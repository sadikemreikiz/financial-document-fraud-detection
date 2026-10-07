from decimal import Decimal
from typing import Any

import pytest

from docfraud.detectors.rules import (
    RULES,
    Status,
    adjustments,
    change_given,
    check,
    explain,
    items_reconcile,
    line_amounts,
    repeats_agree,
    tax_rates,
    totals_reconcile,
)
from docfraud.reading.fields import AmountRole, LineItem, ReceiptFields, TaxSummaryRow

R = AmountRole


def receipt(
    items: tuple[str, ...] = (), rows: tuple[TaxSummaryRow, ...] = (), **amounts: Any
) -> ReceiptFields:
    return ReceiptFields(
        items=tuple(LineItem(amount=Decimal(a)) for a in items),
        amounts={R(role): tuple(Decimal(v) for v in values) for role, values in amounts.items()},
        tax_summary=rows,
    )


def row(rate: str, taxable: str, tax: str | None, code: str = "SR") -> TaxSummaryRow:
    return TaxSummaryRow(
        code=code,
        rate_percent=Decimal(rate),
        taxable_amount=Decimal(taxable),
        tax=None if tax is None else Decimal(tax),
    )


def test_service_charge_and_rounding_link_the_totals() -> None:
    # X51005663298: 41.30 + service charge 4.13 + tax 2.73 = 48.16, rounded to 48.15
    f = receipt(
        total_excluding_tax=["41.30"], service_charge=["4.13"], tax=["2.73"],
        total_including_tax=["48.16"], rounding=["-0.01"], total_due=["48.15"],
    )  # fmt: skip
    outcome = totals_reconcile(f)
    assert outcome.status is Status.PASS
    assert "+ service charge 4.13 + tax 2.73 = total including tax 48.16" in outcome.reason


def test_a_tax_inclusive_subtotal_leads_down_to_the_total_excluding_tax() -> None:
    f = receipt(subtotal=["84.80"], total_excluding_tax=["80.00"], tax=["4.80"],
                total_including_tax=["84.80"])  # fmt: skip
    assert totals_reconcile(f).status is Status.PASS


def test_an_edited_total_breaks_the_chain() -> None:
    f = receipt(subtotal=["20.00"], tax=["1.20"], total_due=["31.20"])
    outcome = totals_reconcile(f)
    assert outcome.status is Status.FAIL
    assert "total due 31.20 does not follow from subtotal 20.00" in outcome.reason
    assert totals_reconcile(receipt(total_due=["5.00"])).status is Status.NOT_APPLICABLE


def test_tax_from_the_summary_stands_in_when_no_tax_amount_is_printed() -> None:
    # X51005441402: 150.00 + service charge 15.00 + tax 9.90 (only in the summary) = 174.90
    rows = (row("10", "150.00", "15.00", code="SVC"), row("6", "165.00", "9.90"))
    f = receipt(rows=rows, subtotal=["150.00"], service_charge=["15.00"], total_due=["174.90"])
    assert totals_reconcile(f).status is Status.PASS
    printed = receipt(rows=rows, tax=["9.00"], subtotal=["150.00"])
    assert [a.label for a in adjustments(printed)] == ["tax"]


def test_delivery_note_items_plus_tax_give_the_amount_due() -> None:
    # X51006556823: no subtotal printed; items 26.16 + tax 0.64 = 26.80
    f = receipt(items=("21.30", "-5.94", "-3.60", "14.40"), tax=["0.64"], total_due=["26.80"])
    outcome = items_reconcile(f)
    assert outcome.status is Status.PASS
    assert outcome.reason == "line amounts 26.16 + tax 0.64 = total due 26.80"


def test_items_that_do_not_reach_the_total_fail() -> None:
    f = receipt(items=("5.00", "6.00"), subtotal=["15.00"])
    assert items_reconcile(f).status is Status.FAIL
    assert items_reconcile(receipt(subtotal=["1.00"])).status is Status.NOT_APPLICABLE


def test_line_amounts() -> None:
    good = ReceiptFields(
        items=(LineItem(quantity=Decimal(2), unit_price=Decimal("2.80"), amount=Decimal("5.60")),)
    )
    bad = ReceiptFields(
        items=(
            LineItem(
                description="TEA",
                quantity=Decimal(17),
                unit_price=Decimal("0.85"),
                amount=Decimal("15.25"),
            ),
        )
    )
    assert line_amounts(good).status is Status.PASS
    assert line_amounts(bad).reason == "TEA: 17 x 0.85 = 14.45, printed 15.25"
    assert line_amounts(receipt(items=("1.00",))).status is Status.NOT_APPLICABLE


@pytest.mark.parametrize(
    ("tendered", "status"),
    [
        (["20.00"], Status.PASS),
        (["-20.00"], Status.PASS),  # cash printed with a minus sign
        (["18.70", "20.00"], Status.PASS),  # a cash line repeating the amount due
        (["10.00", "10.00"], Status.PASS),  # a split payment
        (["25.00"], Status.FAIL),
    ],
)
def test_change_given(tendered: list[str], status: Status) -> None:
    f = receipt(total_due=["18.70"], tendered=tendered, change=["1.30"])
    assert change_given(f).status is status


def test_change_needs_due_payment_and_change() -> None:
    assert (
        change_given(receipt(total_due=["1.00"], change=["0.00"])).status is Status.NOT_APPLICABLE
    )


def test_a_repeated_tax_with_another_value_fails_but_a_zero_tax_line_does_not() -> None:
    # X51005719823 (forged): tax edited to 21.26 in one place, 1.26 left in the other
    assert repeats_agree(receipt(tax=["21.26", "1.26"])).status is Status.FAIL
    assert repeats_agree(receipt(tax=["1.08", "0.00"])).status is Status.NOT_APPLICABLE
    assert repeats_agree(receipt(total_due=["5.00", "5.00"])).status is Status.PASS


def test_tax_rates_allow_half_a_cent_per_line() -> None:
    rows = (row("6", "13.70", "0.80"),)  # 0.822: off by 0.022 over five lines
    five_lines = receipt(items=("1",) * 5, rows=rows)
    one_line = receipt(items=("1",), rows=rows)
    assert tax_rates(five_lines).status is Status.PASS
    assert tax_rates(one_line).status is Status.FAIL
    assert tax_rates(receipt(rows=(row("6", "10", None),))).status is Status.NOT_APPLICABLE


def test_explain_prefers_the_simplest_combination() -> None:
    f = receipt(tax=["1.00"], rounding=["0.00"], service_charge=["2.00"])
    assert explain(Decimal("10.00"), Decimal("12.00"), adjustments(f)) == " + service charge 2.00"
    assert (
        explain(Decimal("10.00"), Decimal("13.00"), adjustments(f))
        == " + service charge 2.00 + tax 1.00"
    )
    assert explain(Decimal("10.00"), Decimal("10.00"), adjustments(f)) == ""
    assert explain(Decimal("10.00"), Decimal("16.00"), adjustments(f)) is None


def test_check_applies_every_rule() -> None:
    assert set(check(ReceiptFields())) == set(RULES)
    assert all(o.status is Status.NOT_APPLICABLE for o in check(ReceiptFields()).values())
