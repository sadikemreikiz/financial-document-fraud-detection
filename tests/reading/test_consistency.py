from decimal import Decimal

import pytest

from docfraud.reading.consistency import (
    CHECKS,
    change_math,
    items_sum,
    line_math,
    repeats_agree,
    rounding_math,
    tax_math,
)
from docfraud.reading.fields import AmountRole, LineItem, ReceiptFields

D = Decimal
R = AmountRole


def fields(**amounts: tuple[str, ...]) -> ReceiptFields:
    return ReceiptFields(amounts={R(k): tuple(D(v) for v in vs) for k, vs in amounts.items()})


def test_items_sum() -> None:
    items = (LineItem(amount=D("1.50")), LineItem(amount=D("2.50")), LineItem())
    assert items_sum(ReceiptFields(items=items, amounts={R.TOTAL_DUE: (D("4.00"),)})) is True
    assert items_sum(ReceiptFields(items=items, amounts={R.SUBTOTAL: (D("4.50"),)})) is False
    assert items_sum(ReceiptFields(items=items)) is None


def test_line_math() -> None:
    good = LineItem(quantity=D(2), unit_price=D("1.25"), amount=D("2.50"))
    bad = LineItem(quantity=D(2), unit_price=D("1.25"), amount=D("2.60"))
    assert line_math(ReceiptFields(items=(good, LineItem(amount=D(1))))) is True
    assert line_math(ReceiptFields(items=(good, bad))) is False
    assert line_math(ReceiptFields(items=(LineItem(amount=D(1)),))) is None


@pytest.mark.parametrize(
    ("check", "ok", "broken"),
    [
        (
            tax_math,
            fields(total_excluding_tax=("10.00",), tax=("0.60",), total_including_tax=("10.60",)),
            fields(total_excluding_tax=("10.00",), tax=("0.60",), total_including_tax=("11.60",)),
        ),
        (
            rounding_math,
            fields(total_including_tax=("10.62",), rounding=("-0.02",), total_due=("10.60",)),
            fields(total_including_tax=("10.62",), rounding=("0.02",), total_due=("10.60",)),
        ),
        (
            change_math,
            fields(total_due=("7.50",), tendered=("10.00",), change=("2.50",)),
            fields(total_due=("7.50",), tendered=("10.00",), change=("3.50",)),
        ),
        (
            repeats_agree,
            fields(total_due=("5.00", "5.00"), tendered=("5.00", "9.00")),
            fields(total_due=("5.00", "50.00")),
        ),
    ],
)
def test_relations(check: object, ok: ReceiptFields, broken: ReceiptFields) -> None:
    assert callable(check)
    assert check(ok) is True
    assert check(broken) is False
    assert check(ReceiptFields()) is None


def test_change_falls_back_to_the_total_including_tax() -> None:
    f = fields(total_including_tax=("7.50",), tendered=("10.00",), change=("2.50",))
    assert change_math(f) is True


def test_registry_names_every_check() -> None:
    assert set(CHECKS.values()) == {
        items_sum,
        line_math,
        tax_math,
        rounding_math,
        change_math,
        repeats_agree,
    }
