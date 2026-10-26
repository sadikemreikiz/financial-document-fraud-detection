from decimal import Decimal

import pytest

from docfraud.data.models import BBox
from docfraud.detectors.rules import RULES, Status, check
from docfraud.linking.values import (
    DATE,
    ITEM_AMOUNT,
    ITEM_DESCRIPTION,
    ITEM_QUANTITY,
    ITEM_UNIT_PRICE,
    RULE_VALUES,
    TAX_ROW_RATE,
    TAX_ROW_TAX,
    TAX_ROW_TAXABLE,
    LocatedValue,
    located_values,
    read_by,
)
from docfraud.reading.fields import (
    AmountRole,
    FieldSources,
    ItemSources,
    LineItem,
    ReceiptFields,
    Source,
    TaxRowSources,
    TaxSummaryRow,
)

R = AmountRole
D = Decimal


def full_receipt() -> ReceiptFields:
    """A receipt on which every rule applies and passes."""
    return ReceiptFields(
        merchant="SHOP",
        dates=("2018-01-02",),
        items=(
            LineItem(description="TEA", quantity=D(1), unit_price=D("4.00"), amount=D("4.00")),
            LineItem(description="BUN", quantity=D(2), unit_price=D("3.00"), amount=D("6.00")),
        ),
        amounts={
            R.SUBTOTAL: (D("10.00"),),
            R.TAX: (D("0.60"),),
            R.TOTAL_INCLUDING_TAX: (D("10.60"),),
            R.TOTAL_DUE: (D("10.60"), D("10.60")),
            R.TENDERED: (D("20.00"),),
            R.CHANGE: (D("9.40"),),
        },
        tax_summary=(
            TaxSummaryRow(code="SR", rate_percent=D(6), taxable_amount=D("10.00"), tax=D("0.60")),
        ),
    )


def perturbed(fields: ReceiptFields, role: str) -> ReceiptFields:
    """The receipt with the first printed value of one role changed by 1.11."""
    step = D("1.11")
    if role in {r.value for r in AmountRole}:
        amounts = dict(fields.amounts)
        head, *rest = amounts[R(role)]
        amounts[R(role)] = (head + step, *rest)
        return fields.model_copy(update={"amounts": amounts})
    part = {ITEM_QUANTITY: "quantity", ITEM_UNIT_PRICE: "unit_price", ITEM_AMOUNT: "amount"}
    if role in part:
        first = fields.items[0]
        changed = first.model_copy(update={part[role]: getattr(first, part[role]) + step})
        return fields.model_copy(update={"items": (changed, *fields.items[1:])})
    column = {TAX_ROW_RATE: "rate_percent", TAX_ROW_TAXABLE: "taxable_amount", TAX_ROW_TAX: "tax"}
    if role in column:
        row = fields.tax_summary[0]
        changed_row = row.model_copy(update={column[role]: getattr(row, column[role]) + step})
        return fields.model_copy(update={"tax_summary": (changed_row,)})
    assert role == DATE
    return fields.model_copy(update={"dates": ("2019-03-04",)})


def test_every_rule_has_a_value_map() -> None:
    assert set(RULE_VALUES) == set(RULES)


@pytest.mark.parametrize("rule", sorted(RULES))
def test_a_rule_does_not_read_values_outside_its_map(rule: str) -> None:
    fields = full_receipt()
    assert check(fields)[rule].status is Status.PASS
    printed = {r.value for r in fields.amounts} | {
        ITEM_QUANTITY,
        ITEM_UNIT_PRICE,
        ITEM_AMOUNT,
        TAX_ROW_RATE,
        TAX_ROW_TAXABLE,
        TAX_ROW_TAX,
        DATE,
    }
    for role in sorted(printed - RULE_VALUES[rule]):
        assert check(perturbed(fields, role))[rule].status is Status.PASS, role
    read = sorted(printed & RULE_VALUES[rule])
    assert any(check(perturbed(fields, role))[rule].status is Status.FAIL for role in read)


def source(row: int, box: BBox | None = None) -> Source:
    return Source(row=row, start=0, end=4, segments=(0,), box=box)


def test_located_values_keep_amounts_items_tax_rows_and_dates_with_a_box() -> None:
    box = BBox(x=1, y=2, width=3, height=4)
    sources = FieldSources(
        merchant=(source(0, box),),
        dates=(source(1, box),),
        items=(
            ItemSources(
                description=(source(2, box),), quantity=source(2, box), amount=source(2, None)
            ),
        ),
        amounts={R.TOTAL_DUE: (source(3, box), source(4, None))},
        tax_summary=(TaxRowSources(rate_percent=source(5, box), tax=source(5, box)),),
        item_count=source(6, box),
    )
    found = located_values(sources)
    assert sorted(v.role for v in found) == sorted(
        [DATE, ITEM_DESCRIPTION, ITEM_QUANTITY, "total_due", TAX_ROW_RATE, TAX_ROW_TAX]
    )
    assert all(v.box == box for v in found)
    assert LocatedValue("total_due", box) in found


def test_read_by_joins_the_maps_of_the_named_rules() -> None:
    assert read_by([]) == frozenset()
    assert (
        read_by(["change_given", "tax_rates"])
        == RULE_VALUES["change_given"] | RULE_VALUES["tax_rates"]
    )
    with pytest.raises(KeyError):
        read_by(["no such rule"])
