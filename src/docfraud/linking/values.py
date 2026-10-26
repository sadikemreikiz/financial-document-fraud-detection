"""The values a reader read and located, and which of them each accounting rule reads.

Link 1 of ADR-0029 (section 5) needs the box of every read value that a forger would change: an
amount, a part of a line item or a date. The merchant, the item count and other text are left
out. Link 2 needs to know which values a failing rule reads. The rules of ADR-0012 stay as they
are; :data:`RULE_VALUES` names, for each of them, every value role its code may read, so that a
value counts for a failing rule whichever of its inputs the rule found wrong.
"""

from collections.abc import Iterable
from dataclasses import dataclass

from docfraud.data.models import BBox
from docfraud.reading.fields import AmountRole, FieldSources, Source

R = AmountRole

ITEM_QUANTITY = "item quantity"
ITEM_UNIT_PRICE = "item unit price"
ITEM_AMOUNT = "item amount"
ITEM_DESCRIPTION = "item description"
TAX_ROW_RATE = "tax row rate"
TAX_ROW_TAXABLE = "tax row taxable amount"
TAX_ROW_TAX = "tax row tax"
DATE = "date"

_TOTALS = (R.SUBTOTAL, R.TOTAL_EXCLUDING_TAX, R.TOTAL_INCLUDING_TAX, R.TOTAL_DUE)
_ADJUSTMENTS = (R.DISCOUNT, R.SERVICE_CHARGE, R.TAX, R.ROUNDING)

RULE_VALUES: dict[str, frozenset[str]] = {
    "totals_reconcile": frozenset({*(r.value for r in (*_TOTALS, *_ADJUSTMENTS)), TAX_ROW_TAX}),
    "items_reconcile": frozenset(
        {ITEM_AMOUNT, *(r.value for r in (*_TOTALS, *_ADJUSTMENTS)), TAX_ROW_TAX}
    ),
    "line_amounts": frozenset({ITEM_QUANTITY, ITEM_UNIT_PRICE, ITEM_AMOUNT}),
    "change_given": frozenset(
        r.value for r in (R.TOTAL_DUE, R.TOTAL_INCLUDING_TAX, R.TENDERED, R.CHANGE)
    ),
    "repeats_agree": frozenset(r.value for r in (*_TOTALS, R.TAX, R.ROUNDING, R.CHANGE)),
    "tax_rates": frozenset({TAX_ROW_RATE, TAX_ROW_TAXABLE, TAX_ROW_TAX}),
}
"""Every value role each rule of :data:`docfraud.detectors.rules.RULES` reads, from its code.

Amounts are named by their :class:`~docfraud.reading.fields.AmountRole`. A total, discount,
service charge, tax or rounding enters ``totals_reconcile`` and ``items_reconcile`` as a step
between totals; the taxes of the tax summary stand in for a missing tax line.
"""


@dataclass(frozen=True, slots=True)
class LocatedValue:
    """A read value that a forger might change, and where it is printed."""

    role: str  # an amount role, a part of a line item, a part of a tax row, or the date
    box: BBox


def located_values(sources: FieldSources) -> tuple[LocatedValue, ...]:
    """The amounts, line-item parts, tax-row parts and dates of a reading that have a box."""
    found: list[tuple[str, Source | None]] = [
        (role.value, source) for role, group in sources.amounts.items() for source in group
    ]
    for item in sources.items:
        found += [
            (ITEM_QUANTITY, item.quantity),
            (ITEM_UNIT_PRICE, item.unit_price),
            (ITEM_AMOUNT, item.amount),
            *((ITEM_DESCRIPTION, s) for s in item.description),
        ]
    for row in sources.tax_summary:
        found += [
            (TAX_ROW_RATE, row.rate_percent),
            (TAX_ROW_TAXABLE, row.taxable_amount),
            (TAX_ROW_TAX, row.tax),
        ]
    found += [(DATE, s) for s in sources.dates]
    return tuple(
        LocatedValue(role, source.box)
        for role, source in found
        if source is not None and source.box is not None
    )


def read_by(rules: Iterable[str]) -> frozenset[str]:
    """The value roles that any of the named rules reads.

    Raises:
        KeyError: If a rule is not one of :data:`RULE_VALUES`.
    """
    return frozenset().union(*(RULE_VALUES[rule] for rule in rules))
