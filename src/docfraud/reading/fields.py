"""Structured receipt content shared by gold annotations, the parser and (later) LLM extraction.

Amounts are recorded *as printed*, never recomputed: a forged receipt must be read as it is,
inconsistencies included, because the inconsistency is the evidence. Every amount carries its
role in the receipt's arithmetic, and a role may be printed several times (for example the total
in the summary and again next to the payment), which lets rules check that repetitions agree.
"""

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class AmountRole(StrEnum):
    """The arithmetic role of a printed amount. Definitions: docs/annotation-guidelines.md."""

    SUBTOTAL = "subtotal"
    DISCOUNT = "discount"
    SERVICE_CHARGE = "service_charge"
    TOTAL_EXCLUDING_TAX = "total_excluding_tax"
    TAX = "tax"
    TOTAL_INCLUDING_TAX = "total_including_tax"
    ROUNDING = "rounding"
    TOTAL_DUE = "total_due"
    TENDERED = "tendered"
    CHANGE = "change"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class LineItem(_Frozen):
    """One purchased line as printed. Missing parts are ``None``."""

    description: str = ""
    quantity: Decimal | None = None
    unit_price: Decimal | None = None
    amount: Decimal | None = None


class TaxSummaryRow(_Frozen):
    """One row of a tax summary table, e.g. ``SR @6%  13.00  0.78``."""

    code: str = ""
    rate_percent: Decimal | None = None
    taxable_amount: Decimal | None = None
    tax: Decimal | None = None


class ReceiptFields(_Frozen):
    """Everything the accounting rules need from one receipt."""

    merchant: str | None = None
    dates: tuple[str, ...] = Field(default=(), description="Printed dates, normalised to ISO")
    items: tuple[LineItem, ...] = ()
    amounts: dict[AmountRole, tuple[Decimal, ...]] = Field(default_factory=dict)
    tax_summary: tuple[TaxSummaryRow, ...] = ()
    item_count: int | None = Field(default=None, description="Printed number of items")
    quantity_total: Decimal | None = Field(default=None, description="Printed total quantity")

    def first(self, role: AmountRole) -> Decimal | None:
        """The first printed amount of a role, or ``None``."""
        values = self.amounts.get(role, ())
        return values[0] if values else None
