"""Structured receipt content shared by gold annotations, the parser and LLM extraction.

Amounts are recorded *as printed*, never recomputed: a forged receipt must be read as it is,
inconsistencies included, because the inconsistency is the evidence. Every amount carries its
role in the receipt's arithmetic, and a role may be printed several times (for example the total
in the summary and again next to the payment), which lets rules check that repetitions agree.

A reader also records where each value is printed (:class:`FieldSources`), in a structure that
mirrors the fields, so that evidence found in the pixels can be linked to the value it changes
(ADR-0024). Gold annotations carry values only.
"""

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field

from docfraud.data.models import BBox


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


class Source(_Frozen):
    """Where a value is printed: characters ``start:end`` of the text of one row.

    ``row`` indexes the rows the reader was given; ``start`` and ``end`` are offsets into that
    row's text (:attr:`docfraud.reading.layout.Row.text`). ``segments`` are the row's segments
    the span touches, and ``box`` encloses those of them that are positioned (``None`` if none
    is). For a number the span covers its digits; for text it covers the row.
    """

    row: int = Field(ge=0)
    start: int = Field(ge=0)
    end: int = Field(gt=0)
    segments: tuple[int, ...]
    box: BBox | None = None


class ItemSources(_Frozen):
    """Where the parts of one :class:`LineItem` are printed; ``None`` where a part is."""

    description: tuple[Source, ...] = ()
    quantity: Source | None = None
    unit_price: Source | None = None
    amount: Source | None = None


class TaxRowSources(_Frozen):
    """Where the parts of one :class:`TaxSummaryRow` are printed."""

    code: Source | None = None
    rate_percent: Source | None = None
    taxable_amount: Source | None = None
    tax: Source | None = None


class FieldSources(_Frozen):
    """Where each value of a :class:`ReceiptFields` is printed, in the same structure.

    Tuples run parallel to the fields: ``items[i]`` locates ``ReceiptFields.items[i]`` and
    ``amounts[role][k]`` locates ``ReceiptFields.amounts[role][k]``.
    """

    merchant: tuple[Source, ...] = ()
    dates: tuple[Source, ...] = ()
    items: tuple[ItemSources, ...] = ()
    amounts: dict[AmountRole, tuple[Source, ...]] = Field(default_factory=dict)
    tax_summary: tuple[TaxRowSources, ...] = ()
    item_count: Source | None = None
    quantity_total: Source | None = None


class Extraction(_Frozen):
    """The fields a reader extracted from one receipt and where each of them is printed."""

    fields: ReceiptFields
    sources: FieldSources
