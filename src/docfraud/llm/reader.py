"""From an LLM's answer to fields: every value comes from the labelled text, never from the model.

ADR-0025, decision 1: code accepts only labels and rows that exist on the receipt, takes the
values from the text and keeps where they are printed. A reference that does not exist, or holds
no value of the right kind, leaves its field empty and is counted as a rejection.

Text fields are read from the rows the model names, by fixed rules: the merchant is the text of
its rows; an item's description is the text of its rows without the item's own numbers and
without codes of six or more digits (annotation guidelines, section 4). Amounts of one role and
dates are put in print order, and a date printed twice is kept once (section 5).
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass
from decimal import Decimal

from pydantic import ValidationError

from docfraud.llm.answer import Answer
from docfraud.llm.labels import Label, LabelKind, LabelledRows
from docfraud.llm.provider import Response
from docfraud.reading.fields import (
    AmountRole,
    Extraction,
    FieldSources,
    ItemSources,
    LineItem,
    ReceiptFields,
    Source,
    TaxRowSources,
    TaxSummaryRow,
)
from docfraud.reading.layout import Row
from docfraud.reading.sources import whole_row

_CODE = re.compile(r"(?<![\d.,])\d{6,}(?![\d.,])")
_SPACES = re.compile(r"\s+")


@dataclass(frozen=True, slots=True)
class Rejection:
    """A reference in an answer that does not hold what its field needs."""

    field: str
    reference: str
    reason: str


@dataclass(frozen=True, slots=True)
class LlmReading:
    """The fields an LLM reader extracted from one receipt, and what was rejected on the way."""

    extraction: Extraction
    rejections: tuple[Rejection, ...]

    @property
    def fields(self) -> ReceiptFields:
        """The extracted fields."""
        return self.extraction.fields


EMPTY = Extraction(fields=ReceiptFields(), sources=FieldSources())


def read_response(response: Response, labelled: LabelledRows, rows: Sequence[Row]) -> LlmReading:
    """Turn a stored response into fields; a response without a valid answer yields none."""
    if response.answer is None:
        reason = response.error or response.stop_reason or "no answer"
        return LlmReading(EMPTY, (Rejection("answer", "", reason),))
    try:
        answer = Answer.model_validate_json(response.answer)
    except ValidationError:
        return LlmReading(EMPTY, (Rejection("answer", "", "not in the answer form"),))
    return read_answer(answer, labelled, rows)


def read_answer(answer: Answer, labelled: LabelledRows, rows: Sequence[Row]) -> LlmReading:
    """Turn an answer into fields with their sources, rejecting what does not hold."""
    check = _Checker(labelled, rows)
    merchant_rows = check.rows("merchant_rows", answer.merchant_rows)
    merchant = _text(rows, merchant_rows, ()) or None

    named = [
        check.label(f"dates[{n}]", name, LabelKind.DATE) for n, name in enumerate(answer.dates)
    ]
    dates: list[Label] = []
    for label in sorted((label for label in named if label is not None), key=_position):
        if all(label.day != kept.day for kept in dates):
            dates.append(label)

    items: list[tuple[LineItem, ItemSources]] = []
    for n, item in enumerate(answer.items):
        quantity = check.number(f"items[{n}].quantity", item.quantity)
        unit_price = check.number(f"items[{n}].unit_price", item.unit_price)
        amount = check.number(f"items[{n}].amount", item.amount)
        own = [label for label in (quantity, unit_price, amount) if label is not None]
        item_rows = check.rows(f"items[{n}].rows", item.rows)
        items.append(
            (
                LineItem(
                    description=_text(rows, item_rows, own, drop_codes=True),
                    quantity=_value(quantity),
                    unit_price=_value(unit_price),
                    amount=_value(amount),
                ),
                ItemSources(
                    description=tuple(whole_row(rows, row) for row in item_rows),
                    quantity=_source(quantity),
                    unit_price=_source(unit_price),
                    amount=_source(amount),
                ),
            )
        )

    amounts: dict[AmountRole, list[Label]] = {}
    for n, entry in enumerate(answer.amounts):
        printed = check.number(f"amounts[{n}]", entry.label)
        if printed is not None:
            amounts.setdefault(entry.role, []).append(printed)
    for labels in amounts.values():
        labels.sort(key=_position)

    tax_rows: list[tuple[TaxSummaryRow, TaxRowSources]] = []
    for n, row in enumerate(answer.tax_summary):
        rate = check.number(f"tax_summary[{n}].rate_percent", row.rate_percent)
        taxable = check.number(f"tax_summary[{n}].taxable_amount", row.taxable_amount)
        tax = check.number(f"tax_summary[{n}].tax", row.tax)
        tax_rows.append(
            (
                TaxSummaryRow(
                    rate_percent=_value(rate), taxable_amount=_value(taxable), tax=_value(tax)
                ),
                TaxRowSources(
                    rate_percent=_source(rate), taxable_amount=_source(taxable), tax=_source(tax)
                ),
            )
        )

    item_count = check.whole_number("item_count", answer.item_count)
    quantity_total = check.number("quantity_total", answer.quantity_total)
    fields = ReceiptFields(
        merchant=merchant,
        dates=tuple(label.day.isoformat() for label in dates if label.day is not None),
        items=tuple(item for item, _ in items),
        amounts={role: tuple(_number(label) for label in found) for role, found in amounts.items()},
        tax_summary=tuple(row for row, _ in tax_rows),
        item_count=None if item_count is None else int(_number(item_count)),
        quantity_total=_value(quantity_total),
    )
    sources = FieldSources(
        merchant=tuple(whole_row(rows, row) for row in merchant_rows),
        dates=tuple(label.source for label in dates),
        items=tuple(source for _, source in items),
        amounts={role: tuple(label.source for label in found) for role, found in amounts.items()},
        tax_summary=tuple(source for _, source in tax_rows),
        item_count=_source(item_count),
        quantity_total=_source(quantity_total),
    )
    return LlmReading(Extraction(fields=fields, sources=sources), tuple(check.rejections))


class _Checker:
    """Looks up references and records the ones that do not hold."""

    def __init__(self, labelled: LabelledRows, rows: Sequence[Row]) -> None:
        self.labels = labelled.labels
        self.row_count = len(rows)
        self.rejections: list[Rejection] = []

    def reject(self, field: str, reference: str, reason: str) -> None:
        self.rejections.append(Rejection(field, reference, reason))

    def label(self, field: str, name: str | None, kind: LabelKind) -> Label | None:
        if name is None:
            return None
        label = self.labels.get(name)
        if label is None:
            self.reject(field, name, "no such label")
            return None
        if label.kind is not kind:
            self.reject(field, name, f"not a {kind.value}")
            return None
        return label

    def number(self, field: str, name: str | None) -> Label | None:
        label = self.label(field, name, LabelKind.NUMBER)
        if label is not None and label.number is None:
            self.reject(field, label.name, "separators cannot be read")
            return None
        return label

    def whole_number(self, field: str, name: str | None) -> Label | None:
        label = self.number(field, name)
        if label is not None and _number(label) != _number(label).to_integral_value():
            self.reject(field, label.name, "not a whole number")
            return None
        return label

    def rows(self, field: str, rows: Sequence[int]) -> list[int]:
        kept: list[int] = []
        for row in rows:
            if not 0 <= row < self.row_count:
                self.reject(field, str(row), "no such row")
            elif row not in kept:
                kept.append(row)
        return kept


def _number(label: Label) -> Decimal:
    if label.number is None:  # pragma: no cover - checked numbers always have a value
        msg = f"label {label.name} has no value"
        raise ValueError(msg)
    return label.number


def _value(label: Label | None) -> Decimal | None:
    return None if label is None else _number(label)


def _source(label: Label | None) -> Source | None:
    return None if label is None else label.source


def _position(label: Label) -> tuple[int, int]:
    return label.source.row, label.source.start


def _text(
    rows: Sequence[Row], chosen: Sequence[int], own: Sequence[Label], *, drop_codes: bool = False
) -> str:
    """The text of the chosen rows, without the given labels' printed numbers."""
    parts: list[str] = []
    for row in chosen:
        text = rows[row].text
        spans = sorted(
            (label.source.start, label.source.end) for label in own if label.source.row == row
        )
        for start, end in reversed(spans):
            text = text[:start] + " " + text[end:]
        if drop_codes:
            text = _CODE.sub(" ", text)
        parts.append(text)
    return _SPACES.sub(" ", " ".join(parts)).strip()
