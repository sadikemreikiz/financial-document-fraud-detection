"""Rule-based receipt parser: rows of positioned text to :class:`ReceiptFields`.

The parser reads amounts as printed and assigns them roles from their labels. It never checks or
repairs the arithmetic: that is the job of the accounting rules, which need the inconsistencies
of forged receipts intact. Label vocabulary and role definitions follow
``docs/annotation-guidelines.md``.

Every value keeps the row and the characters it was read from (:func:`extract_fields`). Tracking
them does not change what is read: the values are those of parser v2 as frozen at ``ef22443``.
"""

import re
from collections.abc import Sequence
from datetime import date
from decimal import Decimal, InvalidOperation
from typing import NamedTuple

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
from docfraud.reading.sources import Normalised, locate, normalise

_AMOUNT = re.compile(
    r"(?<![\w.,])(?:RM|MYR|\$)?\s?(?P<sign>-)?\s?(?:RM|MYR|\$)?\s?"
    r"(?P<value>\d{1,3}(?:,\d{3})+\.\d{2}|\d{1,7}\.\d{2})(?P<trail>-)?(?![\w.,]*\d)"
    r"(?!\s?(?:MM|CM|ML|MG|KG|GM|G|L|M|OZ|LTR|LITRES?)\b)"
)
_HAS_LETTERS = re.compile(r"[A-Z]{2,}")
_PRICE = re.compile(r"(?<![\w.,])(?:RM|\$)?(?P<value>\d{1,7}\.\d{2,4})(?![\w.,]*\d)")
_WORD = re.compile(r"[A-Z]{3,}")
_CODES = re.compile(r"^\d{1,3}\.\s|^(?:SR|ZRL?|S|Z)\s+(?=\S*\d)|\b[A-Z]?\d{6,}\b-?|\b\d{4}-\d{4}\b")
_HEADER_BLOCK = re.compile(
    r"\bINVOICE\b|\bRECEIPT\b|\b(?:GST|TAX|CO\.?)\s?(?:ID|REG)|\bTEL\b|\bCASHIER\b"
    r"|\bCOUNTER\b|\bTABLE\b|\bSHOPPING HOURS\b|\bHRS\b|\bSALESPERSON\b|\bBILL\b"
)
_PER_CODE_TAX = re.compile(r"^(?P<rate>\d+(?:\.\d+)?)\s?%\s*GST\s+(?P<code>[A-Z])\b")
_RELIEF_TAX = re.compile(r"^RELIEF\s+GST\s+(?P<code>[A-Z])\b")

# Ordered: the first pattern that matches a label decides its role. Tax comes first because tax
# lines often mention totals ("GST @6% INCLUDED IN TOTAL"); explicit final totals come before
# rounding because they often mention it ("TOTAL AFTER ROUNDING").
_ROLE_PATTERNS: tuple[tuple[AmountRole, re.Pattern[str]], ...] = (
    (
        AmountRole.TAX,
        re.compile(
            r"\b(?:GST|SST|TAX)\b.*\bINCLUDED\b|\bINCLUDES?\b.*\b(?:GST|SST|TAX)\b"
            r"|^(?:TOTAL\s+)?(?:GST|SST|TAX)\b(?!.*\b(?:ID|REG|NO|CODE|SUMMARY|INVOICE)\b)"
            r"|\b(?:GST|SST|TAX) (?:PAYABLE|AMOUNT)\b"
        ),
    ),
    (
        AmountRole.TOTAL_DUE,
        re.compile(
            r"\bTOTAL\b.*\b(?:AFTER|ROUND(?:ED|ING)?)\b|\bROUNDED\b"
            r"|\b(?:TOTAL|AMT|AMOUNT)\b.*\bPAYABLE\b|\bNETT?\s+TOTAL\b|\bGRAND\b"
            r"|\bTO BE PAID\b|\b(?:AMOUNT|BALANCE)\s+DUE\b"
        ),
    ),
    (AmountRole.ROUNDING, re.compile(r"\bROUND(?:ING)?\b|\bRND\b|^ADJ(?:USTMENT)?\b")),
    (AmountRole.CHANGE, re.compile(r"\bCHANGE\b|\bBAKI\b")),
    (
        AmountRole.TOTAL_EXCLUDING_TAX,
        re.compile(r"\bTOTAL\b.*\bEXC(?:L|LUD(?:E|ING)|LUSIVE|\.)?\b|\bAMOUNT\b.*\bEXCL"),
    ),
    (AmountRole.TOTAL_INCLUDING_TAX, re.compile(r"\bTOTAL\b.*\bINC(?:L|LUSIVE|\.)?\b")),
    (AmountRole.SUBTOTAL, re.compile(r"\bSUB[\s-]?TOTAL\b|\bSUBTTL\b")),
    (AmountRole.DISCOUNT, re.compile(r"\bDISC(?:OUNT)?\b")),
    (AmountRole.SERVICE_CHARGE, re.compile(r"\bSERVICE\b|\bSVC\b|\bS\.?\s?CHARGE\b|\bSERV\b")),
    (
        AmountRole.TENDERED,
        re.compile(
            r"\bCASH\b(?!\s*(?:SALES?|INV|BILL|REFUND|COUNTER|RECEIPT|NO))|\bTENDER(?:ED)?\b"
            r"|\bPAID\b|\bVISA\b|\bMASTER(?:CARD)?\b|\bCREDIT\s*CARD\b|\bDEBIT\b|\bCARD\b"
            r"|\bE-?WALLET\b|\bPAYMENT\b"
        ),
    ),
    (AmountRole.TOTAL_DUE, re.compile(r"\bTOTAL\b|^TL\b|\bNETT?\b|\bAMOUNT DUE\b|\bGRAND\b")),
)
_TOTAL_INCLUDES = re.compile(r"^TOTAL\b.*\bINCLUDES?\b")
_EXPLICIT_FINAL = re.compile(
    r"PAYABLE|ROUND|NETT?\b|GRAND|FINAL|AMOUNT DUE|TOTAL DUE|\bAFTER\b|TO BE PAID"
)
_NOT_A_TOTAL = re.compile(
    r"\bTOTAL\s+(?:QTY|QUANTITY|ITEMS?|PCS|SAVINGS?|DISCOUNT|POINTS?|WEIGHT|GROSS)\b"
    r"|\d+\s?%\s*SUPPLIES"
)
_SUPPLIES = re.compile(r"(?P<rate>\d+(?:\.\d+)?)\s?%\s*SUPPLIES")
_SUMMARY_HEADER = re.compile(r"\b(?:GST|SST|TAX)\s+SUMMARY\b|\bTAX CODE\b|\bSUMMARY\b")
_SUMMARY_TOTAL = re.compile(r"^TOTAL\b")
_SUMMARY_ROW = re.compile(r"^(?P<code>[A-Z]{1,3})\b.*?(?P<rate>\d+(?:\.\d+)?)\s?%")
_ITEM_HEADER = re.compile(r"\b(?:DESCRIPTION|ITEM|PRODUCT|QTY|U\.?\s?PRICE|PRICE|AMT|AMOUNT)\b")
_QUANTITY = re.compile(
    r"(?<![\d.])(?:(?P<qty>\d{1,4}(?:\.\d{1,3})?)\s?[Xx]|(?P<count>\d{1,3})\s?\*(?=\s+\d))\s"
)
_UNIT_QUANTITY = re.compile(
    r"(?:^|\s)(?P<qty>\d{1,4}(?:\.\d{1,3})?)\s+"
    r"(?:PKT|PCS|PC|ROLL|UNITS?|EA|BOX|BTL|KG|SET|PAIR|CTN|BAG|TIN|PACK)\b"
)
_LITRES = re.compile(r"(?P<qty>\d{1,4}\.\d{2})\s+LITRE\b")
_PER_LITRE = re.compile(r"(?P<price>\d{1,2}\.\d{3})\s*RM\s*/?\s*LI?TRE|RM\s*/\s*LITRE")
_BARE_QUANTITY = re.compile(r"(?:^|\s)(?P<qty>\d{1,3})\s+(?=(?:RM|\$)?\s?-?\d{1,7}\.\d{2}\b)")
_ITEMS_END = re.compile(
    r"\bTOT(?:AL)?\.?\s*(?:QTY|QUANTITY|ITEMS?)\b|\bITEM\(?S\)?\s*:|\bNO\.?\s*OF\s*ITEMS?\b"
    r"|\bITEM\s+(?:COUNT|QTY)\b"
    r"|\bSUB[\s-]?TOTAL\b|^TOTAL\b"
)
_TAX_LINE = re.compile(r"^(?:GST|SST|TAX)\b")
_ITEM_COUNT = re.compile(
    r"\b(?:ITEM\(?S\)?|NO\.?\s*OF\s*ITEMS?|TOTAL\s+ITEMS?|ITEM\s+COUNT)\s*:?\s*(?P<n>\d{1,3})\b"
)
_QTY_TOTAL = re.compile(
    r"\b(?:TOTAL\s+QTY|QTY\(?S\)?|TOTAL\s+QUANTITY)\s*:?\s*(?P<n>\d{1,4}(?:\.\d+)?)\b"
)
_MERCHANT_HINT = re.compile(
    r"\b(?:SDN\.?\s*BHD|BHD|ENTERPRISE|TRADING|RESTORAN|RESTAURANT|STORE|MART|SHOP|CAFE|"
    r"HARDWARE|PHARMACY|BAKERIES|BOOK|STATIONERY|CO\.)\b"
)
_NOT_NAME = re.compile(
    r"^(?:NO\.?\s?\d|LOT\b|\d+[,&]|\(?CO\.?\s?(?:NO|REG)|COMPANY\s+(?:NO|REG)|ROC\b|BR\s+NO)"
    r"|\b(?:CASH\s+SALES?|REF\s+NO|THANK|YOUR\s+ORDER|COPY|GOODS\s+SOLD|PLEASE|WELCOME)\b"
)
_NAME_CONTINUES = re.compile(
    r"^(?:\(M\)\s*)?(?:SDN\.?\s*BHD|S/B|BHD|CO\.\s|CO\.?\s*\(M\)|ENTERPRISE|TRADING)"
    r"|^[A-Z]+\)\s*SDN|^[A-Z']+\s+SDN\.?\s*BHD\.?$"
)
_REGISTRATION = re.compile(
    r"\(?\b(?:CO\.?\s*(?:NO|REG)\.?\s*:?\s*)?[A-Z]{0,2}\d{5,10}-?[A-Z]\b\)?,?"
    r"|\(\s*\d{5,10}\s[A-Z]\s*\)|\(\s*(?:CO\.?)?\s*$"
)
_NOT_MERCHANT = re.compile(r"\b(?:TAX INVOICE|INVOICE|RECEIPT|GST|TEL|FAX|NO\.)\b")
_MONTHS = {
    month: number
    for number, month in enumerate(
        ("JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"),
        start=1,
    )
}
_DATE_DMY = re.compile(r"(?<!\d)(?P<d>\d{1,2})[/.-](?P<m>\d{1,2})[/.-](?P<y>\d{4}|\d{2})(?!\d)")
_DATE_YMD = re.compile(r"(?<!\d)(?P<y>20\d{2})[/.-](?P<m>\d{1,2})[/.-](?P<d>\d{1,2})(?!\d)")
_DATE_TEXT = re.compile(
    r"(?<!\d)(?P<d>\d{1,2})[\s-]*(?P<m>JAN|FEB|MAR|APR|MAY|JUN|JUL|AUG|SEP|OCT|NOV|DEC)[A-Z]*"
    r"[\s,-]*(?P<y>\d{4}|\d{2})(?!\d)"
)


class _Pos(NamedTuple):
    """Characters ``start:end`` of the normalised text of row ``row``."""

    row: int
    start: int
    end: int


type _Number = tuple[Decimal, int, int]  # value and the span of its digits in a text


class _Labelled(NamedTuple):
    role: AmountRole
    value: Decimal
    explicit: bool
    label: str
    label_row: int  # the row of the label
    pos: _Pos  # where the value is printed: the label's row or the row below it


class _Found(NamedTuple):
    value: Decimal
    explicit: bool  # the label names a final total explicitly
    label_row: int  # the row of the label, which orders amounts of one role
    pos: _Pos


class _TaxPos(NamedTuple):
    code: _Pos | None = None
    rate_percent: _Pos | None = None
    taxable_amount: _Pos | None = None
    tax: _Pos | None = None


class _ItemPos(NamedTuple):
    description: int  # the row the description was read from
    quantity: _Pos | None
    unit_price: _Pos | None
    amount: _Pos


def parse_receipt(rows: Sequence[Row]) -> ReceiptFields:
    """Extract structured fields from rows in reading order.

    Args:
        rows: Printed rows, top to bottom, as produced by
            :func:`docfraud.reading.layout.group_rows`.
    """
    return extract_fields(rows).fields


def extract_fields(rows: Sequence[Row]) -> Extraction:
    """Extract structured fields from rows in reading order, with where each value is printed.

    Args:
        rows: Printed rows, top to bottom, as produced by
            :func:`docfraud.reading.layout.group_rows`.
    """
    texts = [_clean(row.text) for row in rows]
    normalised = [normalise(row.text) for row in rows]
    if [n.text for n in normalised] != texts:  # pragma: no cover - normalise() mirrors _clean()
        msg = "the normalised row texts differ from the texts the parser reads"
        raise RuntimeError(msg)
    labelled, tax_summary = _scan(texts)
    first_total_row, amounts = _assign_roles(labelled)
    merchant, merchant_rows = _merchant(texts)
    dates = _dates(texts)
    items = _items(texts, end=first_total_row)
    item_count = _first_int(_ITEM_COUNT, texts)
    quantity_total = _first_decimal(_QTY_TOTAL, texts)

    fields = ReceiptFields(
        merchant=merchant,
        dates=tuple(iso for iso, _ in dates),
        items=tuple(item for item, _ in items),
        amounts={role: tuple(f.value for f in found) for role, found in amounts.items()},
        tax_summary=tuple(row for row, _ in tax_summary),
        item_count=item_count[0] if item_count else None,
        quantity_total=quantity_total[0] if quantity_total else None,
    )
    at = _Locator(rows, normalised)
    sources = FieldSources(
        merchant=tuple(at.whole(row) for row in merchant_rows),
        dates=tuple(at(pos) for _, pos in dates),
        items=tuple(
            ItemSources(
                description=(at.whole(pos.description),),
                quantity=at.maybe(pos.quantity),
                unit_price=at.maybe(pos.unit_price),
                amount=at(pos.amount),
            )
            for _, pos in items
        ),
        amounts={role: tuple(at(f.pos) for f in found) for role, found in amounts.items()},
        tax_summary=tuple(
            TaxRowSources(
                code=at.maybe(pos.code),
                rate_percent=at.maybe(pos.rate_percent),
                taxable_amount=at.maybe(pos.taxable_amount),
                tax=at.maybe(pos.tax),
            )
            for _, pos in tax_summary
        ),
        item_count=at.maybe(item_count and item_count[1]),
        quantity_total=at.maybe(quantity_total and quantity_total[1]),
    )
    return Extraction(fields=fields, sources=sources)


class _Locator:
    """Turns positions in the normalised row texts into sources on the original rows."""

    def __init__(self, rows: Sequence[Row], normalised: Sequence[Normalised]) -> None:
        self.rows = rows
        self.normalised = normalised

    def __call__(self, pos: _Pos) -> Source:
        start, end = self.normalised[pos.row].original_span(pos.start, pos.end)
        return locate(self.rows, pos.row, start, end)

    def maybe(self, pos: _Pos | None) -> Source | None:
        return None if pos is None else self(pos)

    def whole(self, row: int) -> Source:
        return locate(self.rows, row, 0, len(self.rows[row].text))


def _scan(texts: Sequence[str]) -> tuple[list[_Labelled], list[tuple[TaxSummaryRow, _TaxPos]]]:
    """Labelled amounts and tax summary rows, row by row."""
    labelled: list[_Labelled] = []
    tax_summary: list[tuple[TaxSummaryRow, _TaxPos]] = []
    in_summary = False
    for index, text in enumerate(texts):
        if _SUMMARY_HEADER.search(text):
            in_summary = True
            continue
        numbers = _amount_matches(text)
        per_code = _per_code_tax(text, numbers, index)
        if per_code is not None:
            tax_summary.append(per_code)
            continue
        supplies = _SUPPLIES.search(text)
        if supplies and numbers:
            tax_summary.append(
                (
                    TaxSummaryRow(
                        rate_percent=Decimal(supplies["rate"]), taxable_amount=numbers[-1][0]
                    ),
                    _TaxPos(
                        rate_percent=_Pos(index, *supplies.span("rate")),
                        taxable_amount=_at(index, numbers[-1]),
                    ),
                )
            )
            continue
        if in_summary:
            row = _summary_row(text, numbers, index)
            if row is not None:
                tax_summary.append(row)
                continue
            if _SUMMARY_TOTAL.search(text) and len(numbers) >= 2:  # noqa: PLR2004 - amount, tax
                continue  # the summary table's own total row, not a receipt total
        subtotal = _quantity_row_total(text)
        if subtotal is not None:
            labelled.append(
                _Labelled(
                    AmountRole.SUBTOTAL, subtotal[0], False, text, index, _at(index, subtotal)
                )
            )
        labelled.extend(_labelled_amounts(text, texts, index))
    return labelled, tax_summary


def _assign_roles(
    labelled: Sequence[_Labelled],
) -> tuple[int | None, dict[AmountRole, tuple[_Found, ...]]]:
    """The first receipt-total row and the amounts of each role, in print order."""
    first_total_row = min(
        (
            entry.label_row
            for entry in labelled
            if entry.role is not AmountRole.TENDERED
            and not _item_discount(entry.role, entry.value, entry.label_row, None)
        ),
        default=None,
    )
    amounts: dict[AmountRole, list[_Found]] = {}
    inclusive: list[_Found] = []
    for entry in labelled:
        if _item_discount(entry.role, entry.value, entry.label_row, first_total_row):
            continue  # a discount printed under an item belongs to the items
        if entry.role is AmountRole.TAX and _TOTAL_INCLUDES.search(entry.label):
            # resolved once all totals are known
            inclusive.append(_Found(entry.value, False, entry.label_row, entry.pos))
            continue
        amounts.setdefault(entry.role, []).append(
            _Found(entry.value, entry.explicit, entry.label_row, entry.pos)
        )
    resolved = _resolve_inclusive(amounts, inclusive)
    resolved = _resolve_total_before_tax(resolved)
    return first_total_row, _resolve_plain_totals(resolved)


def _at(index: int, number: _Number) -> _Pos:
    return _Pos(index, number[1], number[2])


def _item_discount(role: AmountRole, value: Decimal, index: int, first_total: int | None) -> bool:
    """A negative discount before the first receipt total is an item-level line.

    Receipt-level discounts follow the item list, or are printed unsigned; ``first_total`` is
    ``None`` while it is being determined.
    """
    return (
        role is AmountRole.DISCOUNT and value < 0 and (first_total is None or index < first_total)
    )


def _per_code_tax(
    text: str, numbers: Sequence[_Number], index: int
) -> tuple[TaxSummaryRow, _TaxPos] | None:
    """Per-code tax lines of fuel receipts, e.g. ``6.00% GST A RM 5.66``."""
    if not numbers:
        return None
    tax = numbers[-1]
    if match := _PER_CODE_TAX.search(text):
        return (
            TaxSummaryRow(code=match["code"], rate_percent=Decimal(match["rate"]), tax=tax[0]),
            _TaxPos(
                code=_Pos(index, *match.span("code")),
                rate_percent=_Pos(index, *match.span("rate")),
                tax=_at(index, tax),
            ),
        )
    if match := _RELIEF_TAX.search(text):
        return (
            TaxSummaryRow(code=match["code"], tax=tax[0]),
            _TaxPos(code=_Pos(index, *match.span("code")), tax=_at(index, tax)),
        )
    return None


def _quantity_row_total(text: str) -> _Number | None:
    """The unlabelled item total printed on a total-quantity row, e.g. ``TOTAL QTY: 3 14.20``."""
    match = _QTY_TOTAL.search(text)
    if match is None:
        return None
    numbers = _amount_matches(text[match.end() :])
    if not numbers:
        return None
    value, start, end = numbers[-1]
    return value, start + match.end(), end + match.end()


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.upper()).strip()


def _parse_amount(match: re.Match[str]) -> Decimal | None:
    try:
        value = Decimal(match["value"].replace(",", ""))
    except InvalidOperation:  # pragma: no cover - the pattern only admits valid decimals
        return None
    return -value if match["sign"] or match["trail"] else value


def _amount_matches(text: str) -> list[_Number]:
    """The amounts of a text, each with the span of its digits."""
    return [
        (value, match.start("value"), match.end("value"))
        for match in _AMOUNT.finditer(text)
        if (value := _parse_amount(match)) is not None
    ]


def _amounts(text: str) -> list[Decimal]:
    return [value for value, _, _ in _amount_matches(text)]


def _role(label: str) -> tuple[AmountRole, bool] | None:
    if not label or _NOT_A_TOTAL.search(label):
        return None
    for role, pattern in _ROLE_PATTERNS:
        if pattern.search(label):
            return role, bool(_EXPLICIT_FINAL.search(label))
    return None


def _labelled_amounts(text: str, texts: Sequence[str], index: int) -> list[_Labelled]:
    """Pair each role label in a row with the amount that follows it.

    A row like ``CASH 52.00 CHANGE 25.00`` yields two pairs. A label without an amount takes the
    amount of the next row if that row holds nothing but an amount.
    """
    pieces: list[tuple[str, _Number | None]] = []
    cursor = 0
    for match in _AMOUNT.finditer(text):
        label = text[cursor : match.start()].strip(" :")
        value = _parse_amount(match)
        number = None if value is None else (value, match.start("value"), match.end("value"))
        pieces.append((label, number))
        cursor = match.end()
    tail = text[cursor:].strip(" :")
    if tail and _HAS_LETTERS.search(tail):
        pieces.append((tail, None))

    pairs: list[_Labelled] = []
    for position, (label, printed) in enumerate(pieces):
        found = _role(label) if _HAS_LETTERS.search(label) else None
        if found is None:
            continue
        role, explicit = found
        is_last = position == len(pieces) - 1
        if printed is not None or not is_last:
            if printed is not None:
                pairs.append(
                    _Labelled(role, printed[0], explicit, label, index, _at(index, printed))
                )
            continue
        below = _amount_only_row(texts, index + 1)
        if below is not None:
            pairs.append(_Labelled(role, below[0], explicit, label, index, _at(index + 1, below)))
    return pairs


def _amount_only_row(texts: Sequence[str], index: int) -> _Number | None:
    if index >= len(texts) or _HAS_LETTERS.search(texts[index].replace("RM", "")):
        return None
    numbers = _amount_matches(texts[index])
    return numbers[-1] if len(numbers) == 1 else None


def _resolve_inclusive(
    amounts: dict[AmountRole, list[_Found]], inclusive: list[_Found]
) -> dict[AmountRole, list[_Found]]:
    """Resolve lines such as ``TOTAL INCLUDES GST 6% 9.00``.

    Receipts use this wording both for the tax contained in the total and for the total itself.
    If another final total is printed, the line states the tax; otherwise it is the total.
    """
    if inclusive:
        has_total = AmountRole.TOTAL_DUE in amounts or AmountRole.TOTAL_INCLUDING_TAX in amounts
        role = AmountRole.TAX if has_total else AmountRole.TOTAL_INCLUDING_TAX
        amounts.setdefault(role, []).extend(inclusive)
    return amounts


def _resolve_total_before_tax(
    amounts: dict[AmountRole, list[_Found]],
) -> dict[AmountRole, list[_Found]]:
    """A plain TOTAL followed by a tax line and a later total is the subtotal.

    For example ``TOTAL (MYR) 20.00 / GST @ 6% 1.20 / NET TOTAL 21.20``
    (docs/annotation-guidelines.md, section 3).
    """
    taxes = [found.label_row for found in amounts.get(AmountRole.TAX, [])]
    totals = [
        found.label_row
        for role in (AmountRole.TOTAL_DUE, AmountRole.TOTAL_INCLUDING_TAX)
        for found in amounts.get(role, [])
    ]
    kept: list[_Found] = []
    for found in amounts.get(AmountRole.TOTAL_DUE, []):
        later_tax = [t for t in taxes if t > found.label_row]
        if not found.explicit and later_tax and any(t > min(later_tax) for t in totals):
            amounts.setdefault(AmountRole.SUBTOTAL, []).append(found)
        else:
            kept.append(found)
    if AmountRole.TOTAL_DUE in amounts:
        amounts[AmountRole.TOTAL_DUE] = kept
    return amounts


def _resolve_plain_totals(
    amounts: dict[AmountRole, list[_Found]],
) -> dict[AmountRole, tuple[_Found, ...]]:
    """Reclassify totals printed before the final one.

    A plain TOTAL that precedes an explicit payable, rounded or nett total, and any total
    followed by a rounding line and a later total, is the amount before rounding, i.e. the total
    including tax.
    """
    due = sorted(amounts.get(AmountRole.TOTAL_DUE, []), key=lambda found: found.label_row)
    roundings = [found.label_row for found in amounts.get(AmountRole.ROUNDING, [])]
    last_explicit = max((n for n, found in enumerate(due) if found.explicit), default=-1)
    moved: list[_Found] = []
    kept: list[_Found] = []
    for n, found in enumerate(due):
        before_final = n < last_explicit and not found.explicit
        before_rounding = n < len(due) - 1 and any(
            found.label_row < r < due[-1].label_row for r in roundings
        )
        (moved if before_final or before_rounding else kept).append(found)
    if moved:
        amounts.setdefault(AmountRole.TOTAL_INCLUDING_TAX, []).extend(moved)
        amounts[AmountRole.TOTAL_DUE] = kept
    return {
        role: tuple(sorted(found, key=lambda f: f.label_row))
        for role, found in amounts.items()
        if found
    }


def _summary_row(
    text: str, numbers: Sequence[_Number], index: int
) -> tuple[TaxSummaryRow, _TaxPos] | None:
    match = _SUMMARY_ROW.search(text)
    if match is None or not numbers:
        return None
    taxable = numbers[0] if len(numbers) >= 2 else None  # noqa: PLR2004 - amount and tax columns
    return (
        TaxSummaryRow(
            code=match["code"],
            rate_percent=Decimal(match["rate"]),
            taxable_amount=None if taxable is None else taxable[0],
            tax=numbers[-1][0],
        ),
        _TaxPos(
            code=_Pos(index, *match.span("code")),
            rate_percent=_Pos(index, *match.span("rate")),
            taxable_amount=None if taxable is None else _at(index, taxable),
            tax=_at(index, numbers[-1]),
        ),
    )


def _items(texts: Sequence[str], *, end: int | None) -> tuple[tuple[LineItem, _ItemPos], ...]:
    """Line items between the header block and the first receipt total.

    A row with amounts is an item; rows without amounts carry descriptions. Receipts print a
    description on the item's own row, on the row(s) above it, or on the row(s) below it; the
    last two are told apart by whether text follows the last item row.
    """
    stop = end if end is not None else len(texts)
    numbered = [i for i in range(stop) if _amounts(texts[i])]
    if not numbered:
        return ()
    header = [
        i
        for i in range(numbered[0])
        if _ITEM_HEADER.search(texts[i]) or _HEADER_BLOCK.search(texts[i]) or _has_date(texts[i])
    ]
    start = header[-1] + 1 if header else 0
    stop = next(
        (
            i
            for i in range(start, stop)
            if _ITEMS_END.search(texts[i]) or (_TAX_LINE.search(texts[i]) and _amounts(texts[i]))
        ),
        stop,
    )
    fuel = [
        i for i in range(start, stop) if _LITRES.search(texts[i]) or _PER_LITRE.search(texts[i])
    ]
    numbered = [i for i in numbered if start <= i < stop and i not in fuel]
    if not numbered:
        return ()
    if fuel and len(numbered) == 1:
        return (_fuel_item(texts, numbered[0], fuel),)
    below = any(_describes(texts[i]) for i in range(numbered[-1] + 1, stop))
    items = []
    for position, index in enumerate(numbered):
        own, quantity, prices, amount = _item_row(texts[index])
        description, described = own, index
        if not _WORD.search(own):
            if below:
                following = numbered[position + 1] if position + 1 < len(numbered) else stop
                group = range(index + 1, following)
            else:
                previous = numbered[position - 1] if position else start - 1
                group = range(previous + 1, index)
            description, described = next(
                ((d, i) for i in group if _HAS_LETTERS.search(d := _describe(texts[i]))),
                (own, index),
            )
        chosen = _unit_price([price for price, _, _ in prices], quantity and quantity[0], amount[0])
        unit_price = None if chosen is None else prices[chosen]
        items.append(
            (
                LineItem(
                    description=description,
                    quantity=None if quantity is None else quantity[0],
                    unit_price=None if unit_price is None else unit_price[0],
                    amount=amount[0],
                ),
                _ItemPos(
                    description=described,
                    quantity=None if quantity is None else _at(index, quantity),
                    unit_price=None if unit_price is None else _at(index, unit_price),
                    amount=_at(index, amount),
                ),
            )
        )
    return tuple(items)


def _fuel_item(texts: Sequence[str], index: int, fuel: Sequence[int]) -> tuple[LineItem, _ItemPos]:
    """A fuel sale: litres and price per litre are printed on rows of their own."""
    text = texts[index]
    joined = " ".join(texts[i] for i in fuel)
    litres, per_litre = _LITRES.search(joined), _PER_LITRE.search(joined)
    amount = _amount_matches(text)[-1]
    has_price = per_litre is not None and per_litre["price"] is not None
    return (
        LineItem(
            description=_describe(text),
            quantity=Decimal(litres["qty"]) if litres else None,
            unit_price=Decimal(per_litre["price"]) if per_litre and has_price else None,
            amount=amount[0],
        ),
        _ItemPos(
            description=index,
            quantity=_in_rows(texts, fuel, litres.span("qty")) if litres else None,
            unit_price=_in_rows(texts, fuel, per_litre.span("price"))
            if per_litre and has_price
            else None,
            amount=_at(index, amount),
        ),
    )


def _in_rows(texts: Sequence[str], rows: Sequence[int], span: tuple[int, int]) -> _Pos:
    """The position of ``span`` of the rows' texts joined by single spaces."""
    cursor = 0
    for row in rows:
        row_end = cursor + len(texts[row])
        if span[0] < row_end:
            return _Pos(row, span[0] - cursor, span[1] - cursor)
        cursor = row_end + 1
    msg = f"span {span} lies beyond the joined rows"  # pragma: no cover - spans come from them
    raise ValueError(msg)  # pragma: no cover


def _item_row(text: str) -> tuple[str, _Number | None, list[_Number], _Number]:
    """Description, quantity, candidate unit prices and amount of a row with amounts.

    Numbers come with the span of their digits in ``text``.
    """
    amount = _amount_matches(text)[-1]
    quantity_match = (
        _QUANTITY.search(text) or _UNIT_QUANTITY.search(text) or _BARE_QUANTITY.search(text)
    )
    quantity: _Number | None = None
    description = text
    removed: tuple[int, int] | None = None
    if quantity_match:
        group = "qty" if quantity_match["qty"] is not None else "count"
        quantity = (Decimal(quantity_match[group]), *quantity_match.span(group))
        description = text[: quantity_match.start(group)]
        removed = quantity_match.span()
        text = text[: quantity_match.start()] + " " + text[quantity_match.end() :]
    prices = [
        (Decimal(m["value"]), *_restore(m.span("value"), removed)) for m in _PRICE.finditer(text)
    ][:-1]
    return _describe(description), quantity, prices, amount


def _restore(span: tuple[int, int], removed: tuple[int, int] | None) -> tuple[int, int]:
    """Map a span of a text whose characters ``removed`` became one space back to the original."""
    if removed is None:
        return span
    shift = removed[1] - removed[0] - 1
    start, end = span
    return (
        start if start < removed[0] else start + shift,
        end if end <= removed[0] else end + shift,
    )


def _describe(text: str) -> str:
    """A row's text without amounts, quantities, product codes and tax codes."""
    text = _QUANTITY.sub(" ", _PRICE.sub(" ", _AMOUNT.sub(" ", text)))
    text = _CODES.sub(" ", text.strip())
    text = re.sub(
        r"^(?:PKT|PCS|PC|ROLL|UNITS?|EA|BOX|BTL|SET|PAIR|CTN|BAG|TIN|PACK)\b", " ", text.strip()
    )
    text = re.sub(r"[^A-Z0-9()&./\-'+ ]", " ", text).strip()
    text = re.sub(r"\b(?:SRS?|ZRL?|S|Z|T|X|N|E|A|D|ST|NS)$", " ", text)
    return re.sub(r"\s+", " ", text).strip(" -")


def _describes(text: str) -> bool:
    return bool(_WORD.search(_describe(text))) and not _ITEM_COUNT.search(text)


def _has_date(text: str) -> bool:
    return any(p.search(text) for p in (_DATE_YMD, _DATE_TEXT, _DATE_DMY))


def _unit_price(prices: Sequence[Decimal], quantity: Decimal | None, amount: Decimal) -> int | None:
    """Which price column of an item row is the unit price.

    Rows may print several price columns (for example excluding and including tax). The unit
    price is the column for which quantity x price equals the amount; if none does, the first
    column is taken, so a manipulated line still reads as inconsistent.
    """
    if not prices:
        return None
    if quantity is not None:
        for n, price in enumerate(prices):
            if abs(quantity * price - amount) <= Decimal("0.01"):
                return n
    return 0


def _merchant(texts: Sequence[str]) -> tuple[str | None, tuple[int, ...]]:
    """The first business name in the header, without its registration number, and its rows.

    Names broken over two rows (``PASARAYA BORONG PINTAR`` / ``SDN BHD``) are joined.
    """
    head = texts[:8]
    for index, text in enumerate(head):
        if not _HAS_LETTERS.search(text) or _NOT_MERCHANT.search(text) or _NOT_NAME.search(text):
            continue
        name = text
        rows = [index]
        for offset, following in enumerate(head[index + 1 : index + 3], start=index + 1):
            if not _NAME_CONTINUES.search(following):
                break
            name = f"{name} {following}"
            rows.append(offset)
        name = _REGISTRATION.sub(" ", name)
        name = re.sub(r"\s+(?:PHONE|TEL)\b.*|[\s,.:]+$", "", re.sub(r"\s+", " ", name)).strip()
        if _HAS_LETTERS.search(name):
            return name, tuple(rows)
    return None, ()


def _dates(texts: Sequence[str]) -> tuple[tuple[str, _Pos], ...]:
    found: list[tuple[str, _Pos]] = []
    for index, text in enumerate(texts):
        for pattern in (_DATE_YMD, _DATE_TEXT, _DATE_DMY):
            for match in pattern.finditer(text):
                parsed = _to_date(match)
                if parsed is not None and parsed.isoformat() not in (d for d, _ in found):
                    found.append((parsed.isoformat(), _Pos(index, *match.span())))
    return tuple(found)


def _to_date(match: re.Match[str]) -> date | None:
    year = int(match["y"])
    year = year + 2000 if year < 100 else year  # noqa: PLR2004 - two-digit years mean 20YY
    month_text = match["m"]
    month = _MONTHS.get(month_text[:3]) if month_text.isalpha() else int(month_text)
    try:
        return date(year, month or 0, int(match["d"]))
    except ValueError:
        return None


def _first_int(pattern: re.Pattern[str], texts: Sequence[str]) -> tuple[int, _Pos] | None:
    found = next(((i, m) for i, t in enumerate(texts) if (m := pattern.search(t))), None)
    if found is None:
        return None
    index, match = found
    return int(match["n"]), _Pos(index, *match.span("n"))


def _first_decimal(pattern: re.Pattern[str], texts: Sequence[str]) -> tuple[Decimal, _Pos] | None:
    found = next(((i, m) for i, t in enumerate(texts) if (m := pattern.search(t))), None)
    if found is None:
        return None
    index, match = found
    return Decimal(match["n"]), _Pos(index, *match.span("n"))
