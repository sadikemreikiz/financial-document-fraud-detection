"""Rule-based receipt parser: rows of positioned text to :class:`ReceiptFields`.

The parser reads amounts as printed and assigns them roles from their labels. It never checks or
repairs the arithmetic: that is the job of the accounting rules, which need the inconsistencies
of forged receipts intact. Label vocabulary and role definitions follow
``docs/annotation-guidelines.md``.
"""

import re
from collections.abc import Sequence
from datetime import date
from decimal import Decimal, InvalidOperation

from docfraud.reading.fields import AmountRole, LineItem, ReceiptFields, TaxSummaryRow
from docfraud.reading.layout import Row

_AMOUNT = re.compile(
    r"(?<![\w.,])(?:RM|MYR|\$)?\s?(?P<sign>-)?\s?(?:RM|MYR|\$)?\s?"
    r"(?P<value>\d{1,3}(?:,\d{3})+\.\d{2}|\d{1,7}\.\d{2})(?P<trail>-)?(?![\w.,]*\d)"
)
_HAS_LETTERS = re.compile(r"[A-Z]{2,}")

# Ordered: the first pattern that matches a label decides its role.
_ROLE_PATTERNS: tuple[tuple[AmountRole, re.Pattern[str]], ...] = (
    (
        AmountRole.TOTAL_DUE,
        re.compile(
            r"\bTOTAL\b.*\bAFTER\b|\bROUNDED\b|\b(?:TOTAL|AMT|AMOUNT)\b.*\bPAYABLE\b"
            r"|\bNETT?\s+TOTAL\b|\bGRAND\b"
        ),
    ),
    (AmountRole.ROUNDING, re.compile(r"\bROUND(?:ING)?\b|\bRND\b|^ADJ(?:USTMENT)?\b")),
    (AmountRole.CHANGE, re.compile(r"\bCHANGE\b|\bBAKI\b")),
    (
        AmountRole.TAX,
        re.compile(
            r"\b(?:GST|SST|TAX)\b.*\bINCLUDED\b"
            r"|^(?:TOTAL\s+)?(?:GST|SST|TAX)\b(?!.*\b(?:ID|REG|NO|CODE|SUMMARY|INVOICE)\b)"
            r"|\b(?:GST|SST|TAX) (?:PAYABLE|AMOUNT)\b"
        ),
    ),
    (
        AmountRole.TOTAL_EXCLUDING_TAX,
        re.compile(r"\bTOTAL\b.*\bEXC(?:L|LUDING|\.)?\b|\bAMOUNT\b.*\bEXCL"),
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
    (AmountRole.TOTAL_DUE, re.compile(r"\bTOTAL\b|\bNETT?\b|\bAMOUNT DUE\b|\bGRAND\b")),
)
_EXPLICIT_FINAL = re.compile(r"PAYABLE|ROUNDED|NETT?\b|GRAND|FINAL|AMOUNT DUE|TOTAL DUE|\bAFTER\b")
_NOT_A_TOTAL = re.compile(
    r"\bTOTAL\s+(?:QTY|QUANTITY|ITEMS?|PCS|SAVINGS?|DISCOUNT|POINTS?|WEIGHT)\b|\d+\s?%\s*SUPPLIES"
)
_SUPPLIES = re.compile(r"(?P<rate>\d+(?:\.\d+)?)\s?%\s*SUPPLIES")
_SUMMARY_HEADER = re.compile(r"\b(?:GST|SST|TAX)\s+SUMMARY\b|\bTAX CODE\b|\bSUMMARY\b")
_SUMMARY_TOTAL = re.compile(r"^TOTAL\b")
_SUMMARY_ROW = re.compile(r"^(?P<code>[A-Z]{1,3})\b.*?(?P<rate>\d+(?:\.\d+)?)\s?%")
_ITEM_HEADER = re.compile(r"\b(?:DESCRIPTION|ITEM|PRODUCT|QTY|U\.?\s?PRICE|PRICE|AMT|AMOUNT)\b")
_QUANTITY = re.compile(r"(?<![\d.])(?P<qty>\d{1,4}(?:\.\d{1,3})?)\s?[Xx*]\s")
_BARE_QUANTITY = re.compile(r"(?:^|\s)(?P<qty>\d{1,3})\s+(?=(?:RM|\$)?\s?-?\d{1,7}\.\d{2}\b)")
_ITEMS_END = re.compile(
    r"\bTOT(?:AL)?\.?\s*(?:QTY|QUANTITY|ITEMS?)\b|\bITEM\(?S\)?\s*:|\bNO\.?\s*OF\s*ITEMS?\b"
    r"|\bSUB[\s-]?TOTAL\b|^TOTAL\b|^(?:GST|SST|TAX)\b"
)
_ITEM_COUNT = re.compile(
    r"\b(?:ITEM\(?S\)?|NO\.?\s*OF\s*ITEMS?|TOTAL\s+ITEMS?)\s*:?\s*(?P<n>\d{1,3})\b"
)
_QTY_TOTAL = re.compile(
    r"\b(?:TOTAL\s+QTY|QTY\(?S\)?|TOTAL\s+QUANTITY)\s*:?\s*(?P<n>\d{1,4}(?:\.\d+)?)\b"
)
_MERCHANT_HINT = re.compile(
    r"\b(?:SDN\.?\s*BHD|BHD|ENTERPRISE|TRADING|RESTORAN|RESTAURANT|STORE|MART|SHOP|CAFE|"
    r"HARDWARE|PHARMACY|BAKERIES|BOOK|STATIONERY|CO\.)\b"
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


def parse_receipt(rows: Sequence[Row]) -> ReceiptFields:
    """Extract structured fields from rows in reading order.

    Args:
        rows: Printed rows, top to bottom, as produced by
            :func:`docfraud.reading.layout.group_rows`.
    """
    texts = [_clean(row.text) for row in rows]
    amounts: dict[AmountRole, list[tuple[Decimal, bool]]] = {}
    tax_summary: list[TaxSummaryRow] = []
    first_total_row: int | None = None
    in_summary = False

    for index, text in enumerate(texts):
        if _SUMMARY_HEADER.search(text):
            in_summary = True
            continue
        supplies = _SUPPLIES.search(text)
        values = _amounts(text)
        if supplies and values:
            tax_summary.append(
                TaxSummaryRow(rate_percent=Decimal(supplies["rate"]), taxable_amount=values[-1])
            )
            continue
        if in_summary:
            row = _summary_row(text, values)
            if row is not None:
                tax_summary.append(row)
                continue
            if _SUMMARY_TOTAL.search(text) and len(values) >= 2:  # noqa: PLR2004 - amount, tax
                continue  # the summary table's own total row, not a receipt total
        for role, value, explicit in _labelled_amounts(text, texts, index):
            amounts.setdefault(role, []).append((value, explicit))
            if first_total_row is None and role is not AmountRole.TENDERED:
                first_total_row = index

    return ReceiptFields(
        merchant=_merchant(texts),
        dates=_dates(texts),
        items=_items(texts, end=first_total_row),
        amounts=_resolve_plain_totals(amounts),
        tax_summary=tuple(tax_summary),
        item_count=_first_int(_ITEM_COUNT, texts),
        quantity_total=_first_decimal(_QTY_TOTAL, texts),
    )


def _clean(text: str) -> str:
    return re.sub(r"\s+", " ", text.upper()).strip()


def _parse_amount(match: re.Match[str]) -> Decimal | None:
    try:
        value = Decimal(match["value"].replace(",", ""))
    except InvalidOperation:  # pragma: no cover - the pattern only admits valid decimals
        return None
    return -value if match["sign"] or match["trail"] else value


def _amounts(text: str) -> list[Decimal]:
    return [v for m in _AMOUNT.finditer(text) if (v := _parse_amount(m)) is not None]


def _role(label: str) -> tuple[AmountRole, bool] | None:
    if not label or _NOT_A_TOTAL.search(label):
        return None
    for role, pattern in _ROLE_PATTERNS:
        if pattern.search(label):
            return role, bool(_EXPLICIT_FINAL.search(label))
    return None


def _labelled_amounts(
    text: str, texts: Sequence[str], index: int
) -> list[tuple[AmountRole, Decimal, bool]]:
    """Pair each role label in a row with the amount that follows it.

    A row like ``CASH 52.00 CHANGE 25.00`` yields two pairs. A label without an amount takes the
    amount of the next row if that row holds nothing but an amount.
    """
    pieces: list[tuple[str, Decimal | None]] = []
    cursor = 0
    for match in _AMOUNT.finditer(text):
        label = text[cursor : match.start()].strip(" :")
        pieces.append((label, _parse_amount(match)))
        cursor = match.end()
    tail = text[cursor:].strip(" :")
    if tail and _HAS_LETTERS.search(tail):
        pieces.append((tail, None))

    pairs: list[tuple[AmountRole, Decimal, bool]] = []
    for position, (label, printed) in enumerate(pieces):
        found = _role(label) if _HAS_LETTERS.search(label) else None
        if found is None:
            continue
        role, explicit = found
        is_last = position == len(pieces) - 1
        value = (
            printed if printed is not None or not is_last else _amount_only_row(texts, index + 1)
        )
        if value is not None:
            pairs.append((role, value, explicit))
    return pairs


def _amount_only_row(texts: Sequence[str], index: int) -> Decimal | None:
    if index >= len(texts) or _HAS_LETTERS.search(texts[index].replace("RM", "")):
        return None
    values = _amounts(texts[index])
    return values[-1] if len(values) == 1 else None


def _resolve_plain_totals(
    amounts: dict[AmountRole, list[tuple[Decimal, bool]]],
) -> dict[AmountRole, tuple[Decimal, ...]]:
    """Reclassify a plain TOTAL that precedes an explicit payable, rounded or nett total.

    Such a plain total is the amount before rounding, i.e. the total including tax.
    """
    due = amounts.get(AmountRole.TOTAL_DUE, [])
    if any(explicit for _, explicit in due[1:]):
        last_explicit = max(i for i, (_, explicit) in enumerate(due) if explicit)
        moved = [v for i, (v, explicit) in enumerate(due) if i < last_explicit and not explicit]
        if moved:
            amounts.setdefault(AmountRole.TOTAL_INCLUDING_TAX, []).extend((v, False) for v in moved)
            amounts[AmountRole.TOTAL_DUE] = [
                pair for i, pair in enumerate(due) if not (i < last_explicit and not pair[1])
            ]
    return {role: tuple(v for v, _ in pairs) for role, pairs in amounts.items() if pairs}


def _summary_row(text: str, values: list[Decimal]) -> TaxSummaryRow | None:
    match = _SUMMARY_ROW.search(text)
    if match is None or not values:
        return None
    taxable = values[0] if len(values) >= 2 else None  # noqa: PLR2004 - amount and tax columns
    return TaxSummaryRow(
        code=match["code"],
        rate_percent=Decimal(match["rate"]),
        taxable_amount=taxable,
        tax=values[-1],
    )


def _items(texts: Sequence[str], *, end: int | None) -> tuple[LineItem, ...]:
    start = next((i + 1 for i, t in enumerate(texts) if _ITEM_HEADER.search(t)), 0)
    stop = end if end is not None else len(texts)
    stop = next((i for i in range(start, stop) if _ITEMS_END.search(texts[i])), stop)
    items: list[LineItem] = []
    pending: list[str] = []
    for text in texts[start:stop]:
        values = _amounts(text)
        if not values:
            if _HAS_LETTERS.search(text) and not _ITEM_COUNT.search(text):
                pending.append(text)
            continue
        quantity_match = _QUANTITY.search(text) or _BARE_QUANTITY.search(text)
        description = text[: quantity_match.start("qty")] if quantity_match else text
        description = _QUANTITY.sub(" ", _AMOUNT.sub(" ", description))
        description = re.sub(r"\b[A-Z]{1,3}\b$|[^A-Z0-9()&./\- ]", " ", description.strip())
        description = re.sub(r"\s+", " ", description).strip()
        if not _HAS_LETTERS.search(description) and pending:
            description = pending[-1]
        pending.clear()
        items.append(
            LineItem(
                description=description,
                quantity=Decimal(quantity_match["qty"]) if quantity_match else None,
                unit_price=values[0] if len(values) >= 2 else None,  # noqa: PLR2004
                amount=values[-1],
            )
        )
    return tuple(items)


def _merchant(texts: Sequence[str]) -> str | None:
    head = [t for t in texts[:6] if _HAS_LETTERS.search(t)]
    for text in head:
        if _MERCHANT_HINT.search(text) and not _NOT_MERCHANT.search(text):
            return text
    return next((t for t in head if not _NOT_MERCHANT.search(t)), None)


def _dates(texts: Sequence[str]) -> tuple[str, ...]:
    found: list[str] = []
    for text in texts:
        for pattern in (_DATE_YMD, _DATE_TEXT, _DATE_DMY):
            for match in pattern.finditer(text):
                parsed = _to_date(match)
                if parsed is not None and parsed.isoformat() not in found:
                    found.append(parsed.isoformat())
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


def _first_int(pattern: re.Pattern[str], texts: Sequence[str]) -> int | None:
    match = next((m for t in texts if (m := pattern.search(t))), None)
    return int(match["n"]) if match else None


def _first_decimal(pattern: re.Pattern[str], texts: Sequence[str]) -> Decimal | None:
    match = next((m for t in texts if (m := pattern.search(t))), None)
    return Decimal(match["n"]) if match else None
