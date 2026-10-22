"""Labels on every number and date of a receipt's rows: the only values an LLM reader can name.

The model reads the same rows the parser reads, numbered, with a label in front of every number
and date, and answers with labels; code takes the values from the text (ADR-0025, decision 1).
A model therefore cannot write a value that is not printed, and a forged value is read as
printed. Labels are delimited by white square brackets, which no receipt text of the dataset
contains, so that they cannot be confused with printed brackets such as ``[50G]``.
"""

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import date
from decimal import Decimal, InvalidOperation
from enum import StrEnum

from docfraud.reading.dates import find_dates
from docfraud.reading.fields import Source
from docfraud.reading.layout import Row
from docfraud.reading.sources import locate, normalise

OPEN = "\N{MATHEMATICAL LEFT WHITE SQUARE BRACKET}"
CLOSE = "\N{MATHEMATICAL RIGHT WHITE SQUARE BRACKET}"

_DIGITS = re.compile(r"(?<![\d.,])(?:\d+(?:[.,]\d+)*|\.\d+)")
_BEFORE_SIGN = frozenset(" (:=@/")  # characters after which a hyphen is a minus sign
_CURRENCY = "RM"


class LabelKind(StrEnum):
    """What a label stands for."""

    NUMBER = "number"
    DATE = "date"


@dataclass(frozen=True, slots=True)
class Label:
    """One labelled number or date.

    ``printed`` is the text as printed, with a number's sign; ``source`` covers a number's
    digits, or the whole date. A number has its value in ``number`` (``None`` if its separators
    cannot be read unambiguously), a date its calendar day in ``day``.
    """

    name: str
    kind: LabelKind
    printed: str
    source: Source
    number: Decimal | None = None
    day: date | None = None


@dataclass(frozen=True, slots=True)
class LabelledRows:
    """The rows as the reader sees them (one line ``r<i>: ...`` per row) and the labels."""

    text: str
    labels: Mapping[str, Label]


@dataclass(frozen=True, slots=True)
class _Found:
    """A number or date before it is named: ``at`` is where its label goes."""

    at: int
    kind: LabelKind
    printed: str
    source: Source
    number: Decimal | None = None
    day: date | None = None


def label_rows(rows: Sequence[Row]) -> LabelledRows:
    """Number the rows and put a label in front of every date and number in them.

    Dates are labelled ``d1``, ``d2``, ... and numbers ``n1``, ``n2``, ... in reading order;
    the digits of a date are not labelled again as numbers.
    """
    lines: list[str] = []
    labels: dict[str, Label] = {}
    counts = dict.fromkeys(LabelKind, 0)
    for index, row in enumerate(rows):
        pieces: list[str] = []
        cursor = 0
        for found in sorted(_find(rows, index), key=lambda f: f.at):
            counts[found.kind] += 1
            name = f"{found.kind.value[0]}{counts[found.kind]}"
            labels[name] = Label(
                name, found.kind, found.printed, found.source, found.number, found.day
            )
            pieces += [row.text[cursor : found.at], f"{OPEN}{name}{CLOSE}"]
            cursor = found.at
        pieces.append(row.text[cursor:])
        lines.append(f"r{index}: {''.join(pieces)}")
    return LabelledRows(text="\n".join(lines), labels=labels)


def _find(rows: Sequence[Row], index: int) -> list[_Found]:
    """The dates and numbers of one row."""
    text = rows[index].text
    normalised = normalise(text)
    found: list[_Found] = []
    for day, start, end in find_dates(normalised.text):
        begin, finish = normalised.original_span(start, end)
        source = locate(rows, index, begin, finish)
        found.append(_Found(begin, LabelKind.DATE, text[begin:finish], source, day=day))
    dates = [(f.at, f.at + len(f.printed)) for f in found]
    for match in _DIGITS.finditer(text):
        start, end = match.span()
        if any(start < finish and begin < end for begin, finish in dates):
            continue
        sign = _sign_start(text, start)
        trailing = _trailing_minus(text, end)
        value = _read_number(match.group())
        if value is not None and (sign is not None or trailing):
            value = -value
        first = start if sign is None else sign
        printed = text[first : end + 1 if trailing else end]
        source = locate(rows, index, start, end)
        found.append(_Found(first, LabelKind.NUMBER, printed, source, number=value))
    return found


def _sign_start(text: str, digits: int) -> int | None:
    """Where the minus sign of the number whose digits start at ``digits`` begins, if any.

    Accepted forms: ``-0.01``, ``- 0.01``, ``RM-0.20``, ``RM -0.20``, ``-RM0.02``, ``-RM 0.02``
    and ``- RM0.02``. A hyphen inside a word, as in ``TP-24``, is not a sign.
    """
    cursor = digits
    if cursor > 0 and text[cursor - 1] == " ":
        cursor -= 1
    if text[max(cursor - 2, 0) : cursor].upper() == _CURRENCY:
        cursor -= 2
        if cursor > 0 and text[cursor - 1] == " ":
            cursor -= 1
    if cursor == 0 or text[cursor - 1] != "-":
        return None
    hyphen = cursor - 1
    before = text[:hyphen]
    if not before or before[-1] in _BEFORE_SIGN or before[-2:].upper() == _CURRENCY:
        return hyphen
    return None


def _trailing_minus(text: str, end: int) -> bool:
    """Whether the number ending at ``end`` is followed by a minus sign, as in ``2.00-``."""
    if end >= len(text) or text[end] != "-":
        return False
    rest = text[end + 1 :].lstrip(" ")
    return not rest[:1].isdigit()


def _read_number(digits: str) -> Decimal | None:
    """The value of digits with separators, or ``None`` if the separators are ambiguous.

    A comma is a thousands separator before groups of three digits or when a point follows it;
    a single comma before one or two digits is a decimal comma. Several points cannot be read.
    """
    if "," in digits and "." in digits:
        if digits.rfind(".") < digits.rfind(","):
            digits = digits.replace(".", "").replace(",", ".")
        else:
            digits = digits.replace(",", "")
    elif "," in digits:
        head, *groups = digits.split(",")
        if len(head) <= 3 and all(len(g) == 3 for g in groups):  # noqa: PLR2004 - thousands
            digits = digits.replace(",", "")
        elif len(groups) == 1 and len(groups[0]) <= 2:  # noqa: PLR2004 - decimal comma
            digits = f"{head}.{groups[0]}"
        else:
            return None
    if digits.count(".") > 1:
        return None
    try:
        return Decimal(digits)
    except InvalidOperation:  # pragma: no cover - only digits and one point remain
        return None
