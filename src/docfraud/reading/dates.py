"""Printed dates: the formats receipts use and their reading as calendar dates.

Shared by the parser and the labels of the LLM reader, so that both readers recognise the same
dates. The patterns match upper-case text.
"""

import re
from datetime import date

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

#: Date formats, in the order in which a reader tries them.
DATE_PATTERNS: tuple[re.Pattern[str], ...] = (_DATE_YMD, _DATE_TEXT, _DATE_DMY)


def to_date(match: re.Match[str]) -> date | None:
    """The calendar date of a match of one of :data:`DATE_PATTERNS`, or ``None`` if invalid."""
    year = int(match["y"])
    year = year + 2000 if year < 100 else year  # noqa: PLR2004 - two-digit years mean 20YY
    month_text = match["m"]
    month = _MONTHS.get(month_text[:3]) if month_text.isalpha() else int(month_text)
    try:
        return date(year, month or 0, int(match["d"]))
    except ValueError:
        return None


def find_dates(text: str) -> list[tuple[date, int, int]]:
    """The valid dates in an upper-case text with their spans, left to right, not overlapping.

    Where matches of different formats overlap, the one that starts first is kept, and of two
    that start at the same place, the format tried first.
    """
    found = [
        (parsed, match.start(), match.end())
        for pattern in DATE_PATTERNS
        for match in pattern.finditer(text)
        if (parsed := to_date(match)) is not None
    ]
    kept: list[tuple[date, int, int]] = []
    for item in sorted(found, key=lambda f: f[1]):  # stable: ties keep the format order
        if not kept or item[1] >= kept[-1][2]:
            kept.append(item)
    return kept
