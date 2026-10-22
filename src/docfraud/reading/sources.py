"""Locating extracted values on the page: from a span of a row's text to segments and a box.

Readers work on row texts, but evidence linking needs the pixels. A :class:`Source` names a
span of one row's text together with the segments it touches and their box. The parser matches
its patterns on a normalised copy of each row (upper case, single spaces), so this module also
keeps, for every character of that copy, the character of the row it came from.
"""

import re
from collections.abc import Sequence
from dataclasses import dataclass

from docfraud.reading.fields import Source
from docfraud.reading.layout import Row, union

_SPACE = re.compile(r"\s")


@dataclass(frozen=True, slots=True)
class Normalised:
    """A row's text in upper case with whitespace collapsed, and where each character came from.

    ``origin[i]`` is the offset in the original text of character ``i`` of ``text``.
    """

    text: str
    origin: tuple[int, ...]

    def original_span(self, start: int, end: int) -> tuple[int, int]:
        """The span of the original text that characters ``start:end`` of ``text`` come from.

        Raises:
            ValueError: If the span is empty or outside the text.
        """
        if not 0 <= start < end <= len(self.text):
            msg = f"span {start}:{end} is not inside a text of length {len(self.text)}"
            raise ValueError(msg)
        return self.origin[start], self.origin[end - 1] + 1


def normalise(text: str) -> Normalised:
    r"""Upper-case ``text``, collapse whitespace runs to one space and strip both ends.

    The result equals ``re.sub(r"\s+", " ", text.upper()).strip()``, the normalisation the
    parser matches its patterns on.
    """
    chars: list[str] = []
    origin: list[int] = []
    for offset, char in enumerate(text):
        for upper in char.upper():
            if _SPACE.match(upper) and chars and chars[-1] == " ":
                continue
            chars.append(" " if _SPACE.match(upper) else upper)
            origin.append(offset)
    while chars and chars[0] == " ":
        del chars[0], origin[0]
    while chars and chars[-1] == " ":
        del chars[-1], origin[-1]
    return Normalised("".join(chars), tuple(origin))


def locate(rows: Sequence[Row], row: int, start: int, end: int) -> Source:
    """The source of characters ``start:end`` of ``rows[row].text``.

    Raises:
        ValueError: If the span is empty or outside the row's text.
    """
    segments = rows[row].segments
    if not 0 <= start < end <= len(rows[row].text):
        msg = f"span {start}:{end} is not inside row {row}"
        raise ValueError(msg)
    touched: list[int] = []
    cursor = 0
    for index, segment in enumerate(segments):
        segment_end = cursor + len(segment.text)
        if cursor < end and start < segment_end:
            touched.append(index)
        cursor = segment_end + 1  # Row.text joins segments with one space
    boxes = [box for i in touched if (box := segments[i].box) is not None]
    return Source(row=row, start=start, end=end, segments=tuple(touched), box=union(boxes))


def whole_row(rows: Sequence[Row], row: int) -> Source:
    """The source of a whole row, for values read from it as text."""
    return locate(rows, row, 0, len(rows[row].text))
