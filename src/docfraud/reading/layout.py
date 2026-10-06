"""Page layout: rebuilding printed rows from positioned text.

Transcript line order is not reading order: labels such as ``TOTAL`` and ``CASH`` may be listed
together and their amounts several lines later. What reliably ties a label to its value is that
both are printed on the same row. This module groups positioned text into rows.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

from docfraud.data.models import BBox


@dataclass(frozen=True, slots=True)
class Segment:
    """A piece of text and, when known, where it is printed."""

    text: str
    box: BBox | None = None


@dataclass(frozen=True, slots=True)
class Row:
    """Segments printed on the same row, left to right."""

    segments: tuple[Segment, ...]

    @property
    def text(self) -> str:
        """The row as one string, segments separated by single spaces."""
        return " ".join(s.text for s in self.segments)

    @property
    def box(self) -> BBox | None:
        """The box enclosing all positioned segments of the row."""
        return union([s.box for s in self.segments if s.box is not None])


def union(boxes: Sequence[BBox]) -> BBox | None:
    """Smallest box enclosing all given boxes, or ``None`` for no boxes."""
    if not boxes:
        return None
    x0, y0 = min(b.x for b in boxes), min(b.y for b in boxes)
    x1, y1 = max(b.x2 for b in boxes), max(b.y2 for b in boxes)
    return BBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


def group_rows(segments: Iterable[Segment], *, min_overlap: float = 0.5) -> list[Row]:
    """Group positioned segments into rows, top to bottom.

    A segment joins a row when its vertical extent overlaps the row's by at least
    ``min_overlap`` of the smaller height. Comparing against the row's running extent rather
    than its first member tolerates slightly skewed scans.

    Segments without a box cannot be placed and are returned as one-segment rows at the end, in
    their original order.

    Args:
        segments: Text pieces, typically OCR lines or words, or aligned transcript lines.
        min_overlap: Required vertical overlap as a fraction of the smaller height.
    """
    items = list(segments)
    placed = sorted(
        ((s, s.box) for s in items if s.box is not None), key=lambda pair: _center_y(pair[1])
    )
    rows: list[list[Segment]] = []
    extents: list[tuple[int, int]] = []
    for segment, box in placed:
        for index, (top, bottom) in enumerate(extents):
            overlap = min(bottom, box.y2) - max(top, box.y)
            if overlap >= min_overlap * min(bottom - top, box.height):
                rows[index].append(segment)
                extents[index] = (min(top, box.y), max(bottom, box.y2))
                break
        else:
            rows.append([segment])
            extents.append((box.y, box.y2))
    ordered = sorted(zip(extents, rows, strict=True), key=lambda pair: pair[0][0])
    result = [Row(tuple(sorted(row, key=_left))) for _, row in ordered]
    return result + [Row((s,)) for s in items if s.box is None]


def _center_y(box: BBox) -> float:
    return box.y + box.height / 2


def _left(segment: Segment) -> int:
    return segment.box.x if segment.box is not None else 0
