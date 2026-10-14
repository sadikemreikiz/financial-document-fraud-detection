"""Where training tiles lie on a page and what their pixels are labelled (ADR-0022, decision 3).

Tiles are ``TILE`` pixels square and start at multiples of the JPEG block size, so that every
tile keeps the 8 x 8 grid of the receipt aligned, as system 6's windows do. A tile around a box
(a modified region, or an OCR word for a tile without an edit) is drawn among the aligned
positions that keep the box inside with a margin of ``MARGIN`` pixels; along an axis where the
page edge leaves less room, the edge is the limit. Along an axis where no aligned position keeps
that margin, the margin shrinks to nothing, and where none even contains the box, the tile is
drawn among the positions that overlap it (:func:`axis_positions`).

Each pixel of a tile is labelled forged inside a modified rectangle, ignored inside a copy source
outside every modified rectangle, which the loss leaves out as ADR-0015 leaves it out of the
pixel metrics, and clean elsewhere (:func:`labels`).
"""

from collections.abc import Sequence

import numpy as np
import numpy.typing as npt

from docfraud.data.models import BBox
from docfraud.forensics.trufor import GRID

TILE = 384
"""Side of a training tile in pixels, a multiple of ``GRID``."""

MARGIN = 32
"""Pixels kept between a box and the edge of a tile drawn around it, where the page allows."""

CLEAN, FORGED, IGNORED = 0, 1, 2
"""Labels of the pixels of a tile."""

type Positions = npt.NDArray[np.int64]


def _aligned(length: int) -> Positions:
    """Every aligned start of a tile inside an axis of ``length`` pixels.

    Raises:
        ValueError: If the axis is shorter than a tile.
    """
    if length < TILE:
        msg = f"a page side of {length} px is shorter than a tile of {TILE} px"
        raise ValueError(msg)
    return np.arange(0, length - TILE + 1, GRID, dtype=np.int64)


def axis_positions(start: int, end: int, length: int) -> Positions:
    """Aligned starts of a tile around the span ``[start, end)`` of an axis of ``length`` pixels.

    The first non-empty of: the starts that keep a margin of ``MARGIN`` pixels on both sides of
    the span, or as much as the page edge allows; those that contain the span; those that
    overlap it. Empty only for a span in the last few pixels of an axis, beyond the last aligned
    tile.

    Raises:
        ValueError: If the span is empty or outside the axis, or the axis is shorter than a tile.
    """
    if not 0 <= start < end <= length:
        msg = f"the span [{start}, {end}) does not lie in an axis of {length} px"
        raise ValueError(msg)
    starts = _aligned(length)
    last = int(starts[-1])
    contains = (starts <= start) & (starts + TILE >= end)
    for margin in (MARGIN, 0):
        lowest = min(max(end + margin - TILE, 0), last)
        highest = max(min(start - margin, last), 0)
        kept = starts[contains & (starts >= lowest) & (starts <= highest)]
        if kept.size:
            return kept
    return starts[(starts > start - TILE) & (starts < end)]


def position_around(
    box: BBox, shape: tuple[int, int], rng: np.random.Generator
) -> tuple[int, int] | None:
    """The top-left corner ``(y, x)`` of a tile drawn around a box; ``None`` if none can hold it.

    Each axis is drawn uniformly among its :func:`axis_positions`, which makes the corner
    uniform among the positions allowed on both axes.
    """
    height, width = shape
    rows = axis_positions(box.y, min(box.y2, height), height)
    columns = axis_positions(box.x, min(box.x2, width), width)
    if not rows.size or not columns.size:
        return None
    return int(rng.choice(rows)), int(rng.choice(columns))


def uniform_position(shape: tuple[int, int], rng: np.random.Generator) -> tuple[int, int]:
    """The top-left corner ``(y, x)`` of a tile drawn uniformly among the aligned positions."""
    height, width = shape
    return int(rng.choice(_aligned(height))), int(rng.choice(_aligned(width)))


def labels(
    corner: tuple[int, int], modified: Sequence[BBox], ignored: Sequence[BBox]
) -> npt.NDArray[np.uint8]:
    """The label of every pixel of the tile whose top-left corner is ``corner``.

    ``FORGED`` inside a modified rectangle, ``IGNORED`` inside an ignored box (a copy source)
    and outside every modified rectangle, ``CLEAN`` elsewhere.
    """
    y, x = corner
    tile = np.full((TILE, TILE), CLEAN, dtype=np.uint8)
    for value, boxes in ((IGNORED, ignored), (FORGED, modified)):
        for box in boxes:
            top, bottom = max(box.y - y, 0), min(box.y2 - y, TILE)
            left, right = max(box.x - x, 0), min(box.x2 - x, TILE)
            if top < bottom and left < right:
                tile[top:bottom, left:right] = value
    return tile


def cut(page: npt.NDArray[np.uint8], corner: tuple[int, int]) -> npt.NDArray[np.uint8]:
    """A copy of the tile of a page whose top-left corner is ``corner``.

    Raises:
        ValueError: If the tile does not lie inside the page.
    """
    y, x = corner
    height, width = page.shape
    if not (0 <= y <= height - TILE and 0 <= x <= width - TILE):
        msg = f"a tile at {corner} does not lie inside a page of {page.shape}"
        raise ValueError(msg)
    return np.array(page[y : y + TILE, x : x + TILE])


def flip(tile: npt.NDArray[np.uint8], *, vertical: bool, horizontal: bool) -> npt.NDArray[np.uint8]:
    """The tile mirrored top to bottom and/or left to right.

    Every pixel value is kept, and because ``TILE`` is a multiple of 8 the mirrored tile still
    has the JPEG grid aligned: mirroring an 8 x 8 block only changes the sign of some of its
    coefficients.
    """
    axes = tuple(axis for axis, flipped in ((0, vertical), (1, horizontal)) if flipped)
    return np.ascontiguousarray(np.flip(tile, axis=axes)) if axes else tile
