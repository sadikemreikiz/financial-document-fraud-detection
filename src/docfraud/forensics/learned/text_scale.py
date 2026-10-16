"""The scale of a receipt's text, for the document scores of ADR-0023 and their diagnostic.

A page scan holds one receipt at a higher resolution: about as many words as another receipt,
with letters about 2.3 times as high. The highest value of a map is a maximum over pixels, so a
word of a page scan gives it more chances. ADR-0023 therefore adds document scores that average
the map over squares tied to the receipt's text height before the highest value is taken
(:func:`highest_square_mean`), and the development report shows the diagnostic behind that record:
the highest value inside each OCR word box, after averaging over blocks at four scales
(:func:`word_peaks`).

The text height is the median height of the receipt's OCR word boxes (:func:`text_height`).
"""

from collections.abc import Sequence
from enum import StrEnum

import numpy as np
import numpy.typing as npt

from docfraud.data.models import BBox

PAGE_SCAN_SIDE = 6000
"""Pixels of the longer side above which a receipt is a page scan."""

TEXT_HEIGHT_PAGE_SCAN = 91.0
"""Median text height of the training split's page scans, for one without any OCR word."""

TEXT_HEIGHT_OTHER = 40.0
"""Median text height of the training split's other receipts, for one without any OCR word."""

QUARTER = 4
"""The text height over the side of the square of score 3 (ADR-0023, decision 1)."""

HALF = 2
"""The text height over the side of the square of score 4."""

FIXED_SIDE = 10
"""Side in pixels of the diagnostic's control blocks, the same on every receipt."""

_ROWS = 512
"""Rows of square positions measured at once, which bounds memory and rounding."""


class WordScale(StrEnum):
    """The blocks over which the diagnostic of ADR-0023 averages the map inside a word box."""

    PIXEL = "pixel"  # blocks of 1 px: the highest value inside the box
    QUARTER = "a quarter of the text height"
    HALF = "half the text height"
    FIXED = f"{FIXED_SIDE} px on every receipt"  # the control


def page_scan(height: int, width: int) -> bool:
    """Whether a receipt of this size is a page scan."""
    return max(height, width) > PAGE_SCAN_SIDE


def on_page(words: Sequence[BBox], height: int, width: int) -> list[BBox]:
    """The word boxes that start inside the page; their parts outside it are not measured."""
    return [box for box in words if box.x < width and box.y < height]


def text_height(words: Sequence[BBox], height: int, width: int) -> float:
    """The median height of the receipt's word boxes, or that of its kind without any word."""
    boxes = on_page(words, height, width)
    if not boxes:
        return TEXT_HEIGHT_PAGE_SCAN if page_scan(height, width) else TEXT_HEIGHT_OTHER
    return float(np.median([box.height for box in boxes]))


def square_side(height: float, divisor: int) -> int:
    """The side of a score's square: the text height over ``divisor``, rounded, at least 1 px.

    Rounding is Python's, to the nearest integer with ties to even, as in the diagnostic.
    """
    return max(1, round(height / divisor))


def highest_square_mean(values: npt.NDArray[np.float64], side: int) -> float:
    """The highest mean of the map over a ``side`` x ``side`` square inside the page.

    Every position counts. A side longer than the page is shortened to it. The sums are made a
    band of rows at a time, so that rounding stays far below the differences between maps.

    Raises:
        ValueError: If the side is not positive.
    """
    if side < 1:
        msg = f"a square has a positive side, not {side}"
        raise ValueError(msg)
    height, width = values.shape
    side = min(side, height, width)
    best = -np.inf
    for start in range(0, height - side + 1, _ROWS):
        stop = min(start + _ROWS, height - side + 1)  # the squares whose top row is in the band
        band = np.asarray(values[start : stop + side - 1], dtype=np.float64)
        down = np.zeros((band.shape[0] + 1, width))
        np.cumsum(band, axis=0, out=down[1:])
        columns = down[side:] - down[:-side]  # sums of ``side`` rows, one per top row and column
        across = np.zeros((columns.shape[0], width + 1))
        np.cumsum(columns, axis=1, out=across[:, 1:])
        best = max(best, float((across[:, side:] - across[:, :-side]).max()))
    return best / (side * side)


def block_peak(cell: npt.NDArray[np.float64], side: int) -> float:
    """The highest mean over the ``side`` x ``side`` blocks of a word box, from its corner.

    Blocks do not overlap and incomplete blocks at the far edges are left out; a box smaller than
    ``side`` in a direction is one block in that direction.
    """
    height, width = cell.shape
    rows, columns = min(side, height), min(side, width)
    kept_rows, kept_columns = (height // rows) * rows, (width // columns) * columns
    blocks = cell[:kept_rows, :kept_columns].reshape(
        kept_rows // rows, rows, kept_columns // columns, columns
    )
    return float(blocks.mean(axis=(1, 3)).max())


def word_sides(height: float) -> dict[WordScale, int]:
    """The block side of each scale of the diagnostic, for a receipt of this text height."""
    return {
        WordScale.PIXEL: 1,
        WordScale.QUARTER: square_side(height, QUARTER),
        WordScale.HALF: square_side(height, HALF),
        WordScale.FIXED: FIXED_SIDE,
    }


def word_peaks(
    values: npt.NDArray[np.float64], words: Sequence[BBox]
) -> dict[WordScale, tuple[float, ...]]:
    """For every scale, the block peak of each word box on the page, in the order of the boxes."""
    height, width = values.shape
    boxes = on_page(words, height, width)
    sides = word_sides(text_height(words, height, width))
    cells = [values[box.y : min(box.y2, height), box.x : min(box.x2, width)] for box in boxes]
    return {scale: tuple(block_peak(cell, side) for cell in cells) for scale, side in sides.items()}
