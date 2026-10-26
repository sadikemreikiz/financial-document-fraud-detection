"""Copy-move inside a receipt: two regions that are exact duplicates (ADR-0029, ADR-0030).

A receipt is a lossless PNG file, so characters copied and pasted inside it keep the grey levels
of their source, scan noise included, while a character printed twice differs by that noise.
The detector therefore looks for duplicates that are too exact to have been printed: it hashes
every square block of the page, at every position, and finds equal hashes at positions at least
one block apart (the exact-match search of Fridrich, Soukal and Lukáš, 2003). Every match is then
compared pixel by pixel, so a collision of the hash cannot make one.

- **Block side.** A quarter of the receipt's text height (the median height of its OCR words),
  so that a pasted character holds many blocks.
- **Informative blocks.** A block counts only if at least a quarter of its pixels are
  unsaturated (neither 0 nor 255) and it spans ink: its darkest and lightest pixels differ by at
  least :data:`INK_CONTRAST` grey levels. Blocks of blank paper clipped to white, or holding a
  single speck of dust on it, are identical everywhere and would match by chance; so do blocks
  of the repeating textures of compressed paper.
- **Pairs.** Matching blocks are grouped by their offset, the shift from one copy to the other,
  and each offset's matches are split into connected groups; the largest group of any offset is
  the receipt's pair. A hash shared by more than :data:`MAX_GROUP` positions is a repeated
  structure, not a copy, and is skipped.
- **Document score.** The area of one region of the pair covered by duplicated blocks, in
  units of the squared text height, roughly the number of characters copied; 0 without a pair.

The detector reports the pair without naming either region as the copy (ADR-0030). Copies
pasted as their ink alone, slightly scaled or rotated, or made of blank paper are not exact
blocks and are not found.
"""

from collections import defaultdict
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from docfraud.data.models import BBox
from docfraud.forensics.learned.text_scale import text_height

BLOCK_DIVISOR = 4
"""The block side is the text height divided by this."""

MIN_BLOCK = 4
"""The smallest block side in pixels."""

INK_CONTRAST = 60
"""Grey levels between the darkest and lightest pixel of a block that spans ink."""

UNSATURATED_SHARE = 0.25
"""Share of a block's pixels that must lie strictly between 0 and 255."""

MAX_GROUP = 16
"""Positions sharing one hash above which the hash is a repeated structure."""

WHITE = 255
"""The grey level of paper clipped to white; 0 is ink clipped to black."""

_ROW_BASE = np.uint64(0x9E3779B97F4A7C15)
_COLUMN_BASE = np.uint64(0xC2B2AE3D27D4EB4F)

type Page = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class CopyMovePair:
    """Two regions of a receipt that are exact duplicates of each other.

    ``first`` is the upper region, or the left one when both start on the same row; neither is
    named as the copy (ADR-0030).
    """

    first: BBox
    second: BBox
    blocks: int  # duplicated block positions in the group
    area: int  # pixels of each region covered by those blocks


@dataclass(frozen=True, slots=True)
class CopyMoveEvidence:
    """What copy-move found on one receipt."""

    text_height: float
    block: int  # the block side in pixels
    pair: CopyMovePair | None  # the largest duplicated group, if any

    @property
    def document_score(self) -> float:
        """Duplicated area in units of the squared text height; 0 without a pair."""
        return self.pair.area / self.text_height**2 if self.pair else 0.0


def block_side(height: float) -> int:
    """The block side for a receipt whose text is ``height`` pixels high."""
    return max(MIN_BLOCK, round(height / BLOCK_DIVISOR))


def block_sums(values: npt.NDArray[np.int64], side: int) -> npt.NDArray[np.int64]:
    """Sum of ``values`` over the ``side`` x ``side`` block starting at every position."""
    total = np.zeros((values.shape[0] + 1, values.shape[1] + 1), dtype=np.int64)
    total[1:, 1:] = values.cumsum(axis=0).cumsum(axis=1)
    return total[side:, side:] - total[:-side, side:] - total[side:, :-side] + total[:-side, :-side]


def block_hashes(page: Page, side: int) -> npt.NDArray[np.uint64]:
    """A 64-bit polynomial hash of the block starting at every position.

    Equal blocks have equal hashes. Unequal blocks rarely share one, but a polynomial hash with
    fixed bases is not random, so :func:`duplicate_pair` confirms every match on the pixels.
    """
    pixels = page.astype(np.uint64)
    height, width = pixels.shape
    rows = np.zeros((height, width - side + 1), dtype=np.uint64)
    for j in range(side):
        rows = rows * _ROW_BASE + pixels[:, j : width - side + 1 + j]
    blocks = np.zeros((height - side + 1, width - side + 1), dtype=np.uint64)
    for i in range(side):
        blocks = blocks * _COLUMN_BASE + rows[i : height - side + 1 + i]
    return blocks


def informative_blocks(
    page: Page,
    side: int,
    *,
    contrast: int = INK_CONTRAST,
    unsaturated_share: float = UNSATURATED_SHARE,
) -> npt.NDArray[np.bool_]:
    """Whether the block starting at every position may count as a duplicate."""
    origin = -(side // 2)  # the filter window then starts at its output position
    highest = ndimage.maximum_filter(page, size=side, origin=origin, mode="nearest")
    lowest = ndimage.minimum_filter(page, size=side, origin=origin, mode="nearest")
    height, width = page.shape
    spread = (highest.astype(np.int16) - lowest.astype(np.int16))[
        : height - side + 1, : width - side + 1
    ]
    unsaturated = block_sums(((page > 0) & (page < WHITE)).astype(np.int64), side)
    keep = (unsaturated >= unsaturated_share * side * side) & (spread > 0)
    return np.asarray(keep & (spread >= contrast), dtype=np.bool_)


def _offset_matches(
    page: Page, hashes: npt.NDArray[np.uint64], keep: npt.NDArray[np.bool_], side: int
) -> dict[tuple[int, int], list[tuple[int, int]]]:
    """Positions of the upper block of each duplicated pair, by offset to the other block.

    Positions come in row-major order and the sort is stable, so within a group of equal hashes
    a later position lies below an earlier one, or on the same row to its right.
    """
    rows, columns = np.nonzero(keep)
    keys = hashes[rows, columns]
    order = np.argsort(keys, kind="stable")
    keys, rows, columns = keys[order], rows[order], columns[order]
    starts = np.r_[0, np.flatnonzero(np.diff(keys)) + 1]
    ends = np.r_[starts[1:], keys.size]
    shared = (ends - starts > 1) & (ends - starts <= MAX_GROUP)
    matches: dict[tuple[int, int], list[tuple[int, int]]] = defaultdict(list)
    for start, end in zip(starts[shared], ends[shared], strict=True):
        for i in range(start, end):
            for j in range(i + 1, end):
                dy, dx = int(rows[j] - rows[i]), int(columns[j] - columns[i])
                if max(abs(dy), abs(dx)) < side:  # overlapping blocks of one region
                    continue
                first = page[rows[i] : rows[i] + side, columns[i] : columns[i] + side]
                second = page[rows[j] : rows[j] + side, columns[j] : columns[j] + side]
                if not np.array_equal(first, second):  # the hash collided
                    continue
                matches[(dy, dx)].append((int(rows[i]), int(columns[i])))
    return matches


def _largest_group(
    positions: Sequence[tuple[int, int]], offset: tuple[int, int], side: int
) -> CopyMovePair:
    """The largest connected group of one offset's matches, as a pair of regions."""
    points = np.array(positions)
    top, left = points.min(axis=0)
    grid = np.zeros(tuple(points.max(axis=0) - (top, left) + 1), dtype=bool)
    grid[points[:, 0] - top, points[:, 1] - left] = True
    labels, _ = ndimage.label(grid, structure=np.ones((3, 3), dtype=bool))
    sizes = np.bincount(labels.ravel())
    sizes[0] = 0
    group_rows, group_columns = np.nonzero(labels == int(np.argmax(sizes)))
    group = np.zeros(
        (group_rows.max() - group_rows.min() + 1, group_columns.max() - group_columns.min() + 1),
        dtype=np.int64,
    )
    group[group_rows - group_rows.min(), group_columns - group_columns.min()] = 1
    # a pixel is covered by any block that starts up to ``side - 1`` before it
    area = int((block_sums(np.pad(group, side - 1), side) > 0).sum())
    first = BBox(
        x=int(left + group_columns.min()),
        y=int(top + group_rows.min()),
        width=int(group.shape[1] + side - 1),
        height=int(group.shape[0] + side - 1),
    )
    dy, dx = offset
    second = BBox(x=first.x + dx, y=first.y + dy, width=first.width, height=first.height)
    return CopyMovePair(first, second, blocks=int(group_rows.size), area=area)


def duplicate_pair(
    page: Page,
    side: int,
    *,
    contrast: int = INK_CONTRAST,
    unsaturated_share: float = UNSATURATED_SHARE,
) -> CopyMovePair | None:
    """The largest pair of exactly duplicated regions on a page, or ``None``.

    Raises:
        ValueError: If the page is smaller than one block.
    """
    if min(page.shape) < side:
        msg = f"a page of {page.shape} holds no block of side {side}"
        raise ValueError(msg)
    keep = informative_blocks(page, side, contrast=contrast, unsaturated_share=unsaturated_share)
    matches = _offset_matches(page, block_hashes(page, side), keep, side)
    pairs = [_largest_group(positions, offset, side) for offset, positions in matches.items()]
    return max(pairs, key=lambda p: (p.area, p.blocks), default=None)


def copy_move_evidence(page: Page, words: Sequence[BBox]) -> CopyMoveEvidence:
    """Copy-move on one normalised page, given its OCR word boxes for the text height."""
    height = text_height(words, *page.shape)
    side = block_side(height)
    return CopyMoveEvidence(height, side, duplicate_pair(page, side))
