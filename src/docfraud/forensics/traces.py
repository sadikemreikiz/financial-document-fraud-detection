"""Traces of editing that code can measure and a person cannot see (step 4.3, decision 2).

Two traces, as ``docs/invisible-traces-study.md`` fixes them: the noise level of a box, measured
on its paper pixels after a median high-pass filter, and the 8 x 8 block grid that JPEG
compression left in the receipts before they were stored as PNG. A region of a receipt is
compared with control boxes of the same size on the receipt's unedited text.

Every function receives pixels as arrays and returns numbers; nothing is shown to a person.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from docfraud.data.models import BBox
from docfraud.evaluation.localisation import (
    BASELINE_SEED,
    overlapping,
    random_boxes,
    receipt_key,
)

PAPER_WINDOW = 5
"""Side of the neighbourhood that must hold no ink for a pixel to count as paper."""

PAPER_MADS = 3.0
"""Paper level: the receipt's median gray value minus this many median absolute deviations."""

MIN_PAPER_PIXELS = 50
"""Paper pixels a box needs for its noise level to be measured."""

MAD_TO_STD = 1.4826
"""Factor that turns a median absolute deviation into a standard deviation for normal noise."""

GRID_PERIOD = 8
"""Side of a JPEG block."""

NULL_PERIODS = (7, 9)
"""Periods that JPEG does not use; their contrast shows how large a peak arises by chance."""

GRID_RATIO = 2.0
"""How many times the period-8 contrast must exceed the null contrast for a detectable grid."""

CONTROLS = 200
"""Control boxes per region."""

MAX_CONTROL_DRAWS = 20_000
"""Random boxes drawn at most per region while collecting its controls."""

OUTSIDE = (2.5, 97.5)
"""Percentiles of the controls outside which a region stands out."""

_CONTROL_STREAM = 2
_BATCH = 1_000


def residual(page: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """The page minus its 3 x 3 median filter: fine noise and edges, without the content."""
    smooth = np.asarray(ndimage.median_filter(page, size=3), dtype=np.float32)
    return np.asarray(page, dtype=np.float32) - smooth


def paper_mask(page: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
    """Pixels whose neighbourhood holds no value below the receipt's paper level."""
    median = float(np.median(page))
    mad = float(np.median(np.abs(page.astype(np.float32) - median)))
    level = median - PAPER_MADS * mad
    darkest = ndimage.minimum_filter(page, size=PAPER_WINDOW)
    return np.asarray(darkest >= level)


def noise_level(
    noise: npt.NDArray[np.float32], paper: npt.NDArray[np.bool_], box: BBox
) -> float | None:
    """Robust standard deviation of the residual over a box's paper pixels.

    Returns:
        The noise level, or ``None`` if the box holds fewer than ``MIN_PAPER_PIXELS`` paper
        pixels.
    """
    rows, columns = slice(box.y, box.y2), slice(box.x, box.x2)
    values = noise[rows, columns][paper[rows, columns]]
    if values.size < MIN_PAPER_PIXELS:
        return None
    return MAD_TO_STD * float(np.median(np.abs(values - np.median(values))))


def excess_jumps(page: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
    """The jump across each column boundary minus the mean jump across its two neighbours.

    Column ``k`` of the result belongs to the boundary between columns ``k + 1`` and ``k + 2``
    of the page. Pass the transposed page for row boundaries.
    """
    jumps = np.abs(np.diff(page.astype(np.float32), axis=1))
    return np.asarray(jumps[:, 1:-1] - (jumps[:, :-2] + jumps[:, 2:]) / 2, dtype=np.float32)


def _profile(
    sums: npt.NDArray[np.float64],
    counts: npt.NDArray[np.float64],
    boundaries: npt.NDArray[np.intp],
    period: int,
) -> npt.NDArray[np.float64]:
    """Mean excess jump at each position modulo ``period``, over the given boundaries."""
    profile = np.full(period, np.nan)
    phases = boundaries % period
    for phase in range(period):
        chosen = phases == phase
        total = counts[chosen].sum()
        if total > 0:
            profile[phase] = sums[chosen].sum() / total
    return profile


def _contrast(profile: npt.NDArray[np.float64]) -> float:
    return float(np.nanmax(profile) - np.nanmedian(profile))


@dataclass(frozen=True, slots=True)
class Direction:
    """The grid of a receipt across columns or across rows."""

    ratio: float  # period-8 contrast over the larger null contrast
    phase: int  # boundary position modulo 8 with the largest excess jump


@dataclass(frozen=True, slots=True)
class Grid:
    """The JPEG grid of a receipt, measured outside its annotated rectangles."""

    columns: Direction
    rows: Direction

    @property
    def detectable(self) -> bool:
        """Whether the period-8 peak clearly exceeds chance in both directions."""
        return self.columns.ratio >= GRID_RATIO and self.rows.ratio >= GRID_RATIO


def _direction(excess: npt.NDArray[np.float32], kept: npt.NDArray[np.bool_]) -> Direction:
    sums = np.where(kept, excess, 0).sum(axis=0, dtype=np.float64)
    counts = kept.sum(axis=0).astype(np.float64)
    boundaries = np.arange(excess.shape[1]) + 1
    contrasts = {
        period: _contrast(_profile(sums, counts, boundaries, period))
        for period in (GRID_PERIOD, *NULL_PERIODS)
    }
    null = max(contrasts[p] for p in NULL_PERIODS)
    peak = contrasts[GRID_PERIOD]
    ratio = peak / null if null > 0 else (float("inf") if peak > 0 else 0.0)
    phase = int(np.nanargmax(_profile(sums, counts, boundaries, GRID_PERIOD)))
    return Direction(ratio=ratio, phase=phase)


@dataclass(frozen=True, slots=True)
class Jumps:
    """Excess jumps of a receipt across column and row boundaries."""

    columns: npt.NDArray[np.float32]  # (height, width - 3)
    rows: npt.NDArray[np.float32]  # (width, height - 3), from the transposed page

    @classmethod
    def of(cls, page: npt.NDArray[np.uint8]) -> Jumps:
        """Compute both directions from a page."""
        return cls(columns=excess_jumps(page), rows=excess_jumps(page.T))


def receipt_grid(jumps: Jumps, paper: npt.NDArray[np.bool_], rectangles: Iterable[BBox]) -> Grid:
    """The grid of a receipt, measured on paper outside the annotated rectangles.

    A boundary counts only where the four pixels its excess jump uses are paper pixels outside
    every annotated rectangle: the edges of print can repeat at a regular spacing (a monospaced
    font, a fixed line pitch) and would otherwise create or hide a peak.
    """
    usable = paper.copy()
    for r in rectangles:
        usable[r.y : r.y2, r.x : r.x2] = False
    kept_columns = usable[:, :-3] & usable[:, 1:-2] & usable[:, 2:-1] & usable[:, 3:]
    kept_rows = (usable[:-3, :] & usable[1:-2, :] & usable[2:-1, :] & usable[3:, :]).T
    return Grid(
        columns=_direction(jumps.columns, kept_columns),
        rows=_direction(jumps.rows, kept_rows),
    )


def _box_agreement(excess: npt.NDArray[np.float32], box: BBox, phase: int) -> float:
    """Profile value at the receipt's phase minus the profile median, inside one box."""
    first = max(box.x - 1, 0)  # result column k is boundary k + 1
    last = min(box.x2 - 3, excess.shape[1] - 1)  # boundaries between columns of the box
    if last < first:
        return float("nan")
    block = excess[box.y : box.y2, first : last + 1]
    sums = block.sum(axis=0, dtype=np.float64)
    counts = np.full(sums.shape, float(block.shape[0]))
    profile = _profile(sums, counts, np.arange(first, last + 1) + 1, GRID_PERIOD)
    return float(profile[phase] - np.nanmedian(profile))


def grid_agreement(jumps: Jumps, grid: Grid, box: BBox) -> float:
    """How strongly a box shows the receipt's grid, as the mean over both directions."""
    across_columns = _box_agreement(jumps.columns, box, grid.columns.phase)
    transposed = BBox(x=box.y, y=box.x, width=box.height, height=box.width)
    across_rows = _box_agreement(jumps.rows, transposed, grid.rows.phase)
    return float(np.mean([across_columns, across_rows]))


def region_rng(doc_id: str, index: int) -> np.random.Generator:
    """The random generator of one region's controls."""
    return np.random.default_rng([BASELINE_SEED, receipt_key(doc_id), _CONTROL_STREAM, index])


def control_boxes(
    region: BBox,
    page: BBox,
    word_boxes: Sequence[BBox],
    rectangles: Sequence[BBox],
    rng: np.random.Generator,
) -> list[BBox]:
    """Up to ``CONTROLS`` boxes of the region's size on unedited text of the receipt.

    Boxes are drawn as for the random baseline and kept when they overlap no annotated
    rectangle, until enough are kept or ``MAX_CONTROL_DRAWS`` have been drawn.
    """
    kept: list[BBox] = []
    drawn = 0
    while len(kept) < CONTROLS and drawn < MAX_CONTROL_DRAWS:
        x, y = random_boxes(region, page, word_boxes, rng, _BATCH)
        drawn += _BATCH
        free = ~overlapping(x, y, region, rectangles)
        kept += [
            BBox(x=int(a), y=int(b), width=region.width, height=region.height)
            for a, b in zip(x[free], y[free], strict=True)
        ]
    return kept[:CONTROLS]


@dataclass(frozen=True, slots=True)
class Comparison:
    """A region's value against the values of its controls."""

    value: float
    percentile: float  # share of controls below the value, ties counting half, in percent
    stands_out: str | None  # "below", "above" or None


def compare(value: float, controls: Sequence[float]) -> Comparison:
    """Where a value lies among its controls; outside the 2.5th to 97.5th percentile stands out.

    Raises:
        ValueError: If a value is not finite or there are no controls.
    """
    values = np.asarray(controls, dtype=np.float64)
    if values.size == 0 or not np.isfinite(values).all() or not np.isfinite(value):
        msg = "a comparison needs a finite value and finite controls"
        raise ValueError(msg)
    below = float((values < value).sum() + 0.5 * (values == value).sum())
    low, high = np.percentile(values, OUTSIDE)
    side = "below" if value < low else "above" if value > high else None
    return Comparison(value=value, percentile=100 * below / values.size, stands_out=side)
