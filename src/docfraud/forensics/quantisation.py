"""How well each 8 x 8 block of a receipt still fits its earlier JPEG compression.

JPEG stores each 8 x 8 block as discrete cosine transform (DCT) coefficients rounded to
multiples of a step per coefficient. A receipt that was decoded from JPEG and stored losslessly
keeps that lattice: recomputing the DCT on the same block grid gives values close to multiples
of the steps. Content pasted off the grid, drawn anew or painted over no longer fits. The
comparison study is fixed in ``docs/jpeg-trace-comparison.md``.

Every function receives pixels as arrays and returns numbers; nothing is shown to a person.
"""

from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from scipy.fft import dctn

from docfraud.data.models import BBox

BLOCK = 8
"""Side of a JPEG block."""

WHITE = 255
"""Largest gray value; a pixel at 0 or here may have been clipped."""

STEPS = np.arange(2, 65)
"""Candidate quantisation steps of a coefficient."""

MIN_STEP = 3
"""Smallest step a coefficient needs to be used; rounding to whole gray values hides less."""

MIN_VALUES = 200
"""Values beyond half a step a coefficient needs for its fit at that step to be defined."""

COEFFICIENT_FIT = 0.5
"""Fit a coefficient needs at its step to be used for the blocks of the receipt."""

ALIGNMENT_COEFFICIENTS = ((0, 1), (1, 0), (1, 1), (0, 2), (2, 0), (1, 2), (2, 1))
"""Low-frequency coefficients that locate the block grid; text gives them large values."""

ALIGNMENT_SAMPLE = 4_000
"""Block positions sampled to score each of the 64 alignments when a generator is given."""

LATTICE_MARGIN = 0.25
"""How far the best alignment's score must exceed the next best for a detectable lattice."""

MIN_BLOCKS = 4
"""Measurable blocks a box needs for its fit to be defined."""


def coefficient_fit(values: npt.NDArray[np.float64], step: float) -> float:
    """Mean cosine of ``2 pi value / step`` over the values beyond half a step from zero.

    Values on the lattice give 1, values spread evenly over a step give about 0. Values within
    half a step of zero fit every large step equally and are left out.

    Returns:
        The fit, or NaN when fewer than ``MIN_VALUES`` values lie beyond half a step.
    """
    beyond = values[np.abs(values) >= step / 2]
    if beyond.size < MIN_VALUES:
        return float("nan")
    return float(np.mean(np.cos(2 * np.pi * beyond / step)))


def best_step(values: npt.NDArray[np.float64]) -> tuple[int, float]:
    """The candidate step with the highest fit, and that fit (NaN if no step is defined)."""
    fits = np.array([coefficient_fit(values, float(step)) for step in STEPS])
    if np.isnan(fits).all():
        return 0, float("nan")
    index = int(np.nanargmax(fits))
    return int(STEPS[index]), float(fits[index])


def _blocks(
    page: npt.NDArray[np.uint8], offset: tuple[int, int]
) -> tuple[npt.NDArray[np.uint8], int, int]:
    """Every whole block of the grid that starts at ``offset``, as (rows, columns, 8, 8)."""
    dy, dx = offset
    rows, columns = (page.shape[0] - dy) // BLOCK, (page.shape[1] - dx) // BLOCK
    view = page[dy : dy + rows * BLOCK, dx : dx + columns * BLOCK]
    return view.reshape(rows, BLOCK, columns, BLOCK).swapaxes(1, 2), rows, columns


def measurable(blocks: npt.NDArray[np.uint8]) -> npt.NDArray[np.bool_]:
    """Blocks without a pixel clipped to 0 or 255, where rounding alone moved the values."""
    return np.asarray((blocks.min(axis=(-2, -1)) > 0) & (blocks.max(axis=(-2, -1)) < WHITE))


def dct(blocks: npt.NDArray[np.uint8]) -> npt.NDArray[np.float64]:
    """The 8 x 8 DCT of each block as JPEG computes it (orthonormal, gray values minus 128)."""
    return np.asarray(dctn(blocks.astype(np.float64) - 128, axes=(-2, -1), norm="ortho"))


def _outside(
    shape: tuple[int, int], offset: tuple[int, int], rectangles: Iterable[BBox]
) -> npt.NDArray[np.bool_]:
    """Blocks of the grid at ``offset`` that share no pixel with any rectangle."""
    rows, columns = (shape[0] - offset[0]) // BLOCK, (shape[1] - offset[1]) // BLOCK
    free = np.ones((rows, columns), dtype=bool)
    for r in rectangles:
        top, left = (r.y - offset[0]) // BLOCK, (r.x - offset[1]) // BLOCK
        bottom = (r.y2 - 1 - offset[0]) // BLOCK + 1
        right = (r.x2 - 1 - offset[1]) // BLOCK + 1
        free[max(top, 0) : max(bottom, 0), max(left, 0) : max(right, 0)] = False
    return free


def alignment_score(
    page: npt.NDArray[np.uint8],
    offset: tuple[int, int],
    positions: npt.NDArray[np.intp],
    rectangles: Iterable[BBox],
) -> float:
    """Mean best fit of the alignment coefficients over sampled measurable blocks at an offset.

    Args:
        page: The normalised page.
        offset: Row and column where the block grid starts, each in 0..7.
        positions: Sampled block indices (row, column) on the grid at offset (0, 0), shape (n, 2).
        rectangles: Annotated rectangles; blocks sharing a pixel with one are left out.
    """
    blocks, rows, columns = _blocks(page, offset)
    chosen = positions[(positions[:, 0] < rows) & (positions[:, 1] < columns)]
    free = _outside(page.shape, offset, rectangles)[chosen[:, 0], chosen[:, 1]]
    sample = blocks[chosen[:, 0], chosen[:, 1]][free]
    sample = sample[measurable(sample)]
    if sample.shape[0] == 0:
        return float("nan")
    coefficients = dct(sample)
    fits = [best_step(coefficients[:, u, v])[1] for u, v in ALIGNMENT_COEFFICIENTS]
    return float(np.nanmean(fits)) if not np.isnan(fits).all() else float("nan")


@dataclass(frozen=True, slots=True)
class Lattice:
    """The JPEG lattice of a receipt, estimated outside its annotated rectangles."""

    offset: tuple[int, int]  # row and column where the best block grid starts
    score: float  # alignment score of the best offset (NaN if nothing was measurable)
    runner_up: float  # highest alignment score among the other 63 offsets
    steps: npt.NDArray[np.int_]  # (8, 8) step per coefficient; 0 where the coefficient is unused

    @property
    def detectable(self) -> bool:
        """Whether one alignment clearly fits better than every other and some steps are used."""
        if not np.isfinite(self.score) or bool((self.steps == 0).all()):
            return False
        return not np.isfinite(self.runner_up) or self.score - self.runner_up >= LATTICE_MARGIN


def receipt_lattice(
    page: npt.NDArray[np.uint8],
    rectangles: Iterable[BBox],
    rng: np.random.Generator | None = None,
) -> Lattice:
    """Locate the block grid among the 64 alignments, then estimate a step per coefficient.

    The alignment is scored on every block position, or on ``ALIGNMENT_SAMPLE`` positions drawn
    with ``rng`` when one is given (as the JPEG trace comparison did); the steps are estimated on
    every measurable block of the best alignment. Blocks sharing a pixel with an annotated
    rectangle are left out of both.
    """
    rectangles = list(rectangles)
    rows, columns = (page.shape[0] - BLOCK) // BLOCK, (page.shape[1] - BLOCK) // BLOCK
    unused = np.zeros((BLOCK, BLOCK), dtype=int)
    if rows <= 0 or columns <= 0:
        return Lattice((0, 0), float("nan"), float("nan"), unused)
    if rng is None:
        flat = np.arange(rows * columns)
    else:
        flat = rng.choice(rows * columns, size=min(ALIGNMENT_SAMPLE, rows * columns), replace=False)
    positions = np.stack([flat // columns, flat % columns], axis=1)
    scores = np.array(
        [
            [alignment_score(page, (dy, dx), positions, rectangles) for dx in range(BLOCK)]
            for dy in range(BLOCK)
        ]
    )
    if np.isnan(scores).all():
        return Lattice((0, 0), float("nan"), float("nan"), unused)
    best = np.unravel_index(int(np.nanargmax(scores)), scores.shape)
    offset = (int(best[0]), int(best[1]))
    others = np.delete(scores.ravel(), best[0] * BLOCK + best[1])
    runner_up = float(np.nanmax(others)) if not np.isnan(others).all() else float("nan")
    blocks, _, _ = _blocks(page, offset)
    keep = _outside(page.shape, offset, rectangles) & measurable(blocks)
    coefficients = dct(blocks[keep])
    steps = unused.copy()
    for u in range(BLOCK):
        for v in range(BLOCK):
            step, fit = best_step(coefficients[:, u, v])
            if np.isfinite(fit) and fit >= COEFFICIENT_FIT and step >= MIN_STEP:
                steps[u, v] = step
    return Lattice(offset, float(scores[best]), runner_up, steps)


def block_fits(page: npt.NDArray[np.uint8], lattice: Lattice) -> npt.NDArray[np.float64]:
    """The fit of every block of the receipt's grid: the mean cosine over its used coefficients.

    Returns:
        A (rows, columns) array; NaN for blocks that are not measurable.
    """
    blocks, rows, columns = _blocks(page, lattice.offset)
    fits = np.full((rows, columns), np.nan)
    used = lattice.steps > 0
    if not used.any():
        return fits
    keep = measurable(blocks)
    values = dct(blocks[keep])[:, used]
    fits[keep] = np.cos(2 * np.pi * values / lattice.steps[used]).mean(axis=1)
    return fits


def box_fit(fits: npt.NDArray[np.float64], lattice: Lattice, box: BBox) -> float | None:
    """Mean fit of the measurable blocks that share a pixel with a box.

    Returns:
        The fit, or ``None`` if fewer than ``MIN_BLOCKS`` such blocks are measurable.
    """
    dy, dx = lattice.offset
    top, left = max((box.y - dy) // BLOCK, 0), max((box.x - dx) // BLOCK, 0)
    bottom = max((box.y2 - 1 - dy) // BLOCK + 1, 0)
    right = max((box.x2 - 1 - dx) // BLOCK + 1, 0)
    chosen = fits[top:bottom, left:right]
    chosen = chosen[np.isfinite(chosen)]
    if chosen.size < MIN_BLOCKS:
        return None
    return float(chosen.mean())
