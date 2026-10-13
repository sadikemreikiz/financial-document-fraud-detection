"""Measuring the maps of pixel-forensics systems (ADR-0015).

Every system that produces a map is measured by the same rules. Its highest-scoring region is
derived from the map by one rule without parameters (:func:`top_region`); the region is compared
with the annotated rectangles and with a random box of the same size on the receipt's text, the
localisation guard of ADR-0014 (:func:`baseline_hit_rate`, :func:`localisation_guard`); its
pixels are scored per receipt (:func:`pixel_average_precision`, :func:`pixel_f1_iou`); and its
document scores are checked for growing with receipt size (:func:`size_dependence`).

Nothing here views an image: the functions receive maps, boxes and sizes.
"""

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from typing import cast

import numpy as np
import numpy.typing as npt
from scipy import ndimage, stats

from docfraud.data.models import BBox, Document, RegionRole
from docfraud.evaluation.bootstrap import DEFAULT_RESAMPLES, Estimate, ratio, stratified
from docfraud.evaluation.detection import average_precision

REGION_LEVEL = 0.5
"""Share of the way from the lowest to the highest map value at which the region is cut."""

BASELINE_DRAWS = 1_000
"""Random boxes per receipt in the baseline of the localisation guard (ADR-0014)."""

BASELINE_SEED = 0
"""Seed of every receipt's random generators, together with a hash of its identifier."""

LARGE_RECTANGLE_SHARE = 0.05
"""Share of the image above which a modified rectangle puts its receipt out of the
sensitivity check of ADR-0015, decision 4."""

THRESHOLD_SAMPLE = 100_000
"""Pixels sampled from each validation receipt to form the candidate pixel thresholds."""

_EIGHT_NEIGHBOURS = np.ones((3, 3), dtype=bool)
_MAP_DIMENSIONS = 2
_MIN_CORRELATED = 3  # receipts needed for a rank correlation
_BASELINE, _SAMPLE = 0, 1  # streams of a receipt's generator

type Map = npt.NDArray[np.floating] | npt.NDArray[np.integer]


def _checked(score_map: Map) -> npt.NDArray[np.float64]:
    values = np.asarray(score_map, dtype=np.float64)
    if values.ndim != _MAP_DIMENSIONS or values.size == 0:
        msg = "a map is a non-empty two-dimensional array"
        raise ValueError(msg)
    if not np.isfinite(values).all():
        msg = "a map holds finite values only"
        raise ValueError(msg)
    return values


def top_region(score_map: Map) -> BBox:
    """The highest-scoring region of a map (ADR-0015, decision 2).

    The bounding box of the pixels that are eight-connected to the first maximum in row-major
    order and score at least halfway between the lowest and the highest value. A map with one
    value everywhere yields the whole page.

    Raises:
        ValueError: If the map is empty, not two-dimensional or holds a value that is not finite.
    """
    values = _checked(score_map)
    height, width = values.shape
    low, high = float(values.min()), float(values.max())
    if high == low:
        return BBox(x=0, y=0, width=width, height=height)
    peak = np.unravel_index(int(np.argmax(values)), values.shape)
    labels, _ = ndimage.label(values >= low + REGION_LEVEL * (high - low), _EIGHT_NEIGHBOURS)
    component = np.asarray(labels) == labels[peak]
    rows = np.flatnonzero(component.any(axis=1))
    columns = np.flatnonzero(component.any(axis=0))
    return BBox(
        x=int(columns[0]),
        y=int(rows[0]),
        width=int(columns[-1] - columns[0] + 1),
        height=int(rows[-1] - rows[0] + 1),
    )


def annotated_rectangles(document: Document) -> tuple[list[BBox], list[BBox]]:
    """The modified rectangles and the original areas (copy sources) of a receipt."""
    modified = [r.box for r in document.regions if r.role is RegionRole.MODIFIED]
    sources = [r.box for r in document.regions if r.role is RegionRole.SOURCE]
    return modified, sources


def hits(region: BBox, rectangles: Iterable[BBox]) -> bool:
    """Whether a region shares at least one pixel with any of the rectangles."""
    return any(region.overlaps(rectangle) for rectangle in rectangles)


def receipt_rng(doc_id: str, stream: int = _BASELINE) -> np.random.Generator:
    """A receipt's own random generator, seeded with the baseline seed and its identifier.

    Its draws do not depend on which other receipts are measured or in which order.
    """
    key = int.from_bytes(hashlib.sha256(doc_id.encode()).digest()[:8], "big")
    return np.random.default_rng([BASELINE_SEED, key, stream])


def baseline_hit_rate(
    region: BBox,
    page: BBox,
    word_boxes: Iterable[BBox],
    rectangles: Sequence[BBox],
    rng: np.random.Generator,
) -> float:
    """The share of random boxes of the region's size that hit a rectangle (ADR-0014).

    Each box is centred on a pixel drawn uniformly from the union of the OCR word boxes and
    shifted into the page where it would cross an edge.

    Raises:
        ValueError: If the region does not fit on the page or there is no word box.
    """
    if not region.fits_within(page.width, page.height):
        msg = "the region does not fit on the page"
        raise ValueError(msg)
    text = np.zeros((page.height, page.width), dtype=bool)
    for box in word_boxes:
        text[box.y : box.y2, box.x : box.x2] = True
    pixels = np.flatnonzero(text)
    if pixels.size == 0:
        msg = "a receipt without OCR word boxes has no random baseline"
        raise ValueError(msg)
    centres = pixels[rng.integers(0, pixels.size, size=BASELINE_DRAWS)]
    rows, columns = np.divmod(centres, page.width)
    x = np.clip(columns - region.width // 2, 0, page.width - region.width)
    y = np.clip(rows - region.height // 2, 0, page.height - region.height)
    hit = np.zeros(BASELINE_DRAWS, dtype=bool)
    for r in rectangles:
        hit |= (x < r.x2) & (r.x < x + region.width) & (y < r.y2) & (r.y < y + region.height)
    return float(hit.mean())


@dataclass(frozen=True, slots=True)
class Guard:
    """The localisation guard of a system: its hit rate against the random baseline."""

    hit_rate: Estimate
    baseline: float
    receipts: int

    @property
    def detects(self) -> bool:
        """Whether the lower bound of the hit rate exceeds the baseline (ADR-0014)."""
        return self.hit_rate.low > self.baseline


def localisation_guard(
    region_hits: Sequence[bool],
    baselines: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Guard:
    """The hit rate with its bootstrap interval over receipts, and the mean baseline.

    Raises:
        ValueError: If the inputs differ in length or are empty.
    """
    if len(region_hits) != len(baselines) or not region_hits:
        msg = "one hit and one baseline per receipt, for at least one receipt"
        raise ValueError(msg)
    count = len(region_hits)
    estimate = ratio([float(h) for h in region_hits], [1.0] * count, resamples=resamples, seed=seed)
    return Guard(estimate, float(np.mean(baselines)), count)


def has_large_rectangle(
    document: Document, page: BBox, share: float = LARGE_RECTANGLE_SHARE
) -> bool:
    """Whether a modified rectangle covers more than ``share`` of the image (ADR-0015, 4)."""
    return any(r.box.area > share * page.area for r in document.modified_regions)


def _truth(
    shape: tuple[int, ...], modified: Iterable[BBox], sources: Iterable[BBox]
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.bool_]]:
    """The modified pixels, and the pixels that count (all but source-only pixels)."""
    positive = np.zeros(shape, dtype=bool)
    for r in modified:
        positive[r.y : r.y2, r.x : r.x2] = True
    if not positive.any():
        msg = "a receipt without modified pixels has no pixel metric"
        raise ValueError(msg)
    ignored = np.zeros(shape, dtype=bool)
    for r in sources:
        ignored[r.y : r.y2, r.x : r.x2] = True
    return positive, ~(ignored & ~positive)


def pixel_average_precision(
    score_map: Map, modified: Iterable[BBox], sources: Iterable[BBox]
) -> float:
    """Average precision of one receipt's pixels; source-only pixels are ignored (ADR-0015).

    Raises:
        ValueError: If the map is invalid or the receipt has no modified pixel.
    """
    values = _checked(score_map)
    positive, counted = _truth(values.shape, modified, sources)
    return average_precision(positive[counted], values[counted])


def f1_at_thresholds(
    score_map: Map,
    thresholds: npt.ArrayLike,
    modified: Iterable[BBox],
    sources: Iterable[BBox],
) -> npt.NDArray[np.float64]:
    """Pixel F1 of one receipt at each threshold, flagging pixels that score at least it."""
    values = _checked(score_map)
    positive, counted = _truth(values.shape, modified, sources)
    tp, fp, fn = _counts(values, positive, counted, np.asarray(thresholds, dtype=np.float64))
    denominator = 2 * tp + fp + fn
    return np.divide(2 * tp, denominator, out=np.zeros(tp.shape), where=denominator > 0)


def pixel_f1_iou(
    score_map: Map, threshold: float, modified: Iterable[BBox], sources: Iterable[BBox]
) -> tuple[float, float]:
    """Pixel F1 and IoU of one receipt at a threshold; source-only pixels are ignored."""
    values = _checked(score_map)
    positive, counted = _truth(values.shape, modified, sources)
    tp, fp, fn = (float(c[0]) for c in _counts(values, positive, counted, np.array([threshold])))
    return 2 * tp / (2 * tp + fp + fn), tp / (tp + fp + fn)


def _counts(
    values: npt.NDArray[np.float64],
    positive: npt.NDArray[np.bool_],
    counted: npt.NDArray[np.bool_],
    thresholds: npt.NDArray[np.float64],
) -> tuple[npt.NDArray[np.float64], ...]:
    """True positives, false positives and false negatives at each threshold."""
    hits_sorted = np.sort(values[positive])
    misses_sorted = np.sort(values[counted & ~positive])
    tp = hits_sorted.size - np.searchsorted(hits_sorted, thresholds, side="left")
    fp = misses_sorted.size - np.searchsorted(misses_sorted, thresholds, side="left")
    return tp.astype(float), fp.astype(float), (hits_sorted.size - tp).astype(float)


def threshold_sample(score_map: Map, doc_id: str) -> npt.NDArray[np.float64]:
    """Map values sampled from one receipt to form the candidate thresholds (ADR-0015, 5)."""
    values = _checked(score_map).ravel()
    if values.size <= THRESHOLD_SAMPLE:
        return values
    chosen = receipt_rng(doc_id, _SAMPLE).choice(values.size, THRESHOLD_SAMPLE, replace=False)
    return values[chosen]


def threshold_candidates(samples: Sequence[npt.NDArray[np.float64]]) -> npt.NDArray[np.float64]:
    """The 1st to 99th percentiles of the pooled samples, without repeats, in increasing order."""
    return np.unique(np.percentile(np.concatenate(samples), np.arange(1, 100)))


def choose_pixel_threshold(
    candidates: npt.ArrayLike, f1_per_receipt: Sequence[npt.ArrayLike]
) -> float:
    """The candidate with the highest mean F1 over receipts; ties go to the higher threshold."""
    values = np.asarray(candidates, dtype=np.float64)
    mean_f1 = np.mean(np.asarray(f1_per_receipt, dtype=np.float64), axis=0)
    best = np.flatnonzero(mean_f1 == mean_f1.max())
    return float(values[best].max())


def spearman(first: npt.ArrayLike, second: npt.ArrayLike) -> float:
    """Spearman rank correlation, ties ranked by their mean; not a number if undefined."""
    a, b = np.asarray(first, dtype=np.float64), np.asarray(second, dtype=np.float64)
    if np.ptp(a) == 0 or np.ptp(b) == 0:
        return float("nan")
    return cast("float", stats.spearmanr(a, b).statistic)


def size_dependence(
    scores: Sequence[float],
    pixel_counts: Sequence[int],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> Estimate:
    """Rank correlation of document score and receipt size, with an interval (ADR-0015, 6).

    Measured on authentic receipts; an interval that excludes zero means that the document
    score grows (or shrinks) with the size of the receipt.

    Raises:
        ValueError: If the inputs differ in length or hold fewer than three receipts.
    """
    if len(scores) != len(pixel_counts) or len(scores) < _MIN_CORRELATED:
        msg = "one score and one size per receipt, for at least three receipts"
        raise ValueError(msg)
    s, n = np.asarray(scores, dtype=np.float64), np.asarray(pixel_counts, dtype=np.float64)
    return stratified(
        lambda draw: spearman(s[draw], n[draw]),
        np.zeros(len(s), dtype=int),
        resamples=resamples,
        seed=seed,
    )
