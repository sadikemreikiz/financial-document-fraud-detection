"""Bootstrap confidence intervals that resample documents, not individual items.

Items such as amounts on a receipt are not independent: an engine that fails on one blurry
receipt fails on many of its amounts at once. Resampling whole documents keeps that correlation
and gives honest intervals.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

DEFAULT_RESAMPLES = 10_000


@dataclass(frozen=True, slots=True)
class Estimate:
    """A point estimate with a percentile confidence interval."""

    value: float
    low: float
    high: float
    level: float = 0.95

    @property
    def excludes_zero(self) -> bool:
        """Whether the interval lies entirely above or below zero."""
        return self.low > 0 or self.high < 0


def _check(numerators: Sequence[Sequence[float]], denominators: Sequence[float]) -> None:
    sizes = {len(denominators), *(len(n) for n in numerators)}
    if len(sizes) != 1 or not denominators:
        msg = "all inputs need one entry per document and at least one document"
        raise ValueError(msg)
    if sum(denominators) <= 0:
        msg = "the denominators must sum to a positive number"
        raise ValueError(msg)


def ratio(
    numerator: Sequence[float],
    denominator: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Ratio of sums (e.g. amounts found / amounts present) with a document-level interval."""
    _check([numerator], denominator)
    num, den = np.asarray(numerator, float), np.asarray(denominator, float)
    samples = _resample([num], den, resamples, seed)[0]
    return _estimate(num.sum() / den.sum(), samples, level)


def paired_ratio_difference(  # noqa: PLR0913 - mirrors ratio() for two systems
    numerator_a: Sequence[float],
    numerator_b: Sequence[float],
    denominator: Sequence[float],
    *,
    resamples: int = DEFAULT_RESAMPLES,
    level: float = 0.95,
    seed: int = 0,
) -> Estimate:
    """Difference ``b - a`` of two ratios measured on the same documents.

    Both systems are resampled with the same document draws, which is what makes the
    comparison paired.
    """
    _check([numerator_a, numerator_b], denominator)
    a, b = np.asarray(numerator_a, float), np.asarray(numerator_b, float)
    den = np.asarray(denominator, float)
    ratio_a, ratio_b = _resample([a, b], den, resamples, seed)
    return _estimate((b.sum() - a.sum()) / den.sum(), ratio_b - ratio_a, level)


def _resample(
    numerators: list[np.ndarray], denominator: np.ndarray, resamples: int, seed: int
) -> list[np.ndarray]:
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, len(denominator), size=(resamples, len(denominator)))
    totals = denominator[draws].sum(axis=1)
    totals[totals == 0] = np.nan  # a draw without any items carries no information
    return [numerator[draws].sum(axis=1) / totals for numerator in numerators]


def _estimate(value: float, samples: np.ndarray, level: float) -> Estimate:
    tail = (1 - level) / 2 * 100
    low, high = np.nanpercentile(samples, [tail, 100 - tail])
    return Estimate(value=float(value), low=float(low), high=float(high), level=level)
