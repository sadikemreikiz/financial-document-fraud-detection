"""Document-level detection metrics, as pre-registered in the evaluation protocol (section 5).

A detector gives every document a score; higher means more likely forged, and the forged class
is positive. Threshold metrics compare ``score >= threshold``. The metrics are implemented in
numpy so that they can be bootstrapped 10,000 times quickly; the tests check them against
scikit-learn.

Ties matter here: accounting rules produce few distinct scores, so many documents share one.
Every metric treats tied documents alike, never by their order in the input.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from functools import partial
from typing import Any

import numpy as np
import numpy.typing as npt

from docfraud.evaluation.bootstrap import DEFAULT_RESAMPLES, Estimate, stratified

ECE_BINS = 10
REVIEW_BUDGETS = (0.10, 0.20)

type Labels = npt.NDArray[np.bool_]
type Scores = npt.NDArray[np.float64]


def as_arrays(labels: object, scores: object) -> tuple[Labels, Scores]:
    """Validate and convert labels (forged = true) and scores.

    Raises:
        ValueError: If the lengths differ, there are no documents, a score is not finite, or
            only one class is present.
    """
    y = np.asarray(labels, dtype=bool)
    s = np.asarray(scores, dtype=float)
    if y.ndim != 1 or y.shape != s.shape or y.size == 0:
        msg = "labels and scores need one entry per document and at least one document"
        raise ValueError(msg)
    if not np.all(np.isfinite(s)):
        msg = "scores must be finite"
        raise ValueError(msg)
    if y.all() or not y.any():
        msg = "both classes must be present"
        raise ValueError(msg)
    return y, s


@dataclass(frozen=True, slots=True)
class Counts:
    """Confusion counts of a thresholded detector."""

    tp: int
    fp: int
    fn: int
    tn: int

    @classmethod
    def of(cls, y: Labels, predicted: Labels) -> Counts:
        """Count from labels and binary predictions."""
        return cls(
            tp=int(np.sum(y & predicted)),
            fp=int(np.sum(~y & predicted)),
            fn=int(np.sum(y & ~predicted)),
            tn=int(np.sum(~y & ~predicted)),
        )

    @property
    def precision(self) -> float:
        """Share of flagged documents that are forged; 0 when nothing is flagged."""
        flagged = self.tp + self.fp
        return self.tp / flagged if flagged else 0.0

    @property
    def recall(self) -> float:
        """Share of forged documents that are flagged."""
        positives = self.tp + self.fn
        return self.tp / positives if positives else 0.0

    @property
    def f1(self) -> float:
        """Harmonic mean of precision and recall; 0 when there are no true positives."""
        denominator = 2 * self.tp + self.fp + self.fn
        return 2 * self.tp / denominator if denominator else 0.0

    @property
    def specificity(self) -> float:
        """Share of authentic documents that are not flagged."""
        negatives = self.tn + self.fp
        return self.tn / negatives if negatives else 0.0

    @property
    def balanced_accuracy(self) -> float:
        """Mean of recall and specificity."""
        return (self.recall + self.specificity) / 2

    @property
    def mcc(self) -> float:
        """Matthews correlation coefficient; 0 when any marginal is empty."""
        tp, fp, fn, tn = self.tp, self.fp, self.fn, self.tn
        denominator = float(np.sqrt(float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))))
        return (tp * tn - fp * fn) / denominator if denominator else 0.0


def choose_threshold(labels: object, scores: object) -> float:
    """The threshold that maximises F1 of the forged class; ties go to the higher threshold.

    Candidates are the distinct scores, so the chosen threshold flags at least one document.
    """
    y, s = as_arrays(labels, scores)
    candidates = np.unique(s)[::-1]  # highest first, so that ties keep the higher threshold
    f1 = [Counts.of(y, s >= t).f1 for t in candidates]
    return float(candidates[int(np.argmax(f1))])


def average_precision(y: Labels, s: Scores) -> float:
    """Area under the precision-recall curve as a step function (no interpolation)."""
    order = np.argsort(-s, kind="stable")
    y_sorted, s_sorted = y[order], s[order]
    last_of_tie = np.r_[np.flatnonzero(np.diff(s_sorted)), s_sorted.size - 1]
    tp = np.cumsum(y_sorted)[last_of_tie]
    flagged = last_of_tie + 1
    precision = tp / flagged
    recall = tp / tp[-1]
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def roc_auc(y: Labels, s: Scores) -> float:
    """Probability that a forged document scores above an authentic one, ties counting half."""
    ranks = _average_ranks(s)
    n_pos, n_neg = int(y.sum()), int((~y).sum())
    return float((ranks[y].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def _average_ranks(s: Scores) -> npt.NDArray[np.float64]:
    order = np.argsort(s, kind="stable")
    sorted_scores = s[order]
    starts = np.r_[0, np.flatnonzero(np.diff(sorted_scores)) + 1]
    ends = np.r_[starts[1:], s.size]
    ranks = np.empty(s.size)
    ranks[order] = np.repeat((starts + ends + 1) / 2, ends - starts)
    return ranks


def brier(y: Labels, p: Scores) -> float:
    """Mean squared error of probabilities."""
    return float(np.mean((p - y) ** 2))


def expected_calibration_error(y: Labels, p: Scores, bins: int = ECE_BINS) -> float:
    """Calibration error over equal-mass bins of the probabilities.

    Bin edges are quantiles of the probabilities and documents are binned by value, so tied
    probabilities always share a bin; bins are then only approximately equal in size.
    """
    edges = np.quantile(p, np.linspace(0, 1, bins + 1)[1:-1])
    assigned = np.searchsorted(edges, p, side="right")
    total = 0.0
    for b in np.unique(assigned):
        part = assigned == b
        total += part.sum() * abs(float(np.mean(y[part])) - float(np.mean(p[part])))
    return total / y.size


def recall_at_budget(y: Labels, s: Scores, budget: float) -> float:
    """Recall when only the highest-scoring ``budget`` share of documents is reviewed.

    Documents tied at the cut-off are reviewed in part, each with the same probability, which
    gives the expected recall under a random choice among them.
    """
    capacity = budget * y.size
    total = 0.0
    reviewed = 0.0
    for value in np.unique(s)[::-1]:
        tied = s == value
        n = int(tied.sum())
        share = min(1.0, max(0.0, (capacity - reviewed) / n))
        total += share * float(y[tied].sum())
        reviewed += n
        if reviewed >= capacity:
            break
    return total / float(y.sum())


type Metric = Callable[[Labels, Scores], float]


def threshold_metrics(threshold: float) -> dict[str, Metric]:
    """Metrics of ``score >= threshold``."""

    def counted(name: str) -> Metric:
        return lambda y, s: float(getattr(Counts.of(y, s >= threshold), name))

    names = ("f1", "precision", "recall", "mcc", "balanced_accuracy")
    return {name: counted(name) for name in names}


def ranking_metrics() -> dict[str, Metric]:
    """Threshold-free metrics, including recall at the pre-registered review budgets."""
    metrics: dict[str, Metric] = {"average_precision": average_precision, "roc_auc": roc_auc}
    for budget in REVIEW_BUDGETS:
        metrics[f"recall_at_{round(budget * 100)}pct_review"] = partial(
            recall_at_budget, budget=budget
        )
    return metrics


def calibration_metrics() -> dict[str, Metric]:
    """Metrics that only make sense for probabilities."""
    return {"brier": brier, "expected_calibration_error": expected_calibration_error}


def evaluate(  # noqa: PLR0913 - inputs plus resampling settings
    labels: object,
    scores: object,
    threshold: float,
    *,
    probabilities: bool = False,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict[str, Estimate]:
    """All document-level metrics with stratified bootstrap intervals.

    Raises:
        ValueError: If ``probabilities`` is set but a score lies outside [0, 1].
    """
    y, s = as_arrays(labels, scores)
    metrics = threshold_metrics(threshold) | ranking_metrics()
    if probabilities:
        if s.min() < 0 or s.max() > 1:
            msg = "calibration metrics need probabilities in [0, 1]"
            raise ValueError(msg)
        metrics |= calibration_metrics()
    return {
        name: stratified(_on(metric, y, s), y, resamples=resamples, seed=seed)
        for name, metric in metrics.items()
    }


def compare(  # noqa: PLR0913 - two systems, each with its scores and threshold
    labels: object,
    baseline: object,
    candidate: object,
    baseline_threshold: float,
    candidate_threshold: float,
    *,
    resamples: int = DEFAULT_RESAMPLES,
    seed: int = 0,
) -> dict[str, Estimate]:
    """Paired differences ``candidate - baseline``, both evaluated on the same draws."""
    y, a = as_arrays(labels, baseline)
    _, b = as_arrays(labels, candidate)
    of_a = threshold_metrics(baseline_threshold) | ranking_metrics()
    of_b = threshold_metrics(candidate_threshold) | ranking_metrics()
    return {
        name: stratified(
            _difference(of_a[name], of_b[name], y, a, b), y, resamples=resamples, seed=seed
        )
        for name in of_a
    }


type Statistic = Callable[[npt.NDArray[np.intp]], float]


def _on(metric: Metric, y: Labels, s: Scores) -> Statistic:
    return lambda idx: metric(y[idx], s[idx])


def _difference(of_a: Metric, of_b: Metric, y: Labels, a: Scores, b: Scores) -> Statistic:
    return lambda idx: of_b(y[idx], b[idx]) - of_a(y[idx], a[idx])


def summary(
    labels: object,
    scores: object,
    threshold: float,
    doc_ids: Sequence[str],
    *,
    probabilities: bool = False,
) -> dict[str, Any]:
    """JSON-ready results of one detector: sizes, metrics with intervals, counts and scores."""
    y, s = as_arrays(labels, scores)
    counts = Counts.of(y, s >= threshold)
    return {
        "n_documents": int(y.size),
        "n_forged": int(y.sum()),
        "threshold": threshold,
        "metrics": {
            name: {"value": e.value, "low": e.low, "high": e.high}
            for name, e in evaluate(y, s, threshold, probabilities=probabilities).items()
        },
        "counts": {"tp": counts.tp, "fp": counts.fp, "fn": counts.fn, "tn": counts.tn},
        "scores": dict(zip(doc_ids, (float(v) for v in s), strict=True)),
    }
