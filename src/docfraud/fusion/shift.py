"""Do the fusion inputs look the same on both splits? (ADR-0029, section 7 (d)).

The training inputs come from the cross-validation models and the validation inputs from the
models trained on the whole training split (section 2). Before the validation check, every
input's distribution is compared between the two, without labels, and every difference is
reported; nothing is decided by it.

For every input the comparison gives the two sides' summaries and the Kolmogorov-Smirnov
distance D, the largest gap between their cumulative distributions; for an input that is 0 or
1, D is the difference between the shares of 1. Its p-value comes from a permutation test: the
receipts' sides are shuffled and D computed again, so the p-value stays valid where values tie,
as in the 0/1 inputs and the many zeros of system 1 and copy-move, where the asymptotic p-value
of the KS test is conservative. Both were tried on random splits of the training inputs before
this module was written ([input-shift-method.md](../../../reports/fusion/input-shift-method.md)),
with a classifier that tells the two sides apart, which was dropped: it found a shift of one
input less often than the KS distance did. The 95th percentile of the shuffled distances is
reported as the distance a difference must reach to give p < 0.05.

Each input is compared at the level it varies on: the page findings once per receipt, the
rules' score per text source, system 2 per seed and the links per text source and seed, 21
comparisons in all. Inputs that should not differ between text sources or seeds are checked
to be equal. About one comparison in twenty is expected below 0.05 by chance; no correction is
made. A difference can come from the models or from the receipts themselves; without labels the
two cannot be told apart.
"""

from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from docfraud.fusion.development_report import TRACK_NAMES
from docfraud.fusion.inputs import (
    COPY_MOVE_INPUT,
    LINK_1,
    LINK_2,
    METADATA_INPUT,
    PRACTICAL,
    RULES_INPUT,
    SYSTEM_1_EVIDENCE,
    SYSTEM_1_INPUT,
    SYSTEM_2_INPUT,
)
from docfraud.fusion.training_inputs import InputsTable

type Vector = npt.NDArray[np.float64]

RUN_NAME = "fusion-input-shift"
PERMUTATIONS = 10_000
_BATCH = 1_000  # permutations drawn at once

RECEIPT, TRACK, SEED, TRACK_AND_SEED = "receipt", "track", "seed", "track and seed"
VARIES_BY = {
    RULES_INPUT: TRACK,
    SYSTEM_2_INPUT: SEED,
    SYSTEM_1_INPUT: RECEIPT,
    SYSTEM_1_EVIDENCE: RECEIPT,
    COPY_MOVE_INPUT: RECEIPT,
    LINK_1: TRACK_AND_SEED,
    LINK_2: TRACK_AND_SEED,
    METADATA_INPUT: RECEIPT,
}
"""What each input depends on besides the receipt: the text source, system 2's seed, or both."""


def ks_distance(first: Vector, second: Vector) -> float:
    """The largest gap between the cumulative distributions of two samples."""
    values = np.unique(np.concatenate([first, second]))
    below_first = np.searchsorted(np.sort(first), values, side="right") / first.size
    below_second = np.searchsorted(np.sort(second), values, side="right") / second.size
    return float(np.abs(below_first - below_second).max())


@dataclass(frozen=True, slots=True)
class Permuted:
    """A distance with its permutation p-value and the distance that p < 0.05 needs."""

    distance: float
    p_value: float
    critical: float  # the 95th percentile of the shuffled distances


def permutation_test(
    first: Vector, second: Vector, permutations: int = PERMUTATIONS, seed: int = 0
) -> Permuted:
    """The KS distance of two samples against its distribution when sides are shuffled.

    The p-value counts the observed split among the permutations, so it is never 0.
    """
    pooled = np.concatenate([first, second])
    order = np.argsort(pooled, kind="stable")
    ordered = pooled[order]
    # the last position of every distinct value, where the cumulative distributions are read
    ends = np.flatnonzero(np.r_[ordered[1:] != ordered[:-1], True])
    n, total = first.size, pooled.size
    side = np.zeros(total, dtype=bool)
    side[:n] = True
    observed = _distances(side[order][None, :], ends, n, total)[0]
    rng = np.random.default_rng(seed)
    shuffled = np.concatenate(
        [
            _distances(rng.permuted(np.tile(side, (size, 1)), axis=1)[:, order], ends, n, total)
            for size in _batches(permutations)
        ]
    )
    return Permuted(
        distance=float(observed),
        p_value=float((1 + np.sum(shuffled >= observed - 1e-12)) / (1 + permutations)),
        critical=float(np.quantile(shuffled, 0.95)),
    )


def _batches(permutations: int) -> Iterator[int]:
    for start in range(0, permutations, _BATCH):
        yield min(_BATCH, permutations - start)


def _distances(
    sides: npt.NDArray[np.bool_], ends: npt.NDArray[np.intp], n: int, total: int
) -> Vector:
    """The KS distance of each row's split, with the receipts in sorted order of their values."""
    in_first = np.cumsum(sides, axis=1)[:, ends]
    in_second = (ends + 1) - in_first
    gaps = np.abs(in_first / n - in_second / (total - n))
    return np.asarray(gaps.max(axis=1), dtype=float)


def summary(values: Vector) -> dict[str, float]:
    """One side's mean, share of zeros, median and 90th percentile."""
    return {
        "mean": float(values.mean()),
        "zeros": float(np.mean(values == 0)),
        "median": float(np.median(values)),
        "q90": float(np.quantile(values, 0.9)),
    }


def _column(table: InputsTable, name: str, track: str, seed: int) -> tuple[list[str], Vector]:
    """The receipts of one text source and seed, and one input of each.

    Raises:
        ValueError: If the table has no rows for them.
    """
    rows = table.rows(track, seed)
    if not rows:
        msg = f"no rows for {track}, seed {seed}"
        raise ValueError(msg)
    return [r.receipt for r in rows], np.array([r.inputs[name] for r in rows], dtype=float)


def _scopes(name: str, seeds: Sequence[int]) -> list[tuple[str | None, int | None]]:
    """The text sources and seeds an input is compared at; None where it does not vary."""
    tracks: list[str | None] = list(TRACK_NAMES)
    by_seed: list[int | None] = list(seeds)
    varies = VARIES_BY[name]
    return [
        (t, s)
        for t in (tracks if varies in {TRACK, TRACK_AND_SEED} else [None])
        for s in (by_seed if varies in {SEED, TRACK_AND_SEED} else [None])
    ]


def _compared(
    table: InputsTable, name: str, track: str | None, seed: int | None, seeds: Sequence[int]
) -> Vector:
    """An input's values at one scope, checked to be equal wherever it should not vary.

    Raises:
        ValueError: If a text source or seed has no rows, its receipts differ from the others',
            or the input differs between text sources or seeds that it does not depend on.
    """
    tracks = [track] if track is not None else list(TRACK_NAMES)
    at = [seed] if seed is not None else list(seeds)
    found = [_column(table, name, t, s) for t in tracks for s in at]
    receipts, values = found[0]
    if any(other != receipts for other, _ in found[1:]):
        msg = "the text sources and seeds of a table hold different receipts"
        raise ValueError(msg)
    if any(not np.array_equal(values, other) for _, other in found[1:]):
        msg = f"{name} differs between text sources or seeds it does not depend on"
        raise ValueError(msg)
    return values


def input_shift(
    training: InputsTable,
    validation: InputsTable,
    *,
    seeds: Sequence[int],
    permutations: int = PERMUTATIONS,
) -> dict[str, Any]:
    """Every input of both splits compared at the level it varies on, with no label read."""
    comparisons: list[dict[str, Any]] = []
    below = 0
    for name in PRACTICAL:
        for track, seed in _scopes(name, seeds):
            first = _compared(training, name, track, seed, seeds)
            second = _compared(validation, name, track, seed, seeds)
            test = permutation_test(first, second, permutations)
            below += test.p_value < 0.05  # noqa: PLR2004 - the usual level, stated in the report
            comparisons.append(
                {
                    "input": name,
                    "track": track,
                    "seed": seed,
                    "training": summary(first),
                    "validation": summary(second),
                    "distance": test.distance,
                    "p_value": test.p_value,
                    "critical": test.critical,
                }
            )
    return {
        "receipts": {
            "training": len(training.rows(next(iter(TRACK_NAMES)), seeds[0])),
            "validation": len(validation.rows(next(iter(TRACK_NAMES)), seeds[0])),
        },
        "permutations": permutations,
        "comparisons": comparisons,
        "below_0_05": below,
    }
