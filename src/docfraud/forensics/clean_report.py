"""The report of the clean measurement: every Phase 4 system on the clean folds, once (ADR-0020).

Every table is made from what the measurement kept (:class:`CleanData`): the records and results
of the 36 runs of systems 2 to 4, the results of systems 1 and 6 and of the probe in every clean
round, with the receipts of the clean folds and their OCR words. Nothing here runs a system. The
tables are those that ADR-0014, ADR-0015, ADR-0017, ADR-0018 and ADR-0022 fix, with what the
record of the clean measurement (``docs/clean-measurement.md``, section 2) added before any
system produced an output on a clean fold. Nothing in this report chooses anything.

Each system's document scores are pooled over the four clean rounds (:func:`pooled`). Systems 2
to 4 have a model per round and seed; their headline numbers come from seed 0, chosen in advance
(evaluation protocol, section 6), and the spread over seeds is shown apart. Thresholds come from
the validation split, each round's for its own model: a receipt is flagged when its score reaches
the threshold of its round.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass
from functools import partial

import numpy as np
import numpy.typing as npt

from docfraud.data.models import BBox, Document, Label
from docfraud.evaluation.bootstrap import DEFAULT_RESAMPLES, Estimate, ratio, stratified
from docfraud.evaluation.breakdowns import breakdown, flagged_by_budget, rules_blind
from docfraud.evaluation.design_fold import CLEAN_FOLDS, EARLIER_CONTACTS
from docfraud.evaluation.detection import (
    REVIEW_BUDGETS,
    Metric,
    average_precision,
    calibration_metrics,
    choose_threshold,
    recall_at_budget,
    roc_auc,
    threshold_metrics,
)
from docfraud.evaluation.localisation import (
    LARGE_RECTANGLE_SHARE,
    Guard,
    baseline_hit_rate,
    hits,
    localisation_guard,
    receipt_rng,
    size_dependence,
    spearman,
)
from docfraud.evaluation.runs import CodeVersion
from docfraud.forensics.clean import (
    LATTICE,
    LATTICE_SCORE,
    PROBE,
    TRUFOR,
    TRUFOR_SCORE,
    ProbeResults,
)
from docfraud.forensics.learned.inference import HIGHEST, QUARTER_SQUARE
from docfraud.forensics.learned.results import Group, MeasuredReceipt, RunResults
from docfraud.forensics.learned.sampler import System
from docfraud.forensics.learned.text_scale import page_scan
from docfraud.forensics.learned.training import RunRecord, binary_entropy, failure
from docfraud.forensics.probe import PROPERTIES
from docfraud.forensics.trufor import CITATION
from docfraud.markdown import Cell, Table, render_table

MAP_SYSTEMS = (LATTICE, *(int(s) for s in System), TRUFOR)
"""The systems with a map, in the order of ADR-0014 (decision 6)."""

NAMES = {
    LATTICE: "system 1, JPEG lattice",
    int(System.SYNTHETIC): "system 2, synthetic forgeries",
    int(System.REAL): "system 3, real forgeries",
    int(System.BOTH): "system 4, both",
    PROBE: "system 5, global-trace probe",
    TRUFOR: "system 6, TruFor",
}

SCORES = {
    LATTICE: LATTICE_SCORE,
    **{int(s): QUARTER_SQUARE for s in System},
    TRUFOR: TRUFOR_SCORE,
}
"""The document score each frozen system uses (docs/clean-measurement.md, section 1)."""

HEADLINE_SEED = 0
BUDGET = 0.10
"""The review budget of the breakdowns (docs/clean-measurement.md, section 2)."""

WINDOW = 500
"""Steps over which the failure rule of ADR-0022 averages the loss."""

type LearnedKey = tuple[int, System, int]  # held-out fold, system, seed


@dataclass(frozen=True, slots=True)
class CleanData:
    """What the report of the clean measurement is made from."""

    records: Mapping[LearnedKey, RunRecord]
    learned: Mapping[LearnedKey, RunResults]
    untrained: Mapping[tuple[int, int], RunResults]  # held-out fold and system 1 or 6
    probe: Mapping[int, ProbeResults]  # by held-out fold
    documents: Mapping[str, Document]  # every receipt of the clean folds
    words: Mapping[str, Sequence[BBox]]  # their OCR word boxes
    rounds: tuple[int, ...] = CLEAN_FOLDS
    earlier: frozenset[str] = EARLIER_CONTACTS  # receipts seen before ADR-0014 (decision 1)
    resamples: int = DEFAULT_RESAMPLES  # of every bootstrap interval

    @property
    def seeds(self) -> list[int]:
        """The seeds of systems 2 to 4."""
        return sorted({seed for _, _, seed in self.learned})


def results_of(data: CleanData, system: int, held_out: int, seed: int) -> RunResults:
    """The results of a system in a round: of its run with the seed, for systems 2 to 4."""
    if system in (LATTICE, TRUFOR):
        return data.untrained[(held_out, system)]
    return data.learned[(held_out, System(system), seed)]


@dataclass(frozen=True, slots=True)
class Pooled:
    """One system's receipts of every clean round, each with its round's document threshold."""

    system: int
    score: str
    receipts: tuple[MeasuredReceipt, ...]
    thresholds: tuple[float, ...]
    rounds: tuple[int, ...]  # the round of each receipt

    @property
    def labels(self) -> npt.NDArray[np.bool_]:
        """Whether each receipt is forged."""
        return np.array([r.forged for r in self.receipts], dtype=bool)

    @property
    def scores(self) -> npt.NDArray[np.float64]:
        """The document score of each receipt."""
        return np.array([r.scores[self.score] for r in self.receipts], dtype=np.float64)

    @property
    def margins(self) -> npt.NDArray[np.float64]:
        """Each score minus its round's threshold: a receipt is flagged when this is not below 0."""
        return self.scores - np.array(self.thresholds, dtype=np.float64)

    def only(self, keep: Callable[[MeasuredReceipt], bool]) -> Pooled:
        """The receipts that ``keep`` accepts, with their thresholds."""
        chosen = [i for i, r in enumerate(self.receipts) if keep(r)]
        return Pooled(
            self.system,
            self.score,
            tuple(self.receipts[i] for i in chosen),
            tuple(self.thresholds[i] for i in chosen),
            tuple(self.rounds[i] for i in chosen),
        )


def pooled(data: CleanData, system: int, seed: int = HEADLINE_SEED) -> Pooled:
    """A system's held-out receipts of every clean round, in round and identifier order.

    Raises:
        ValueError: If the receipts are not exactly those of the clean folds, once each.
    """
    score = SCORES[system]
    receipts: list[MeasuredReceipt] = []
    thresholds: list[float] = []
    rounds: list[int] = []
    for held_out in data.rounds:
        results = results_of(data, system, held_out, seed)
        threshold = choose_threshold(
            [r.forged for r in results.validation], [r.scores[score] for r in results.validation]
        )
        held = sorted((r for r in results.measured if r.group is Group.HELD_OUT), key=_key)
        receipts += held
        thresholds += [threshold] * len(held)
        rounds += [held_out] * len(held)
    keys = [r.key for r in receipts]
    if len(set(keys)) != len(keys) or set(keys) != set(data.documents):
        msg = f"system {system}, seed {seed}: the receipts are not those of the clean folds"
        raise ValueError(msg)
    return Pooled(system, score, tuple(receipts), tuple(thresholds), tuple(rounds))


def _key(receipt: MeasuredReceipt) -> str:
    return receipt.key


def _interval(estimate: Estimate, digits: int = 3) -> str:
    return f"{estimate.value:.{digits}f} ({estimate.low:.{digits}f} to {estimate.high:.{digits}f})"


def _signed(estimate: Estimate) -> str:
    return f"{estimate.value:+.3f} ({estimate.low:+.3f} to {estimate.high:+.3f})"


def _metric(
    metric: Metric,
    labels: npt.NDArray[np.bool_],
    values: npt.NDArray[np.float64],
    resamples: int,
) -> Estimate:
    """A document metric with the stratified bootstrap interval of the evaluation protocol."""
    return stratified(lambda d: metric(labels[d], values[d]), labels, resamples=resamples)


def _mean(values: Iterable[float], resamples: int) -> Estimate | None:
    """A mean over receipts with its bootstrap interval, if there are any."""
    chosen = [float(v) for v in values]
    if not chosen:
        return None
    return ratio(chosen, [1.0] * len(chosen), resamples=resamples)


def _page(receipt: MeasuredReceipt) -> BBox:
    return BBox(x=0, y=0, width=receipt.width, height=receipt.height)


def large(receipt: MeasuredReceipt) -> bool:
    """Whether a modified rectangle covers more than 5 % of the page (ADR-0015, section 4)."""
    area = receipt.width * receipt.height
    return any(box.area > LARGE_RECTANGLE_SHARE * area for box in receipt.modified)


def evidence(receipt: MeasuredReceipt) -> bool:
    """Whether the map holds more than one value (ADR-0017)."""
    return receipt.scores[HIGHEST] > receipt.lowest


@dataclass(frozen=True, slots=True)
class Localised:
    """Whether a forged receipt's top region hits, and its random baseline (ADR-0015, 3)."""

    hit: bool
    baseline: float


def localised(
    data: CleanData, receipts: Iterable[MeasuredReceipt], targets: str = "both"
) -> dict[str, Localised]:
    """The hit and random baseline of every forged receipt that has the targets.

    The targets are its modified rectangles and copy sources (``both``, as the guard counts
    them), or its copy sources only (``sources``).
    """
    found: dict[str, Localised] = {}
    for r in receipts:
        rectangles = [*r.modified, *r.sources] if targets == "both" else list(r.sources)
        if not r.forged or not rectangles or (targets == "both" and not r.modified):
            continue
        found[r.key] = Localised(
            hits(r.top_region, rectangles),
            baseline_hit_rate(
                r.top_region, _page(r), data.words[r.key], rectangles, receipt_rng(r.key)
            ),
        )
    return found


def guard(found: Mapping[str, Localised], resamples: int) -> Guard | None:
    """The localisation guard over receipts, if there are any."""
    if not found:
        return None
    chosen = list(found.values())
    return localisation_guard(
        [c.hit for c in chosen], [c.baseline for c in chosen], resamples=resamples
    )


def _verdict(found: Guard | None) -> str | None:
    if found is None:
        return None
    return "detects forgeries" if found.detects else "probable shortcut"


def _guard_cells(found: Guard | None) -> list[Cell]:
    if found is None:
        return [0, None, None, None]
    return [found.receipts, _interval(found.hit_rate), found.baseline, _verdict(found)]


def runs_table(data: CleanData) -> Table:
    """How each run of systems 2 to 4 trained and how it fared under the failure rule."""
    rows: list[list[Cell]] = []
    for key in sorted(data.records):
        held_out, system, seed = key
        record = data.records[key]
        window = int(record.settings["failure_window"])
        rows.append(
            [
                held_out,
                f"system {int(system)}, seed {seed}",
                record.status,
                record.steps_done,
                f"{sum(record.seconds) / 3600:.2f}",
                float(np.mean(record.losses[-window:])),
                binary_entropy(float(np.mean(record.forged_shares[-window:]))),
                failure(record) or "learned",
                f"`{data.learned[key].model}`",
            ]
        )
    return Table(
        "Runs of systems 2 to 4",
        [
            "Round",
            "Run",
            "Status",
            "Steps",
            "Hours",
            f"Mean loss, last {WINDOW} steps",
            "Loss of the best constant",
            "Failure rule",
            "Network",
        ],
        rows,
        "Each round holds out its clean fold and trains on the other four. Failure rule of "
        "ADR-0022 (decision 5): a run diverges if its loss is ever not finite and does not learn "
        "if its mean loss over the last 500 steps is not below that of the best constant "
        "prediction on the same tiles.",
    )


def untrained_table(data: CleanData) -> Table:
    """What measured systems 1 and 6 and trained the probe in every round."""
    rows: list[list[Cell]] = []
    for held_out in data.rounds:
        for system in (LATTICE, TRUFOR):
            results = data.untrained[(held_out, system)]
            rows.append(
                [
                    held_out,
                    NAMES[system],
                    len(results.validation),
                    len(results.measured),
                    f"`{results.model}`",
                ]
            )
        probe = data.probe[held_out]
        rows.append(
            [
                held_out,
                NAMES[PROBE],
                f"trained on {probe.trained_on} ({probe.trained_forged} forged)",
                len(probe.receipts),
                f"`{probe.model}`",
            ]
        )
    return Table(
        "Systems without a run",
        ["Round", "System", "Validation receipts", "Held-out receipts", "Model"],
        rows,
        "Systems 1 and 6 are not trained, so every round measures the same model; each round is "
        "still measured on its own, its validation split choosing its thresholds. The probe is a "
        "random forest trained in each round on the receipts of the other four folds (ADR-0018).",
    )


def primary_table(data: CleanData) -> Table:
    """The primary metric of ADR-0014 (decision 4), also without the receipts of earlier contact."""
    rows: list[list[Cell]] = []
    for system in MAP_SYSTEMS:
        found = pooled(data, system)
        kept = np.array([r.key not in data.earlier for r in found.receipts], dtype=bool)
        rows.append([NAMES[system], *_primary_cells(found.labels, found.scores, kept, data)])
    labels, scores, kept = _probe_arrays(data)
    rows.append([f"{NAMES[PROBE]} (not a detector)", *_primary_cells(labels, scores, kept, data)])
    return Table(
        "Primary metric: document-level average precision",
        [
            "System",
            "Receipts",
            "Forged",
            "Chance",
            "Average precision (95 % interval)",
            "ROC-AUC",
            "Without earlier contact: receipts",
            "Forged",
            "Chance",
            "Average precision (95 % interval)",
        ],
        rows,
        "Average precision of the document score over the pooled scores of the clean rounds "
        "(ADR-0014, decision 4); the interval is the stratified bootstrap of the evaluation "
        f"protocol ({data.resamples:,} resamples, seed 0). Chance: the share of forged receipts. "
        f"Systems 2 to 4: seed 0. Without earlier contact: the {_earlier(data)} receipts viewed "
        "before ADR-0014 are left out, a sensitivity check (decision 1).",
    )


def _earlier(data: CleanData) -> int:
    """How many of the measured receipts were viewed before ADR-0014."""
    return len(data.earlier & set(data.documents))


def _probe_arrays(
    data: CleanData,
) -> tuple[npt.NDArray[np.bool_], npt.NDArray[np.float64], npt.NDArray[np.bool_]]:
    """The probe's pooled labels and scores, and which receipts are not of earlier contact."""
    receipts = [r for held_out in data.rounds for r in data.probe[held_out].receipts]
    return (
        np.array([r.forged for r in receipts], dtype=bool),
        np.array([r.score for r in receipts], dtype=np.float64),
        np.array([r.key not in data.earlier for r in receipts], dtype=bool),
    )


def _primary_cells(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    kept: npt.NDArray[np.bool_],
    data: CleanData,
) -> list[Cell]:
    """Size, chance and AP of all receipts, their ROC-AUC, then the same without earlier contact."""
    ap = _metric(average_precision, labels, scores, data.resamples)
    y, s = labels[kept], scores[kept]
    without = _metric(average_precision, y, s, data.resamples)
    return [
        int(labels.size),
        int(labels.sum()),
        float(labels.mean()),
        _interval(ap),
        roc_auc(labels, scores),
        int(y.size),
        int(y.sum()),
        float(y.mean()),
        _interval(without),
    ]


def _difference(
    data: CleanData, first: Pooled, second: Pooled, keep: Callable[[str], bool]
) -> Estimate:
    """The paired bootstrap of ``second - first`` in average precision on the same receipts."""
    by_key = {r.key: r.scores[second.score] for r in second.receipts}
    chosen = [i for i, r in enumerate(first.receipts) if keep(r.key)]
    labels = first.labels[chosen]
    a = first.scores[chosen]
    b = np.array([by_key[first.receipts[i].key] for i in chosen], dtype=np.float64)
    return stratified(
        lambda d: average_precision(labels[d], b[d]) - average_precision(labels[d], a[d]),
        labels,
        resamples=data.resamples,
    )


def comparisons(data: CleanData) -> list[tuple[str, int, int, Estimate, Estimate]]:
    """RQ5 and the secondary comparison (ADR-0014, decision 6).

    Each as its name, the two systems, the difference and the difference without the receipts of
    earlier contact.
    """
    found: list[tuple[str, int, int, Estimate, Estimate]] = []
    for name, base, other in (
        ("RQ5: system 4 against system 3", int(System.REAL), int(System.BOTH)),
        ("Secondary: system 2 against system 3", int(System.REAL), int(System.SYNTHETIC)),
    ):
        first, second = pooled(data, base), pooled(data, other)
        found.append(
            (
                name,
                base,
                other,
                _difference(data, first, second, lambda _key: True),
                _difference(data, first, second, lambda key: key not in data.earlier),
            )
        )
    return found


def comparison_table(data: CleanData) -> Table:
    """The comparisons of ADR-0014 (decision 6) by the paired bootstrap."""
    rows: list[list[Cell]] = []
    for name, base, other, difference, without in comparisons(data):
        rows.append(
            [
                name,
                average_precision(pooled(data, base).labels, pooled(data, base).scores),
                average_precision(pooled(data, other).labels, pooled(data, other).scores),
                _signed(difference),
                difference.excludes_zero,
                _signed(without),
            ]
        )
    return Table(
        "Comparisons of the learned models",
        [
            "Comparison",
            "AP, system 3",
            "AP, the other system",
            "Difference (95 % interval)",
            "Difference claimed",
            "Without earlier contact",
        ],
        rows,
        "The other system's average precision minus system 3's, seed 0, by the paired bootstrap "
        "of the evaluation protocol: both systems are resampled with the same receipts, within "
        "each class. A difference is claimed only if its interval excludes zero (ADR-0014, "
        "decision 4); the comparison without the receipts of earlier contact decides nothing.",
    )


def guard_table(data: CleanData) -> Table:
    """The localisation guard on all forged receipts, without large ones and without contact."""
    rows: list[list[Cell]] = []
    for system in MAP_SYSTEMS:
        receipts = {r.key: r for r in pooled(data, system).receipts}
        found = localised(data, receipts.values())
        subsets = (
            found,
            {k: v for k, v in found.items() if not large(receipts[k])},
            {k: v for k, v in found.items() if k not in data.earlier},
        )
        cells: list[Cell] = [NAMES[system]]
        for subset in subsets:
            cells += _guard_cells(guard(subset, data.resamples))
        rows.append(cells)
    columns = ["System"]
    for name in ("all", "without large rectangles", "without earlier contact"):
        columns += [f"Forged, {name}", "Hit rate (95 % interval)", "Random baseline", "Verdict"]
    return Table(
        "Localisation guard",
        columns,
        rows,
        "The top region of each forged receipt with region annotations (ADR-0015, section 2) "
        "hits if it shares a pixel with a modified rectangle or a copy source on the page; the "
        "baseline is the share of 1,000 boxes of the same size, centred on OCR text, that hit. A "
        "system detects forgeries if the lower bound of its hit rate's interval (bootstrap over "
        "receipts) exceeds the baseline; otherwise its result is a probable shortcut (ADR-0014, "
        "decision 4). A map without evidence makes the whole page its top region, which hits with "
        "a baseline of 1 (ADR-0015, section 2), so such receipts raise a system's hit rate and its "
        "baseline alike. Large rectangles cover more than 5 % of the page (ADR-0015, section 4). "
        "Systems 2 to 4: seed 0.",
    )


def source_table(data: CleanData) -> Table:
    """How often the top region finds the source of a copy, where one is annotated."""
    rows: list[list[Cell]] = []
    for system in MAP_SYSTEMS:
        found = localised(data, pooled(data, system).receipts, targets="sources")
        rows.append([NAMES[system], *_guard_cells(guard(found, data.resamples))])
    return Table(
        "Copy sources",
        [
            "System",
            "Forged with a copy source",
            "Hit rate (95 % interval)",
            "Random baseline",
            "Verdict",
        ],
        rows,
        "Copy-move source localisation (evaluation protocol, section 5): the top region against "
        "the annotated original areas only, on the receipts that have one; the baseline and the "
        "verdict follow the guard's rules. Systems 2 to 4: seed 0.",
    )


def secondary_table(data: CleanData) -> Table:
    """The secondary document metrics, at each round's validation threshold and at budgets."""
    rows: list[list[Cell]] = []
    for system in MAP_SYSTEMS:
        found = pooled(data, system)
        rows.append(
            [NAMES[system], *_secondary_cells(found.labels, found.scores, found.margins, data)]
        )
    labels = pooled(data, MAP_SYSTEMS[0]).labels
    flat = np.zeros(labels.size)
    rows.append(["flag everything", *_secondary_cells(labels, flat, flat, data)])
    share = float(labels.mean())
    return Table(
        "Secondary document metrics",
        [
            "System",
            "F1",
            "Precision",
            "Recall",
            "MCC",
            "Balanced accuracy",
            *(f"Recall at {round(100 * b)} %" for b in REVIEW_BUDGETS),
            "ROC-AUC",
        ],
        rows,
        "Threshold metrics flag a receipt whose score reaches the threshold that maximises F1 on "
        "its own round's validation scores (ADR-0014, decision 4). Recall at a review budget: "
        "only that share of the highest scores is reviewed, ties in part. Flag everything is the "
        f"reference of ADR-0013: its F1, {2 * share / (1 + share):.3f}, is the floor of F1; "
        "the chance level of ROC-AUC is 0.5 and that of recall at budget b is b. Systems 2 to 4: "
        "seed 0; 95 % intervals by the stratified bootstrap.",
    )


def _secondary_cells(
    labels: npt.NDArray[np.bool_],
    scores: npt.NDArray[np.float64],
    margins: npt.NDArray[np.float64],
    data: CleanData,
) -> list[Cell]:
    metrics: list[tuple[Metric, npt.NDArray[np.float64]]] = [
        *((metric, margins) for metric in threshold_metrics(0.0).values()),
        *((partial(recall_at_budget, budget=budget), scores) for budget in REVIEW_BUDGETS),
        (roc_auc, scores),
    ]
    return [
        _interval(_metric(metric, labels, values, data.resamples)) for metric, values in metrics
    ]


def calibration_table(data: CleanData) -> Table:
    """Brier score and expected calibration error, where the score is a probability of forgery."""
    rows: list[list[Cell]] = []
    for system in MAP_SYSTEMS:
        if system != TRUFOR:
            rows.append([NAMES[system], "not a probability", "not a probability"])
            continue
        found = pooled(data, system)
        rows.append(
            [
                NAMES[system],
                *(
                    _interval(_metric(metric, found.labels, found.scores, data.resamples))
                    for metric in calibration_metrics().values()
                ),
            ]
        )
    return Table(
        "Calibration",
        ["System", "Brier score (95 % interval)", "Expected calibration error (95 % interval)"],
        rows,
        "The evaluation protocol (section 5) lists both. Only system 6's document score is a "
        "probability that the receipt is forged, the sigmoid of TruFor's detection output, so they "
        "are reported for it alone (docs/clean-measurement.md, section 2). Expected calibration "
        "error over 10 bins of equal size.",
    )


def pixel_table(data: CleanData) -> Table:
    """Pixel metrics on the forged receipts with region annotations (ADR-0015, section 5)."""
    rows: list[list[Cell]] = []
    for system in MAP_SYSTEMS:
        found = pooled(data, system)
        pixels = [r.pixels for r in found.receipts if r.pixels is not None]
        thresholds = [
            results_of(data, system, k, HEADLINE_SEED).threshold.value for k in data.rounds
        ]
        rows.append(
            [
                NAMES[system],
                ", ".join(f"{t:.3g}" for t in thresholds),
                len(pixels),
                *_mean_cells([p.average_precision for p in pixels], data.resamples),
                _plain_mean(p.chance for p in pixels),
                *_mean_cells([p.f1 for p in pixels], data.resamples),
                *_mean_cells([p.iou for p in pixels], data.resamples),
            ]
        )
    return Table(
        "Pixel metrics",
        [
            "System",
            "Pixel thresholds by round",
            "Forged",
            "Pixel AP (95 % interval)",
            "Chance",
            "F1 (95 % interval)",
            "IoU (95 % interval)",
        ],
        rows,
        "Per forged receipt with region annotations, then averaged with equal weight; pixels of a "
        "copy source outside every modified rectangle are ignored (ADR-0015, section 5). Chance: "
        "the mean share of modified pixels. F1 and IoU at each round's pixel threshold, chosen on "
        "the forged receipts of the validation split. Intervals by the bootstrap over receipts. "
        "Systems 2 to 4: seed 0.",
    )


def _mean_cells(values: Sequence[float], resamples: int) -> list[Cell]:
    estimate = _mean(values, resamples)
    return [None if estimate is None else _interval(estimate)]


def _plain_mean(values: Iterable[float]) -> float | None:
    chosen = list(values)
    return float(np.mean(chosen)) if chosen else None


def seed_table(data: CleanData) -> Table:
    """The spread over seeds of systems 2 to 4 (evaluation protocol, section 6)."""
    rows: list[list[Cell]] = []
    for system in System:
        values: dict[str, list[float]] = {"ap": [], "hit": [], "pixel": []}
        for seed in data.seeds:
            found = pooled(data, int(system), seed)
            ap = average_precision(found.labels, found.scores)
            located = localised(data, found.receipts)
            hit = float(np.mean([v.hit for v in located.values()])) if located else float("nan")
            pixel = _plain_mean(r.pixels.average_precision for r in found.receipts if r.pixels)
            values["ap"].append(ap)
            values["hit"].append(hit)
            values["pixel"].append(float("nan") if pixel is None else pixel)
            rows.append([NAMES[int(system)], seed, ap, hit, pixel])
        rows.append(
            [
                NAMES[int(system)],
                "mean ± standard deviation",
                *(_spread(values[name]) for name in ("ap", "hit", "pixel")),
            ]
        )
    return Table(
        "Seeds",
        ["System", "Seed", "Average precision", "Hit rate", "Pixel AP"],
        rows,
        "Each seed's runs of the rounds, measured as seed 0 is; the headline numbers above "
        "are seed 0's, chosen in advance (evaluation protocol, section 6). Standard deviation over "
        "the seeds, with one degree of freedom removed.",
    )


def _spread(values: Sequence[float]) -> str:
    if len(values) < 2:  # noqa: PLR2004 - a spread needs two seeds
        return f"{values[0]:.3f}" if values else "—"
    return f"{float(np.mean(values)):.3f} ± {float(np.std(values, ddof=1)):.3f}"


def blind_table(data: CleanData) -> Table:
    """Every metric on the forged receipts that the accounting rules cannot see (ADR-0014, 4)."""
    blind = {k for k, d in data.documents.items() if d.label is Label.FORGED and rules_blind(d)}
    rows: list[list[Cell]] = []
    for system in MAP_SYSTEMS:
        found = pooled(data, system).only(lambda r: not r.forged or r.key in blind)
        y = found.labels
        f1 = threshold_metrics(0.0)["f1"]
        located = localised(data, found.receipts)
        pixels = [r.pixels.average_precision for r in found.receipts if r.pixels is not None]
        rows.append(
            [
                NAMES[system],
                int(y.sum()),
                float(y.mean()),
                _interval(_metric(average_precision, y, found.scores, data.resamples)),
                roc_auc(y, found.scores),
                _interval(_metric(f1, y, found.margins, data.resamples)),
                recall_at_budget(y, found.scores, BUDGET),
                *_guard_cells(guard(located, data.resamples)),
                _plain_mean(pixels),
            ]
        )
    return Table(
        "Forged receipts the accounting rules cannot see",
        [
            "System",
            "Forged",
            "Chance",
            "Average precision (95 % interval)",
            "ROC-AUC",
            "F1 (95 % interval)",
            "Recall at 10 %",
            "Forged with regions",
            "Hit rate (95 % interval)",
            "Random baseline",
            "Verdict",
            "Pixel AP",
        ],
        rows,
        "The forged clean-fold receipts whose annotated changes avoid amounts and products, "
        "against every authentic receipt of the clean folds (ADR-0014, decision 4). Thresholds, "
        "budget, guard and pixel metrics as above. Systems 2 to 4: seed 0.",
    )


def breakdown_tables(data: CleanData) -> list[Table]:
    """Forged receipts found and authentic ones flagged at the 10 % budget, by group."""
    documents = [data.documents[k] for k in sorted(data.documents)]
    found = {
        system: breakdown(
            documents,
            flagged_by_budget(
                {r.key: r.scores[SCORES[system]] for r in pooled(data, system).receipts}, BUDGET
            ),
        )
        for system in MAP_SYSTEMS
    }
    first = found[MAP_SYSTEMS[0]]
    forged_rows: list[list[Cell]] = []
    for kind, groups in first["forged"].items():
        for name, counts in groups.items():
            forged_rows.append(
                [
                    kind,
                    name,
                    int(counts["receipts"]),
                    *(f"{found[s]['forged'][kind][name]['found']:.1f}" for s in MAP_SYSTEMS),
                ]
            )
    authentic_rows: list[list[Cell]] = [
        [
            name,
            int(counts["receipts"]),
            *(f"{found[s]['authentic'][name]['flagged']:.1f}" for s in MAP_SYSTEMS),
        ]
        for name, counts in first["authentic"].items()
    ]
    columns = [f"System {s}" for s in MAP_SYSTEMS]
    note = (
        "At the 10 % review budget over the pooled scores of the clean rounds, as Phase 3 "
        "reported its learned scorer (docs/clean-measurement.md, section 2); receipts tied at the "
        "cut-off count in part, so the numbers are expected values. A receipt with several "
        "techniques or entities counts in each. Descriptive: the groups are small. Systems 2 to 4: "
        "seed 0."
    )
    return [
        Table(
            "Forged receipts found at the 10 % review budget",
            ["Kind", "Group", "Receipts", *columns],
            forged_rows,
            note,
        ),
        Table(
            "Authentic receipts flagged at the 10 % review budget",
            ["Benign annotation", "Receipts", *columns],
            authentic_rows,
            note,
        ),
    ]


def size_table(data: CleanData) -> Table:
    """The size check of ADR-0015 on each clean fold's authentic receipts, descriptive only."""
    rows: list[list[Cell]] = []
    for system in MAP_SYSTEMS:
        found = pooled(data, system)
        for held_out in data.rounds:
            authentic = [
                r
                for r, k in zip(found.receipts, found.rounds, strict=True)
                if k == held_out and not r.forged
            ]
            shown = [r for r in authentic if evidence(r)]
            without = [r for r in authentic if not evidence(r)]
            rows.append(
                [
                    NAMES[system],
                    held_out,
                    len(authentic),
                    len(shown),
                    sum(page_scan(r.height, r.width) for r in without),
                    len(without) - sum(page_scan(r.height, r.width) for r in without),
                    _size(shown, found.score, data.resamples),
                    _size(authentic, found.score, data.resamples),
                ]
            )
    return Table(
        "Size check, descriptive",
        [
            "System",
            "Round",
            "Authentic",
            "With evidence",
            "Without evidence, page scans",
            "Without evidence, others",
            "Spearman, with evidence (95 % interval)",
            "Spearman, all",
        ],
        rows,
        "The rank correlation between document score and pixel count on each clean fold's "
        "authentic receipts (ADR-0015, section 6), on the receipts with evidence, whose map is not "
        "constant (ADR-0017), and on all of them. The folds are checked one by one, since each "
        "round of systems 2 to 4 has its own models (seed 0). No verdict is drawn: with no effect "
        f"at all, one of {len(data.rounds)} checks excludes zero with a probability of about "
        f"{100 * (1 - 0.95 ** len(data.rounds)):.0f} % (docs/clean-measurement.md, section 2). "
        "Page scans: a longer side above 6000 px.",
    )


def _size(receipts: Sequence[MeasuredReceipt], score: str, resamples: int) -> str | None:
    values = [r.scores[score] for r in receipts]
    sizes = [r.height * r.width for r in receipts]
    if len(receipts) < 3 or np.ptp(values) == 0 or np.ptp(sizes) == 0:  # noqa: PLR2004
        return None  # a rank correlation needs three receipts that differ
    estimate = size_dependence(values, sizes, resamples=resamples)
    return _signed(estimate)


def evidence_table(data: CleanData) -> Table:
    """On how many receipts each map holds evidence: the coverage system 1's AP is bounded by."""
    rows: list[list[Cell]] = []
    for system in MAP_SYSTEMS:
        receipts = pooled(data, system).receipts
        groups = (
            [r for r in receipts if r.forged],
            [r for r in receipts if not r.forged],
            [r for r in receipts if page_scan(r.height, r.width)],
        )
        counts = [f"{sum(evidence(r) for r in chosen)} of {len(chosen)}" for chosen in groups]
        rows.append([NAMES[system], *counts])
    return Table(
        "Receipts with evidence",
        ["System", "Forged", "Authentic", "Page scans"],
        rows,
        "A receipt has evidence when its map is not constant (ADR-0017). System 1 has none where "
        "no JPEG lattice can be measured, which bounds its average precision (ADR-0016). Systems "
        "2 to 4: seed 0.",
    )


def probe_table(data: CleanData) -> Table:
    """Whether the dataset carries a global trace (ADR-0018, decision 4)."""
    labels, scores, kept = _probe_arrays(data)
    rows: list[list[Cell]] = []
    for name, chosen in (
        ("all", np.ones(labels.size, dtype=bool)),
        ("without earlier contact", kept),
    ):
        y, s = labels[chosen], scores[chosen]
        ap = _metric(average_precision, y, s, data.resamples)
        share = float(y.mean())
        rows.append(
            [
                name,
                int(y.size),
                int(y.sum()),
                share,
                _interval(ap),
                roc_auc(y, s),
                ("carries a global trace" if ap.low > share else "no global trace found")
                if name == "all"
                else None,
            ]
        )
    return Table(
        "Global-trace probe",
        [
            "Receipts",
            "Count",
            "Forged",
            "Chance",
            "Average precision (95 % interval)",
            "ROC-AUC",
            "Verdict",
        ],
        rows,
        "System 5, a diagnostic and not a detector (ADR-0018): in each round a random forest of "
        "500 trees learns 28 global properties of the receipts of the other four folds and scores "
        "the held-out fold. The dataset carries a global trace if the lower bound of the pooled "
        "average precision's interval exceeds the share of forged receipts (decision 4); the "
        "check without the receipts of earlier contact decides nothing.",
    )


def property_table(data: CleanData) -> Table:
    """Each property of the probe on its own over the clean folds."""
    receipts = [r for held_out in data.rounds for r in data.probe[held_out].receipts]
    labels = np.array([r.forged for r in receipts], dtype=bool)
    rows: list[list[Cell]] = []
    for name in PROPERTIES:
        values = np.array([r.properties[name] for r in receipts], dtype=np.float64)
        defined = np.isfinite(values)
        both = labels[defined].any() and not labels[defined].all()
        rows.append(
            [
                name,
                int((~defined).sum()),
                roc_auc(labels[defined], values[defined]) if both else None,
            ]
        )
    return Table(
        "Properties of the probe on the clean folds",
        ["Property", "Empty", "ROC-AUC alone"],
        rows,
        "Each property used as a fraud score on the receipts where it is defined, so that a trace "
        "can be traced to its source (ADR-0018, decision 5); 0.5 means no signal, below 0.5 a "
        "signal in the other direction.",
    )


def agreement_table(data: CleanData) -> Table:
    """How each system's document score follows the probe's (ADR-0018, decision 5)."""
    probe = {r.key: r.score for k in data.rounds for r in data.probe[k].receipts}
    rows: list[list[Cell]] = []
    for system in MAP_SYSTEMS:
        found = pooled(data, system)
        rows.append([NAMES[system], spearman([probe[r.key] for r in found.receipts], found.scores)])
    return Table(
        "Document scores against the probe",
        ["System", "Spearman with the probe's score"],
        rows,
        "The rank correlation over every receipt of the clean folds. A system whose score follows "
        "the probe may use the same global trace, a warning next to the localisation guard. "
        "Systems 2 to 4: seed 0.",
    )


def time_table(data: CleanData) -> Table:
    """Time per receipt and peak memory on the reference machine (evaluation protocol, 5)."""
    rows: list[list[Cell]] = []
    for system in MAP_SYSTEMS:
        results = (
            [data.untrained[(k, system)] for k in data.rounds]
            if system in (LATTICE, TRUFOR)
            else [data.learned[key] for key in sorted(data.learned) if int(key[1]) == system]
        )
        seconds = [r.seconds for res in results for r in (*res.validation, *res.measured)]
        training = [
            sum(record.seconds) for key, record in data.records.items() if int(key[1]) == system
        ]
        peaks = [r.peak_memory for r in results if r.peak_memory is not None]
        rows.append(
            [
                NAMES[system],
                len(seconds),
                f"{float(np.median(seconds)):.2f}",
                f"{float(np.percentile(seconds, 95)):.2f}",
                f"{sum(training) / 3600:.1f}" if training else None,
                f"{max(peaks) / 2**30:.1f}" if peaks else None,
            ]
        )
    return Table(
        "Times and memory",
        [
            "System",
            "Receipts analysed",
            "Median seconds per receipt",
            "95th percentile",
            "Training hours",
            "Peak memory, GiB",
        ],
        rows,
        "On the reference machine. Seconds: the system on a whole receipt, validation included, "
        "over every round (and seed); TruFor's are those of the run that filled its cache. Peak "
        "memory: the most resident memory of the measuring process, as the operating system "
        "counts it, which may leave out what the Apple GPU allocates.",
    )


def _commits(codes: Iterable[CodeVersion]) -> str:
    """The commits that code ran from, each marked if the working tree differed from it."""
    named = sorted(
        {
            f"`{(code.commit or 'unknown')[:7]}`"
            + ("" if code.dirty is False else " with uncommitted changes")
            for code in codes
        }
    )
    return f"commit {', '.join(named)}" if len(named) == 1 else f"commits {', '.join(named)}"


def summary(data: CleanData) -> str:
    """The answers in a few sentences, each from the tables below."""
    lines = []
    for name, _, _, difference, _ in comparisons(data):
        claimed = (
            "claimed" if difference.excludes_zero else "not claimed: the interval includes zero"
        )
        lines.append(f"- {name}: average precision {_signed(difference)}, {claimed}.")
    for system in MAP_SYSTEMS:
        found = guard(localised(data, pooled(data, system).receipts), data.resamples)
        name = NAMES[system]
        lines.append(f"- {name[0].upper()}{name[1:]}: {_verdict(found)} by the localisation guard.")
    labels, scores, _ = _probe_arrays(data)
    ap = _metric(average_precision, labels, scores, data.resamples)
    trace = "carries a global trace" if ap.low > float(labels.mean()) else "shows no global trace"
    lines.append(f"- The global-trace probe: the dataset {trace} (ADR-0018).")
    return "\n".join(lines)


STATED = """\
Stated with the results, as the records require:

- Conventions that no data chose (ADR-0022): the learning rate and its schedule, the number of \
steps, the shares of tiles, the 50 % mix of system 4 and the factor of the failure rule. Another \
mix or length could change the answer to RQ5; no search for the best one was made.
- Limitations of systems 2 to 4 (ADR-0022): system 4 sees each real forgery half as often as \
system 3; rectangles teach the unchanged paper inside them as forged; large rectangles stay as \
annotated; on page scans a tile spans about 11 characters, against about 25 on other receipts; \
training on the Apple GPU is not repeatable bit for bit. Their document score was added after the \
development round's results (ADR-0023), and system 2 kept part of its page-scan effect at the \
scale of the text on the design fold.
- System 1 sees only receipts that kept a JPEG lattice, and no page scan (ADR-0016); its trace is \
the most exposed to re-encoding, which the robustness track of Phase 7 measures.
- TruFor (system 6) is for informational and nonprofit use; credit: {citation}. The ImageNet \
terms limit systems 2 to 4 to non-commercial research.
"""


def report(data: CleanData) -> str:
    """The report of the clean measurement, in Markdown."""
    trained = _commits(r.code for r in data.records.values())
    measured = _commits(
        [r.code for r in data.learned.values()]
        + [r.code for r in data.untrained.values()]
        + [r.code for r in data.probe.values()]
    )
    tables = [
        primary_table(data),
        comparison_table(data),
        guard_table(data),
        secondary_table(data),
        calibration_table(data),
        pixel_table(data),
        source_table(data),
        seed_table(data),
        blind_table(data),
        *breakdown_tables(data),
        evidence_table(data),
        size_table(data),
        probe_table(data),
        property_table(data),
        agreement_table(data),
        runs_table(data),
        untrained_table(data),
        time_table(data),
    ]
    forged = sum(d.label is Label.FORGED for d in data.documents.values())
    folds = ", ".join(map(str, data.rounds))
    several = len(data.rounds) > 1
    trial = (
        f"> **Trial.** {'Rounds' if several else 'Round'} {folds} {'are' if several else 'is'} not "
        "the clean measurement: this report only tries the commands, nothing in it is a result, "
        "and it is deleted after the trial.\n\n"
        if tuple(data.rounds) != CLEAN_FOLDS
        else ""
    )
    header = (
        "# Clean measurement: the Phase 4 systems on the clean folds\n\n"
        "> Generated by `experiments/clean_measurement.py` (`make clean-measurement`). Do not edit "
        "by hand. Not part of `make reports`: the systems run on the Apple GPU.\n\n"
        f"{trial}"
        "The single measurement of every Phase 4 system on the clean folds "
        "([ADR-0020](../../docs/decisions/0020-clean-folds-measured-together.md)), recorded in "
        "[clean-measurement.md](../../docs/clean-measurement.md): "
        f"{len(data.documents)} receipts of {'folds' if several else 'fold'} {folds}, {forged} of "
        "them forged. No result of "
        "a system was looked at before every system had run, and nothing below chose "
        "anything.\n\n"
        f"- Systems 2 to 4 trained at {trained}; every system measured at {measured}.\n\n"
        f"{summary(data)}\n\n"
        f"{STATED.format(citation=CITATION)}\n"
    )
    return header + "\n".join(render_table(table) for table in tables)
