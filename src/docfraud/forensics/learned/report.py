"""The development report of systems 2 to 4 (ADR-0022, decision 5).

Every table is made from what the round kept (:class:`RoundData`): the runs' records, their
results and the repeat check, with the design fold's OCR words, benign marks and pages. Nothing
here runs a network. The tables are those ADR-0022 lists, in its order, and one fixed before the
round ran: whether the top region of an authentic receipt lies on blank paper, where TruFor's
highest score lay on every page scan. Every number comes from the design fold or the synthetic
development set and is optimistic.

The document score follows the fixed order of decision 1 (:func:`chosen_score`): the first
candidate whose size check passes for the seed-0 run of every system.
"""

from collections.abc import Callable, Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from docfraud.data.benign_marks import Mark
from docfraud.data.models import BBox, Label, Technique
from docfraud.evaluation.bootstrap import DEFAULT_RESAMPLES, Estimate
from docfraud.evaluation.detection import average_precision
from docfraud.evaluation.localisation import (
    LARGE_RECTANGLE_SHARE,
    baseline_hit_rate,
    hits,
    localisation_guard,
    receipt_rng,
    size_dependence,
)
from docfraud.evaluation.mark_alarms import check_marks
from docfraud.evaluation.runs import CodeVersion
from docfraud.forensics.learned.inference import HIGHEST, SCORE_ORDER
from docfraud.forensics.learned.results import Group, MeasuredReceipt, RunResults
from docfraud.forensics.learned.rounds import Repeatability
from docfraud.forensics.learned.sampler import System
from docfraud.forensics.learned.training import RunRecord, binary_entropy, failure
from docfraud.markdown import Cell, Table, render_table

WINDOW = 500
"""Steps over which a learning curve averages the loss, as the failure rule does."""

LARGE_SIDE = 6000
"""Pixels of the longer side above which a receipt is a page scan."""

SIZES = ("one pass", "tiled", "page scan")

NEAR_CUT = 8
"""Pixels from a cut within which a peak lies at the cut: one JPEG block."""

INK = 128
"""Grey value below which a pixel is print, as in system 6's report."""

BLANK_SHARE = 0.001
"""Share of print below which a region is blank paper, as in system 6's report."""

type RunKey = tuple[System, int]


@dataclass(frozen=True, slots=True)
class RoundData:
    """What the report of a round is made from."""

    records: Mapping[RunKey, RunRecord]
    results: Mapping[RunKey, RunResults]
    repeatability: Repeatability
    words: Mapping[str, Sequence[BBox]]  # the OCR word boxes of every design-fold receipt
    flagged: Mapping[str, Label]  # receipts the dataset marks as annotated, with their labels
    marks: Mapping[str, Sequence[Mark]]  # the final benign marks of the flagged receipts
    ink: Callable[[str, BBox], float]  # the share of print in a box of a design-fold page
    resamples: int = DEFAULT_RESAMPLES  # of every bootstrap interval

    @property
    def runs(self) -> list[RunKey]:
        """The runs, system by system and seed by seed."""
        return sorted(self.results)


def ink_share(page: npt.NDArray[np.uint8], box: BBox) -> float:
    """The share of a box's pixels that are darker than ``INK``."""
    return float((page[box.y : box.y2, box.x : box.x2] < INK).mean())


def size_class(receipt: MeasuredReceipt) -> str:
    """One pass, tiled, or a page scan, which is always tiled."""
    if max(receipt.height, receipt.width) > LARGE_SIDE:
        return "page scan"
    return "one pass" if receipt.windows == 1 else "tiled"


def _group(results: RunResults, group: Group) -> list[MeasuredReceipt]:
    return [r for r in results.measured if r.group is group]


def evidence(receipt: MeasuredReceipt) -> bool:
    """Whether the map holds more than one value (ADR-0017)."""
    return receipt.scores[HIGHEST] > receipt.lowest


def size_check(results: RunResults, score: str, resamples: int = DEFAULT_RESAMPLES) -> Estimate:
    """ADR-0015 (section 6) on the held-out fold's authentic receipts with evidence."""
    authentic = [r for r in _group(results, Group.HELD_OUT) if not r.forged and evidence(r)]
    return size_dependence(
        [r.scores[score] for r in authentic],
        [r.height * r.width for r in authentic],
        resamples=resamples,
    )


def passes(estimate: Estimate) -> bool:
    """Whether the interval of a size check includes zero."""
    return not estimate.excludes_zero


def chosen_score(
    results: Mapping[RunKey, RunResults], resamples: int = DEFAULT_RESAMPLES
) -> str | None:
    """The first candidate whose size check passes for seed 0 of every system, if one does.

    Raises:
        KeyError: If the seed-0 run of a system is missing.
    """
    seed_zero = [results[(system, 0)] for system in System]
    return next(
        (
            name
            for name in SCORE_ORDER
            if all(passes(size_check(r, name, resamples)) for r in seed_zero)
        ),
        None,
    )


def _label(key: RunKey) -> str:
    system, seed = key
    return f"system {int(system)}, seed {seed}"


def _mean(values: Iterable[float]) -> float | None:
    chosen = list(values)
    return float(np.mean(chosen)) if chosen else None


def _with_means(
    data: RoundData, row: Callable[[RunKey], list[Cell]], averaged: Sequence[int]
) -> list[list[Cell]]:
    """A row per run and, after each system's seeds, their mean in the ``averaged`` columns."""
    rows: list[list[Cell]] = []
    for system in System:
        keys = [key for key in data.runs if key[0] is system]
        own = [row(key) for key in keys]
        rows += own
        if len(own) > 1:
            mean: list[Cell] = [f"system {int(system)}, mean of {len(own)} seeds"]
            for column in range(1, len(own[0])):
                values = [r[column] for r in own]
                numbers = [float(v) for v in values if isinstance(v, float)]
                usable = column in averaged and len(numbers) == len(values)
                mean.append(float(np.mean(numbers)) if usable else None)
            rows.append(mean)
    return rows


def runs_table(data: RoundData) -> Table:
    """How each run trained and how it fared under the failure rule of decision 5."""
    rows: list[list[Cell]] = []
    for key in data.runs:
        record = data.records[key]
        window = int(record.settings["failure_window"])
        rows.append(
            [
                _label(key),
                record.status,
                record.steps_done,
                f"{100 * record.prior:.2f} %",
                f"{float(np.median(record.seconds)):.2f}",
                f"{sum(record.seconds) / 3600:.2f}",
                float(np.mean(record.losses[-window:])),
                binary_entropy(float(np.mean(record.forged_shares[-window:]))),
                failure(record) or "learned",
                f"`{data.results[key].model}`",
            ]
        )
    return Table(
        "Runs",
        [
            "Run",
            "Status",
            "Steps",
            "Initial forged share",
            "Median seconds per step",
            "Hours",
            f"Mean loss, last {WINDOW} steps",
            "Loss of the best constant",
            "Failure rule",
            "Network",
        ],
        rows,
        "Failure rule of ADR-0022 (decision 5): a run diverges if its loss is ever not finite and "
        "does not learn if its mean loss over the last 500 steps is not below that of the best "
        "constant prediction on the same tiles. Initial forged share: in the run's first 100 "
        "steps, which sets the initial output bias. Seconds include the wait for each batch.",
    )


def curve_table(data: RoundData) -> Table:
    """The mean loss of every window of steps over that of the best constant prediction."""
    steps = max(record.steps_done for record in data.records.values())
    starts = range(0, steps, WINDOW)
    rows: list[list[Cell]] = []
    for key in data.runs:
        record = data.records[key]
        cells: list[Cell] = [_label(key)]
        for start in starts:
            losses = record.losses[start : start + WINDOW]
            shares = record.forged_shares[start : start + WINDOW]
            constant = binary_entropy(float(np.mean(shares))) if shares else 0.0
            cells.append(float(np.mean(losses)) / constant if constant else None)
        rows.append(cells)
    return Table(
        "Learning curves",
        ["Run", *(f"steps {start + 1}-{start + WINDOW}" for start in starts)],
        rows,
        "The mean loss of each window of steps over the loss of the best constant prediction on "
        "the same tiles, the share of forged pixels among them. The loss follows the share of "
        "forged pixels in each batch, which the ratio removes; below 1, the network predicts "
        "better than that share alone.",
    )


def repeat_table(repeat: Repeatability) -> Table:
    """Whether the first steps of one run, trained twice, agree."""
    return Table(
        "Repeatability",
        [
            "Run",
            "Steps",
            "Device",
            "Tensors",
            "Tensors that differ",
            "Largest weight difference",
            "Largest loss difference",
        ],
        [
            [
                f"system {repeat.system}, seed {repeat.seed}",
                repeat.steps,
                repeat.device,
                repeat.tensors,
                len(repeat.differing),
                f"{repeat.largest_difference:.2e}",
                f"{repeat.largest_loss_difference:.2e}",
            ]
        ],
        "The first steps of one run, trained twice from scratch before the round (ADR-0022, "
        "decision 4); PyTorch cannot make training on the Apple GPU repeatable bit for bit, so "
        "seeds 1 and 2 show how much a run varies beyond this.",
    )


def _interval(estimate: Estimate) -> str:
    return f"{estimate.value:+.2f} ({estimate.low:+.2f} to {estimate.high:+.2f})"


def score_table(data: RoundData) -> Table:
    """Both candidate document scores of every run: average precision and the size check."""
    chance = 0.0

    def row(key: RunKey) -> list[Cell]:
        nonlocal chance
        receipts = _group(data.results[key], Group.HELD_OUT)
        labels = np.array([r.forged for r in receipts])
        chance = float(labels.mean())
        cells: list[Cell] = [_label(key)]
        for name in SCORE_ORDER:
            check = size_check(data.results[key], name, data.resamples)
            scores = np.array([r.scores[name] for r in receipts])
            cells += [average_precision(labels, scores), _interval(check), passes(check)]
        return cells

    rows = _with_means(data, row, averaged=(1, 4))
    columns = ["Run"]
    for position, name in enumerate(SCORE_ORDER, start=1):
        columns += [f"{position}. AP, {name}", f"{position}. Size check", f"{position}. Passes"]
    return Table(
        "Document scores",
        columns,
        rows,
        f"Average precision over the design fold (chance {chance:.3f}). Size check: the Spearman "
        "rank correlation of document score and pixel count over the authentic receipts with "
        "evidence, with its 95 % interval (bootstrap over receipts, 10,000 resamples, seed 0; "
        "ADR-0015, section 6, and ADR-0017); it passes if the interval includes zero. By the "
        "fixed order of ADR-0022 (decision 1), the seed-0 runs decide for all three systems. No "
        "choice depends on average precision. Optimistic.",
    )


def _page(receipt: MeasuredReceipt) -> BBox:
    return BBox(x=0, y=0, width=receipt.width, height=receipt.height)


def large(receipt: MeasuredReceipt) -> bool:
    """Whether a modified rectangle covers more than 5 % of the page (ADR-0015, section 4)."""
    area = receipt.width * receipt.height
    return any(box.area > LARGE_RECTANGLE_SHARE * area for box in receipt.modified)


def guard_table(data: RoundData, group: Group) -> Table:
    """The localisation guard of every run, on all forged receipts and without large ones."""

    def row(key: RunKey) -> list[Cell]:
        forged = [r for r in _group(data.results[key], group) if r.modified]
        found = {r.key: hits(r.top_region, [*r.modified, *r.sources]) for r in forged}
        baselines = {
            r.key: baseline_hit_rate(
                r.top_region,
                _page(r),
                data.words[r.key],
                [*r.modified, *r.sources],
                receipt_rng(r.key),
            )
            for r in forged
        }
        cells: list[Cell] = [_label(key)]
        for chosen in (forged, [r for r in forged if not large(r)]):
            guard = localisation_guard(
                [found[r.key] for r in chosen],
                [baselines[r.key] for r in chosen],
                resamples=data.resamples,
            )
            cells += [
                len(chosen),
                sum(found[r.key] for r in chosen),
                f"{guard.hit_rate.value:.3f} ({guard.hit_rate.low:.3f} to "
                f"{guard.hit_rate.high:.3f})",
                guard.baseline,
                guard.detects,
            ]
        return cells

    rows = [row(key) for key in data.runs]
    columns = ["Run"]
    for name in ("all", "without large rectangles"):
        columns += [
            f"Forged, {name}",
            "Hits",
            "Hit rate (95 % interval)",
            "Random baseline",
            "Detects",
        ]
    real = group is Group.HELD_OUT
    return Table(
        "Top region on forged receipts" if real else "Top region on synthetic forgeries",
        columns,
        rows,
        "Top region, hit, random baseline and guard as ADR-0014 and ADR-0015 fix them: the hit "
        "counts a modified rectangle or a copy source on the page; the baseline is the share of "
        "1,000 boxes of the same size, centred on OCR text, that hit; a system detects forgeries "
        "if the lower bound of its hit rate's interval (bootstrap over receipts) exceeds the "
        "baseline. Large rectangles cover more than 5 % of the page (ADR-0015, section 4)."
        + ("" if real else " A synthetic forgery's OCR words are those of its base.")
        + " Optimistic.",
    )


def pixel_table(data: RoundData) -> Table:
    """Pixel metrics on the real and the synthetic forgeries (ADR-0015, section 5)."""

    def row(key: RunKey) -> list[Cell]:
        results = data.results[key]
        cells: list[Cell] = [_label(key), results.threshold.value]
        for group in (Group.HELD_OUT, Group.SYNTHETIC):
            pixels = [r.pixels for r in _group(results, group) if r.pixels is not None]
            cells += [
                len(pixels),
                _mean(p.average_precision for p in pixels),
                _mean(p.chance for p in pixels),
                _mean(p.f1 for p in pixels),
                _mean(p.iou for p in pixels),
            ]
        return cells

    rows = _with_means(data, row, averaged=(1, 3, 4, 5, 6, 8, 9, 10, 11))
    columns = ["Run", "Pixel threshold"]
    for name in ("real", "synthetic"):
        columns += [f"Forged, {name}", "Pixel AP", "Chance", "F1", "IoU"]
    return Table(
        "Pixel metrics",
        columns,
        rows,
        "Per forged receipt, then averaged over receipts with equal weight; pixels of a copy "
        "source outside every modified rectangle are ignored (ADR-0015, section 5). Chance: the "
        "mean share of modified pixels. F1 and IoU at the run's threshold, chosen on the forged "
        "receipts of the validation split, whose own results are not shown. Optimistic.",
    )


def synthetic_score_table(data: RoundData, score: str) -> Table:
    """How the chosen document score ranks synthetic forgeries among the authentic receipts."""
    chance = 0.0

    def row(key: RunKey) -> list[Cell]:
        nonlocal chance
        results = data.results[key]
        authentic = [r for r in _group(results, Group.HELD_OUT) if not r.forged]
        forged = _group(results, Group.SYNTHETIC)
        labels = np.array([False] * len(authentic) + [True] * len(forged))
        chance = float(labels.mean())
        scores = np.array([r.scores[score] for r in (*authentic, *forged)])
        return [_label(key), len(forged), average_precision(labels, scores)]

    rows = _with_means(data, row, averaged=(2,))
    return Table(
        "Document score on synthetic forgeries",
        ["Run", "Synthetic forgeries", f"AP, {score}"],
        rows,
        f"The synthetic development set against the design fold's authentic receipts, which are "
        f"its bases (chance {chance:.3f}). Optimistic.",
    )


def technique_table(data: RoundData) -> Table:
    """For every technique and system: how often an edit raises the map, and its pixel AP."""
    rows: list[list[Cell]] = []
    first = data.results[data.runs[0]]
    for technique in Technique:
        edits = [
            e for r in _group(first, Group.SYNTHETIC) for e in r.edits if e.technique is technique
        ]
        if not edits:
            continue
        cells: list[Cell] = [technique.value, len(edits)]
        for system in System:
            seen: list[bool] = []
            precision: list[float] = []
            for key in (k for k in data.runs if k[0] is system):
                results = data.results[key]
                before = {r.key: r.scores[HIGHEST] for r in _group(results, Group.HELD_OUT)}
                for receipt in _group(results, Group.SYNTHETIC):
                    for edit in receipt.edits:
                        if edit.technique is technique:
                            seen.append(edit.peak > before[receipt.key])
                            precision.append(edit.average_precision)
            cells += [
                f"{100 * float(np.mean(seen)):.0f} %" if seen else None,
                _mean(precision),
            ]
        cells.append(_mean(e.chance for e in edits))
        rows.append(cells)
    columns = ["Technique", "Edits"]
    for system in System:
        columns += [f"System {int(system)}, seen", f"System {int(system)}, edit AP"]
    columns.append("Chance of the edit AP")
    return Table(
        "Synthetic edits by technique",
        columns,
        rows,
        "An edit is seen if the highest map value inside its rectangle exceeds the highest value "
        "of the same receipt without the edit, measured by the same network. Edit AP: the edit's "
        "pixels against the clean pixels of its receipt, the other edits ignored. Pooled over the "
        "seeds of each system. The 3 CPO edits are too few to read. Optimistic.",
    )


def mark_table(data: RoundData, score: str) -> Table:
    """For every run: how often the top region of a flagged receipt lies on a benign mark."""
    rows: list[list[Cell]] = []
    for key in data.runs:
        receipts = {r.key: r for r in _group(data.results[key], Group.HELD_OUT)}
        checks = [
            check_marks(
                doc_id,
                label,
                receipts[doc_id].top_region,
                _page(receipts[doc_id]),
                data.words[doc_id],
                data.marks.get(doc_id, ()),
                evidence=evidence(receipts[doc_id]),
                score=receipts[doc_id].scores[score],
            )
            for doc_id, label in sorted(data.flagged.items())
        ]
        cells: list[Cell] = [_label(key)]
        for label in (Label.AUTHENTIC, Label.FORGED):
            chosen = [c for c in checks if c.label is label]
            cells += [
                len(chosen),
                sum(c.on_mark for c in chosen),
                _mean(c.baseline for c in chosen),
            ]
        rows.append(cells)
    return Table(
        "Top region on a benign mark",
        [
            "Run",
            "Flagged authentic",
            "On a mark",
            "Random baseline",
            "Flagged forged",
            "On a mark",
            "Random baseline",
        ],
        rows,
        "Flagged: the dataset marks a handwritten or digital annotation. On a mark: the top "
        "region shares a pixel with a final box of the benign marks guide. Baseline: the mean "
        "share of 1,000 boxes of the same size, centred on OCR text, that overlap a mark. Design "
        "fold.",
    )


def blank_table(data: RoundData) -> Table:
    """Whether the top region of an authentic receipt lies on paper without print."""
    rows: list[list[Cell]] = []
    for key in data.runs:
        authentic = [r for r in _group(data.results[key], Group.HELD_OUT) if not r.forged]
        cells: list[Cell] = [_label(key)]
        for size in SIZES:
            chosen = [r for r in authentic if size_class(r) == size]
            blank = sum(data.ink(r.key, r.top_region) < BLANK_SHARE for r in chosen)
            cells.append(f"{blank} of {len(chosen)}")
        rows.append(cells)
    return Table(
        "Top region on blank paper",
        ["Run", *(f"Authentic, {size}" for size in SIZES)],
        rows,
        f"Blank: fewer than {100 * BLANK_SHARE:.1f} % of the region's pixels are darker than "
        f"{INK}, the rule of system 6's report, in which TruFor's highest-scoring tile was blank "
        "on every page scan of the design fold; ADR-0022 adds tiles drawn anywhere on authentic "
        "receipts for that reason. Fixed before the round ran. Design fold.",
    )


def cut_table(data: RoundData) -> Table:
    """The maps of receipts in several windows, at the cuts between them."""
    rows: list[list[Cell]] = []
    for key in data.runs:
        cuts = [r.cuts for r in _group(data.results[key], Group.HELD_OUT) if r.cuts is not None]
        ratios = [ratio for _, ratio in cuts]
        rows.append(
            [
                _label(key),
                len(cuts),
                sum(distance <= NEAR_CUT for distance, _ in cuts),
                sum(ratio > 1 for ratio in ratios),
                float(np.median(ratios)) if ratios else None,
                max(ratios) if ratios else None,
            ]
        )
    return Table(
        "The cuts between windows",
        [
            "Run",
            "Receipts in several windows",
            f"Peak within {NEAR_CUT} px of a cut",
            "Jumps larger across cuts than inside windows",
            "Median ratio",
            "Largest ratio",
        ],
        rows,
        "A cut lies between two neighbouring pixels that come from different windows. Ratio: the "
        "99th percentile of the jumps between neighbouring pixels across the cuts over that "
        "inside the windows. Each window gives the network another context, so some jump is "
        "expected; a defect of the joining would put values in the wrong place, which the tests "
        "exclude. Design fold.",
    )


def time_table(data: RoundData) -> Table:
    """Training and inference times on the reference machine."""
    rows: list[list[Cell]] = []
    for key in data.runs:
        results = data.results[key]
        seconds: dict[str, list[float]] = {size: [] for size in SIZES}
        for receipt in results.measured:
            seconds[size_class(receipt)].append(receipt.seconds)
        total = sum(r.seconds for r in results.validation) + sum(
            r.seconds for r in results.measured
        )
        rows.append(
            [
                _label(key),
                f"{sum(data.records[key].seconds) / 3600:.2f}",
                *(
                    f"{float(np.median(seconds[size])):.2f}" if seconds[size] else None
                    for size in SIZES
                ),
                f"{total / 60:.1f}",
            ]
        )
    return Table(
        "Times",
        [
            "Run",
            "Training hours",
            *(f"Median seconds, {size}" for size in SIZES),
            "Network minutes, all receipts",
        ],
        rows,
        "On the reference machine. Seconds: the network on a whole receipt of the design fold or "
        "the synthetic set; network minutes add validation. Measuring the maps takes more time "
        "on top.",
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


def report(data: RoundData) -> str:
    """The development report of a round, in Markdown."""
    score = chosen_score(data.results, data.resamples)
    trained = _commits(r.code for r in data.records.values())
    measured = _commits(r.code for r in data.results.values())
    outcome = (
        f"By the fixed order of ADR-0022 (decision 1), the document score of systems 2 to 4 is "
        f"the **{score}**."
        if score
        else "Neither candidate document score passes its size check: by ADR-0022 (decision 1), "
        "development stops and the next step is decided in a new record."
    )
    tables = [
        runs_table(data),
        curve_table(data),
        repeat_table(data.repeatability),
        score_table(data),
        guard_table(data, Group.HELD_OUT),
        pixel_table(data),
        *([synthetic_score_table(data, score)] if score else []),
        guard_table(data, Group.SYNTHETIC),
        technique_table(data),
        mark_table(data, score or SCORE_ORDER[0]),
        blank_table(data),
        cut_table(data),
        time_table(data),
    ]
    header = (
        "# Learned models development: design fold\n\n"
        "> Generated by `experiments/learned_development.py` (`make learned`). Do not edit by "
        "hand. Not part of `make reports`: the networks run on the Apple GPU.\n\n"
        "The development round of systems 2 to 4 "
        "([ADR-0022](../../docs/decisions/0022-learned-models.md)), recorded in "
        "[learned-models-development.md](../../docs/learned-models-development.md). Every number "
        "comes from the design fold or the synthetic development set and is optimistic.\n\n"
        f"- Trained at {trained}; measured at {measured}.\n\n"
        f"{outcome}\n\n"
    )
    return header + "\n".join(render_table(table) for table in tables)
