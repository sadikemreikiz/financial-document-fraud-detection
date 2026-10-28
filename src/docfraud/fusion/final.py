"""Phase 7: the frozen system applied once to the test split (ADR-0034, decisions 3 to 6).

Every row of the ablation table is the logistic regression that Phase 5's check fitted on the
whole training split (:mod:`docfraud.fusion.validation`); it is fitted again here, which is
deterministic, and must give the coefficients the check recorded. Its Platt scaling, its max-F1
threshold and the review policy's thresholds are those the check set on the validation split,
applied unchanged: nothing is fitted or chosen on the receipts measured. F1 at the frozen
threshold, the Brier score and the expected calibration error are therefore unbiased here, and
F1 gets an interval, as ADR-0013's success criterion needs.

The results take the form of the check's, so that one rehearsal can show that this code, applied
to the validation split with the check's record, gives back what the check recorded
(:func:`rehearsal_mismatches`). Every comparison of section 8 of ADR-0029, RQ3, the review
policy, RQ4 and the grounding of every audit report are computed as the check computed them.
"""

import math
from collections.abc import Callable, Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from docfraud.data.models import BBox, Document, Label
from docfraud.decision import BUDGETS, ReviewPolicy
from docfraud.evaluation.bootstrap import DEFAULT_RESAMPLES, Estimate, stratified
from docfraud.evaluation.detection import (
    Counts,
    average_precision,
    brier,
    expected_calibration_error,
    recall_at_budget,
    roc_auc,
)
from docfraud.fusion.development import (
    CATCHERS,
    PRACTICAL_VARIANTS,
    SINGLE_SOURCES,
    VARIANTS,
    Variant,
    difference,
    false_alarms,
    first_regions,
    forged_groups,
    group_contributions,
    matrix,
    rq3,
    rq4,
)
from docfraud.fusion.development_report import TRACK_NAMES
from docfraud.fusion.model import FusionModel, Platt
from docfraud.fusion.training_inputs import InputsTable, ReceiptInputs
from docfraud.fusion.validation import FUSION_ROW, Checked, grounding, policy_result

RUN_NAME = "fusion-test-check"
REHEARSAL_RUN = "fusion-check-rehearsal"
RESAVED_RUN = "fusion-resaved-check"

CHECK_CODE = (
    "src/docfraud/decision.py",
    "src/docfraud/evaluation",
    "src/docfraud/fusion",
    "src/docfraud/linking",
    "src/docfraud/reporting",
)
"""The code that turns the inputs and the check's record into the results."""

PUBLISHED_F1 = 0.284
"""ADR-0013: F1's lower bound must exceed the best published result, F1 28.39 on this split."""

CHANCE_RECALL = 0.10
"""ADR-0013: recall's lower bound at the 10 % budget must exceed what chance finds there."""

TOLERANCE = 1e-9
"""How far a refitted model may lie from the recorded one."""

type Labels = npt.NDArray[np.bool_]
type Vector = npt.NDArray[np.float64]

RANKING: dict[str, Callable[[Labels, Vector], float]] = {
    "average_precision": average_precision,
    "roc_auc": roc_auc,
    "recall_at_10pct": lambda y, p: recall_at_budget(y, p, 0.10),
    "recall_at_20pct": lambda y, p: recall_at_budget(y, p, 0.20),
}
"""The ranking columns of the ablation table, as :func:`~docfraud.fusion.development.metrics`."""


def _estimate(e: Estimate) -> dict[str, float]:
    return {"value": e.value, "low": e.low, "high": e.high}


def _on(metric: Callable[[Labels, Vector], float], y: Labels, p: Vector) -> Callable[[Any], float]:
    return lambda draw: metric(y[draw], p[draw])


def frozen_row(
    training: Sequence[ReceiptInputs],
    training_labels: Labels,
    rows: Sequence[ReceiptInputs],
    variant: Variant,
    recorded: Mapping[str, Any],
) -> Checked:
    """A row as the check left it: fitted on the training split, with its recorded Platt scaling.

    Raises:
        ValueError: If the model fitted again is not the one the check recorded.
    """
    model = FusionModel.fit(matrix(training, variant), training_labels, variant.inputs)
    names = model.names
    scaler = model.scaler
    found = {
        "coefficients": model.coefficients,
        "intercept": {"": model.intercept},
        "means": dict(zip(names, map(float, scaler.mean_), strict=True)),
        "scales": dict(zip(names, map(float, scaler.scale_), strict=True)),
    }
    stated = {
        "coefficients": recorded["coefficients"],
        "intercept": {"": recorded["intercept"]},
        "means": recorded["means"],
        "scales": recorded["scales"],
    }
    for part, values in found.items():
        if values.keys() != stated[part].keys() or any(
            not math.isclose(v, stated[part][k], rel_tol=0, abs_tol=TOLERANCE)
            for k, v in values.items()
        ):
            msg = f"the row {variant.name!r} fitted again is not the recorded one ({part})"
            raise ValueError(msg)
    platt = Platt(recorded["platt"]["slope"], recorded["platt"]["offset"])
    return Checked(model, platt, recorded["calibrated"], model.log_odds(matrix(rows, variant)))


def frozen_metrics(
    y: Labels, checked: Checked, threshold: float, resamples: int = DEFAULT_RESAMPLES
) -> dict[str, Any]:
    """The ablation table's columns, with F1 at the frozen threshold and its interval."""
    p = checked.probabilities
    found: dict[str, Any] = {
        name: _estimate(stratified(_on(m, y, p), y, resamples=resamples))
        for name, m in RANKING.items()
    }
    flagged = p >= threshold
    f1 = stratified(lambda d: Counts.of(y[d], flagged[d]).f1, y, resamples=resamples)
    found["f1"] = _estimate(f1)
    found["brier"] = brier(y, p) if checked.calibrated else None
    found["ece"] = expected_calibration_error(y, p) if checked.calibrated else None
    found["threshold"] = threshold
    return found


def success(table: Mapping[str, Any]) -> dict[str, Any]:
    """ADR-0013's success criterion for a row: both lower bounds above their references."""
    f1, recall = table["f1"], table["recall_at_10pct"]
    return {
        "f1": f1,
        "recall_at_10pct": recall,
        "f1_above_published": f1["low"] > PUBLISHED_F1,
        "recall_above_chance": recall["low"] > CHANCE_RECALL,
        "met": f1["low"] > PUBLISHED_F1 and recall["low"] > CHANCE_RECALL,
    }


def frozen_track(  # noqa: PLR0913 - the rows of each seed, their labels and the record
    training_by_seed: Mapping[int, Sequence[ReceiptInputs]],
    training_labels: Labels,
    rows_by_seed: Mapping[int, Sequence[ReceiptInputs]],
    y: Labels,
    recorded: Mapping[str, Any],
    *,
    headline_seed: int = 0,
    resamples: int = DEFAULT_RESAMPLES,
) -> tuple[dict[str, Any], dict[str, Checked]]:
    """The ablation table, the comparisons and the seed spread of a track, frozen.

    ``recorded`` is the track's results in the check's record. Returns the results and every
    row at the headline seed.
    """
    training = training_by_seed[headline_seed]
    rows = rows_by_seed[headline_seed]
    checked = {
        v.name: frozen_row(training, training_labels, rows, v, recorded["fitted"][v.name])
        for v in VARIANTS
    }
    fusion = checked[FUSION_ROW.name].probabilities
    comparisons = {
        name: difference(y, fusion, checked[name].probabilities, resamples)
        for name in (*(v.name for v in SINGLE_SOURCES), "score-level baseline")
    }
    seeds = {}
    for seed in sorted(rows_by_seed):
        model = FusionModel.fit(
            matrix(training_by_seed[seed], FUSION_ROW), training_labels, FUSION_ROW.inputs
        )
        seeds[str(seed)] = average_precision(
            y, model.log_odds(matrix(rows_by_seed[seed], FUSION_ROW))
        )
    table = {
        name: frozen_metrics(y, c, recorded["table"][name]["threshold"], resamples)
        for name, c in checked.items()
    }
    return {
        "receipts": int(y.size),
        "forged": int(y.sum()),
        "table": table,
        "comparisons": comparisons,
        "fusion_beats_every_single_source": all(
            comparisons[v.name]["above_zero"] for v in SINGLE_SOURCES
        ),
        "seeds": seeds,
        "success": success(table[FUSION_ROW.name]),
    }, checked


def frozen_check(  # noqa: PLR0913 - both tables, the labels, what RQ4 places boxes on, the record
    training: InputsTable,
    training_labels: Labels,
    measured: InputsTable,
    documents: Sequence[Document],
    pages: Mapping[str, tuple[BBox, Sequence[BBox]]],
    *,
    recorded: Mapping[str, Any],
    seeds: Sequence[int],
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Every result of the check, per track, on ``documents``, with the check's record applied.

    ``documents`` are the measured receipts in the order of ``measured`` and ``pages`` the page
    and OCR word boxes of those forged with region annotations (RQ4).

    Raises:
        ValueError: If a table's rows are not the receipts given, in order, or a row fitted
            again is not the recorded one.
    """
    y = np.array([d.label is Label.FORGED for d in documents])
    receipts = [d.doc_id for d in documents]
    results: dict[str, Any] = {"rq3": {}, "false_alarms": {}, "rq4": {}, "grounding": {}}
    policies: dict[str, Any] = {}
    for track, name in TRACK_NAMES.items():
        training_by_seed = {s: training.rows(track, s) for s in seeds}
        rows_by_seed = {s: measured.rows(track, s) for s in seeds}
        if any(len(rows) != training_labels.size for rows in training_by_seed.values()):
            msg = f"the training table's {track} rows do not match the training labels"
            raise ValueError(msg)
        if any([r.receipt for r in rows] != receipts for rows in rows_by_seed.values()):
            msg = f"the measured table's {track} rows are not the receipts given"
            raise ValueError(msg)
        results[track], checked = frozen_track(
            training_by_seed,
            training_labels,
            rows_by_seed,
            y,
            recorded[track],
            headline_seed=seeds[0],
            resamples=resamples,
        )
        rows, fusion = rows_by_seed[seeds[0]], checked[FUSION_ROW.name]
        stated = recorded["policy"][track]
        policies[track] = {
            key: policy_result(
                y,
                fusion.probabilities,
                ReviewPolicy(found["budget"], found["threshold"]),
                resamples,
            )
            for key, found in stated.items()
        }
        policy = ReviewPolicy(BUDGETS[0], stated[f"{BUDGETS[0] * 100:.0f} %"]["threshold"])
        models = [fusion.model] * len(rows)
        probabilities = {c: checked[c].probabilities for c in CATCHERS}
        results["rq3"][track] = rq3(
            group_contributions(rows, models), probabilities, receipts, forged_groups(documents)
        )
        results["false_alarms"][track] = false_alarms(documents, probabilities)
        results["rq4"][track] = rq4(
            documents,
            first_regions(rows, models),
            list(policy.reviewed(fusion.probabilities)),
            pages,
            resamples=resamples,
        )
        results["grounding"][track] = grounding(rows, fusion, policy, name)
    ocr_training, ocr_rows = training.rows("ocr", seeds[0]), measured.rows("ocr", seeds[0])
    t3 = recorded["T3"]
    practical = {
        v.name: frozen_row(ocr_training, training_labels, ocr_rows, v, t3["fitted"][v.name])
        for v in PRACTICAL_VARIANTS
    }
    results["T3"] = {
        "table": {"fusion of track T2": results["ocr"]["table"]["fusion"]}
        | {
            name: frozen_metrics(y, c, t3["table"][name]["threshold"], resamples)
            for name, c in practical.items()
        }
    }
    results["policy"] = policies
    return results


def subset(
    table: InputsTable, documents: Sequence[Document], left_out: frozenset[str]
) -> tuple[InputsTable, list[Document]]:
    """The table and the receipts without those left out, in the same order."""
    kept = tuple(r for r in table.receipts if r.receipt not in left_out)
    return InputsTable(receipts=kept), [d for d in documents if d.doc_id not in left_out]


def resaved_check(  # noqa: PLR0913 - both tables, the labels, the saved tables, the record
    training: InputsTable,
    training_labels: Labels,
    measured: InputsTable,
    resaved: Mapping[int, InputsTable],
    y: Labels,
    *,
    recorded: Mapping[str, Any],
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Track R: the frozen fusion model of T2 on the test receipts saved again (ADR-0034, 4).

    ``resaved`` holds, per JPEG quality, the inputs of the saved receipts at system 2's seed 0;
    ``measured`` the test inputs, whose T2 rows at seed 0 are the reference. ``recorded`` is the
    T2 results of the check's record. Each quality gets the ablation table's columns at the
    frozen threshold, the paired difference in average precision from the original images, and
    whether ADR-0013's two conditions would hold. Descriptive.

    Raises:
        ValueError: If a table's rows are not the same receipts in the same order.
    """
    training_rows = training.rows("ocr", 0)
    original_rows = measured.rows("ocr", 0)
    stated = recorded["fitted"][FUSION_ROW.name]
    threshold = recorded["table"][FUSION_ROW.name]["threshold"]
    original = frozen_row(training_rows, training_labels, original_rows, FUSION_ROW, stated)
    found: dict[str, Any] = {}
    for quality, table in resaved.items():
        rows = table.rows("ocr", 0)
        if [r.receipt for r in rows] != [r.receipt for r in original_rows]:
            msg = f"the receipts saved at quality {quality} are not the test receipts"
            raise ValueError(msg)
        saved = frozen_row(training_rows, training_labels, rows, FUSION_ROW, stated)
        metrics = frozen_metrics(y, saved, threshold, resamples)
        found[str(quality)] = {
            "table": metrics,
            "against_original": difference(
                y, original.probabilities, saved.probabilities, resamples
            ),
            "success": success(metrics),
        }
    found["original"] = frozen_metrics(y, original, threshold, resamples)
    return found


def _same(recorded: Any, rebuilt: Any, path: str, out: list[str]) -> None:
    """Add to ``out`` every path where the rebuilt value differs from the recorded one."""
    if isinstance(recorded, Mapping) and isinstance(rebuilt, Mapping):
        old = {str(k): v for k, v in recorded.items()}  # a record's keys are text
        new = {str(k): v for k, v in rebuilt.items()}
        for key in sorted(old.keys() | new.keys()):
            if key not in old or key not in new:
                out.append(f"{path}/{key}")
            else:
                _same(old[key], new[key], f"{path}/{key}", out)
    elif isinstance(recorded, list | tuple) and isinstance(rebuilt, list | tuple):
        if len(recorded) != len(rebuilt):
            out.append(path)
        else:
            for i, (a, b) in enumerate(zip(recorded, rebuilt, strict=True)):
                _same(a, b, f"{path}/{i}", out)
    elif recorded != rebuilt:
        out.append(path)


def rehearsal_mismatches(recorded: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> list[str]:
    """Where :func:`frozen_check` on the validation split differs from the check's record.

    Everything the check recorded must come back exactly, except what only the frozen
    application adds or changes in form: F1 gains an interval, whose value must equal the
    recorded F1; and each track gains the success criterion. The check's fitted models are not
    results here; :func:`frozen_row` already compared them.
    """
    out: list[str] = []
    for key, value in recorded.items():
        if key in TRACK_NAMES:
            for part, found in value.items():
                if part == "fitted":
                    continue
                if part == "table":
                    _same(_flat_f1(found), _flat_f1(rebuilt[key]["table"]), f"/{key}/table", out)
                else:
                    _same(found, rebuilt[key].get(part), f"/{key}/{part}", out)
        elif key == "T3":
            _same(_flat_f1(value["table"]), _flat_f1(rebuilt["T3"]["table"]), "/T3/table", out)
        else:
            _same(value, rebuilt.get(key), f"/{key}", out)
    return out


def _flat_f1(table: Mapping[str, Any]) -> dict[str, Any]:
    """A table with each row's F1 as its value alone, as the check recorded it."""
    return {
        row: dict(found) | {"f1": f["value"] if isinstance(f := found["f1"], Mapping) else f}
        for row, found in table.items()
    }
