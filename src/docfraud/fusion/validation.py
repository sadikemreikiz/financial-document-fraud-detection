"""Phase 5's check on the validation split, made once when the system is frozen (ADR-0029).

Every row of the ablation table (section 9) is the logistic regression of its inputs fitted on
the whole training split, on the out-of-fold inputs, and applied to the validation inputs, which
components trained on the whole training split made (section 2): standard stacking. Platt
scaling is then fitted per row on the validation labels (section 6). It keeps the order of the
scores, so average precision, ROC-AUC and recall at a budget are unbiased (section 1); F1 at the
threshold that maximises it there, Brier score and expected calibration error are measured on
the receipts they were fitted on, and labelled so (section 7 (c)). A row whose Platt fit would
reverse the order keeps the model's own probabilities: its ranking metrics stand, its
calibration metrics are not reported, and the report says so.

The comparisons of section 8 apply the rules of the claims, as a check. The review policy's
thresholds are set on the fusion model's calibrated probabilities without labels (section 11).
With every row's model, Platt scaling and max-F1 threshold, they are what Phase 7 applies
unchanged to the test split, so the results keep them. RQ3 and RQ4 are measured as on the
training split (:mod:`docfraud.fusion.development`), from the fusion model fitted on the whole
training split, and every receipt's audit report is checked for grounding (section 12).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from docfraud.data.models import BBox, Document, Label
from docfraud.decision import BUDGETS, ReviewPolicy, outcome
from docfraud.evaluation.bootstrap import DEFAULT_RESAMPLES, ratio
from docfraud.evaluation.detection import average_precision, choose_threshold
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
    metrics,
    rq3,
    rq4,
)
from docfraud.fusion.development_report import TRACK_NAMES
from docfraud.fusion.model import FusionModel, Platt, ReversedOrderError
from docfraud.fusion.training_inputs import InputsTable, ReceiptInputs
from docfraud.reporting.assessment import assess
from docfraud.reporting.grounding import check_grounding
from docfraud.reporting.text import render_report

type Labels = npt.NDArray[np.bool_]
type Vector = npt.NDArray[np.float64]

RUN_NAME = "fusion-validation-check"
FUSION_ROW = VARIANTS[-1]

IDENTITY = Platt(1.0, 0.0)
"""The map from log-odds to the model's own probabilities, for a row Platt scaling would
reverse."""


@dataclass(frozen=True, slots=True)
class Checked:
    """A row fitted on the whole training split and calibrated on the validation split."""

    model: FusionModel
    platt: Platt  # :data:`IDENTITY` if Platt scaling would reverse the order
    calibrated: bool
    log_odds: Vector  # of the validation receipts

    @property
    def probabilities(self) -> Vector:
        """The validation receipts' calibrated probabilities (the model's own if uncalibrated)."""
        return self.platt.probabilities(self.log_odds)


def check_row(
    training: Sequence[ReceiptInputs],
    training_labels: Labels,
    validation: Sequence[ReceiptInputs],
    validation_labels: Labels,
    variant: Variant,
) -> Checked:
    """Fit a row on the training split's out-of-fold inputs, then calibrate it on validation."""
    model = FusionModel.fit(matrix(training, variant), training_labels, variant.inputs)
    log_odds = model.log_odds(matrix(validation, variant))
    try:
        return Checked(model, Platt.fit(log_odds, validation_labels), True, log_odds)
    except ReversedOrderError:
        return Checked(model, IDENTITY, False, log_odds)


def fitted(checked: Checked) -> dict[str, Any]:
    """What Phase 7 applies to the test split: the model's parameters and its Platt scaling."""
    names = checked.model.names
    scaler = checked.model.scaler
    return {
        "coefficients": checked.model.coefficients,
        "intercept": checked.model.intercept,
        "means": dict(zip(names, map(float, scaler.mean_), strict=True)),
        "scales": dict(zip(names, map(float, scaler.scale_), strict=True)),
        "platt": {"slope": checked.platt.slope, "offset": checked.platt.offset},
        "calibrated": checked.calibrated,
    }


def row_metrics(y: Labels, checked: Checked, resamples: int = DEFAULT_RESAMPLES) -> dict[str, Any]:
    """The ablation table's columns on the validation split, with the max-F1 threshold."""
    found = metrics(y, checked.probabilities, resamples)
    found["threshold"] = choose_threshold(y, checked.probabilities)
    if not checked.calibrated:
        found["brier"] = found["ece"] = None
    return found


def check_track(  # noqa: PLR0913 - the rows of each seed on both splits and their labels
    training_by_seed: Mapping[int, Sequence[ReceiptInputs]],
    training_labels: Labels,
    validation_by_seed: Mapping[int, Sequence[ReceiptInputs]],
    validation_labels: Labels,
    *,
    variants: Sequence[Variant] = VARIANTS,
    headline_seed: int = 0,
    resamples: int = DEFAULT_RESAMPLES,
) -> tuple[dict[str, Any], dict[str, Checked]]:
    """The ablation table, the comparisons and the seed spread of a track on the validation split.

    Returns the results and every row at the headline seed.

    Raises:
        ValueError: If the two splits do not have the same seeds.
    """
    if set(training_by_seed) != set(validation_by_seed):
        msg = f"seeds {sorted(training_by_seed)} on training, {sorted(validation_by_seed)} here"
        raise ValueError(msg)
    y = validation_labels
    rows = (training_by_seed[headline_seed], training_labels, validation_by_seed[headline_seed], y)
    checked = {v.name: check_row(*rows, v) for v in variants}
    fusion = checked[FUSION_ROW.name].probabilities
    comparisons = {
        name: difference(y, fusion, checked[name].probabilities, resamples)
        for name in (*(v.name for v in SINGLE_SOURCES), "score-level baseline")
    }
    seeds = {
        str(seed): average_precision(
            y,
            check_row(
                training_by_seed[seed], training_labels, validation_by_seed[seed], y, FUSION_ROW
            ).log_odds,
        )
        for seed in sorted(training_by_seed)
    }
    return {
        "receipts": int(y.size),
        "forged": int(y.sum()),
        "table": {name: row_metrics(y, c, resamples) for name, c in checked.items()},
        "comparisons": comparisons,
        "fusion_beats_every_single_source": all(
            comparisons[v.name]["above_zero"] for v in SINGLE_SOURCES
        ),
        "seeds": seeds,
        "fitted": {name: fitted(c) for name, c in checked.items()},
    }, checked


def _estimate(numerator: Vector, denominator: Vector, resamples: int) -> dict[str, float]:
    e = ratio(numerator.tolist(), denominator.tolist(), resamples=resamples)
    return {"value": e.value, "low": e.low, "high": e.high}


def policy_results(
    y: Labels, probabilities: Vector, resamples: int = DEFAULT_RESAMPLES
) -> dict[str, Any]:
    """The review policy at each budget: its threshold and what it does (section 11).

    The intervals resample receipts with the threshold held fixed.
    """
    out: dict[str, Any] = {}
    for budget in BUDGETS:
        policy = ReviewPolicy.set_on(probabilities, budget)
        reviewed = policy.reviewed(probabilities).astype(float)
        found = (policy.reviewed(probabilities) & y).astype(float)
        counts = outcome(policy, probabilities, y)
        out[f"{budget * 100:.0f} %"] = {
            "budget": budget,
            "threshold": policy.threshold,
            "receipts": counts.receipts,
            "reviewed": counts.reviewed,
            "forged": counts.forged,
            "forged_reviewed": counts.forged_reviewed,
            "review_share": _estimate(reviewed, np.ones(y.size), resamples),
            "recall": _estimate(found, y.astype(float), resamples),
            "precision": _estimate(found, reviewed, resamples),
        }
    return out


def grounding(
    rows: Sequence[ReceiptInputs], checked: Checked, policy: ReviewPolicy, track: str
) -> dict[str, Any]:
    """The grounding rate of every receipt's audit report, with every violation listed."""
    facts = 0
    violations: list[str] = []
    for row in rows:
        assessment = assess(
            row.receipt,
            track,
            {name: row.inputs[name] for name in checked.model.names},
            row.findings,
            model=checked.model,
            platt=checked.platt,
            policy=policy,
        )
        found = check_grounding(render_report(assessment), assessment)
        facts += found.facts
        violations += [f"{row.receipt}: {v}" for v in found.violations]
    return {
        "reports": len(rows),
        "facts": facts,
        "violations": violations,
        "rate": 1 - len(violations) / facts if facts else 1.0,
    }


def validation_check(  # noqa: PLR0913 - both tables, the labels and what RQ4 places boxes on
    training: InputsTable,
    training_labels: Labels,
    validation: InputsTable,
    documents: Sequence[Document],
    pages: Mapping[str, tuple[BBox, Sequence[BBox]]],
    *,
    seeds: Sequence[int],
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Every result of the check, per track, from both tables and the validation receipts.

    ``documents`` are the validation receipts in the order of the validation table, and
    ``pages`` the page and OCR word boxes of those forged with region annotations (RQ4).

    Raises:
        ValueError: If a table's rows are not the receipts given, in order.
    """
    y = np.array([d.label is Label.FORGED for d in documents])
    receipts = [d.doc_id for d in documents]
    results: dict[str, Any] = {"rq3": {}, "false_alarms": {}, "rq4": {}, "grounding": {}}
    policies = {}
    for track, name in TRACK_NAMES.items():
        training_by_seed = {s: training.rows(track, s) for s in seeds}
        validation_by_seed = {s: validation.rows(track, s) for s in seeds}
        if any(len(rows) != training_labels.size for rows in training_by_seed.values()):
            msg = f"the training table's {track} rows do not match the training labels"
            raise ValueError(msg)
        if any([r.receipt for r in rows] != receipts for rows in validation_by_seed.values()):
            msg = f"the validation table's {track} rows are not the validation receipts"
            raise ValueError(msg)
        results[track], checked = check_track(
            training_by_seed,
            training_labels,
            validation_by_seed,
            y,
            headline_seed=seeds[0],
            resamples=resamples,
        )
        rows, fusion = validation_by_seed[seeds[0]], checked[FUSION_ROW.name]
        policies[track] = policy_results(y, fusion.probabilities, resamples)
        policy = ReviewPolicy.set_on(fusion.probabilities, BUDGETS[0])
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
    ocr_training, ocr_validation = training.rows("ocr", seeds[0]), validation.rows("ocr", seeds[0])
    practical = {
        v.name: check_row(ocr_training, training_labels, ocr_validation, y, v)
        for v in PRACTICAL_VARIANTS
    }
    results["T3"] = {
        "table": {"fusion of track T2": results["ocr"]["table"]["fusion"]}
        | {name: row_metrics(y, c, resamples) for name, c in practical.items()},
        "fitted": {name: fitted(c) for name, c in practical.items()},
    }
    results["policy"] = policies
    return results
