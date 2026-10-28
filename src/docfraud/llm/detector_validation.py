"""The frozen detector's check on the validation split, made once (ADR-0031, sections 3 and 4).

Every row runs the frozen instruction on the validation receipts: the main row (image and OCR
text) and the text row (transcript) on the frozen model, Claude Sonnet 5.5; the main row on
Claude Haiku 5.5; and, if the rule of ADR-0032 added it, the main row on Claude Opus 5.5. A
receipt whose answer failed the form ranks below every scored one (ADR-0031, section 2).

Platt scaling is fitted per row on the validation labels, on the score itself. It keeps the
order of the scores, so average precision, ROC-AUC and recall at a budget are unbiased; F1 at the
threshold that maximises it there, Brier score and expected calibration error are measured on
the receipts they were fitted on, and labelled so, as for fusion (ADR-0029, section 7 (c)). A
row whose Platt fit would reverse the order keeps the model's own score, read as a percentage,
and has no calibration metrics. Each row's Platt scaling and threshold are what Phase 7 applies
to the test split.

Each row is set against the frozen fusion model of its track (T2 for the main rows, T1 for the
text row) by the paired bootstrap of average precision. All of it is descriptive except the main
row on the frozen model against fusion of track T2: RQ6's pre-named comparison, applied here as a
check (ADR-0031, section 3). Fusion's probabilities are rebuilt from what Phase 5's check
recorded, and must give the average precision it recorded.
"""

import math
from collections import Counter
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from docfraud.evaluation.bootstrap import DEFAULT_RESAMPLES
from docfraud.evaluation.detection import average_precision, choose_threshold
from docfraud.fusion.development import difference, metrics
from docfraud.fusion.development_report import TRACK_NAMES
from docfraud.fusion.model import Platt, ReversedOrderError
from docfraud.fusion.training_inputs import InputsTable
from docfraud.fusion.validation import FUSION_ROW, applied
from docfraud.llm.detector import HAIKU, DetectorRow, Reading
from docfraud.llm.detector_rounds import OPUS, scores
from docfraud.llm.provider import Settings

type Labels = npt.NDArray[np.bool_]
type Vector = npt.NDArray[np.float64]

RUN_NAME = "llm-detector-validation"

FUSION_TRACK = {DetectorRow.IMAGE_AND_OCR: "ocr", DetectorRow.TRANSCRIPT: "transcript"}
"""The fusion track each input is set against: the same text source."""


@dataclass(frozen=True, slots=True)
class ValidationRow:
    """One row of the detector on the validation split: an input and a model."""

    row: DetectorRow
    settings: Settings

    @property
    def name(self) -> str:
        """How the record and the report name the row."""
        return f"{self.row.value}, {self.settings.model}"

    @property
    def track(self) -> str:
        """The fusion track the row is set against."""
        return FUSION_TRACK[self.row]


def validation_rows(frozen: Settings, *, stronger: bool) -> tuple[ValidationRow, ...]:
    """The rows in the report's order; the first is RQ6's, the frozen detector's main row.

    ``stronger`` is whether ADR-0032's rule added the row of Claude Opus 5.5.
    """
    rows = [
        ValidationRow(DetectorRow.IMAGE_AND_OCR, frozen),
        ValidationRow(DetectorRow.TRANSCRIPT, frozen),
        ValidationRow(DetectorRow.IMAGE_AND_OCR, HAIKU),
    ]
    if stronger:
        rows.append(ValidationRow(DetectorRow.IMAGE_AND_OCR, OPUS))
    return tuple(rows)


@dataclass(frozen=True, slots=True)
class Calibrated:
    """A row's scores on the validation receipts and their Platt scaling.

    ``platt`` is ``None`` if Platt scaling would reverse the order of the scores.
    """

    scores: Vector
    platt: Platt | None

    @property
    def probabilities(self) -> Vector:
        """Calibrated probabilities, or the model's own score read as a percentage."""
        if self.platt is None:
            return np.clip(self.scores, 0, 100) / 100
        return self.platt.probabilities(self.scores)


def calibrate(row_scores: Vector, y: Labels) -> Calibrated:
    """Fit Platt scaling on the scores of these receipts (the validation split)."""
    try:
        return Calibrated(row_scores, Platt.fit(row_scores, y))
    except ReversedOrderError:
        return Calibrated(row_scores, None)


def rq6_answer(difference_of_ap: Mapping[str, float]) -> str:
    """RQ6's answer from fusion minus the detector, by the rule of ADR-0031, section 3."""
    if difference_of_ap["low"] > 0:
        return "the specialised components detect better"
    if difference_of_ap["high"] < 0:
        return "a general language model detects better"
    return "inconclusive"


def row_result(  # noqa: PLR0913 - the row, its answers, its cost and what it is set against
    y: Labels,
    row: ValidationRow,
    readings: Sequence[Reading],
    cost_usd: str,
    fusion: Vector,
    *,
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """One row's metrics, its comparison with fusion and what Phase 7 applies.

    Raises:
        ValueError: If there is not one reading per receipt.
    """
    if len(readings) != y.size:
        msg = f"{len(readings)} readings for {y.size} receipts"
        raise ValueError(msg)
    calibrated = calibrate(scores(readings), y)
    p = calibrated.probabilities
    table = metrics(y, p, resamples) | {"threshold": choose_threshold(y, p)}
    if calibrated.platt is None:
        table["brier"] = table["ece"] = None
    platt = calibrated.platt
    return {
        "input": row.row.value,
        "settings": row.settings.model_dump(mode="json"),
        "track": row.track,
        "failures": dict(Counter(r.failure for r in readings if r.suspicion is None)),
        "cost_usd": cost_usd,
        "table": table,
        "against_fusion": difference(y, fusion, p, resamples),
        "fitted": {
            "platt": None if platt is None else {"slope": platt.slope, "offset": platt.offset},
            "threshold": table["threshold"],
        },
    }


def fusion_probabilities(
    check: Mapping[str, Any], table: InputsTable, receipts: Sequence[str], y: Labels, seed: int
) -> dict[str, Vector]:
    """Each track's frozen fusion probabilities on the validation receipts.

    ``check`` holds the results of Phase 5's validation check, whose fitted fusion model is
    applied to the validation inputs of system 2 seed ``seed``, the check's headline seed.

    Raises:
        ValueError: If a track's rows are not the receipts given, in order, or the rebuilt
            probabilities do not give the average precision the check recorded.
    """
    found = {}
    for track, name in TRACK_NAMES.items():
        rows = table.rows(track, seed)
        if [r.receipt for r in rows] != list(receipts):
            msg = f"the validation table's {track} rows are not the validation receipts"
            raise ValueError(msg)
        p = applied(check[track]["fitted"][FUSION_ROW.name], rows, FUSION_ROW)
        recorded = check[track]["table"][FUSION_ROW.name]["average_precision"]["value"]
        rebuilt = average_precision(y, p)
        if not math.isclose(rebuilt, recorded, rel_tol=0, abs_tol=1e-9):
            msg = f"fusion of {name} gives AP {rebuilt:.6f} here, {recorded:.6f} in the check"
            raise ValueError(msg)
        found[track] = p
    return found


def detector_validation(  # noqa: PLR0913, PLR0917 - the rows, their answers and costs, and fusion
    y: Labels,
    rows: Sequence[ValidationRow],
    readings: Mapping[str, Sequence[Reading]],
    costs: Mapping[str, str],
    fusion: Mapping[str, Vector],
    fusion_tables: Mapping[str, Mapping[str, Any]],
    *,
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Every result of the check, from each row's readings and cost and fusion of each track.

    ``fusion_tables`` are fusion's rows of Phase 5's ablation table, by track, shown beside the
    detector's.
    """
    found = {
        row.name: row_result(
            y, row, readings[row.name], costs[row.name], fusion[row.track], resamples=resamples
        )
        for row in rows
    }
    first = rows[0]
    rq6 = found[first.name]["against_fusion"]
    return {
        "receipts": int(y.size),
        "forged": int(y.sum()),
        "rows": found,
        "fusion": dict(fusion_tables),
        "rq6": {
            "row": first.name,
            "track": first.track,
            "difference": rq6,
            "answer": rq6_answer(rq6),
        },
    }
