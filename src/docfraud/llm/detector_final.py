"""The frozen detector on the test split, once (ADR-0031, sections 3 and 10; ADR-0034, 7).

Every row runs the frozen instruction on the test receipts with the Platt scaling and the max-F1
threshold that its validation run set (:mod:`docfraud.llm.detector_validation`), applied
unchanged: nothing is fitted on the receipts measured, so F1 at the threshold, the Brier score
and the expected calibration error are unbiased and F1 gets an interval.

- **RQ6**, the ninth pre-named test: fusion of T2 against the main row on the frozen model, by
  the paired bootstrap of average precision and the rule of ADR-0031 (section 3).
- **The paper's result**: the text row against ChatGPT's F1 by ADR-0013's two conditions,
  descriptive.

The rows run in two parts, so that what is claimed is measured first (ADR-0034, decision 10):
the frozen model's two rows, then the other models' rows. Before either, a rehearsal applies the
same code to the validation receipts from the stored answers, without sending a request, and must
give back the validation run's record: Phase 6b changed the provider's code after the detector
was frozen, so this, not a check that no file changed, shows that the requests and their reading
are as they were.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from docfraud.evaluation.bootstrap import DEFAULT_RESAMPLES
from docfraud.fusion.development import difference
from docfraud.fusion.final import differences_in, flat_f1, metrics_at, success
from docfraud.fusion.model import Platt
from docfraud.llm.detector import DetectorRow, Reading
from docfraud.llm.detector_rounds import DETECTOR_CODE, scores
from docfraud.llm.detector_validation import Calibrated, ValidationRow, rq6_answer

type Labels = npt.NDArray[np.bool_]
type Vector = npt.NDArray[np.float64]

CLAIMS, DESCRIPTIVE = "claims", "descriptive"
RUN_NAMES = {CLAIMS: "llm-detector-test", DESCRIPTIVE: "llm-detector-test-descriptive"}
REHEARSAL_RUN = "llm-detector-rehearsal"

TEST_CODE = (
    *DETECTOR_CODE,
    "src/docfraud/fusion",
    "src/docfraud/llm/detector_final.py",
    "src/docfraud/llm/detector_rounds.py",
    "src/docfraud/llm/detector_validation.py",
    "src/docfraud/llm/run.py",
    "src/docfraud/llm/store.py",
)
"""What makes the detector's requests and results; unchanged from the rehearsal to the test."""


def part_rows(rows: Sequence[ValidationRow], part: str) -> tuple[ValidationRow, ...]:
    """The rows of one part: the frozen model's main and text rows, or the other models' rows.

    Raises:
        ValueError: If ``part`` is neither.
    """
    if part == CLAIMS:
        return tuple(rows[:2])
    if part == DESCRIPTIVE:
        return tuple(rows[2:])
    msg = f"no part {part!r}: {CLAIMS} or {DESCRIPTIVE}"
    raise ValueError(msg)


def frozen_calibration(readings: Sequence[Reading], fitted: Mapping[str, Any]) -> Calibrated:
    """A row's scores with the Platt scaling its validation run recorded, or none."""
    platt = fitted["platt"]
    return Calibrated(
        scores(readings), None if platt is None else Platt(platt["slope"], platt["offset"])
    )


def row_result(  # noqa: PLR0913 - the row, its answers, cost, fusion and what was recorded
    y: Labels,
    row: ValidationRow,
    readings: Sequence[Reading],
    cost_usd: str,
    fusion: Vector,
    *,
    fitted: Mapping[str, Any],
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """One row's metrics at its frozen threshold and its comparison with fusion.

    Raises:
        ValueError: If there is not one reading per receipt.
    """
    if len(readings) != y.size:
        msg = f"{len(readings)} readings for {y.size} receipts"
        raise ValueError(msg)
    calibrated = frozen_calibration(readings, fitted)
    p = calibrated.probabilities
    table = metrics_at(
        y,
        p,
        fitted["threshold"],
        calibrated=calibrated.platt is not None,
        resamples=resamples,
    )
    return {
        "input": row.row.value,
        "settings": row.settings.model_dump(mode="json"),
        "track": row.track,
        "failures": dict(Counter(r.failure for r in readings if r.suspicion is None)),
        "cost_usd": cost_usd,
        "table": table,
        "against_fusion": difference(y, fusion, p, resamples),
        "success": success(table),
    }


def detector_test(  # noqa: PLR0913 - the rows, their answers and costs, fusion and the record
    y: Labels,
    rows: Sequence[ValidationRow],
    readings: Mapping[str, Sequence[Reading]],
    costs: Mapping[str, str],
    fusion: Mapping[str, Vector],
    *,
    recorded: Mapping[str, Any],
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """Every result of the given rows, with each row's fitted calibration from ``recorded``.

    ``recorded`` is the validation run's results. If the rows hold RQ6's row, the frozen main
    row, the results hold RQ6's answer; if they hold the frozen text row, ADR-0013's conditions
    for it, set against the paper's result.
    """
    found = {
        row.name: row_result(
            y,
            row,
            readings[row.name],
            costs[row.name],
            fusion[row.track],
            fitted=recorded["rows"][row.name]["fitted"],
            resamples=resamples,
        )
        for row in rows
    }
    results: dict[str, Any] = {"receipts": int(y.size), "forged": int(y.sum()), "rows": found}
    rq6 = recorded["rq6"]["row"]
    if rq6 in found:
        difference_of_ap = found[rq6]["against_fusion"]
        results["rq6"] = {
            "row": rq6,
            "track": found[rq6]["track"],
            "difference": difference_of_ap,
            "answer": rq6_answer(difference_of_ap),
        }
    for name, row in found.items():
        if row["input"] == DetectorRow.TRANSCRIPT.value:
            results["paper"] = {"row": name} | row["success"]
    return results


def rehearsal_mismatches(recorded: Mapping[str, Any], rebuilt: Mapping[str, Any]) -> list[str]:
    """Where :func:`detector_test` on the validation split differs from that run's record.

    Every row's input, model, failures, cost, metrics and comparison with fusion must come back,
    with F1's value equal to the recorded F1, and so must RQ6. What only the frozen application
    adds is left out: F1's interval and ADR-0013's conditions.
    """
    out: list[str] = []
    for name, row in recorded["rows"].items():
        found = rebuilt["rows"].get(name)
        if found is None:
            out.append(f"/rows/{name}")
            continue
        for part in ("input", "settings", "track", "failures", "cost_usd", "against_fusion"):
            out += differences_in(row[part], found[part], f"/rows/{name}/{part}")
        tables = flat_f1({"row": row["table"]}), flat_f1({"row": found["table"]})
        out += differences_in(*tables, f"/rows/{name}/table")
    out += differences_in(recorded["rq6"], rebuilt.get("rq6"), "/rq6")
    return out
