import copy
import json
from typing import Any

import pytest

from docfraud.fusion.development_report import TRACK_NAMES
from docfraud.llm.detector import SONNET
from docfraud.llm.detector_final import (
    CLAIMS,
    DESCRIPTIVE,
    detector_test,
    part_rows,
    rehearsal_mismatches,
)
from docfraud.llm.detector_validation import (
    detector_validation,
    fusion_probabilities,
    validation_rows,
)
from tests.llm.test_detector_validation import RESAMPLES, _readings, phase_5


def test_the_claims_are_the_frozen_models_rows_and_run_first() -> None:
    rows = validation_rows(SONNET, stronger=True)
    assert part_rows(rows, CLAIMS) == rows[:2]
    assert [r.settings.model for r in part_rows(rows, DESCRIPTIVE)] == [
        "claude-haiku-5-5",
        "claude-opus-5-5",
    ]
    with pytest.raises(ValueError, match="no part"):
        part_rows(rows, "all")


def _validation_run() -> tuple[dict[str, Any], dict[str, Any]]:
    """The arguments of a validation run on made-up receipts, and its record's results."""
    check, validation, receipts, y = phase_5()
    rows = validation_rows(SONNET, stronger=True)
    given: dict[str, Any] = {
        "y": y,
        "rows": rows,
        "readings": {
            r.name: _readings(y, reverse=r.settings.model == "claude-haiku-5-5") for r in rows
        },
        "costs": {r.name: "0.0100" for r in rows},
        "fusion": fusion_probabilities(check, validation, receipts, y, 0),
    }
    tables = {track: check[track]["table"]["fusion"] for track in TRACK_NAMES}
    recorded = detector_validation(**given, fusion_tables=tables, resamples=RESAMPLES)
    return given, json.loads(json.dumps(recorded))


def test_the_test_runs_code_gives_back_the_validation_run_from_its_record() -> None:
    given, recorded = _validation_run()
    rebuilt = detector_test(**given, recorded=recorded, resamples=RESAMPLES)
    assert rehearsal_mismatches(recorded, rebuilt) == []
    main = rebuilt["rows"]["image and OCR text, claude-sonnet-5-5"]
    assert main["table"]["threshold"] == recorded["rows"][main_name()]["fitted"]["threshold"]
    assert main["table"]["f1"]["low"] <= main["table"]["f1"]["value"] <= main["table"]["f1"]["high"]
    assert rebuilt["rq6"]["answer"] == recorded["rq6"]["answer"]
    assert rebuilt["paper"]["row"] == "transcript, claude-sonnet-5-5"
    haiku = rebuilt["rows"]["image and OCR text, claude-haiku-5-5"]
    assert (haiku["table"]["brier"], haiku["table"]["ece"]) == (None, None)  # its own score
    changed = copy.deepcopy(rebuilt)
    changed["rows"][main_name()]["cost_usd"] = "0.0200"
    del changed["rq6"]
    assert rehearsal_mismatches(recorded, changed) == [f"/rows/{main_name()}/cost_usd", "/rq6"]
    del changed["rows"][main_name()]
    assert f"/rows/{main_name()}" in rehearsal_mismatches(recorded, changed)


def main_name() -> str:
    return "image and OCR text, claude-sonnet-5-5"


def test_a_part_without_the_frozen_main_row_gives_no_rq6_answer() -> None:
    given, recorded = _validation_run()
    descriptive = part_rows(given["rows"], DESCRIPTIVE)
    found = detector_test(**(given | {"rows": descriptive}), recorded=recorded, resamples=RESAMPLES)
    assert "rq6" not in found
    assert "paper" not in found
    assert list(found["rows"]) == [r.name for r in descriptive]
    short = {r.name: given["readings"][r.name][1:] for r in descriptive}
    with pytest.raises(ValueError, match="readings for"):
        detector_test(**(given | {"rows": descriptive, "readings": short}), recorded=recorded)
