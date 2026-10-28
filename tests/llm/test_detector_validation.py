from typing import Any

import numpy as np
import pytest

from docfraud.fusion.development_report import TRACK_NAMES
from docfraud.fusion.training_inputs import InputsTable
from docfraud.fusion.validation import FUSION_ROW, check_row, check_track
from docfraud.llm.detector import SONNET, Reading
from docfraud.llm.detector_rounds import OPUS, UNSCORED
from docfraud.llm.detector_validation import (
    Calibrated,
    calibrate,
    detector_validation,
    fusion_probabilities,
    rq6_answer,
    validation_rows,
)
from docfraud.llm.detector_validation_report import UNCALIBRATED, render_report
from tests.fusion.test_validation import SEEDS, labels, tables

RESAMPLES = 20


def test_the_rows_start_with_rq6s_and_add_the_stronger_model_only_by_its_rule() -> None:
    rows = validation_rows(SONNET, stronger=False)
    assert [r.name for r in rows] == [
        "image and OCR text, claude-sonnet-5-5",
        "transcript, claude-sonnet-5-5",
        "image and OCR text, claude-haiku-5-5",
    ]
    assert [r.track for r in rows] == ["ocr", "transcript", "ocr"]
    assert validation_rows(SONNET, stronger=True)[-1].settings == OPUS


def test_platt_scaling_keeps_the_order_or_the_row_keeps_its_own_score() -> None:
    y = np.array([False, False, True, True, False, True])
    s = np.array([10.0, 60.0, 80.0, 40.0, 30.0, 70.0])
    kept = calibrate(s, y)
    assert kept.platt is not None
    assert kept.platt.slope > 0
    assert np.all(np.diff(kept.probabilities[np.argsort(s)]) > 0)
    reversed_order = calibrate(s, ~y)
    assert reversed_order.platt is None
    assert np.allclose(reversed_order.probabilities, s / 100)
    unscored = Calibrated(np.array([UNSCORED, 50.0]), None)
    assert np.allclose(unscored.probabilities, [0.0, 0.5])


def test_rq6_is_answered_by_where_the_interval_lies() -> None:
    assert rq6_answer({"value": 0.3, "low": 0.1, "high": 0.5}) == (
        "the specialised components detect better"
    )
    assert rq6_answer({"value": -0.3, "low": -0.5, "high": -0.1}) == (
        "a general language model detects better"
    )
    assert rq6_answer({"value": 0.1, "low": -0.1, "high": 0.3}) == "inconclusive"


def phase_5() -> tuple[dict[str, Any], InputsTable, list[str], np.ndarray]:
    """Phase 5's check of both tracks on the toy receipts, as its record holds it."""
    training, y_train, validation, documents = tables()
    y = labels(documents)
    check = {
        track: check_track(
            {s: training.rows(track, s) for s in SEEDS},
            y_train,
            {s: validation.rows(track, s) for s in SEEDS},
            y,
            resamples=RESAMPLES,
        )[0]
        for track in TRACK_NAMES
    }
    return check, validation, [d.doc_id for d in documents], y


def test_fusion_is_rebuilt_from_the_check_and_must_give_its_average_precision() -> None:
    check, validation, receipts, y = phase_5()
    found = fusion_probabilities(check, validation, receipts, y, 0)
    training, y_train, _, _ = tables()
    for track in TRACK_NAMES:
        fitted_again = check_row(
            training.rows(track, 0), y_train, validation.rows(track, 0), y, FUSION_ROW
        )
        assert np.allclose(found[track], fitted_again.probabilities)
    with pytest.raises(ValueError, match="not the validation receipts"):
        fusion_probabilities(check, validation, receipts[::-1], y, 0)
    check["ocr"]["table"]["fusion"]["average_precision"]["value"] += 0.01
    with pytest.raises(ValueError, match="in the check"):
        fusion_probabilities(check, validation, receipts, y, 0)


def _readings(y: np.ndarray, *, reverse: bool = False) -> list[Reading]:
    """Scores that mostly follow the labels, one answer that failed its form."""
    found = []
    for i, forged in enumerate(y):
        high = (bool(forged) ^ (i % 7 == 0)) ^ reverse  # a few against the labels
        found.append(Reading(None, failure="refusal") if i == 1 else Reading(75 if high else 25))
    return found


def test_the_check_measures_every_row_against_fusion_and_answers_rq6() -> None:
    check, validation, receipts, y = phase_5()
    fusion = fusion_probabilities(check, validation, receipts, y, 0)
    rows = validation_rows(SONNET, stronger=True)
    readings = {r.name: _readings(y, reverse=r.settings.model == "claude-haiku-5-5") for r in rows}
    results = detector_validation(
        y,
        rows,
        readings,
        dict.fromkeys(readings, "0.0100"),
        fusion,
        {track: check[track]["table"]["fusion"] for track in TRACK_NAMES},
        resamples=RESAMPLES,
    )
    assert (results["receipts"], results["forged"]) == (30, 10)
    assert list(results["rows"]) == [r.name for r in rows]
    main = results["rows"][rows[0].name]
    assert main["failures"] == {"refusal": 1}
    assert main["fitted"]["platt"] is not None
    assert main["fitted"]["threshold"] == main["table"]["threshold"]
    assert main["against_fusion"] == results["rq6"]["difference"]
    assert results["rq6"]["answer"] == rq6_answer(main["against_fusion"])
    haiku = results["rows"][rows[2].name]
    assert haiku["fitted"]["platt"] is None
    assert (haiku["table"]["brier"], haiku["table"]["ece"]) == (None, None)
    with pytest.raises(ValueError, match="readings for"):
        detector_validation(
            y, rows[:1], {rows[0].name: readings[rows[0].name][1:]}, {rows[0].name: "0"}, fusion, {}
        )
    report = render_report(results, "c" * 40)
    assert "**RQ6, as a check.**" in report
    assert f"{rows[2].name} {UNCALIBRATED}" in report
    assert "What Phase 7 applies to the test split" in report
    assert "| image and OCR text, claude-sonnet-5-5 | refusal | 1 |" in report
    assert "fusion of T2 (OCR), Phase 5" in report
