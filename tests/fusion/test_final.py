import copy
import json
from typing import Any

import numpy as np
import pytest

from docfraud.fusion.development import VARIANTS
from docfraud.fusion.final import (
    frozen_check,
    frozen_metrics,
    frozen_row,
    rehearsal_mismatches,
    resaved_check,
    subset,
    success,
)
from docfraud.fusion.model import Platt
from docfraud.fusion.validation import FUSION_ROW, IDENTITY, Checked, validation_check
from tests.fusion.test_validation import SEEDS, labels, pages, tables

RESAMPLES = 50


def recorded_check() -> dict[str, Any]:
    """The check on the made-up tables, as its run record keeps it."""
    training, y_train, validation, documents = tables()
    found = validation_check(
        training, y_train, validation, documents, pages(documents), seeds=SEEDS, resamples=RESAMPLES
    )
    kept: dict[str, Any] = json.loads(json.dumps(found))
    return kept


def test_the_frozen_check_on_the_validation_split_gives_back_the_checks_record() -> None:
    training, y_train, validation, documents = tables()
    check = recorded_check()
    rebuilt = frozen_check(
        training,
        y_train,
        validation,
        documents,
        pages(documents),
        recorded=check,
        seeds=SEEDS,
        resamples=RESAMPLES,
    )
    assert rehearsal_mismatches(check, rebuilt) == []
    fusion = rebuilt["ocr"]["table"]["fusion"]
    assert fusion["f1"]["value"] == check["ocr"]["table"]["fusion"]["f1"]
    assert fusion["f1"]["low"] <= fusion["f1"]["value"] <= fusion["f1"]["high"]
    assert rebuilt["ocr"]["success"]["f1"] == fusion["f1"]
    moved = copy.deepcopy(rebuilt)
    moved["ocr"]["table"]["fusion"]["roc_auc"]["value"] += 0.01
    moved["policy"]["ocr"]["10 %"]["reviewed"] += 1
    assert rehearsal_mismatches(check, moved) == [
        "/ocr/table/fusion/roc_auc/value",
        "/T3/table/fusion of track T2/roc_auc/value",  # T3's table shows the same row
        "/policy/ocr/10 %/reviewed",
    ]


def test_a_row_fitted_again_must_be_the_recorded_one() -> None:
    training, y_train, validation, _ = tables()
    recorded = recorded_check()["ocr"]["fitted"]["fusion"]
    rows = (training.rows("ocr", 0), y_train, validation.rows("ocr", 0), FUSION_ROW)
    row = frozen_row(*rows, recorded)
    assert row.platt == Platt(recorded["platt"]["slope"], recorded["platt"]["offset"])
    changed = copy.deepcopy(recorded)
    changed["coefficients"]["rules"] += 1e-6
    with pytest.raises(ValueError, match="is not the recorded one \\(coefficients\\)"):
        frozen_row(*rows, changed)
    other = recorded_check()["ocr"]["fitted"][VARIANTS[0].name]
    with pytest.raises(ValueError, match="is not the recorded one"):
        frozen_row(*rows, other)


def test_f1_is_measured_at_the_frozen_threshold_with_an_interval() -> None:
    y = np.array([True, True, False, False, False, False])
    log_odds = np.array([3.0, -1.0, 2.0, -2.0, -3.0, -4.0])
    checked = Checked(None, IDENTITY, False, log_odds)  # type: ignore[arg-type]
    found = frozen_metrics(y, checked, threshold=0.5, resamples=RESAMPLES)
    assert found["f1"]["value"] == pytest.approx(0.5)  # one of two forged found, one false alarm
    assert found["threshold"] == 0.5
    assert (found["brier"], found["ece"]) == (None, None)  # uncalibrated: no calibration metrics
    calibrated = frozen_metrics(y, Checked(None, IDENTITY, True, log_odds), 0.5, RESAMPLES)  # type: ignore[arg-type]
    assert calibrated["brier"] is not None


@pytest.mark.parametrize(
    ("f1_low", "recall_low", "met"),
    [(0.30, 0.20, True), (0.284, 0.20, False), (0.30, 0.10, False)],
)
def test_the_success_criterion_needs_both_lower_bounds_above_their_references(
    f1_low: float, recall_low: float, met: bool
) -> None:
    table = {
        "f1": {"value": 0.5, "low": f1_low, "high": 0.7},
        "recall_at_10pct": {"value": 0.3, "low": recall_low, "high": 0.4},
    }
    assert success(table)["met"] is met


def test_the_sensitivity_subset_leaves_the_same_receipts_out_of_both() -> None:
    _, _, validation, documents = tables()
    left_out = frozenset({"V00", "V03"})
    table, kept = subset(validation, documents, left_out)
    assert [d.doc_id for d in kept] == [d.doc_id for d in documents if d.doc_id not in left_out]
    assert {r.receipt for r in table.receipts} == {d.doc_id for d in kept}


def test_track_r_sets_the_saved_receipts_against_the_original_ones() -> None:
    training, y_train, validation, documents = tables()
    y = labels(documents)
    recorded = recorded_check()["ocr"]
    found = resaved_check(
        training, y_train, validation, {95: validation}, y, recorded=recorded, resamples=RESAMPLES
    )
    assert found["95"]["table"] == found["original"]  # the same inputs measure the same
    assert found["95"]["against_original"]["value"] == 0.0
    assert set(found["95"]["success"]) >= {"f1_above_published", "recall_above_chance", "met"}
    shuffled = validation.model_copy(update={"receipts": validation.receipts[::-1]})
    with pytest.raises(ValueError, match="not the test receipts"):
        resaved_check(
            training, y_train, validation, {75: shuffled}, y, recorded=recorded, resamples=RESAMPLES
        )


def test_the_frozen_check_refuses_tables_that_are_not_the_receipts_given() -> None:
    training, y_train, validation, documents = tables()
    check = recorded_check()
    arguments: dict[str, Any] = {"recorded": check, "seeds": SEEDS, "resamples": RESAMPLES}
    with pytest.raises(ValueError, match="do not match the training labels"):
        frozen_check(training, y_train[1:], validation, documents, pages(documents), **arguments)
    with pytest.raises(ValueError, match="are not the receipts given"):
        frozen_check(training, y_train, validation, documents[1:], pages(documents), **arguments)


def test_a_rehearsal_names_every_value_missing_added_or_of_another_length() -> None:
    recorded = {"rq4": {"ocr": {"by_entity": [1, 2], "hits": 3}}}
    rebuilt = {"rq4": {"ocr": {"by_entity": [1], "located": 3}}}
    assert rehearsal_mismatches(recorded, rebuilt) == [
        "/rq4/ocr/by_entity",
        "/rq4/ocr/hits",
        "/rq4/ocr/located",
    ]
