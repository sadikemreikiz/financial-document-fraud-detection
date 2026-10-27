from pathlib import Path
from typing import Any

import numpy as np
import pytest

from docfraud.data.models import BBox, Document, Entity, Label, Region, RegionRole, Split, Technique
from docfraud.decision import ReviewPolicy
from docfraud.fusion.development import PRACTICAL_VARIANTS, SINGLE_SOURCES, VARIANTS, matrix
from docfraud.fusion.evidence import SystemTwo
from docfraud.fusion.model import FusionModel, Platt
from docfraud.fusion.training_inputs import InputsTable
from docfraud.fusion.validation import (
    FUSION_ROW,
    IDENTITY,
    check_row,
    check_track,
    grounding,
    policy_results,
    row_metrics,
    validation_check,
)
from docfraud.fusion.validation_inputs import rules_scorers, validation_inputs
from docfraud.fusion.validation_report import render_report
from tests.fusion.test_training_inputs import DUE_BOX, FAIL, PAGE, PASS, SOURCES, table

SEEDS = (0, 1, 2)
N_TRAIN, N_VALIDATION = 40, 30
PAGE_BOX = BBox(x=0, y=0, width=300, height=200)
WORDS = [BBox(x=10, y=10 + 30 * k, width=200, height=24) for k in range(5)]


def _forged(i: int) -> bool:
    return i % 3 == 0


def _fails(i: int) -> bool:
    """Most forged validation receipts fail a rule, and a few authentic ones do too."""
    return (_forged(i) and i % 2 == 0) or (not _forged(i) and i % 5 == 0) or i in {3, 9}


def validation_documents() -> list[Document]:
    modified = Region(
        box=DUE_BOX,
        role=RegionRole.MODIFIED,
        techniques=frozenset({Technique.CPI}),
        entity=Entity.TOTAL_PAYMENT,
    )
    return [
        Document(
            dataset="toy",
            doc_id=f"V{i:02d}",
            source_id=f"V{i:02d}",
            split=Split.VAL,
            label=Label.FORGED if _forged(i) else Label.AUTHENTIC,
            image_path=Path(f"V{i:02d}.png"),
            regions=(modified,) if _forged(i) else (),
            has_region_annotations=_forged(i),
            handwritten_annotation=not _forged(i) and i % 4 == 1,
            editing_software="gimp" if _forged(i) else None,
        )
        for i in range(N_VALIDATION)
    ]


def tables() -> tuple[InputsTable, np.ndarray, InputsTable, list[Document]]:
    """A training table, its labels, a validation table made like the real one, and receipts."""
    training = table(n=N_TRAIN, seeds=SEEDS)
    y_train = np.array([i % 4 == 0 for i in range(N_TRAIN)])
    outcomes = [FAIL if f else PASS for f in y_train]
    scorers = rules_scorers({"transcript": outcomes, "ocr": [PASS] * N_TRAIN}, list(y_train))
    documents = validation_documents()
    receipts = [d.doc_id for d in documents]
    readings = [(FAIL if _fails(i) else PASS, SOURCES) for i in range(N_VALIDATION)]
    system_2 = {
        seed: {
            r: SystemTwo(0.1 * ((i + seed) % 4), BBox(x=105, y=62, width=8, height=8))
            for i, r in enumerate(receipts)
        }
        for seed in SEEDS
    }
    validation = validation_inputs(
        receipts,
        {"transcript": readings, "ocr": readings},
        page=[PAGE] * N_VALIDATION,
        system_2=system_2,
        scorers=scorers,
    )
    return training, y_train, validation, documents


def labels(documents: list[Document]) -> np.ndarray:
    return np.array([d.label is Label.FORGED for d in documents])


def pages(documents: list[Document]) -> dict[str, tuple[BBox, list[BBox]]]:
    return {d.doc_id: (PAGE_BOX, WORDS) for d in documents if d.label is Label.FORGED}


def test_a_row_is_fitted_on_the_training_split_and_calibrated_on_the_validation_split() -> None:
    training, y_train, validation, documents = tables()
    y = labels(documents)
    rows_t, rows_v = training.rows("transcript", 0), validation.rows("transcript", 0)
    checked = check_row(rows_t, y_train, rows_v, y, FUSION_ROW)
    alone = FusionModel.fit(matrix(rows_t, FUSION_ROW), y_train, FUSION_ROW.inputs)
    assert checked.model.coefficients == alone.coefficients
    assert np.allclose(checked.log_odds, alone.log_odds(matrix(rows_v, FUSION_ROW)))
    assert checked.calibrated
    assert checked.platt == Platt.fit(checked.log_odds, y)
    order = np.argsort(checked.log_odds)
    assert np.all(np.diff(checked.probabilities[order]) >= 0)


def test_a_row_that_platt_scaling_would_reverse_keeps_its_own_probabilities() -> None:
    training, y_train, validation, documents = tables()
    reversed_labels = ~labels(documents)
    rows = (training.rows("transcript", 0), y_train, validation.rows("transcript", 0))
    checked = check_row(*rows, reversed_labels, FUSION_ROW)
    assert (checked.calibrated, checked.platt) == (False, IDENTITY)
    assert np.allclose(checked.probabilities, 1 / (1 + np.exp(-checked.log_odds)))
    found = row_metrics(reversed_labels, checked, resamples=50)
    assert (found["brier"], found["ece"]) == (None, None)
    assert 0 <= found["threshold"] <= 1


def test_the_track_check_holds_every_row_comparison_seed_and_what_phase_7_applies() -> None:
    training, y_train, validation, documents = tables()
    by_seed = {s: training.rows("ocr", s) for s in SEEDS}
    results, checked = check_track(
        by_seed, y_train, {s: validation.rows("ocr", s) for s in SEEDS}, labels(documents),
        resamples=50,
    )  # fmt: skip
    names = [v.name for v in VARIANTS]
    assert list(results["table"]) == list(results["fitted"]) == list(checked) == names
    assert set(results["comparisons"]) == {v.name for v in SINGLE_SOURCES} | {
        "score-level baseline"
    }
    assert set(results["seeds"]) == {"0", "1", "2"}
    assert (results["receipts"], results["forged"]) == (N_VALIDATION, 10)
    fusion = results["fitted"]["fusion"]
    assert set(fusion) == {"coefficients", "intercept", "means", "scales", "platt", "calibrated"}
    assert list(fusion["coefficients"]) == list(FUSION_ROW.inputs)
    with pytest.raises(ValueError, match="seeds"):
        check_track(by_seed, y_train, {0: validation.rows("ocr", 0)}, labels(documents))


def test_the_policy_reviews_the_highest_share_by_a_threshold_set_without_labels() -> None:
    p = np.linspace(0.01, 0.99, 50)
    y = np.zeros(50, dtype=bool)
    y[[49, 45, 30, 2]] = True
    found = policy_results(y, p, resamples=50)
    assert set(found) == {"10 %", "20 %"}
    ten = found["10 %"]
    assert ten["threshold"] == ReviewPolicy.set_on(p, 0.10).threshold == pytest.approx(p[-5])
    assert (ten["reviewed"], ten["forged_reviewed"], ten["forged"]) == (5, 2, 4)
    assert ten["recall"]["value"] == pytest.approx(0.5)
    assert ten["precision"]["value"] == pytest.approx(0.4)
    assert ten["review_share"]["value"] == pytest.approx(0.1)
    assert ten["recall"]["low"] <= 0.5 <= ten["recall"]["high"]
    other = policy_results(~y, p, resamples=50)
    assert other["10 %"]["threshold"] == ten["threshold"]  # labels do not move the threshold


def test_every_audit_report_is_grounded() -> None:
    training, y_train, validation, documents = tables()
    rows = validation.rows("transcript", 0)
    checked = check_row(
        training.rows("transcript", 0), y_train, rows, labels(documents), FUSION_ROW
    )
    policy = ReviewPolicy.set_on(checked.probabilities, 0.10)
    found = grounding(rows, checked, policy, "T1 (transcripts)")
    assert (found["reports"], found["violations"], found["rate"]) == (N_VALIDATION, [], 1.0)
    assert found["facts"] > N_VALIDATION * 3


def check_results() -> dict[str, Any]:
    """The results of the check on the small made-up tables."""
    training, y_train, validation, documents = tables()
    return validation_check(
        training, y_train, validation, documents, pages(documents), seeds=SEEDS, resamples=50
    )


def test_the_check_measures_every_section_on_both_tracks() -> None:
    results = check_results()
    assert set(results) == {
        "transcript", "ocr", "T3", "policy", "rq3", "false_alarms", "rq4", "grounding",
    }  # fmt: skip
    for track in ("transcript", "ocr"):
        assert results["rq4"][track]["receipts"] == 10
        assert results["rq3"][track]["technique"]["CPI"]["receipts"] == 10
        assert results["grounding"][track]["violations"] == []
        alarms = results["false_alarms"][track]["fusion"]
        assert set(alarms) == {"handwritten", "none"}
        assert set(results["policy"][track]) == {"10 %", "20 %"}
    assert list(results["T3"]["table"]) == [
        "fusion of track T2",
        *(v.name for v in PRACTICAL_VARIANTS),
    ]
    assert set(results["T3"]["fitted"]) == {v.name for v in PRACTICAL_VARIANTS}


def test_the_check_refuses_tables_that_are_not_the_receipts_given() -> None:
    training, y_train, validation, documents = tables()
    with pytest.raises(ValueError, match="not the validation receipts"):
        validation_check(
            training, y_train, validation, documents[::-1], {}, seeds=SEEDS, resamples=50
        )
    with pytest.raises(ValueError, match="do not match the training labels"):
        validation_check(
            training, y_train[1:], validation, documents, {}, seeds=SEEDS, resamples=50
        )


def test_the_report_renders_every_section_and_marks_uncalibrated_rows() -> None:
    results = check_results()
    results["ocr"]["fitted"]["copy-move"]["calibrated"] = False
    text = render_report(results, "0123456789")
    for title in (
        "Ablation table, T1 (transcripts)",
        "Comparisons of section 8, T2 (OCR)",
        "Fusion by system 2 seed",
        "What Phase 7 applies to the test split",
        "The review policy",
        "Track T3",
        "RQ3 by technique, T2 (OCR)",
        "False alarms on authentic receipts, by benign mark",
        "RQ4: the report's first region",
        "Hits among the receipts sent to review",
        "Grounding of the audit reports",
    ):
        assert title in text
    assert "`0123456`" in text
    assert "copy-move †" in text
    assert "on the 30 validation receipts" in text
    assert "the 3rd highest of 30 at 10 %" in text
    assert "Every grounding violation" not in text
    results["grounding"]["ocr"]["violations"] = ["V01: 0.123"]
    assert "V01: 0.123" in render_report(results, None)


def test_the_report_notes_ceilings_tiny_bounds_and_shares_of_one() -> None:
    results = check_results()
    results["ocr"]["comparisons"]["copy-move"] |= {"low": 0.00018, "above_zero": True}
    policy = results["policy"]["transcript"]["10 %"]
    policy |= {"reviewed": 3, "forged_reviewed": 3}
    text = render_report(results, None)
    assert "(0.0002 to" in text
    assert "At most 3 receipts fit in a 10 % budget and 10 are forged" in text
    assert "T1 (transcripts) at 10 %, precision 3 of 3: 0.292 to 1.000" in text
