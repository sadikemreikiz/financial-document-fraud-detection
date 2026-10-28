import numpy as np
import pytest

from docfraud.llm.agent_cases import AUTHENTIC_CASES, development_cases, out_of_fold
from tests.fusion.test_training_inputs import table

RECEIPTS = [f"R{i:02d}" for i in range(40)]


def test_development_takes_the_reviewed_design_fold_receipts_and_the_highest_authentic() -> None:
    p = np.linspace(0.99, 0.01, 40)  # R00 highest
    forged = np.array([i in {0, 1, 2, 3, 7} for i in range(40)])
    design = {"R01", "R02", "R03", "R05", "R06", "R07", "R08", "R09", "R30"}
    found = development_cases(RECEIPTS, forged, p, design)
    assert found.forged == ("R01", "R02", "R03")  # the highest 10 % are R00 to R03
    assert len(found.authentic) == AUTHENTIC_CASES
    assert found.authentic == ("R05", "R06", "R08")  # R07 is forged, R30 ranks lower
    assert found.receipts == found.forged + found.authentic
    assert found.pilot == ("R01", "R02", "R05")
    with pytest.raises(ValueError, match="authentic"):
        development_cases(RECEIPTS, forged & (np.arange(40) != 2), p, design)  # R02 authentic


def test_every_training_receipt_is_assessed_by_the_model_of_its_fold() -> None:
    rows = table(n=40, seeds=(0,)).rows("ocr", 0)
    forged = np.array([i % 4 == 0 for i in range(40)])
    fold_of = np.array([i % 5 for i in range(40)])
    found = out_of_fold(rows, forged, fold_of)
    assert list(found.assessments) == RECEIPTS
    probabilities = [found.assessments[r].probability for r in RECEIPTS]
    assert np.allclose(probabilities, found.probabilities)
    first = found.assessments["R00"]
    assert (first.track, first.budget) == ("T2 (OCR)", 0.1)
    assert all(a.threshold == first.threshold for a in found.assessments.values())
