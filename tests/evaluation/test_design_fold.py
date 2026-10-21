import re
from pathlib import Path

import pytest

from docfraud.data.findit import SENSITIVITY_EXCLUSIONS
from docfraud.evaluation.design_fold import (
    CLEAN_FOLDS,
    DESIGN_FOLD,
    DESIGN_FOLD_IDS,
    EARLIER_CONTACTS,
    NotInDesignFoldError,
    require_design_fold,
)
from docfraud.evaluation.folds import CV_FOLDS, FOLD_SEED, FoldAssignment, load_folds

ROOT = Path(__file__).resolve().parents[2]
REPORT = ROOT / "reports" / "forensics" / "preregistration-data.md"


@pytest.fixture(scope="module")
def folds() -> FoldAssignment:
    return load_folds(ROOT / "annotations" / "findit2" / "cv-folds.json")


def test_design_fold_is_fold_2_of_the_stored_folds(folds: FoldAssignment) -> None:
    assert (folds.dataset, folds.n_folds, folds.seed) == ("findit2", CV_FOLDS, FOLD_SEED)
    assert folds.documents(DESIGN_FOLD) == DESIGN_FOLD_IDS
    assert len(DESIGN_FOLD_IDS) == 113


def test_the_clean_folds_are_the_other_four(folds: FoldAssignment) -> None:
    assert sorted((*CLEAN_FOLDS, DESIGN_FOLD)) == list(range(CV_FOLDS))
    assert sum(len(folds.documents(fold)) for fold in CLEAN_FOLDS) == 449


def test_earlier_contacts_are_the_exclusion_list_of_the_report(folds: FoldAssignment) -> None:
    match = re.search(r"The (\d+) clean-fold receipts above: (.+)\.\n", REPORT.read_text("utf-8"))
    assert match, f"exclusion list not found in {REPORT.name}"
    listed = match[2].split(", ")
    assert len(listed) == int(match[1]) == 98
    assert frozenset(listed) == EARLIER_CONTACTS
    assert all(folds.folds[i] != DESIGN_FOLD for i in EARLIER_CONTACTS)


def test_design_fold_receipts_may_be_viewed() -> None:
    require_design_fold("X51009568881")  # one identifier, not its characters
    require_design_fold(sorted(DESIGN_FOLD_IDS))
    require_design_fold([])


def test_clean_fold_and_other_split_receipts_are_refused(folds: FoldAssignment) -> None:
    clean = min(folds.documents(0))
    test_split = min(SENSITIVITY_EXCLUSIONS)  # test receipts are in no fold
    assert test_split not in folds.folds
    for refused in (clean, test_split, "X00000000000"):
        with pytest.raises(NotInDesignFoldError, match=f"ADR-0014\\): {refused}$"):
            require_design_fold(refused)


def test_one_refused_receipt_stops_the_whole_request(folds: FoldAssignment) -> None:
    clean = min(folds.documents(1))
    with pytest.raises(NotInDesignFoldError, match=f"^1 receipt\\(s\\) .*: {clean}$"):
        require_design_fold([*sorted(DESIGN_FOLD_IDS)[:3], clean])


def test_long_refusals_name_only_the_first_receipts() -> None:
    refused = sorted(EARLIER_CONTACTS)
    with pytest.raises(NotInDesignFoldError, match=r"^98 receipt.*, X51005365179 and 93 more$"):
        require_design_fold(refused)
