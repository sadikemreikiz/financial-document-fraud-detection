import logging
from pathlib import Path

import pytest

from docfraud.data.models import Document, Split
from docfraud.data.splits import TestSetLockedError, check_disjoint, select


def doc(doc_id: str, split: Split, source: str | None = None, dataset: str = "d") -> Document:
    return Document(
        dataset=dataset,
        doc_id=doc_id,
        source_id=source or f"s/{doc_id}",
        split=split,
        image_path=Path(f"{doc_id}.png"),
    )


DOCS = (doc("1", Split.TRAIN), doc("2", Split.VAL), doc("3", Split.TEST))


def test_select_returns_one_split() -> None:
    assert [d.doc_id for d in select(DOCS, Split.TRAIN)] == ["1"]
    assert [d.doc_id for d in select(DOCS, Split.VAL)] == ["2"]


def test_test_split_is_locked() -> None:
    with pytest.raises(TestSetLockedError, match="locked"):
        select(DOCS, Split.TEST)


def test_unlocking_requires_a_reason() -> None:
    with pytest.raises(ValueError, match="reason"):
        select(DOCS, Split.TEST, unlock_test=True)


def test_unlocking_is_logged(caplog: pytest.LogCaptureFixture) -> None:
    with caplog.at_level(logging.WARNING):
        docs = select(DOCS, Split.TEST, unlock_test=True, reason="final evaluation")
    assert [d.doc_id for d in docs] == ["3"]
    assert "final evaluation" in caplog.text


def test_check_disjoint_accepts_clean_splits() -> None:
    check_disjoint(DOCS)


def test_check_disjoint_finds_duplicates() -> None:
    with pytest.raises(ValueError, match=r"duplicate documents: \['d/1'\]"):
        check_disjoint([*DOCS, doc("1", Split.TRAIN)])


def test_check_disjoint_finds_cross_dataset_source_leakage() -> None:
    other = doc("9", Split.TRAIN, source="s/3", dataset="other")
    with pytest.raises(ValueError, match=r"sources in several splits: \['s/3'\]"):
        check_disjoint([*DOCS, other])
