from pathlib import Path

import pytest

from docfraud.data.models import Document, Label, Split
from docfraud.evaluation.folds import (
    FoldAssignment,
    assign_folds,
    check_folds,
    load_folds,
    save_folds,
)


def documents(n: int = 60, forged_every: int = 5) -> list[Document]:
    docs = [
        Document(
            dataset="toy",
            doc_id=f"D{i:03d}",
            source_id=f"S{i:03d}",
            split=Split.TRAIN,
            label=Label.FORGED if i % forged_every == 0 else Label.AUTHENTIC,
            image_path=Path(f"D{i:03d}.png"),
        )
        for i in range(n)
    ]
    # an authentic/forged twin: same physical receipt, so it must stay in one fold
    twin = docs[0].model_copy(update={"doc_id": "TWIN", "label": Label.AUTHENTIC})
    held_out = docs[1].model_copy(update={"split": Split.TEST})
    return [*docs, twin, held_out]


def test_folds_cover_training_only_keep_twins_together_and_stratify() -> None:
    docs = documents()
    folds = assign_folds(docs)
    assert set(folds.folds) == {d.doc_id for d in docs if d.split is Split.TRAIN}
    assert folds.folds["TWIN"] == folds.folds["D000"]
    forged = {d.doc_id for d in docs if d.label is Label.FORGED and d.split is Split.TRAIN}
    per_fold = [len(folds.documents(f) & forged) for f in range(folds.n_folds)]
    assert max(per_fold) - min(per_fold) <= 1


def test_folds_depend_only_on_the_seed() -> None:
    docs = documents()
    assert assign_folds(docs) == assign_folds(list(reversed(docs)))
    assert assign_folds(docs, seed=1) != assign_folds(docs)


def test_stored_folds_round_trip(tmp_path: Path) -> None:
    folds = assign_folds(documents())
    path = tmp_path / "folds.json"
    save_folds(folds, path)
    assert load_folds(path) == folds


def test_stale_folds_are_detected() -> None:
    docs = documents()
    folds = assign_folds(docs)
    extra = docs[2].model_copy(update={"doc_id": "NEW", "source_id": "NEW"})
    with pytest.raises(ValueError, match="missing \\['NEW'\\]"):
        check_folds(folds, [*docs, extra])


def test_split_receipts_and_one_class_folds_are_detected() -> None:
    docs = documents()
    folds = assign_folds(docs)
    moved = dict(folds.folds) | {"TWIN": (folds.folds["D000"] + 1) % folds.n_folds}
    with pytest.raises(ValueError, match="different folds"):
        check_folds(folds.model_copy(update={"folds": moved}), docs)
    only_authentic = dict(folds.folds)
    forged = [d.doc_id for d in docs if d.label is Label.FORGED and d.split is Split.TRAIN]
    for doc_id in [*forged, "TWIN"]:  # the twin follows its forged copy
        only_authentic[doc_id] = 0
    broken = FoldAssignment(
        dataset="toy", split=Split.TRAIN, n_folds=5, seed=0, folds=only_authentic
    )
    with pytest.raises(ValueError, match="without both classes"):
        check_folds(broken, docs)


def test_mixed_datasets_and_unlabelled_documents_are_rejected() -> None:
    docs = documents()
    other = docs[2].model_copy(update={"dataset": "other"})
    with pytest.raises(ValueError, match="exactly one dataset"):
        assign_folds([*docs, other])
    unlabelled = docs[2].model_copy(update={"doc_id": "U", "source_id": "U", "label": None})
    with pytest.raises(ValueError, match="needs a label"):
        assign_folds([*docs, unlabelled])
