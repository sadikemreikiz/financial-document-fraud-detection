from pathlib import Path
from typing import Any, cast

import numpy as np
import pytest
from sklearn.pipeline import Pipeline

from docfraud.baselines import bow
from docfraud.baselines.report import render_report
from docfraud.data.models import Document, Label, Split
from docfraud.evaluation.folds import assign_folds
from docfraud.evaluation.runs import JsonRunStore, RunRecord, run


def corpus(root: Path, n: int = 60, forged_every: int = 4) -> list[Document]:
    """Receipts whose forged copies carry a telltale word, so the classifier can learn them."""
    root.mkdir(parents=True, exist_ok=True)
    docs = []
    for i in range(n):
        forged = i % forged_every == 0
        transcript = root / f"R{i:03d}.txt"
        words = "TOTAL 12.00 CASH 20.00 CHANGE 8.00"
        transcript.write_text(f"SHOP {i}\n{words}\n{'EDITED' if forged else 'PRINTED'}\n", "utf-8")
        docs.append(
            Document(
                dataset="toy",
                doc_id=f"R{i:03d}",
                source_id=f"R{i:03d}",
                split=Split.TEST if i >= n - 12 else Split.TRAIN,
                label=Label.FORGED if forged else Label.AUTHENTIC,
                image_path=root / f"R{i:03d}.png",
                transcript_path=transcript,
            )
        )
    return docs


def test_the_baseline_learns_a_separable_signal(tmp_path: Path) -> None:
    docs = corpus(tmp_path)
    train = [d for d in docs if d.split is Split.TRAIN]
    test = [d for d in docs if d.split is Split.TEST]
    fitted = bow.fit(train)
    assert fitted.max_iter == bow.DEFAULT_MAX_ITER
    predicted = fitted.scores(test) > bow.THRESHOLD
    assert (predicted == bow.labels(test)).all()


def test_cross_validation_scores_every_training_receipt_out_of_fold(tmp_path: Path) -> None:
    docs = corpus(tmp_path)
    cv = bow.cross_validate(docs, assign_folds(docs))
    assert len(cv.doc_ids) == 48
    assert len(cv.max_iter) == 5
    assert ((cv.scores > bow.THRESHOLD) == cv.labels).all()


def test_cross_validation_needs_a_fold_for_every_training_receipt(tmp_path: Path) -> None:
    docs = corpus(tmp_path)
    folds = assign_folds(docs)
    late = docs[1].model_copy(update={"doc_id": "LATE", "source_id": "LATE"})
    with pytest.raises(ValueError, match="without a fold"):
        bow.cross_validate([*docs, late], folds)


def test_the_iteration_limit_is_raised_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    train = [d for d in corpus(tmp_path) if d.split is Split.TRAIN]
    monkeypatch.setattr(bow, "DEFAULT_MAX_ITER", 1)
    assert bow.fit(train).max_iter == bow.RAISED_MAX_ITER
    monkeypatch.setattr(bow, "RAISED_MAX_ITER", 1)
    with pytest.raises(RuntimeError, match="did not converge"):
        bow.fit(train)


class Undecided:
    def predict_proba(self, texts: list[str]) -> np.ndarray:
        return np.full((len(texts), 2), 0.5)


def test_a_probability_on_the_threshold_is_refused(tmp_path: Path) -> None:
    docs = corpus(tmp_path)[:2]
    fitted = bow.Fitted(model=cast(Pipeline, Undecided()), max_iter=1)
    with pytest.raises(ValueError, match="threshold exactly"):
        fitted.scores(docs)


def test_inputs_need_transcripts_and_labels(tmp_path: Path) -> None:
    doc = corpus(tmp_path)[0]
    with pytest.raises(ValueError, match="without transcripts"):
        bow.texts([doc.model_copy(update={"transcript_path": None})])
    with pytest.raises(ValueError, match="needs a label"):
        bow.labels([doc.model_copy(update={"label": None})])


def record(tmp_path: Path, name: str, results: dict[str, Any], unlock: str = "") -> RunRecord:
    store = JsonRunStore(tmp_path / "runs")
    with run(name, store=store, config={}, seeds={}, dataset_sha256={}) as current:
        if unlock:
            from docfraud.data.splits import select  # noqa: PLC0415 - only this helper needs it

            select([], Split.TEST, unlock_test=True, reason=unlock)
        for key, value in results.items():
            current.log(key, value)
    assert current.record is not None
    return current.record


def detection(value: float) -> dict[str, Any]:
    estimate = {"value": value, "low": value - 0.1, "high": value + 0.1}
    names = [
        "precision", "recall", "f1", "average_precision", "roc_auc", "mcc",
        "recall_at_10pct_review", "recall_at_20pct_review",
    ]  # fmt: skip
    return {
        "n_documents": 218,
        "n_forged": 35,
        "metrics": dict.fromkeys(names, estimate),
        "counts": {"tp": 5, "fp": 6, "fn": 30, "tn": 177},
        "max_iter": 100,
    }


def test_report_before_and_after_the_test_run(tmp_path: Path) -> None:
    cv = record(tmp_path, "bow-lr-cv", detection(0.2))
    before = render_report(cv, None)
    assert "Not run yet" in before
    assert "| F1 | 20.00 [10.0, 30.0] |" in before
    test = record(
        tmp_path,
        "bow-lr-test",
        detection(0.2) | {"published": bow.PUBLISHED, "reproduced": True},
        unlock="baseline reproduction",
    )
    after = render_report(cv, test)
    assert "| F1 | 17.78 | 20.00 [10.0, 30.0] |" in after
    assert "| Forged found / authentic flagged / forged missed | 4 / 6 / 31 | 5 / 6 / 30 |" in after
    assert '"baseline reproduction"' in after
    assert "**Reproduction passed.**" in after
