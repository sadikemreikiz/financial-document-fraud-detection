"""Cross-validation folds over the training split, fixed once and stored.

Development uses stratified 5-fold cross-validation on the training split (evaluation protocol,
section 3). The folds are drawn once with a fixed seed and stored, so that every experiment,
now and later, is measured on exactly the same partition even if the drawing code or a library
changes. Copies of one physical receipt share a ``source_id`` and are kept in the same fold
(ADR-0009), so a model is never tested on a twin of a receipt it was trained on.
"""

from collections import defaultdict
from collections.abc import Iterable
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict
from sklearn.model_selection import StratifiedGroupKFold

from docfraud.data.models import Document, Label, Split

CV_FOLDS = 5
FOLD_SEED = 0


class FoldAssignment(BaseModel):
    """The cross-validation fold of every training document."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    split: Split
    n_folds: int
    seed: int
    folds: dict[str, int]

    def documents(self, fold: int) -> frozenset[str]:
        """Identifiers of the documents held out in one fold."""
        return frozenset(doc_id for doc_id, f in self.folds.items() if f == fold)


def assign_folds(
    documents: Iterable[Document], *, n_folds: int = CV_FOLDS, seed: int = FOLD_SEED
) -> FoldAssignment:
    """Draw stratified folds over the training split, keeping copies of a receipt together.

    The result depends only on the seed and the sorted training documents.

    Raises:
        ValueError: If documents from several datasets are given or a document has no label.
    """
    train = sorted((d for d in documents if d.split is Split.TRAIN), key=lambda d: d.doc_id)
    datasets = {d.dataset for d in train}
    if len(datasets) != 1:
        msg = f"expected training documents of exactly one dataset, got {sorted(datasets)}"
        raise ValueError(msg)
    if any(d.label is None for d in train):
        msg = "every training document needs a label to be stratified"
        raise ValueError(msg)
    labels = np.array([d.label is Label.FORGED for d in train], dtype=int)
    groups = [d.source_id for d in train]
    splitter = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=seed)
    folds: dict[str, int] = {}
    for fold, (_, held_out) in enumerate(splitter.split(np.zeros(len(train)), labels, groups)):
        for index in held_out:
            folds[train[int(index)].doc_id] = fold
    assignment = FoldAssignment(
        dataset=datasets.pop(), split=Split.TRAIN, n_folds=n_folds, seed=seed, folds=folds
    )
    check_folds(assignment, train)
    return assignment


def check_folds(assignment: FoldAssignment, documents: Iterable[Document]) -> None:
    """Verify that stored folds still match the training documents.

    Raises:
        ValueError: If a training document is unassigned or unknown, a physical receipt is split
            across folds, or a fold lacks one of the classes.
    """
    train = [d for d in documents if d.split is Split.TRAIN and d.dataset == assignment.dataset]
    ids = {d.doc_id for d in train}
    missing, unknown = sorted(ids - assignment.folds.keys()), sorted(assignment.folds.keys() - ids)
    if missing or unknown:
        msg = f"folds do not match the training split: missing {missing}, unknown {unknown}"
        raise ValueError(msg)
    folds_by_source: dict[str, set[int]] = defaultdict(set)
    labels_by_fold: dict[int, set[Label | None]] = defaultdict(set)
    for d in train:
        folds_by_source[d.source_id].add(assignment.folds[d.doc_id])
        labels_by_fold[assignment.folds[d.doc_id]].add(d.label)
    split_sources = sorted(s for s, f in folds_by_source.items() if len(f) > 1)
    if split_sources:
        msg = f"copies of one receipt are in different folds: {split_sources}"
        raise ValueError(msg)
    expected = {Label.AUTHENTIC, Label.FORGED}
    incomplete = sorted(f for f in range(assignment.n_folds) if labels_by_fold[f] != expected)
    if incomplete:
        msg = f"folds without both classes: {incomplete}"
        raise ValueError(msg)


def save_folds(assignment: FoldAssignment, path: Path) -> None:
    """Write folds as JSON, documents in sorted order so that diffs stay readable."""
    ordered = assignment.model_copy(update={"folds": dict(sorted(assignment.folds.items()))})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(ordered.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_folds(path: Path) -> FoldAssignment:
    """Read stored folds."""
    return FoldAssignment.model_validate_json(path.read_text(encoding="utf-8"))
