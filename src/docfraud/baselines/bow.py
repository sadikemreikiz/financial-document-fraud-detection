"""The published bag-of-words + logistic-regression baseline (ADR-0011).

Martínez Tornés et al. (ICDAR 2023) describe it only as word counts with logistic regression, so
the most literal configuration is used: scikit-learn defaults, trained on the training split,
forged when the predicted probability exceeds 0.5. It is reproduced to check our splits, labels
and metric against the paper, and measured in cross-validation as the text baseline of Phase 3.
"""

import warnings
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.exceptions import ConvergenceWarning
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from docfraud.data.models import Document, Label
from docfraud.data.normalize import read_transcript
from docfraud.evaluation.folds import FoldAssignment

THRESHOLD = 0.5
DEFAULT_MAX_ITER = 100
RAISED_MAX_ITER = 1_000

# Table 4 of the paper: 218 test receipts, 35 forged.
PUBLISHED = {"precision": 0.4000, "recall": 0.1143, "f1": 0.1778, "tp": 4, "fp": 6, "fn": 31}


@dataclass(frozen=True, slots=True)
class Fitted:
    """A trained baseline and the iteration limit it needed to converge."""

    model: Pipeline
    max_iter: int

    def scores(self, documents: Sequence[Document]) -> npt.NDArray[np.float64]:
        """Probability of forgery for each document.

        Raises:
            ValueError: If a probability is exactly the threshold, where ``>= 0.5`` and the
                classifier's own ``> 0.5`` would disagree.
        """
        probabilities = np.asarray(self.model.predict_proba(texts(documents))[:, 1], dtype=float)
        if np.any(probabilities == THRESHOLD):
            msg = "a probability equals the decision threshold exactly"
            raise ValueError(msg)
        return probabilities


def texts(documents: Sequence[Document]) -> list[str]:
    """Normalised transcripts (track T1).

    Raises:
        ValueError: If a document has no transcript.
    """
    missing = [d.doc_id for d in documents if d.transcript_path is None]
    if missing:
        msg = f"documents without transcripts: {missing}"
        raise ValueError(msg)
    return [read_transcript(d.transcript_path) for d in documents if d.transcript_path]


def labels(documents: Sequence[Document]) -> npt.NDArray[np.bool_]:
    """True for forged documents.

    Raises:
        ValueError: If a document has no label.
    """
    if any(d.label is None for d in documents):
        msg = "every document needs a label"
        raise ValueError(msg)
    return np.array([d.label is Label.FORGED for d in documents], dtype=bool)


def fit(documents: Sequence[Document]) -> Fitted:
    """Train with scikit-learn defaults; raise the iteration limit once if lbfgs did not converge.

    Raises:
        RuntimeError: If the classifier does not converge with the raised limit either.
    """
    x, y = texts(documents), labels(documents)
    for max_iter in (DEFAULT_MAX_ITER, RAISED_MAX_ITER):
        model = Pipeline(
            [("counts", CountVectorizer()), ("classifier", LogisticRegression(max_iter=max_iter))]
        )
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            model.fit(x, y)
        if not any(issubclass(w.category, ConvergenceWarning) for w in caught):
            return Fitted(model=model, max_iter=max_iter)
    msg = f"logistic regression did not converge within {RAISED_MAX_ITER} iterations"
    raise RuntimeError(msg)


@dataclass(frozen=True, slots=True)
class CrossValidated:
    """Out-of-fold probabilities for every training document."""

    doc_ids: tuple[str, ...]
    labels: npt.NDArray[np.bool_]
    scores: npt.NDArray[np.float64]
    max_iter: tuple[int, ...]  # per fold


def cross_validate(documents: Sequence[Document], folds: FoldAssignment) -> CrossValidated:
    """Train on four folds and score the fifth, for each fold in turn.

    Raises:
        ValueError: If a document has no fold.
    """
    train = sorted(
        (d for d in documents if d.split is folds.split and d.dataset == folds.dataset),
        key=lambda d: d.doc_id,
    )
    unassigned = [d.doc_id for d in train if d.doc_id not in folds.folds]
    if unassigned:
        msg = f"documents without a fold: {unassigned}"
        raise ValueError(msg)
    fold_of = np.array([folds.folds[d.doc_id] for d in train])
    scores = np.zeros(len(train))
    iterations = []
    for fold in range(folds.n_folds):
        held_out = np.flatnonzero(fold_of == fold)
        fitted = fit([train[i] for i in np.flatnonzero(fold_of != fold)])
        scores[held_out] = fitted.scores([train[i] for i in held_out])
        iterations.append(fitted.max_iter)
    return CrossValidated(
        doc_ids=tuple(d.doc_id for d in train),
        labels=labels(train),
        scores=scores,
        max_iter=tuple(iterations),
    )
