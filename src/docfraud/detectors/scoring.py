"""From rule outcomes to a forgery score (ADR-0012).

Each rule contributes two binary features per receipt: it failed, or it could not be applied.
Passing is the reference. A logistic regression with scikit-learn's default settings learns
from the training split how much each outcome says about forgery; nothing is tuned. Its
coefficients are part of the explanation: they state how much a failing rule raises the score.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression

from docfraud.detectors.rules import RULES, Outcome, Status

type Matrix = npt.NDArray[np.float64]

FEATURES = tuple(f"{rule}:{status}" for rule in RULES for status in ("fail", "not_applicable"))


def features(outcomes: Sequence[Mapping[str, Outcome]]) -> Matrix:
    """One row per receipt, one column per rule and non-passing outcome (``FEATURES``)."""
    return np.array(
        [
            [
                float(o[rule].status is status)
                for rule in RULES
                for status in (Status.FAIL, Status.NOT_APPLICABLE)
            ]
            for o in outcomes
        ],
        dtype=float,
    ).reshape(len(outcomes), len(FEATURES))


def any_fail(outcomes: Sequence[Mapping[str, Outcome]]) -> npt.NDArray[np.float64]:
    """1 when at least one rule fails, else 0: the reference score without learned parameters."""
    return np.array(
        [float(any(x.status is Status.FAIL for x in o.values())) for o in outcomes], dtype=float
    )


@dataclass(frozen=True, slots=True)
class Scorer:
    """A fitted logistic regression over rule outcomes."""

    model: LogisticRegression

    @classmethod
    def fit(cls, x: Matrix, y: npt.NDArray[np.bool_]) -> Scorer:
        """Fit with scikit-learn's default settings."""
        return cls(LogisticRegression().fit(x, y))

    def scores(self, x: Matrix) -> npt.NDArray[np.float64]:
        """Probability of forgery per receipt."""
        return np.asarray(self.model.predict_proba(x)[:, 1], dtype=float)

    def weights(self) -> dict[str, float]:
        """Coefficient of each feature: how much it raises (or lowers) the log-odds of forgery."""
        return dict(zip(FEATURES, map(float, self.model.coef_[0]), strict=True))


def out_of_fold(
    x: Matrix, y: npt.NDArray[np.bool_], fold_of: npt.NDArray[np.int_]
) -> npt.NDArray[np.float64]:
    """Score each receipt with a scorer fitted on the other folds."""
    scores = np.zeros(len(y))
    for fold in np.unique(fold_of):
        held_out = fold_of == fold
        scores[held_out] = Scorer.fit(x[~held_out], y[~held_out]).scores(x[held_out])
    return scores
