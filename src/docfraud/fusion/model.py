"""The fusion model, its calibration and its checks (ADR-0028; ADR-0029, sections 6, 7 and 10).

- **Model.** scikit-learn's logistic regression with its default settings (L2 penalty, C = 1),
  nothing tuned, on inputs standardised with the mean and spread of the receipts it is fitted
  on. An input that does not vary there keeps a spread of 1, as scikit-learn's scaler does.
- **Out of fold.** On the training split, every receipt is scored by a model fitted on the other
  folds (:func:`out_of_fold`); validation and test receipts by the model fitted on the whole
  training split.
- **Calibration.** Platt scaling: a logistic regression of the labels on the model's log-odds,
  fitted on the validation split by maximum likelihood. It keeps the order of the scores, which
  ADR-0029 (section 1) requires; a fit that would reverse it is refused.
- **Contributions** (RQ3). The log-odds split exactly into the intercept and one term per input,
  the coefficient times the standardised value.
- **Calibration slope** (section 7 (a)): the slope of a logistic regression of the labels on the
  logit of out-of-fold probabilities; about 0.94 is expected, below 0.9 is overfitting.
"""

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

type Matrix = npt.NDArray[np.float64]
type Labels = npt.NDArray[np.bool_]
type Vector = npt.NDArray[np.float64]

_EDGE = 1e-12  # probabilities are kept this far from 0 and 1 before taking their logit


@dataclass(frozen=True, slots=True)
class FusionModel:
    """A logistic regression on standardised inputs, with its inputs' names."""

    names: tuple[str, ...]
    scaler: StandardScaler
    model: LogisticRegression

    @classmethod
    def fit(cls, x: Matrix, y: Labels, names: Sequence[str]) -> FusionModel:
        """Fit on these receipts: their mean and spread, then the regression.

        Raises:
            ValueError: If the columns do not match the names or one class is missing.
        """
        if x.ndim != 2 or x.shape[1] != len(names):  # noqa: PLR2004 - a matrix has two axes
            msg = f"{x.shape} inputs for the {len(names)} names {tuple(names)}"
            raise ValueError(msg)
        if np.unique(y).size != 2:  # noqa: PLR2004 - forged and authentic
            msg = "fitting needs forged and authentic receipts"
            raise ValueError(msg)
        scaler = StandardScaler().fit(x)
        model = LogisticRegression().fit(scaler.transform(x), y)
        return cls(tuple(names), scaler, model)

    @property
    def coefficients(self) -> dict[str, float]:
        """The weight of each standardised input on the log-odds of forgery."""
        return dict(zip(self.names, map(float, self.model.coef_[0]), strict=True))

    def log_odds(self, x: Matrix) -> Vector:
        """The log-odds of forgery of each receipt, before calibration."""
        return np.asarray(self.model.decision_function(self.scaler.transform(x)), dtype=float)

    def contributions(self, x: Matrix) -> Matrix:
        """Each input's term of the log-odds; with the intercept they sum to :meth:`log_odds`."""
        return np.asarray(self.scaler.transform(x) * self.model.coef_[0], dtype=float)

    @property
    def intercept(self) -> float:
        """The log-odds of a receipt whose every input lies at its mean."""
        return float(self.model.intercept_[0])


def out_of_fold(
    x: Matrix, y: Labels, fold_of: npt.NDArray[np.int_], names: Sequence[str]
) -> tuple[Vector, dict[int, FusionModel]]:
    """Log-odds of every receipt from the model fitted on the other folds, and those models."""
    log_odds = np.zeros(len(y))
    models: dict[int, FusionModel] = {}
    for fold in np.unique(fold_of):
        held = fold_of == fold
        models[int(fold)] = FusionModel.fit(x[~held], y[~held], names)
        log_odds[held] = models[int(fold)].log_odds(x[held])
    return log_odds, models


def _logit(p: Vector) -> Vector:
    q = np.clip(p, _EDGE, 1 - _EDGE)
    return np.asarray(np.log(q / (1 - q)), dtype=float)


class ReversedOrderError(ValueError):
    """Platt scaling would reverse the order of the scores."""


@dataclass(frozen=True, slots=True)
class Platt:
    """A logistic map from log-odds to calibrated probabilities."""

    slope: float
    offset: float

    @classmethod
    def fit(cls, log_odds: Vector, y: Labels) -> Platt:
        """Fit by maximum likelihood on these receipts (the validation split).

        Raises:
            ReversedOrderError: If the fitted slope is not positive, which would reverse the order.
        """
        model = LogisticRegression(C=np.inf).fit(log_odds[:, None], y)
        slope, offset = float(model.coef_[0][0]), float(model.intercept_[0])
        if slope <= 0:
            msg = f"Platt scaling would reverse the order of the scores (slope {slope:.3f})"
            raise ReversedOrderError(msg)
        return cls(slope, offset)

    def probabilities(self, log_odds: Vector) -> Vector:
        """Calibrated probabilities of forgery."""
        return np.asarray(1 / (1 + np.exp(-(self.slope * log_odds + self.offset))), dtype=float)


def calibration_slope(y: Labels, probabilities: Vector) -> float:
    """The slope of the labels' logistic regression on the logit of the probabilities."""
    model = LogisticRegression(C=np.inf).fit(_logit(probabilities)[:, None], y)
    return float(model.coef_[0][0])


def probabilities(log_odds: Vector) -> Vector:
    """The model's own probabilities, before calibration."""
    return np.asarray(1 / (1 + np.exp(-log_odds)), dtype=float)
