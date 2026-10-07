"""Measuring the accounting rules on the training split and, once, on the validation split.

The procedure is fixed in ADR-0012: rule outcomes are scored by a logistic regression fitted in
the stored folds; the operating threshold maximises F1 on the out-of-fold scores; "any rule
fails" is reported alongside. On the validation split the scorer is fitted on the whole training
split and applied with the threshold found in cross-validation.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from docfraud.detectors.rules import RULES, Outcome, Status
from docfraud.detectors.scoring import Scorer, any_fail, features, out_of_fold
from docfraud.evaluation.detection import choose_threshold, summary

type Outcomes = Sequence[Mapping[str, Outcome]]


def flag_all(y: npt.NDArray[np.bool_], doc_ids: Sequence[str]) -> dict[str, Any]:
    """The detector that flags every receipt: the floor that any F1 has to be read against.

    With one forged receipt in six, flagging everything already reaches F1 = 2p / (1 + p), about
    0.29, with no information at all.
    """
    return summary(y, np.ones(len(y)), 1.0, doc_ids)


def rule_counts(outcomes: Outcomes, y: npt.NDArray[np.bool_]) -> dict[str, dict[str, Any]]:
    """Per rule and label: on how many receipts it applied and on how many it failed."""
    counts: dict[str, dict[str, Any]] = {}
    for rule in RULES:
        counts[rule] = {}
        for label, keep in (("authentic", False), ("forged", True)):
            group = [o[rule] for o, forged in zip(outcomes, y, strict=True) if forged == keep]
            counts[rule][label] = {
                "applicable": sum(x.status is not Status.NOT_APPLICABLE for x in group),
                "fail": sum(x.status is Status.FAIL for x in group),
            }
    return counts


def cross_validated(
    outcomes: Outcomes,
    y: npt.NDArray[np.bool_],
    fold_of: npt.NDArray[np.int_],
    doc_ids: Sequence[str],
) -> dict[str, Any]:
    """Out-of-fold scores, the operating threshold, metrics and the weights of a full fit."""
    x = features(outcomes)
    oof = out_of_fold(x, y, fold_of)
    threshold = choose_threshold(y, oof)
    return {
        "threshold": threshold,
        "learned": summary(y, oof, threshold, doc_ids, probabilities=True),
        "any_fail": summary(y, any_fail(outcomes), 1.0, doc_ids),
        "flag_all": flag_all(y, doc_ids),
        "weights": Scorer.fit(x, y).weights(),
        "rules": rule_counts(outcomes, y),
    }


def held_out(  # noqa: PLR0913, PLR0917 - training and held-out data with outcomes and labels
    train_outcomes: Outcomes,
    train_y: npt.NDArray[np.bool_],
    fold_of: npt.NDArray[np.int_],
    outcomes: Outcomes,
    y: npt.NDArray[np.bool_],
    doc_ids: Sequence[str],
) -> dict[str, Any]:
    """Fit on the whole training split, threshold from its cross-validation, score held-out data."""
    x_train = features(train_outcomes)
    threshold = choose_threshold(train_y, out_of_fold(x_train, train_y, fold_of))
    scorer = Scorer.fit(x_train, train_y)
    return {
        "threshold": threshold,
        "learned": summary(
            y, scorer.scores(features(outcomes)), threshold, doc_ids, probabilities=True
        ),
        "any_fail": summary(y, any_fail(outcomes), 1.0, doc_ids),
        "flag_all": flag_all(y, doc_ids),
        "weights": scorer.weights(),
        "rules": rule_counts(outcomes, y),
    }
