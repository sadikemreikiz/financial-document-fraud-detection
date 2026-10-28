"""Which receipts the agent develops on, and what the system concluded about them (ADR-0031, 5).

Development uses design-fold receipts of the training split: those that the out-of-fold fusion
model of track T2 ranks among the highest 10 % of the training split, all of them forged, and
the :data:`AUTHENTIC_CASES` highest-ranked authentic ones, so that the agent also meets
receipts that are not forged. The pilot takes three of them: the two forged receipts and the
authentic receipt ranked highest.

On the training split each receipt is scored by the fusion model of the fold that held it out,
without Platt scaling, which is fitted only on the validation split; the review policy is set
on those out-of-fold probabilities at the first budget. This is how Phase 5's training report
measures RQ4, so a development case's assessment is the one the system made of it out of fold.
"""

from collections.abc import Collection, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from docfraud.decision import BUDGETS, ReviewPolicy
from docfraud.fusion.development import fold_models, in_budget, score
from docfraud.fusion.development_report import TRACK_NAMES
from docfraud.fusion.training_inputs import ReceiptInputs
from docfraud.fusion.validation import FUSION_ROW, IDENTITY
from docfraud.reporting.assessment import Assessment, assess

AUTHENTIC_CASES = 3
PILOT_FORGED, PILOT_AUTHENTIC = 2, 1
TRACK = "ocr"
"""The agent runs on track T2 only (ADR-0031, section 5)."""


@dataclass(frozen=True, slots=True)
class Development:
    """The development cases, each kind ranked by its out-of-fold probability, highest first."""

    forged: tuple[str, ...]
    authentic: tuple[str, ...]

    @property
    def receipts(self) -> tuple[str, ...]:
        """Every development case, the forged first."""
        return self.forged + self.authentic

    @property
    def pilot(self) -> tuple[str, ...]:
        """The pilot's cases: the forged and the authentic ones ranked highest."""
        return self.forged[:PILOT_FORGED] + self.authentic[:PILOT_AUTHENTIC]


def development_cases(
    receipts: Sequence[str],
    forged: npt.NDArray[np.bool_],
    probabilities: npt.NDArray[np.float64],
    design_fold: Collection[str],
) -> Development:
    """The development cases among the training receipts.

    Raises:
        ValueError: If a design-fold receipt among the highest 10 % is not forged.
    """
    top = np.array(in_budget(receipts, probabilities))
    design = np.array([r in design_fold for r in receipts])
    order = np.argsort(-probabilities, kind="stable")
    chosen = [i for i in order if top[i] and design[i]]
    if not all(forged[i] for i in chosen):
        msg = "a design-fold receipt among the highest 10 % is authentic"
        raise ValueError(msg)
    authentic = [i for i in order if design[i] and not forged[i] and not top[i]]
    return Development(
        tuple(receipts[i] for i in chosen),
        tuple(receipts[i] for i in authentic[:AUTHENTIC_CASES]),
    )


@dataclass(frozen=True, slots=True)
class OutOfFold:
    """Track T2's fusion on the training split, out of fold, with its review policy."""

    probabilities: npt.NDArray[np.float64]
    assessments: dict[str, Assessment]


def out_of_fold(
    rows: Sequence[ReceiptInputs], forged: npt.NDArray[np.bool_], fold_of: npt.NDArray[np.int_]
) -> OutOfFold:
    """Every training receipt's assessment by the fusion model of the fold that held it out."""
    scored = score(rows, FUSION_ROW, forged, fold_of)
    policy = ReviewPolicy.set_on(scored.probabilities, BUDGETS[0])
    assessments = {
        row.receipt: assess(
            row.receipt,
            TRACK_NAMES[TRACK],
            {name: row.inputs[name] for name in model.names},
            row.findings,
            model=model,
            platt=IDENTITY,
            policy=policy,
        )
        for row, model in zip(rows, fold_models(scored, fold_of), strict=True)
    }
    return OutOfFold(scored.probabilities, assessments)
