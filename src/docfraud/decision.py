"""The decision policy: accept a receipt or send it to review (ADR-0029, section 11).

A receipt is sent to review if its calibrated probability of forgery is at least the threshold
of a review budget, otherwise accepted. The threshold for a budget of 10 % is the 19th highest
calibrated probability of the 190 validation receipts: in general, the ``round(budget * n)``-th
highest of the receipts it is set on. Setting it reads no label. It is then applied unchanged,
so the share of receipts sent to review elsewhere may differ from the budget, and is reported.

There is no automatic rejection: rejecting a receipt is the reviewer's decision after reading
the audit report (ADR-0029, section 11, amending step 5 of ADR-0005).
"""

from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt

BUDGETS = (0.10, 0.20)
"""The review budget of the policy, and the alternative setting that is reported."""


class Decision(StrEnum):
    """What the policy does with a receipt."""

    ACCEPT = "accept"
    REVIEW = "review"


@dataclass(frozen=True, slots=True)
class ReviewPolicy:
    """Send to review every receipt whose calibrated probability reaches the threshold."""

    budget: float
    threshold: float

    @classmethod
    def set_on(cls, probabilities: npt.ArrayLike, budget: float = BUDGETS[0]) -> ReviewPolicy:
        """The threshold at which ``budget`` of these receipts would be reviewed; no label used.

        Raises:
            ValueError: If the budget would review no receipt or every receipt.
        """
        p = np.sort(np.asarray(probabilities, dtype=float))[::-1]
        reviewed = round(budget * p.size)
        if not 0 < reviewed < p.size:
            msg = f"a budget of {budget} reviews {reviewed} of {p.size} receipts"
            raise ValueError(msg)
        return cls(budget, float(p[reviewed - 1]))

    def decide(self, probability: float) -> Decision:
        """Review or accept one receipt."""
        return Decision.REVIEW if probability >= self.threshold else Decision.ACCEPT

    def reviewed(self, probabilities: npt.ArrayLike) -> npt.NDArray[np.bool_]:
        """Which receipts the policy sends to review."""
        return np.asarray(probabilities, dtype=float) >= self.threshold


@dataclass(frozen=True, slots=True)
class PolicyOutcome:
    """What the policy did on labelled receipts."""

    receipts: int
    reviewed: int
    forged: int
    forged_reviewed: int

    @property
    def review_share(self) -> float:
        """The share of receipts sent to review."""
        return self.reviewed / self.receipts

    @property
    def recall(self) -> float:
        """The share of forged receipts sent to review."""
        return self.forged_reviewed / self.forged if self.forged else 0.0

    @property
    def precision(self) -> float:
        """The share of reviewed receipts that are forged."""
        return self.forged_reviewed / self.reviewed if self.reviewed else 0.0


def outcome(
    policy: ReviewPolicy, probabilities: npt.ArrayLike, forged: npt.ArrayLike
) -> PolicyOutcome:
    """Count what the policy reviews among labelled receipts."""
    reviewed = policy.reviewed(probabilities)
    y = np.asarray(forged, dtype=bool)
    return PolicyOutcome(
        receipts=int(y.size),
        reviewed=int(reviewed.sum()),
        forged=int(y.sum()),
        forged_reviewed=int((reviewed & y).sum()),
    )
