"""What the system concludes about one receipt, and the evidence behind it (ADR-0029, 10-12).

An assessment holds the calibrated probability of forgery, the decision of the review policy and
the evidence ordered by how much it moves the fusion model's log-odds. Contributions belong to
the model's inputs, so evidence is grouped by input: the findings of that input (every failing
rule for the rules' score; the top region of system 1 for its score and evidence together; the
links of each kind) with the input's contribution. Each input also has a finding that states its
score, so every number a report gives is a finding's.

The report's first region (RQ4) is the location of the evidence with the largest positive
contribution among the evidence that has one; a location of several boxes, such as the values
of failing rules or a copy-move pair, hits if any of its boxes hits (ADR-0030).
"""

from collections.abc import Mapping, Sequence

import numpy as np
from pydantic import BaseModel, ConfigDict

from docfraud.data.models import BBox
from docfraud.decision import Decision, ReviewPolicy
from docfraud.detectors.finding import (
    COPY_MOVE,
    LINKING,
    METADATA,
    RULES,
    SYSTEM_1,
    SYSTEM_2,
    Finding,
)
from docfraud.fusion.inputs import (
    COPY_MOVE_INPUT,
    LINK_1,
    LINK_2,
    METADATA_INPUT,
    RULES_INPUT,
    SYSTEM_1_EVIDENCE,
    SYSTEM_1_INPUT,
    SYSTEM_2_INPUT,
)
from docfraud.fusion.model import FusionModel, Platt

GROUPS: dict[tuple[str, ...], tuple[str, str | None]] = {
    (RULES_INPUT,): (RULES, None),
    (SYSTEM_2_INPUT,): (SYSTEM_2, None),
    (SYSTEM_1_INPUT, SYSTEM_1_EVIDENCE): (SYSTEM_1, None),
    (COPY_MOVE_INPUT,): (COPY_MOVE, None),
    (LINK_1,): (LINKING, "region on a read value"),
    (LINK_2,): (LINKING, "region on a value a failing rule reads"),
    (METADATA_INPUT,): (METADATA, None),
}
"""Which inputs form one piece of evidence, and the detector (and kind) of its findings."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Evidence(_Frozen):
    """The findings behind one input of the fusion model, with that input's contribution."""

    inputs: tuple[str, ...]
    contribution: float  # the input's term of the log-odds of forgery
    findings: tuple[Finding, ...]

    @property
    def boxes(self) -> tuple[BBox, ...]:
        """Where the evidence lies: every box of its findings, without repeats."""
        return tuple(dict.fromkeys(b for f in self.findings for b in f.boxes))


class Assessment(_Frozen):
    """The system's conclusion on one receipt of one track."""

    receipt: str
    track: str
    probability: float  # calibrated
    decision: Decision
    budget: float
    threshold: float
    evidence: tuple[Evidence, ...]  # the strongest push towards forgery first

    def first_region(self) -> tuple[BBox, ...]:
        """The boxes of the evidence with the largest positive contribution that has a location.

        Empty if no evidence with a positive contribution has one; RQ4 counts that as a miss.
        """
        return first_region(self.evidence)


def first_region(evidence: Sequence[Evidence]) -> tuple[BBox, ...]:
    """The boxes of the first evidence, in order, that pushes towards forgery and has a location.

    Empty if there is none.
    """
    for item in evidence:
        if item.contribution > 0 and item.boxes:
            return item.boxes
    return ()


def score_finding(inputs: Sequence[str], values: Mapping[str, float]) -> Finding:
    """A finding that states the values of an input group."""
    stated = "; ".join(f"{name} {values[name]:.3f}" for name in inputs)
    return Finding(
        detector="fusion input",
        kind="input value",
        score=values[inputs[0]],
        reason=f"input values: {stated}",
        details={name: values[name] for name in inputs},
    )


def assess(  # noqa: PLR0913 - the receipt, its inputs and findings, and the fitted system
    receipt: str,
    track: str,
    inputs: Mapping[str, float],
    findings: Sequence[Finding],
    *,
    model: FusionModel,
    platt: Platt,
    policy: ReviewPolicy,
) -> Assessment:
    """Score one receipt and gather its evidence, ordered by contribution.

    Raises:
        ValueError: If the inputs are not the model's.
    """
    evidence = gather_evidence(inputs, findings, model)  # checks that the inputs are the model's
    x = np.array([[inputs[name] for name in model.names]], dtype=float)
    probability = float(platt.probabilities(model.log_odds(x))[0])
    return Assessment(
        receipt=receipt,
        track=track,
        probability=probability,
        decision=policy.decide(probability),
        budget=policy.budget,
        threshold=policy.threshold,
        evidence=evidence,
    )


def gather_evidence(
    inputs: Mapping[str, float], findings: Sequence[Finding], model: FusionModel
) -> tuple[Evidence, ...]:
    """A receipt's evidence grouped by the model's inputs, the strongest push towards forgery first.

    It needs no calibration, so it also serves the out-of-fold models of the training split.

    Raises:
        ValueError: If the inputs are not the model's.
    """
    if list(inputs) != list(model.names):
        msg = f"inputs {tuple(inputs)} are not the model's {model.names}"
        raise ValueError(msg)
    x = np.array([[inputs[name] for name in model.names]], dtype=float)
    terms = dict(zip(model.names, model.contributions(x)[0], strict=True))
    evidence = []
    for group, (detector, kind) in GROUPS.items():
        if not set(group) <= set(model.names):
            continue
        own = tuple(
            f for f in findings if f.detector == detector and (kind is None or f.kind == kind)
        )
        evidence.append(
            Evidence(
                inputs=group,
                contribution=float(sum(terms[name] for name in group)),
                findings=(score_finding(group, inputs), *own),
            )
        )
    evidence.sort(key=lambda e: -e.contribution)
    return tuple(evidence)
