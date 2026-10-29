"""The demo's cards: the system's outputs for the receipts shown (ADR-0035, 1 and 3).

The ``demo-cards`` run record keeps, for each receipt the rule of :mod:`docfraud.demo.gallery`
shows, its calibrated probability, the decision at every budget, the evidence with its
contributions and regions, and the template audit report. It keeps nothing from the dataset but
the outcome of each receipt shown, the SHA-256 of its image and counts over the test split: the
images and annotations are read when the page is built.
"""

from collections.abc import Callable, Mapping, Sequence
from typing import Any

from docfraud.data.models import BBox, Document, Label
from docfraud.decision import Decision, ReviewPolicy
from docfraud.demo.gallery import choose, outcomes
from docfraud.reporting.assessment import Assessment, Evidence
from docfraud.reporting.text import render_report

RUN_NAME = "demo-cards"

TOWARDS = 3
"""How many pieces of evidence towards forgery a card lists before its full report."""


def budget_name(policy: ReviewPolicy) -> str:
    """A policy's budget as reports name it, such as ``10 %``."""
    return f"{policy.budget * 100:.0f} %"


def box(found: BBox) -> list[int]:
    """A box as the record keeps it: x, y, width and height in pixels."""
    return [found.x, found.y, found.width, found.height]


def evidence(item: Evidence) -> dict[str, Any]:
    """One piece of evidence: its inputs, contribution, findings' reasons and regions."""
    return {
        "inputs": list(item.inputs),
        "contribution": item.contribution,
        "reasons": [finding.reason for finding in item.findings],
        "boxes": [box(b) for b in item.boxes],
    }


def card(found: Assessment, policies: Sequence[ReviewPolicy]) -> dict[str, Any]:
    """What a card shows of one assessment, taken from it without editing."""
    towards = [e for e in found.evidence if e.contribution > 0]
    # evidence is ordered from the strongest push towards forgery to the strongest against
    against = [e for e in reversed(found.evidence) if e.contribution <= 0]
    return {
        "probability": found.probability,
        "decisions": {budget_name(p): p.decide(found.probability).value for p in policies},
        "towards": [evidence(e) for e in towards[:TOWARDS]],
        "against": [evidence(e) for e in against[:1]],
        # the boxes the audit report's overlay draws: every located finding pushing towards forgery
        "overlay": [
            {"detector": finding.detector, "boxes": [box(b) for b in finding.boxes]}
            for item in towards
            for finding in item.findings
            if finding.boxes
        ],
        "first_region": [box(b) for b in found.first_region()],
        "report": render_report(found),
    }


def _marked(document: Document) -> bool:
    return document.handwritten_annotation or document.digital_annotation


def alarms(
    probabilities: Mapping[str, float],
    documents: Mapping[str, Document],
    policy: ReviewPolicy,
) -> dict[str, int]:
    """The genuine receipts a policy sends to review, and how many carry a recorded mark.

    The marks are the dataset's flags: handwriting, highlighting and stamps count as handwritten;
    a digitally added note counts as digital only when the receipt has no handwritten mark.
    """
    genuine = [r for r in probabilities if documents[r].label is Label.AUTHENTIC]
    flagged = [r for r in genuine if policy.decide(probabilities[r]) is Decision.REVIEW]
    return {
        "genuine": len(genuine),
        "genuine_with_marks": sum(_marked(documents[r]) for r in genuine),
        "false_alarms": len(flagged),
        "with_marks": sum(_marked(documents[r]) for r in flagged),
        "handwritten": sum(documents[r].handwritten_annotation for r in flagged),
        "digital_only": sum(
            documents[r].digital_annotation and not documents[r].handwritten_annotation
            for r in flagged
        ),
    }


def demo_cards(
    assessments: Mapping[str, Assessment],
    documents: Sequence[Document],
    policies: Sequence[ReviewPolicy],
    digest: Callable[[Document], str],
) -> dict[str, Any]:
    """The results of the ``demo-cards`` run.

    Args:
        assessments: Every test receipt's assessment of track T2.
        documents: The test receipts, for their labels and recorded marks.
        policies: The review policies a card states; the first decides the outcomes.
        digest: The SHA-256 of a receipt's image file.

    Raises:
        ValueError: If no policy is given, or a receipt has no assessment.
    """
    if not policies:
        msg = "the demo needs the review policy that decides the outcomes"
        raise ValueError(msg)
    by_id = {d.doc_id: d for d in documents}
    if set(by_id) != set(assessments):
        msg = "every test receipt needs an assessment, and only those"
        raise ValueError(msg)
    probabilities = {r: a.probability for r, a in assessments.items()}
    forged = {r: by_id[r].label is Label.FORGED for r in assessments}
    deciding = policies[0]
    reviewed = {r: deciding.decide(p) is Decision.REVIEW for r, p in probabilities.items()}
    ranked = outcomes(probabilities, forged, reviewed)
    gallery = {
        name: [
            {
                "receipt": ranked[name][rank],
                "rank": rank,
                "probability": probabilities[ranked[name][rank]],
                "sha256": digest(by_id[ranked[name][rank]]),
                "card": card(assessments[ranked[name][rank]], policies),
            }
            for rank in ranks
        ]
        for name, ranks in choose(ranked).items()
    }
    return {
        "policies": [{"budget": p.budget, "threshold": p.threshold} for p in policies],
        "receipts": len(assessments),
        "forged": sum(forged.values()),
        "outcomes": {name: len(members) for name, members in ranked.items()},
        "gallery": gallery,
        "alarms": {budget_name(p): alarms(probabilities, by_id, p) for p in policies},
    }
