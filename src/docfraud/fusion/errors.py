"""Phase 7's error analysis: what the system decides wrongly, and why (ADR-0034, decision 8).

It looks at what the system decides: the review policy of T2 at the 10 % budget, with the fusion
model, Platt scaling and threshold of Phase 5's check applied unchanged. Its errors are the
forged receipts it accepts (misses) and the authentic receipts it sends to review (false alarms).

Each error gets a cause computed from the system's own outputs, by the first rule that applies,
so that no cause is chosen after the error is seen:

- a miss: *nothing found*, no input pushes towards forgery; *found at the edit, too weak*, the
  audit report's first region shares a pixel with an annotated modified rectangle or copy
  source, but the probability stays under the threshold; *found elsewhere*, evidence pushes
  towards forgery but the first region misses the edit, has no location, or the receipt has no
  annotation to judge it by;
- a false alarm: the group of the input that pushes most towards forgery.

Beside its cause, each error shows what the annotations say of a miss (the edited entities, the
techniques and the editing software), whether the dataset marks handwritten or digital
annotations on a false alarm, its probability and its largest contributions. One sentence per
error, drafted from these and the audit report, is added by hand and checked by the owner.
"""

from collections import Counter
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from docfraud.data.models import Document, Label, RegionRole
from docfraud.decision import BUDGETS, Decision, ReviewPolicy
from docfraud.evaluation.localisation import annotated_rectangles, hits
from docfraud.fusion.development_report import TRACK_NAMES
from docfraud.fusion.final import frozen_row
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
from docfraud.fusion.training_inputs import ReceiptInputs
from docfraud.fusion.validation import FUSION_ROW
from docfraud.reporting.assessment import Assessment, Evidence, assess

RUN_NAME = "fusion-test-errors"
TRACK = "ocr"
"""The track whose decisions are analysed: T2, what the system does from the image alone."""

NOTHING_FOUND = "nothing found"
TOO_WEAK = "found at the edit, too weak"
ELSEWHERE = "found elsewhere"
NOTHING_PUSHES = "no input pushes towards forgery"

GROUP_OF: dict[str, str] = {
    RULES_INPUT: "accounting rules",
    SYSTEM_2_INPUT: "pixel evidence",
    SYSTEM_1_INPUT: "pixel evidence",
    SYSTEM_1_EVIDENCE: "pixel evidence",
    COPY_MOVE_INPUT: "copy-move",
    LINK_1: "links",
    LINK_2: "links",
    METADATA_INPUT: "file metadata",
}
"""The group of each fusion input, as ADR-0034 (decision 8) names them."""

LARGEST = 3
"""How many of an error's largest contributions it shows."""

type Labels = npt.NDArray[np.bool_]


def frozen_assessments(
    training: Sequence[ReceiptInputs],
    training_labels: Labels,
    rows: Sequence[ReceiptInputs],
    check: Mapping[str, Any],
) -> tuple[ReviewPolicy, dict[str, Assessment]]:
    """Every receipt's T2 assessment, with what Phase 5's check recorded applied unchanged.

    Raises:
        ValueError: If the fusion model fitted again is not the recorded one.
    """
    fused = frozen_row(
        training, training_labels, rows, FUSION_ROW, check[TRACK]["fitted"]["fusion"]
    )
    stated = check["policy"][TRACK][f"{BUDGETS[0] * 100:.0f} %"]
    policy = ReviewPolicy(stated["budget"], stated["threshold"])
    return policy, {
        row.receipt: assess(
            row.receipt,
            TRACK_NAMES[TRACK],
            {name: row.inputs[name] for name in fused.model.names},
            row.findings,
            model=fused.model,
            platt=fused.platt,
            policy=policy,
        )
        for row in rows
    }


def _group(evidence: Evidence) -> str:
    return GROUP_OF[evidence.inputs[0]]


def miss_cause(assessment: Assessment, document: Document) -> str:
    """Why a forged receipt was accepted, by the first rule that applies."""
    if not any(e.contribution > 0 for e in assessment.evidence):
        return NOTHING_FOUND
    modified, sources = annotated_rectangles(document)
    region = assessment.first_region()
    if region and any(hits(box, [*modified, *sources]) for box in region):
        return TOO_WEAK
    return ELSEWHERE


def false_alarm_cause(assessment: Assessment) -> str:
    """Why an authentic receipt was sent to review: the group pushing most towards forgery."""
    pushing = [e for e in assessment.evidence if e.contribution > 0]
    if not pushing:
        return NOTHING_PUSHES
    return _group(max(pushing, key=lambda e: e.contribution))


def _contributions(assessment: Assessment) -> list[dict[str, Any]]:
    ordered = sorted(assessment.evidence, key=lambda e: e.contribution, reverse=True)
    return [
        {"group": _group(e), "inputs": list(e.inputs), "contribution": e.contribution}
        for e in ordered[:LARGEST]
    ]


def error(assessment: Assessment, document: Document) -> dict[str, Any] | None:
    """One receipt's error, or ``None`` if the system decided it rightly."""
    forged = document.label is Label.FORGED
    reviewed = assessment.decision is Decision.REVIEW
    if forged == reviewed:
        return None
    found: dict[str, Any] = {
        "receipt": document.doc_id,
        "kind": "miss" if forged else "false alarm",
        "probability": assessment.probability,
        "largest_contributions": _contributions(assessment),
    }
    if forged:
        modified = [r for r in document.regions if r.role is RegionRole.MODIFIED]
        found |= {
            "cause": miss_cause(assessment, document),
            "annotated": bool(document.regions),
            "entities": sorted({r.entity.value for r in modified if r.entity is not None}),
            "techniques": sorted({t.value for r in modified for t in r.techniques}),
            "software": document.editing_software,
        }
    else:
        found |= {
            "cause": false_alarm_cause(assessment),
            "handwritten_annotation": document.handwritten_annotation,
            "digital_annotation": document.digital_annotation,
        }
    return found


def error_analysis(
    documents: Sequence[Document], assessments: Mapping[str, Assessment], policy: ReviewPolicy
) -> dict[str, Any]:
    """Every error of the policy on these receipts, with counts by cause and by kind of edit."""
    errors = [found for d in documents if (found := error(assessments[d.doc_id], d)) is not None]
    misses = [e for e in errors if e["kind"] == "miss"]
    alarms = [e for e in errors if e["kind"] == "false alarm"]
    return {
        "track": TRACK,
        "policy": {"budget": policy.budget, "threshold": policy.threshold},
        "receipts": len(documents),
        "forged": sum(d.label is Label.FORGED for d in documents),
        "reviewed": sum(a.decision is Decision.REVIEW for a in assessments.values()),
        "misses": {
            "count": len(misses),
            "by_cause": dict(Counter(e["cause"] for e in misses)),
            "by_entity": dict(Counter(x for e in misses for x in e["entities"])),
            "by_technique": dict(Counter(x for e in misses for x in e["techniques"])),
        },
        "false_alarms": {
            "count": len(alarms),
            "by_cause": dict(Counter(e["cause"] for e in alarms)),
            "with_benign_marks": sum(
                e["handwritten_annotation"] or e["digital_annotation"] for e in alarms
            ),
        },
        "errors": errors,
    }
