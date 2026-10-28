"""The frozen agent's run on the validation split, measured once (ADR-0031, sections 5 and 6).

The agent runs only on the receipts that the review policy of track T2 sends to review. They are
the receipts of Phase 5's validation check: its fusion model is fitted again on the whole
training split, checked against the probabilities the check recorded, and its policy set on
them at the first budget, so that each case's assessment is the audit report the system makes
of it.

Each case file is measured as ADR-0031 (section 6) fixes:

- **grounding**: every number and box of the rendered case file looked up in what the agent
  received, with every violation listed;
- **cost and latency** per case, median and 95th percentile;
- **where it points**, descriptive: on the forged receipts with region annotations, whether the
  location of the first observation hits an annotated modified rectangle or copy source,
  against the random baseline of ADR-0015 and ADR-0030, next to the audit report's first
  region on the same receipts.
"""

import math
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from docfraud.data.models import BBox, Document, Label
from docfraud.decision import BUDGETS, ReviewPolicy
from docfraud.evaluation.bootstrap import DEFAULT_RESAMPLES
from docfraud.evaluation.localisation import (
    annotated_rectangles,
    any_box_baseline,
    localisation_guard,
    receipt_rng,
)
from docfraud.fusion.development_report import TRACK_NAMES
from docfraud.fusion.training_inputs import ReceiptInputs
from docfraud.fusion.validation import FUSION_ROW, applied, check_row
from docfraud.llm.agent import Investigation
from docfraud.llm.agent_cases import TRACK
from docfraud.llm.case_file import CaseFile, first_location
from docfraud.reporting.assessment import Assessment, assess
from docfraud.reporting.localisation import FirstRegion, first_region_hit

RUN_NAME = "llm-agent-validation"

type Labels = npt.NDArray[np.bool_]
type Page = tuple[BBox, Sequence[BBox]]
"""A page and its OCR word boxes, on which the random baseline places its boxes."""

TOLERANCE = 1e-9
"""How far the refitted fusion model's probabilities may lie from the recorded ones."""


@dataclass(frozen=True, slots=True)
class Queue:
    """The receipts the review policy sends to review, with their assessments."""

    policy: ReviewPolicy
    assessments: dict[str, Assessment]
    """By receipt, in the order of the validation table."""


def review_queue(
    training: Sequence[ReceiptInputs],
    training_labels: Labels,
    validation: Sequence[ReceiptInputs],
    labels: Labels,
    check: Mapping[str, Any],
) -> Queue:
    """Track T2's review queue on the validation split, as Phase 5's check left it.

    ``check`` holds the results of that check: the fitted fusion model and the policy, which
    the rebuilt ones must reproduce.

    Raises:
        ValueError: If the rebuilt probabilities, threshold or queue differ from the recorded.
    """
    checked = check_row(training, training_labels, validation, labels, FUSION_ROW)
    recorded = applied(check[TRACK]["fitted"][FUSION_ROW.name], validation, FUSION_ROW)
    if not np.allclose(checked.probabilities, recorded, rtol=0, atol=TOLERANCE):
        msg = "the refitted fusion model does not give the probabilities the check recorded"
        raise ValueError(msg)
    policy = ReviewPolicy.set_on(checked.probabilities, BUDGETS[0])
    stated = check["policy"][TRACK][f"{BUDGETS[0] * 100:.0f} %"]
    reviewed = policy.reviewed(checked.probabilities)
    if not math.isclose(policy.threshold, stated["threshold"], rel_tol=0, abs_tol=TOLERANCE) or (
        int(reviewed.sum()) != stated["reviewed"]
    ):
        msg = "the review policy is not the one the check recorded"
        raise ValueError(msg)
    assessments = {
        row.receipt: assess(
            row.receipt,
            TRACK_NAMES[TRACK],
            {name: row.inputs[name] for name in checked.model.names},
            row.findings,
            model=checked.model,
            platt=checked.platt,
            policy=policy,
        )
        for row, sent in zip(validation, reviewed, strict=True)
        if sent
    }
    return Queue(policy, assessments)


def first_location_hit(
    case_file: CaseFile | None, document: Document, page: BBox, word_boxes: Sequence[BBox]
) -> FirstRegion:
    """Judge the location of a case file's first observation against the annotated edits.

    As for the audit report's first region: a case file without one, or without a located first
    observation, is a miss with a baseline of 0.
    """
    location = first_location(case_file) if case_file is not None else None
    if location is None:
        return FirstRegion(located=False, hit=False, baseline=0.0)
    modified, sources = annotated_rectangles(document)
    targets = [*modified, *sources]
    hit = any(location.overlaps(t) for t in targets)
    rng = receipt_rng(document.doc_id)
    return FirstRegion(True, hit, any_box_baseline((location,), page, word_boxes, targets, rng))


def _pointing(found: FirstRegion) -> dict[str, Any]:
    return {"located": found.located, "hit": found.hit, "baseline": found.baseline}


def measured(
    investigation: Investigation,
    document: Document,
    assessment: Assessment,
    page: Page | None,
) -> dict[str, Any]:
    """One case of the run: its form, grounding, cost, time and, if located, where it points.

    ``page`` is given for a forged receipt with region annotations, and only for one.
    """
    grounding = investigation.grounding()
    found = investigation.read.case_file
    case: dict[str, Any] = {
        "receipt": investigation.receipt,
        "forged": document.label is Label.FORGED,
        "failure": investigation.read.failure if found is None else None,
        "observations": len(found.observations) if found is not None else None,
        "facts": grounding.facts if grounding is not None else None,
        "violations": list(grounding.violations) if grounding is not None else None,
        "tool_calls": len(investigation.calls),
        "steps": len(investigation.responses),
        "tokens": investigation.tokens,
        "cost_usd": float(investigation.cost_usd),
        "seconds": investigation.seconds,
        "first_location": None,
        "report_first_region": None,
    }
    if page is not None:
        box, words = page
        case["first_location"] = _pointing(first_location_hit(found, document, box, words))
        case["report_first_region"] = _pointing(first_region_hit(assessment, document, box, words))
    return case


def _guard(found: Sequence[Mapping[str, Any]], resamples: int) -> dict[str, Any] | None:
    if not found:
        return None
    guard = localisation_guard(
        [f["hit"] for f in found], [f["baseline"] for f in found], resamples=resamples
    )
    e = guard.hit_rate
    return {
        "receipts": guard.receipts,
        "located": sum(f["located"] for f in found),
        "hits": sum(f["hit"] for f in found),
        "hit_rate": {"value": e.value, "low": e.low, "high": e.high},
        "baseline": guard.baseline,
        "detects": guard.detects,
    }


def agent_validation(
    cases: Sequence[Mapping[str, Any]], *, resamples: int = DEFAULT_RESAMPLES
) -> dict[str, Any]:
    """The run's results from its measured cases (:func:`measured`), with every case kept.

    Raises:
        ValueError: If there is no case.
    """
    if not cases:
        msg = "the review queue is empty"
        raise ValueError(msg)
    failures: dict[str, int] = {}
    for case in cases:
        if case["failure"] is not None:
            failures[case["failure"]] = failures.get(case["failure"], 0) + 1
    violations = [f"{c['receipt']}: {v}" for c in cases for v in c["violations"] or ()]
    facts = sum(c["facts"] or 0 for c in cases)
    costs = [c["cost_usd"] for c in cases]
    seconds = [c["seconds"] for c in cases if c["seconds"] is not None]
    located = [c for c in cases if c["first_location"] is not None]
    return {
        "receipts": len(cases),
        "forged": sum(c["forged"] for c in cases),
        "failures": failures,
        "grounding": {
            "case_files": len(cases) - sum(failures.values()),
            "facts": facts,
            "violations": violations,
            "rate": 1 - len(violations) / facts if facts else 1.0,
        },
        "cost_usd": {
            "total": sum(costs),
            "median": float(np.median(costs)),
            "p95": float(np.percentile(costs, 95)),
        },
        "seconds": {
            "timed": len(seconds),
            "median": float(np.median(seconds)) if seconds else None,
            "p95": float(np.percentile(seconds, 95)) if seconds else None,
        },
        "first_location": _guard([c["first_location"] for c in located], resamples),
        "report_first_region": _guard([c["report_first_region"] for c in located], resamples),
        "cases": list(cases),
    }
