"""Findings: the contract between the layers of the system (ADR-0005; architecture).

A detector's output becomes one or more findings: which detector found what, how strongly, where
on the page and in words. Linking relates findings, fusion turns their scores into inputs, and
the audit report can state nothing that is not a finding. Every sentence a finding carries is
built from the detector's output alone, never from labels or annotations.

The functions below turn each detector's output into its findings:

- **Accounting rules**: one finding per failing rule, located by the boxes of every read value
  the rule reads (:data:`~docfraud.linking.values.RULE_VALUES`).
- **Systems 2 and 1**: the system's document score and its top region (ADR-0015, section 2);
  system 1 has no region on a receipt without evidence.
- **Copy-move**: the document score and the pair of identical regions (ADR-0030).
- **File metadata**: the editing traits of the file container, without a location (track T3).
"""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict

from docfraud.data.models import BBox
from docfraud.detectors.copy_move import CopyMoveEvidence
from docfraud.detectors.rules import Outcome, Status
from docfraud.evaluation.localisation import top_region
from docfraud.forensics.lattice_map import LatticeEvidence
from docfraud.linking.values import RULE_VALUES, LocatedValue

RULES = "accounting rules"
SYSTEM_2 = "system 2"
SYSTEM_1 = "system 1"
COPY_MOVE = "copy-move"
METADATA = "file metadata"
LINKING = "linking"

PIXEL_DETECTORS = frozenset({SYSTEM_2, SYSTEM_1, COPY_MOVE})
"""Detectors whose regions are suspicious regions for linking (ADR-0029, section 5)."""

type Detail = str | int | float | bool


class Finding(BaseModel):
    """One piece of evidence about one receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    detector: str
    kind: str
    score: float  # the detector's score behind the finding
    boxes: tuple[BBox, ...] = ()  # where on the page; empty if the finding has no location
    values: tuple[str, ...] = ()  # roles of the read values it concerns
    reason: str  # what was found, in words, from the detector's output only
    details: dict[str, Detail] = {}


def _box(box: BBox) -> str:
    return f"({box.x}, {box.y}, {box.width} x {box.height} px)"


def rule_findings(
    outcomes: Mapping[str, Outcome], values: Sequence[LocatedValue]
) -> tuple[Finding, ...]:
    """One finding per failing rule, at the read values the rule reads."""
    findings = []
    for rule, outcome in outcomes.items():
        if outcome.status is not Status.FAIL:
            continue
        read = [v for v in values if v.role in RULE_VALUES[rule]]
        findings.append(
            Finding(
                detector=RULES,
                kind="failing rule",
                score=1.0,
                boxes=tuple(v.box for v in read),
                values=tuple(sorted({v.role for v in read})),
                reason=f"{rule}: {outcome.reason}",
                details={"rule": rule},
            )
        )
    return tuple(findings)


def region_finding(detector: str, score: float, region: BBox | None) -> Finding:
    """A pixel system's document score and its top region, if it has one."""
    if region is None:
        reason = f"{detector} scores the receipt {score:.3f} and has no evidence on it"
    else:
        reason = (
            f"{detector} scores the receipt {score:.3f}; its most suspicious region is "
            f"{_box(region)}"
        )
    return Finding(
        detector=detector,
        kind="suspicious region",
        score=score,
        boxes=() if region is None else (region,),
        reason=reason,
        details={"evidence": region is not None},
    )


def lattice_finding(evidence: LatticeEvidence) -> Finding:
    """System 1's finding: its top region only where it has evidence (ADR-0017)."""
    has_evidence = bool(evidence.measured.any())
    region = top_region(evidence.pixel_map()) if has_evidence else None
    return region_finding(SYSTEM_1, evidence.document_score(), region)


def copy_move_finding(evidence: CopyMoveEvidence) -> Finding | None:
    """Copy-move's finding: the pair of identical regions, if it found one."""
    pair = evidence.pair
    if pair is None:
        return None
    return Finding(
        detector=COPY_MOVE,
        kind="duplicated regions",
        score=evidence.document_score,
        boxes=(pair.first, pair.second),
        reason=(
            f"copy-move scores the receipt {evidence.document_score:.3f}: the regions "
            f"{_box(pair.first)} and {_box(pair.second)} are pixel for pixel identical over "
            f"{pair.area} px"
        ),
        details={"area": pair.area, "text_height": evidence.text_height},
    )


def metadata_finding(traits: Sequence[str]) -> Finding | None:
    """The editing traits of the file container, if it carries any (track T3 only)."""
    if not traits:
        return None
    return Finding(
        detector=METADATA,
        kind="editing traits",
        score=1.0,
        reason=f"the file carries traces of an editing program: {', '.join(traits)}",
        details={"traits": len(traits)},
    )
