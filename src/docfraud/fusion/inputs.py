"""The inputs of the fusion model, built from findings only (ADR-0029, sections 6, 7 (e), 13).

Per receipt and text source there are seven inputs: the accounting rules' score (the Phase 3
scorer over the rule outcomes), the document scores of systems 2 and 1, whether system 1 has
evidence, copy-move's document score, and the two kinds of link. Track T3 adds an eighth, whether
the file carries an editing trait; tracks T1 and T2 never see it.

The inputs come from the rules' score, the findings and the links, never from a document: no
label and no annotation can reach them.
"""

from collections.abc import Sequence

from docfraud.detectors.finding import COPY_MOVE, METADATA, SYSTEM_1, SYSTEM_2, Finding
from docfraud.linking.links import Links

RULES_INPUT = "rules"
SYSTEM_2_INPUT = "system 2"
SYSTEM_1_INPUT = "system 1"
SYSTEM_1_EVIDENCE = "system 1 evidence"
COPY_MOVE_INPUT = "copy-move"
LINK_1 = "link 1"
LINK_2 = "link 2"
METADATA_INPUT = "file metadata"

BASELINE = (RULES_INPUT, SYSTEM_2_INPUT, SYSTEM_1_INPUT, SYSTEM_1_EVIDENCE, COPY_MOVE_INPUT)
"""The score-level baseline of ADR-0005 (step 4): the fusion inputs without the links."""

FUSION = (*BASELINE, LINK_1, LINK_2)
"""The fusion model's inputs in tracks T1 and T2."""

PRACTICAL = (*FUSION, METADATA_INPUT)
"""The fusion model's inputs in track T3."""

SINGLE_SOURCES: dict[str, tuple[str, ...]] = {
    "accounting rules": (RULES_INPUT,),
    "pixel evidence": (SYSTEM_2_INPUT, SYSTEM_1_INPUT, SYSTEM_1_EVIDENCE),
    "copy-move": (COPY_MOVE_INPUT,),
}
"""The single sources RQ2 compares fusion with (ADR-0029, section 8)."""


def _one(findings: Sequence[Finding], detector: str) -> Finding | None:
    found = [f for f in findings if f.detector == detector]
    if len(found) > 1:
        msg = f"one {detector} finding expected per receipt, got {len(found)}"
        raise ValueError(msg)
    return found[0] if found else None


def fusion_inputs(
    rules_score: float, findings: Sequence[Finding], links: Links, *, practical: bool = False
) -> dict[str, float]:
    """One receipt's inputs, in the order of :data:`FUSION` (or :data:`PRACTICAL` for T3).

    A detector without a finding (copy-move without a pair, a file without editing traits)
    contributes 0.

    Raises:
        ValueError: If a pixel system has no finding, or a detector has more than one.
    """
    system_2, system_1 = _one(findings, SYSTEM_2), _one(findings, SYSTEM_1)
    if system_2 is None or system_1 is None:
        msg = "systems 2 and 1 report a finding on every receipt"
        raise ValueError(msg)
    copy_move = _one(findings, COPY_MOVE)
    inputs = {
        RULES_INPUT: rules_score,
        SYSTEM_2_INPUT: system_2.score,
        SYSTEM_1_INPUT: system_1.score,
        SYSTEM_1_EVIDENCE: float(bool(system_1.details.get("evidence"))),
        COPY_MOVE_INPUT: copy_move.score if copy_move else 0.0,
        LINK_1: float(links.region_on_value),
        LINK_2: float(links.failing_value),
    }
    if practical:
        inputs[METADATA_INPUT] = float(_one(findings, METADATA) is not None)
    return inputs
