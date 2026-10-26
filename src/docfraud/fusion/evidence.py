"""A receipt's findings and fusion inputs, from its page and readings only (ADR-0029, 5-7).

The pixel findings that do not depend on the text source or on system 2's seed are found once
per receipt: system 1's lattice, copy-move's pair and the file's editing traits. System 2's
finding comes from its stored output for the seed at hand. Per text source, the frozen reader's
fields give the rule findings and the located values, and from all of them the links and the
inputs follow.

Nothing here takes a document: labels and annotations cannot reach a finding or an input
(ADR-0029, section 7 (e)).
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass

import numpy as np
import numpy.typing as npt

from docfraud.data.models import BBox
from docfraud.detectors.copy_move import copy_move_evidence
from docfraud.detectors.finding import (
    SYSTEM_2,
    Finding,
    copy_move_finding,
    lattice_finding,
    metadata_finding,
    region_finding,
    rule_findings,
)
from docfraud.detectors.rules import Outcome
from docfraud.forensics.lattice_map import lattice_evidence
from docfraud.fusion.inputs import fusion_inputs
from docfraud.linking.links import link, link_findings
from docfraud.linking.values import located_values
from docfraud.reading.fields import FieldSources


@dataclass(frozen=True, slots=True)
class SystemTwo:
    """System 2's output on one receipt: its document score and its top region."""

    score: float
    region: BBox


def page_findings(
    page: npt.NDArray[np.uint8], words: Sequence[BBox], traits: Sequence[str]
) -> tuple[Finding, ...]:
    """The findings of system 1, copy-move and the file-metadata check on one page."""
    found = (
        lattice_finding(lattice_evidence(page)),
        copy_move_finding(copy_move_evidence(page, words)),
        metadata_finding(traits),
    )
    return tuple(f for f in found if f is not None)


def receipt_evidence(
    rules_score: float,
    system_2: SystemTwo,
    page: Sequence[Finding],
    outcomes: Mapping[str, Outcome],
    sources: FieldSources,
) -> tuple[dict[str, float], tuple[Finding, ...]]:
    """One receipt's inputs (those of track T3, whose first seven are T1's or T2's) and findings.

    ``page`` holds the findings of :func:`page_findings`; ``outcomes`` and ``sources`` are the
    accounting rules' outcomes on the frozen reader's fields of one text source, and where those
    fields are printed.
    """
    values = located_values(sources)
    detected = (
        region_finding(SYSTEM_2, system_2.score, system_2.region),
        *page,
        *rule_findings(outcomes, values),
    )
    links = link(detected, values)
    findings = (*detected, *link_findings(links))
    return fusion_inputs(rules_score, findings, links, practical=True), findings
