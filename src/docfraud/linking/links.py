"""The two kinds of link between findings of different modalities (ADR-0029, 5; ADR-0030).

1. **A suspicious region lies on a read value**: a region of a pixel finding (the top region of
   system 2, that of system 1 where it has evidence, either region of a copy-move pair) shares a
   pixel with the box of an amount, a line-item part, a tax-row part or a date the reader read.
2. **That value is read by a failing accounting rule**, according to the fixed map
   :data:`~docfraud.linking.values.RULE_VALUES`.

Links take findings and the reader's located values only, so they cannot see labels or
annotations. No threshold is involved: regions are the detectors' top regions, never a region
cut from a map at a threshold chosen on the validation split.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from docfraud.data.models import BBox
from docfraud.detectors.finding import LINKING, PIXEL_DETECTORS, RULES, Finding
from docfraud.linking.values import LocatedValue, read_by


@dataclass(frozen=True, slots=True)
class Links:
    """The read values that suspicious regions lie on, and those of them a failing rule reads."""

    on_value: tuple[LocatedValue, ...]
    on_failing_value: tuple[LocatedValue, ...]

    @property
    def region_on_value(self) -> bool:
        """Link 1: a suspicious region lies on a read value."""
        return bool(self.on_value)

    @property
    def failing_value(self) -> bool:
        """Link 2: such a value is read by a failing rule."""
        return bool(self.on_failing_value)


def suspicious_regions(findings: Sequence[Finding]) -> tuple[BBox, ...]:
    """The regions of the pixel findings."""
    return tuple(box for f in findings if f.detector in PIXEL_DETECTORS for box in f.boxes)


def failing_rules(findings: Sequence[Finding]) -> tuple[str, ...]:
    """The rules whose findings say that they fail."""
    return tuple(str(f.details["rule"]) for f in findings if f.detector == RULES)


def link(findings: Sequence[Finding], values: Sequence[LocatedValue]) -> Links:
    """Relate a receipt's pixel findings to its read values and its failing rules."""
    regions = suspicious_regions(findings)
    on_value = tuple(v for v in values if any(v.box.overlaps(r) for r in regions))
    read = read_by(failing_rules(findings))
    return Links(on_value, tuple(v for v in on_value if v.role in read))


def link_findings(links: Links) -> tuple[Finding, ...]:
    """The links as findings, located at the values they concern."""

    def finding(kind: str, values: Sequence[LocatedValue], reason: str) -> Finding:
        roles = tuple(sorted({v.role for v in values}))
        return Finding(
            detector=LINKING,
            kind=kind,
            score=1.0,
            boxes=tuple(v.box for v in values),
            values=roles,
            reason=f"{reason}: {', '.join(roles)}",
        )

    found = []
    if links.region_on_value:
        found.append(
            finding("region on a read value", links.on_value, "a suspicious region lies on")
        )
    if links.failing_value:
        found.append(
            finding(
                "region on a value a failing rule reads",
                links.on_failing_value,
                "a suspicious region lies on a value that a failing rule reads",
            )
        )
    return tuple(found)
