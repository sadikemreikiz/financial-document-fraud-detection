"""The audit report of one receipt, from a fixed template (ADR-0029, section 12; ADR-0006).

No language model writes it: every sentence is either fixed wording or a finding's reason, and
every number comes from the assessment, which :mod:`docfraud.reporting.grounding` checks
independently. Evidence is listed by its contribution to the log-odds, the strongest push
towards forgery first.
"""

from collections.abc import Sequence

from docfraud.data.models import BBox
from docfraud.reporting.assessment import Assessment, Evidence


def box_text(box: BBox) -> str:
    """A box as reports state it."""
    return f"({box.x}, {box.y}, {box.width} x {box.height} px)"


def _evidence(item: Evidence) -> list[str]:
    lines = [f"- **{' + '.join(item.inputs)}** ({item.contribution:+.3f} on the log-odds)"]
    lines += [f"  - {finding.reason}" for finding in item.findings]
    if item.boxes:
        lines.append(f"  - located at {'; '.join(box_text(b) for b in item.boxes)}")
    return lines


def _section(title: str, items: Sequence[Evidence]) -> list[str]:
    """A titled list; its order is the order of the evidence, so no number is printed for it."""
    if not items:
        return []
    lines = ["", f"## {title}", ""]
    for item in items:
        lines += _evidence(item)
    return lines


def render_report(assessment: Assessment) -> str:
    """The report in Markdown."""
    towards = [e for e in assessment.evidence if e.contribution > 0]
    # evidence is ordered from the strongest push towards forgery to the strongest against
    against = [e for e in reversed(assessment.evidence) if e.contribution <= 0]
    lines = [
        f"# Audit report: {assessment.receipt}",
        "",
        f"- Track: {assessment.track}",
        f"- Probability of forgery: {assessment.probability:.3f} (calibrated)",
        f"- Decision: {assessment.decision.value}; receipts from {assessment.threshold:.3f} "
        f"are reviewed, a review budget of {assessment.budget * 100:.0f} %",
        "",
        "The system accepts a receipt or sends it to review. Rejecting it is the reviewer's "
        "decision.",
    ]
    lines += _section("Evidence towards forgery, strongest first", towards)
    lines += _section("Evidence against forgery, strongest first", against)
    return "\n".join(lines) + "\n"
