"""The agent's development on its 12 cases and its frozen version (ADR-0031, 5; ADR-0033).

Version 1 of the instruction runs on the development cases with the model and the token budget
that the pilot kept. At most one revision follows, run on the same cases with the same code; it
replaces version 1 only if all three conditions of ADR-0033 hold against it:

- fewer grounding violations in all, over the case files in the fixed form;
- no more cases without a case file in the fixed form;
- on the forged cases, the first observation's location on target at least as often.

A case without a case file, because the answer failed the form or never came, counts against the
version that produced it: as a failure, and on a forged case as a miss. The audit report's first
region on the same forged cases is recorded beside the agent's, to read, not to decide.

The kept version is frozen at a clean commit, with the code that builds the agent's requests and
tool outputs and checks its case files; the validation and test splits see that version.
"""

from collections.abc import Mapping
from decimal import Decimal
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from docfraud.data.models import Document, Label
from docfraud.evaluation.localisation import annotated_rectangles, hits
from docfraud.llm.agent import Investigation
from docfraud.llm.case_file import first_location
from docfraud.llm.provider import Settings
from docfraud.reporting.assessment import Assessment

VERSIONS = (1, 2)
"""The versions development may write: version 1 and at most one revision."""

AGENT_CODE = (
    "src/docfraud/llm/agent.py",
    "src/docfraud/llm/agent_tools.py",
    "src/docfraud/llm/case_file.py",
    "src/docfraud/llm/detector.py",
    "src/docfraud/llm/claude.py",
    "src/docfraud/llm/provider.py",
    "src/docfraud/reading",
    "src/docfraud/reporting/assessment.py",
    "src/docfraud/data/normalize.py",
)
"""The code that builds the agent's requests and tool outputs and checks its case files: the
loop, the tools (whose crops ``detector.py`` fits to the API's limits), the case file and its
grounding check (ADR-0033, item 4), the provider, the reading of the receipt, the assessment and
the normalised page. Frozen with the instruction."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ToolUse(_Frozen):
    """One tool call of a case, as its record keeps it."""

    tool: str
    arguments: dict[str, Any]


class CaseResult(_Frozen):
    """What one version of the agent did on one development case."""

    receipt: str
    forged: bool
    failure: str | None
    """Why the case has no case file in the fixed form; ``None`` if it has one."""
    observations: int | None
    facts: int | None
    violations: tuple[str, ...] | None
    """The numbers and boxes of the rendered case file that no tool output gave."""
    first_location_hit: bool | None
    """On a forged case, whether the first observation's location shares a pixel with an
    annotated modified rectangle or copy source (ADR-0031, section 6); ``None`` if authentic."""
    report_hit: bool | None
    """The same for the audit report's first region, on a forged case; ``None`` if authentic."""
    tool_calls: tuple[ToolUse, ...]
    steps: int
    tokens: int
    cost_usd: str
    seconds: float | None


class AgentRound(_Frozen):
    """What one version of the agent's instruction did on the development cases."""

    version: int = Field(ge=1)
    settings: Settings
    token_budget: int
    instruction: str
    instruction_sha256: str
    commit: str
    cases: tuple[CaseResult, ...]

    @property
    def failures(self) -> int:
        """The cases without a case file in the fixed form."""
        return sum(c.failure is not None for c in self.cases)

    @property
    def violations(self) -> int:
        """The grounding violations of all its case files."""
        return sum(len(c.violations or ()) for c in self.cases)

    @property
    def forged(self) -> int:
        """The forged cases."""
        return sum(c.forged for c in self.cases)

    @property
    def hits(self) -> int:
        """The forged cases whose first observation's location is on target."""
        return sum(c.first_location_hit is True for c in self.cases)

    @property
    def report_hits(self) -> int:
        """The forged cases whose audit report's first region is on target."""
        return sum(c.report_hit is True for c in self.cases)

    @property
    def cost_usd(self) -> Decimal:
        """What its cases' steps cost, stored answers included."""
        return sum((Decimal(c.cost_usd) for c in self.cases), Decimal(0))


class FrozenAgent(_Frozen):
    """The version of the agent that the validation and test splits see."""

    version: int
    settings: Settings
    token_budget: int
    instruction: str
    instruction_sha256: str
    commit: str


def case_result(
    investigation: Investigation, document: Document, assessment: Assessment
) -> CaseResult:
    """One case of a round: its form, its grounding and, if forged, where it points."""
    grounding = investigation.grounding()
    found = investigation.read.case_file
    forged = document.label is Label.FORGED
    first_hit = report_hit = None
    if forged:
        modified, sources = annotated_rectangles(document)
        targets = [*modified, *sources]
        first = first_location(found) if found is not None else None
        first_hit = first is not None and hits(first, targets)
        report_hit = any(hits(box, targets) for box in assessment.first_region())
    return CaseResult(
        receipt=investigation.receipt,
        forged=forged,
        failure=investigation.read.failure if found is None else None,
        observations=len(found.observations) if found is not None else None,
        facts=grounding.facts if grounding is not None else None,
        violations=grounding.violations if grounding is not None else None,
        first_location_hit=first_hit,
        report_hit=report_hit,
        tool_calls=tuple(
            ToolUse(tool=c.name, arguments=dict(c.arguments)) for c in investigation.calls
        ),
        steps=len(investigation.responses),
        tokens=investigation.tokens,
        cost_usd=f"{investigation.cost_usd:.4f}",
        seconds=investigation.seconds,
    )


def replaces(new: AgentRound, old: AgentRound) -> bool:
    """Whether ``new`` meets all three conditions of ADR-0033 (item 3) against ``old``."""
    return new.violations < old.violations and new.failures <= old.failures and new.hits >= old.hits


def kept_version(rounds: Mapping[int, AgentRound]) -> int:
    """The version to freeze, by the rule fixed in advance (ADR-0033, item 3).

    Raises:
        ValueError: If version 1 has not run, a version outside :data:`VERSIONS` has, or the
            versions ran on different cases.
    """
    if 1 not in rounds:
        msg = "version 1 has not run on the development cases"
        raise ValueError(msg)
    if set(rounds) - set(VERSIONS):
        msg = f"development allows versions {VERSIONS} only"
        raise ValueError(msg)
    first = rounds[1]
    second = rounds.get(2)
    if second is None:
        return 1
    if [c.receipt for c in second.cases] != [c.receipt for c in first.cases]:
        msg = "the two versions ran on different cases"
        raise ValueError(msg)
    return 2 if replaces(second, first) else 1


def save_round(record: AgentRound, path: Path) -> None:
    """Write a development round as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_rounds(directory: Path) -> dict[int, AgentRound]:
    """Every development round written to ``directory``, by version."""
    found = [
        AgentRound.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(directory.glob("round-v*.json"))
    ]
    return {r.version: r for r in found}
