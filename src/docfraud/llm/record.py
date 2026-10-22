"""Records of the development rounds and of the frozen version (ADR-0025, decision 3).

Each development round leaves a machine-readable record next to its report. The frozen version
is the round with the most receipts whose rule inputs are all exact on gold set A, both text
sources counted; ties go to the earlier round. It is frozen once, from a clean commit, before
gold set B is read.
"""

from collections.abc import Mapping, Sequence

from pydantic import BaseModel, ConfigDict, Field

from docfraud.llm.provider import Settings


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RoundRecord(_Frozen):
    """What one development round used and how many receipts it read exactly."""

    round: int = Field(ge=1)
    settings: Settings
    instruction: str
    instruction_sha256: str
    receipts: int
    exact: Mapping[str, int]  # per text source, receipts with all rule inputs exact

    @property
    def exact_total(self) -> int:
        """Exact readings over both text sources."""
        return sum(self.exact.values())


class FrozenVersion(_Frozen):
    """The version of the LLM reader that reads gold set B and, if it passes, everything else."""

    round: int
    settings: Settings
    instruction: str
    instruction_sha256: str
    commit: str


def best_round(records: Sequence[RoundRecord]) -> RoundRecord:
    """The round to freeze: most exact readings over both text sources, the earlier on ties.

    Raises:
        ValueError: If there is no round.
    """
    if not records:
        msg = "no development round has been recorded"
        raise ValueError(msg)
    return min(records, key=lambda r: (-r.exact_total, r.round))
