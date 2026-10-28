"""The detector's development on the design fold and its frozen version (ADR-0031, section 2).

Version 1 of the instruction runs on the 113 design-fold receipts, for both rows. At most one
revision follows; it is kept only if its average precision on the design fold, main row,
exceeds version 1's by at least :data:`MARGIN` and it fails the answer form on no more receipts.
The kept version is frozen at a clean commit and is the one the validation and test splits see.

A receipt without a score, because the answer failed its form or never came, ranks below every
scored receipt: a failure counts against the version that produced it.

Once frozen, the main row is checked once on a stronger model over the same receipts
(ADR-0032); by the same margin, that check decides whether the stronger model also runs on the
validation and test splits.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from docfraud.evaluation.detection import average_precision, roc_auc
from docfraud.llm.detector import SONNET, DetectorRow, Reading
from docfraud.llm.provider import Settings

MARGIN = 0.05
"""How much higher a revision's average precision must be to replace version 1 (a convention)."""

VERSIONS = (1, 2)
"""The versions development may write: version 1 and at most one revision."""

UNSCORED = -1.0
"""The score of a receipt without one: below every score the model can give."""

OPUS = SONNET.model_copy(update={"model": "claude-opus-5-5"})
"""The stronger model the frozen main row is checked on: every other setting is the same."""

DETECTOR_CODE = (
    "src/docfraud/llm/detector.py",
    "src/docfraud/llm/claude.py",
    "src/docfraud/llm/provider.py",
    "src/docfraud/reading",
    "src/docfraud/data/normalize.py",
)
"""The code that builds the detector's requests from a receipt; frozen with the instruction."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class RowResult(_Frozen):
    """One row of one version on the design fold."""

    receipts: int
    forged: int
    failures: int
    average_precision: float
    roc_auc: float
    cost_usd: str


class DevelopmentRound(_Frozen):
    """What one version of the instruction did on the design fold."""

    version: int = Field(ge=1)
    settings: Settings
    instruction: str
    instruction_sha256: str
    commit: str | None
    rows: dict[str, RowResult]

    def main_row(self) -> RowResult:
        """The row that decides between versions: image and OCR text."""
        return self.rows[DetectorRow.IMAGE_AND_OCR.value]


class FrozenDetector(_Frozen):
    """The version of the detector that the validation and test splits see."""

    version: int
    settings: Settings
    instruction: str
    instruction_sha256: str
    commit: str


class StrongerModelCheck(_Frozen):
    """The frozen main row on a stronger model, on the design fold (ADR-0032)."""

    settings: Settings
    instruction: str
    instruction_sha256: str
    commit: str | None
    result: RowResult
    frozen_result: RowResult
    """The frozen main row on its own model, over the same receipts."""
    row_added: bool
    """Whether the stronger model also runs on the validation and test splits."""


def scores(readings: Sequence[Reading]) -> npt.NDArray[np.float64]:
    """Every receipt's score, with :data:`UNSCORED` for a receipt without one."""
    return np.array(
        [UNSCORED if r.suspicion is None else float(r.suspicion) for r in readings], dtype=float
    )


def row_result(forged: Sequence[bool], readings: Sequence[Reading], cost_usd: str) -> RowResult:
    """The design-fold result of one row: ranking metrics, failures and cost.

    Raises:
        ValueError: If there is not one reading per receipt.
    """
    if len(forged) != len(readings):
        msg = "one reading per receipt"
        raise ValueError(msg)
    y = np.asarray(forged, dtype=bool)
    s = scores(readings)
    return RowResult(
        receipts=int(y.size),
        forged=int(y.sum()),
        failures=sum(r.suspicion is None for r in readings),
        average_precision=average_precision(y, s),
        roc_auc=roc_auc(y, s),
        cost_usd=cost_usd,
    )


def clearly_better(new: RowResult, old: RowResult) -> bool:
    """Whether ``new`` ranks at least :data:`MARGIN` higher and fails on no more receipts."""
    return new.average_precision >= old.average_precision + MARGIN and new.failures <= old.failures


def kept_version(rounds: Mapping[int, DevelopmentRound]) -> int:
    """The version to freeze, by the rule fixed in advance.

    Raises:
        ValueError: If version 1 has not run, or a version outside :data:`VERSIONS` has.
    """
    if 1 not in rounds:
        msg = "version 1 has not run on the design fold"
        raise ValueError(msg)
    if set(rounds) - set(VERSIONS):
        msg = f"development allows versions {VERSIONS} only"
        raise ValueError(msg)
    first = rounds[1].main_row()
    second = rounds[2].main_row() if 2 in rounds else None  # noqa: PLR2004 - the revision
    return 2 if second is not None and clearly_better(second, first) else 1


def save_round(record: DevelopmentRound, path: Path) -> None:
    """Write a development round as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")


def load_rounds(directory: Path) -> dict[int, DevelopmentRound]:
    """Every development round written to ``directory``, by version."""
    found = [
        DevelopmentRound.model_validate_json(p.read_text(encoding="utf-8"))
        for p in sorted(directory.glob("round-v*.json"))
    ]
    return {r.version: r for r in found}
