"""The realism review of ADR-0021 (decision 5): can the owner see the synthetic edits?

The owner found 2 of the 45 real edits of the design fold at 100 %. The review asks the same
question of synthetic edits made on design-fold receipts: 40 regions drawn with a fixed seed
from the synthetic development set, shown as the real ones were shown (the receipt's width and
200 px above and below the edit, without a box until visibility is answered), followed by
whether the edit looks like one a forger could make. If 5 or more of the 40 are found at 100 %,
the engine counts as clearly more visible than the real edits. The technique of an item is
never shown while it is reviewed; it is joined afterwards from the forgery's record.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from itertools import pairwise
from pathlib import Path
from typing import Self

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator

from docfraud.data.models import BBox, Technique
from docfraud.data.observation import Display, Visibility
from docfraud.evaluation.design_fold import require_design_fold
from docfraud.markdown import Table
from docfraud.synthesis.engine import ForgeryRecord, Stream

REVIEW_VERSION = 1
"""Version of the review log; a log of another version is refused."""

REVIEW_ITEMS = 40
"""Synthetic edits the owner reviews in a round of the review."""

REVIEW_SEED = 20261014
"""Seed of the review sample, together with the round of the review."""

FOUND_LIMIT = 5
"""Edits found at 100 % from which the engine counts as clearly more visible than real edits:
with the real rate of 2 in 45, 5 or more of 40 occur by chance with probability 0.031."""

MAX_ROUNDS = 2
"""Rounds of the review: the first, and one more after a revision of the engine."""

REAL_FOUND, REAL_REGIONS = 2, 45
"""The owner's result on the real edits of the design fold (step 4.2)."""

BREAK = 600.0
"""Seconds between two answers above which the reviewer is taken to have paused."""


class Plausibility(StrEnum):
    """Whether an edit looks like one a forger could make."""

    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReviewAnswer(_Frozen):
    """The owner's answers on one synthetic edit."""

    visibility: Visibility
    plausible: Plausibility
    note: str = ""
    display: Display
    answered_at: datetime
    commit: str | None
    language: str


class ReviewLog(_Frozen):
    """Every answer of one reviewer in one round of the review."""

    version: int = REVIEW_VERSION
    reviewer: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    round: int = Field(ge=1, le=MAX_ROUNDS)
    answers: dict[str, ReviewAnswer] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _current_version(self) -> Self:
        if self.version != REVIEW_VERSION:
            msg = f"review log version {self.version}, expected {REVIEW_VERSION}"
            raise ValueError(msg)
        return self


@dataclass(frozen=True, slots=True)
class ReviewItem:
    """One synthetic edit shown to the reviewer."""

    key: str  # base and number of the region, for example "X51005230617-2"
    base: str
    number: int  # the region's position in its forgery's record
    box: BBox


def review_sample(records: Iterable[ForgeryRecord], round_number: int) -> tuple[ReviewItem, ...]:
    """The edits of a round of the review, drawn from the development set with a fixed seed.

    Every region of every forgery is equally likely, so the techniques come in the engine's own
    mix, as the real regions came in the dataset's.

    Raises:
        ValueError: If a record is not of the development set or there are too few regions.
    """
    regions: list[ReviewItem] = []
    for record in sorted(records, key=lambda r: r.base):
        if record.stream is not Stream.DEVELOPMENT or record.index != 0:
            msg = f"{record.base}: the review draws from the development set only"
            raise ValueError(msg)
        require_design_fold(record.base)
        regions.extend(
            ReviewItem(f"{record.base}-{n}", record.base, n, region.box)
            for n, region in enumerate(record.regions)
        )
    if len(regions) < REVIEW_ITEMS:
        msg = f"{len(regions)} regions, the review needs {REVIEW_ITEMS}"
        raise ValueError(msg)
    picks = np.random.default_rng([REVIEW_SEED, round_number]).choice(
        len(regions), REVIEW_ITEMS, replace=False
    )
    return tuple(regions[int(i)] for i in picks)


def log_path(directory: Path, reviewer: str, round_number: int) -> Path:
    """Where a reviewer's answers of one round are kept."""
    return directory / f"{reviewer}-{round_number}.json"


def load_log(path: Path, reviewer: str, round_number: int) -> ReviewLog:
    """Read a review log, or start an empty one.

    Raises:
        ValueError: If the log belongs to another reviewer or round.
    """
    if not path.exists():
        return ReviewLog(reviewer=reviewer, round=round_number)
    log = ReviewLog.model_validate_json(path.read_text(encoding="utf-8"))
    if (log.reviewer, log.round) != (reviewer, round_number):
        msg = f"{path} is round {log.round} of {log.reviewer!r}"
        raise ValueError(msg)
    return log


def save_log(log: ReviewLog, path: Path) -> None:
    """Write a review log atomically."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(log.model_dump_json(indent=2) + "\n", encoding="utf-8")
    temporary.replace(path)


@dataclass(frozen=True, slots=True)
class Outcome:
    """What a finished round of the review found."""

    visibility: dict[Visibility, int]
    plausible: dict[Plausibility, int]
    by_technique: dict[Technique, dict[Visibility, int]]

    @property
    def found(self) -> int:
        """Edits found at 100 %."""
        return self.visibility.get(Visibility.AT_100, 0)

    @property
    def exceeded(self) -> bool:
        """Whether the engine counts as clearly more visible than the real edits."""
        return self.found >= FOUND_LIMIT


def outcome(
    log: ReviewLog, items: Sequence[ReviewItem], records: Iterable[ForgeryRecord]
) -> Outcome:
    """Count a finished round; techniques are joined from the forgeries' records.

    Raises:
        ValueError: If an item of the sample has no answer, or an answer no item.
    """
    keys = {item.key for item in items}
    if set(log.answers) != keys:
        missing = len(keys - set(log.answers))
        msg = f"the round is not finished: {missing} of {len(keys)} items unanswered"
        raise ValueError(msg)
    by_base = {record.base: record for record in records}
    visibility: Counter[Visibility] = Counter()
    plausible: Counter[Plausibility] = Counter()
    by_technique: dict[Technique, Counter[Visibility]] = {}
    for item in items:
        answer = log.answers[item.key]
        visibility[answer.visibility] += 1
        plausible[answer.plausible] += 1
        technique = by_base[item.base].regions[item.number].technique
        by_technique.setdefault(technique, Counter())[answer.visibility] += 1
    return Outcome(dict(visibility), dict(plausible), {t: dict(c) for t, c in by_technique.items()})


def seconds_between(times: Iterable[datetime]) -> list[float]:
    """Seconds between consecutive answers, leaving out pauses longer than ``BREAK``."""
    ordered = sorted(times)
    gaps = [(b - a).total_seconds() for a, b in pairwise(ordered)]
    return [gap for gap in gaps if gap <= BREAK]


def review_tables(  # noqa: PLR0913 - the outcome, its comparison and its sources
    result: Outcome,
    real: Counter[Visibility],
    items: Sequence[ReviewItem],
    log: ReviewLog,
    records: Iterable[ForgeryRecord],
    *,
    real_seconds: Sequence[float],
) -> list[Table]:
    """The tables of the review report.

    Visibility against the real edits, visibility by technique, plausibility, the time taken
    per answer, and the notes.
    """
    choices = list(Visibility)
    visible = Table(
        "Visible at 100 %",
        ["Edits", "Count", *(v.value for v in choices)],
        [
            ["real, step 4.2", sum(real.values()), *(real.get(v, 0) for v in choices)],
            [
                f"synthetic, round {log.round}",
                len(items),
                *(result.visibility.get(v, 0) for v in choices),
            ],
        ],
        f"The engine counts as clearly more visible than the real edits if {FOUND_LIMIT} or more "
        f"of the {REVIEW_ITEMS} synthetic edits are found at 100 % (ADR-0021, decision 5). Real: "
        "the owner's answers on the 45 modified regions of the design fold.",
    )
    technique = Table(
        "By technique",
        ["Technique", "Edits", *(v.value for v in choices)],
        [
            [t.value, sum(c.values()), *(c.get(v, 0) for v in choices)]
            for t, c in sorted(result.by_technique.items(), key=lambda pair: pair[0].value)
        ],
        "Techniques were not shown during the review; they are joined from the records.",
    )
    plausible = Table(
        "Looks like an edit a forger could make",
        ["Answer", "Edits"],
        [[p.value, result.plausible.get(p, 0)] for p in Plausibility],
    )
    synthetic_seconds = seconds_between(a.answered_at for a in log.answers.values())
    timing = Table(
        "Time per answer",
        ["Edits", "Answers timed", "Median seconds", "25th percentile", "75th percentile"],
        [
            [name, len(gaps), *(float(np.percentile(gaps, q)) for q in (50, 25, 75))]
            for name, gaps in (
                ("real, step 4.2, visibility only", real_seconds),
                (f"synthetic, round {log.round}, both questions", synthetic_seconds),
            )
            if gaps
        ],
        f"Seconds between consecutive answers, pauses of more than {BREAK:.0f} s left out. "
        "For the real edits, only answers that asked visibility alone, not the full form. "
        "Similar times suggest that both sets of edits were looked at with similar care.",
    )
    by_base = {record.base: record for record in records}
    notes = Table(
        "Notes",
        ["Edit", "Technique", "Visibility", "Note"],
        [
            [
                position,
                by_base[item.base].regions[item.number].technique.value,
                log.answers[item.key].visibility.value,
                log.answers[item.key].note.strip(),
            ]
            for position, item in enumerate(items, start=1)
            if log.answers[item.key].note.strip()
        ],
        "Free notes of the owner, in the order the edits were shown.",
    )
    return [visible, technique, plausible, timing, notes]
