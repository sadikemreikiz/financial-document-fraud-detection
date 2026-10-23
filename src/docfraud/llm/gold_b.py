"""Gold set B: the one reading by the frozen LLM reader and parser v2, and what it decides.

ADR-0025, decision 4: the frozen version reads gold set B once from each text source, and parser
v2 reads the same rows again. Both readers' fields are kept here, together with what is in
dispute among them and the gold annotation. No metric is computed from them until every question
of the adjudication has a final answer (ADR-0026); then the gold set with the owner's answers in
place decides, per text source, whether the LLM reader replaces parser v2 (decision 5).

The LLM's requests on gold set B are sent one by one rather than to the batch service, at twice
the price, so that the time to each answer is measured (decision 6).
"""

from collections import Counter, defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from docfraud.detectors.rules import Status, check
from docfraud.llm.adjudication import GOLD, Answer, Kind, Question, adjudicated, questions_for
from docfraud.llm.reader import Rejection
from docfraud.llm.record import FrozenVersion
from docfraud.reading.fields import Extraction, ReceiptFields
from docfraud.reading.pipeline import TextSource

LLM = "LLM"
PARSER = "parser v2"
READERS = (PARSER, LLM)  # the baseline first

#: The code that turns a receipt's rows into a request and an answer into fields, and the
#: parser it is compared with. Gold set B, and the receipts of ADR-0027, are read only while it
#: is as frozen (ADR-0025).
READER_CODE = (
    "src/docfraud/llm/answer.py",
    "src/docfraud/llm/claude.py",
    "src/docfraud/llm/labels.py",
    "src/docfraud/llm/provider.py",
    "src/docfraud/llm/reader.py",
    "src/docfraud/llm/run.py",
    "src/docfraud/reading",
    "src/docfraud/data/normalize.py",
)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


def reading_name(reader: str, source: TextSource) -> str:
    """How one reader on one text source is named among the readings of a receipt."""
    return f"{reader} {source.value}"


class ReaderOutput(_Frozen):
    """What one reader extracted from one receipt on one text source."""

    doc_id: str
    reader: str
    source: TextSource
    extraction: Extraction
    rejections: tuple[Rejection, ...] = ()
    seconds: float | None = None  # time to the answer of a request sent on its own

    @property
    def name(self) -> str:
        """The name of this reading among the readings of its receipt."""
        return reading_name(self.reader, self.source)


class Measurement(_Frozen):
    """The one reading of a set of receipts: the version, the commit, the cost and every output.

    Used for gold set B (ADR-0025) and for the receipts of ADR-0027.
    """

    version: FrozenVersion
    commit: str
    measured_at: datetime
    receipts: int
    cost_usd: Decimal
    readings: tuple[ReaderOutput, ...]


def save_measurement(measurement: Measurement, path: Path) -> None:
    """Keep the measurement; it holds the readers' fields, so the owner does not open it."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(measurement.model_dump_json(indent=1) + "\n", encoding="utf-8")


def load_measurement(path: Path) -> Measurement:
    """The measurement kept by :func:`save_measurement`."""
    return Measurement.model_validate_json(path.read_text(encoding="utf-8"))


def questions_on(
    gold: Mapping[str, ReceiptFields], readings: Sequence[ReaderOutput]
) -> list[Question]:
    """What is in dispute on every receipt among the gold annotation and the four readings.

    Raises:
        ValueError: If a receipt does not have one reading per reader and text source.
    """
    by_doc: dict[str, dict[str, ReceiptFields]] = defaultdict(dict)
    for reading in readings:
        by_doc[reading.doc_id][reading.name] = reading.extraction.fields
    expected = {reading_name(r, s) for r in READERS for s in TextSource}
    found: list[Question] = []
    for doc_id in sorted(gold):
        if set(by_doc[doc_id]) != expected:
            msg = f"{doc_id} needs one reading per reader and text source"
            raise ValueError(msg)
        found += questions_for(doc_id, {GOLD: gold[doc_id], **by_doc[doc_id]})
    return found


def unanswered(questions: Sequence[Question], answers: Mapping[str, Answer]) -> list[str]:
    """The keys of the questions that have no final answer yet."""
    return [q.key for q in questions if q.key not in answers or not answers[q.key].done]


def adjudicated_gold(
    gold: Mapping[str, ReceiptFields],
    questions: Sequence[Question],
    answers: Mapping[str, Answer],
) -> dict[str, ReceiptFields]:
    """Gold set B with every final answer in place; receipts without a question are unchanged.

    Raises:
        ValueError: If a question has no final answer.
    """
    by_doc: dict[str, list[Question]] = defaultdict(list)
    for question in questions:
        by_doc[question.doc_id].append(question)
    return {
        doc_id: adjudicated(fields, by_doc[doc_id], answers) if doc_id in by_doc else fields
        for doc_id, fields in gold.items()
    }


@dataclass(frozen=True, slots=True)
class AdjudicationCounts:
    """What the adjudication found for one kind of question (ADR-0025, decision 6).

    ``agreed`` counts, per reading, the questions whose final answer equals that reading.
    """

    kind: Kind
    questions: int
    first_matched: int  # the owner's first answer equalled a reading
    gold_changed: int  # the final answer differs from the gold annotation
    agreed: Mapping[str, int]


def adjudication_counts(
    questions: Sequence[Question], answers: Mapping[str, Answer]
) -> list[AdjudicationCounts]:
    """The counts per kind of question, in the order of :class:`Kind`.

    Raises:
        ValueError: If a question has no final answer.
    """
    found: list[AdjudicationCounts] = []
    for kind in Kind:
        asked = [q for q in questions if q.kind is kind]
        agreed: Counter[str] = Counter()
        first_matched = gold_changed = 0
        for question in asked:
            answer = answers.get(question.key)
            final = None if answer is None else answer.final
            if answer is None or final is None:
                msg = f"question {question.key} has no final answer"
                raise ValueError(msg)
            first_matched += answer.matched
            for candidate in question.candidates:
                if candidate.listing.same_as(final):
                    agreed.update(candidate.sources)
                elif GOLD in candidate.sources:
                    gold_changed += 1
        found.append(AdjudicationCounts(kind, len(asked), first_matched, gold_changed, agreed))
    return found


@dataclass(frozen=True, slots=True)
class Latency:
    """Seconds from sending a request on its own to its answer, over the timed answers."""

    timed: int
    median: float
    p95: float


def latency(seconds: Sequence[float | None]) -> Latency | None:
    """Median and 95th percentile of the timed answers, or ``None`` if none was timed."""
    timed = [s for s in seconds if s is not None]
    if not timed:
        return None
    return Latency(len(timed), float(np.median(timed)), float(np.percentile(timed, 95)))


def false_alarms(fields: Mapping[str, ReceiptFields], authentic: frozenset[str]) -> int:
    """Authentic receipts on which at least one accounting rule fails with these fields."""
    return sum(
        any(o.status is Status.FAIL for o in check(f).values())
        for doc_id, f in fields.items()
        if doc_id in authentic
    )
