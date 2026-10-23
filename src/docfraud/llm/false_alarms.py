"""The LLM reader judged once more, by the accounting rules' false alarms (ADR-0027).

The frozen version and parser v2 read the training receipts outside both gold sets once from
each text source. A receipt raises an alarm when at least one accounting rule fails with a
reader's fields. Every receipt is labelled authentic or forged, so no annotation is needed: per
text source, the LLM reader replaces parser v2 only if it raises fewer false alarms on the
authentic receipts and its catch rate on the forged ones falls by less than its false-alarm rate
(:func:`docfraud.llm.decision.decide_by_alarms`).
"""

from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass

import numpy as np

from docfraud.detectors.rules import RULES, Status, check
from docfraud.evaluation.bootstrap import Estimate, stratified
from docfraud.llm.decision import AlarmDecision, decide_by_alarms
from docfraud.llm.gold_b import LLM, PARSER, READERS, ReaderOutput
from docfraud.reading.fields import ReceiptFields
from docfraud.reading.pipeline import TextSource

#: The commit of ADR-0027; the accounting rules must be unchanged since.
FIXED_AT = "888aeb65c1d0a7df6c95b1c836c6317ed2feafac"

#: The accounting rules the test applies.
RULES_CODE = ("src/docfraud/detectors/rules.py",)


@dataclass(frozen=True, slots=True)
class Alarms:
    """How many receipts of each class raise an alarm with one reader on one text source."""

    false_alarms: int  # authentic receipts on which at least one rule fails
    catches: int  # forged receipts on which at least one rule fails


@dataclass(frozen=True, slots=True)
class TrackResult:
    """What one text source gives under ADR-0027; readers in the order of ``READERS``."""

    decision: AlarmDecision
    alarms: dict[str, Alarms]
    separation: Estimate  # LLM minus parser v2 in catch rate minus false-alarm rate
    rules: dict[str, tuple[dict[str, int], dict[str, int]]]  # failures on authentic, on forged
    rejected: int  # receipts with a rejected reference in the LLM's answer


def failing(fields: ReceiptFields) -> tuple[str, ...]:
    """The accounting rules that fail with these fields, in rule order."""
    return tuple(name for name, outcome in check(fields).items() if outcome.status is Status.FAIL)


def rule_counts(failures: Iterable[Sequence[str]]) -> dict[str, int]:
    """For every accounting rule, on how many receipts it fails."""
    counted = Counter(name for names in failures for name in names)
    return {name: counted[name] for name in RULES}


def separation_difference(
    forged: Sequence[bool], parser: Sequence[bool], llm: Sequence[bool]
) -> Estimate:
    """LLM minus parser v2 in catch rate minus false-alarm rate, with a paired interval.

    Receipts are resampled within each class, so that both classes keep their size (evaluation
    protocol, section 6), and both readers are evaluated on the same draws.

    Raises:
        ValueError: If the sequences differ in length or a class is empty.
    """
    label = np.asarray(forged, dtype=bool)
    if not len(label) == len(parser) == len(llm) or label.all() or not label.any():
        msg = "one alarm per receipt and reader, with authentic and forged receipts, is needed"
        raise ValueError(msg)
    gain = np.asarray(llm, dtype=float) - np.asarray(parser, dtype=float)

    def statistic(rows: np.ndarray) -> float:
        drawn = label[rows]
        return float(gain[rows[drawn]].mean() - gain[rows[~drawn]].mean())

    return stratified(statistic, label.astype(int))


def judge(
    track: str,
    source: TextSource,
    forged: Mapping[str, bool],
    readings: Sequence[ReaderOutput],
) -> TrackResult:
    """Apply the rule of ADR-0027 to one text source.

    Args:
        track: The name of the text source in the report.
        source: The text source; readings of other sources are ignored.
        forged: Every receipt of the test, and whether it is forged.
        readings: One reading per receipt and reader from ``source``.

    Raises:
        ValueError: If a receipt lacks a reading.
    """
    read = {(r.reader, r.doc_id): r for r in readings if r.source is source}
    receipts = sorted(forged)
    if any((reader, d) not in read for reader in READERS for d in receipts):
        msg = f"every receipt needs one reading per reader from {source.value}"
        raise ValueError(msg)
    failed = {r: [failing(read[r, d].extraction.fields) for d in receipts] for r in READERS}
    alarmed = {r: [bool(f) for f in failed[r]] for r in READERS}
    labels = [forged[d] for d in receipts]
    authentic = {r: _of_class(alarmed[r], labels, forged=False) for r in READERS}
    caught = {r: _of_class(alarmed[r], labels, forged=True) for r in READERS}
    return TrackResult(
        decision=decide_by_alarms(
            track, (authentic[PARSER], authentic[LLM]), (caught[PARSER], caught[LLM])
        ),
        alarms={r: Alarms(sum(authentic[r]), sum(caught[r])) for r in READERS},
        separation=separation_difference(labels, alarmed[PARSER], alarmed[LLM]),
        rules={
            r: (
                rule_counts(_of_class(failed[r], labels, forged=False)),
                rule_counts(_of_class(failed[r], labels, forged=True)),
            )
            for r in READERS
        },
        rejected=sum(bool(read[LLM, d].rejections) for d in receipts),
    )


def _of_class[T](values: Sequence[T], labels: Sequence[bool], *, forged: bool) -> list[T]:
    """The values of the receipts of one class (``labels`` true for forged), in order."""
    return [value for value, label in zip(values, labels, strict=True) if label is forged]
