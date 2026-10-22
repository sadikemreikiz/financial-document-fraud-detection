"""Adjudication of gold set B: what is in dispute, what the owner answered, and the result.

ADR-0025 (decision 4) and ADR-0026: on gold set B, five readings of every receipt are compared,
the gold annotation and the LLM and parser v2 on both text sources. Where they do not all agree,
the owner reads the receipt image and answers first without seeing any candidate. Amounts are
asked per role; items and tax summary rows as the receipt's whole list. If the first answer
equals one candidate it is final; otherwise the candidates are shown without their sources and
the owner gives the final answer. The adjudicated gold set is the gold annotation with every
final answer put in place of the list it answers.
"""

import json
import random
from collections.abc import Callable, Mapping, Sequence
from datetime import datetime
from decimal import Decimal
from enum import StrEnum
from pathlib import Path

from pydantic import BaseModel, ConfigDict

from docfraud.evaluation.extraction import item_matches, maximum_matching, tax_row_matches
from docfraud.reading.fields import AmountRole, LineItem, ReceiptFields, TaxSummaryRow

GOLD = "gold"
#: The seed of the question order, fixed before any question exists.
ORDER_SEED = 20261022


class Kind(StrEnum):
    """What a question asks for."""

    AMOUNTS = "amounts"  # every amount of one role
    ITEMS = "items"  # the receipt's whole item list
    TAX_ROWS = "tax_rows"  # the receipt's whole tax summary


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Listing(_Frozen):
    """A list of values for one question: amounts, items or tax summary rows."""

    amounts: tuple[Decimal, ...] = ()
    items: tuple[LineItem, ...] = ()
    tax_rows: tuple[TaxSummaryRow, ...] = ()

    def same_as(self, other: Listing) -> bool:
        """Equal under the matching rules of the annotation guidelines, section 7."""
        return (
            sorted(self.amounts) == sorted(other.amounts)
            and _same(self.items, other.items, item_matches)
            and _same(self.tax_rows, other.tax_rows, tax_row_matches)
        )


class Candidate(_Frozen):
    """One distinct listing and the readings that gave it; the owner never sees ``sources``."""

    sources: tuple[str, ...]
    listing: Listing


class Question(_Frozen):
    """One thing in dispute on one receipt."""

    key: str
    doc_id: str
    kind: Kind
    role: AmountRole | None = None
    candidates: tuple[Candidate, ...]


class Answer(_Frozen):
    """The owner's answer: first without candidates, then final.

    ``matched`` says whether the first answer equalled a candidate, which made it final.
    """

    first: Listing
    final: Listing | None = None
    matched: bool = False
    answered_at: datetime
    commit: str | None = None

    @property
    def done(self) -> bool:
        """Whether the answer is final."""
        return self.final is not None


class AdjudicationLog(_Frozen):
    """Every answer of one owner, keyed by question."""

    reviewer: str
    answers: Mapping[str, Answer] = {}


def questions_for(doc_id: str, readings: Mapping[str, ReceiptFields]) -> list[Question]:
    """What is in dispute on one receipt among its readings (the gold set among them).

    Raises:
        ValueError: If the gold annotation is not among the readings.
    """
    if GOLD not in readings:
        msg = f"the readings of {doc_id} must include the gold annotation"
        raise ValueError(msg)
    found: list[Question] = []
    roles = sorted({role for fields in readings.values() for role in fields.amounts})
    for role in roles:
        listings = {name: Listing(amounts=f.amounts.get(role, ())) for name, f in readings.items()}
        found += _question(doc_id, Kind.AMOUNTS, listings, role)
    found += _question(
        doc_id, Kind.ITEMS, {name: Listing(items=f.items) for name, f in readings.items()}
    )
    found += _question(
        doc_id,
        Kind.TAX_ROWS,
        {name: Listing(tax_rows=f.tax_summary) for name, f in readings.items()},
    )
    return found


def in_order(questions: Sequence[Question]) -> list[Question]:
    """The questions in the fixed random order in which the owner answers them."""
    ordered = sorted(questions, key=lambda q: q.key)
    random.Random(ORDER_SEED).shuffle(ordered)  # noqa: S311 - a fixed order, not a secret
    return ordered


def shown_candidates(question: Question) -> list[Listing]:
    """The candidate listings as the owner sees them: without sources, in a fixed random order."""
    listings = [candidate.listing for candidate in question.candidates]
    random.Random(f"{ORDER_SEED}/{question.key}").shuffle(listings)  # noqa: S311 - fixed order
    return listings


def first_answer(question: Question, listing: Listing, at: datetime, commit: str | None) -> Answer:
    """Record a first answer; it is final if it equals a candidate."""
    matched = any(listing.same_as(c.listing) for c in question.candidates)
    return Answer(
        first=listing,
        final=listing if matched else None,
        matched=matched,
        answered_at=at,
        commit=commit,
    )


def adjudicated(
    gold: ReceiptFields, questions: Sequence[Question], answers: Mapping[str, Answer]
) -> ReceiptFields:
    """The gold annotation of one receipt with every final answer put in place.

    Raises:
        ValueError: If a question of the receipt has no final answer.
    """
    amounts = dict(gold.amounts)
    items, tax_rows = gold.items, gold.tax_summary
    for question in questions:
        answer = answers.get(question.key)
        if answer is None or answer.final is None:
            msg = f"question {question.key} has no final answer"
            raise ValueError(msg)
        final = answer.final
        if question.kind is Kind.AMOUNTS and question.role is not None:
            amounts.pop(question.role, None)
            if final.amounts:
                amounts[question.role] = final.amounts
        elif question.kind is Kind.ITEMS:
            items = _with_gold_text(final.items, gold.items)
        else:
            tax_rows = _with_gold_codes(final.tax_rows, gold.tax_summary)
    return gold.model_copy(update={"amounts": amounts, "items": items, "tax_summary": tax_rows})


def load_log(path: Path, reviewer: str) -> AdjudicationLog:
    """The owner's answers so far, or an empty log.

    Raises:
        ValueError: If the log belongs to another reviewer.
    """
    if not path.exists():
        return AdjudicationLog(reviewer=reviewer)
    log = AdjudicationLog.model_validate_json(path.read_text(encoding="utf-8"))
    if log.reviewer != reviewer:
        msg = f"{path} belongs to {log.reviewer}, not {reviewer}"
        raise ValueError(msg)
    return log


def save_log(log: AdjudicationLog, path: Path) -> None:
    """Write the log, replacing the file only once the new one is complete."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(".tmp")
    temporary.write_text(log.model_dump_json(indent=1) + "\n", encoding="utf-8")
    temporary.replace(path)


def save_questions(questions: Sequence[Question], path: Path) -> None:
    """Keep the questions; they hold the candidates, so the owner does not open this file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = [q.model_dump(mode="json") for q in questions]
    path.write_text(json.dumps(payload, indent=1) + "\n", encoding="utf-8")


def load_questions(path: Path) -> list[Question]:
    """The questions kept by :func:`save_questions`."""
    return [Question.model_validate(q) for q in json.loads(path.read_text(encoding="utf-8"))]


def _question(
    doc_id: str, kind: Kind, listings: Mapping[str, Listing], role: AmountRole | None = None
) -> list[Question]:
    candidates: list[Candidate] = []
    for name, listing in listings.items():
        for n, candidate in enumerate(candidates):
            if candidate.listing.same_as(listing):
                candidates[n] = Candidate(
                    sources=(*candidate.sources, name), listing=candidate.listing
                )
                break
        else:
            candidates.append(Candidate(sources=(name,), listing=listing))
    if len(candidates) == 1:
        return []
    key = f"{doc_id}:{kind.value}" + (f":{role.value}" if role is not None else "")
    return [Question(key=key, doc_id=doc_id, kind=kind, role=role, candidates=tuple(candidates))]


def _same[T](a: Sequence[T], b: Sequence[T], matches: Callable[[T, T], bool]) -> bool:
    """Whether two lists pair up completely; both directions, as matching may be one-sided."""
    if len(a) != len(b):
        return False
    return len(maximum_matching(a, b, matches)) == len(maximum_matching(b, a, matches)) == len(a)


def _with_gold_text(items: Sequence[LineItem], gold: Sequence[LineItem]) -> tuple[LineItem, ...]:
    """Final items, each with the description of the gold item it matches unless typed."""
    pairs = dict(maximum_matching(items, gold, item_matches))
    return tuple(
        item
        if item.description or n not in pairs
        else item.model_copy(update={"description": gold[pairs[n]].description})
        for n, item in enumerate(items)
    )


def _with_gold_codes(
    rows: Sequence[TaxSummaryRow], gold: Sequence[TaxSummaryRow]
) -> tuple[TaxSummaryRow, ...]:
    """Final tax rows, each with the code of the gold row it matches."""
    pairs = dict(maximum_matching(rows, gold, tax_row_matches))
    return tuple(
        row if n not in pairs else row.model_copy(update={"code": gold[pairs[n]].code})
        for n, row in enumerate(rows)
    )
