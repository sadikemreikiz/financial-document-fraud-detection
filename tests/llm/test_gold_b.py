from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from docfraud.llm.adjudication import GOLD, Kind, Listing, first_answer
from docfraud.llm.gold_b import (
    LLM,
    PARSER,
    Measurement,
    ReaderOutput,
    adjudicated_gold,
    adjudication_counts,
    false_alarms,
    load_measurement,
    questions_on,
    reading_name,
    save_measurement,
    unanswered,
)
from docfraud.llm.reader import Rejection
from docfraud.llm.record import FrozenVersion
from docfraud.llm.run import COMPARISON_SETTINGS
from docfraud.reading.fields import AmountRole, Extraction, FieldSources, ReceiptFields
from docfraud.reading.pipeline import TextSource

D = Decimal
AT = datetime(2026, 10, 22, tzinfo=UTC)
T1, T2 = TextSource.TRANSCRIPT, TextSource.OCR


def fields(**amounts: str) -> ReceiptFields:
    return ReceiptFields.model_validate({"amounts": {r: [v] for r, v in amounts.items()}})


RIGHT = fields(total_due="30.00", tendered="50.00", change="20.00")
WRONG = fields(total_due="3.00", tendered="50.00", change="20.00")


def output(doc_id: str, reader: str, source: TextSource, read: ReceiptFields) -> ReaderOutput:
    return ReaderOutput(
        doc_id=doc_id,
        reader=reader,
        source=source,
        extraction=Extraction(fields=read, sources=FieldSources()),
    )


def four(doc_id: str, llm_transcript: ReceiptFields = RIGHT) -> list[ReaderOutput]:
    """The four readings of a receipt; only the LLM on the transcript may differ."""
    return [
        output(doc_id, LLM, T1, llm_transcript),
        output(doc_id, LLM, T2, RIGHT),
        output(doc_id, PARSER, T1, RIGHT),
        output(doc_id, PARSER, T2, RIGHT),
    ]


def test_questions_name_every_reading_of_a_receipt_in_dispute() -> None:
    found = questions_on({"R1": RIGHT, "R2": RIGHT}, [*four("R1"), *four("R2", WRONG)])
    (question,) = found
    assert (question.key, question.kind) == ("R2:amounts:total_due", Kind.AMOUNTS)
    assert [c.sources for c in question.candidates] == [
        (GOLD, reading_name(LLM, T2), reading_name(PARSER, T1), reading_name(PARSER, T2)),
        (reading_name(LLM, T1),),
    ]
    assert reading_name(LLM, T1) == "LLM transcript"


def test_every_receipt_needs_all_four_readings() -> None:
    with pytest.raises(ValueError, match="R1 needs one reading per reader"):
        questions_on({"R1": RIGHT}, four("R1")[:3])


def test_the_owner_corrects_the_gold_set_where_a_reader_was_right() -> None:
    # The gold annotation is the misread one here: the LLM on the transcript read it right.
    asked = questions_on({"R1": WRONG, "R2": RIGHT}, [*four("R1", RIGHT), *four("R2")])
    (question,) = asked
    listing = Listing(amounts=(D("30.00"),))
    answers = {question.key: first_answer(question, listing, AT, "abc")}
    assert unanswered(asked, answers) == []
    gold = adjudicated_gold({"R1": WRONG, "R2": RIGHT}, asked, answers)
    assert gold == {"R1": RIGHT, "R2": RIGHT}  # a receipt without questions is unchanged
    (amounts, items, tax_rows) = adjudication_counts(asked, answers)
    assert (amounts.questions, amounts.first_matched, amounts.gold_changed) == (1, 1, 1)
    assert amounts.agreed == {
        reading_name(r, s): 1 for r in (LLM, PARSER) for s in TextSource
    }  # the gold annotation alone was wrong
    assert (items.kind, items.questions, tax_rows.questions) == (Kind.ITEMS, 0, 0)


def test_a_question_without_a_final_answer_stops_the_evaluation() -> None:
    asked = questions_on({"R1": RIGHT}, four("R1", WRONG))
    differs = Listing(amounts=(D("31.00"),))  # equals no reading, so not final yet
    answers = {asked[0].key: first_answer(asked[0], differs, AT, None)}
    assert unanswered(asked, answers) == [asked[0].key]
    assert unanswered(asked, {}) == [asked[0].key]
    with pytest.raises(ValueError, match="no final answer"):
        adjudication_counts(asked, answers)
    with pytest.raises(ValueError, match="no final answer"):
        adjudicated_gold({"R1": RIGHT}, asked, answers)


def test_false_alarms_count_authentic_receipts_whose_rules_fail() -> None:
    read = {"A1": WRONG, "A2": RIGHT, "F1": WRONG}  # WRONG breaks paid - due = change
    assert false_alarms(read, frozenset({"A1", "A2"})) == 1
    assert false_alarms({"A2": RIGHT}, frozenset({"A2"})) == 0


def test_the_measurement_is_kept_with_rejections(tmp_path: Path) -> None:
    rejected = output("R1", LLM, T1, ReceiptFields()).model_copy(
        update={"rejections": (Rejection("answer", "", "not answered"),)}
    )
    measurement = Measurement(
        version=FrozenVersion(
            round=2,
            settings=COMPARISON_SETTINGS["claude-haiku-5-5"],
            instruction="extraction-v2.md",
            instruction_sha256="0" * 64,
            commit="abc",
        ),
        commit="def",
        measured_at=AT,
        receipts=1,
        cost_usd=D("0.0123"),
        readings=(rejected, *four("R1")[1:]),
    )
    path = tmp_path / "gold-b" / "measurement.json"
    save_measurement(measurement, path)
    assert load_measurement(path) == measurement
    assert load_measurement(path).readings[0].rejections[0].reason == "not answered"
    assert AmountRole.TOTAL_DUE in load_measurement(path).readings[1].extraction.fields.amounts
