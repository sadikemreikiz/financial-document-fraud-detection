from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from docfraud.llm.adjudication import (
    GOLD,
    AdjudicationLog,
    Kind,
    Listing,
    adjudicated,
    first_answer,
    in_order,
    load_log,
    load_questions,
    questions_for,
    save_log,
    save_questions,
    shown_candidates,
)
from docfraud.reading.fields import AmountRole, LineItem, ReceiptFields, TaxSummaryRow

D = Decimal
AT = datetime(2026, 10, 22, tzinfo=UTC)


def fields(**values: object) -> ReceiptFields:
    return ReceiptFields.model_validate(values)


GOLD_FIELDS = fields(
    amounts={"total_due": ["30.00"], "tendered": ["52.00"]},
    items=[
        {"description": "TAUCU KISAR", "quantity": "2", "unit_price": "9.50", "amount": "19.00"}
    ],
    tax_summary=[{"code": "SR", "rate_percent": "6", "taxable_amount": "13.00", "tax": "0.78"}],
)


def readings(**others: ReceiptFields) -> dict[str, ReceiptFields]:
    return {GOLD: GOLD_FIELDS, **others}


def test_nothing_is_asked_when_all_readings_agree() -> None:
    same = GOLD_FIELDS.model_copy(update={"merchant": "ANOTHER NAME"})  # not a rule input
    assert questions_for("R1", readings(llm=same, parser=GOLD_FIELDS)) == []


def test_amounts_are_asked_per_role_with_the_readings_grouped() -> None:
    misread = GOLD_FIELDS.model_copy(
        update={"amounts": {AmountRole.TOTAL_DUE: (D("3.00"),), AmountRole.TENDERED: (D("52.00"),)}}
    )
    (question,) = questions_for("R1", readings(llm=GOLD_FIELDS, parser=misread))
    assert (question.key, question.kind, question.role) == (
        "R1:amounts:total_due",
        Kind.AMOUNTS,
        AmountRole.TOTAL_DUE,
    )
    assert [(c.sources, c.listing.amounts) for c in question.candidates] == [
        ((GOLD, "llm"), (D("30.00"),)),
        (("parser",), (D("3.00"),)),
    ]


def test_items_and_tax_rows_are_asked_as_whole_lists() -> None:
    other = GOLD_FIELDS.model_copy(update={"items": (), "tax_summary": ()})
    kinds = [q.kind for q in questions_for("R1", readings(llm=other))]
    assert kinds == [Kind.ITEMS, Kind.TAX_ROWS]


def test_the_gold_annotation_must_be_among_the_readings() -> None:
    with pytest.raises(ValueError, match="gold annotation"):
        questions_for("R1", {"llm": GOLD_FIELDS})


def test_listings_compare_without_order_and_by_the_matching_rules() -> None:
    assert Listing(amounts=(D(1), D(2))).same_as(Listing(amounts=(D("2.00"), D(1))))
    assert not Listing(amounts=(D(1),)).same_as(Listing(amounts=(D(1), D(1))))
    printed = LineItem(quantity=D(2), unit_price=D("9.50"), amount=D("19.00"))
    assert Listing(items=(printed,)).same_as(Listing(items=(LineItem(amount=D("19.00")),)))
    assert not Listing(items=(printed,)).same_as(Listing(items=(LineItem(amount=D("19.01")),)))


def test_the_order_of_questions_and_candidates_is_fixed_and_hides_the_sources() -> None:
    misread = GOLD_FIELDS.model_copy(update={"amounts": {}, "items": (), "tax_summary": ()})
    questions = questions_for("R1", readings(llm=misread)) + questions_for(
        "R2", readings(llm=misread)
    )
    assert [q.key for q in in_order(questions)] == [
        q.key for q in in_order(list(reversed(questions)))
    ]
    assert sorted(q.key for q in in_order(questions)) == sorted(q.key for q in questions)
    shown = shown_candidates(questions[0])
    assert all(isinstance(listing, Listing) for listing in shown)
    assert shown == shown_candidates(questions[0])


def test_a_first_answer_equal_to_a_candidate_is_final() -> None:
    misread = GOLD_FIELDS.model_copy(
        update={"amounts": {AmountRole.TOTAL_DUE: (D("3.00"),), AmountRole.TENDERED: (D("52.00"),)}}
    )
    (question,) = questions_for("R1", readings(llm=misread))
    final = first_answer(question, Listing(amounts=(D("30.00"),)), AT, "abc")
    assert (final.matched, final.done) == (True, True)
    pending = first_answer(question, Listing(amounts=(D("33.00"),)), AT, None)
    assert (pending.matched, pending.done) == (False, False)


def test_final_answers_replace_what_they_answer_in_the_gold_annotation() -> None:
    misread = GOLD_FIELDS.model_copy(
        update={"amounts": {AmountRole.TENDERED: (D("52.00"),)}, "items": (), "tax_summary": ()}
    )
    questions = questions_for("R1", readings(llm=misread))
    answers = {
        "R1:amounts:total_due": first_answer(questions[0], Listing(), AT, None),  # not printed
        "R1:items": first_answer(
            questions[1],
            Listing(items=(LineItem(quantity=D(2), unit_price=D("9.50"), amount=D("19.00")),)),
            AT,
            None,
        ),
        "R1:tax_rows": first_answer(
            questions[2],
            Listing(
                tax_rows=(
                    TaxSummaryRow(rate_percent=D(6), taxable_amount=D("13.00"), tax=D("0.78")),
                )
            ),
            AT,
            None,
        ),
    }
    result = adjudicated(GOLD_FIELDS, questions, answers)
    assert AmountRole.TOTAL_DUE not in result.amounts
    assert result.amounts[AmountRole.TENDERED] == (D("52.00"),)
    assert result.items[0].description == "TAUCU KISAR"  # kept from the matching gold item
    assert result.tax_summary[0].code == "SR"
    with pytest.raises(ValueError, match="no final answer"):
        adjudicated(GOLD_FIELDS, questions, {})


def test_logs_and_questions_are_kept_on_disk(tmp_path: Path) -> None:
    log = AdjudicationLog(reviewer="owner")
    save_log(log, tmp_path / "log.json")
    assert load_log(tmp_path / "log.json", "owner") == log
    assert load_log(tmp_path / "none.json", "owner") == AdjudicationLog(reviewer="owner")
    with pytest.raises(ValueError, match="belongs to owner"):
        load_log(tmp_path / "log.json", "someone else")
    questions = questions_for("R1", readings(llm=GOLD_FIELDS.model_copy(update={"items": ()})))
    save_questions(questions, tmp_path / "questions.json")
    assert load_questions(tmp_path / "questions.json") == questions
