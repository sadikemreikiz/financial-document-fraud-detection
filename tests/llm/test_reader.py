import json
from decimal import Decimal
from typing import Any

import pytest

from docfraud.llm.answer import Answer
from docfraud.llm.labels import LabelledRows, label_rows
from docfraud.llm.provider import Response, Usage
from docfraud.llm.reader import Rejection, read_answer, read_response
from docfraud.reading.fields import AmountRole, ReceiptFields, TaxSummaryRow
from docfraud.reading.layout import Row, Segment

D = Decimal


def rows(*lines: str) -> list[Row]:
    return [Row((Segment(line),)) for line in lines]


LINES = rows(
    "SEMBOYAN TEGAS SDN BHD",
    "DATE : 06-03-2018 07:24:06",
    "TAUCU KISAR",
    "9556655112694 2 X 9.50 19.00 SR",
    "TOTAL QTY: 2",
    "TOTAL ROUNDED 19.00",
    "CASH : 52.00",
    "CHANGE : 33.00",
    "SR 6% 17.92 1.08",
    "DATE 06-03-2018",
)
LABELLED = label_rows(LINES)


def name(labelled: LabelledRows, row: int, printed: str) -> str:
    """The label of the number or date printed as ``printed`` on ``row``."""
    (found,) = [
        label.name
        for label in labelled.labels.values()
        if label.source.row == row and label.printed == printed
    ]
    return found


def n(row: int, printed: str) -> str:
    """The label of a number or date on the rows of ``LINES``."""
    return name(LABELLED, row, printed)


def answer(**change: Any) -> Answer:
    form: dict[str, Any] = {
        "merchant_rows": [0],
        "dates": [n(9, "06-03-2018"), n(1, "06-03-2018")],
        "items": [
            {
                "rows": [2, 3],
                "quantity": n(3, "2"),
                "unit_price": n(3, "9.50"),
                "amount": n(3, "19.00"),
            }
        ],
        "amounts": [
            {"role": "change", "label": n(7, "33.00")},
            {"role": "tendered", "label": n(6, "52.00")},
            {"role": "total_due", "label": n(5, "19.00")},
        ],
        "tax_summary": [
            {"rate_percent": n(8, "6"), "taxable_amount": n(8, "17.92"), "tax": n(8, "1.08")}
        ],
        "item_count": None,
        "quantity_total": n(4, "2"),
    }
    return Answer.model_validate(form | change)


def test_values_and_their_sources_come_from_the_labelled_text() -> None:
    reading = read_answer(answer(), LABELLED, LINES)
    assert reading.rejections == ()
    fields = reading.fields
    assert fields.merchant == "SEMBOYAN TEGAS SDN BHD"
    assert fields.dates == ("2018-03-06",)  # printed twice, kept once
    (item,) = fields.items
    assert (item.quantity, item.unit_price, item.amount) == (D(2), D("9.50"), D("19.00"))
    assert item.description == "TAUCU KISAR X SR"  # own numbers and the barcode left out
    assert fields.amounts == {
        AmountRole.CHANGE: (D("33.00"),),
        AmountRole.TENDERED: (D("52.00"),),
        AmountRole.TOTAL_DUE: (D("19.00"),),
    }
    assert fields.tax_summary == (
        TaxSummaryRow(rate_percent=D(6), taxable_amount=D("17.92"), tax=D("1.08")),
    )
    assert (fields.item_count, fields.quantity_total) == (None, D(2))
    sources = reading.extraction.sources
    assert [s.row for s in sources.dates] == [1]  # the first printing
    assert [s.row for s in sources.items[0].description] == [2, 3]
    (tendered,) = sources.amounts[AmountRole.TENDERED]
    assert LINES[6].text[tendered.start : tendered.end] == "52.00"


def test_amounts_of_one_role_are_put_in_print_order() -> None:
    lines = rows("TOTAL 6.50", "TOTAL 7.00")
    labelled = label_rows(lines)
    form = Answer(
        merchant_rows=[],
        dates=[],
        items=[],
        amounts=[
            {"role": "total_due", "label": name(labelled, 1, "7.00")},  # type: ignore[list-item]
            {"role": "total_due", "label": name(labelled, 0, "6.50")},  # type: ignore[list-item]
        ],
        tax_summary=[],
        item_count=None,
        quantity_total=None,
    )
    reading = read_answer(form, labelled, lines)
    assert reading.fields.amounts[AmountRole.TOTAL_DUE] == (D("6.50"), D("7.00"))
    assert [s.row for s in reading.extraction.sources.amounts[AmountRole.TOTAL_DUE]] == [0, 1]


def test_references_that_do_not_hold_are_rejected() -> None:
    lines = [*LINES, *rows("CODE 10.0.0.1")]
    labelled = label_rows(lines)
    date = name(labelled, 1, "06-03-2018")
    reading = read_answer(
        answer(
            merchant_rows=[0, 0, 50],
            dates=[name(labelled, 6, "52.00")],
            amounts=[
                {"role": "tendered", "label": "n999"},
                {"role": "change", "label": date},
                {"role": "total_due", "label": name(labelled, 10, "10.0.0.1")},
            ],
            item_count=name(labelled, 3, "9.50"),
        ),
        labelled,
        lines,
    )
    assert reading.rejections == (
        Rejection("merchant_rows", "50", "no such row"),
        Rejection("dates[0]", name(labelled, 6, "52.00"), "not a date"),
        Rejection("amounts[0]", "n999", "no such label"),
        Rejection("amounts[1]", date, "not a number"),
        Rejection("amounts[2]", name(labelled, 10, "10.0.0.1"), "separators cannot be read"),
        Rejection("item_count", name(labelled, 3, "9.50"), "not a whole number"),
    )
    fields = reading.fields
    assert (fields.merchant, fields.dates, fields.amounts, fields.item_count) == (
        "SEMBOYAN TEGAS SDN BHD",
        (),
        {},
        None,
    )


def test_a_whole_number_count() -> None:
    reading = read_answer(answer(item_count=name(LABELLED, 4, "2")), LABELLED, LINES)
    assert reading.fields.item_count == 2


def response(answer_text: str | None, **change: Any) -> Response:
    base: dict[str, Any] = {
        "key": "k",
        "model": "claude-haiku-5-5",
        "answer": answer_text,
        "stop_reason": "end_turn",
        "usage": Usage(),
        "batch": True,
    }
    return Response.model_validate(base | change)


@pytest.mark.parametrize(
    ("stored", "reason"),
    [
        (response(None, stop_reason="refusal"), "refusal"),
        (response(None, stop_reason=None, error="overloaded_error"), "overloaded_error"),
        (response(None, stop_reason=None), "no answer"),
        (response('{"total": "19.00"}'), "not in the answer form"),
    ],
)
def test_a_response_without_a_valid_answer_reads_nothing(stored: Response, reason: str) -> None:
    reading = read_response(stored, LABELLED, LINES)
    assert reading.fields == ReceiptFields()
    assert reading.rejections == (Rejection("answer", "", reason),)


def test_a_stored_answer_reads_like_the_answer_itself() -> None:
    form = answer()
    stored = response(json.dumps(form.model_dump(mode="json")))
    assert read_response(stored, LABELLED, LINES) == read_answer(form, LABELLED, LINES)
