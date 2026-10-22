from datetime import date
from decimal import Decimal

import pytest

from docfraud.data.models import BBox
from docfraud.llm.labels import CLOSE, OPEN, LabelKind, label_rows
from docfraud.reading.layout import Row, Segment

D = Decimal


def rows(*lines: str) -> list[Row]:
    return [Row((Segment(line),)) for line in lines]


def tagged(name: str) -> str:
    return f"{OPEN}{name}{CLOSE}"


def test_rows_are_numbered_and_every_number_and_date_is_labelled() -> None:
    labelled = label_rows(rows("CASH : 52.00", "DATE 06-03-2018 07:24"))
    assert labelled.text.splitlines() == [
        f"r0: CASH : {tagged('n1')}52.00",
        f"r1: DATE {tagged('d1')}06-03-2018 {tagged('n2')}07:{tagged('n3')}24",
    ]
    cash, day = labelled.labels["n1"], labelled.labels["d1"]
    assert (cash.kind, cash.number, cash.printed) == (LabelKind.NUMBER, D("52.00"), "52.00")
    assert (day.kind, day.day, day.printed) == (LabelKind.DATE, date(2018, 3, 6), "06-03-2018")
    assert (cash.source.row, cash.source.start, cash.source.end) == (0, 7, 12)


@pytest.mark.parametrize(
    ("line", "printed", "value"),
    [
        ("ROUNDING ADJ: -0.01", "-0.01", "-0.01"),
        ("ADJ - 0.01", "- 0.01", "-0.01"),
        ("DISCOUNT RM-0.20", "-0.20", "-0.20"),
        ("DISCOUNT rm -0.20", "-0.20", "-0.20"),
        ("ROUNDING: -RM0.02", "-RM0.02", "-0.02"),
        ("ROUNDING -RM 0.02", "-RM 0.02", "-0.02"),
        ("ROUNDING - RM0.02", "- RM0.02", "-0.02"),
        ("REFUND 2.00-", "2.00-", "-2.00"),
        ("TP-24", "24", "24"),
        ("BIG ITEM 1,234.50", "1,234.50", "1234.50"),
        ("PRICE 12,50", "12,50", "12.50"),
        ("QTY 1,234", "1,234", "1234"),
        ("EURO 1.234,50", "1.234,50", "1234.50"),
        ("HALF .50", ".50", "0.50"),
    ],
)
def test_signs_and_separators(line: str, printed: str, value: str) -> None:
    (label,) = label_rows(rows(line)).labels.values()
    assert (label.printed, label.number) == (printed, D(value))


def test_the_label_goes_before_the_sign_and_the_source_covers_the_digits() -> None:
    lines = rows("ROUNDING ADJ: -0.01")
    labelled = label_rows(lines)
    assert labelled.text == f"r0: ROUNDING ADJ: {tagged('n1')}-0.01"
    source = labelled.labels["n1"].source
    assert lines[0].text[source.start : source.end] == "0.01"


@pytest.mark.parametrize("digits", ["10.0.0.1", "12,5,1", "12,5013"])
def test_ambiguous_separators_give_no_value(digits: str) -> None:
    (label,) = label_rows(rows(f"CODE {digits}")).labels.values()
    assert (label.printed, label.number) == (digits, None)


def test_a_hyphen_before_more_digits_is_not_a_trailing_minus() -> None:
    labels = label_rows(rows("TEL: 03- 55423228")).labels
    assert [(label.printed, label.number) for label in labels.values()] == [
        ("03", D(3)),
        ("55423228", D(55423228)),
    ]


def test_printed_brackets_stay_apart_from_labels() -> None:
    labelled = label_rows(rows("100 PLUS [1.5L]"))
    assert labelled.text == f"r0: {tagged('n1')}100 PLUS [{tagged('n2')}1.5L]"


def test_sources_carry_the_segment_and_its_box() -> None:
    box = BBox(x=200, y=50, width=60, height=14)
    labelled = label_rows([Row((Segment("TOTAL"), Segment("6.50", box)))])
    source = labelled.labels["n1"].source
    assert (source.segments, source.box) == ((1,), box)


def test_no_rows_give_no_text_and_no_labels() -> None:
    labelled = label_rows([])
    assert (labelled.text, dict(labelled.labels)) == ("", {})
