import re

import pytest

from docfraud.data.models import BBox
from docfraud.reading.layout import Row, Segment
from docfraud.reading.sources import locate, normalise, whole_row


@pytest.mark.parametrize(
    "text",
    [
        "Cash :  52.00",
        "  total\t rm 1,234.50  ",
        "",
        "   ",
        "straße 1.00",
        "a\N{NO-BREAK SPACE}b",
        "x\n\ny",
    ],
)
def test_normalise_matches_the_parser_normalisation(text: str) -> None:
    assert normalise(text).text == re.sub(r"\s+", " ", text.upper()).strip()


def test_normalised_spans_map_back_to_the_original() -> None:
    text = "Cash :   52.00 "
    normalised = normalise(text)
    start = normalised.text.index("52.00")
    begin, end = normalised.original_span(start, start + 5)
    assert text[begin:end] == "52.00"


def test_characters_that_grow_when_upper_cased_map_to_their_source() -> None:
    normalised = normalise("ß 1")
    assert normalised.text == "SS 1"
    assert normalised.original_span(0, 2) == (0, 1)
    assert normalised.original_span(3, 4) == (2, 3)


@pytest.mark.parametrize(("start", "end"), [(2, 2), (-1, 1), (0, 9)])
def test_original_span_rejects_empty_or_outside_spans(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="not inside"):
        normalise("ab").original_span(start, end)


BOX_A = BBox(x=0, y=0, width=50, height=10)
BOX_B = BBox(x=100, y=2, width=40, height=10)
ROW = Row((Segment("CASH", BOX_A), Segment("52.00", BOX_B)))


def test_locate_names_the_segments_of_the_span_and_their_box() -> None:
    source = locate([ROW], 0, 5, 10)
    assert ROW.text[source.start : source.end] == "52.00"
    assert source.segments == (1,)
    assert source.box == BOX_B


def test_a_whole_row_touches_every_segment() -> None:
    source = whole_row([ROW], 0)
    assert (source.start, source.end, source.segments) == (0, 10, (0, 1))
    assert source.box == BBox(x=0, y=0, width=140, height=12)


def test_unpositioned_segments_give_no_box() -> None:
    source = locate([Row((Segment("6.50"),))], 0, 0, 4)
    assert source.segments == (0,)
    assert source.box is None


@pytest.mark.parametrize(("start", "end"), [(8, 20), (3, 3)])
def test_locate_rejects_spans_outside_the_row(start: int, end: int) -> None:
    with pytest.raises(ValueError, match="not inside row 0"):
        locate([ROW], 0, start, end)
