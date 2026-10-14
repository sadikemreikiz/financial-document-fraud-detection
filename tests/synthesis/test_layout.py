import numpy as np
import pytest

from docfraud.data.models import BBox, Entity
from docfraud.reading.ocr import OcrResult, Token
from docfraud.synthesis.layout import Kind, _cells, entities, kind, locate, otsu, words
from tests.synthesis.pages import INK, PAPER, draw_receipt


def test_otsu_separates_ink_from_paper() -> None:
    values = np.array([20] * 50 + [230] * 200, dtype=np.uint8)
    assert 20 < otsu(values) < 230


@pytest.mark.parametrize(
    ("character", "expected"),
    [
        ("7", Kind.DIGIT),
        ("A", Kind.UPPER),
        ("z", Kind.LOWER),
        (".", Kind.SEPARATOR),
        ("é", Kind.SEPARATOR),
    ],
)
def test_characters_are_classed(character: str, expected: Kind) -> None:
    assert kind(character) is expected


def test_the_characters_of_a_printed_word_are_found_between_gaps() -> None:
    page, ocr = draw_receipt()
    token = next(t for t in ocr.tokens if t.text == "48.75")
    characters, method, ink, paper, threshold = locate(page, token.box, token.text)
    assert method == "gaps"
    assert "".join(c.text for c in characters) == "48.75"
    assert [c.box.x for c in characters] == sorted(c.box.x for c in characters)
    assert all(c.box.x >= token.box.x and c.box.x2 <= token.box.x2 for c in characters)
    assert ink < threshold < paper
    assert abs(ink - INK) < 40
    assert abs(paper - PAPER) < 10


def test_ink_of_a_neighbouring_line_is_left_out() -> None:
    page, ocr = draw_receipt()
    token = next(t for t in ocr.tokens if t.text == "70.00")
    tall = BBox(
        x=token.box.x, y=token.box.y - 30, width=token.box.width, height=token.box.height + 30
    )
    plain, _, _, _, _ = locate(page, token.box, token.text)
    reaching, _, _, _, _ = locate(page, tall, token.text)
    assert [c.box for c in reaching] == [c.box for c in plain]


def test_spaces_print_no_character() -> None:
    page, ocr = draw_receipt()
    date = next(t for t in ocr.tokens if t.text == "02/01/2019")
    characters, _, _, _, _ = locate(page, date.box, "02/01 /2019")
    assert len(characters) == 10


def test_words_without_contrast_get_no_characters() -> None:
    page = np.full((40, 80), 200, dtype=np.uint8)
    page[10:30, 10:20] = 190
    characters, method, *_ = locate(page, BBox(x=0, y=0, width=80, height=40), "1")
    assert characters == ()
    assert method == "none"
    assert locate(page, BBox(x=0, y=0, width=80, height=40), " ")[1] == "none"


def _ink(*runs: tuple[int, int]) -> np.ndarray:
    ink = np.zeros((10, 60), dtype=bool)
    for start, end in runs:
        ink[2:8, start:end] = True
    return ink


def test_touching_characters_are_split_and_broken_ones_merged() -> None:
    assert _cells(_ink((0, 8), (12, 20)), 2) == ([(0, 8), (12, 20)], "gaps")
    split = _cells(_ink((0, 8), (12, 28)), 3)  # the second run holds two characters
    assert split is not None
    assert split[1] == "split"
    assert len(split[0]) == 3
    merged = _cells(_ink((0, 4), (5, 8), (14, 22)), 2)  # the first character is broken
    assert merged == ([(0, 8), (14, 22)], "merge")
    speck = _ink((0, 8), (12, 20))
    speck[0, 40] = True  # a single pixel of noise is not a character
    assert _cells(speck, 2) == ([(0, 8), (12, 20)], "gaps")


def test_an_even_division_that_cuts_through_ink_is_refused() -> None:
    # Two runs for three characters, neither wide enough to hold two: an even division would
    # cut through the first run, so the word is not used.
    assert _cells(_ink((0, 10), (12, 22)), 3) is None
    assert _cells(np.zeros((10, 20), dtype=bool), 1) is None


def test_entities_follow_the_layout_of_a_receipt() -> None:
    _, ocr = draw_receipt()
    found = dict(zip((t.text for t in ocr.tokens), entities(ocr), strict=False))
    assert found["MAKMUR"] is Entity.COMPANY
    assert found["02/01/2019"] is Entity.METADATA
    assert found["12.50"] is Entity.PRODUCT  # a header with a label and no amount ends nothing
    assert found["64.45"] is Entity.TOTAL_PAYMENT
    assert found["5.55"] is Entity.TOTAL_PAYMENT
    assert found["TOTAL"] is Entity.OTHER
    assert found["03-1234"] is Entity.OTHER


def test_month_names_are_dates() -> None:
    tokens = (
        Token(text="26-Mar-2018", box=BBox(x=0, y=0, width=50, height=10), confidence=1.0, line=0),
    )
    assert entities(OcrResult(engine="t", width=60, height=20, tokens=tokens)) == [Entity.METADATA]


def test_every_ocr_word_becomes_a_word() -> None:
    page, ocr = draw_receipt()
    found = words(page, ocr)
    assert [w.text for w in found] == [t.text for t in ocr.tokens]
    amount = next(w for w in found if w.text == "64.45")
    assert amount.editable == (0, 1, 3, 4)
    assert amount.baseline == max(c.box.y2 for c in amount.characters)
