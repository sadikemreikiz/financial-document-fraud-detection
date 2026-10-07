import re
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image, ImageDraw, ImageFont

from docfraud.data.models import BBox
from docfraud.reading.ocr import (
    CachedOcr,
    OcrResult,
    RapidOcrEngine,
    RapidOcrSettings,
    Token,
    tokens_from_word_results,
)


def quad(x0: float, y0: float, x1: float, y1: float) -> list[list[float]]:
    return [[x0, y0], [x1, y0], [x1, y1], [x0, y1]]


def test_word_results_become_tokens_with_lines() -> None:
    result = tokens_from_word_results(
        "engine",
        100,
        50,
        [
            [("TOTAL", 0.99, quad(10, 5, 40, 15)), ("12.50", 0.97, quad(60.2, 5.7, 90.1, 15))],
            [("CASH", 1.2, quad(10, 20, 30, 30))],
        ],
    )
    assert [t.text for t in result.tokens] == ["TOTAL", "12.50", "CASH"]
    assert [t.line for t in result.tokens] == [0, 0, 1]
    assert result.tokens[1].box == BBox(x=60, y=5, width=31, height=10)
    assert result.tokens[2].confidence == 1.0
    assert result.lines == ["TOTAL 12.50", "CASH"]
    assert result.text == "TOTAL 12.50\nCASH\n"


def test_boxes_are_clipped_and_unusable_words_dropped() -> None:
    result = tokens_from_word_results(
        "engine",
        50,
        20,
        [
            [
                ("EDGE", 0.9, quad(-5, -3, 60, 25)),
                ("  ", 0.9, quad(1, 1, 5, 5)),
                ("NOBOX", 0.9, None),
                ("OUTSIDE", 0.9, quad(70, 1, 80, 5)),
            ]
        ],
    )
    assert [t.text for t in result.tokens] == ["EDGE"]
    assert result.tokens[0].box == BBox(x=0, y=0, width=50, height=20)


def test_empty_page() -> None:
    result = tokens_from_word_results("engine", 10, 10, [])
    assert result.tokens == ()
    assert result.text == ""


class CountingEngine:
    identifier = "fake-1"

    def __init__(self) -> None:
        self.calls = 0

    def recognize(self, image: npt.NDArray[np.uint8]) -> OcrResult:
        self.calls += 1
        token = Token(
            text=str(int(image.sum())), box=BBox(x=0, y=0, width=1, height=1), confidence=1, line=0
        )
        return OcrResult(
            engine=self.identifier, width=image.shape[1], height=image.shape[0], tokens=(token,)
        )


def test_cache_runs_the_engine_once_per_page(tmp_path: Path) -> None:
    engine = CountingEngine()
    cached = CachedOcr(engine, tmp_path)
    page = np.full((4, 3), 7, dtype=np.uint8)
    first = cached.recognize(page)
    second = cached.recognize(page.copy())
    assert first == second
    assert engine.calls == 1
    assert cached.identifier == "fake-1"
    assert len(list((tmp_path / "fake-1").glob("*.json"))) == 1


def test_cache_distinguishes_content_and_shape(tmp_path: Path) -> None:
    engine = CountingEngine()
    cached = CachedOcr(engine, tmp_path)
    cached.recognize(np.zeros((4, 3), dtype=np.uint8))
    cached.recognize(np.ones((4, 3), dtype=np.uint8))
    cached.recognize(np.zeros((3, 4), dtype=np.uint8))
    assert engine.calls == 3


def test_rapidocr_identifier_reflects_settings() -> None:
    engine = RapidOcrEngine(RapidOcrSettings(model_type="medium", max_side_len=4000))
    assert re.fullmatch(r"rapidocr-[\d.]+-PP-OCRv6-medium-4000", engine.identifier)


def render_receipt() -> npt.NDArray[np.uint8]:
    image = Image.new("L", (520, 200), 255)
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=44)
    draw.text((30, 40), "TOTAL", fill=0, font=font)
    draw.text((300, 40), "12.50", fill=0, font=font)
    draw.text((30, 120), "CASH 20.00", fill=0, font=font)
    return np.asarray(image, dtype=np.uint8)


@pytest.mark.models
def test_rapidocr_reads_a_rendered_receipt() -> None:
    page = render_receipt()
    result = RapidOcrEngine().recognize(page)
    texts = [t.text for t in result.tokens]
    assert "12.50" in texts
    assert "20.00" in "".join(texts)
    assert (result.width, result.height) == (520, 200)
    assert all(t.box.fits_within(520, 200) for t in result.tokens)
    total = next(t for t in result.tokens if t.text == "12.50")
    assert 250 <= total.box.x <= 330
    assert total.line != next(t for t in result.tokens if "20.00" in t.text).line
