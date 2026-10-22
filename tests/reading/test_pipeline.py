from decimal import Decimal
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from docfraud.data.models import BBox, Document
from docfraud.reading.fields import AmountRole
from docfraud.reading.ocr import OcrResult, Token
from docfraud.reading.pipeline import TextSource, ocr_segments, read_document


def token(text: str, x: int, y: int, line: int) -> Token:
    return Token(
        text=text, box=BBox(x=x, y=y, width=12 * len(text), height=14), confidence=0.9, line=line
    )


class FakeEngine:
    identifier = "fake"

    def __init__(self, tokens: tuple[Token, ...]) -> None:
        self.tokens = tokens

    def recognize(self, image: npt.NDArray[np.uint8]) -> OcrResult:
        height, width = image.shape
        return OcrResult(engine="fake", width=width, height=height, tokens=self.tokens)


# The OCR misreads the total (6.5O); the transcript has it right and lists lines out of order.
OCR_TOKENS = (
    token("TOTAL", 10, 50, 0),
    token("6.5O", 200, 50, 1),
    token("CASH", 10, 80, 2),
    token("10.00", 200, 80, 3),
)


@pytest.fixture
def document(tmp_path: Path) -> Document:
    image = tmp_path / "r.png"
    Image.new("L", (300, 120), 255).save(image)
    transcript = tmp_path / "r.txt"
    transcript.write_text("TOTAL\nCASH\n6.50\n10.00\n", encoding="utf-8")
    return Document(
        dataset="toy", doc_id="r", source_id="toy/r", image_path=image, transcript_path=transcript
    )


def test_ocr_segments_join_words_per_line() -> None:
    segments = ocr_segments(OcrResult(engine="e", width=300, height=120, tokens=OCR_TOKENS[:2]))
    assert [s.text for s in segments] == ["TOTAL", "6.5O"]


def test_transcript_source_uses_transcript_text_with_ocr_layout(document: Document) -> None:
    reading = read_document(document, FakeEngine(OCR_TOKENS), TextSource.TRANSCRIPT)
    assert [r.text for r in reading.rows] == ["TOTAL 6.50", "CASH 10.00"]
    assert reading.fields.amounts[AmountRole.TOTAL_DUE] == (Decimal("6.50"),)
    (total,) = reading.sources.amounts[AmountRole.TOTAL_DUE]
    assert (total.row, total.segments, total.box) == (0, (1,), OCR_TOKENS[1].box)


def test_ocr_source_reads_what_the_engine_saw(document: Document) -> None:
    reading = read_document(document, FakeEngine(OCR_TOKENS), TextSource.OCR)
    assert [r.text for r in reading.rows] == ["TOTAL 6.5O", "CASH 10.00"]
    assert AmountRole.TOTAL_DUE not in reading.fields.amounts
    assert reading.fields.amounts[AmountRole.TENDERED] == (Decimal("10.00"),)


def test_transcript_source_requires_a_transcript(document: Document) -> None:
    without = document.model_copy(update={"transcript_path": None})
    with pytest.raises(ValueError, match="no transcript"):
        read_document(without, FakeEngine(OCR_TOKENS), TextSource.TRANSCRIPT)
