from docfraud.data.models import BBox
from docfraud.reading.align import align_transcript
from docfraud.reading.layout import group_rows
from docfraud.reading.ocr import OcrResult, Token


def page(*words: tuple[str, int, int, int]) -> OcrResult:
    """Words as (text, x, y, line)."""
    tokens = tuple(
        Token(text=t, box=BBox(x=x, y=y, width=10 * len(t), height=12), confidence=1, line=line)
        for t, x, y, line in words
    )
    return OcrResult(engine="test", width=400, height=400, tokens=tokens)


RECEIPT = page(
    ("TOTAL", 10, 100, 0),
    ("QTY:", 70, 100, 0),
    ("6", 120, 100, 0),
    ("TOTAL", 10, 130, 1),
    ("6.50", 300, 130, 2),
    ("CASH", 10, 160, 3),
    ("6.50", 300, 160, 4),
)


def boxes_y(lines: list[str]) -> list[int | None]:
    return [s.box.y if s.box else None for s in align_transcript(lines, RECEIPT)]


def test_multi_word_lines_are_located() -> None:
    assert boxes_y(["TOTAL QTY: 6"]) == [100]


def test_repeated_amounts_follow_their_transcript_neighbours() -> None:
    # Transcript order lists each amount right after its label, as the dataset usually does.
    assert boxes_y(["TOTAL QTY: 6", "TOTAL", "6.50", "CASH", "6.50"]) == [100, 130, 130, 160, 160]


def test_rows_rebuilt_from_aligned_transcript() -> None:
    segments = align_transcript(["CASH", "6.50", "TOTAL", "6.50"], RECEIPT)
    assert [r.text for r in group_rows(segments)] == ["TOTAL 6.50", "CASH 6.50"]


def test_lines_without_a_match_stay_unplaced() -> None:
    assert boxes_y(["THANK YOU", "CASH"]) == [None, 160]


def test_tokens_are_not_used_twice() -> None:
    # Three transcript "6.50" lines but only two printed ones: the third stays unplaced.
    placed = boxes_y(["6.50", "6.50", "6.50"])
    assert sorted(y for y in placed if y is not None) == [130, 160]
    assert placed.count(None) == 1


def test_ocr_noise_is_tolerated() -> None:
    noisy = page(("T0TAL", 10, 50, 0), ("12.5O", 200, 50, 0))
    assert [s.box is not None for s in align_transcript(["TOTAL 12.50"], noisy)] == [True]


def test_empty_inputs() -> None:
    assert align_transcript([], RECEIPT) == []
    assert align_transcript(["TOTAL"], page())[0].box is None
