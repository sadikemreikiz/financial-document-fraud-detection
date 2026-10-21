import numpy as np
import pytest

from docfraud.data.models import BBox
from docfraud.forensics.learned.text_scale import (
    FIXED_SIDE,
    TEXT_HEIGHT_OTHER,
    TEXT_HEIGHT_PAGE_SCAN,
    WordScale,
    block_peak,
    highest_square_mean,
    on_page,
    page_scan,
    square_side,
    text_height,
    word_peaks,
    word_sides,
)


def box(x: int, y: int, width: int, height: int) -> BBox:
    return BBox(x=x, y=y, width=width, height=height)


def brute_force(values: np.ndarray, side: int) -> float:
    height, width = values.shape
    return max(
        float(values[y : y + side, x : x + side].mean())
        for y in range(height - side + 1)
        for x in range(width - side + 1)
    )


def test_a_page_scan_has_a_longer_side_above_6000_px() -> None:
    assert page_scan(6001, 4000)
    assert page_scan(4000, 6001)
    assert not page_scan(6000, 6000)


def test_the_text_height_is_the_median_height_of_the_word_boxes() -> None:
    words = [box(0, 0, 30, 20), box(0, 30, 30, 40), box(0, 80, 30, 25), box(0, 120, 30, 31)]
    assert text_height(words, 200, 100) == 28.0


def test_boxes_that_start_outside_the_page_are_not_measured() -> None:
    words = [box(0, 0, 30, 20), box(100, 0, 10, 90), box(0, 200, 10, 90), box(90, 190, 40, 30)]
    assert on_page(words, 200, 100) == [words[0], words[3]]
    assert text_height(words, 200, 100) == 25.0


def test_a_receipt_without_words_takes_the_height_of_its_kind() -> None:
    assert text_height([], 800, 400) == TEXT_HEIGHT_OTHER == 40.0
    assert text_height([], 7016, 4961) == TEXT_HEIGHT_PAGE_SCAN == 91.0
    assert text_height([box(500, 0, 10, 10)], 800, 400) == TEXT_HEIGHT_OTHER


@pytest.mark.parametrize(
    ("height", "divisor", "side"),
    [(40.0, 4, 10), (90.0, 4, 22), (90.0, 2, 45), (42.0, 4, 10), (46.0, 4, 12), (2.0, 4, 1)],
)
def test_the_side_of_a_square_follows_the_text_height(
    height: float, divisor: int, side: int
) -> None:
    assert square_side(height, divisor) == side


@pytest.mark.parametrize("side", [1, 2, 3, 7, 12])
def test_the_highest_square_mean_is_that_of_every_position(side: int) -> None:
    values = np.random.default_rng(side).random((23, 31))
    assert highest_square_mean(values, side) == pytest.approx(brute_force(values, side), abs=1e-12)


def test_squares_are_measured_across_bands_of_rows(monkeypatch: pytest.MonkeyPatch) -> None:
    from docfraud.forensics.learned import text_scale  # noqa: PLC0415 - patched below

    monkeypatch.setattr(text_scale, "_ROWS", 4)
    values = np.zeros((40, 30))
    values[17:22, 9:14] = 1.0  # a square that starts in one band and ends in the next
    assert highest_square_mean(values, 5) == pytest.approx(1.0, abs=1e-12)
    values = np.random.default_rng(1).random((40, 30))
    assert highest_square_mean(values, 6) == pytest.approx(brute_force(values, 6), abs=1e-12)


def test_a_square_of_one_pixel_gives_the_highest_value() -> None:
    values = np.random.default_rng(2).random((50, 40))
    assert highest_square_mean(values, 1) == pytest.approx(values.max(), abs=1e-12)


def test_a_square_longer_than_the_page_is_shortened_to_it() -> None:
    values = np.random.default_rng(3).random((8, 20))
    assert highest_square_mean(values, 30) == pytest.approx(brute_force(values, 8), abs=1e-12)


def test_a_square_has_a_positive_side() -> None:
    with pytest.raises(ValueError, match="positive side"):
        highest_square_mean(np.zeros((5, 5)), 0)


def test_squares_average_out_an_isolated_high_value() -> None:
    values = np.full((60, 60), 0.1)
    values[30, 30] = 0.9  # one pixel
    values[5:15, 40:50] = 0.5  # a patch as large as the square
    assert highest_square_mean(values, 1) == pytest.approx(0.9)
    assert highest_square_mean(values, 10) == pytest.approx(0.5)


def test_block_peaks_use_blocks_that_do_not_overlap() -> None:
    cell = np.arange(35 * 23, dtype=float).reshape(35, 23)
    assert block_peak(cell, 10) == cell[20:30, 10:20].mean()
    assert block_peak(cell, 1) == cell.max()
    assert block_peak(cell, 50) == cell.mean()  # smaller than a block: one block
    assert block_peak(cell[:5], 10) == cell[:5, 10:20].mean()


def test_word_peaks_are_measured_at_four_scales() -> None:
    values = np.zeros((300, 200))
    values[100:140, 20:60] = 0.8  # a word of 40 px, all suspicious
    values[200, 100] = 1.0  # one pixel in a second word
    words = [box(20, 100, 40, 40), box(90, 190, 40, 40), box(250, 10, 20, 20)]
    peaks = word_peaks(values, words)
    assert list(peaks) == list(WordScale)
    assert peaks[WordScale.PIXEL] == (0.8, 1.0)  # the third box starts outside the page
    assert peaks[WordScale.QUARTER] == pytest.approx((0.8, 1 / 100))
    assert peaks[WordScale.HALF] == pytest.approx((0.8, 1 / 400))
    assert peaks[WordScale.FIXED] == pytest.approx((0.8, 1 / 100))


def test_the_scales_of_the_diagnostic_follow_the_text_except_the_control() -> None:
    assert word_sides(40.0) == {
        WordScale.PIXEL: 1,
        WordScale.QUARTER: 10,
        WordScale.HALF: 20,
        WordScale.FIXED: FIXED_SIDE,
    }
    assert word_sides(90.0)[WordScale.QUARTER] == 22
    assert word_sides(90.0)[WordScale.FIXED] == FIXED_SIDE
