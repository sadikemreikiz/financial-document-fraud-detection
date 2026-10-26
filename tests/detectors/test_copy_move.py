import numpy as np
import numpy.typing as npt
import pytest

from docfraud.data.models import BBox
from docfraud.detectors.copy_move import (
    MAX_GROUP,
    MIN_BLOCK,
    CopyMoveEvidence,
    block_hashes,
    block_side,
    block_sums,
    copy_move_evidence,
    duplicate_pair,
    informative_blocks,
)

type Page = npt.NDArray[np.uint8]


def noisy_page(height: int = 160, width: int = 240, seed: int = 0) -> Page:
    """Paper with scan noise, less than the contrast of ink: no two blocks are equal."""
    return np.random.default_rng(seed).integers(200, 250, size=(height, width), dtype=np.uint8)


def ink_patch(height: int, width: int, seed: int = 1) -> Page:
    """A patch with dark ink and light paper, as a printed character."""
    rng = np.random.default_rng(seed)
    patch = rng.integers(180, 250, size=(height, width), dtype=np.uint8)
    patch[height // 4 : 3 * height // 4, width // 3 : 2 * width // 3] = rng.integers(
        10, 80, size=(3 * height // 4 - height // 4, 2 * width // 3 - width // 3)
    )
    return patch


def test_block_side_is_a_quarter_of_the_text_height_and_at_least_the_minimum() -> None:
    assert block_side(43.0) == 11
    assert block_side(8.0) == MIN_BLOCK


def test_block_sums_add_every_block() -> None:
    values = np.arange(20, dtype=np.int64).reshape(4, 5)
    sums = block_sums(values, 2)
    assert sums.shape == (3, 4)
    assert sums[1, 2] == values[1:3, 2:4].sum()


def test_equal_blocks_have_equal_hashes_and_others_differ() -> None:
    page = noisy_page(40, 50)
    page[30:36, 40:46] = page[2:8, 3:9]
    hashes = block_hashes(page, 6)
    assert hashes.shape == (35, 45)
    assert hashes[30, 40] == hashes[2, 3]
    assert len(np.unique(hashes)) == hashes.size - 1


def test_an_exact_copy_is_found_as_a_pair_with_its_area() -> None:
    page = noisy_page()
    page[20:50, 30:50] = ink_patch(30, 20)
    page[100:130, 180:200] = page[20:50, 30:50]
    pair = duplicate_pair(page, 6)
    assert pair is not None
    assert pair.first == BBox(x=30, y=20, width=20, height=30)
    assert pair.second == BBox(x=180, y=100, width=20, height=30)
    assert pair.area == 600


def test_the_upper_region_comes_first_whichever_was_pasted() -> None:
    page = noisy_page()
    page[100:130, 30:50] = ink_patch(30, 20)
    page[20:50, 180:200] = page[100:130, 30:50]
    pair = duplicate_pair(page, 6)
    assert pair is not None
    assert pair.first.y == 20
    assert pair.second.y == 100


def test_a_page_without_a_copy_has_no_pair() -> None:
    page = noisy_page()
    page[20:50, 30:50] = ink_patch(30, 20, seed=1)
    page[100:130, 180:200] = ink_patch(30, 20, seed=2)  # the same shape printed twice
    assert duplicate_pair(page, 6) is None


def test_specks_of_dust_on_white_paper_do_not_match() -> None:
    page = np.full((120, 160), 255, dtype=np.uint8)
    page[10, 10] = page[80, 120] = 0
    page[40, 40] = page[90, 20] = 100
    assert not informative_blocks(page, 6, contrast=0).any()
    assert duplicate_pair(page, 6) is None


def test_flat_paper_is_not_informative() -> None:
    page = np.full((60, 80), 128, dtype=np.uint8)
    assert not informative_blocks(page, 6, contrast=0).any()


def test_a_copy_of_blank_paper_is_not_found() -> None:
    page = noisy_page()
    page[100:130, 180:200] = page[20:50, 30:50]  # paper texture only, no ink
    assert duplicate_pair(page, 6) is None
    assert duplicate_pair(page, 6, contrast=0) is not None


def test_a_hash_shared_by_many_positions_is_skipped() -> None:
    page = noisy_page(200, 320)
    patch = ink_patch(8, 8)
    for k in range(MAX_GROUP + 1):
        row, column = divmod(k, 6)
        page[10 + 40 * row : 18 + 40 * row, 10 + 50 * column : 18 + 50 * column] = patch
    assert duplicate_pair(page, 8) is None


def test_a_pattern_repeating_within_one_block_is_not_a_copy() -> None:
    page = noisy_page()
    stripes = ink_patch(12, 2)
    page[60:72, 100:109] = np.tile(stripes, (1, 5))[:, :9]  # repeats every 2 px over 9 px
    assert duplicate_pair(page, 6) is None


def test_a_page_smaller_than_a_block_is_refused() -> None:
    with pytest.raises(ValueError, match="holds no block"):
        duplicate_pair(noisy_page(5, 40), 6)


def test_evidence_scores_the_area_in_squared_text_heights_and_leaves_the_page_unchanged() -> None:
    page = noisy_page()
    page[20:50, 30:50] = ink_patch(30, 20)
    page[100:130, 180:200] = page[20:50, 30:50]
    before = page.copy()
    words = [BBox(x=10, y=10 + 30 * k, width=60, height=24) for k in range(4)]
    evidence = copy_move_evidence(page, words)
    assert evidence.block == 6
    assert evidence.pair is not None
    assert evidence.document_score == pytest.approx(600 / 24**2)
    assert np.array_equal(page, before)
    assert copy_move_evidence(page, words) == evidence


def test_no_pair_scores_zero() -> None:
    assert CopyMoveEvidence(text_height=24.0, block=6, pair=None).document_score == 0.0
