import io

import numpy as np
import numpy.typing as npt
import pytest
from PIL import Image

from docfraud.data.models import BBox
from docfraud.forensics.traces import (
    CONTROLS,
    Jumps,
    compare,
    control_boxes,
    excess_jumps,
    grid_agreement,
    noise_level,
    paper_mask,
    receipt_grid,
    region_rng,
    residual,
)


def box(x: int, y: int, width: int, height: int) -> BBox:
    return BBox(x=x, y=y, width=width, height=height)


def receipt_like(seed: int, height: int = 400, width: int = 320) -> npt.NDArray[np.uint8]:
    """Paper with noise and dark text-like bars at a regular pitch of 24 and 36 px."""
    rng = np.random.default_rng(seed)
    page = rng.normal(225, 6, size=(height, width))
    for top in range(20, height - 20, 24):  # 24 = 3 x 8: print repeats on the JPEG period
        for left in range(16, width - 40, 36):  # 36 = 4 x 9: and on a null period
            page[top : top + 10, left : left + 6 + int(rng.integers(0, 20))] = 40
    return np.clip(page, 0, 255).astype(np.uint8)


def jpeg(page: npt.NDArray[np.uint8], quality: int = 60) -> npt.NDArray[np.uint8]:
    buffer = io.BytesIO()
    Image.fromarray(page).save(buffer, format="JPEG", quality=quality)
    return np.asarray(Image.open(io.BytesIO(buffer.getvalue())))


def test_excess_jumps_peak_at_a_step_between_columns() -> None:
    page = np.zeros((3, 10), dtype=np.uint8)
    page[:, 5:] = 100  # a step between columns 4 and 5
    excess = excess_jumps(page)
    assert excess.shape == (3, 7)
    assert int(np.argmax(excess[0])) + 1 == 4  # result column k is boundary k + 1


def test_a_jpeg_page_has_a_grid_and_noise_has_none() -> None:
    page = jpeg(receipt_like(0))
    grid = receipt_grid(Jumps.of(page), paper_mask(page), [])
    assert grid.detectable  # periodic print does not hide the grid: only paper is used
    assert (grid.columns.phase, grid.rows.phase) == (7, 7)  # boundary after every 8th pixel
    plain = receipt_like(0)
    assert not receipt_grid(Jumps.of(plain), paper_mask(plain), []).detectable


def test_a_patch_pasted_off_the_grid_disagrees_with_it() -> None:
    page = jpeg(receipt_like(1)).copy()
    patch = box(160, 200, 80, 40)
    page[200:240, 160:240] = page[103:143, 61:141]  # moved by 3 px off the 8 x 8 grid
    whole = box(0, 0, page.shape[1], page.shape[0])
    jumps = Jumps.of(page)
    grid = receipt_grid(jumps, paper_mask(page), [patch])
    words = [box(0, 0, page.shape[1], page.shape[0])]
    controls = control_boxes(patch, whole, words, [patch], region_rng("X1", 0))
    agreements = [grid_agreement(jumps, grid, c) for c in controls]
    result = compare(grid_agreement(jumps, grid, patch), agreements)
    assert result.stands_out == "below"


def test_a_flat_patch_stands_out_by_its_noise() -> None:
    page = receipt_like(2)
    patch = box(100, 150, 60, 30)
    page[150:180, 100:160] = 225  # no paper noise at all
    noise, paper = residual(page), paper_mask(page)
    whole = box(0, 0, page.shape[1], page.shape[0])
    controls = control_boxes(patch, whole, [whole], [patch], region_rng("X1", 0))
    values = [noise_level(noise, paper, c) for c in controls]
    measured = [v for v in values if v is not None]
    flat = noise_level(noise, paper, patch)
    assert flat == 0.0
    assert compare(flat, measured).stands_out == "below"


def test_paper_excludes_the_neighbourhood_of_ink() -> None:
    page = np.full((20, 20), 230, dtype=np.uint8)
    page[10, 10] = 20
    paper = paper_mask(page)
    assert not paper[8:13, 8:13].any()
    assert paper[0:7, 0:7].all()
    noise = residual(page)
    assert noise_level(noise, paper, box(8, 8, 5, 5)) is None  # no paper pixel in the box


def test_controls_avoid_annotated_rectangles_and_are_reproducible() -> None:
    whole = box(0, 0, 200, 100)
    region, edited = box(0, 0, 10, 10), box(0, 0, 100, 100)  # the left half is edited
    controls = control_boxes(region, whole, [whole], [edited], region_rng("X1", 3))
    assert len(controls) == CONTROLS
    assert all(c.x >= 100 for c in controls)
    assert controls == control_boxes(region, whole, [whole], [edited], region_rng("X1", 3))
    assert controls != control_boxes(region, whole, [whole], [edited], region_rng("X1", 4))
    nowhere = control_boxes(region, whole, [whole], [whole], region_rng("X1", 3))
    assert nowhere == []


def test_comparisons_place_a_value_among_its_controls() -> None:
    controls = list(range(100))
    assert compare(50.0, controls).percentile == pytest.approx(50.5)
    assert compare(-1.0, controls).stands_out == "below"
    assert compare(99.5, controls).stands_out == "above"
    assert compare(50.0, controls).stands_out is None
    with pytest.raises(ValueError, match="finite"):
        compare(float("nan"), controls)
    with pytest.raises(ValueError, match="finite"):
        compare(1.0, [])
