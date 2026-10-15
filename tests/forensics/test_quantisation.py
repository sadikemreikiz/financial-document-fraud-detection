import io

import numpy as np
import numpy.typing as npt
from PIL import Image
from PIL.JpegImagePlugin import JpegImageFile

from docfraud.data.models import BBox
from docfraud.forensics.quantisation import (
    MIN_VALUES,
    best_step,
    block_fits,
    box_fit,
    coefficient_fit,
    measurable,
    receipt_lattice,
)
from docfraud.forensics.traces import compare, control_boxes, region_rng


def box(x: int, y: int, width: int, height: int) -> BBox:
    return BBox(x=x, y=y, width=width, height=height)


def scattered_text(seed: int, height: int = 480, width: int = 400) -> npt.NDArray[np.uint8]:
    """Noisy paper with dark strokes at random places, sizes and tones; nothing repeats.

    Print that repeats exactly on the block period gives discrete DCT values that fit a
    lattice by themselves; real print does not repeat that exactly.
    """
    rng = np.random.default_rng(seed)
    page = rng.normal(225, 6, size=(height, width))
    for _ in range(height * width // 80):
        top, left = int(rng.integers(0, height - 10)), int(rng.integers(0, width - 8))
        tall, wide = int(rng.integers(3, 10)), int(rng.integers(1, 8))
        page[top : top + tall, left : left + wide] = rng.integers(20, 120)
    return np.clip(page, 1, 254).astype(np.uint8)  # no clipped pixel before compression


def jpeg(
    page: npt.NDArray[np.uint8], quality: int = 75
) -> tuple[npt.NDArray[np.uint8], npt.NDArray[np.int_]]:
    """The page after JPEG compression, and the file's luminance quantisation table."""
    buffer = io.BytesIO()
    Image.fromarray(page).save(buffer, format="JPEG", quality=quality)
    image = Image.open(io.BytesIO(buffer.getvalue()))
    assert isinstance(image, JpegImageFile)
    table = np.array(image.quantization[0]).reshape(8, 8)  # natural order in Pillow
    return np.asarray(image), table


def test_a_coefficient_on_the_lattice_fits_and_spread_values_do_not() -> None:
    rng = np.random.default_rng(0)
    on = 12.0 * rng.integers(-20, 21, size=2_000) + rng.normal(0, 0.3, size=2_000)
    assert coefficient_fit(on, 12) > 0.9
    assert best_step(on)[0] == 12  # not 6, 4, 3 or 2: those divide 12 but fit noise worse
    spread = rng.uniform(-200, 200, size=2_000)
    assert abs(coefficient_fit(spread, 12)) < 0.1
    near_zero = rng.uniform(-5, 5, size=2_000)  # fits every large step; left out
    assert np.isnan(coefficient_fit(near_zero, 12))
    assert np.isnan(coefficient_fit(on[: MIN_VALUES - 1], 12))


def test_the_lattice_of_a_jpeg_page_is_found_with_its_steps() -> None:
    page, table = jpeg(scattered_text(0))
    lattice = receipt_lattice(page, [], np.random.default_rng(0))
    assert lattice.detectable
    assert lattice.offset == (0, 0)
    used = lattice.steps > 0
    assert used.sum() >= 10
    assert (lattice.steps[used] == table[used]).all()


def test_the_alignment_follows_a_cropped_grid() -> None:
    page, _ = jpeg(scattered_text(1))
    lattice = receipt_lattice(page[3:, 5:], [], np.random.default_rng(0))
    assert lattice.detectable
    assert lattice.offset == (5, 3)  # the next block starts 5 rows and 3 columns in


def test_a_page_never_compressed_has_no_lattice() -> None:
    lattice = receipt_lattice(scattered_text(2), [], np.random.default_rng(0))
    assert not lattice.detectable


def test_a_patch_pasted_off_the_grid_does_not_fit() -> None:
    page = jpeg(scattered_text(3))[0].copy()
    patch = box(160, 240, 80, 40)
    page[240:280, 160:240] = page[103:143, 61:141]  # 1 row and 3 columns off the grid
    lattice = receipt_lattice(page, [patch], np.random.default_rng(0))
    assert lattice.detectable
    fits = block_fits(page, lattice)
    whole = box(0, 0, page.shape[1], page.shape[0])
    controls = control_boxes(patch, whole, [whole], [patch], region_rng("X1", 0))
    values = [box_fit(fits, lattice, c) for c in controls]
    measured = [v for v in values if v is not None]
    patched = box_fit(fits, lattice, patch)
    assert patched is not None
    assert compare(patched, measured).stands_out == "below"


def test_clipped_blocks_and_small_boxes_are_not_measured() -> None:
    blocks = np.full((3, 8, 8), 128, dtype=np.uint8)
    blocks[1, 0, 0], blocks[2, 7, 7] = 255, 0
    assert measurable(blocks).tolist() == [True, False, False]
    page, _ = jpeg(scattered_text(4))
    lattice = receipt_lattice(page, [], np.random.default_rng(0))
    fits = block_fits(page, lattice)
    assert box_fit(fits, lattice, box(8, 8, 16, 8)) is None  # two blocks, fewer than four
    assert box_fit(fits, lattice, box(8, 8, 32, 32)) is not None


def test_annotated_rectangles_are_left_out_of_the_estimate() -> None:
    page, _ = jpeg(scattered_text(5))
    whole = box(0, 0, page.shape[1], page.shape[0])
    lattice = receipt_lattice(page, [whole], np.random.default_rng(0))
    assert not lattice.detectable
    assert np.isnan(lattice.score)
    assert (lattice.steps == 0).all()
