import numpy as np

from docfraud.evaluation.localisation import hits, top_region
from docfraud.forensics.lattice_map import lattice_evidence
from tests.forensics.test_quantisation import box, jpeg, scattered_text


def test_the_top_region_of_the_map_finds_a_patch_pasted_off_the_grid() -> None:
    page = jpeg(scattered_text(3))[0].copy()
    page[240:280, 160:240] = page[103:143, 61:141]  # 1 row and 3 columns off the grid
    evidence = lattice_evidence(page)  # no annotation: the patch is part of the estimate
    assert evidence.lattice.detectable
    values = evidence.pixel_map()
    assert values.shape == page.shape
    assert np.isfinite(values).all()
    assert values.min() >= 0
    assert values.max() <= 2
    assert hits(top_region(values), [box(160, 240, 80, 40)])


def test_a_page_without_a_lattice_has_no_evidence() -> None:
    evidence = lattice_evidence(scattered_text(2))
    assert not evidence.lattice.detectable
    assert not evidence.pixel_map().any()


def test_pixels_outside_the_grid_and_clipped_blocks_have_no_evidence() -> None:
    page = jpeg(scattered_text(6))[0][3:, 5:].copy()  # the grid starts at row 5, column 3
    page[101:109, 99:107] = 255  # one whole block of the grid clipped to white
    evidence = lattice_evidence(page)
    assert evidence.lattice.offset == (5, 3)
    values = evidence.pixel_map()
    assert not values[:5].any()
    assert not values[:, :3].any()
    assert not values[101:109, 99:107].any()
    assert values[5:, 3:].any()


def test_the_map_depends_on_the_page_only() -> None:
    page = jpeg(scattered_text(7))[0]
    first, second = lattice_evidence(page), lattice_evidence(page.copy())
    np.testing.assert_array_equal(first.pixel_map(), second.pixel_map())


def test_the_document_score_rises_with_a_patch_and_is_zero_without_evidence() -> None:
    clean = jpeg(scattered_text(3))[0]
    patched = clean.copy()
    patched[240:280, 160:240] = clean[103:143, 61:141]
    assert lattice_evidence(patched).document_score() > lattice_evidence(clean).document_score()
    assert lattice_evidence(scattered_text(2)).document_score() == 0.0


def test_only_measured_blocks_enter_the_document_score() -> None:
    page = jpeg(scattered_text(6))[0].copy()
    page[96:160, 96:160] = 255  # 64 blocks clipped to white: no evidence, not a perfect fit
    evidence = lattice_evidence(page)
    assert not evidence.measured[12:20, 12:20].any()
    values = evidence.suspicion[evidence.measured]
    expected = values.max() - np.percentile(values, 99)
    assert evidence.document_score() == expected
