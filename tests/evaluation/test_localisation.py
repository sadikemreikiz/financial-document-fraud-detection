from pathlib import Path

import numpy as np
import pytest
from sklearn.metrics import average_precision_score

from docfraud.data.models import BBox, Document, Label, Region, RegionRole, Split
from docfraud.evaluation.localisation import (
    BASELINE_DRAWS,
    THRESHOLD_SAMPLE,
    annotated_rectangles,
    any_box_baseline,
    baseline_hit_rate,
    choose_pixel_threshold,
    edit_average_precisions,
    f1_at_thresholds,
    has_large_rectangle,
    hits,
    localisation_guard,
    pixel_average_precision,
    pixel_f1_iou,
    positive_share,
    receipt_rng,
    size_dependence,
    spearman,
    threshold_candidates,
    threshold_sample,
    top_region,
)


def box(x: int, y: int, width: int, height: int) -> BBox:
    return BBox(x=x, y=y, width=width, height=height)


PAGE = box(0, 0, 100, 60)


def test_the_region_grows_from_the_peak_over_eight_connected_pixels() -> None:
    score_map = np.zeros((60, 100))
    score_map[10:13, 20:25] = 0.8  # the peak's blob, joined diagonally to the next pixel
    score_map[13, 25] = 0.6
    score_map[11, 22] = 1.0
    score_map[40:45, 70:90] = 0.9  # a separate, lower blob
    score_map[20, 20] = 0.4  # below the half height: not part of the region
    assert top_region(score_map) == box(20, 10, 6, 4)


def test_ties_take_the_first_peak_and_a_flat_map_the_whole_page() -> None:
    score_map = np.zeros((60, 100), dtype=np.float32)
    score_map[50, 5] = score_map[5, 50] = 2.0  # equal peaks: the first in row-major order
    assert top_region(score_map) == box(50, 5, 1, 1)
    assert top_region(np.full((60, 100), 3)) == PAGE


@pytest.mark.parametrize("bad", [np.zeros((0, 4)), np.zeros(5), np.array([[0.0, np.nan]])])
def test_invalid_maps_are_refused(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="map"):
        top_region(bad)


def document(regions: list[Region]) -> Document:
    return Document(
        dataset="toy",
        doc_id="X1",
        source_id="X1",
        split=Split.TRAIN,
        label=Label.FORGED,
        image_path=Path("x.png"),
        regions=tuple(regions),
        has_region_annotations=True,
    )


def test_rectangles_hits_and_large_rectangles() -> None:
    modified = Region(box=box(10, 10, 5, 5), role=RegionRole.MODIFIED)
    source = Region(box=box(50, 10, 5, 5), role=RegionRole.SOURCE)
    receipt = document([modified, source])
    assert annotated_rectangles(receipt) == ([modified.box], [source.box])
    assert hits(box(14, 14, 3, 3), [modified.box])
    assert not hits(box(15, 15, 3, 3), [modified.box])
    assert not has_large_rectangle(receipt, PAGE)  # 25 px of 6000
    assert has_large_rectangle(receipt, box(0, 0, 20, 20))  # 25 px of 400


def test_the_baseline_draws_boxes_on_text_inside_the_page() -> None:
    words = [box(0, 0, 10, 4)]  # text in the top left corner only
    near = [box(0, 0, 2, 2)]
    rate = baseline_hit_rate(box(0, 0, 30, 20), PAGE, words, near, receipt_rng("X1"))
    assert rate == 1.0  # every box is shifted into the page, so it covers the corner
    far = [box(90, 50, 5, 5)]
    assert baseline_hit_rate(box(0, 0, 3, 3), PAGE, words, far, receipt_rng("X1")) == 0.0
    half = baseline_hit_rate(box(0, 0, 1, 1), PAGE, words, [box(0, 0, 5, 4)], receipt_rng("X1"))
    assert 0.4 < half < 0.6  # half of the text pixels lie in the rectangle
    assert half * BASELINE_DRAWS == round(half * BASELINE_DRAWS)


def test_the_baseline_is_reproducible_per_receipt() -> None:
    args = (box(0, 0, 2, 2), PAGE, [box(0, 0, 100, 60)], [box(0, 0, 50, 60)])
    assert baseline_hit_rate(*args, receipt_rng("X1")) == baseline_hit_rate(
        *args, receipt_rng("X1")
    )
    assert baseline_hit_rate(*args, receipt_rng("X1")) != baseline_hit_rate(
        *args, receipt_rng("X2")
    )


def test_the_baseline_of_several_boxes_places_each_independently() -> None:
    words = [box(0, 0, 100, 60)]
    left, right = [box(0, 0, 50, 60)], [box(50, 0, 50, 60)]
    one = baseline_hit_rate(box(0, 0, 2, 2), PAGE, words, left, receipt_rng("X1"))
    assert any_box_baseline([box(0, 0, 2, 2)], PAGE, words, left, receipt_rng("X1")) == one
    two = any_box_baseline([box(0, 0, 2, 2)] * 2, PAGE, words, left, receipt_rng("X1"))
    assert 0.7 < two < 0.8  # 1 - 0.5 x 0.5 for independent boxes; equal boxes would give 0.5
    both_halves = any_box_baseline(
        [box(0, 0, 2, 2)] * 2, PAGE, words, left + right, receipt_rng("X1")
    )
    assert both_halves == 1.0
    with pytest.raises(ValueError, match="without a region"):
        any_box_baseline([], PAGE, words, left, receipt_rng("X1"))


def test_the_baseline_needs_text_and_a_region_that_fits() -> None:
    with pytest.raises(ValueError, match="no random baseline"):
        baseline_hit_rate(box(0, 0, 2, 2), PAGE, [], [], receipt_rng("X1"))
    with pytest.raises(ValueError, match="does not fit"):
        baseline_hit_rate(box(0, 0, 200, 2), PAGE, [PAGE], [], receipt_rng("X1"))


def test_the_guard_compares_the_interval_with_the_baseline() -> None:
    guard = localisation_guard([True] * 18 + [False] * 2, [0.3] * 20, resamples=2000)
    assert (guard.receipts, guard.hit_rate.value) == (20, 0.9)
    assert guard.baseline == pytest.approx(0.3)
    assert guard.detects
    assert not localisation_guard([True, False] * 10, [0.45] * 20, resamples=2000).detects
    with pytest.raises(ValueError, match="one hit and one baseline"):
        localisation_guard([True], [0.1, 0.2])


def test_pixel_average_precision_ignores_copy_sources_and_matches_scikit_learn() -> None:
    rng = np.random.default_rng(3)
    score_map = rng.random((60, 100))
    modified, source = box(10, 10, 20, 10), box(60, 30, 10, 10)
    truth = np.zeros((60, 100), dtype=bool)
    truth[10:20, 10:30] = True
    counted = np.ones((60, 100), dtype=bool)
    counted[30:40, 60:70] = False
    expected = average_precision_score(truth[counted], score_map[counted])
    assert pixel_average_precision(score_map, [modified], [source]) == pytest.approx(expected)
    score_map[30:40, 60:70] = 2.0  # a high score on the copy source changes nothing
    assert pixel_average_precision(score_map, [modified], [source]) == pytest.approx(expected)
    with pytest.raises(ValueError, match="without modified pixels"):
        pixel_average_precision(score_map, [], [source])


def test_the_chance_level_is_the_share_of_modified_pixels_among_those_counted() -> None:
    modified, source = box(10, 10, 20, 10), box(60, 30, 10, 10)
    assert positive_share((60, 100), [modified], []) == 200 / 6000
    assert positive_share((60, 100), [modified], [source]) == 200 / 5900
    with pytest.raises(ValueError, match="without modified pixels"):
        positive_share((60, 100), [], [source])


def random_box(rng: np.random.Generator, height: int, width: int) -> BBox:
    y, x = int(rng.integers(0, height - 1)), int(rng.integers(0, width - 1))
    return box(x, y, int(rng.integers(1, width - x + 1)), int(rng.integers(1, height - y + 1)))


def test_each_edit_is_measured_as_the_only_modified_rectangle_with_the_others_ignored() -> None:
    rng = np.random.default_rng(5)
    for _ in range(200):
        height, width = (int(n) for n in rng.integers(6, 30, size=2))
        score_map = rng.integers(0, 5, size=(height, width)).astype(float)  # many ties
        edits = [random_box(rng, height, width) for _ in range(int(rng.integers(1, 4)))]
        sources = [random_box(rng, height, width) for _ in range(int(rng.integers(0, 2)))]
        measured = edit_average_precisions(score_map, edits, sources)
        for edit, result in zip(edits, measured, strict=True):
            others = [o for o in edits if o is not edit] + sources
            assert result.average_precision == pytest.approx(
                pixel_average_precision(score_map, [edit], others), abs=1e-12
            )
            assert result.chance == pytest.approx(
                positive_share((height, width), [edit], others), abs=1e-12
            )


def test_an_edit_scored_above_every_clean_pixel_has_average_precision_one() -> None:
    score_map = np.zeros((60, 100))
    score_map[10:20, 10:30] = 1.0
    score_map[40:50, 10:30] = 2.0  # another edit, ignored when the first is measured
    first, second = box(10, 10, 20, 10), box(10, 40, 20, 10)
    measured = edit_average_precisions(score_map, [first, second], [])
    assert [m.average_precision for m in measured] == [1.0, 1.0]
    assert measured[0].chance == 200 / 5800
    with pytest.raises(ValueError, match="no edit"):
        edit_average_precisions(score_map, [], [])
    with pytest.raises(ValueError, match="holds no pixel"):
        edit_average_precisions(score_map, [box(200, 200, 5, 5)], [])


def test_pixel_f1_and_iou_at_one_or_many_thresholds() -> None:
    score_map = np.zeros((60, 100))
    score_map[10:20, 10:30] = 1.0  # exactly the modified rectangle
    score_map[0:10, 10:30] = 0.5  # as large again next to it
    modified = [box(10, 10, 20, 10)]
    assert pixel_f1_iou(score_map, 1.0, modified, []) == (1.0, 1.0)
    f1, iou = pixel_f1_iou(score_map, 0.5, modified, [])
    assert f1 == pytest.approx(2 / 3)
    assert iou == 0.5
    sources = [box(10, 0, 20, 10)]  # the extra pixels are a copy source: ignored
    assert pixel_f1_iou(score_map, 0.5, modified, sources) == (1.0, 1.0)
    curve = f1_at_thresholds(score_map, [0.5, 1.0, 2.0], modified, [])
    np.testing.assert_allclose(curve, [2 / 3, 1.0, 0.0])


def test_the_pixel_threshold_comes_from_equal_samples_and_mean_f1() -> None:
    large = np.arange(THRESHOLD_SAMPLE + 10, dtype=float).reshape(1, -1)
    sample = threshold_sample(large, "X1")
    assert sample.size == THRESHOLD_SAMPLE
    np.testing.assert_array_equal(sample, threshold_sample(large, "X1"))
    small = np.array([[1.0, 2.0], [3.0, 4.0]])
    np.testing.assert_array_equal(threshold_sample(small, "X2"), [1.0, 2.0, 3.0, 4.0])
    candidates = threshold_candidates([small.ravel(), small.ravel()])
    assert (candidates[0], candidates[-1]) == (1.0, 4.0)  # pooled: 1, 1, 2, 2, 3, 3, 4, 4
    assert np.all(np.diff(candidates) > 0)
    assert choose_pixel_threshold([0.1, 0.2, 0.3], [[0.5, 0.8, 0.8], [0.5, 0.6, 0.6]]) == 0.3


def test_size_dependence_finds_scores_that_grow_with_size() -> None:
    sizes = list(range(1, 41))
    growing = size_dependence([s * 2.0 for s in sizes], sizes, resamples=2000)
    assert growing.value == pytest.approx(1.0)
    assert growing.excludes_zero
    shuffled = np.random.default_rng(0).permutation(40).astype(float)
    independent = size_dependence(list(shuffled), sizes, resamples=2000)
    assert not independent.excludes_zero
    assert np.isnan(spearman([1.0, 1.0, 1.0], [1.0, 2.0, 3.0]))
    with pytest.raises(ValueError, match="at least three receipts"):
        size_dependence([1.0, 2.0], [1, 2])
