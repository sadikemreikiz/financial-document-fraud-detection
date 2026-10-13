from pathlib import Path

import numpy as np
import pytest

from docfraud.forensics.probe import (
    PROPERTIES,
    PropertyCache,
    as_matrix,
    global_properties,
    probe_scores,
)
from tests.forensics.test_quantisation import jpeg, scattered_text


def test_the_probe_sees_the_28_properties_in_a_fixed_order() -> None:
    found = global_properties(scattered_text(0))
    assert len(PROPERTIES) == len(set(PROPERTIES)) == 28
    assert tuple(found) == PROPERTIES
    assert as_matrix([found, found]).shape == (2, 28)


def test_size_and_brightness_of_a_known_page() -> None:
    page = np.zeros((10, 20), dtype=np.uint8)
    page[:, 10:] = 255
    found = global_properties(page)
    assert (found["height"], found["width"], found["pixels"]) == (10, 20, 200)
    assert found["height over width"] == 0.5
    assert found["mean"] == 127.5
    assert found["standard deviation"] == 127.5
    assert (found["percentile 1"], found["percentile 99"]) == (0, 255)
    assert (found["share at 255"], found["share at 0"]) == (0.5, 0.5)
    assert found["distinct values"] == 2


def test_jpeg_properties_tell_a_compressed_page_from_a_plain_one() -> None:
    compressed = global_properties(jpeg(scattered_text(0))[0])
    assert compressed["lattice detectable"] == 1
    assert compressed["coefficients with a step"] >= 10
    assert np.isfinite(compressed["DC step"])
    assert np.isfinite(compressed["median step"])
    plain = global_properties(scattered_text(2))
    assert plain["lattice detectable"] == 0
    assert plain["coefficients with a step"] == 0
    assert np.isnan(plain["DC step"])
    assert np.isnan(plain["median step"])


def test_properties_that_are_not_defined_are_empty() -> None:
    white = global_properties(np.full((64, 64), 255, dtype=np.uint8))  # every block clipped
    assert np.isnan(white["lattice score"])
    assert np.isnan(white["lattice margin"])
    assert white["measurable block share"] == 0
    assert white["paper noise"] == 0  # paper clipped to white has no noise
    assert all(np.isfinite(v) or np.isnan(v) for v in white.values())


def test_the_forest_scores_receipts_and_accepts_empty_values() -> None:
    rng = np.random.default_rng(0)
    forged = np.arange(200) % 5 == 0
    train = rng.normal(size=(200, 3))
    train[:, 0] += 3 * forged  # one property carries a trace
    train[rng.random(train.shape) < 0.1] = np.nan
    test = np.array([[3.0, 0.0, np.nan], [0.0, 0.0, 0.0]])
    scores = probe_scores(train, forged, test)
    assert scores[0] > scores[1]
    np.testing.assert_array_equal(scores, probe_scores(train, forged, test))  # seed 0
    with pytest.raises(ValueError, match="forged and authentic"):
        probe_scores(train, np.zeros(200, dtype=bool), test)


def test_the_cache_returns_stored_properties_and_keys_them_by_page(tmp_path: Path) -> None:
    cache = PropertyCache(tmp_path)
    page = np.full((64, 64), 255, dtype=np.uint8)
    first = cache.properties(page)
    stored = list(cache.directory.glob("*.json"))
    assert len(stored) == 1
    again = cache.properties(page)
    assert again.keys() == first.keys()
    assert all(
        a == b or (np.isnan(a) and np.isnan(b))
        for a, b in zip(first.values(), again.values(), strict=True)
    )
    other = page.copy()
    other[0, 0] = 0
    cache.properties(other)
    assert len(list(cache.directory.glob("*.json"))) == 2
