from itertools import pairwise
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from docfraud.data.models import BBox
from docfraud.forensics import trufor
from docfraud.forensics.trufor import (
    GRID,
    OVERLAP,
    TILE,
    CachedTruFor,
    analyse,
    axis_windows,
    owners,
    sigmoid,
    stitch,
    tiles,
)


class FakeModel:
    """Returns the window itself as its map, and the window's mean as its logit."""

    identifier = "fake-1"

    def __init__(self) -> None:
        self.windows: list[tuple[int, int]] = []

    def predict(self, window: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.float32], float]:
        self.windows.append(window.shape)
        return (window / 255).astype(np.float32), float(window.mean())


def noise(height: int, width: int, seed: int = 0) -> npt.NDArray[np.uint8]:
    return np.random.default_rng(seed).integers(0, 256, size=(height, width), dtype=np.uint8)


@pytest.fixture
def small_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Tiles of 48 pixels overlapping by 16, so that every pixel can be checked."""
    monkeypatch.setattr(trufor, "TILE", 48)
    monkeypatch.setattr(trufor, "OVERLAP", 16)


@pytest.mark.parametrize("length", [1, 300, TILE])
def test_an_axis_no_longer_than_a_tile_is_one_window(length: int) -> None:
    assert axis_windows(length) == ((0, length),)


def test_an_axis_has_at_least_one_pixel() -> None:
    with pytest.raises(ValueError, match="one pixel"):
        axis_windows(0)


@pytest.mark.parametrize(
    "length",
    [
        *(TILE + 1, TILE + 7, TILE + 8, 2 * TILE - OVERLAP + 3, 2260, 2785, 4961, 6689, 7016),
        *range(TILE + 1, 5 * TILE, 97),
    ],
)
def test_long_axes_are_tiled_on_the_jpeg_grid_with_overlap(length: int) -> None:
    windows = axis_windows(length)
    starts = [start for start, _ in windows]
    assert starts[0] == 0
    assert all(start % GRID == 0 for start in starts)
    assert starts == sorted(set(starts))
    assert windows[-1][0] + windows[-1][1] == length  # the last window reaches the end
    assert all(size == TILE for _, size in windows[:-1])
    assert TILE <= windows[-1][1] < TILE + GRID
    for (start, size), (following, _) in pairwise(windows):
        assert start + size - following >= OVERLAP


def test_the_page_scans_of_the_design_fold_take_20_or_24_tiles() -> None:
    assert len(tiles(7016, 4961)) == 6 * 4
    assert len(tiles(6689, 4846)) == 5 * 4


def test_a_page_of_at_most_one_tile_of_pixels_is_one_window_whatever_its_shape() -> None:
    assert tiles(6000, 300) == (BBox(x=0, y=0, width=300, height=6000),)
    assert tiles(TILE, TILE) == (BBox(x=0, y=0, width=TILE, height=TILE),)
    assert tiles(TILE + 1, TILE) == (BBox(x=0, y=0, width=TILE, height=TILE + 1),)  # extended
    assert len(tiles(TILE + GRID, TILE)) == 2  # a block more: two rows of tiles
    assert {w.width for w in tiles(2260, 963)} == {963}  # a narrow receipt is tiled in height


def test_overlaps_are_cut_where_both_windows_are_equally_far_inside() -> None:
    owner = owners([(0, 10), (6, 10)], 16)  # the overlap is positions 6 to 9
    assert owner.tolist() == [0] * 8 + [1] * 8  # 7: 2 from window 0's edge, 1 from window 1's
    assert owners([(0, 9), (6, 10)], 16).tolist() == [0] * 8 + [1] * 8  # 7 is a tie: earlier
    assert owners([(0, 16)], 16).tolist() == [0] * 16  # the page edges are no window edges


def test_windows_that_leave_a_gap_are_refused() -> None:
    with pytest.raises(ValueError, match="cover"):
        owners([(0, 5), (6, 10)], 16)


@pytest.mark.usefixtures("small_tiles")
@pytest.mark.parametrize(("height", "width"), [(100, 70), (49, 200), (130, 130), (48, 97)])
def test_every_pixel_comes_from_the_tile_it_lies_farthest_inside_of(
    height: int, width: int
) -> None:
    windows = tiles(height, width)
    assert len(windows) > 1
    maps = [
        np.full((w.height, w.width), index, dtype=np.float32) for index, w in enumerate(windows)
    ]
    chosen = stitch((height, width), windows, maps)

    def depth(window: BBox, y: int, x: int) -> int:
        """Distance of a pixel to the nearest edge of the window inside the page."""
        edges = [
            y - window.y if window.y > 0 else height,
            window.y2 - 1 - y if window.y2 < height else height,
            x - window.x if window.x > 0 else width,
            window.x2 - 1 - x if window.x2 < width else width,
        ]
        return min(edges)

    for y in range(height):
        for x in range(width):
            covering = [w for w in windows if w.y <= y < w.y2 and w.x <= x < w.x2]
            best = max(depth(w, y, x) for w in covering)
            assert depth(windows[int(chosen[y, x])], y, x) == best


@pytest.mark.usefixtures("small_tiles")
def test_the_stitched_map_puts_every_value_where_its_pixel_is() -> None:
    page = noise(130, 101)
    output = analyse(page, FakeModel())
    assert len(output.windows) > 1
    np.testing.assert_array_equal(output.localisation, page.astype(np.float32) / 255)


@pytest.mark.usefixtures("small_tiles")
def test_the_integrity_score_is_the_highest_over_tiles() -> None:
    page = np.zeros((130, 101), dtype=np.uint8)
    page[120:, 90:] = 200  # only the last tile sees the bright corner
    output = analyse(page, FakeModel())
    assert output.logits[-1] == max(output.logits) > 0
    assert output.integrity_score() == sigmoid(max(output.logits))


def test_a_small_page_is_analysed_whole() -> None:
    model = FakeModel()
    output = analyse(noise(300, 200), model)
    assert model.windows == [(300, 200)]
    assert output.windows == (BBox(x=0, y=0, width=200, height=300),)
    assert output.seconds >= 0


def test_only_grayscale_pages_are_analysed() -> None:
    with pytest.raises(ValueError, match="grayscale"):
        analyse(np.zeros((10, 10, 3), dtype=np.uint8), FakeModel())
    with pytest.raises(ValueError, match="grayscale"):
        analyse(np.zeros((10, 10)), FakeModel())  # type: ignore[arg-type]


def test_a_map_of_the_wrong_size_is_refused() -> None:
    class Shrinking(FakeModel):
        def predict(self, window: npt.NDArray[np.uint8]) -> tuple[npt.NDArray[np.float32], float]:
            return np.zeros((window.shape[0] // 2, window.shape[1]), dtype=np.float32), 0.0

    with pytest.raises(ValueError, match="map of"):
        analyse(noise(30, 20), Shrinking())
    windows = tiles(2 * TILE, TILE)
    with pytest.raises(ValueError, match="map of"):
        stitch((2 * TILE, TILE), windows, [np.zeros((1, 1), dtype=np.float32)] * len(windows))
    with pytest.raises(ValueError, match="grid"):
        stitch((2 * TILE, TILE), windows, [])


def test_scores_near_one_stay_distinct() -> None:
    assert sigmoid(0) == 0.5
    assert sigmoid(17) < sigmoid(18) < 1.0


@pytest.mark.usefixtures("small_tiles")
def test_the_cache_returns_the_stored_output_without_running_the_model(tmp_path: Path) -> None:
    page = noise(130, 101)
    model = FakeModel()
    first = CachedTruFor(model, tmp_path).analyse(page)
    calls = len(model.windows)
    again = CachedTruFor(model, tmp_path).analyse(page)
    assert len(model.windows) == calls
    np.testing.assert_array_equal(again.localisation, first.localisation)
    assert again.windows == first.windows
    assert again.logits == first.logits
    assert again.seconds == first.seconds
    CachedTruFor(model, tmp_path).analyse(noise(130, 101, seed=1))  # another page: a miss
    assert len(model.windows) == 2 * calls
    assert [p.parent.name for p in tmp_path.rglob("*.npz")] == ["fake-1-tile48-overlap16"] * 2
