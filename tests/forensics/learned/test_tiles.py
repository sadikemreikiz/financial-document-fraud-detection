import numpy as np
import pytest

from docfraud.data.models import BBox
from docfraud.forensics.learned.tiles import (
    CLEAN,
    FORGED,
    IGNORED,
    MARGIN,
    TILE,
    axis_positions,
    cut,
    flip,
    labels,
    position_around,
    uniform_position,
)
from docfraud.forensics.trufor import GRID


def test_a_small_span_is_kept_inside_with_the_margin() -> None:
    starts = axis_positions(500, 540, 2000)
    assert starts.size
    assert (starts % GRID == 0).all()
    assert (starts <= 500 - MARGIN).all()
    assert (starts + TILE >= 540 + MARGIN).all()
    every = np.arange(0, 2000 - TILE + 1, GRID)
    expected = every[(every <= 500 - MARGIN) & (every + TILE >= 540 + MARGIN)]
    assert np.array_equal(starts, expected)


def test_near_the_page_edge_the_edge_limits_the_margin() -> None:
    assert axis_positions(10, 50, 2000).tolist() == [0]
    last = (2000 - TILE) // GRID * GRID
    assert axis_positions(1990, 1995, 2000).tolist() == [last]


def test_the_margin_shrinks_before_the_span_is_cut() -> None:
    span = TILE - MARGIN  # too wide for a margin on both sides
    starts = axis_positions(1000, 1000 + span, 3000)
    assert starts.size
    assert (starts <= 1000).all()
    assert (starts + TILE >= 1000 + span).all()


def test_a_span_wider_than_a_tile_is_overlapped() -> None:
    starts = axis_positions(1000, 1600, 3000)
    assert starts.size
    every = np.arange(0, 3000 - TILE + 1, GRID)
    assert np.array_equal(starts, every[(every < 1600) & (every + TILE > 1000)])


def test_a_span_beyond_the_last_aligned_tile_has_no_position() -> None:
    length = TILE + 15  # the last aligned tile ends 7 px before the page does
    assert axis_positions(length - 5, length, length).size == 0


def test_a_span_partly_beyond_the_last_aligned_tile_is_overlapped() -> None:
    length = TILE + 15
    starts = axis_positions(length - 10, length, length)
    assert starts.tolist() == [8]


@pytest.mark.parametrize(("start", "end", "length"), [(5, 5, 1000), (-1, 5, 1000), (5, 1001, 1000)])
def test_a_span_must_lie_in_the_axis(start: int, end: int, length: int) -> None:
    with pytest.raises(ValueError, match="span"):
        axis_positions(start, end, length)


def test_a_page_shorter_than_a_tile_is_refused() -> None:
    with pytest.raises(ValueError, match="shorter than a tile"):
        axis_positions(0, 10, TILE - 1)


def test_positions_around_a_box_cover_every_allowed_corner() -> None:
    rng = np.random.default_rng(0)
    box = BBox(x=700, y=300, width=60, height=20)
    corners = [position_around(box, (1500, 1200), rng) for _ in range(4000)]
    assert {c[0] for c in corners if c} == set(axis_positions(300, 320, 1500).tolist())
    assert {c[1] for c in corners if c} == set(axis_positions(700, 760, 1200).tolist())


def test_a_box_no_tile_reaches_has_no_position() -> None:
    box = BBox(x=TILE + 10, y=0, width=4, height=4)
    assert position_around(box, (1000, TILE + 15), np.random.default_rng(0)) is None


def test_uniform_positions_are_aligned_and_inside() -> None:
    rng = np.random.default_rng(1)
    corners = [uniform_position((1001, 777), rng) for _ in range(500)]
    assert all(y % GRID == 0 and x % GRID == 0 for y, x in corners)
    assert all(0 <= y <= 1001 - TILE and 0 <= x <= 777 - TILE for y, x in corners)
    assert len(set(corners)) > 100


def test_labels_mark_forged_and_ignored_pixels() -> None:
    modified = [BBox(x=110, y=210, width=20, height=10)]
    sources = [BBox(x=100, y=200, width=40, height=40), BBox(x=1000, y=1000, width=5, height=5)]
    tile = labels((200, 100), modified, sources)
    assert tile.shape == (TILE, TILE)
    assert (tile[10:20, 10:30] == FORGED).all()
    assert (tile[0:40, 0:40] != CLEAN).all()
    assert (tile[0:10, 0:40] == IGNORED).all()
    assert (tile == FORGED).sum() == 200
    assert (tile == IGNORED).sum() == 1600 - 200
    assert (tile == CLEAN).sum() == TILE * TILE - 1600


def test_labels_clip_boxes_at_the_tile_edge() -> None:
    tile = labels((0, 0), [BBox(x=TILE - 4, y=TILE - 2, width=50, height=50)], [])
    assert (tile == FORGED).sum() == 8


def test_cut_copies_the_tile() -> None:
    page = np.arange(500 * 600, dtype=np.int64).reshape(500, 600).astype(np.uint8)
    tile = cut(page, (16, 40))
    assert np.array_equal(tile, page[16 : 16 + TILE, 40 : 40 + TILE])
    tile[0, 0] ^= 1
    assert tile[0, 0] != page[16, 40]


@pytest.mark.parametrize("corner", [(-8, 0), (0, 600 - TILE + 8), (500, 0)])
def test_cut_refuses_a_tile_outside_the_page(corner: tuple[int, int]) -> None:
    with pytest.raises(ValueError, match="inside a page"):
        cut(np.zeros((500, 600), dtype=np.uint8), corner)


@pytest.mark.parametrize(
    ("vertical", "horizontal"), [(False, False), (True, False), (False, True), (True, True)]
)
def test_flips_mirror_pixels_and_keep_their_values(vertical: bool, horizontal: bool) -> None:
    tile = np.random.default_rng(2).integers(0, 256, size=(TILE, TILE), dtype=np.uint8)
    mirrored = flip(tile, vertical=vertical, horizontal=horizontal)
    expected = tile[:: -1 if vertical else 1, :: -1 if horizontal else 1]
    assert np.array_equal(mirrored, expected)
    assert mirrored.flags.c_contiguous
    assert np.array_equal(np.sort(mirrored, axis=None), np.sort(tile, axis=None))


def test_a_flip_keeps_the_jpeg_grid_aligned() -> None:
    tile = np.zeros((TILE, TILE), dtype=np.uint8)
    tile[::GRID, ::GRID] = 255  # the first pixel of every 8 x 8 block
    rows, columns = np.nonzero(flip(tile, vertical=True, horizontal=True))
    assert rows.size == (TILE // GRID) ** 2
    assert (rows % GRID == GRID - 1).all()  # now the last pixel of every block
    assert (columns % GRID == GRID - 1).all()
