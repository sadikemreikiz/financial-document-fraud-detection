import numpy as np
import numpy.typing as npt
import pytest

from docfraud.data.models import BBox
from docfraud.forensics.final import map_results, measured, pixel_threshold
from docfraud.forensics.learned.inference import LearnedOutput
from docfraud.forensics.learned.results import Group, Item

EDIT = BBox(x=40, y=20, width=10, height=10)
SOURCE = BBox(x=10, y=40, width=10, height=10)
WORDS = (BBox(x=5, y=5, width=60, height=10), BBox(x=5, y=40, width=60, height=10))


class Spot:
    """A map that is high on the edit of every forged page it is given, low elsewhere."""

    identifier = "spot"

    def analyse(self, page: npt.NDArray[np.uint8]) -> LearnedOutput:
        rng = np.random.default_rng(int(page.sum()) % 1000)
        logits = rng.normal(-4, 1, size=page.shape).astype(np.float32)
        if page.max() > 0:  # the forged pages carry a mark
            logits[EDIT.y : EDIT.y2, EDIT.x : EDIT.x2] += 8
        return LearnedOutput(logits, (BBox(x=0, y=0, width=80, height=60),), 0.1)


def item(key: str, *, forged: bool, sources: tuple[BBox, ...] = ()) -> Item:
    page = np.zeros((60, 80), dtype=np.uint8)
    if forged:
        page[0, int(key[1:])] = 1
    return Item(
        key,
        Group.HELD_OUT,
        page,
        forged,
        modified=(EDIT,) if forged else (),
        sources=sources,
        words=WORDS,
    )


def test_the_pixel_threshold_is_chosen_on_the_forged_receipts_with_an_edit_only() -> None:
    items = [item("R1", forged=True), item("R2", forged=True), item("R3", forged=False)]
    found = pixel_threshold(Spot(), items)
    assert found.receipts == 2
    assert 0 < found.value < 1
    with pytest.raises(ValueError, match="no forged validation receipt"):
        pixel_threshold(Spot(), [item("R3", forged=False)])


def test_each_receipt_is_measured_at_the_frozen_threshold_and_summed_up() -> None:
    items = [
        item("R1", forged=True, sources=(SOURCE,)),
        item("R2", forged=True),
        item("R3", forged=False),
    ]
    receipts = measured(Spot(), items, 0.5)
    assert [r.key for r in receipts] == ["R1", "R2", "R3"]
    assert receipts[2].pixels is None  # an authentic receipt has no pixel metrics
    words = {r.key: list(WORDS) for r in receipts}
    found = map_results(receipts, words, 0.5, resamples=20)
    assert (found["threshold"], found["forged_with_regions"]) == (0.5, 2)
    assert found["pixel_average_precision"]["value"] > found["chance"]
    assert found["guard"]["receipts"] == 2
    assert found["guard"]["hit_rate"]["value"] == 1.0  # the top region lies on the edit
    assert found["sources"]["receipts"] == 1
    assert found["sources"]["hit_rate"]["value"] == 0.0  # and not on the copy's source


def test_a_split_without_forged_receipts_has_nothing_to_measure() -> None:
    receipts = measured(Spot(), [item("R3", forged=False)], 0.5)
    found = map_results(receipts, {"R3": list(WORDS)}, 0.5, resamples=20)
    assert found["forged_with_regions"] == 0
    assert found["pixel_average_precision"] is None
    assert (found["chance"], found["guard"], found["sources"]) == (None, None, None)
