from docfraud.data.models import BBox, Label
from docfraud.evaluation.mark_alarms import MarkCheck, check_marks, mark_tables
from tests.data.test_benign_marks import mark

PAGE = BBox(x=0, y=0, width=100, height=200)
WORDS = [BBox(x=10, y=150, width=80, height=20)]  # the text sits low on the page
NOTE = mark(box={"x": 5, "y": 5, "width": 40, "height": 20})  # handwriting above the header
TICK = mark(box={"x": 60, "y": 150, "width": 10, "height": 10}, kind="pen_mark")


def checked(doc_id: str, label: Label, region: BBox, score: float = 0.5) -> MarkCheck:
    return check_marks(doc_id, label, region, PAGE, WORDS, [NOTE, TICK], evidence=True, score=score)


def test_a_region_that_shares_a_pixel_with_a_mark_is_on_it() -> None:
    on_note = checked("XA", Label.AUTHENTIC, BBox(x=44, y=24, width=5, height=5))
    assert on_note.on_mark
    assert on_note.kinds == ("handwriting",)
    both = checked("XA", Label.AUTHENTIC, BBox(x=0, y=0, width=100, height=200))
    assert both.kinds == ("handwriting", "pen_mark")
    beside = checked("XA", Label.AUTHENTIC, BBox(x=45, y=25, width=5, height=5))
    assert not beside.on_mark


def test_the_baseline_places_boxes_on_the_text_and_is_seeded_by_the_receipt() -> None:
    region = BBox(x=0, y=0, width=4, height=4)
    first = checked("XA", Label.AUTHENTIC, region)
    assert 0 < first.baseline < 1  # a box on the text overlaps the tick, never the note
    assert checked("XA", Label.AUTHENTIC, region).baseline == first.baseline
    without = check_marks("XA", Label.AUTHENTIC, region, PAGE, WORDS, [], evidence=True, score=0.5)
    assert without.baseline == 0
    assert not without.on_mark


def test_the_tables_count_receipts_with_evidence_and_rank_false_alarms() -> None:
    region = BBox(x=44, y=24, width=5, height=5)
    checks = [
        checked("XA", Label.AUTHENTIC, region, score=0.9),
        checked("XB", Label.AUTHENTIC, BBox(x=45, y=25, width=5, height=5), score=0.2),
        check_marks(
            "XC", Label.AUTHENTIC, PAGE, PAGE, WORDS, [NOTE], evidence=False, score=0.0
        ),  # without evidence: left out although the page overlaps the note
        checked("XD", Label.FORGED, region, score=0.7),
    ]
    summary, listing = mark_tables(checks, [0.9, 0.2, 0.0, 0.95])
    assert summary.rows[0][1:4] == [3, 2, 1]
    assert summary.rows[1][1:4] == [1, 1, 1]
    assert listing.rows == [["XA", "handwriting", 0.9, 2]]
    assert listing.columns[-1] == "Rank among 4 authentic"
