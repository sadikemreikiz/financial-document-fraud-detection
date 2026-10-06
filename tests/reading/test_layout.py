from docfraud.data.models import BBox
from docfraud.reading.layout import Row, Segment, group_rows, union


def seg(text: str, x: int, y: int, w: int = 20, h: int = 10) -> Segment:
    return Segment(text, BBox(x=x, y=y, width=w, height=h))


def texts(rows: list[Row]) -> list[str]:
    return [row.text for row in rows]


def test_label_and_value_on_the_same_row_are_joined_left_to_right() -> None:
    rows = group_rows([seg("6.50", 200, 101), seg("TOTAL", 10, 100), seg("CASH", 10, 130)])
    assert texts(rows) == ["TOTAL 6.50", "CASH"]


def test_rows_follow_the_page_not_the_input_order() -> None:
    rows = group_rows([seg("C", 0, 300), seg("A", 0, 100), seg("B", 0, 200)])
    assert texts(rows) == ["A", "B", "C"]


def test_slight_skew_stays_on_one_row() -> None:
    rows = group_rows([seg("A", 0, 100), seg("B", 100, 104), seg("C", 200, 108)])
    assert texts(rows) == ["A B C"]


def test_insufficient_overlap_starts_a_new_row() -> None:
    rows = group_rows([seg("A", 0, 100), seg("B", 100, 107)], min_overlap=0.5)
    assert texts(rows) == ["A", "B"]


def test_segments_without_position_are_kept_at_the_end_in_order() -> None:
    rows = group_rows([Segment("x"), seg("A", 0, 100), Segment("y")])
    assert texts(rows) == ["A", "x", "y"]


def test_generators_are_accepted() -> None:
    rows = group_rows(s for s in [seg("A", 0, 0), Segment("free")])
    assert texts(rows) == ["A", "free"]


def test_row_box_and_union() -> None:
    row = group_rows([seg("A", 10, 100, 20, 10), seg("B", 50, 102, 30, 10)])[0]
    assert row.box == BBox(x=10, y=100, width=70, height=12)
    assert union([]) is None
    assert Row((Segment("free"),)).box is None
