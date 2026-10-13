from datetime import timedelta
from pathlib import Path

import pytest

from docfraud.data.benign_marks import ReceiptMarks, final_marks, image_sizes, marks_plan
from docfraud.data.benign_marks_report import build_report, render_report
from docfraud.markdown import Table
from tests.data.test_benign_marks import (
    DIGITAL,
    FORGED,
    HAND,
    check,
    check_log,
    mark,
    mark_log,
    marks_documents,
)


def rounds(root: Path) -> tuple[list[Table], str]:
    documents = marks_documents(root)
    plan = marks_plan(documents)
    first = mark_log(
        {
            HAND: ReceiptMarks(marks=(mark(), mark(kind="pen_mark", touches_print="yes"))),
            DIGITAL: ReceiptMarks(marks=(mark(kind="handwriting"),)),
            FORGED: ReceiptMarks(note="no mark found"),
        }
    )
    first_check = check_log({HAND: "correct", DIGITAL: "wrong_box", FORGED: "missing"})
    later = first_check.receipts[HAND].checked_at + timedelta(minutes=3)
    first_check = first_check.model_copy(
        update={"receipts": first_check.receipts | {HAND: check(checked_at=later)}}
    )
    box = {"x": 4, "y": 9, "width": 8, "height": 4}  # overlaps the forged region (5, 10, 10, 8)
    second = mark_log(
        {
            DIGITAL: ReceiptMarks(marks=(mark(kind="handwriting", touches_print="unclear"),)),
            FORGED: ReceiptMarks(marks=(mark(kind="stamp", box=box),)),
        },
        round_number=2,
    )
    second_check = check_log({DIGITAL: "correct", FORGED: "wrong_box"}, round_number=2)
    third = mark_log({FORGED: ReceiptMarks(marks=(mark(kind="stamp", box=box),))}, 3)
    third_check = check_log({FORGED: "missing_and_wrong"}, 3)
    checks = [first_check, second_check, third_check]
    final = final_marks(plan, [first, second, third], checks)
    tables = build_report(documents, plan, final, checks, image_sizes(documents, plan))
    return tables, render_report(tables)


def table(tables: list[Table], title: str) -> list[list[object]]:
    return [list(row) for row in next(t for t in tables if t.title == title).rows]


@pytest.fixture
def report(tmp_path: Path) -> tuple[list[Table], str]:
    return rounds(tmp_path / "d")


def test_coverage_and_checks_follow_the_rounds(report: tuple[list[Table], str]) -> None:
    tables, _ = report
    assert table(tables, "Receipts and boxes") == [
        ["digital note", "authentic", 1, 1, 1],
        ["handwriting", "authentic", 1, 1, 2],
        ["handwriting", "forged", 1, 1, 1],
        ["all", "", 3, 3, 4],
    ]
    assert table(tables, "The owner's check") == [
        [1, 3, 1, 1, 1, 0, 1, "3.0"],
        [2, 2, 1, 0, 1, 0, 1, "0.0"],
        [3, 1, 0, 0, 0, 1, 0, "0.0"],
    ]
    note = next(t for t in tables if t.title == "The owner's check").note
    assert f"Disputed after the last round: {FORGED}" in note


def test_kinds_touching_size_and_position_are_counted(report: tuple[list[Table], str]) -> None:
    tables, _ = report
    assert table(tables, "Kinds of mark")[-1] == ["all", 4, 3]
    assert table(tables, "Kinds of mark")[0] == ["handwriting", 2, 2]
    assert table(tables, "Marks touching print")[-1] == ["all", 4, 1, 2, 1]
    sizes = table(tables, "Size of the boxes")
    assert sizes[0] == ["handwriting", 2, "2.08 %", "2.08 %", "2.08 %", 5]  # 50 px of 40 x 60
    assert table(tables, "Position on the page")[-1] == ["all", 4, 4, 0, 0]


def test_flags_and_forged_regions_are_compared(report: tuple[list[Table], str]) -> None:
    tables, text = report
    flags = table(tables, "Flags against marks")
    assert flags == [
        ["flagged for handwriting, no mark of a hand kind", 2, 0, None],
        ["flagged for a digital note, no digital note", 1, 1, DIGITAL],
        ["digital note found, not flagged for one", 2, 0, None],
        ["mark of a hand kind found, flagged for a digital note only", 1, 1, DIGITAL],
    ]
    assert table(tables, "Marks on forged receipts") == [[FORGED, 1, 1, 1]]
    assert text.startswith("# Benign marks: design fold")
