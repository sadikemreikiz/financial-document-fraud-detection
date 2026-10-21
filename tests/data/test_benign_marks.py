from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import ValidationError

from docfraud.data.benign_marks import (
    MAX_ROUNDS,
    CheckLog,
    Mark,
    MarkLog,
    ReceiptCheck,
    ReceiptMarks,
    check_problems,
    export_views,
    final_marks,
    image_sizes,
    load_check_log,
    load_mark_log,
    load_rounds,
    mark_problems,
    marks_plan,
    round_path,
    round_receipts,
)
from docfraud.data.models import BBox, Document
from docfraud.data.normalize import load_grayscale
from docfraud.data.observation import ASSISTANT, save_log
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS, NotInDesignFoldError
from tests.fixtures.findit import Entry, annotation, load_toy, region, write_dataset

# Identifiers of real receipts, so that the viewing lock applies; the images are toy images.
HAND, DIGITAL, FORGED, PLAIN = "X00016469670", "X00016469671", "X51005230617", "X51005230621"
CLEAN = "X00016469622"
NOW = datetime(2026, 10, 13, tzinfo=UTC)


def marks_documents(root: Path) -> tuple[Document, ...]:
    entries = [
        Entry(HAND, handwritten="1"),
        Entry(DIGITAL, digital="1"),
        Entry(
            FORGED,
            forged="1",
            handwritten="1",
            annotation=annotation(FORGED, [region(x=5, y=10, width=10, height=8)]),
        ),
        Entry(PLAIN),
        Entry(CLEAN, handwritten="1"),
    ]
    return load_toy(write_dataset(root, entries))


def mark(**changes: object) -> Mark:
    values: dict[str, object] = {
        "box": {"x": 2, "y": 3, "width": 10, "height": 5},
        "kind": "handwriting",
        "touches_print": "no",
    }
    return Mark.model_validate(values | changes)


def mark_log(receipts: dict[str, ReceiptMarks], round_number: int = 1) -> MarkLog:
    return MarkLog(
        annotator=ASSISTANT, round=round_number, commit="abc1234", marked_at=NOW, receipts=receipts
    )


def check(verdict: str = "correct", **changes: object) -> ReceiptCheck:
    values: dict[str, object] = {"verdict": verdict, "checked_at": NOW, "commit": "abc1234"}
    return ReceiptCheck.model_validate(values | changes)


def check_log(verdicts: dict[str, str], round_number: int = 1) -> CheckLog:
    receipts = {doc_id: check(v) for doc_id, v in verdicts.items()}
    return CheckLog(checker="emre", round=round_number, receipts=receipts)


def test_toy_identifiers_sit_where_the_tests_assume() -> None:
    assert {HAND, DIGITAL, FORGED, PLAIN} <= DESIGN_FOLD_IDS
    assert CLEAN not in DESIGN_FOLD_IDS


def test_the_plan_takes_the_flagged_design_fold_receipts_in_a_seeded_order(
    tmp_path: Path,
) -> None:
    documents = marks_documents(tmp_path / "d")
    plan = marks_plan(documents)
    assert sorted(plan) == sorted([HAND, DIGITAL, FORGED])  # forged too; not unflagged or clean
    assert marks_plan(reversed(documents)) == plan


def test_marks_and_checks_validate_their_content() -> None:
    with pytest.raises(ValidationError, match="needs a note"):
        mark(kind="other")
    assert mark(kind="other", note="a sticker").note == "a sticker"
    with pytest.raises(ValidationError, match="a correct receipt has none"):
        check("correct", points=[{"x": 1, "y": 1}])
    assert check("missing", points=[{"x": 1, "y": 1}]).points[0].x == 1
    with pytest.raises(ValidationError, match="format version 0"):
        CheckLog(checker="emre", round=1, version=0)
    with pytest.raises(ValidationError, match="format version 2"):
        MarkLog.model_validate(mark_log({}).model_dump() | {"version": 2})
    with pytest.raises(ValidationError, match="does not check its own marks"):
        CheckLog(checker=ASSISTANT, round=1)
    with pytest.raises(ValidationError):
        CheckLog(checker="emre", round=MAX_ROUNDS + 1)


def test_each_round_takes_the_receipts_not_accepted_before() -> None:
    plan = ("a", "b", "c")
    assert round_receipts(plan, []) == plan
    first = check_log({"a": "correct", "b": "missing", "c": "wrong_box"})
    assert round_receipts(plan, [first]) == ("b", "c")
    second = check_log({"b": "correct", "c": "missing_and_wrong"}, round_number=2)
    assert round_receipts(plan, [first, second]) == ("c",)
    with pytest.raises(ValueError, match="round 2 found where round 1"):
        round_receipts(plan, [second])
    with pytest.raises(ValueError, match="must cover its 3 receipts"):
        round_receipts(plan, [check_log({"a": "correct"})])


def test_mark_and_check_problems_name_receipts_and_pages() -> None:
    pages = {"a": BBox(x=0, y=0, width=40, height=60), "b": BBox(x=0, y=0, width=40, height=60)}
    outside = mark(box={"x": 35, "y": 3, "width": 10, "height": 5})
    log = mark_log({"a": ReceiptMarks(marks=(mark(), outside)), "z": ReceiptMarks()})
    assert mark_problems(log, 2, ["a", "b"], pages) == [
        "marks of round 1, not 2",
        "z: not a receipt of this round",
        "b: missing",
        "a: box 2 lies outside the page",
    ]
    checks = CheckLog(
        checker="emre",
        round=1,
        receipts={"a": check("missing", points=[{"x": 40, "y": 1}]), "z": check()},
    )
    assert check_problems(checks, ["a", "b"], pages) == [
        "z: not a receipt of this round",
        "a: a point lies outside the page",
    ]
    assert check_problems(check_log({"a": "correct"}), ["a", "b"], pages) == []


def test_final_marks_take_the_latest_round_of_each_receipt() -> None:
    plan = ("a", "b")
    first_marks = mark_log({"a": ReceiptMarks(marks=(mark(),)), "b": ReceiptMarks(note="none")})
    first_check = check_log({"a": "correct", "b": "missing"})
    second_marks = mark_log({"b": ReceiptMarks(marks=(mark(kind="stamp"),))}, round_number=2)
    second_check = check_log({"b": "correct"}, round_number=2)

    final = final_marks(plan, [first_marks, second_marks], [first_check, second_check])

    assert final.rounds == 2
    assert final.accepted_in == {"a": 1, "b": 2}
    assert final.marks["b"].marks[0].kind == "stamp"
    assert final.marks["a"] == first_marks.receipts["a"]
    with pytest.raises(ValueError, match="round 2 is due"):
        final_marks(plan, [first_marks], [first_check])
    with pytest.raises(ValueError, match="needs its marks and a finished check"):
        final_marks(plan, [first_marks], [])
    with pytest.raises(ValueError, match="marks of round 2 must cover its 1 receipts"):
        final_marks(plan, [first_marks, first_marks], [first_check, second_check])


def test_receipts_not_accepted_after_the_last_round_are_disputed() -> None:
    marks = [mark_log({"a": ReceiptMarks()}, n) for n in range(1, MAX_ROUNDS + 1)]
    checks = [check_log({"a": "wrong_box"}, n) for n in range(1, MAX_ROUNDS + 1)]
    final = final_marks(("a",), marks, checks)
    assert final.accepted_in == {"a": None}
    assert final.rounds == MAX_ROUNDS


def test_rounds_are_saved_and_loaded(tmp_path: Path) -> None:
    assert round_path(tmp_path, "emre", 2) == tmp_path / "emre-2.json"
    assert load_check_log(tmp_path / "emre-1.json", "emre", 1) == CheckLog(checker="emre", round=1)
    marks = mark_log({"a": ReceiptMarks(marks=(mark(),))})
    save_log(marks, round_path(tmp_path, ASSISTANT, 1))
    save_log(check_log({"a": "correct"}), round_path(tmp_path, "emre", 1))
    assert load_mark_log(round_path(tmp_path, ASSISTANT, 1)) == marks
    mark_logs, check_logs = load_rounds(tmp_path, ASSISTANT, "emre")
    assert (mark_logs, [c.round for c in check_logs]) == ([marks], [1])
    with pytest.raises(ValueError, match="is round 1 of 'emre', not round 2 of 'emre'"):
        load_check_log(round_path(tmp_path, "emre", 1), "emre", 2)


def test_image_sizes_come_from_the_header(tmp_path: Path) -> None:
    documents = marks_documents(tmp_path / "d")
    assert image_sizes(documents, [HAND]) == {HAND: BBox(x=0, y=0, width=40, height=60)}


def test_views_show_the_page_at_native_resolution_with_a_grid_and_the_boxes(
    tmp_path: Path,
) -> None:
    root = tmp_path / "d"
    write_dataset(root, [Entry(HAND, handwritten="1")])
    noise = np.random.default_rng(1).integers(0, 256, size=(700, 1700), dtype=np.uint8)
    Image.fromarray(noise).save(root / "train" / f"{HAND}.png")
    documents = load_toy(root)
    page = load_grayscale(root / "train" / f"{HAND}.png")

    files = export_views(documents, [HAND], tmp_path / "views")

    assert [f.name for f in files] == [
        f"01-{HAND}-overview.png",
        f"01-{HAND}-overview-grid.png",
        f"01-{HAND}-tile-0-0.png",
        f"01-{HAND}-tile-0-1.png",
    ]
    overview, _, first, second = (np.asarray(Image.open(f)) for f in files)
    assert overview.shape == (618, 1500)  # reduced so that the longer side is 1500 px
    assert first.shape == (700, 1500, 3)
    assert second.shape == (700, 200, 3)
    assert (first[155, 155] == page[155, 155]).all()  # the page itself is unchanged
    assert tuple(first[50, 100]) == (0, 150, 255)  # grid line at page x = 100
    assert tuple(second[50, 100]) == (0, 150, 255)  # grid line at page x = 1600

    box = Mark.model_validate(
        {"box": {"x": 1650, "y": 300, "width": 40, "height": 20}, "kind": "stamp"}
        | {"touches_print": "yes"}
    )
    marked = export_views(
        documents, [HAND], tmp_path / "marked", {HAND: ReceiptMarks(marks=(box,))}
    )
    assert [f.name for f in marked] == [
        f"01-{HAND}-overview-grid-marked.png",
        f"01-{HAND}-tile-0-0-marked.png",
        f"01-{HAND}-tile-0-1-marked.png",
    ]
    tile = np.asarray(Image.open(marked[2]))
    assert tuple(tile[298, 160]) == (220, 0, 0)  # outline just above the box
    assert (tile[305:315, 155:185] == page[305:315, 1655:1685, None]).all()


def test_views_refuse_receipts_outside_the_design_fold(tmp_path: Path) -> None:
    documents = marks_documents(tmp_path / "d")
    with pytest.raises(NotInDesignFoldError):
        export_views(documents, [CLEAN], tmp_path / "views")
