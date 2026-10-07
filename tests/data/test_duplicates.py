from pathlib import Path

import pytest
from PIL import Image, ImageDraw

from docfraud.data.duplicates import (
    Candidate,
    DistinctPair,
    DuplicateGroup,
    DuplicateKind,
    find_candidates,
    group_index,
    resolve_duplicates,
    undocumented,
)
from docfraud.data.models import Document, Label, Split
from docfraud.data.splits import check_disjoint


def doc(
    doc_id: str, split: Split, label: Label = Label.AUTHENTIC, path: Path = Path("x")
) -> Document:
    return Document(
        dataset="d",
        doc_id=doc_id,
        source_id=f"s/{doc_id}",
        split=split,
        label=label,
        image_path=path / f"{doc_id}.png",
        transcript_path=path / f"{doc_id}.txt",
    )


COPY, TWIN = DuplicateKind.COPY, DuplicateKind.TWIN


def ids(documents: list[Document]) -> list[str]:
    return [d.doc_id for d in documents]


def test_the_evaluation_split_copy_is_kept() -> None:
    docs = [doc("a", Split.TRAIN), doc("b", Split.TEST), doc("c", Split.VAL)]
    kept, dropped = resolve_duplicates(docs, [DuplicateGroup(("a", "b", "c"), COPY, "same")])
    assert ids(kept) == ["b"]
    assert {d.document.doc_id: d.kept for d in dropped} == {"a": ("b",), "c": ("b",)}
    assert all("kept in test" in d.reason for d in dropped)


def test_extra_same_label_copies_in_one_split_are_dropped() -> None:
    docs = [doc("b", Split.TRAIN), doc("a", Split.TRAIN)]
    kept, dropped = resolve_duplicates(docs, [DuplicateGroup(("b", "a"), COPY, "same")])
    assert ids(kept) == ["a"]
    assert dropped[0].reason == "extra copy with the same label in the same split"


def test_twins_in_one_split_are_both_kept_and_share_a_source() -> None:
    docs = [doc("orig", Split.VAL), doc("fake", Split.VAL, Label.FORGED), doc("other", Split.VAL)]
    kept, dropped = resolve_duplicates(docs, [DuplicateGroup(("orig", "fake"), TWIN, "forged")])
    assert dropped == []
    assert ids(kept) == ["orig", "fake", "other"]
    assert kept[0].source_id == kept[1].source_id == "s/fake"
    assert kept[2].source_id == "s/other"


def test_a_twin_in_training_is_dropped_when_its_sibling_is_tested() -> None:
    docs = [doc("fake", Split.TRAIN, Label.FORGED), doc("orig", Split.TEST)]
    kept, _ = resolve_duplicates(docs, [DuplicateGroup(("fake", "orig"), TWIN, "forged")])
    assert ids(kept) == ["orig"]


def test_resolution_removes_split_leakage() -> None:
    docs = [doc("a", Split.TRAIN), doc("b", Split.TEST)]
    kept, _ = resolve_duplicates(docs, [DuplicateGroup(("a", "b"), COPY, "same")])
    check_disjoint(kept)
    merged = [d.model_copy(update={"source_id": "s/a"}) for d in docs]
    with pytest.raises(ValueError, match="several splits"):
        check_disjoint(merged)


def test_unknown_members_raise() -> None:
    with pytest.raises(KeyError):
        resolve_duplicates([doc("a", Split.TRAIN)], [DuplicateGroup(("a", "zz"), COPY, "same")])


def test_undocumented_and_group_index() -> None:
    groups = [DuplicateGroup(("a", "b", "c"), COPY, "same")]
    distinct = [DistinctPair(("x", "y"), "different receipt")]
    candidates = [
        Candidate(("a", "c"), ("identical pixels",), 1.0),
        Candidate(("y", "x"), ("shared rare identifier",), 0.8),
        Candidate(("a", "x"), ("near-identical image",), 0.9),
    ]
    assert [c.members for c in undocumented(candidates, groups, distinct)] == [("a", "x")]
    assert group_index(groups)["c"] is groups[0]


def write(path: Path, doc_id: str, text: str, mark: tuple[int, int] | None = None) -> None:
    image = Image.new("L", (120, 160), 255)
    draw = ImageDraw.Draw(image)
    for row in range(0, 160, 20):
        draw.rectangle([10, row + 5, 90, row + 12], fill=0)
    if mark:
        draw.rectangle([mark[0], mark[1], mark[0] + 3, mark[1] + 3], fill=128)
    image.save(path / f"{doc_id}.png")
    (path / f"{doc_id}.txt").write_text(text, encoding="utf-8")


def test_find_candidates_on_small_pages(tmp_path: Path) -> None:
    header = "SHOP SDN BHD\nTEL 03-1234\nTAX INVOICE\n"
    write(tmp_path, "copy1", header + "DATE 01/02/2018 10:15\nTOTAL 9.90\n")
    write(tmp_path, "copy2", header + "DATE 01/02/2018 10:15\nTOTAL 9.90\n")
    write(tmp_path, "forged", header + "DATE 01/02/2018 10:15\nTOTAL 99.90\n", mark=(100, 140))
    write(tmp_path, "other", "OTHER SHOP\nDATE 05/05/2018 09:00\nTOTAL 1.00\n", mark=(50, 3))
    docs = [doc(n, Split.TRAIN, path=tmp_path) for n in ("copy1", "copy2", "forged", "other")]
    found = {c.members: c for c in find_candidates(docs)}
    assert found[("copy1", "copy2")].signals == ("identical pixels",)
    assert "same date and time" in found[("copy1", "forged")].signals
    assert not any("other" in pair for pair in found)
