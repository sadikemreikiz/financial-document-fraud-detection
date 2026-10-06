from collections.abc import Mapping
from pathlib import Path

import pytest

from docfraud.data.duplicates import DuplicateGroup, DuplicateKind
from docfraud.data.findit import (
    DUPLICATE_RESOLUTIONS,
    KNOWN_CORRECTIONS,
    UNANNOTATED_FORGERIES,
    DataIntegrityError,
)
from docfraud.data.models import Document, Entity, Label, RegionRole, Split, Technique
from tests.fixtures.findit import (
    Entry,
    annotation,
    load_toy,
    region,
    standard_entries,
    write_dataset,
)


def load(
    root: Path,
    duplicate_resolutions: Mapping[str, tuple[Split, Label]] | None = None,
    unannotated_forgeries: frozenset[str] = frozenset(),
    duplicate_groups: tuple[DuplicateGroup, ...] = (),
) -> dict[str, Document]:
    """Load a miniature dataset; corrections default to none."""
    documents = load_toy(
        root,
        duplicate_resolutions=duplicate_resolutions or {},
        unannotated_forgeries=unannotated_forgeries,
        duplicate_groups=duplicate_groups,
    )
    return {d.doc_id: d for d in documents}


def test_loads_documents_in_split_then_id_order(findit_root: Path) -> None:
    docs = list(load(findit_root).values())
    assert [d.doc_id for d in docs] == ["A1", "F1", "F2", "A2", "A3", "F3"]
    assert [d.split for d in docs] == [Split.TRAIN] * 3 + [Split.VAL] + [Split.TEST] * 2


def test_document_fields(findit_root: Path) -> None:
    docs = load(findit_root)
    a1, f1, a2 = docs["A1"], docs["F1"], docs["A2"]
    assert a1.label is Label.AUTHENTIC
    assert a1.handwritten_annotation
    assert not a1.digital_annotation
    assert a2.digital_annotation
    assert f1.label is Label.FORGED
    assert f1.dataset == "findit2"
    assert f1.source_id == "sroie/F1"
    assert f1.image_path == findit_root / "train" / "F1.png"
    assert f1.transcript_path == findit_root / "train" / "F1.txt"
    assert f1.editing_software == "gimp"
    assert docs["F3"].editing_software is None


def test_regions_are_parsed_with_roles_techniques_and_entities(findit_root: Path) -> None:
    docs = load(findit_root)
    modified, source = docs["F1"].regions
    assert modified.role is RegionRole.MODIFIED
    assert modified.techniques == {Technique.CPI, Technique.PIX}
    assert modified.entity is Entity.TOTAL_PAYMENT
    assert (modified.box.x, modified.box.y, modified.box.width, modified.box.height) == (1, 2, 3, 4)
    assert source.role is RegionRole.SOURCE
    assert source.techniques == frozenset()
    assert source.entity is Entity.METADATA
    string_form, french = docs["F2"].regions
    assert string_form.techniques == {Technique.CPI}
    assert french.techniques == {Technique.OTHER}


def test_known_corrections_document_the_configured_defects() -> None:
    documented = {c.doc_id for c in KNOWN_CORRECTIONS}
    assert documented == set(DUPLICATE_RESOLUTIONS) | UNANNOTATED_FORGERIES


def defects() -> list[Entry]:
    dup = annotation("D1", [region()])
    return [
        *standard_entries(),
        Entry("D1", split="train", forged="0", annotation=dup, image_mode="RGBA"),
        Entry("D1", split="val", forged="1", annotation=dup, folder="train"),
        Entry("U1", split="val", forged="1"),
    ]


def test_documented_corrections_are_applied(tmp_path: Path) -> None:
    root = write_dataset(tmp_path / "d", defects())
    docs = load(
        root,
        duplicate_resolutions={"D1": (Split.TRAIN, Label.FORGED)},
        unannotated_forgeries=frozenset({"U1"}),
    )
    assert len(docs) == 8
    assert docs["D1"].split is Split.TRAIN
    assert docs["D1"].label is Label.FORGED
    assert docs["U1"].label is Label.FORGED
    assert not docs["U1"].has_region_annotations


def test_undocumented_duplicate_is_an_error(tmp_path: Path) -> None:
    root = write_dataset(tmp_path / "d", defects())
    with pytest.raises(DataIntegrityError, match="several splits"):
        load(root, unannotated_forgeries=frozenset({"U1"}))


def test_undocumented_missing_annotation_is_an_error(tmp_path: Path) -> None:
    root = write_dataset(tmp_path / "d", defects())
    with pytest.raises(DataIntegrityError, match="no region annotations"):
        load(root, duplicate_resolutions={"D1": (Split.TRAIN, Label.FORGED)})


def test_stale_correction_is_an_error(findit_root: Path) -> None:
    with pytest.raises(DataIntegrityError, match="no longer apply"):
        load(findit_root, unannotated_forgeries=frozenset({"GONE"}))


def test_resolution_must_point_to_an_existing_row(tmp_path: Path) -> None:
    root = write_dataset(tmp_path / "d", defects())
    with pytest.raises(DataIntegrityError, match="exactly one row"):
        load(
            root,
            duplicate_resolutions={"D1": (Split.TEST, Label.FORGED)},
            unannotated_forgeries=frozenset({"U1"}),
        )


def with_extra(tmp_path: Path, entry: Entry) -> Path:
    return write_dataset(tmp_path / "d", [*standard_entries(), entry])


@pytest.mark.parametrize(
    ("entry", "message"),
    [
        (Entry("X", annotation=annotation("X", [region()])), "authentic but carries"),
        (Entry("X", forged="1", annotation=annotation("Y", [region()])), "does not describe"),
        (Entry("X", forged="1", annotation="{'filename': "), "not a valid literal"),
        (Entry("X", forged="1", annotation=annotation("X", [region(shape="polygon")])), "shape"),
        (Entry("X", forged="1", annotation=annotation("X", [region(entity="Tax")])), "attributes"),
        (
            Entry("X", forged="1", annotation=annotation("X", [region(original="maybe")])),
            "attributes",
        ),
        (
            Entry("X", forged="1", annotation=annotation("X", [region(modified={"AI": True})])),
            "techniques",
        ),
        (Entry("X", forged="2"), "0/1 flag"),
    ],
    ids=[
        "authentic-annotated",
        "wrong-file",
        "bad-literal",
        "shape",
        "entity",
        "role",
        "technique",
        "flag",
    ],
)
def test_inconsistencies_are_errors(tmp_path: Path, entry: Entry, message: str) -> None:
    with pytest.raises(DataIntegrityError, match=message):
        load(with_extra(tmp_path, entry))


def test_unexpected_header_is_an_error(findit_root: Path) -> None:
    (findit_root / "val.txt").write_text("name,forged\n", encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="header"):
        load(findit_root)


def test_unexpected_image_name_is_an_error(findit_root: Path) -> None:
    path = findit_root / "val.txt"
    path.write_text(path.read_text(encoding="utf-8").replace("A2.png", "A2.jpg"), encoding="utf-8")
    with pytest.raises(DataIntegrityError, match="image name"):
        load(findit_root)


def test_missing_files_are_reported(findit_root: Path) -> None:
    (findit_root / "test" / "A3.txt").unlink()
    with pytest.raises(FileNotFoundError, match=r"A3\.txt"):
        load(findit_root)


def test_duplicate_groups_are_resolved_towards_the_test_split(findit_root: Path) -> None:
    group = DuplicateGroup(("A1", "A3"), DuplicateKind.COPY, "same receipt")
    docs = load(findit_root, duplicate_groups=(group,))
    assert "A1" not in docs
    assert docs["A3"].split is Split.TEST


def test_duplicate_groups_with_unknown_members_are_errors(findit_root: Path) -> None:
    group = DuplicateGroup(("A1", "GONE"), DuplicateKind.COPY, "same receipt")
    with pytest.raises(DataIntegrityError, match="unknown documents"):
        load(findit_root, duplicate_groups=(group,))


def test_registered_groups_and_distinct_pairs_do_not_overlap() -> None:
    from docfraud.data.findit import DUPLICATE_GROUPS, REVIEWED_DISTINCT  # noqa: PLC0415

    members = [m for g in DUPLICATE_GROUPS for m in g.members]
    assert len(members) == len(set(members)), "a document may belong to one group only"
    for pair in REVIEWED_DISTINCT:
        assert not any(set(pair.members) <= set(g.members) for g in DUPLICATE_GROUPS)
