from pathlib import Path

import pytest

from docfraud.data.audit import audit_dataset, build_audit, inspect_files, roc_auc
from docfraud.data.models import Document
from docfraud.markdown import Cell, Table
from tests.fixtures.findit import Entry, load_toy, standard_entries, write_dataset


def documents(root: Path) -> tuple[Document, ...]:
    return load_toy(root)


@pytest.mark.parametrize(
    ("scores", "labels", "expected"),
    [
        ([0.1, 0.2, 0.8, 0.9], [False, False, True, True], 1.0),
        ([0.9, 0.8, 0.2, 0.1], [False, False, True, True], 0.0),
        ([0.5, 0.5, 0.5, 0.5], [False, True, False, True], 0.5),
        ([1, 2, 2, 3], [False, False, True, True], 0.875),
    ],
)
def test_roc_auc(scores: list[float], labels: list[bool], expected: float) -> None:
    assert roc_auc(scores, labels) == pytest.approx(expected)


def test_roc_auc_needs_both_classes() -> None:
    with pytest.raises(ValueError, match="positive and one negative"):
        roc_auc([0.1, 0.2], [True, True])


def test_inspect_files_measures_format_traits(findit_root: Path) -> None:
    traits = {t.document.doc_id: t for t in map(inspect_files, documents(findit_root))}
    f1, a3 = traits["F1"], traits["A3"]
    assert f1.has_alpha
    assert f1.crlf
    assert not f1.normalized_crlf
    assert f1.effectively_gray
    assert (f1.width, f1.height) == (40, 60)
    assert f1.normalized_transcript_bytes == len(b"TOTAL 99.00\n")
    assert a3.mode == "RGB"
    assert not a3.has_alpha


def by_first_column(table: Table) -> dict[str, list[Cell]]:
    return {str(row[0]): list(row[1:]) for row in table.rows}


def tables(root: Path) -> dict[str, Table]:
    return {t.title: t for t in build_audit([inspect_files(d) for d in documents(root)])}


def test_audit_composition_and_alpha_rule(findit_root: Path) -> None:
    audit = tables(findit_root)
    composition = {k: v[:3] for k, v in by_first_column(audit["Composition"]).items()}
    assert composition == {
        "train": [3, 2, 1],
        "val": [1, 0, 1],
        "test": [2, 1, 1],
        "total": [6, 3, 3],
    }
    alpha = {
        row[0]: row[1:] for row in audit["Shortcut rule: 'the image has an alpha channel'"].rows
    }
    assert alpha["train"] == [1.0, 0.5, pytest.approx(2 / 3)]
    assert alpha["test"] == [0.0, 0.0, 0.0]


def test_audit_region_statistics(findit_root: Path) -> None:
    rows = by_first_column(tables(findit_root)["Forged regions"])
    assert rows["Modified regions"] == [4]
    assert rows["Copy-source regions"] == [1]
    assert rows["Modified regions per document (median / max)"] == ["1 / 2"]
    assert rows["Region area in px² (median / 90th percentile)"] == ["12 / 12"]


def test_audit_handles_forgeries_without_regions(tmp_path: Path) -> None:
    root = write_dataset(tmp_path / "d", [Entry("A"), Entry("U", forged="1")])
    docs = load_toy(root, unannotated_forgeries=frozenset({"U"}))
    audit = {t.title: t for t in build_audit([inspect_files(d) for d in docs])}
    rows = by_first_column(audit["Forged regions"])
    assert rows["Region area in px² (median / 90th percentile)"] == ["\N{EM DASH}"]
    integrity = by_first_column(audit["Integrity checks"])
    assert integrity["Forged documents without region annotations"] == ["U"]


def test_audit_dataset_writes_markdown(findit_root: Path, tmp_path: Path) -> None:
    output = tmp_path / "reports" / "audit.md"
    audit_dataset(documents(findit_root), output, source="Toy source")
    text = output.read_text(encoding="utf-8")
    assert text.startswith("# Data audit: Find-it-again")
    assert "Toy source" in text
    assert "### Shortcut audit: binary traits" in text


def test_single_region_dataset(tmp_path: Path) -> None:
    entries = [e for e in standard_entries() if e.doc_id in {"A1", "F3"}]
    root = write_dataset(tmp_path / "d", entries)
    rows = by_first_column(tables(root)["Forged regions"])
    assert rows["Region area in % of the page (median / 90th percentile)"] == ["0.500 / 0.500"]


def test_duplicate_table_lists_kept_and_removed_copies(findit_root: Path) -> None:
    from docfraud.data.duplicates import (  # noqa: PLC0415
        DuplicateGroup,
        DuplicateKind,
        resolve_duplicates,
    )

    groups = [DuplicateGroup(("A1", "A3"), DuplicateKind.COPY, "identical pixels")]
    kept, dropped = resolve_duplicates(list(documents(findit_root)), groups)
    table = {t.title: t for t in build_audit([inspect_files(d) for d in kept], groups, dropped)}
    rows = table["Copies of the same receipt"].rows
    assert rows == [["copy", "identical pixels", "A3", "A1 (train, authentic)"]]
    assert "1 documents removed" in table["Copies of the same receipt"].note
