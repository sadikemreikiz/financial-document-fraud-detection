from pathlib import Path

import pytest

from docfraud.data.gold import (
    GoldManifest,
    annotation_path,
    development_documents,
    draw_gold_set,
    load_annotations,
    load_manifest,
    save_manifest,
)
from docfraud.data.models import Document, Label, Split


def doc(i: int, split: Split, label: Label, dataset: str = "d") -> Document:
    return Document(
        dataset=dataset,
        doc_id=f"D{i:03}",
        source_id=f"s/{i}",
        split=split,
        label=label,
        image_path=Path(f"{i}.png"),
    )


DOCS = [
    *(doc(i, Split.TRAIN, Label.AUTHENTIC) for i in range(30)),
    *(doc(i, Split.TRAIN, Label.FORGED) for i in range(30, 40)),
    *(doc(i, Split.VAL, Label.AUTHENTIC) for i in range(40, 50)),
    *(doc(i, Split.TEST, Label.FORGED) for i in range(50, 60)),
]


def test_draw_is_stratified_and_training_only() -> None:
    manifest = draw_gold_set(DOCS, n_authentic=8, n_forged=3)
    by_id = {d.doc_id: d for d in DOCS}
    assert len(manifest.authentic) == 8
    assert len(manifest.forged) == 3
    assert all(by_id[i].split is Split.TRAIN for i in manifest.doc_ids)
    assert all(by_id[i].label is Label.AUTHENTIC for i in manifest.authentic)
    assert all(by_id[i].label is Label.FORGED for i in manifest.forged)


def test_draw_depends_only_on_seed_and_identifiers() -> None:
    first = draw_gold_set(DOCS, n_authentic=8, n_forged=3, seed=7)
    assert draw_gold_set(list(reversed(DOCS)), n_authentic=8, n_forged=3, seed=7) == first
    assert draw_gold_set(DOCS, n_authentic=8, n_forged=3, seed=8) != first


def test_strata_must_be_large_enough() -> None:
    with pytest.raises(ValueError, match="cannot draw 11 forged"):
        draw_gold_set(DOCS, n_authentic=1, n_forged=11)


def test_one_dataset_only() -> None:
    mixed = [*DOCS, doc(99, Split.TRAIN, Label.AUTHENTIC, dataset="other")]
    with pytest.raises(ValueError, match="exactly one dataset"):
        draw_gold_set(mixed)


def test_development_documents_exclude_gold_and_non_training(tmp_path: Path) -> None:
    manifest = draw_gold_set(DOCS, n_authentic=8, n_forged=3)
    development = development_documents(DOCS, manifest)
    assert len(development) == 40 - 11
    assert not {d.doc_id for d in development} & manifest.doc_ids
    assert all(d.split is Split.TRAIN for d in development)
    path = tmp_path / "gold" / "manifest.json"
    save_manifest(manifest, path)
    assert load_manifest(path) == manifest
    assert isinstance(load_manifest(path), GoldManifest)


def annotation_json(doc_id: str, total: str = "5.00") -> str:
    return (
        f'{{"doc_id": "{doc_id}", "annotator": "test", '
        f'"fields": {{"amounts": {{"total_due": ["{total}"]}}}}}}'
    )


def test_annotations_are_loaded_and_validated(tmp_path: Path) -> None:
    manifest = draw_gold_set(DOCS, n_authentic=2, n_forged=1)
    first = sorted(manifest.doc_ids)[0]
    (tmp_path / f"{first}.json").write_text(annotation_json(first), encoding="utf-8")
    save_manifest(manifest, tmp_path / "manifest.json")
    loaded = load_annotations(tmp_path, manifest)
    assert list(loaded) == [first]
    assert (
        str(loaded[first].fields.amounts)
        == "{<AmountRole.TOTAL_DUE: 'total_due'>: (Decimal('5.00'),)}"
    )
    assert annotation_path(tmp_path, first) == tmp_path / f"{first}.json"


@pytest.mark.parametrize("name", ["X999", "mismatch"])
def test_annotations_outside_the_manifest_are_rejected(tmp_path: Path, name: str) -> None:
    manifest = draw_gold_set(DOCS, n_authentic=2, n_forged=1)
    inside = sorted(manifest.doc_ids)[0]
    doc_id = "X999" if name == "X999" else inside
    file_name = "X999.json" if name == "X999" else "other.json"
    (tmp_path / file_name).write_text(annotation_json(doc_id), encoding="utf-8")
    with pytest.raises(ValueError, match="not a gold receipt"):
        load_annotations(tmp_path, manifest)
