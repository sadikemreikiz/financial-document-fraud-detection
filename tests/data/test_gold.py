from pathlib import Path

import pytest

from docfraud.data.gold import (
    GoldManifest,
    development_documents,
    draw_gold_set,
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
