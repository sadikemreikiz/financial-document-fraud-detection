"""The receipts' images and annotations kept for the deployment, on made-up receipts."""

import json
from pathlib import Path

import pytest
from PIL import Image

from docfraud.data.download import sha256_of
from docfraud.data.models import BBox, Document, Entity, Label, Region, RegionRole, Technique
from docfraud.demo.assets import ANNOTATIONS, AssetMismatchError, export, image_name, load
from tests.llm.test_agent_rounds import document

EDIT = BBox(x=2, y=3, width=4, height=5)
SOURCE = BBox(x=10, y=3, width=4, height=5)


def receipt(tmp_path: Path, name: str, label: Label) -> Document:
    path = tmp_path / "data" / f"{name}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGBA", (20, 12), (255, 255, 255, 255)).save(path)
    regions = (
        (
            Region(
                box=EDIT,
                role=RegionRole.MODIFIED,
                techniques=frozenset({Technique.CPI}),
                entity=Entity.PRODUCT,
            ),
            Region(box=SOURCE, role=RegionRole.SOURCE),
        )
        if label is Label.FORGED
        else ()
    )
    return document(name, label).model_copy(
        update={
            "image_path": path,
            "regions": regions,
            "has_region_annotations": bool(regions),
            "handwritten_annotation": label is Label.AUTHENTIC,
        }
    )


def test_export_copies_the_files_unchanged_and_writes_their_annotations(tmp_path: Path) -> None:
    documents = [receipt(tmp_path, "F1", Label.FORGED), receipt(tmp_path, "G1", Label.AUTHENTIC)]
    out = tmp_path / "assets"
    digests = export(documents, out)
    for d in documents:
        copied = out / image_name(d.doc_id)
        assert copied.read_bytes() == d.image_path.read_bytes()
        assert digests[d.doc_id] == sha256_of(d.image_path)
    written = json.loads((out / ANNOTATIONS).read_text(encoding="utf-8"))
    assert written["F1"] == {
        "label": "forged",
        "width": 20,
        "height": 12,
        "regions": [
            {"box": [2, 3, 4, 5], "role": "modified", "techniques": ["cpi"], "entity": "product"},
            {"box": [10, 3, 4, 5], "role": "source", "techniques": [], "entity": None},
        ],
        "handwritten": False,
        "digital": False,
    }
    assert written["G1"]["handwritten"] is True
    assert load(out, digests) == {r: written[r] for r in digests}


def test_a_receipt_without_a_label_is_not_exported(tmp_path: Path) -> None:
    unlabelled = receipt(tmp_path, "U1", Label.AUTHENTIC).model_copy(update={"label": None})
    with pytest.raises(ValueError, match="no label"):
        export([unlabelled], tmp_path / "assets")


def test_load_refuses_anything_that_is_not_what_was_pinned(tmp_path: Path) -> None:
    out = tmp_path / "assets"
    digests = export([receipt(tmp_path, "F1", Label.FORGED)], out)
    with pytest.raises(AssetMismatchError, match="pins"):
        load(out, {"F1": "0" * 64})
    with pytest.raises(AssetMismatchError, match="missing"):
        load(out, digests | {"F2": "0" * 64})
    (out / ANNOTATIONS).write_text("{}", encoding="utf-8")
    with pytest.raises(AssetMismatchError, match="no annotation"):
        load(out, digests)
    (out / ANNOTATIONS).unlink()
    with pytest.raises(AssetMismatchError, match="missing"):
        load(out, digests)
