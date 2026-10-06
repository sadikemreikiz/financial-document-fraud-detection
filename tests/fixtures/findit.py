"""Builders for miniature Find-it-again datasets used by unit tests."""

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pytest
from PIL import Image

HEADER = ["image", "digital annotation", "handwritten annotation", "forged", "forgery annotations"]
WHITE: dict[str, Any] = {"L": 255, "LA": (255, 255), "RGB": (255, 255, 255)}


@dataclass(frozen=True)
class Entry:
    doc_id: str
    split: str = "train"
    forged: str = "0"
    digital: str = "0"
    handwritten: str = "0"
    annotation: str = "0"
    image_mode: str = "L"
    transcript: bytes = b"TOTAL 10.00\n"
    folder: str | None = None


def region(
    *,
    x: int = 1,
    y: int = 2,
    width: int = 3,
    height: int = 4,
    modified: Any = None,
    entity: str = "Total/payment",
    original: str = "no",
    shape: str = "rect",
) -> dict[str, Any]:
    return {
        "shape_attributes": {"name": shape, "x": x, "y": y, "width": width, "height": height},
        "region_attributes": {
            "Modified area": {"CPI": True} if modified is None else modified,
            "Entity type": entity,
            "Original area": original,
        },
    }


def annotation(doc_id: str, regions: list[dict[str, Any]], software: str = "gimp") -> str:
    return repr(
        {
            "filename": f"{doc_id}.png",
            "size": 1,
            "regions": regions,
            "file_attributes": {"Software used": software, "Comment": ""},
        }
    )


def write_dataset(root: Path, entries: list[Entry]) -> Path:
    root.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        with (root / f"{split}.txt").open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(HEADER)
            for e in entries:
                if e.split == split:
                    writer.writerow(
                        [f"{e.doc_id}.png", e.digital, e.handwritten, e.forged, e.annotation]
                    )
    for e in entries:
        folder = root / (e.folder or e.split)
        folder.mkdir(exist_ok=True)
        image = folder / f"{e.doc_id}.png"
        if not image.exists():
            colour = WHITE.get(e.image_mode, (255, 255, 255, 255))
            Image.new(e.image_mode, (40, 60), colour).save(image)
            (folder / f"{e.doc_id}.txt").write_bytes(e.transcript)
    return root


def standard_entries() -> list[Entry]:
    """A small, consistent dataset that needs no corrections."""
    return [
        Entry("A1", handwritten="1"),
        Entry(
            "F1",
            forged="1",
            image_mode="RGBA",
            transcript=b"TOTAL 99.00\r\n",
            annotation=annotation(
                "F1",
                [
                    region(modified={"CPI": True, "PIX": True}),
                    region(x=10, modified={"None": True}, original="yes", entity="Metadata"),
                ],
            ),
        ),
        Entry(
            "F2",
            forged="1",
            annotation=annotation(
                "F2", [region(modified="CPI", entity="Product"), region(modified={"Autre": True})]
            ),
        ),
        Entry("A2", split="val", digital="1"),
        Entry("A3", split="test", image_mode="RGB"),
        Entry(
            "F3",
            split="test",
            forged="1",
            annotation=annotation("F3", [region(entity="Company")], software=""),
        ),
    ]


@pytest.fixture
def findit_root(tmp_path: Path) -> Path:
    return write_dataset(tmp_path / "findit2", standard_entries())
