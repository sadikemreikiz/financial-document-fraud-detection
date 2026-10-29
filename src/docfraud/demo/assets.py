"""The shown receipts' images and annotations, kept outside this repository (ADR-0035, 6).

:func:`export` writes them, as the dataset distributes them, to a directory that becomes the
owner's private repository; :func:`load` reads them back when the page is built, and refuses an
image whose SHA-256 is not the one the ``demo-cards`` record pinned.
"""

import json
import shutil
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

from PIL import Image

from docfraud.data.download import sha256_of
from docfraud.data.models import Document
from docfraud.demo.cards import box

ANNOTATIONS = "annotations.json"


class AssetMismatchError(RuntimeError):
    """Raised when an image or its annotation is missing or differs from what was pinned."""


def image_name(receipt: str) -> str:
    """The file name of a receipt's image, in the assets and on the page."""
    return f"{receipt}.png"


def annotation(document: Document, size: tuple[int, int]) -> dict[str, Any]:
    """What the page shows of the dataset's annotation of one receipt."""
    if document.label is None:
        msg = f"{document.doc_id} has no label"
        raise ValueError(msg)
    return {
        "label": document.label.value,
        "width": size[0],
        "height": size[1],
        "regions": [
            {
                "box": box(region.box),
                "role": region.role.value,
                "techniques": sorted(t.value for t in region.techniques),
                "entity": region.entity.value if region.entity else None,
            }
            for region in document.regions
        ],
        "handwritten": document.handwritten_annotation,
        "digital": document.digital_annotation,
    }


def export(documents: Sequence[Document], destination: Path) -> dict[str, str]:
    """Copy the receipts' image files unchanged and write their annotations.

    Returns:
        The SHA-256 of every copied image, by receipt.
    """
    destination.mkdir(parents=True, exist_ok=True)
    annotations: dict[str, Any] = {}
    digests: dict[str, str] = {}
    for document in documents:
        target = destination / image_name(document.doc_id)
        shutil.copyfile(document.image_path, target)
        with Image.open(target) as image:
            annotations[document.doc_id] = annotation(document, image.size)
        digests[document.doc_id] = sha256_of(target)
    text = json.dumps(annotations, indent=2, sort_keys=True, ensure_ascii=False)
    (destination / ANNOTATIONS).write_text(text + "\n", encoding="utf-8")
    return digests


def load(directory: Path, pinned: Mapping[str, str]) -> dict[str, dict[str, Any]]:
    """The annotations of the pinned receipts, once every image matches its SHA-256.

    Raises:
        AssetMismatchError: If an image or annotation is missing, or an image differs.
    """
    path = directory / ANNOTATIONS
    if not path.exists():
        msg = f"{path} is missing"
        raise AssetMismatchError(msg)
    annotations = json.loads(path.read_text(encoding="utf-8"))
    for receipt, digest in pinned.items():
        image = directory / image_name(receipt)
        if not image.exists():
            msg = f"{image} is missing"
            raise AssetMismatchError(msg)
        found = sha256_of(image)
        if found != digest:
            msg = f"{image} has SHA-256 {found}; the demo's record pins {digest}"
            raise AssetMismatchError(msg)
        if receipt not in annotations:
            msg = f"{path} has no annotation of {receipt}"
            raise AssetMismatchError(msg)
    return {receipt: annotations[receipt] for receipt in pinned}
