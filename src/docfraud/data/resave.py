"""Track R: the test images saved again as JPEG, as a forger who cleans a file might (ADR-0034).

Each image is opened as distributed, rendered on white paper where it is transparent, as
:func:`~docfraud.data.normalize.to_grayscale` renders it, converted to RGB and saved as JPEG at
one quality with Pillow's default settings. Nothing of the original file's metadata is carried
over. The saved file then enters the T2 pipeline from the start, with the receipt's identifier,
split and transcript unchanged (decision 4).
"""

import hashlib
from collections.abc import Sequence
from pathlib import Path

from PIL import Image

from docfraud.data.models import Document
from docfraud.data.normalize import PAPER_WHITE

QUALITIES = (95, 75)
"""The JPEG qualities of track R (evaluation protocol, section 2)."""


def on_paper(image: Image.Image) -> Image.Image:
    """The image in RGB, its transparent pixels composited onto white paper."""
    if image.mode in {"RGBA", "LA", "PA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        paper = Image.new("RGBA", rgba.size, (PAPER_WHITE, PAPER_WHITE, PAPER_WHITE, 255))
        image = Image.alpha_composite(paper, rgba)
    return image.convert("RGB")


def resave(source: Path, quality: int, directory: Path) -> Path:
    """Save ``source`` again as JPEG at ``quality`` into ``directory``, and return the new file.

    The file is named after the source and written every time, so that no run reuses a copy
    made otherwise.
    """
    with Image.open(source) as image:
        rgb = on_paper(image)
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / f"{source.stem}.jpg"
    rgb.save(target, format="JPEG", quality=quality)
    return target


def resaved(documents: Sequence[Document], quality: int, directory: Path) -> list[Document]:
    """The receipts with their images saved again at ``quality`` into ``directory``."""
    return [
        d.model_copy(update={"image_path": resave(d.image_path, quality, directory)})
        for d in documents
    ]


def digest(documents: Sequence[Document]) -> str:
    """One SHA-256 over the images of the receipts, in their order: which files a run read."""
    found = hashlib.sha256()
    for document in documents:
        found.update(hashlib.sha256(document.image_path.read_bytes()).digest())
    return found.hexdigest()
