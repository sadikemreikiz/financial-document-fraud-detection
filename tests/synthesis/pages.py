"""Synthetic receipts for the engine's tests: printed text whose OCR words are known exactly."""

from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont

from docfraud.data.models import BBox, Document, Entity, Label, Region, RegionRole, Split, Technique
from docfraud.evaluation.design_fold import DESIGN_FOLD, DESIGN_FOLD_IDS
from docfraud.evaluation.folds import FoldAssignment
from docfraud.reading.ocr import OcrResult, Token
from docfraud.synthesis.engine import Receipt
from docfraud.synthesis.layout import words

PAPER = 232
INK = 40
SIZE = 24
PADDING = 3
LINES = (
    ("SHOP", "MAKMUR", "SDN", "BHD"),
    ("12", "JALAN", "BESAR"),
    ("TEL", "03-1234"),
    ("TAX", "INVOICE"),
    ("DATE:", "02/01/2019"),
    ("ITEM", "A", "12.50"),
    ("ITEM", "B", "3.20"),
    ("ITEM", "C", "48.75"),
    ("TOTAL", "64.45"),
    ("CASH", "70.00"),
    ("CHANGE", "5.55"),
)


def font(_family: str = "", _bold: bool = False, size: int = SIZE) -> ImageFont.FreeTypeFont:
    """Pillow's own font, which the tests use instead of the downloaded ones."""
    loaded = ImageFont.load_default(size)
    assert isinstance(loaded, ImageFont.FreeTypeFont)
    return loaded


def draw_receipt(
    lines: tuple[tuple[str, ...], ...] = LINES, *, seed: int = 0, width: int = 420
) -> tuple[npt.NDArray[np.uint8], OcrResult]:
    """A page of noisy paper with the lines printed on it, and OCR words that match them."""
    height = 30 + 34 * len(lines)
    noise = np.random.default_rng(seed).normal(PAPER, 3, size=(height, width))
    image = Image.fromarray(np.clip(noise, 0, 255).astype(np.uint8))
    draw = ImageDraw.Draw(image)
    tokens = []
    for number, line in enumerate(lines):
        x, baseline = 20, 40 + 34 * number
        for text in line:
            left, top, right, bottom = (
                round(v) for v in draw.textbbox((x, baseline), text, font=font(), anchor="ls")
            )
            draw.text((x, baseline), text, fill=INK, font=font(), anchor="ls")
            box = BBox(
                x=left - PADDING,
                y=top - PADDING,
                width=right - left + 2 * PADDING,
                height=bottom - top + 2 * PADDING,
            )
            tokens.append(Token(text=text, box=box, confidence=0.99, line=number))
            x = right + 18
    return np.asarray(image).copy(), OcrResult(
        engine="test", width=width, height=height, tokens=tuple(tokens)
    )


@lru_cache(maxsize=16)
def receipt(doc_id: str) -> Receipt:
    """A receipt for the engine; each identifier gets its own noise."""
    page, ocr = draw_receipt(seed=sum(map(ord, doc_id)))
    return Receipt(doc_id, page, words(page, ocr))


def documents(
    tmp: Path, folds: int = 5, per_fold: int = 3
) -> tuple[list[Document], FoldAssignment]:
    """Authentic and forged training documents spread over folds, with images on disk.

    The first receipt of every fold is forged. The design fold uses identifiers of the real design
    fold, so that the development stream has bases; the images are drawn, not the real ones.
    """
    found: list[Document] = []
    assignment: dict[str, int] = {}
    page, _ = draw_receipt()
    design = sorted(DESIGN_FOLD_IDS)
    for fold in range(folds):
        for number in range(per_fold):
            doc_id = design[number] if fold == DESIGN_FOLD else f"X{fold}{number}"
            path = tmp / f"{doc_id}.png"
            Image.fromarray(page).save(path)
            forged = number == 0
            regions = (
                (
                    Region(
                        box=BBox(x=150, y=208, width=30, height=24),
                        role=RegionRole.MODIFIED,
                        techniques=frozenset({Technique.CPI}),
                        entity=Entity.PRODUCT,
                    ),
                )
                if forged
                else ()
            )
            found.append(
                Document(
                    dataset="test",
                    doc_id=doc_id,
                    source_id=doc_id,
                    split=Split.TRAIN,
                    label=Label.FORGED if forged else Label.AUTHENTIC,
                    image_path=path,
                    regions=regions,
                    has_region_annotations=forged,
                )
            )
            assignment[doc_id] = fold
    return found, FoldAssignment(
        dataset="test", split=Split.TRAIN, n_folds=folds, seed=0, folds=assignment
    )
