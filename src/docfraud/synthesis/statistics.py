"""What a cross-validation round's real forgeries look like (ADR-0014 decision 5, ADR-0021 2).

Every number of the engine that comes from data is computed here, in each round, from the
annotations of that round's training folds only: the modified regions, as templates of
(techniques, entity, width in characters), and the number of modified regions per forged
receipt. The engine draws a template per synthetic region, so the shares of techniques and
entities and the widths follow the round's real regions jointly; a region labelled with several
techniques gives one of them, each equally likely, and "other" is left out.

The width of a region is measured in character widths of the OCR words it overlaps, read from
the forged receipt by code; no person views a receipt here.
"""

from collections import Counter
from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from docfraud.data.models import BBox, Document, Entity, Label, RegionRole, Split, Technique
from docfraud.data.normalize import load_grayscale
from docfraud.evaluation.folds import FoldAssignment
from docfraud.reading.ocr import OcrEngine, OcrResult

ROW_OVERLAP = 0.5
"""Vertical overlap, as a share of the smaller height, of two boxes on one printed row."""


@dataclass(frozen=True, slots=True)
class Template:
    """One real modified region of a round's training folds."""

    techniques: tuple[Technique, ...]  # without "other", never empty
    entity: Entity
    characters: int  # its width in character widths of the OCR words it overlaps, at least 1


@dataclass(frozen=True, slots=True)
class RoundStatistics:
    """The real forgeries of one round's training folds, as the engine draws from them."""

    held_out: int  # the fold this round holds out
    templates: tuple[Template, ...]
    regions_per_receipt: tuple[int, ...]  # one count of modified regions per forged receipt
    skipped: dict[str, int]  # modified regions that gave no template, by reason
    widths_measured_on: dict[str, int]  # templates by where their character width was measured


def training_documents(
    documents: Iterable[Document], folds: FoldAssignment, held_out: int
) -> list[Document]:
    """The training documents of a round: those in the training split outside its held-out fold.

    Raises:
        ValueError: If the fold does not exist.
    """
    if not 0 <= held_out < folds.n_folds:
        msg = f"there is no fold {held_out}"
        raise ValueError(msg)
    return [
        d
        for d in documents
        if d.split is Split.TRAIN and d.doc_id in folds.folds and folds.folds[d.doc_id] != held_out
    ]


def character_width(ocr: OcrResult, region: BBox) -> tuple[float, str] | None:
    """The median width of one character near a rectangle, and where it was measured.

    Measured on the OCR words the rectangle overlaps; where it overlaps none, as where a deletion
    removed the text, on the words of its printed row, and failing that on the whole receipt.
    """
    printed = [(t, len(t.text.replace(" ", ""))) for t in ocr.tokens if t.text.strip()]
    for place, chosen in (
        ("words", [(t, n) for t, n in printed if t.box.overlaps(region)]),
        ("row", [(t, n) for t, n in printed if _same_row(t.box, region)]),
        ("receipt", printed),
    ):
        if chosen:
            return float(np.median([t.box.width / n for t, n in chosen])), place
    return None


def _same_row(box: BBox, region: BBox) -> bool:
    """Whether two boxes overlap vertically by at least half of the smaller height."""
    overlap = min(box.y2, region.y2) - max(box.y, region.y)
    return overlap >= ROW_OVERLAP * min(box.height, region.height)


def round_statistics(
    documents: Sequence[Document], folds: FoldAssignment, held_out: int, ocr: OcrEngine
) -> RoundStatistics:
    """The templates and region counts of the forged receipts in a round's training folds.

    Only the annotations of training documents outside the held-out fold are read.
    """
    templates: list[Template] = []
    counts: list[int] = []
    skipped: Counter[str] = Counter()
    measured_on: Counter[str] = Counter()
    for document in sorted(training_documents(documents, folds, held_out), key=lambda d: d.doc_id):
        if document.label is not Label.FORGED:
            continue
        modified = [r for r in document.regions if r.role is RegionRole.MODIFIED]
        if not modified:
            skipped["forged receipt without regions"] += 1
            continue
        counts.append(len(modified))
        page_ocr: OcrResult | None = None
        for region in modified:
            techniques = tuple(sorted(region.techniques - {Technique.OTHER}))
            if not techniques:
                skipped["technique other only" if region.techniques else "no technique"] += 1
                continue
            if region.entity is None:
                skipped["no entity"] += 1
                continue
            if page_ocr is None:
                page_ocr = ocr.recognize(load_grayscale(document.image_path))
            measured = character_width(page_ocr, region.box)
            if measured is None:
                skipped["no OCR word on the receipt"] += 1
                continue
            width, place = measured
            measured_on[place] += 1
            templates.append(
                Template(techniques, region.entity, max(1, round(region.box.width / width)))
            )
    return RoundStatistics(
        held_out, tuple(templates), tuple(counts), dict(skipped), dict(measured_on)
    )
