"""The engine of one cross-validation round, built from the dataset, and its development set."""

from collections.abc import Iterator, Sequence
from functools import lru_cache
from pathlib import Path

from docfraud.data.models import Document
from docfraud.data.normalize import load_grayscale
from docfraud.evaluation.design_fold import DESIGN_FOLD
from docfraud.evaluation.folds import FoldAssignment
from docfraud.reading.ocr import OcrEngine
from docfraud.synthesis.engine import (
    Engine,
    Forgery,
    Receipt,
    ReceiptSource,
    Stream,
    directory_fonts,
)
from docfraud.synthesis.layout import words
from docfraud.synthesis.statistics import round_statistics

RECEIPT_CACHE = 32
"""Receipts kept loaded: bases and the donors of copies from another receipt."""


def receipt_source(documents: Sequence[Document], ocr: OcrEngine) -> ReceiptSource:
    """Receipts as the engine sees them, loaded on demand from the dataset."""
    by_id = {d.doc_id: d for d in documents}

    @lru_cache(maxsize=RECEIPT_CACHE)
    def receipt(doc_id: str) -> Receipt:
        page = load_grayscale(by_id[doc_id].image_path)
        return Receipt(doc_id, page, words(page, ocr.recognize(page)))

    return receipt


def round_engine(
    documents: Sequence[Document],
    folds: FoldAssignment,
    held_out: int,
    ocr: OcrEngine,
    fonts: Path,
) -> Engine:
    """The engine of the round that holds out one fold."""
    statistics = round_statistics(documents, folds, held_out, ocr)
    return Engine(
        statistics, documents, folds, receipt_source(documents, ocr), directory_fonts(fonts)
    )


def development_set(engine: Engine) -> list[Forgery]:
    """One forgery of each authentic design-fold receipt (ADR-0021, decision 6).

    Raises:
        ValueError: If the engine is not of the round that holds out the design fold.
    """
    return list(development_forgeries(engine))


def development_forgeries(engine: Engine) -> Iterator[Forgery]:
    """The forgeries of :func:`development_set`, made one at a time, in the order of their bases.

    Raises:
        ValueError: If the engine is not of the round that holds out the design fold.
    """
    if engine.held_out != DESIGN_FOLD:
        msg = f"the development set belongs to round {DESIGN_FOLD}, not {engine.held_out}"
        raise ValueError(msg)
    return (engine.forge(base, 0, Stream.DEVELOPMENT) for base in engine.pools[Stream.DEVELOPMENT])
