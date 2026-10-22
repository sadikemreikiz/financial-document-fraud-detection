"""Reading pipeline: from a document to rows and structured fields.

Both text sources share the layout recovered by OCR, so that tracks T1 and T2 of the evaluation
protocol differ only in where the text comes from.
"""

from dataclasses import dataclass
from enum import StrEnum

from docfraud.data.models import Document
from docfraud.data.normalize import load_grayscale, read_transcript
from docfraud.reading.align import align_transcript
from docfraud.reading.fields import FieldSources, ReceiptFields
from docfraud.reading.layout import Row, Segment, group_rows, union
from docfraud.reading.ocr import OcrEngine, OcrResult
from docfraud.reading.parser import extract_fields


class TextSource(StrEnum):
    """Where the text of a page comes from."""

    TRANSCRIPT = "transcript"  # track T1: dataset transcript, positioned by alignment
    OCR = "ocr"  # track T2: our own OCR


@dataclass(frozen=True, slots=True)
class Reading:
    """What was read from one document, and where on the page each field value is printed."""

    source: TextSource
    ocr: OcrResult
    rows: tuple[Row, ...]
    fields: ReceiptFields
    sources: FieldSources


def ocr_segments(ocr: OcrResult) -> list[Segment]:
    """One positioned segment per OCR text line."""
    by_line: dict[int, list[int]] = {}
    for index, token in enumerate(ocr.tokens):
        by_line.setdefault(token.line, []).append(index)
    return [
        Segment(
            " ".join(ocr.tokens[i].text for i in members),
            union([ocr.tokens[i].box for i in members]),
        )
        for _, members in sorted(by_line.items())
    ]


def read_document(document: Document, ocr_engine: OcrEngine, source: TextSource) -> Reading:
    """Read a document from the requested text source.

    Raises:
        ValueError: If the transcript source is requested for a document without transcript.
    """
    ocr = ocr_engine.recognize(load_grayscale(document.image_path))
    if source is TextSource.OCR:
        segments = ocr_segments(ocr)
    else:
        if document.transcript_path is None:
            msg = f"{document.uid} has no transcript"
            raise ValueError(msg)
        lines = read_transcript(document.transcript_path).splitlines()
        segments = align_transcript([line for line in lines if line.strip()], ocr)
    rows = tuple(group_rows(segments))
    extraction = extract_fields(rows)
    return Reading(
        source=source, ocr=ocr, rows=rows, fields=extraction.fields, sources=extraction.sources
    )
