"""Paths, options and helpers shared by the command groups.

Loading the dataset, building the OCR engine and checking for a clean commit each happen in one
place here, so that every command does them the same way and tests replace them in one place.
"""

from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console

from docfraud.data.download import SOURCES
from docfraud.data.findit import load_findit2
from docfraud.data.models import Document
from docfraud.evaluation.runs import code_version
from docfraud.reading.ocr import CachedOcr, RapidOcrEngine
from docfraud.reading.pipeline import TextSource

console = Console()

GOLD_MANIFEST = Path("annotations/findit2/gold-fields/manifest.json")
GOLD_B_MANIFEST = Path("annotations/findit2/gold-fields-b/manifest.json")
CV_FOLDS_PATH = Path("annotations/findit2/cv-folds.json")
OCR_CACHE = Path("artifacts/ocr")
REPORTED_RUNS = Path("reports/runs")
TRACKS = {TextSource.TRANSCRIPT: "T1 transcript", TextSource.OCR: "T2 OCR"}


class DatasetName(StrEnum):
    """Datasets known to the command line."""

    FINDIT2 = "findit2"


DataDir = Annotated[Path, typer.Option(help="Root directory for datasets.")]
DatasetArg = Annotated[DatasetName, typer.Argument(help="Dataset to operate on.")]


def findit_documents(data_dir: Path, **options: Any) -> tuple[Document, ...]:
    """The prepared Find-it-again documents (options go to ``load_findit2``)."""
    return load_findit2(data_dir / "raw" / SOURCES[DatasetName.FINDIT2.value].name, **options)


def cached_ocr(cache: Path) -> CachedOcr:
    """The default OCR engine behind the page cache."""
    return CachedOcr(RapidOcrEngine(), cache)


def require_clean_commit() -> None:
    """Stop unless the working tree matches a commit, so a one-time run names its code.

    Raises:
        typer.Exit: If there are uncommitted changes or git is unavailable.
    """
    if code_version().dirty is not False:
        console.print("[red]commit all changes first[/red]: this run must name a clean commit")
        raise typer.Exit(1)
