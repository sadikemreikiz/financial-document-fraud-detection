"""Paths, options and helpers shared by the command groups.

Loading the dataset, building the OCR engine and checking for a clean commit or for unchanged
code each happen in one place here, so that every command does them the same way and tests
replace them in one place.
"""

from collections.abc import Sequence
from enum import StrEnum
from pathlib import Path
from typing import Annotated, Any

import typer
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

from docfraud.data.download import SOURCES, PinnedFile, download
from docfraud.data.findit import load_findit2
from docfraud.data.models import Document
from docfraud.evaluation.runs import changed_since, code_version
from docfraud.reading.ocr import CachedOcr, RapidOcrEngine
from docfraud.reading.pipeline import TextSource

console = Console()

GOLD_MANIFEST = Path("annotations/findit2/gold-fields/manifest.json")
GOLD_B_MANIFEST = Path("annotations/findit2/gold-fields-b/manifest.json")
CV_FOLDS_PATH = Path("annotations/findit2/cv-folds.json")
OCR_CACHE = Path("artifacts/ocr")
REPORTED_RUNS = Path("reports/runs")
TRACKS = {TextSource.TRANSCRIPT: "T1 transcript", TextSource.OCR: "T2 OCR"}
FREEZE_TAG = "phase-7-freeze"
"""The git tag the configuration is frozen at before the test split is unlocked (ADR-0034)."""
TEST_UNLOCK = "ADR-0034: the single evaluation of Phase 7"
"""The reason every command of Phase 7 gives when it unlocks the test split (ADR-0034)."""


class DatasetName(StrEnum):
    """Datasets known to the command line."""

    FINDIT2 = "findit2"


DataDir = Annotated[Path, typer.Option(help="Root directory for datasets.")]
DatasetArg = Annotated[DatasetName, typer.Argument(help="Dataset to operate on.")]


def findit_documents(data_dir: Path, **options: Any) -> tuple[Document, ...]:
    """The prepared Find-it-again documents (options go to ``load_findit2``)."""
    return load_findit2(data_dir / "raw" / SOURCES[DatasetName.FINDIT2.value].name, **options)


def download_with_progress(source: PinnedFile, directory: Path) -> Path:
    """Download and verify an archive into ``directory``, showing a progress bar."""
    columns = (BarColumn(), DownloadColumn(), TransferSpeedColumn())
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task(source.name, total=source.size_bytes)
        path = download(
            source,
            directory,
            on_progress=lambda done, total: progress.update(task, completed=done, total=total),
        )
    console.print(f"[green]verified[/green] {path} ({source.size_bytes:,} bytes)")
    return path


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


def require_unchanged(commit: str, paths: Sequence[str], problem: str) -> None:
    """Stop if the committed files under ``paths`` differ from ``commit``.

    Raises:
        typer.Exit: If they changed, or git cannot compare with the commit.
    """
    changed = changed_since(commit, paths)
    if changed is None:
        console.print(f"[red]cannot compare with commit {commit[:12]}[/red]")
        raise typer.Exit(1)
    if changed:
        console.print(f"[red]{problem}[/red] at {commit[:12]}: " + ", ".join(changed))
        raise typer.Exit(1)


def require_freeze_tag() -> None:
    """Stop unless the committed source is the source tagged :data:`FREEZE_TAG` (ADR-0034, 1).

    Raises:
        typer.Exit: If the tag does not exist or the source changed after it.
    """
    changed = changed_since(FREEZE_TAG, ("src",))
    if changed is None:
        console.print(
            f"[red]no tag {FREEZE_TAG}[/red]: freeze the configuration first (ADR-0034, decision 1)"
        )
        raise typer.Exit(1)
    if changed:
        console.print(f"[red]the source changed after {FREEZE_TAG}[/red]: " + ", ".join(changed))
        raise typer.Exit(1)
