"""Command-line interface for docfraud."""

from enum import StrEnum
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

from docfraud import __version__
from docfraud.data.audit import audit_dataset
from docfraud.data.download import SOURCES, download, safe_extract, verify
from docfraud.data.duplicates import resolve_duplicates
from docfraud.data.findit import DUPLICATE_GROUPS, load_findit2
from docfraud.data.models import Label, Split
from docfraud.data.normalize import load_grayscale
from docfraud.data.splits import select
from docfraud.reading.ocr import CachedOcr, RapidOcrEngine

app = typer.Typer(
    name="docfraud",
    help="Explainable multimodal detection of forged financial documents.",
    no_args_is_help=True,
    add_completion=False,
)
data_app = typer.Typer(help="Download, prepare and audit datasets.", no_args_is_help=True)
app.add_typer(data_app, name="data")
reading_app = typer.Typer(help="OCR and field extraction.", no_args_is_help=True)
app.add_typer(reading_app, name="reading")

console = Console()


class DatasetName(StrEnum):
    """Datasets known to the command line."""

    FINDIT2 = "findit2"


DataDir = Annotated[Path, typer.Option(help="Root directory for datasets.")]
DatasetArg = Annotated[DatasetName, typer.Argument(help="Dataset to operate on.")]


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"docfraud {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Detect forged financial documents and explain why."""


@data_app.command("download")
def data_download(
    dataset: DatasetArg = DatasetName.FINDIT2, data_dir: DataDir = Path("data")
) -> None:
    """Download a dataset archive and verify its SHA-256 checksum."""
    source = SOURCES[dataset.value]
    columns = (BarColumn(), DownloadColumn(), TransferSpeedColumn())
    with Progress(*columns, console=console, transient=True) as progress:
        task = progress.add_task(source.name, total=source.size_bytes)
        path = download(
            source,
            data_dir / "raw",
            on_progress=lambda done, total: progress.update(task, completed=done, total=total),
        )
    console.print(f"[green]verified[/green] {path} ({source.size_bytes:,} bytes)")


@data_app.command("prepare")
def data_prepare(
    dataset: DatasetArg = DatasetName.FINDIT2, data_dir: DataDir = Path("data")
) -> None:
    """Verify a downloaded archive, extract it and validate its contents."""
    source = SOURCES[dataset.value]
    raw = data_dir / "raw"
    archive = raw / source.filename
    verify(source, archive)
    files = safe_extract(archive, raw)
    documents = load_findit2(raw / source.name)
    forged = sum(d.label is Label.FORGED for d in documents)
    console.print(
        f"[green]prepared[/green] {len(files)} files, {len(documents)} documents "
        f"({forged} forged) in {raw / source.name}"
    )


@data_app.command("audit")
def data_audit(
    dataset: DatasetArg = DatasetName.FINDIT2,
    data_dir: DataDir = Path("data"),
    output: Annotated[Path, typer.Option(help="Where to write the Markdown audit.")] = Path(
        "reports/data/findit2-audit.md"
    ),
) -> None:
    """Audit composition, corrections, integrity and shortcuts; write a Markdown report."""
    source = SOURCES[dataset.value]
    raw = load_findit2(data_dir / "raw" / source.name, duplicate_groups=())
    documents, dropped = resolve_duplicates(raw, DUPLICATE_GROUPS)
    with console.status("inspecting documents"):
        audit_dataset(documents, output, source.citation, DUPLICATE_GROUPS, dropped)
    console.print(f"[green]written[/green] {output}")


OCR_CACHE = Path("artifacts/ocr")


@reading_app.command("ocr")
def reading_ocr(
    split: Annotated[
        list[Split], typer.Option(help="Splits to read; the test split stays locked.")
    ] = [Split.TRAIN, Split.VAL],  # noqa: B006 - Typer needs a literal default
    data_dir: DataDir = Path("data"),
    cache: Annotated[Path, typer.Option(help="OCR cache directory.")] = OCR_CACHE,
) -> None:
    """Run OCR on every page of the given splits and cache the results."""
    documents = load_findit2(data_dir / "raw" / SOURCES[DatasetName.FINDIT2.value].name)
    pages = [d for s in split for d in select(documents, s)]
    ocr = CachedOcr(RapidOcrEngine(), cache)
    with Progress(*Progress.get_default_columns(), console=console) as progress:
        task = progress.add_task(ocr.identifier, total=len(pages))
        for document in pages:
            ocr.recognize(load_grayscale(document.image_path))
            progress.advance(task)
    console.print(f"[green]cached[/green] {len(pages)} pages in {ocr.directory}")
