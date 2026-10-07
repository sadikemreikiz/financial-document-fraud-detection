"""`docfraud data`: download, prepare and audit datasets."""

from pathlib import Path
from typing import Annotated

import typer
from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

from docfraud.cli.common import (
    CV_FOLDS_PATH,
    GOLD_MANIFEST,
    DataDir,
    DatasetArg,
    DatasetName,
    console,
    findit_documents,
)
from docfraud.data.audit import audit_dataset
from docfraud.data.download import SOURCES, download, safe_extract, verify
from docfraud.data.duplicates import resolve_duplicates
from docfraud.data.findit import DUPLICATE_GROUPS
from docfraud.data.gold import (
    GOLD_AUTHENTIC,
    GOLD_FORGED,
    GOLD_SEED,
    development_documents,
    draw_gold_set,
    load_manifest,
    save_manifest,
)
from docfraud.data.models import Label
from docfraud.evaluation.folds import (
    CV_FOLDS,
    FOLD_SEED,
    assign_folds,
    save_folds,
)

app = typer.Typer(help="Download, prepare and audit datasets.", no_args_is_help=True)


@app.command("download")
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


@app.command("prepare")
def data_prepare(
    dataset: DatasetArg = DatasetName.FINDIT2, data_dir: DataDir = Path("data")
) -> None:
    """Verify a downloaded archive, extract it and validate its contents."""
    source = SOURCES[dataset.value]
    raw = data_dir / "raw"
    archive = raw / source.filename
    verify(source, archive)
    files = safe_extract(archive, raw)
    documents = findit_documents(data_dir)
    forged = sum(d.label is Label.FORGED for d in documents)
    console.print(
        f"[green]prepared[/green] {len(files)} files, {len(documents)} documents "
        f"({forged} forged) in {raw / source.name}"
    )


@app.command("audit")
def data_audit(
    dataset: DatasetArg = DatasetName.FINDIT2,
    data_dir: DataDir = Path("data"),
    output: Annotated[Path, typer.Option(help="Where to write the Markdown audit.")] = Path(
        "reports/data/findit2-audit.md"
    ),
) -> None:
    """Audit composition, corrections, integrity and shortcuts; write a Markdown report."""
    source = SOURCES[dataset.value]
    raw = findit_documents(data_dir, duplicate_groups=())
    documents, dropped = resolve_duplicates(raw, DUPLICATE_GROUPS)
    with console.status("inspecting documents"):
        audit_dataset(documents, output, source.citation, DUPLICATE_GROUPS, dropped)
    console.print(f"[green]written[/green] {output}")


@app.command("draw-gold")
def data_draw_gold(  # noqa: PLR0913, PLR0917 - one option per draw parameter
    data_dir: DataDir = Path("data"),
    output: Annotated[Path, typer.Option(help="Where to write the manifest.")] = GOLD_MANIFEST,
    exclude: Annotated[
        list[Path] | None, typer.Option(help="Manifests of earlier gold sets to draw around.")
    ] = None,
    authentic: Annotated[int, typer.Option(help="Authentic receipts to draw.")] = GOLD_AUTHENTIC,
    forged: Annotated[int, typer.Option(help="Forged receipts to draw.")] = GOLD_FORGED,
    seed: Annotated[int, typer.Option(help="Random seed of the draw.")] = GOLD_SEED,
) -> None:
    """Draw a gold set for field extraction from the training split (seeded, stratified).

    A gold set is drawn once: an existing manifest is never overwritten. Later gold sets are
    drawn from the training receipts outside every earlier one (``--exclude``).
    """
    if output.exists():
        console.print(f"[red]refusing to overwrite[/red] {output}")
        raise typer.Exit(1)
    documents = findit_documents(data_dir)
    if exclude:
        documents = development_documents(documents, *map(load_manifest, exclude))
    manifest = draw_gold_set(documents, n_authentic=authentic, n_forged=forged, seed=seed)
    save_manifest(manifest, output)
    console.print(
        f"[green]written[/green] {output}: {len(manifest.authentic)} authentic and "
        f"{len(manifest.forged)} forged training receipts (seed {manifest.seed})"
    )


@app.command("draw-folds")
def data_draw_folds(
    data_dir: DataDir = Path("data"),
    output: Annotated[Path, typer.Option(help="Where to write the folds.")] = CV_FOLDS_PATH,
    folds: Annotated[int, typer.Option(help="Number of folds.")] = CV_FOLDS,
    seed: Annotated[int, typer.Option(help="Random seed of the draw.")] = FOLD_SEED,
) -> None:
    """Draw the cross-validation folds of the training split (seeded, stratified, grouped).

    Folds are drawn once: an existing file is never overwritten, so every experiment is measured
    on the same partition.
    """
    if output.exists():
        console.print(f"[red]refusing to overwrite[/red] {output}")
        raise typer.Exit(1)
    documents = findit_documents(data_dir)
    assignment = assign_folds(documents, n_folds=folds, seed=seed)
    save_folds(assignment, output)
    sizes = ", ".join(str(len(assignment.documents(f))) for f in range(assignment.n_folds))
    console.print(f"[green]written[/green] {output}: fold sizes {sizes} (seed {assignment.seed})")
