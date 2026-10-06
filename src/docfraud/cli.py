"""Command-line interface for docfraud."""

import json
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
from docfraud.data.gold import draw_gold_set, load_annotations, load_manifest, save_manifest
from docfraud.data.gold_review import (
    ReviewResult,
    draw_review_sample,
    image_url,
    render_review_page,
    render_review_report,
    score_review,
)
from docfraud.data.models import Label, Split
from docfraud.data.normalize import load_grayscale
from docfraud.data.splits import select
from docfraud.evaluation.extraction import render_extraction_report, score_document
from docfraud.reading.consistency import CHECKS
from docfraud.reading.ocr import CachedOcr, RapidOcrEngine
from docfraud.reading.pipeline import TextSource, read_document

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


GOLD_MANIFEST = Path("annotations/findit2/gold-fields/manifest.json")


@data_app.command("draw-gold")
def data_draw_gold(
    data_dir: DataDir = Path("data"),
    output: Annotated[Path, typer.Option(help="Where to write the manifest.")] = GOLD_MANIFEST,
) -> None:
    """Draw the gold set for field extraction from the training split (seeded, stratified)."""
    documents = load_findit2(data_dir / "raw" / SOURCES[DatasetName.FINDIT2.value].name)
    manifest = draw_gold_set(documents)
    save_manifest(manifest, output)
    console.print(
        f"[green]written[/green] {output}: {len(manifest.authentic)} authentic and "
        f"{len(manifest.forged)} forged training receipts (seed {manifest.seed})"
    )


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


@reading_app.command("check-gold")
def reading_check_gold(
    manifest_path: Annotated[Path, typer.Option("--manifest", help="Gold manifest.")] = (
        GOLD_MANIFEST
    ),
) -> None:
    """Validate gold annotations and list arithmetic that fails on authentic receipts.

    A failing relation on an authentic receipt usually means an annotation typo; it is resolved
    by looking at the image again, never by editing a value until it balances.
    """
    manifest = load_manifest(manifest_path)
    annotations = load_annotations(manifest_path.parent, manifest)
    missing = sorted(manifest.doc_ids - set(annotations))
    console.print(f"{len(annotations)} of {len(manifest.doc_ids)} gold receipts annotated")
    for doc_id, annotation in annotations.items():
        if doc_id not in manifest.authentic:
            continue
        failed = [name for name, check in CHECKS.items() if check(annotation.fields) is False]
        if failed:
            console.print(f"[yellow]re-check[/yellow] {doc_id}: {', '.join(failed)}")
    if missing:
        console.print(f"missing: {', '.join(missing)}")


GOLD_REVIEW = Path("annotations/findit2/gold-review.json")
GOLD_RESOLUTIONS = Path("annotations/findit2/gold-review-resolutions.json")


@reading_app.command("review-gold")
def reading_review_gold(
    data_dir: DataDir = Path("data"),
    manifest_path: Annotated[Path, typer.Option("--manifest", help="Gold manifest.")] = (
        GOLD_MANIFEST
    ),
    output: Annotated[Path, typer.Option(help="Directory for the review page.")] = Path(
        "artifacts/gold-review"
    ),
) -> None:
    """Write a local page for the owner to verify a seeded sample of gold annotations."""
    manifest = load_manifest(manifest_path)
    annotations = load_annotations(manifest_path.parent, manifest)
    sample = draw_review_sample(manifest)
    missing = sorted(set(sample) - set(annotations))
    if missing:
        console.print(f"[red]not annotated yet:[/red] {', '.join(missing)}")
        raise typer.Exit(1)
    documents = load_findit2(data_dir / "raw" / SOURCES[DatasetName.FINDIT2.value].name)
    images = {d.doc_id: d.image_path for d in documents if d.doc_id in sample}
    output.mkdir(parents=True, exist_ok=True)
    urls = {doc_id: image_url(output, path) for doc_id, path in images.items()}
    page = output / "index.html"
    page.write_text(render_review_page(sample, annotations, urls), encoding="utf-8")
    console.print(f"[green]written[/green] {page} ({len(sample)} receipts)")


@reading_app.command("score-review")
def reading_score_review(
    result_path: Annotated[Path, typer.Argument(help="JSON downloaded from the review page.")],
    manifest_path: Annotated[Path, typer.Option("--manifest", help="Gold manifest.")] = (
        GOLD_MANIFEST
    ),
    record: Annotated[Path, typer.Option(help="Where to keep the review.")] = GOLD_REVIEW,
    resolutions: Annotated[
        Path, typer.Option(help="JSON mapping each finding to its resolution, if any.")
    ] = GOLD_RESOLUTIONS,
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = Path(
        "reports/reading/gold-verification.md"
    ),
) -> None:
    """Score a completed review, keep it with the annotations and write the report."""
    manifest = load_manifest(manifest_path)
    annotations = load_annotations(manifest_path.parent, manifest)
    result = ReviewResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    scores = score_review(result, annotations, draw_review_sample(manifest))
    if result_path.resolve() != record.resolve():
        record.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, str] = (
        json.loads(resolutions.read_text(encoding="utf-8")) if resolutions.is_file() else {}
    )
    output.write_text(render_review_report(result, scores, resolved), encoding="utf-8")
    accepted = sum(s.accepted for s in scores)
    total = sum(s.units + s.missed for s in scores)
    console.print(f"[green]written[/green] {output}: {accepted} of {total} accepted")


TRACKS = {TextSource.TRANSCRIPT: "T1 transcript", TextSource.OCR: "T2 OCR"}


@reading_app.command("evaluate-gold")
def reading_evaluate_gold(
    data_dir: DataDir = Path("data"),
    manifest_path: Annotated[Path, typer.Option("--manifest", help="Gold manifest.")] = (
        GOLD_MANIFEST
    ),
    cache: Annotated[Path, typer.Option(help="OCR cache directory.")] = OCR_CACHE,
    predictions: Annotated[
        Path, typer.Option(help="Where to keep the parser's output per receipt.")
    ] = Path("artifacts/gold-evaluation"),
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = Path(
        "reports/reading/gold-evaluation.md"
    ),
) -> None:
    """Evaluate field extraction on the gold set for both text sources (tracks T1 and T2)."""
    manifest = load_manifest(manifest_path)
    annotations = load_annotations(manifest_path.parent, manifest)
    missing = sorted(manifest.doc_ids - set(annotations))
    if missing:
        console.print(f"[red]not annotated yet:[/red] {', '.join(missing)}")
        raise typer.Exit(1)
    documents = load_findit2(data_dir / "raw" / SOURCES[DatasetName.FINDIT2.value].name)
    gold_documents = sorted(
        (d for d in documents if d.doc_id in manifest.doc_ids), key=lambda d: d.doc_id
    )
    ocr = CachedOcr(RapidOcrEngine(), cache)
    scores = {}
    for source, track in TRACKS.items():
        (predictions / source.value).mkdir(parents=True, exist_ok=True)
        track_scores = []
        for document in gold_documents:
            fields = read_document(document, ocr, source).fields
            (predictions / source.value / f"{document.doc_id}.json").write_text(
                fields.model_dump_json(indent=2) + "\n", encoding="utf-8"
            )
            track_scores.append(
                score_document(document.doc_id, annotations[document.doc_id].fields, fields)
            )
        scores[track] = track_scores
    header = (
        "# Field extraction on the gold set\n\n"
        "Generated by `docfraud reading evaluate-gold`; do not edit by hand.\n\n"
        f"The rule-based parser, frozen before annotation, read the {len(gold_documents)} gold "
        f"receipts ({len(manifest.authentic)} authentic, {len(manifest.forged)} forged) from the "
        "dataset transcripts (T1) and from our OCR (T2). Metrics: "
        "docs/annotation-guidelines.md, section 7. OCR engine: "
        f"`{ocr.identifier}`.\n\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_extraction_report(scores, frozenset(manifest.forged), header), encoding="utf-8"
    )
    console.print(f"[green]written[/green] {output}")
