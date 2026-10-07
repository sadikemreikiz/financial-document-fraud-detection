"""`docfraud reading`: oCR and field extraction."""

import json
from pathlib import Path
from typing import Annotated

import typer
from rich.progress import Progress

from docfraud.cli.common import (
    GOLD_MANIFEST,
    OCR_CACHE,
    TRACKS,
    DataDir,
    cached_ocr,
    console,
    findit_documents,
)
from docfraud.data.gold import (
    GoldAnnotation,
    GoldManifest,
    load_annotations,
    load_manifest,
)
from docfraud.data.gold_review import (
    REVIEW_AUTHENTIC,
    REVIEW_FORGED,
    ReviewResult,
    draw_review_sample,
    image_url,
    render_review_page,
    render_review_report,
    score_review,
)
from docfraud.data.models import Split
from docfraud.data.normalize import load_grayscale
from docfraud.data.splits import select
from docfraud.evaluation.extraction import render_extraction_report, score_document
from docfraud.reading.consistency import CHECKS
from docfraud.reading.fields import ReceiptFields
from docfraud.reading.pipeline import read_document

app = typer.Typer(help="OCR and field extraction.", no_args_is_help=True)


@app.command("ocr")
def reading_ocr(
    split: Annotated[
        list[Split], typer.Option(help="Splits to read; the test split stays locked.")
    ] = [Split.TRAIN, Split.VAL],  # noqa: B006 - Typer needs a literal default
    data_dir: DataDir = Path("data"),
    cache: Annotated[Path, typer.Option(help="OCR cache directory.")] = OCR_CACHE,
) -> None:
    """Run OCR on every page of the given splits and cache the results."""
    documents = findit_documents(data_dir)
    pages = [d for s in split for d in select(documents, s)]
    ocr = cached_ocr(cache)
    with Progress(*Progress.get_default_columns(), console=console) as progress:
        task = progress.add_task(ocr.identifier, total=len(pages))
        for document in pages:
            ocr.recognize(load_grayscale(document.image_path))
            progress.advance(task)
    console.print(f"[green]cached[/green] {len(pages)} pages in {ocr.directory}")


@app.command("check-gold")
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
ReviewAuthentic = Annotated[int, typer.Option(help="Authentic receipts in the review sample.")]
ReviewForged = Annotated[int, typer.Option(help="Forged receipts in the review sample.")]
GOLD_RESOLUTIONS = Path("annotations/findit2/gold-review-resolutions.json")


@app.command("review-gold")
def reading_review_gold(
    data_dir: DataDir = Path("data"),
    manifest_path: Annotated[Path, typer.Option("--manifest", help="Gold manifest.")] = (
        GOLD_MANIFEST
    ),
    output: Annotated[Path, typer.Option(help="Directory for the review page.")] = Path(
        "artifacts/gold-review"
    ),
    authentic: ReviewAuthentic = REVIEW_AUTHENTIC,
    forged: ReviewForged = REVIEW_FORGED,
) -> None:
    """Write a local page for the owner to verify a seeded sample of gold annotations."""
    manifest = load_manifest(manifest_path)
    annotations = load_annotations(manifest_path.parent, manifest)
    sample = draw_review_sample(manifest, n_authentic=authentic, n_forged=forged)
    missing = sorted(set(sample) - set(annotations))
    if missing:
        console.print(f"[red]not annotated yet:[/red] {', '.join(missing)}")
        raise typer.Exit(1)
    documents = findit_documents(data_dir)
    images = {d.doc_id: d.image_path for d in documents if d.doc_id in sample}
    output.mkdir(parents=True, exist_ok=True)
    urls = {doc_id: image_url(output, path) for doc_id, path in images.items()}
    page = output / "index.html"
    page.write_text(render_review_page(sample, annotations, urls), encoding="utf-8")
    console.print(f"[green]written[/green] {page} ({len(sample)} receipts)")


@app.command("score-review")
def reading_score_review(  # noqa: PLR0913, PLR0917 - one option per input
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
    authentic: ReviewAuthentic = REVIEW_AUTHENTIC,
    forged: ReviewForged = REVIEW_FORGED,
) -> None:
    """Score a completed review, keep it with the annotations and write the report."""
    manifest = load_manifest(manifest_path)
    annotations = load_annotations(manifest_path.parent, manifest)
    result = ReviewResult.model_validate_json(result_path.read_text(encoding="utf-8"))
    sample = draw_review_sample(manifest, n_authentic=authentic, n_forged=forged)
    scores = score_review(result, annotations, sample)
    if result_path.resolve() != record.resolve():
        record.write_text(
            json.dumps(result.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
        )
    output.parent.mkdir(parents=True, exist_ok=True)
    resolved: dict[str, str] = (
        json.loads(resolutions.read_text(encoding="utf-8")) if resolutions.is_file() else {}
    )
    report = render_review_report(result, scores, resolved, sizes=(authentic, forged))
    output.write_text(report, encoding="utf-8")
    accepted = sum(s.accepted for s in scores)
    total = sum(s.units + s.missed for s in scores)
    console.print(f"[green]written[/green] {output}: {accepted} of {total} accepted")


ReportTitle = Annotated[str, typer.Option(help="Title of the report.")]
ReportNote = Annotated[
    str, typer.Option(help="Paragraph stating which parser ran and what it has seen of the set.")
]


def _load_gold(manifest_path: Path) -> tuple[GoldManifest, dict[str, GoldAnnotation]]:
    manifest = load_manifest(manifest_path)
    annotations = load_annotations(manifest_path.parent, manifest)
    missing = sorted(manifest.doc_ids - set(annotations))
    if missing:
        console.print(f"[red]not annotated yet:[/red] {', '.join(missing)}")
        raise typer.Exit(1)
    return manifest, annotations


def _report_header(title: str, command: str, note: str) -> str:
    header = f"# {title}\n\nGenerated by `docfraud reading {command}`; do not edit by hand.\n\n"
    return header + (f"{note}\n\n" if note else "")


@app.command("evaluate-gold")
def reading_evaluate_gold(  # noqa: PLR0913, PLR0917 - one option per input and output
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
    title: ReportTitle = "Field extraction on the gold set",
    note: ReportNote = "",
) -> None:
    """Evaluate field extraction on the gold set for both text sources (tracks T1 and T2)."""
    manifest, annotations = _load_gold(manifest_path)
    documents = findit_documents(data_dir)
    gold_documents = sorted(
        (d for d in documents if d.doc_id in manifest.doc_ids), key=lambda d: d.doc_id
    )
    ocr = cached_ocr(cache)
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
    header = _report_header(title, "evaluate-gold", note) + (
        f"The rule-based parser read the {len(gold_documents)} gold receipts "
        f"({len(manifest.authentic)} authentic, {len(manifest.forged)} forged) from the "
        "dataset transcripts (T1) and from our OCR (T2). Metrics: "
        "docs/annotation-guidelines.md, section 7. OCR engine: "
        f"`{ocr.identifier}`.\n\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_extraction_report(scores, frozenset(manifest.forged), header), encoding="utf-8"
    )
    console.print(f"[green]written[/green] {output}")


@app.command("compare-gold")
def reading_compare_gold(  # noqa: PLR0913, PLR0917 - one option per input and output
    baseline: Annotated[
        Path, typer.Option(help="Predictions of the baseline, as kept by evaluate-gold.")
    ],
    candidate: Annotated[
        Path, typer.Option(help="Predictions of the candidate, as kept by evaluate-gold.")
    ],
    output: Annotated[Path, typer.Option(help="Where to write the report.")],
    manifest_path: Annotated[Path, typer.Option("--manifest", help="Gold manifest.")] = (
        GOLD_MANIFEST
    ),
    baseline_name: Annotated[str, typer.Option(help="Name of the baseline.")] = "baseline",
    candidate_name: Annotated[str, typer.Option(help="Name of the candidate.")] = "candidate",
    title: ReportTitle = "Comparison of two extractors on the gold set",
    note: ReportNote = "",
) -> None:
    """Compare two extractors' kept predictions on the same gold receipts, per text source."""
    if baseline_name == candidate_name:
        console.print("[red]the baseline and the candidate need different names[/red]")
        raise typer.Exit(1)
    manifest, annotations = _load_gold(manifest_path)
    doc_ids = sorted(manifest.doc_ids)
    sections = []
    for source, track in TRACKS.items():
        scores = {}
        for name, directory in ((baseline_name, baseline), (candidate_name, candidate)):
            paths = [directory / source.value / f"{doc_id}.json" for doc_id in doc_ids]
            absent = [str(p) for p in paths if not p.is_file()]
            if absent:
                console.print(f"[red]no prediction:[/red] {', '.join(absent)}")
                raise typer.Exit(1)
            scores[name] = [
                score_document(
                    doc_id,
                    annotations[doc_id].fields,
                    ReceiptFields.model_validate_json(path.read_text(encoding="utf-8")),
                )
                for doc_id, path in zip(doc_ids, paths, strict=True)
            ]
        sections.append(
            render_extraction_report(scores, frozenset(manifest.forged), f"## {track}\n\n")
        )
    header = _report_header(title, "compare-gold", note) + (
        f"The candidate ({candidate_name}) is compared with the baseline ({baseline_name}) on "
        f"the same {len(doc_ids)} gold "
        f"receipts ({len(manifest.authentic)} authentic, {len(manifest.forged)} forged), "
        "separately for the dataset transcripts (T1) and our OCR (T2). Metrics: "
        "docs/annotation-guidelines.md, section 7.\n\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(header + "\n".join(sections), encoding="utf-8")
    console.print(f"[green]written[/green] {output}")
