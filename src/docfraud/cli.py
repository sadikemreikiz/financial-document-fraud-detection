"""Command-line interface for docfraud."""

import json
from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from enum import StrEnum
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import typer
from rich.console import Console
from rich.progress import BarColumn, DownloadColumn, Progress, TransferSpeedColumn

from docfraud import __version__
from docfraud.baselines import bow
from docfraud.baselines.report import render_report
from docfraud.data.audit import audit_dataset
from docfraud.data.download import SOURCES, download, safe_extract, verify
from docfraud.data.duplicates import resolve_duplicates
from docfraud.data.findit import DUPLICATE_GROUPS, load_findit2
from docfraud.data.gold import (
    GOLD_AUTHENTIC,
    GOLD_FORGED,
    GOLD_SEED,
    GoldAnnotation,
    GoldManifest,
    development_documents,
    draw_gold_set,
    load_annotations,
    load_manifest,
    save_manifest,
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
from docfraud.data.models import Document, Label, Split
from docfraud.data.normalize import load_grayscale
from docfraud.data.splits import select
from docfraud.detectors.evaluation import cross_validated, held_out, rule_counts
from docfraud.detectors.report import render_rules_report
from docfraud.detectors.rules import check
from docfraud.detectors.scoring import any_fail
from docfraud.evaluation.detection import compare, summary
from docfraud.evaluation.extraction import render_extraction_report, score_document
from docfraud.evaluation.folds import (
    CV_FOLDS,
    FOLD_SEED,
    FoldAssignment,
    assign_folds,
    check_folds,
    load_folds,
    save_folds,
)
from docfraud.evaluation.runs import JsonRunStore, Run, code_version, run
from docfraud.reading.consistency import CHECKS
from docfraud.reading.fields import ReceiptFields
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
baseline_app = typer.Typer(help="Published baselines, re-implemented.", no_args_is_help=True)
app.add_typer(baseline_app, name="baseline")
rules_app = typer.Typer(help="Accounting rules.", no_args_is_help=True)
app.add_typer(rules_app, name="rules")

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
GOLD_B_MANIFEST = Path("annotations/findit2/gold-fields-b/manifest.json")


@data_app.command("draw-gold")
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
    documents = load_findit2(data_dir / "raw" / SOURCES[DatasetName.FINDIT2.value].name)
    if exclude:
        documents = development_documents(documents, *map(load_manifest, exclude))
    manifest = draw_gold_set(documents, n_authentic=authentic, n_forged=forged, seed=seed)
    save_manifest(manifest, output)
    console.print(
        f"[green]written[/green] {output}: {len(manifest.authentic)} authentic and "
        f"{len(manifest.forged)} forged training receipts (seed {manifest.seed})"
    )


CV_FOLDS_PATH = Path("annotations/findit2/cv-folds.json")


@data_app.command("draw-folds")
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
    documents = load_findit2(data_dir / "raw" / SOURCES[DatasetName.FINDIT2.value].name)
    assignment = assign_folds(documents, n_folds=folds, seed=seed)
    save_folds(assignment, output)
    sizes = ", ".join(str(len(assignment.documents(f))) for f in range(assignment.n_folds))
    console.print(f"[green]written[/green] {output}: fold sizes {sizes} (seed {assignment.seed})")


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
ReviewAuthentic = Annotated[int, typer.Option(help="Authentic receipts in the review sample.")]
ReviewForged = Annotated[int, typer.Option(help="Forged receipts in the review sample.")]
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
    documents = load_findit2(data_dir / "raw" / SOURCES[DatasetName.FINDIT2.value].name)
    images = {d.doc_id: d.image_path for d in documents if d.doc_id in sample}
    output.mkdir(parents=True, exist_ok=True)
    urls = {doc_id: image_url(output, path) for doc_id, path in images.items()}
    page = output / "index.html"
    page.write_text(render_review_page(sample, annotations, urls), encoding="utf-8")
    console.print(f"[green]written[/green] {page} ({len(sample)} receipts)")


@reading_app.command("score-review")
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


TRACKS = {TextSource.TRANSCRIPT: "T1 transcript", TextSource.OCR: "T2 OCR"}


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


@reading_app.command("evaluate-gold")
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


@reading_app.command("compare-gold")
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


REPORTED_RUNS = Path("reports/runs")
BOW_CV_RUN = "bow-lr-cv"
BOW_TEST_RUN = "bow-lr-test"
BOW_TEST_REASON = "baseline reproduction, evaluation protocol section 3, exception 1 (ADR-0011)"
BOW_CONFIG = {
    "baseline": "bag of words + logistic regression (Martinez Tornes et al., 2023)",
    "decision": "ADR-0011",
    "text": "T1, normalised dataset transcripts",
    "features": "sklearn CountVectorizer(), default settings",
    "classifier": "sklearn LogisticRegression(), default settings",
    "threshold": bow.THRESHOLD,
    "training": "training split",
}


@baseline_app.command("bow-lr")
def baseline_bow_lr(
    data_dir: DataDir = Path("data"),
    runs: Annotated[Path, typer.Option(help="Where reported run records are kept.")] = (
        REPORTED_RUNS
    ),
    folds_path: Annotated[Path, typer.Option("--folds", help="Stored folds.")] = CV_FOLDS_PATH,
    unlock_test: Annotated[
        bool,
        typer.Option(
            "--unlock-test",
            help="Evaluate once on the test split (protocol exception 1) instead of in "
            "cross-validation.",
        ),
    ] = False,
) -> None:
    """Run the published BoW + LR baseline: cross-validated, or once on the test split."""
    source = SOURCES[DatasetName.FINDIT2.value]
    documents = load_findit2(data_dir / "raw" / source.name)
    store = JsonRunStore(runs)
    recorded = partial(
        run, store=store, seeds={"bootstrap": 0}, dataset_sha256={source.name: source.sha256}
    )
    if not unlock_test:
        folds = load_folds(folds_path)
        check_folds(folds, documents)
        with recorded(BOW_CV_RUN, config=BOW_CONFIG | {"folds": str(folds_path)}) as current:
            cv = bow.cross_validate(documents, folds)
            current.log("max_iter", list(cv.max_iter))
            for key, value in summary(cv.labels, cv.scores, bow.THRESHOLD, cv.doc_ids).items():
                current.log(key, value)
        f1 = current.results["metrics"]["f1"]["value"]
        console.print(f"[green]recorded[/green] {BOW_CV_RUN}: cross-validated F1 {f1:.3f}")
        return
    if store.records(BOW_TEST_RUN):
        console.print(f"[red]refusing to run again:[/red] a {BOW_TEST_RUN} record exists in {runs}")
        raise typer.Exit(1)
    if code_version().dirty is not False:
        console.print("[red]commit all changes first[/red]: the test run must name a clean commit")
        raise typer.Exit(1)
    with recorded(BOW_TEST_RUN, config=BOW_CONFIG) as current:
        train = select(documents, Split.TRAIN)
        test = select(documents, Split.TEST, unlock_test=True, reason=BOW_TEST_REASON)
        fitted = bow.fit(train)
        current.log("max_iter", fitted.max_iter)
        doc_ids = [d.doc_id for d in test]
        for key, value in summary(
            bow.labels(test), fitted.scores(test), bow.THRESHOLD, doc_ids
        ).items():
            current.log(key, value)
        current.log("published", bow.PUBLISHED)
        f1 = current.results["metrics"]["f1"]
        current.log("reproduced", bool(f1["low"] <= bow.PUBLISHED["f1"] <= f1["high"]))
    verdict = "passed" if current.results["reproduced"] else "failed"
    console.print(f"[green]recorded[/green] {BOW_TEST_RUN}: reproduction {verdict}")


@baseline_app.command("report")
def baseline_report(
    runs: Annotated[Path, typer.Option(help="Where reported run records are kept.")] = (
        REPORTED_RUNS
    ),
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = Path(
        "reports/baselines/bow-lr.md"
    ),
) -> None:
    """Render the BoW + LR report from the latest cross-validation run and the test run."""
    store = JsonRunStore(runs)
    cv = store.records(BOW_CV_RUN)
    if not cv:
        console.print(f"[red]no {BOW_CV_RUN} run in {runs}[/red]")
        raise typer.Exit(1)
    test = store.records(BOW_TEST_RUN)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(cv[-1], test[-1] if test else None), encoding="utf-8")
    console.print(f"[green]written[/green] {output}")


RULES_CV_RUN = "rules-cv"
RULES_VALIDATION_RUN = "rules-validation"
RULES_CONFIG = {
    "decision": "ADR-0012",
    "rules": "docfraud.detectors.rules.RULES",
    "scorer": "sklearn LogisticRegression(), default settings, over fail / not-applicable flags",
    "threshold": "maximises F1 on out-of-fold training scores, ties to the higher threshold",
    "reader": "parser v2 (frozen)",
}
COMPARED = (
    "f1",
    "average_precision",
    "roc_auc",
    "recall_at_10pct_review",
    "recall_at_20pct_review",
)


def _versus_bow(
    y: object, bow_scores: object, rule_scores: object, threshold: float
) -> dict[str, Any]:
    differences = compare(y, bow_scores, rule_scores, bow.THRESHOLD, threshold)
    return {
        name: {"value": d.value, "low": d.low, "high": d.high, "excludes_zero": d.excludes_zero}
        for name, d in differences.items()
        if name in COMPARED
    }


@rules_app.command("evaluate")
def rules_evaluate(  # noqa: PLR0913, PLR0917 - one option per input
    data_dir: DataDir = Path("data"),
    runs: Annotated[Path, typer.Option(help="Where reported run records are kept.")] = (
        REPORTED_RUNS
    ),
    folds_path: Annotated[Path, typer.Option("--folds", help="Stored folds.")] = CV_FOLDS_PATH,
    cache: Annotated[Path, typer.Option(help="OCR cache directory.")] = OCR_CACHE,
    gold: Annotated[
        list[Path] | None, typer.Option(help="Gold manifests for the correct-reading level.")
    ] = None,
    validation: Annotated[
        bool,
        typer.Option(
            "--validation",
            help="Measure once on the validation split (ADR-0012) instead of in cross-validation.",
        ),
    ] = False,
) -> None:
    """Measure the accounting rules: cross-validated on training, or once on validation."""
    source = SOURCES[DatasetName.FINDIT2.value]
    documents = load_findit2(data_dir / "raw" / source.name)
    folds = load_folds(folds_path)
    check_folds(folds, documents)
    store = JsonRunStore(runs)
    setup = _RulesSetup(
        documents=documents,
        train=sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id),
        folds=folds,
        ocr=CachedOcr(RapidOcrEngine(), cache),
        store=store,
        recorded=partial(
            run, store=store, seeds={"bootstrap": 0}, dataset_sha256={source.name: source.sha256}
        ),
    )
    if validation:
        _rules_validation(setup)
    else:
        _rules_cross_validation(setup, gold or [GOLD_MANIFEST, GOLD_B_MANIFEST])


@dataclass(frozen=True)
class _RulesSetup:
    documents: tuple[Document, ...]
    train: list[Document]
    folds: FoldAssignment
    ocr: CachedOcr
    store: JsonRunStore
    recorded: Callable[..., AbstractContextManager[Run]]

    def outcomes(self, documents: Sequence[Document], source: TextSource) -> list[Any]:
        return [check(read_document(d, self.ocr, source).fields) for d in documents]

    def fold_of(self) -> npt.NDArray[np.int_]:
        return np.array([self.folds.folds[d.doc_id] for d in self.train])


def _rules_cross_validation(setup: _RulesSetup, gold: list[Path]) -> None:
    bow_runs = setup.store.records(BOW_CV_RUN)
    if not bow_runs:
        console.print(f"[red]run the baseline first:[/red] no {BOW_CV_RUN} record")
        raise typer.Exit(1)
    train, y = setup.train, bow.labels(setup.train)
    ids = [d.doc_id for d in train]
    bow_scores = [bow_runs[-1].results["scores"][i] for i in ids]
    with setup.recorded(RULES_CV_RUN, config=RULES_CONFIG) as current:
        for text_source in TRACKS:
            result = cross_validated(setup.outcomes(train, text_source), y, setup.fold_of(), ids)
            oof = [result["learned"]["scores"][i] for i in ids]
            result["versus_bow"] = _versus_bow(y, bow_scores, oof, result["threshold"])
            current.log(text_source.value, result)
        annotations: dict[str, Any] = {}
        for manifest_path in gold:
            annotations |= load_annotations(manifest_path.parent, load_manifest(manifest_path))
        forged = {d.doc_id: d.label is Label.FORGED for d in train}
        gold_ids = sorted(annotations)
        gold_outcomes = [check(annotations[i].fields) for i in gold_ids]
        gold_y = np.array([forged[i] for i in gold_ids], dtype=bool)
        current.log(
            "gold_fields",
            {
                "any_fail": summary(gold_y, any_fail(gold_outcomes), 1.0, gold_ids),
                "rules": rule_counts(gold_outcomes, gold_y),
            },
        )
    console.print(f"[green]recorded[/green] {RULES_CV_RUN}")


def _rules_validation(setup: _RulesSetup) -> None:
    if setup.store.records(RULES_VALIDATION_RUN):
        console.print(f"[red]refusing to run again:[/red] a {RULES_VALIDATION_RUN} record exists")
        raise typer.Exit(1)
    if code_version().dirty is not False:
        console.print("[red]commit all changes first[/red]: the run must name a clean commit")
        raise typer.Exit(1)
    held = sorted(select(setup.documents, Split.VAL), key=lambda d: d.doc_id)
    y, y_train = bow.labels(held), bow.labels(setup.train)
    ids = [d.doc_id for d in held]
    with setup.recorded(RULES_VALIDATION_RUN, config=RULES_CONFIG) as current:
        held_bow = bow.fit(setup.train).scores(held)
        current.log("bow", summary(y, held_bow, bow.THRESHOLD, ids))
        for text_source in TRACKS:
            result = held_out(
                setup.outcomes(setup.train, text_source),
                y_train,
                setup.fold_of(),
                setup.outcomes(held, text_source),
                y,
                ids,
            )
            scores = [result["learned"]["scores"][i] for i in ids]
            result["versus_bow"] = _versus_bow(y, held_bow, scores, result["threshold"])
            current.log(text_source.value, result)
    console.print(f"[green]recorded[/green] {RULES_VALIDATION_RUN}")


@rules_app.command("report")
def rules_report(
    runs: Annotated[Path, typer.Option(help="Where reported run records are kept.")] = (
        REPORTED_RUNS
    ),
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = Path(
        "reports/rules/accounting-rules.md"
    ),
) -> None:
    """Render the accounting-rules report from the latest cross-validation and validation runs."""
    store = JsonRunStore(runs)
    cv, bow_cv = store.records(RULES_CV_RUN), store.records(BOW_CV_RUN)
    if not cv or not bow_cv:
        console.print(f"[red]need {RULES_CV_RUN} and {BOW_CV_RUN} runs in {runs}[/red]")
        raise typer.Exit(1)
    validation = store.records(RULES_VALIDATION_RUN)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_rules_report(cv[-1], bow_cv[-1], validation[-1] if validation else None),
        encoding="utf-8",
    )
    console.print(f"[green]written[/green] {output}")
