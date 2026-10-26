"""`docfraud copy-move`: the copy-move detector's single measurement on the clean folds."""

from pathlib import Path
from typing import Annotated

import typer

from docfraud.cli.common import (
    CV_FOLDS_PATH,
    OCR_CACHE,
    REPORTED_RUNS,
    DataDir,
    DatasetName,
    cached_ocr,
    console,
    findit_documents,
    require_clean_commit,
)
from docfraud.data.download import SOURCES
from docfraud.data.models import Split
from docfraud.data.normalize import load_grayscale
from docfraud.data.splits import select
from docfraud.detectors.copy_move_measurement import (
    CONFIG,
    RUN_NAME,
    measure,
    render_report,
    results,
)
from docfraud.evaluation.design_fold import CLEAN_FOLDS
from docfraud.evaluation.folds import check_folds, load_folds
from docfraud.evaluation.runs import JsonRunStore, run

app = typer.Typer(help="Copy-move inside a receipt.", no_args_is_help=True)

REPORT = Path("reports/fusion/copy-move-clean.md")
Runs = Annotated[Path, typer.Option(help="Where reported run records are kept.")]
Output = Annotated[Path, typer.Option(help="Where to write the report.")]


@app.command("measure")
def copy_move_measure(
    data_dir: DataDir = Path("data"),
    runs: Runs = REPORTED_RUNS,
    folds_path: Annotated[Path, typer.Option("--folds", help="Stored folds.")] = CV_FOLDS_PATH,
    cache: Annotated[Path, typer.Option(help="OCR cache directory.")] = OCR_CACHE,
    output: Output = REPORT,
) -> None:
    """Measure the frozen detector once on the clean folds (ADR-0029, section 4; ADR-0030).

    Refuses to run again, and runs only from a clean commit, which the record names.
    """
    store = JsonRunStore(runs)
    if store.records(RUN_NAME):
        console.print(f"[red]refusing to run again:[/red] a {RUN_NAME} record exists")
        raise typer.Exit(1)
    require_clean_commit()
    documents = findit_documents(data_dir)
    folds = load_folds(folds_path)
    check_folds(folds, documents)
    clean = sorted(
        (d for d in select(documents, Split.TRAIN) if folds.folds[d.doc_id] in CLEAN_FOLDS),
        key=lambda d: d.doc_id,
    )
    ocr = cached_ocr(cache)
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        RUN_NAME,
        store=store,
        config=CONFIG,
        seeds={"bootstrap": 0, "baseline": 0},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        measured = []
        for document in clean:
            page = load_grayscale(document.image_path)
            words = [token.box for token in ocr.recognize(page).tokens]
            measured.append(measure(document, page, words))
        for key, value in results(measured).items():
            current.log(key, value)
    record = store.records(RUN_NAME)[-1]
    _write(render_report(record.results, record.code.commit), output)


@app.command("report")
def copy_move_report(runs: Runs = REPORTED_RUNS, output: Output = REPORT) -> None:
    """Render the report of the clean measurement from its run record."""
    records = JsonRunStore(runs).records(RUN_NAME)
    if not records:
        console.print(f"[red]no {RUN_NAME} record in {runs}[/red]")
        raise typer.Exit(1)
    _write(render_report(records[-1].results, records[-1].code.commit), output)


def _write(text: str, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(text, encoding="utf-8")
    console.print(f"[green]written[/green] {output}")
