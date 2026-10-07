"""`docfraud baseline`: published baselines, re-implemented."""

from functools import partial
from pathlib import Path
from typing import Annotated

import typer

from docfraud.baselines import bow
from docfraud.baselines.report import render_report
from docfraud.cli.common import (
    CV_FOLDS_PATH,
    REPORTED_RUNS,
    DataDir,
    DatasetName,
    console,
    findit_documents,
    require_clean_commit,
)
from docfraud.data.download import SOURCES
from docfraud.data.models import Split
from docfraud.data.splits import select
from docfraud.evaluation.detection import summary
from docfraud.evaluation.folds import (
    check_folds,
    load_folds,
)
from docfraud.evaluation.runs import JsonRunStore, run

app = typer.Typer(help="Published baselines, re-implemented.", no_args_is_help=True)


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


@app.command("bow-lr")
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
    documents = findit_documents(data_dir)
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
    require_clean_commit()
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


@app.command("report")
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
