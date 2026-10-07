"""`docfraud rules`: accounting rules."""

from collections.abc import Callable, Sequence
from contextlib import AbstractContextManager
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import numpy.typing as npt
import typer

from docfraud.baselines import bow
from docfraud.cli.baseline import BOW_CV_RUN
from docfraud.cli.common import (
    CV_FOLDS_PATH,
    GOLD_B_MANIFEST,
    GOLD_MANIFEST,
    OCR_CACHE,
    REPORTED_RUNS,
    TRACKS,
    DataDir,
    DatasetName,
    cached_ocr,
    console,
    findit_documents,
    require_clean_commit,
)
from docfraud.data.download import SOURCES
from docfraud.data.gold import (
    load_annotations,
    load_manifest,
)
from docfraud.data.models import Document, Label, Split
from docfraud.data.splits import select
from docfraud.detectors.evaluation import cross_validated, held_out, rule_counts
from docfraud.detectors.report import render_rules_report
from docfraud.detectors.rules import check
from docfraud.detectors.scoring import any_fail
from docfraud.evaluation.breakdowns import breakdown, flagged_by_budget, flagged_by_threshold
from docfraud.evaluation.detection import compare, summary
from docfraud.evaluation.folds import (
    FoldAssignment,
    check_folds,
    load_folds,
)
from docfraud.evaluation.runs import JsonRunStore, Run, run
from docfraud.reading.ocr import CachedOcr
from docfraud.reading.pipeline import TextSource, read_document

app = typer.Typer(help="Accounting rules.", no_args_is_help=True)


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


@app.command("evaluate")
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
    documents = findit_documents(data_dir)
    folds = load_folds(folds_path)
    check_folds(folds, documents)
    store = JsonRunStore(runs)
    setup = _RulesSetup(
        documents=documents,
        train=sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id),
        folds=folds,
        ocr=cached_ocr(cache),
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
    require_clean_commit()
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


RULES_BREAKDOWN_RUN = "rules-breakdown"
REVIEW_BUDGET = 0.10


@app.command("breakdown")
def rules_breakdown(
    data_dir: DataDir = Path("data"),
    runs: Annotated[Path, typer.Option(help="Where reported run records are kept.")] = (
        REPORTED_RUNS
    ),
) -> None:
    """Group the recorded rule decisions by technique, entity, software and annotation.

    Nothing is measured again: the scores come from the latest cross-validation and the
    validation run records, grouped with the dataset's annotations (evaluation protocol, section 7).
    """
    source = SOURCES[DatasetName.FINDIT2.value]
    by_id = {d.doc_id: d for d in findit_documents(data_dir)}
    store = JsonRunStore(runs)
    scopes = {
        "training": store.records(RULES_CV_RUN),
        "validation": store.records(RULES_VALIDATION_RUN),
    }
    recorded = {scope: records[-1] for scope, records in scopes.items() if records}
    if "training" not in recorded:
        console.print(f"[red]no {RULES_CV_RUN} run in {runs}[/red]")
        raise typer.Exit(1)
    with run(
        RULES_BREAKDOWN_RUN,
        store=store,
        config={
            "from_runs": {s: r.run_id for s, r in recorded.items()},
            "review_budget": REVIEW_BUDGET,
        },
        seeds={},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        for scope, record in recorded.items():
            result: dict[str, Any] = {}
            for text_source in TRACKS:
                track = record.results[text_source.value]
                documents = [by_id[i] for i in track["learned"]["scores"]]
                result[text_source.value] = {
                    "any_fail": breakdown(
                        documents, flagged_by_threshold(track["any_fail"]["scores"], 1.0)
                    ),
                    "learned_at_budget": breakdown(
                        documents, flagged_by_budget(track["learned"]["scores"], REVIEW_BUDGET)
                    ),
                }
            current.log(scope, result)
    console.print(f"[green]recorded[/green] {RULES_BREAKDOWN_RUN}: {', '.join(recorded)}")


@app.command("report")
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
    breakdowns = store.records(RULES_BREAKDOWN_RUN)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_rules_report(
            cv[-1],
            bow_cv[-1],
            validation[-1] if validation else None,
            breakdowns[-1] if breakdowns else None,
        ),
        encoding="utf-8",
    )
    console.print(f"[green]written[/green] {output}")
