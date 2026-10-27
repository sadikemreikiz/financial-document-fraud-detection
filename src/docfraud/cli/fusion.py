"""`docfraud fusion`: the fusion of the findings (ADR-0028, ADR-0029)."""

import hashlib
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated

import numpy as np
import typer
from rich.progress import track as with_progress

from docfraud.cli.common import (
    CV_FOLDS_PATH,
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
    require_unchanged,
)
from docfraud.cli.forensics import ROUNDS_DIR
from docfraud.cli.llm import FROZEN, STORE
from docfraud.data.download import SOURCES
from docfraud.data.gold import load_manifest
from docfraud.data.models import BBox, Document, Label, Split
from docfraud.data.normalize import load_grayscale
from docfraud.data.splits import select
from docfraud.detectors.finding import Finding
from docfraud.detectors.metadata import editing_traits
from docfraud.detectors.rules import Outcome, check
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS, WHOLE_TRAINING
from docfraud.evaluation.folds import check_folds, load_folds
from docfraud.evaluation.runs import JsonRunStore, RunRecord, changed_since, run
from docfraud.forensics.learned.rounds import SEEDS, run_directory
from docfraud.forensics.learned.sampler import System
from docfraud.forensics.learned.training import MODEL
from docfraud.fusion import validation_report
from docfraud.fusion.development import (
    CATCHERS,
    PRACTICAL_VARIANTS,
    false_alarms,
    first_regions,
    fold_models,
    forged_groups,
    group_contributions,
    in_budget,
    metrics,
    rq3,
    rq4,
    score,
    track_results,
)
from docfraud.fusion.development_report import render_report
from docfraud.fusion.evidence import SystemTwo, page_findings
from docfraud.fusion.training_inputs import (
    RUN_NAME,
    load_table,
    save_table,
    stored_system_2,
    summary,
    training_inputs,
)
from docfraud.fusion.validation import RUN_NAME as CHECK_RUN
from docfraud.fusion.validation import validation_check
from docfraud.fusion.validation_inputs import INPUT_CODE, rules_scorers, system_two
from docfraud.fusion.validation_inputs import RUN_NAME as VALIDATION_INPUTS_RUN
from docfraud.fusion.validation_inputs import validation_inputs as validation_table
from docfraud.llm.frozen import FrozenReader
from docfraud.llm.record import FrozenVersion
from docfraud.llm.run import target
from docfraud.llm.store import ResponseStore
from docfraud.reading.fields import FieldSources
from docfraud.reading.ocr import CachedOcr
from docfraud.reading.pipeline import TextSource, read_document

app = typer.Typer(help="Fusion of the findings.", no_args_is_help=True)

TRAINING_INPUTS = Path("artifacts/fusion/training-inputs.json")
VALIDATION_INPUTS = Path("artifacts/fusion/validation-inputs.json")
VALIDATION_REPORT = Path("reports/fusion/validation-check.md")
Runs = Annotated[Path, typer.Option(help="Where reported run records are kept.")]
Output = Annotated[Path, typer.Option(help="Where to write the report.")]

type Reading = tuple[dict[str, Outcome], FieldSources]


def _frozen_reader() -> FrozenReader:  # pragma: no cover - needs the stored LLM answers
    """The frozen LLM reader, over the stored answers."""
    version = FrozenVersion.model_validate_json(FROZEN.read_text(encoding="utf-8"))
    instruction = Path(version.instruction).read_text(encoding="utf-8")
    return FrozenReader(version, instruction, ResponseStore(STORE))


def _words(ocr: CachedOcr, image: np.ndarray) -> list[BBox]:  # pragma: no cover
    """The OCR word boxes of a page."""
    return [token.box for token in ocr.recognize(image).tokens]


def _reading(  # pragma: no cover - needs the stored LLM answers
    document: Document, ocr: CachedOcr, reader: FrozenReader, text_source: TextSource
) -> Reading:
    """The rule outcomes on the frozen reader's fields of one text source, and their sources."""
    rows = read_document(document, ocr, text_source).rows
    extraction = reader.read(target(document.doc_id, text_source, rows)).extraction
    return check(extraction.fields), extraction.sources


@app.command("training-inputs")
def training_inputs_command(  # pragma: no cover - needs the dataset, LLM answers and rounds
    data_dir: DataDir = Path("data"),
    runs: Annotated[Path, typer.Option(help="Where reported run records are kept.")] = (
        REPORTED_RUNS
    ),
    rounds: Annotated[Path, typer.Option(help="Where the Phase 4 rounds are kept.")] = ROUNDS_DIR,
    output: Annotated[Path, typer.Option(help="Where to write the table.")] = TRAINING_INPUTS,
) -> None:
    """Compute the fusion inputs of every training receipt, out of fold (ADR-0029, 14, step 3).

    Per text source and system 2 seed: the rules' score from the Phase 3 scorer fitted on the
    other folds over the frozen LLM reader's fields, system 2's stored output from the round
    that held the receipt out, system 1, copy-move, the file's editing traits and the links. The
    table goes to ``output``; the run record keeps its aggregates.
    """
    documents = findit_documents(data_dir)
    folds = load_folds(CV_FOLDS_PATH)
    check_folds(folds, documents)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    ocr = cached_ocr(OCR_CACHE)
    reader = _frozen_reader()
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        RUN_NAME,
        store=JsonRunStore(runs),
        config={"decision": "ADR-0029", "frozen_reader": str(FROZEN), "rounds": str(rounds)},
        seeds={f"system 2, seed {s}": s for s in SEEDS},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        system_2 = {seed: stored_system_2(rounds, seed) for seed in SEEDS}
        page = []
        readings: dict[str, list[Reading]] = {s.value: [] for s in TRACKS}
        for document in train:
            image = load_grayscale(document.image_path)
            page.append(
                page_findings(image, _words(ocr, image), editing_traits(document.image_path))
            )
            for text_source in TRACKS:
                readings[text_source.value].append(_reading(document, ocr, reader, text_source))
        table = training_inputs(
            [d.doc_id for d in train],
            [d.label is Label.FORGED for d in train],
            [folds.folds[d.doc_id] for d in train],
            readings,
            page=page,
            system_2=system_2,
        )
        save_table(table, output)
        current.log("summary", summary(table))
    console.print(f"[green]written[/green] {output}: {len(table.receipts)} rows")


TRAINING_REPORT = Path("reports/fusion/training-oof.md")
REPORT_RUN = "fusion-training-oof"


@app.command("training-oof")
def training_oof_command(  # pragma: no cover - needs the dataset and the inputs table
    data_dir: DataDir = Path("data"),
    runs: Annotated[Path, typer.Option(help="Where reported run records are kept.")] = (
        REPORTED_RUNS
    ),
    inputs: Annotated[Path, typer.Option(help="The training inputs table.")] = TRAINING_INPUTS,
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = TRAINING_REPORT,
) -> None:
    """Phase 5 on the training split, out of fold: the development reference (ADR-0029, 1, 7-12).

    The ablation table, the comparisons of section 8, the seed spread, the overfitting check,
    the receipts shaped in development left out, track T3, RQ3 and RQ4. Only aggregates go to
    the run record and the report.
    """
    table = load_table(inputs)
    documents = findit_documents(data_dir)
    folds = load_folds(CV_FOLDS_PATH)
    check_folds(folds, documents)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    receipts = [d.doc_id for d in train]
    y = np.array([d.label is Label.FORGED for d in train])
    fold_of = np.array([folds.folds[i] for i in receipts])
    gold_a = load_manifest(GOLD_MANIFEST).doc_ids
    subsets = {
        "without the design fold": np.array([i not in DESIGN_FOLD_IDS for i in receipts]),
        "without gold set A": np.array([i not in gold_a for i in receipts]),
    }
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        REPORT_RUN,
        store=JsonRunStore(runs),
        config={"decision": "ADR-0029", "inputs": str(inputs)},
        seeds={"bootstrap": 0, "baseline": 0},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        scored = {}
        for track in ("transcript", "ocr"):
            rows_by_seed = {seed: table.rows(track, seed) for seed in SEEDS}
            if [r.receipt for r in rows_by_seed[0]] != receipts:
                console.print(f"[red]the table's {track} receipts are not the training split[/red]")
                raise typer.Exit(1)
            results, scored[track] = track_results(rows_by_seed, y, fold_of, subsets=subsets)
            current.log(track, results)
        ocr_rows = table.rows("ocr", 0)
        current.log(
            "T3",
            {"fusion of track T2": metrics(y, scored["ocr"]["fusion"].probabilities)}
            | {
                v.name: metrics(y, score(ocr_rows, v, y, fold_of).probabilities)
                for v in PRACTICAL_VARIANTS
            },
        )
        groups = forged_groups(train)
        probabilities = {
            track: {name: scored[track][name].probabilities for name in CATCHERS}
            for track in ("transcript", "ocr")
        }
        models = {track: fold_models(scored[track]["fusion"], fold_of) for track in scored}
        current.log(
            "rq3",
            {
                track: rq3(
                    group_contributions(table.rows(track, 0), models[track]),
                    probabilities[track],
                    receipts,
                    groups,
                )
                for track in ("transcript", "ocr")
            },
        )
        current.log(
            "false_alarms",
            {track: false_alarms(train, probabilities[track]) for track in ("transcript", "ocr")},
        )
        pages = _pages(train)
        current.log(
            "rq4",
            {
                track: rq4(
                    train,
                    first_regions(table.rows(track, 0), models[track]),
                    in_budget(receipts, scored[track]["fusion"].probabilities),
                    pages,
                )
                for track in ("transcript", "ocr")
            },
        )
    training_report_command(runs, output)


@app.command("training-report")
def training_report_command(
    runs: Annotated[Path, typer.Option(help="Where reported run records are kept.")] = (
        REPORTED_RUNS
    ),
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = TRAINING_REPORT,
) -> None:
    """Render the training-split report of Phase 5 from its latest run record."""
    records = JsonRunStore(runs).records(REPORT_RUN)
    if not records:
        console.print(f"[red]no {REPORT_RUN} record in {runs}[/red]")
        raise typer.Exit(1)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(records[-1].results, records[-1].code.commit), encoding="utf-8")
    console.print(f"[green]written[/green] {output}")


def _pages(  # pragma: no cover - needs the dataset and the OCR cache
    documents: Sequence[Document],
) -> dict[str, tuple[BBox, list[BBox]]]:
    """The page and OCR word boxes of every forged receipt with regions, for RQ4's baseline."""
    ocr = cached_ocr(OCR_CACHE)
    pages = {}
    for document in documents:
        if document.label is Label.FORGED and document.modified_regions:
            image = load_grayscale(document.image_path)
            page = BBox(x=0, y=0, width=image.shape[1], height=image.shape[0])
            pages[document.doc_id] = (page, _words(ocr, image))
    return pages


def unlabelled(document: Document) -> Document:
    """The receipt without its label, its annotations and the dataset's notes on it.

    The validation inputs are computed from such copies, so that nothing that reads a receipt
    there can see whether it is forged: the editing software is recorded for forged receipts
    only, and the marks are annotations too.
    """
    return document.model_copy(
        update={
            "label": None,
            "regions": (),
            "has_region_annotations": False,
            "digital_annotation": False,
            "handwritten_annotation": False,
            "editing_software": None,
        }
    )


def _latest(store: JsonRunStore, name: str) -> RunRecord:
    """The latest record of a run, or stop if there is none."""
    records = store.records(name)
    if not records:
        console.print(f"[red]no {name} record in {store.root}[/red]")
        raise typer.Exit(1)
    return records[-1]


def _refuse_after_the_check(store: JsonRunStore) -> None:
    """Stop if the validation check has been made: it is made once (ADR-0029, section 1)."""
    if store.records(CHECK_RUN):
        console.print(f"[red]refusing to run again:[/red] a {CHECK_RUN} record exists")
        raise typer.Exit(1)


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.command("validation-inputs")
def validation_inputs_command(  # pragma: no cover - needs the dataset, LLM answers and models
    data_dir: DataDir = Path("data"),
    runs: Runs = REPORTED_RUNS,
    rounds: Annotated[Path, typer.Option(help="Where the rounds are kept.")] = ROUNDS_DIR,
    output: Annotated[Path, typer.Option(help="Where to write the table.")] = VALIDATION_INPUTS,
) -> None:
    """Compute the fusion inputs of every validation receipt (ADR-0029, sections 2, 7 (d), 14).

    With the components trained on the whole training split: system 2 of the round that holds
    out no fold, seeds 0 to 2, and the rules' scorer fitted on every training receipt; system 1,
    copy-move, the file's editing traits and the links as on the training split. The validation
    receipts are read without their labels and annotations (:func:`unlabelled`). Runs only from
    a clean commit, only if the code that makes the inputs is the code that made the training
    inputs, and never after the validation check. The table goes to ``output``; the run record
    names the models and keeps aggregates only.
    """
    store = JsonRunStore(runs)
    _refuse_after_the_check(store)
    require_clean_commit()
    training_run = _latest(store, RUN_NAME)
    if training_run.code.commit is None or training_run.code.dirty is not False:
        console.print(f"[red]{training_run.run_id} was not made from a clean commit[/red]")
        raise typer.Exit(1)
    require_unchanged(
        training_run.code.commit,
        INPUT_CODE,
        "the code that makes the inputs changed since the training inputs were made",
    )
    from docfraud.forensics.learned.inference import LearnedModel  # noqa: PLC0415 - PyTorch
    from docfraud.forensics.learned.rounds import WHOLE_RUNS, require_trained  # noqa: PLC0415

    try:
        require_trained(rounds, (WHOLE_TRAINING,), WHOLE_RUNS)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    documents = findit_documents(data_dir)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    validation = sorted(map(unlabelled, select(documents, Split.VAL)), key=lambda d: d.doc_id)
    ocr = cached_ocr(OCR_CACHE)
    reader = _frozen_reader()
    paths = {s: run_directory(rounds, WHOLE_TRAINING, System.SYNTHETIC, s) / MODEL for s in SEEDS}
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        VALIDATION_INPUTS_RUN,
        store=store,
        config={
            "decision": "ADR-0029",
            "frozen_reader": str(FROZEN),
            "training_inputs": training_run.run_id,
            "models": {str(s): {"path": str(p), "sha256": _sha256(p)} for s, p in paths.items()},
        },
        seeds={f"system 2, seed {s}": s for s in SEEDS},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        outcomes = {t.value: [_reading(d, ocr, reader, t)[0] for d in train] for t in TRACKS}
        scorers = rules_scorers(outcomes, [d.label is Label.FORGED for d in train])
        models = {seed: LearnedModel.from_file(path) for seed, path in paths.items()}
        page: list[tuple[Finding, ...]] = []
        system_2: dict[int, dict[str, SystemTwo]] = {seed: {} for seed in SEEDS}
        readings: dict[str, list[Reading]] = {t.value: [] for t in TRACKS}
        for document in with_progress(
            validation, description="validation receipts", console=console
        ):
            image = load_grayscale(document.image_path)
            words = _words(ocr, image)
            page.append(page_findings(image, words, editing_traits(document.image_path)))
            for seed, model in models.items():
                system_2[seed][document.doc_id] = system_two(model.analyse(image), words)
            for text_source in TRACKS:
                readings[text_source.value].append(_reading(document, ocr, reader, text_source))
        table = validation_table(
            [d.doc_id for d in validation],
            readings,
            page=page,
            system_2=system_2,
            scorers=scorers,
        )
        save_table(table, output)
        current.log("summary", summary(table))
        current.log("sha256", _sha256(output))
        current.log("rules_scorers", {t: scorer.weights() for t, scorer in scorers.items()})
        current.log("models", {str(seed): m.identifier for seed, m in models.items()})
        current.log(
            "changed_since_training_inputs", changed_since(training_run.code.commit, ["src"])
        )
    console.print(f"[green]written[/green] {output}: {len(table.receipts)} rows")


@app.command("validation-check")
def validation_check_command(
    data_dir: DataDir = Path("data"),
    runs: Runs = REPORTED_RUNS,
    training: Annotated[Path, typer.Option(help="The training inputs table.")] = TRAINING_INPUTS,
    validation: Annotated[
        Path, typer.Option(help="The validation inputs table.")
    ] = VALIDATION_INPUTS,
    output: Output = VALIDATION_REPORT,
) -> None:
    """Check the frozen system once on the validation split (ADR-0029, sections 1, 8 to 13).

    Refuses to run again. Runs only from a clean commit whose code is that of the validation
    inputs, on the two tables their run records describe. Only aggregates go to the run record
    and the report, with what Phase 7 applies to the test split: every row's model, Platt
    scaling and max-F1 threshold, and the review policy's thresholds.
    """
    store = JsonRunStore(runs)
    _refuse_after_the_check(store)
    require_clean_commit()
    inputs_run = _latest(store, VALIDATION_INPUTS_RUN)
    if inputs_run.code.commit is None or inputs_run.code.dirty is not False:
        console.print(f"[red]{inputs_run.run_id} was not made from a clean commit[/red]")
        raise typer.Exit(1)
    require_unchanged(
        inputs_run.code.commit, ("src",), "the code changed since the validation inputs were made"
    )
    training_run = store.load(inputs_run.config["training_inputs"])
    training_table = load_table(training)
    if summary(training_table) != training_run.results["summary"]:
        console.print(f"[red]{training} is not the table {training_run.run_id} describes[/red]")
        raise typer.Exit(1)
    if _sha256(validation) != inputs_run.results["sha256"]:
        console.print(f"[red]{validation} is not the table {inputs_run.run_id} wrote[/red]")
        raise typer.Exit(1)
    validation_inputs = load_table(validation)
    documents = findit_documents(data_dir)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    checked = sorted(select(documents, Split.VAL), key=lambda d: d.doc_id)
    if [r.receipt for r in training_table.rows("transcript", SEEDS[0])] != [
        d.doc_id for d in train
    ]:
        console.print("[red]the training table's receipts are not the training split[/red]")
        raise typer.Exit(1)
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        CHECK_RUN,
        store=store,
        config={
            "decision": "ADR-0029",
            "training_inputs": training_run.run_id,
            "validation_inputs": inputs_run.run_id,
            "validation_sha256": inputs_run.results["sha256"],
        },
        seeds={"bootstrap": 0, "baseline": 0},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        results = validation_check(
            training_table,
            np.array([d.label is Label.FORGED for d in train]),
            validation_inputs,
            checked,
            _pages(checked),
            seeds=SEEDS,
        )
        for key, value in results.items():
            current.log(key, value)
    validation_report_command(runs, output)


@app.command("validation-report")
def validation_report_command(
    runs: Runs = REPORTED_RUNS, output: Output = VALIDATION_REPORT
) -> None:
    """Render the report of the validation check from its run record."""
    record = _latest(JsonRunStore(runs), CHECK_RUN)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        validation_report.render_report(record.results, record.code.commit), encoding="utf-8"
    )
    console.print(f"[green]written[/green] {output}")
