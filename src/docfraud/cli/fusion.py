"""`docfraud fusion`: the fusion of the findings (ADR-0028, ADR-0029)."""

import hashlib
import resource
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer
from rich.progress import track as with_progress

from docfraud.cli.common import (
    CV_FOLDS_PATH,
    GOLD_MANIFEST,
    OCR_CACHE,
    REPORTED_RUNS,
    TEST_UNLOCK,
    TRACKS,
    DataDir,
    DatasetName,
    cached_ocr,
    console,
    findit_documents,
    require_clean_commit,
    require_freeze_tag,
    require_unchanged,
)
from docfraud.cli.forensics import ROUNDS_DIR
from docfraud.cli.llm import FROZEN, RESAVED, RESAVED_READING_RUN, STORE, TEST_READING_RUN
from docfraud.data.download import SOURCES
from docfraud.data.gold import load_manifest
from docfraud.data.models import BBox, Document, Label, Split
from docfraud.data.normalize import load_grayscale
from docfraud.data.resave import QUALITIES, digest, resaved
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
from docfraud.fusion import shift_report, validation_report
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
from docfraud.fusion.rehearsal import (
    REHEARSAL_RUN,
    RESAVED_INPUTS_RUN,
    TEST_INPUTS_RUN,
    differences,
)
from docfraud.fusion.shift import RUN_NAME as SHIFT_RUN
from docfraud.fusion.shift import input_shift
from docfraud.fusion.training_inputs import (
    RUN_NAME,
    InputsTable,
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
REHEARSAL_INPUTS = Path("artifacts/fusion/rehearsal-inputs.json")
TEST_INPUTS = Path("artifacts/fusion/test-inputs.json")
RESAVED_INPUTS = Path("artifacts/fusion/resaved")
VALIDATION_REPORT = Path("reports/fusion/validation-check.md")
SHIFT_REPORT = Path("reports/fusion/input-shift.md")
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


def _whole_training_models(rounds: Path) -> dict[int, Path]:  # pragma: no cover - PyTorch
    """System 2's models trained on the whole training split, seeds 0 to 2, checked complete.

    Raises:
        typer.Exit: If a model is missing.
    """
    from docfraud.forensics.learned.rounds import WHOLE_RUNS, require_trained  # noqa: PLC0415

    try:
        require_trained(rounds, (WHOLE_TRAINING,), WHOLE_RUNS)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    return {s: run_directory(rounds, WHOLE_TRAINING, System.SYNTHETIC, s) / MODEL for s in SEEDS}


def _models_config(paths: dict[int, Path]) -> dict[str, dict[str, str]]:
    """The models of a run record: each seed's file and its SHA-256."""
    return {str(s): {"path": str(p), "sha256": _sha256(p)} for s, p in paths.items()}


@dataclass(frozen=True, slots=True)
class _Made:
    """One split's inputs, what made them, and the seconds each component took per receipt."""

    table: InputsTable
    scorers: dict[str, Any]
    models: dict[str, str]
    seconds: dict[str, list[float]]


def _split_inputs(
    train: Sequence[Document], receipts: Sequence[Document], paths: dict[int, Path], name: str
) -> _Made:
    """The fusion inputs of ``receipts`` with the components trained on the whole training split.

    The rules' scorer is fitted on the frozen reader's fields of every training receipt; system 1,
    copy-move, the file's editing traits and the links are made as on the training split.
    """
    from docfraud.forensics.learned.inference import LearnedModel  # noqa: PLC0415 - PyTorch

    ocr = cached_ocr(OCR_CACHE)
    reader = _frozen_reader()
    outcomes = {t.value: [_reading(d, ocr, reader, t)[0] for d in train] for t in TRACKS}
    scorers = rules_scorers(outcomes, [d.label is Label.FORGED for d in train])
    models = {seed: LearnedModel.from_file(path) for seed, path in paths.items()}
    page: list[tuple[Finding, ...]] = []
    system_2: dict[int, dict[str, SystemTwo]] = {seed: {} for seed in models}
    readings: dict[str, list[Reading]] = {t.value: [] for t in TRACKS}
    seconds: dict[str, list[float]] = {
        "page and OCR": [],
        "system 1, copy-move and file traits": [],
        "system 2, one seed": [],
        "reading both text sources": [],
    }
    for document in with_progress(receipts, description=name, console=console):
        start = time.perf_counter()
        image = load_grayscale(document.image_path)
        words = _words(ocr, image)
        seconds["page and OCR"].append(time.perf_counter() - start)
        start = time.perf_counter()
        page.append(page_findings(image, words, editing_traits(document.image_path)))
        seconds["system 1, copy-move and file traits"].append(time.perf_counter() - start)
        for seed, model in models.items():
            start = time.perf_counter()
            system_2[seed][document.doc_id] = system_two(model.analyse(image), words)
            seconds["system 2, one seed"].append(time.perf_counter() - start)
        start = time.perf_counter()
        for text_source in TRACKS:
            readings[text_source.value].append(_reading(document, ocr, reader, text_source))
        seconds["reading both text sources"].append(time.perf_counter() - start)
    table = validation_table(
        [d.doc_id for d in receipts], readings, page=page, system_2=system_2, scorers=scorers
    )
    return _Made(
        table,
        {t: scorer.weights() for t, scorer in scorers.items()},
        {str(seed): m.identifier for seed, m in models.items()},
        seconds,
    )


def _efficiency(seconds: dict[str, list[float]]) -> dict[str, Any]:
    """Each component's time per receipt, median and 95th percentile, and the peak memory.

    The memory is the process's peak resident size, in bytes on macOS, the reference machine.
    """
    return {
        "seconds": {
            name: {
                "receipts": len(found),
                "median": float(np.median(found)),
                "p95": float(np.percentile(found, 95)),
            }
            for name, found in seconds.items()
            if found
        },
        "peak_resident_bytes": resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
    }


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
    paths = _whole_training_models(rounds)
    documents = findit_documents(data_dir)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    validation = sorted(map(unlabelled, select(documents, Split.VAL)), key=lambda d: d.doc_id)
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        VALIDATION_INPUTS_RUN,
        store=store,
        config={
            "decision": "ADR-0029",
            "frozen_reader": str(FROZEN),
            "training_inputs": training_run.run_id,
            "models": _models_config(paths),
        },
        seeds={f"system 2, seed {s}": s for s in SEEDS},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        made = _split_inputs(train, validation, paths, "validation receipts")
        table = made.table
        save_table(table, output)
        current.log("summary", summary(table))
        current.log("sha256", _sha256(output))
        current.log("rules_scorers", made.scorers)
        current.log("models", made.models)
        current.log(
            "changed_since_training_inputs", changed_since(training_run.code.commit, ["src"])
        )
    console.print(f"[green]written[/green] {output}: {len(table.receipts)} rows")


@app.command("rehearse-inputs")
def rehearse_inputs_command(
    data_dir: DataDir = Path("data"),
    runs: Runs = REPORTED_RUNS,
    rounds: Annotated[Path, typer.Option(help="Where the rounds are kept.")] = ROUNDS_DIR,
    validation: Annotated[Path, typer.Option(help="The recorded validation inputs table.")] = (
        VALIDATION_INPUTS
    ),
    output: Annotated[Path, typer.Option(help="Where to write the table.")] = REHEARSAL_INPUTS,
) -> None:
    """Make the validation inputs again with today's code and compare them (ADR-0034, 13).

    The test inputs are made only by code that reproduces the recorded validation table exactly
    (:mod:`docfraud.fusion.rehearsal`). Reads no test receipt. The run record says whether the
    table was reproduced and, if not, where it differs.
    """
    store = JsonRunStore(runs)
    require_clean_commit()
    inputs_run = _latest(store, VALIDATION_INPUTS_RUN)
    if _sha256(validation) != inputs_run.results["sha256"]:
        console.print(f"[red]{validation} is not the table {inputs_run.run_id} wrote[/red]")
        raise typer.Exit(1)
    paths = _whole_training_models(rounds)
    documents = findit_documents(data_dir)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    receipts = sorted(map(unlabelled, select(documents, Split.VAL)), key=lambda d: d.doc_id)
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        REHEARSAL_RUN,
        store=store,
        config={
            "decision": "ADR-0034",
            "validation_inputs": inputs_run.run_id,
            "models": _models_config(paths),
        },
        seeds={f"system 2, seed {s}": s for s in SEEDS},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        made = _split_inputs(train, receipts, paths, "validation receipts, again")
        save_table(made.table, output)
        recorded = load_table(validation)
        current.log("reproduced", made.table == recorded)
        current.log("differences", differences(recorded, made.table))
        current.log("sha256", _sha256(output))
    verdict = "reproduced" if current.results["reproduced"] else "[red]not reproduced[/red]"
    console.print(f"the validation inputs: {verdict}")


def _rehearsal(store: JsonRunStore) -> RunRecord:
    """The latest rehearsal, if it reproduced the validation inputs with today's input code.

    Raises:
        typer.Exit: If there is none, it did not reproduce them from a clean commit, or the code
            that makes the inputs changed after it.
    """
    rehearsal = _latest(store, REHEARSAL_RUN)
    if (
        not rehearsal.results["reproduced"]
        or rehearsal.code.commit is None
        or rehearsal.code.dirty is not False
    ):
        console.print(
            f"[red]{rehearsal.run_id} did not reproduce the validation inputs from a clean "
            "commit[/red]"
        )
        raise typer.Exit(1)
    require_unchanged(
        rehearsal.code.commit,
        INPUT_CODE,
        "the code that makes the inputs changed since the rehearsal",
    )
    return rehearsal


@app.command("test-inputs")
def final_test_inputs_command(
    data_dir: DataDir = Path("data"),
    runs: Runs = REPORTED_RUNS,
    rounds: Annotated[Path, typer.Option(help="Where the rounds are kept.")] = ROUNDS_DIR,
    output: Annotated[Path, typer.Option(help="Where to write the table.")] = TEST_INPUTS,
) -> None:
    """Compute the fusion inputs of every test receipt, once (ADR-0034, decisions 1 to 3).

    As the validation inputs were made, by code that the rehearsal showed to reproduce them: it
    runs only at the freeze tag, from a clean commit, after the frozen reader read the test
    split, and only if the code that makes the inputs is unchanged since a rehearsal that
    reproduced the validation table. The test receipts are read without their labels and
    annotations. The run record records the unlock, aggregates and each component's time.
    """
    store = JsonRunStore(runs)
    if store.records(TEST_INPUTS_RUN):
        console.print(f"[red]refusing to run again:[/red] a {TEST_INPUTS_RUN} record exists")
        raise typer.Exit(1)
    require_clean_commit()
    require_freeze_tag()
    rehearsal = _rehearsal(store)
    if not store.records(TEST_READING_RUN):
        console.print("[red]the frozen reader has not read the test split[/red]: llm read-test")
        raise typer.Exit(1)
    paths = _whole_training_models(rounds)
    documents = findit_documents(data_dir)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        TEST_INPUTS_RUN,
        store=store,
        config={
            "decision": "ADR-0034",
            "rehearsal": rehearsal.run_id,
            "test_reading": store.records(TEST_READING_RUN)[-1].run_id,
            "models": _models_config(paths),
        },
        seeds={f"system 2, seed {s}": s for s in SEEDS},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        test = select(documents, Split.TEST, unlock_test=True, reason=TEST_UNLOCK)
        receipts = sorted(map(unlabelled, test), key=lambda d: d.doc_id)
        made = _split_inputs(train, receipts, paths, "test receipts")
        save_table(made.table, output)
        current.log("summary", summary(made.table))
        current.log("sha256", _sha256(output))
        current.log("rules_scorers", made.scorers)
        current.log("models", made.models)
        current.log("efficiency", _efficiency(made.seconds))
    console.print(f"[green]written[/green] {output}: {len(made.table.receipts)} rows")


@app.command("resaved-inputs")
def resaved_inputs_command(
    data_dir: DataDir = Path("data"),
    runs: Runs = REPORTED_RUNS,
    rounds: Annotated[Path, typer.Option(help="Where the rounds are kept.")] = ROUNDS_DIR,
    resaved_to: Annotated[Path, typer.Option(help="Where track R's images go.")] = RESAVED,
    output: Annotated[Path, typer.Option(help="Where to write the tables.")] = RESAVED_INPUTS,
) -> None:
    """Compute track R's inputs: the test receipts saved again as JPEG, once (ADR-0034, 4).

    At every quality of track R, the test images are saved again and enter the T2 pipeline from
    the start, with system 2 at seed 0, as the test inputs are made and under their guards,
    after the frozen reader read the saved images' text. The images must be those it read. One
    table per quality goes to ``output``; the run record records the unlock and aggregates.
    """
    store = JsonRunStore(runs)
    if store.records(RESAVED_INPUTS_RUN):
        console.print(f"[red]refusing to run again:[/red] a {RESAVED_INPUTS_RUN} record exists")
        raise typer.Exit(1)
    require_clean_commit()
    require_freeze_tag()
    rehearsal = _rehearsal(store)
    readings = store.records(RESAVED_READING_RUN)
    if not readings:
        console.print("[red]the frozen reader has not read track R[/red]: llm read-resaved")
        raise typer.Exit(1)
    paths = _whole_training_models(rounds)
    documents = findit_documents(data_dir)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        RESAVED_INPUTS_RUN,
        store=store,
        config={
            "decision": "ADR-0034",
            "rehearsal": rehearsal.run_id,
            "resaved_reading": readings[-1].run_id,
            "qualities": list(QUALITIES),
            "models": _models_config({0: paths[0]}),
        },
        seeds={"system 2, seed 0": 0},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        test = sorted(
            select(documents, Split.TEST, unlock_test=True, reason=TEST_UNLOCK),
            key=lambda d: d.doc_id,
        )
        for quality in QUALITIES:
            saved = resaved(test, quality, resaved_to / f"q{quality}")
            if digest(saved) != readings[-1].results["images"][str(quality)]:
                console.print(f"[red]the images at quality {quality} are not those read[/red]")
                raise typer.Exit(1)
            receipts = list(map(unlabelled, saved))
            made = _split_inputs(train, receipts, {0: paths[0]}, f"test receipts at {quality}")
            table = output / f"q{quality}-inputs.json"
            save_table(made.table, table)
            current.log(f"q{quality}", {"summary": summary(made.table), "sha256": _sha256(table)})
    console.print(f"[green]written[/green] track R's inputs at qualities {QUALITIES} to {output}")


@dataclass(frozen=True, slots=True)
class _Recorded:
    """Both inputs tables, checked against the run records that made them."""

    training_run: RunRecord
    inputs_run: RunRecord
    training: InputsTable
    validation: InputsTable


def _recorded_tables(store: JsonRunStore, training: Path, validation: Path) -> _Recorded:
    """The tables that the latest validation inputs and their training inputs describe.

    Stops unless the validation inputs were made from a clean commit whose source code is the
    current one, and each table on disk is the one its record describes: the training table by
    its aggregates, the validation table by its SHA-256.
    """
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
    return _Recorded(training_run, inputs_run, training_table, load_table(validation))


Training = Annotated[Path, typer.Option(help="The training inputs table.")]
Validation = Annotated[Path, typer.Option(help="The validation inputs table.")]


@app.command("input-shift")
def input_shift_command(
    runs: Runs = REPORTED_RUNS,
    training: Training = TRAINING_INPUTS,
    validation: Validation = VALIDATION_INPUTS,
    output: Output = SHIFT_REPORT,
) -> None:
    """Compare every fusion input between the two splits, without labels (ADR-0029, 7 (d)).

    Reads the two tables only, no receipt and no label, and only those their run records
    describe. Runs only from a clean commit whose source code is that of the validation inputs,
    and never after the validation check, which needs this comparison first.
    """
    store = JsonRunStore(runs)
    _refuse_after_the_check(store)
    require_clean_commit()
    recorded = _recorded_tables(store, training, validation)
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        SHIFT_RUN,
        store=store,
        config={
            "decision": "ADR-0029",
            "training_inputs": recorded.training_run.run_id,
            "validation_inputs": recorded.inputs_run.run_id,
            "validation_sha256": recorded.inputs_run.results["sha256"],
        },
        seeds={"permutation": 0},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        for key, value in input_shift(recorded.training, recorded.validation, seeds=SEEDS).items():
            current.log(key, value)
    input_shift_report_command(runs, output)


@app.command("input-shift-report")
def input_shift_report_command(runs: Runs = REPORTED_RUNS, output: Output = SHIFT_REPORT) -> None:
    """Render the comparison of the inputs on both splits from its run record."""
    record = _latest(JsonRunStore(runs), SHIFT_RUN)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        shift_report.render_report(record.results, record.code.commit), encoding="utf-8"
    )
    console.print(f"[green]written[/green] {output}")


@app.command("validation-check")
def validation_check_command(
    data_dir: DataDir = Path("data"),
    runs: Runs = REPORTED_RUNS,
    training: Training = TRAINING_INPUTS,
    validation: Validation = VALIDATION_INPUTS,
    output: Output = VALIDATION_REPORT,
) -> None:
    """Check the frozen system once on the validation split (ADR-0029, sections 1, 8 to 13).

    Refuses to run again. Runs only from a clean commit whose code is that of the validation
    inputs, on the two tables their run records describe, and only after the inputs of both
    splits have been compared (``input-shift``). Only aggregates go to the run record
    and the report, with what Phase 7 applies to the test split: every row's model, Platt
    scaling and max-F1 threshold, and the review policy's thresholds.
    """
    store = JsonRunStore(runs)
    _refuse_after_the_check(store)
    require_clean_commit()
    recorded = _recorded_tables(store, training, validation)
    training_run, inputs_run = recorded.training_run, recorded.inputs_run
    training_table, validation_inputs = recorded.training, recorded.validation
    if not any(
        r.config.get("validation_inputs") == inputs_run.run_id for r in store.records(SHIFT_RUN)
    ):
        console.print(
            "[red]compare the inputs of both splits first[/red] (ADR-0029, section 7 (d)): "
            "docfraud fusion input-shift"
        )
        raise typer.Exit(1)
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
