"""``docfraud agent``: the investigation agent, its pilot and its development (ADR-0031, 5).

``pilot`` runs version 1 of the agent's instruction on the three pilot cases with Claude Haiku
5.5 and Claude Sonnet 5.5, once, and keeps the model by the rule fixed in advance: Haiku if all
three of its case files take the fixed form, cite evidence in every observation and pass the
grounding check, otherwise Sonnet. It also sets the token budget of a case for development:
twice the most tokens a case of the kept model used, rounded up to a thousand.

``develop`` runs one version of the instruction, once, on the 12 development cases with the
model and the token budget the pilot kept; ``freeze`` keeps a version by the rule of ADR-0033.
Every case file is written beside its record, for the owner to read.

These cases are design-fold receipts of the training split. ``validation`` then runs the frozen
agent once on the validation receipts that track T2's review policy sends to review. Every
command checks the receipts it reads before any request is built.
"""

import hashlib
import json
import math
from collections.abc import Sequence
from pathlib import Path
from typing import Annotated, Any

import numpy as np
import typer

from docfraud.cli.common import (
    CV_FOLDS_PATH,
    OCR_CACHE,
    REPORTED_RUNS,
    TEST_UNLOCK,
    DataDir,
    DatasetName,
    NoRequests,
    cached_ocr,
    console,
    findit_documents,
    require_clean_commit,
    require_freeze_tag,
    require_unchanged,
)
from docfraud.cli.fusion import TEST_INPUTS, TRAINING_INPUTS, VALIDATION_INPUTS, _latest, _pages
from docfraud.cli.llm import FROZEN, LEDGER, STORE, Cache, LedgerPath, Store, _runner
from docfraud.data.download import SOURCES
from docfraud.data.models import Document, Label, Split
from docfraud.data.normalize import load_grayscale
from docfraud.data.splits import select
from docfraud.decision import BUDGETS
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS
from docfraud.evaluation.folds import load_folds
from docfraud.evaluation.runs import JsonRunStore, RunRecord, code_version, run
from docfraud.forensics.learned.rounds import SEEDS
from docfraud.fusion.final import RUN_NAME as TEST_CHECK_RUN
from docfraud.fusion.final import differences_in
from docfraud.fusion.rehearsal import TEST_INPUTS_RUN
from docfraud.fusion.training_inputs import RUN_NAME as TRAINING_INPUTS_RUN
from docfraud.fusion.training_inputs import load_table, summary
from docfraud.fusion.validation import RUN_NAME as CHECK_RUN
from docfraud.llm.agent import (
    HAIKU_AGENT,
    NOT_ANSWERED,
    SONNET_AGENT,
    Investigation,
    investigate,
)
from docfraud.llm.agent_cases import TRACK, Development, development_cases, out_of_fold
from docfraud.llm.agent_final import REHEARSAL_RUN as AGENT_REHEARSAL_RUN
from docfraud.llm.agent_final import RUN_NAME as TEST_RUN
from docfraud.llm.agent_final import TEST_CODE as AGENT_TEST_CODE
from docfraud.llm.agent_final import frozen_queue
from docfraud.llm.agent_rounds import (
    AGENT_CODE,
    VERSIONS,
    AgentRound,
    FrozenAgent,
    case_result,
    kept_version,
    load_rounds,
    save_round,
)
from docfraud.llm.agent_tools import Case, read_fields
from docfraud.llm.agent_validation import RUN_NAME as VALIDATION_RUN
from docfraud.llm.agent_validation import Queue, agent_validation, measured, review_queue
from docfraud.llm.agent_validation_report import render_report
from docfraud.llm.cost import Ledger
from docfraud.llm.frozen import FrozenReader
from docfraud.llm.provider import Settings
from docfraud.llm.record import FrozenVersion
from docfraud.llm.run import Runner, check_documents, target
from docfraud.llm.store import ResponseStore
from docfraud.reading.ocr import CachedOcr
from docfraud.reading.pipeline import TextSource, read_document

app = typer.Typer(help="The investigation agent: its pilot, development and validation run.")

INSTRUCTIONS = Path("src/docfraud/llm/instructions")
AGENT = Path("reports/llm/agent")
PILOT = "pilot.json"
FROZEN_AGENT = "frozen.json"
VALIDATION_REPORT = AGENT / "validation.md"

Output = Annotated[Path, typer.Option(help="Where the agent's records and case files go.")]
Training = Annotated[Path, typer.Option(help="The training inputs table of Phase 5.")]
Validation = Annotated[Path, typer.Option(help="The validation inputs table of Phase 5.")]
Runs = Annotated[Path, typer.Option(help="Where reported run records are kept.")]
Report = Annotated[Path, typer.Option(help="Where to write the report.")]


def instruction_path(version: int) -> Path:
    """Where version ``version`` of the agent's instruction is kept."""
    return INSTRUCTIONS / f"agent-v{version}.md"


def _training(data_dir: Path, training: Path) -> tuple[list[Document], Development, Any]:
    """The training receipts, the development cases and their out-of-fold assessments.

    Raises:
        typer.Exit: If the table is not the one its record describes, or not the training split.
    """
    table = load_table(training)
    records = JsonRunStore(REPORTED_RUNS).records(TRAINING_INPUTS_RUN)
    if not records or summary(table) != records[-1].results["summary"]:
        console.print(f"[red]{training} is not the table its latest record describes[/red]")
        raise typer.Exit(1)
    train = sorted(select(findit_documents(data_dir), Split.TRAIN), key=lambda d: d.doc_id)
    ids = [d.doc_id for d in train]
    rows = table.rows(TRACK, 0)
    if [r.receipt for r in rows] != ids:
        console.print("[red]the training table's receipts are not the training split[/red]")
        raise typer.Exit(1)
    y = np.array([d.label is Label.FORGED for d in train])
    folds = load_folds(Path(CV_FOLDS_PATH))
    found = out_of_fold(rows, y, np.array([folds.folds[i] for i in ids]))
    return train, development_cases(ids, y, found.probabilities, DESIGN_FOLD_IDS), found


def cases(
    documents: Sequence[Document], assessments: Any, ocr: CachedOcr, reader: FrozenReader
) -> dict[str, Case]:
    """What the agent's tools show of each receipt: the system's outputs and the page."""
    found = {}
    for document in documents:
        rows = read_document(document, ocr, TextSource.OCR).rows
        reading = reader.read(target(document.doc_id, TextSource.OCR, rows))
        found[document.doc_id] = Case(
            receipt=document.doc_id,
            assessment=assessments[document.doc_id],
            fields=read_fields(reading.extraction.fields, reading.extraction.sources),
            rows=tuple(rows),
            page=load_grayscale(document.image_path),
        )
    return found


def _reader(frozen: Path, store: Path) -> FrozenReader:
    version = FrozenVersion.model_validate_json(frozen.read_text(encoding="utf-8"))
    text = Path(version.instruction).read_text(encoding="utf-8")
    return FrozenReader(version, text, ResponseStore(store))


def described(investigation: Investigation, forged: bool) -> dict[str, Any]:
    """One case of the pilot, as its record keeps it."""
    grounding = investigation.grounding()
    found = investigation.read.case_file
    return {
        "receipt": investigation.receipt,
        "forged": forged,
        "failure": investigation.read.failure,
        "observations": len(found.observations) if found is not None else None,
        "grounding": None
        if grounding is None
        else {"facts": grounding.facts, "violations": list(grounding.violations)},
        "tool_calls": [
            {"tool": c.name, "arguments": dict(c.arguments)} for c in investigation.calls
        ],
        "steps": len(investigation.responses),
        "tokens": investigation.tokens,
        "cost_usd": f"{investigation.cost_usd:.4f}",
        "seconds": investigation.seconds,
    }


def complete(investigation: Investigation) -> bool:
    """Whether a case file takes the fixed form, cites evidence everywhere and is grounded."""
    grounding = investigation.grounding()
    return grounding is not None and not grounding.violations


def token_budget(investigations: Sequence[Investigation]) -> int:
    """Twice the most tokens a case used, rounded up to a thousand."""
    return math.ceil(2 * max(i.tokens for i in investigations) / 1000) * 1000


@app.command("pilot")
def agent_pilot(  # noqa: PLR0913, PLR0917 - one option per input and output
    data_dir: DataDir = Path("data"),
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    cache: Cache = OCR_CACHE,
    training: Training = TRAINING_INPUTS,
    output: Output = AGENT,
) -> None:
    """Pilot version 1 on three development cases with Haiku and Sonnet, once (ADR-0031, 5)."""
    record = output / PILOT
    if record.exists():
        console.print(f"[red]{record} exists[/red]: the pilot runs once")
        raise typer.Exit(1)
    require_clean_commit()
    path = instruction_path(1)
    instruction = path.read_text(encoding="utf-8")
    train, development, found = _training(data_dir, training)
    check_documents(development.pilot, DESIGN_FOLD_IDS, "the agent's pilot")
    chosen = [d for d in train if d.doc_id in development.pilot]
    by_id = cases(chosen, found.assessments, cached_ocr(cache), _reader(FROZEN, store))
    forged = {d.doc_id: d.label is Label.FORGED for d in chosen}
    runner = _runner(store, ledger)
    runs = {
        settings.model: [
            investigate(by_id[r], settings, instruction, runner) for r in development.pilot
        ]
        for settings in (HAIKU_AGENT, SONNET_AGENT)
    }
    kept = HAIKU_AGENT if all(complete(i) for i in runs[HAIKU_AGENT.model]) else SONNET_AGENT
    files = output / "pilot"
    files.mkdir(parents=True, exist_ok=True)
    for model, done in runs.items():
        for investigation in done:
            rendered = investigation.rendered or f"No case file: {investigation.read.failure}\n"
            (files / f"{model}-{investigation.receipt}.md").write_text(rendered, encoding="utf-8")
    result = {
        "instruction": str(path),
        "commit": code_version().commit,
        "cases": list(development.pilot),
        "models": {
            model: [described(i, forged[i.receipt]) for i in done] for model, done in runs.items()
        },
        "kept": kept.model_dump(mode="json"),
        "token_budget": token_budget(runs[kept.model]),
    }
    record.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    for model, done in runs.items():
        for i in done:
            grounding = i.grounding()
            shown = (
                i.read.failure or f"{len(grounding.violations) if grounding else '?'} violations"
            )
            console.print(
                f"{model} · {i.receipt} · {shown} · {len(i.calls)} calls · {i.tokens} tokens · "
                f"{i.cost_usd:.4f} USD"
            )
    console.print(f"kept {kept.model}; token budget {result['token_budget']}; written {record}")


def _kept(output: Path) -> tuple[Settings, int]:
    """The model and the token budget the pilot kept.

    Raises:
        typer.Exit: If the pilot has not run.
    """
    path = output / PILOT
    if not path.exists():
        console.print(f"[red]{path} does not exist[/red]: run the pilot first")
        raise typer.Exit(1)
    record = json.loads(path.read_text(encoding="utf-8"))
    return Settings.model_validate(record["kept"]), int(record["token_budget"])


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode()).hexdigest()


def _sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@app.command("develop")
def agent_develop(  # noqa: PLR0913, PLR0917 - one option per input and output
    version: Annotated[int, typer.Option(help="The version of the instruction, 1 or 2.")],
    data_dir: DataDir = Path("data"),
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    cache: Cache = OCR_CACHE,
    training: Training = TRAINING_INPUTS,
    output: Output = AGENT,
) -> None:
    """Run one version of the instruction on the 12 development cases, once (ADR-0033, 3).

    Version 2 runs only after version 1, with the agent's code unchanged since version 1 ran,
    so that the two differ by their instruction alone. A run that leaves a step unanswered
    records nothing, so that a fault of the service does not count against a version; running
    it again replays the stored steps and sends only what is missing.
    """
    if version not in VERSIONS:
        console.print(f"[red]development allows versions {VERSIONS} only[/red]")
        raise typer.Exit(1)
    if (output / FROZEN_AGENT).exists():
        console.print("[red]the agent is frozen[/red]: development has ended")
        raise typer.Exit(1)
    record = output / f"round-v{version}.json"
    if record.exists():
        console.print(f"[red]{record} exists[/red]: each version runs once")
        raise typer.Exit(1)
    developed = load_rounds(output)
    if version > 1 and 1 not in developed:
        console.print("[red]run version 1 on the development cases first[/red]")
        raise typer.Exit(1)
    require_clean_commit()
    if version > 1:
        require_unchanged(
            developed[1].commit, AGENT_CODE, "the agent's code changed after version 1 ran"
        )
    settings, budget = _kept(output)
    path = instruction_path(version)
    if not path.exists():
        console.print(f"[red]{path} does not exist[/red]: write version {version} first")
        raise typer.Exit(1)
    instruction = path.read_text(encoding="utf-8")
    train, development, found = _training(data_dir, training)
    check_documents(development.receipts, DESIGN_FOLD_IDS, "the agent's development")
    by_id = {d.doc_id: d for d in train if d.doc_id in development.receipts}
    chosen = [by_id[r] for r in development.receipts]
    shown = cases(chosen, found.assessments, cached_ocr(cache), _reader(FROZEN, store))
    runner = _runner(store, ledger)
    done = [
        investigate(shown[r], settings, instruction, runner, budget=budget)
        for r in development.receipts
    ]
    unanswered = [i.receipt for i in done if i.read.failure == NOT_ANSWERED]
    if unanswered:
        console.print(
            f"[red]{len(unanswered)} of {len(done)} cases were not answered[/red] "
            f"({', '.join(unanswered)}); nothing is recorded. Run the command again to send them."
        )
        raise typer.Exit(1)
    files = output / f"v{version}"
    files.mkdir(parents=True, exist_ok=True)
    for investigation in done:
        rendered = investigation.rendered or f"No case file: {investigation.read.failure}\n"
        (files / f"{investigation.receipt}.md").write_text(rendered, encoding="utf-8")
    result = AgentRound(
        version=version,
        settings=settings,
        token_budget=budget,
        instruction=str(path),
        instruction_sha256=_sha256(instruction),
        commit=str(code_version().commit),
        cases=tuple(case_result(i, by_id[i.receipt], found.assessments[i.receipt]) for i in done),
    )
    save_round(result, record)
    for case in result.cases:
        said = case.failure or f"{len(case.violations or ())} violations"
        pointed = (
            "" if case.first_location_hit is None else f" · on target {case.first_location_hit}"
        )
        console.print(
            f"{case.receipt} · {said}{pointed} · {case.tokens} tokens · {case.cost_usd} USD"
        )
    console.print(
        f"version {version}: {result.failures} without a case file, {result.violations} "
        f"violations, first location on target {result.hits} of {result.forged} "
        f"(audit report {result.report_hits}), {result.cost_usd:.4f} USD; written {record}"
    )


@app.command("freeze")
def agent_freeze(output: Output = AGENT) -> None:
    """Freeze the version the rule of ADR-0033 keeps, at a clean commit."""
    path = output / FROZEN_AGENT
    if path.exists():
        console.print(f"[red]{path} exists[/red]: the agent is frozen once")
        raise typer.Exit(1)
    require_clean_commit()
    developed = load_rounds(output)
    try:
        version = kept_version(developed)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    kept = developed[version]
    text = Path(kept.instruction).read_text(encoding="utf-8")
    if _sha256(text) != kept.instruction_sha256:
        console.print(f"[red]{kept.instruction} changed after version {version} ran[/red]")
        raise typer.Exit(1)
    require_unchanged(
        kept.commit, AGENT_CODE, f"the agent's code changed after version {version} ran"
    )
    frozen = FrozenAgent(
        version=version,
        settings=kept.settings,
        token_budget=kept.token_budget,
        instruction=kept.instruction,
        instruction_sha256=kept.instruction_sha256,
        commit=str(code_version().commit),
    )
    path.write_text(frozen.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]frozen[/green] version {version} at {frozen.commit[:7]}")


def _frozen(output: Path) -> tuple[FrozenAgent, str]:
    """The frozen agent and its instruction, both checked unchanged since the freeze.

    The instruction is checked by its hash, the agent's code by git.

    Raises:
        typer.Exit: If the agent is not frozen, or something changed after it was frozen.
    """
    path = output / FROZEN_AGENT
    if not path.exists():
        console.print(f"[red]{path} does not exist[/red]: freeze the agent first")
        raise typer.Exit(1)
    frozen = FrozenAgent.model_validate_json(path.read_text(encoding="utf-8"))
    text = Path(frozen.instruction).read_text(encoding="utf-8")
    if _sha256(text) != frozen.instruction_sha256:
        console.print(f"[red]{frozen.instruction} changed after it was frozen[/red]")
        raise typer.Exit(1)
    require_unchanged(frozen.commit, AGENT_CODE, "the agent changed after it was frozen")
    return frozen, text


def _labels(documents: Sequence[Document]) -> Any:
    """Whether each receipt is forged.

    Raises:
        typer.Exit: If a receipt has no label.
    """
    if any(d.label is None for d in documents):
        console.print("[red]every receipt must be labelled authentic or forged[/red]")
        raise typer.Exit(1)
    return np.array([d.label is Label.FORGED for d in documents])


def _review_queue(
    check: RunRecord, records: JsonRunStore, data_dir: Path, training: Path, validation: Path
) -> tuple[Queue, list[Document]]:
    """Track T2's review queue on the validation split, rebuilt from Phase 5's check.

    Raises:
        typer.Exit: If a table is not the one the check read, or the queue is not the recorded.
    """
    if _sha256_file(validation) != check.config["validation_sha256"]:
        console.print(f"[red]{validation} is not the table {check.run_id} read[/red]")
        raise typer.Exit(1)
    training_table = load_table(training)
    if summary(training_table) != records.load(check.config["training_inputs"]).results["summary"]:
        console.print(f"[red]{training} is not the table {check.run_id} was fitted on[/red]")
        raise typer.Exit(1)
    documents = findit_documents(data_dir)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    val = sorted(select(documents, Split.VAL), key=lambda d: d.doc_id)
    training_rows = training_table.rows(TRACK, SEEDS[0])
    validation_rows = load_table(validation).rows(TRACK, SEEDS[0])
    if [r.receipt for r in training_rows] != [d.doc_id for d in train] or [
        r.receipt for r in validation_rows
    ] != [d.doc_id for d in val]:
        console.print("[red]the tables' receipts are not the training and validation splits[/red]")
        raise typer.Exit(1)
    try:
        queue = review_queue(
            training_rows, _labels(train), validation_rows, _labels(val), check.results
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    return queue, val


@app.command("validation")
def agent_validation_command(  # noqa: PLR0913, PLR0917 - one option per input and output
    data_dir: DataDir = Path("data"),
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    cache: Cache = OCR_CACHE,
    training: Training = TRAINING_INPUTS,
    validation: Validation = VALIDATION_INPUTS,
    runs: Runs = REPORTED_RUNS,
    output: Output = AGENT,
    report: Report = VALIDATION_REPORT,
) -> None:
    """Run the frozen agent once on the validation receipts sent to review (ADR-0031, 5 and 6).

    Refuses to run again. Runs only from a clean commit, on the review queue of track T2 that
    Phase 5's check left: its fusion model is fitted again on the training table its record
    names and must reproduce the recorded probabilities and policy. A run that leaves a step
    unanswered records nothing; running it again replays the stored steps and sends only what
    is missing. The case files go beside the agent's records, the results to the run record and
    its report.
    """
    records = JsonRunStore(runs)
    if records.records(VALIDATION_RUN):
        console.print(f"[red]refusing to run again:[/red] a {VALIDATION_RUN} record exists")
        raise typer.Exit(1)
    frozen, text = _frozen(output)
    require_clean_commit()
    checks = records.records(CHECK_RUN)
    if not checks:
        console.print(f"[red]no {CHECK_RUN} record in {runs}[/red]: Phase 5's check comes first")
        raise typer.Exit(1)
    check = checks[-1]
    queue, val = _review_queue(check, records, data_dir, training, validation)
    receipts = list(queue.assessments)
    check_documents(receipts, {d.doc_id for d in val}, "the agent's validation run")
    by_id = {d.doc_id: d for d in val}
    chosen = [by_id[r] for r in receipts]
    shown = cases(chosen, queue.assessments, cached_ocr(cache), _reader(FROZEN, store))
    runner = _runner(store, ledger)
    done = [
        investigate(shown[r], frozen.settings, text, runner, budget=frozen.token_budget)
        for r in receipts
    ]
    unanswered = [i.receipt for i in done if i.read.failure == NOT_ANSWERED]
    if unanswered:
        console.print(
            f"[red]{len(unanswered)} of {len(done)} cases were not answered[/red] "
            f"({', '.join(unanswered)}); nothing is recorded. Run the command again to send them."
        )
        raise typer.Exit(1)
    pages = _pages(chosen)
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        VALIDATION_RUN,
        store=records,
        config={
            "decision": "ADR-0031, ADR-0033",
            "frozen": frozen.model_dump(mode="json"),
            "validation_check": check.run_id,
            "validation_sha256": check.config["validation_sha256"],
            "policy": {"budget": queue.policy.budget, "threshold": queue.policy.threshold},
        },
        seeds={"bootstrap": 0, "baseline": 0},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        results = agent_validation(
            [
                measured(i, by_id[i.receipt], queue.assessments[i.receipt], pages.get(i.receipt))
                for i in done
            ]
        )
        for key, value in results.items():
            current.log(key, value)
    # Written once the record is saved: the record names the commit the tree was clean at.
    files = output / "validation"
    files.mkdir(parents=True, exist_ok=True)
    for investigation in done:
        rendered = investigation.rendered or f"No case file: {investigation.read.failure}\n"
        (files / f"{investigation.receipt}.md").write_text(rendered, encoding="utf-8")
    agent_validation_report_command(runs, report)


@app.command("validation-report")
def agent_validation_report_command(
    runs: Runs = REPORTED_RUNS, report: Report = VALIDATION_REPORT
) -> None:
    """Render the report of the agent's validation run from its run record."""
    found = JsonRunStore(runs).records(VALIDATION_RUN)
    if not found:
        console.print(f"[red]no {VALIDATION_RUN} record in {runs}[/red]")
        raise typer.Exit(1)
    record = found[-1]
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(
        render_report(record.results, record.config, record.code.commit), encoding="utf-8"
    )
    console.print(f"[green]written[/green] {report}")


def _frozen_queue(  # noqa: PLR0913, PLR0917 - the record, its store, both splits, both tables
    check: RunRecord,
    records: JsonRunStore,
    train: Sequence[Document],
    documents: Sequence[Document],
    training: Path,
    table: Path,
) -> Queue:
    """Track T2's review queue among ``documents``, with Phase 5's check applied unchanged.

    Raises:
        typer.Exit: If the training table is not the one the check was fitted on, the table's
            rows are not the receipts given, or the fusion model is not the recorded one.
    """
    training_table = load_table(training)
    if summary(training_table) != records.load(check.config["training_inputs"]).results["summary"]:
        console.print(f"[red]{training} is not the table {check.run_id} was fitted on[/red]")
        raise typer.Exit(1)
    rows = load_table(table).rows(TRACK, SEEDS[0])
    if [r.receipt for r in rows] != [d.doc_id for d in documents]:
        console.print(f"[red]the rows of {table} are not the receipts measured[/red]")
        raise typer.Exit(1)
    try:
        return frozen_queue(
            training_table.rows(TRACK, SEEDS[0]), _labels(train), rows, check.results
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error


@app.command("rehearse-test")
def agent_rehearse_test(  # noqa: PLR0913, PLR0917 - one option per input and output
    data_dir: DataDir = Path("data"),
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    cache: Cache = OCR_CACHE,
    training: Training = TRAINING_INPUTS,
    validation: Validation = VALIDATION_INPUTS,
    runs: Runs = REPORTED_RUNS,
    output: Output = AGENT,
) -> None:
    """Apply the test run's code to the validation receipts from stored steps (ADR-0034).

    The queue, built with the check's recorded policy, must be the validation run's; every step
    of every case must be stored, so that none is sent; and every result of the validation run
    must come back. Reads no test receipt.
    """
    records = JsonRunStore(runs)
    require_clean_commit()
    frozen, text = _frozen(output)
    measured_run = _latest(records, VALIDATION_RUN)
    check = _latest(records, CHECK_RUN)
    if _sha256_file(validation) != check.config["validation_sha256"]:
        console.print(f"[red]{validation} is not the table {check.run_id} read[/red]")
        raise typer.Exit(1)
    documents = findit_documents(data_dir)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    val = sorted(select(documents, Split.VAL), key=lambda d: d.doc_id)
    queue = _frozen_queue(check, records, train, val, training, validation)
    receipts = list(queue.assessments)
    if receipts != [c["receipt"] for c in measured_run.results["cases"]]:
        console.print(f"[red]the queue is not the one {measured_run.run_id} measured[/red]")
        raise typer.Exit(1)
    by_id = {d.doc_id: d for d in val}
    chosen = [by_id[r] for r in receipts]
    shown = cases(chosen, queue.assessments, cached_ocr(cache), _reader(FROZEN, store))
    runner = Runner(NoRequests(), ResponseStore(store), Ledger(ledger))
    try:
        done = [
            investigate(shown[r], frozen.settings, text, runner, budget=frozen.token_budget)
            for r in receipts
        ]
    except RuntimeError as error:
        console.print(f"[red]{error}[/red]: the agent no longer asks as it did")
        raise typer.Exit(1) from error
    pages = _pages(chosen)
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        AGENT_REHEARSAL_RUN,
        store=records,
        config={"decision": "ADR-0034", "validation_run": measured_run.run_id},
        seeds={"bootstrap": 0, "baseline": 0},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        rebuilt = agent_validation(
            [
                measured(i, by_id[i.receipt], queue.assessments[i.receipt], pages.get(i.receipt))
                for i in done
            ]
        )
        mismatches = differences_in(measured_run.results, rebuilt)
        current.log("reproduced", not mismatches)
        current.log("mismatches", mismatches)
    verdict = "reproduced" if not mismatches else f"[red]{len(mismatches)} differences[/red]"
    console.print(f"the agent's validation run: {verdict}")


@app.command("test")
def agent_test_command(  # noqa: PLR0913, PLR0917 - one option per input and output
    data_dir: DataDir = Path("data"),
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    cache: Cache = OCR_CACHE,
    training: Training = TRAINING_INPUTS,
    test: Annotated[Path, typer.Option(help="The test inputs table.")] = TEST_INPUTS,
    runs: Runs = REPORTED_RUNS,
    output: Output = AGENT,
) -> None:
    """Run the frozen agent once on the test receipts sent to review (ADR-0034, decision 7).

    The queue is the T2 policy of Phase 5's check applied to the test inputs, and must hold as
    many receipts as the test check counted. Runs once, at the freeze tag, from a clean commit,
    after the test check and a rehearsal that reproduced the validation run with the code
    unchanged since. A run that leaves a step unanswered records nothing. The case files go to
    ``output``/test once the record is saved.
    """
    records = JsonRunStore(runs)
    if records.records(TEST_RUN):
        console.print(f"[red]refusing to run again:[/red] a {TEST_RUN} record exists")
        raise typer.Exit(1)
    require_clean_commit()
    require_freeze_tag()
    frozen, text = _frozen(output)
    rehearsal = _latest(records, AGENT_REHEARSAL_RUN)
    if (
        not rehearsal.results["reproduced"]
        or rehearsal.code.commit is None
        or rehearsal.code.dirty is not False
    ):
        console.print(f"[red]{rehearsal.run_id} did not reproduce the validation run[/red]")
        raise typer.Exit(1)
    require_unchanged(rehearsal.code.commit, AGENT_TEST_CODE, "the agent changed since")
    check = _latest(records, CHECK_RUN)
    test_check = _latest(records, TEST_CHECK_RUN)
    inputs = _latest(records, TEST_INPUTS_RUN)
    if _sha256_file(test) != inputs.results["sha256"]:
        console.print(f"[red]{test} is not the table {inputs.run_id} wrote[/red]")
        raise typer.Exit(1)
    documents = findit_documents(data_dir)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        TEST_RUN,
        store=records,
        config={
            "decision": "ADR-0034",
            "frozen": frozen.model_dump(mode="json"),
            "rehearsal": rehearsal.run_id,
            "test_check": test_check.run_id,
        },
        seeds={"bootstrap": 0, "baseline": 0},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        receipts_of_test = sorted(
            select(documents, Split.TEST, unlock_test=True, reason=TEST_UNLOCK),
            key=lambda d: d.doc_id,
        )
        queue = _frozen_queue(check, records, train, receipts_of_test, training, test)
        counted = test_check.results["policy"][TRACK][f"{BUDGETS[0] * 100:.0f} %"]["reviewed"]
        if len(queue.assessments) != counted:
            console.print(
                f"[red]{len(queue.assessments)} sent to review; the test check counted "
                f"{counted}[/red]"
            )
            raise typer.Exit(1)
        by_id = {d.doc_id: d for d in receipts_of_test}
        chosen = [by_id[r] for r in queue.assessments]
        shown = cases(chosen, queue.assessments, cached_ocr(cache), _reader(FROZEN, store))
        runner = _runner(store, ledger)
        done = [
            investigate(shown[r], frozen.settings, text, runner, budget=frozen.token_budget)
            for r in queue.assessments
        ]
        unanswered = [i.receipt for i in done if i.read.failure == NOT_ANSWERED]
        if unanswered:
            console.print(
                f"[red]{len(unanswered)} of {len(done)} cases were not answered[/red] "
                f"({', '.join(unanswered)}); nothing is recorded. Run the command again to send "
                "them."
            )
            raise typer.Exit(1)
        pages = _pages(chosen)
        results = agent_validation(
            [
                measured(i, by_id[i.receipt], queue.assessments[i.receipt], pages.get(i.receipt))
                for i in done
            ]
        )
        current.log("policy", {"budget": queue.policy.budget, "threshold": queue.policy.threshold})
        for key, value in results.items():
            current.log(key, value)
    # Written once the record is saved: the record names the commit the tree was clean at.
    files = output / "test"
    files.mkdir(parents=True, exist_ok=True)
    for investigation in done:
        rendered = investigation.rendered or f"No case file: {investigation.read.failure}\n"
        (files / f"{investigation.receipt}.md").write_text(rendered, encoding="utf-8")
    console.print(
        f"[green]recorded[/green] {len(done)} case files; grounding "
        f"{results['grounding']['rate']:.4f}"
    )
