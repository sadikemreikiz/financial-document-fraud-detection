"""``docfraud llm``: the LLM field reader on the gold sets (ADR-0024, ADR-0025, ADR-0026).

Every command that can call the API checks which receipts it may read before it builds a
request, sends only requests without a stored answer, and stops before the spending limit.
Gold set A is for development; gold set B is read once by the frozen version, adjudicated by the
owner, and only then evaluated.
"""

import hashlib
import json
import webbrowser
from collections import Counter, defaultdict
from collections.abc import Sequence
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import typer

from docfraud.cli.common import (
    GOLD_B_MANIFEST,
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
from docfraud.data.download import SOURCES
from docfraud.data.gold import GoldAnnotation, GoldManifest, load_annotations, load_manifest
from docfraud.data.local_page import HOST, make_server
from docfraud.data.models import Document, Label, Split
from docfraud.data.normalize import load_grayscale
from docfraud.data.observation_text import PageLanguage
from docfraud.data.splits import select
from docfraud.evaluation.extraction import DocumentScore, score_document
from docfraud.evaluation.runs import JsonRunStore, code_version, run
from docfraud.llm.adjudication import (
    GOLD,
    in_order,
    load_log,
    load_questions,
    save_questions,
)
from docfraud.llm.adjudication_server import AdjudicationApp
from docfraud.llm.claude import ClaudeProvider
from docfraud.llm.cost import Ledger
from docfraud.llm.decision import LADDER, ModelStep, decide_reader, model_step
from docfraud.llm.false_alarms import FIXED_AT, RULES_CODE, judge
from docfraud.llm.frozen import FrozenReader, SplitReading, save_split_reading
from docfraud.llm.gold_b import (
    LLM,
    PARSER,
    READER_CODE,
    READERS,
    Measurement,
    ReaderOutput,
    adjudicated_gold,
    adjudication_counts,
    false_alarms,
    latency,
    load_measurement,
    questions_on,
    reading_name,
    save_measurement,
    unanswered,
)
from docfraud.llm.provider import Request, Settings, Thinking
from docfraud.llm.reader import LlmReading
from docfraud.llm.record import FrozenVersion, RoundRecord, best_round
from docfraud.llm.report import (
    ReaderRun,
    render_false_alarms,
    render_gold_b,
    render_model_comparison,
    render_round,
)
from docfraud.llm.run import COMPARISON_SETTINGS, Runner, Target, check_documents, target
from docfraud.llm.store import ResponseStore
from docfraud.reading.fields import Extraction, ReceiptFields
from docfraud.reading.ocr import CachedOcr
from docfraud.reading.parser import parse_receipt
from docfraud.reading.pipeline import TextSource, read_document

app = typer.Typer(
    help="The LLM field reader: model choice and development on gold set A, measurement on gold "
    "set B, and the reading of the validation split for Phase 5."
)

STORE = Path("artifacts/llm/responses")
LEDGER = Path("artifacts/llm/ledger.jsonl")
CHOICE = Path("reports/llm/model-choice.json")
ROUNDS = Path("reports/llm/development")
FROZEN = Path("reports/llm/frozen.json")
GOLD_B_DIR = Path("artifacts/llm/gold-b")
ADJUDICATION = Path("annotations/findit2/gold-adjudication-b.json")
GOLD_B_REPORT = Path("reports/llm/gold-b.md")
FALSE_ALARMS_DIR = Path("artifacts/llm/false-alarms")
FALSE_ALARMS_REPORT = Path("reports/llm/false-alarms.md")
VALIDATION_READING = Path("reports/llm/validation-reading.json")
TEST_READING_RUN = "llm-test-reading"
GOLD_A = GOLD_MANIFEST
GOLD_B = GOLD_B_MANIFEST
MAX_ROUNDS = 10

Instruction = Annotated[Path, typer.Option(help="The fixed instructions the model reads.")]
Store = Annotated[Path, typer.Option(help="Where responses are kept.")]
LedgerPath = Annotated[Path, typer.Option("--ledger", help="Where spending is recorded.")]
Cache = Annotated[Path, typer.Option(help="OCR cache directory.")]
Frozen = Annotated[Path, typer.Option(help="The frozen version.")]
MeasurementPath = Annotated[
    Path, typer.Option("--measurement", help="Where the reading of gold set B is kept.")
]
QuestionsPath = Annotated[
    Path, typer.Option("--questions", help="Where the questions of the adjudication are kept.")
]
AdjudicationPath = Annotated[
    Path, typer.Option("--log", help="Where the owner's answers are saved.")
]
Reviewer = Annotated[str, typer.Option(help="Name of the person who adjudicates, lower case.")]


@app.command("compare-models")
def llm_compare_models(  # noqa: PLR0913, PLR0917 - one option per input and output
    instruction: Instruction,
    data_dir: DataDir = Path("data"),
    cache: Cache = OCR_CACHE,
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = Path(
        "reports/llm/model-comparison.md"
    ),
    choice: Annotated[Path, typer.Option(help="Where to record the chosen model.")] = CHOICE,
) -> None:
    """Choose the model on gold set A, from the cheapest up (ADR-0025, decision 2)."""
    annotations, targets = _gold_a_targets(data_dir, cache)
    runner = _runner(store, ledger)
    text = instruction.read_text(encoding="utf-8")
    runs: dict[str, ReaderRun] = {}
    exact: dict[str, list[tuple[bool, ...]]] = {}

    def read(model: str) -> None:
        settings = COMPARISON_SETTINGS[model]
        console.print(f"reading gold set A with {model}")
        readings = runner.run(settings, text, targets)
        keys = {r.key for r in runner.requests(settings, text, targets)}
        runs[model] = _summary(model, targets, readings, annotations, _spent(runner.ledger, keys))
        exact[model] = _per_receipt(targets, runs[model])

    chosen = LADDER[0]
    steps: list[ModelStep] = []
    read(chosen)
    for larger in LADDER[1:]:
        read(larger)
        step = model_step(chosen, larger, exact[chosen], exact[larger])
        steps.append(step)
        if not step.larger_is_clearly_better:
            break
        chosen = larger
    parser = _summary("parser v2", targets, None, annotations, None)
    digest = hashlib.sha256(text.encode()).hexdigest()
    header = (
        "# Model comparison on gold set A\n\nGenerated by `docfraud llm compare-models`; do not "
        "edit by hand.\n\n"
        f"Instruction `{instruction}` (SHA-256 `{digest[:12]}`), the same for every model. "
        f"{len(targets) // len(TRACKS)} receipts of gold set A, read from the transcripts (T1) "
        "and from our OCR (T2). Settings: "
        + "; ".join(
            f"`{m}` thinking {COMPARISON_SETTINGS[m].thinking}, effort "
            f"{COMPARISON_SETTINGS[m].effort}"
            for m in runs
        )
        + ".\n\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_model_comparison(header, [*runs.values(), parser], steps, chosen), encoding="utf-8"
    )
    record = {
        "model": chosen,
        "settings": COMPARISON_SETTINGS[chosen].model_dump(mode="json"),
        "instruction": str(instruction),
        "instruction_sha256": digest,
    }
    choice.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]written[/green] {output}; chosen model: {chosen}")


@app.command("develop")
def llm_develop(  # noqa: PLR0913, PLR0917 - one option per input and output
    round_number: Annotated[int, typer.Option("--round", help="Development round, 1 to 10.")],
    instruction: Instruction,
    thinking: Annotated[
        Thinking | None, typer.Option(help="Override the thinking setting.")
    ] = None,
    effort: Annotated[str | None, typer.Option(help="Override the effort setting.")] = None,
    data_dir: DataDir = Path("data"),
    cache: Cache = OCR_CACHE,
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    choice: Annotated[Path, typer.Option(help="The recorded model choice.")] = CHOICE,
    reports: Annotated[Path, typer.Option(help="Directory of the round reports.")] = ROUNDS,
) -> None:
    """Read gold set A with the chosen model in one development round (ADR-0025, decision 3)."""
    if not 1 <= round_number <= MAX_ROUNDS:
        console.print(f"[red]a development round is numbered 1 to {MAX_ROUNDS}[/red]")
        raise typer.Exit(1)
    output = reports / f"round-{round_number:02d}.md"
    if output.exists():
        console.print(f"[red]{output} exists[/red]: a round is recorded once and never rewritten")
        raise typer.Exit(1)
    recorded = json.loads(choice.read_text(encoding="utf-8"))
    update: dict[str, object] = {}
    if thinking is not None:
        update["thinking"] = thinking
    if effort is not None:
        update["effort"] = effort
    settings = Settings.model_validate(recorded["settings"] | update)
    annotations, targets = _gold_a_targets(data_dir, cache)
    runner = _runner(store, ledger)
    text = instruction.read_text(encoding="utf-8")
    readings = runner.run(settings, text, targets)
    keys = {r.key for r in runner.requests(settings, text, targets)}
    run = _summary(settings.model, targets, readings, annotations, _spent(runner.ledger, keys))
    digest = hashlib.sha256(text.encode()).hexdigest()
    header = (
        f"# Development round {round_number} on gold set A\n\nGenerated by `docfraud llm "
        "develop`; do not edit by hand.\n\n"
        f"Model `{settings.model}`, thinking {settings.thinking}, effort {settings.effort}; "
        f"instruction `{instruction}` (SHA-256 `{digest[:12]}`). Gold set A is the development "
        "set, so these numbers are optimistic (ADR-0025).\n\n"
    )
    manifest = _gold_a_manifest()
    alarms = _false_alarms(targets, readings, annotations, frozenset(manifest.authentic))
    reports.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_round(header, run, frozenset(manifest.forged), alarms, len(manifest.authentic)),
        encoding="utf-8",
    )
    record = RoundRecord(
        round=round_number,
        settings=settings,
        instruction=str(instruction),
        instruction_sha256=digest,
        receipts=len(targets) // len(TRACKS),
        exact={
            track: sum(s.rule_inputs_exact for s in scores) for track, scores in run.scores.items()
        },
    )
    output.with_suffix(".json").write_text(
        record.model_dump_json(indent=2) + "\n", encoding="utf-8"
    )
    console.print(f"[green]written[/green] {output}")


@app.command("freeze")
def llm_freeze(
    reports: Annotated[Path, typer.Option(help="Directory of the round reports.")] = ROUNDS,
    frozen: Annotated[Path, typer.Option(help="Where to record the frozen version.")] = FROZEN,
) -> None:
    """Freeze the best development round, once, from a clean commit (ADR-0025, decision 3)."""
    if frozen.exists():
        console.print(f"[red]{frozen} exists[/red]: the version is frozen once")
        raise typer.Exit(1)
    require_clean_commit()
    records = [
        RoundRecord.model_validate_json(path.read_text(encoding="utf-8"))
        for path in sorted(reports.glob("round-*.json"))
    ]
    best = best_round(records)
    text = Path(best.instruction).read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != best.instruction_sha256:
        console.print(f"[red]{best.instruction} changed after round {best.round}[/red]")
        raise typer.Exit(1)
    version = FrozenVersion(
        round=best.round,
        settings=best.settings,
        instruction=best.instruction,
        instruction_sha256=best.instruction_sha256,
        commit=str(code_version().commit),
    )
    frozen.parent.mkdir(parents=True, exist_ok=True)
    frozen.write_text(version.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(
        f"[green]frozen[/green] round {best.round} ({best.exact_total} exact readings of "
        f"{best.receipts * len(best.exact)}) at {version.commit[:12]}"
    )


@app.command("stability")
def llm_stability(  # noqa: PLR0913, PLR0917 - one option per input and output
    data_dir: DataDir = Path("data"),
    cache: Cache = OCR_CACHE,
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    frozen: Frozen = FROZEN,
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = Path(
        "reports/llm/stability.md"
    ),
) -> None:
    """Read gold set A a second time with the frozen version and compare (ADR-0025, decision 3)."""
    version = _frozen(frozen)
    text = Path(version.instruction).read_text(encoding="utf-8")
    _, targets = _gold_a_targets(data_dir, cache)
    runner = _runner(store, ledger)
    first = runner.run(version.settings, text, targets)
    second = runner.run(version.settings, text, targets, repeat=1)
    rows: list[tuple[str, int, int, int]] = []
    for source, label in TRACKS.items():
        pairs = [
            (a.fields, b.fields)
            for t, a, b in zip(targets, first, second, strict=True)
            if t.source is source
        ]
        same_inputs = sum(score_document("", a, b).rule_inputs_exact for a, b in pairs)
        rows.append((label, len(pairs), same_inputs, sum(a == b for a, b in pairs)))
    lines = [
        "# Stability of the frozen version on gold set A",
        "",
        "Generated by `docfraud llm stability`; do not edit by hand.",
        "",
        f"Round {version.round}, model `{version.settings.model}`, read twice. The models do not "
        "accept a sampling temperature, so two readings of the same receipt can differ.",
        "",
        "| Text source | Readings | Same rule inputs | Same fields |",
        "|---|---|---|---|",
        *(f"| {label} | {n} | {inputs} | {fields} |" for label, n, inputs, fields in rows),
        "",
    ]
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text("\n".join(lines), encoding="utf-8")
    console.print(f"[green]written[/green] {output}")


def _runner(store: Path, ledger: Path) -> Runner:
    """A runner with Claude that keeps its answers in ``store`` and its batches beside them."""
    return Runner(ClaudeProvider(pending=store / "batches"), ResponseStore(store), Ledger(ledger))


def _frozen(path: Path) -> FrozenVersion:
    """The frozen version, after checking that its instruction is unchanged.

    Raises:
        typer.Exit: If nothing is frozen yet or the instruction changed.
    """
    if not path.exists():
        console.print(f"[red]{path} does not exist[/red]: freeze a development round first")
        raise typer.Exit(1)
    version = FrozenVersion.model_validate_json(path.read_text(encoding="utf-8"))
    text = Path(version.instruction).read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != version.instruction_sha256:
        console.print(f"[red]{version.instruction} changed after it was frozen[/red]")
        raise typer.Exit(1)
    return version


@app.command("measure-b")
def llm_measure_b(  # noqa: PLR0913, PLR0917 - one option per input and output
    data_dir: DataDir = Path("data"),
    cache: Cache = OCR_CACHE,
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    frozen: Frozen = FROZEN,
    measurement: MeasurementPath = GOLD_B_DIR / "measurement.json",
    questions: QuestionsPath = GOLD_B_DIR / "questions.json",
) -> None:
    """Read gold set B once with the frozen version and parser v2, and list what is in dispute.

    Prints only how many questions there are: no metric on gold set B is computed before every
    question has a final answer (ADR-0025, decision 4). Requests are sent one by one, so that
    the time to each answer is measured (decision 6). A run that leaves a request unanswered
    keeps nothing; running it again sends only what is missing.
    """
    if measurement.exists():
        console.print(f"[red]{measurement} exists[/red]: gold set B is read once")
        raise typer.Exit(1)
    require_clean_commit()
    version = _frozen(frozen)
    require_unchanged(version.commit, READER_CODE, "the reader changed after it was frozen")
    manifest = load_manifest(GOLD_B)
    gold = load_annotations(GOLD_B.parent, manifest)
    if set(gold) != manifest.doc_ids:
        console.print("[red]gold set B is not fully annotated[/red]")
        raise typer.Exit(1)
    documents = sorted(
        (d for d in findit_documents(data_dir) if d.doc_id in manifest.doc_ids),
        key=lambda d: d.doc_id,
    )
    allowed = {d.doc_id for d in documents if d.split is Split.TRAIN} - _gold_a_manifest().doc_ids
    check_documents([d.doc_id for d in documents], allowed, "the measurement on gold set B")
    ocr = cached_ocr(cache)
    reads = [
        (document.doc_id, source, read_document(document, ocr, source))
        for document in documents
        for source in TRACKS
    ]
    targets = [target(doc_id, source, reading.rows) for doc_id, source, reading in reads]
    runner = _runner(store, ledger)
    text = Path(version.instruction).read_text(encoding="utf-8")
    llm = runner.run(version.settings, text, targets, batch=False)
    requests = runner.requests(version.settings, text, targets)
    missing = runner.store.missing(requests)
    if missing:
        console.print(
            f"[red]{len(missing)} of {len(requests)} requests were not answered[/red]; nothing "
            "is kept. Run the command again to send them."
        )
        raise typer.Exit(1)
    outputs = [
        ReaderOutput(
            doc_id=doc_id,
            reader=LLM,
            source=source,
            extraction=answered.extraction,
            rejections=answered.rejections,
            seconds=_seconds(runner.store, request),
        )
        for (doc_id, source, _), answered, request in zip(reads, llm, requests, strict=True)
    ] + [
        ReaderOutput(
            doc_id=doc_id,
            reader=PARSER,
            source=source,
            extraction=Extraction(fields=reading.fields, sources=reading.sources),
        )
        for doc_id, source, reading in reads
    ]
    found = questions_on({doc_id: a.fields for doc_id, a in gold.items()}, outputs)
    save_measurement(
        Measurement(
            version=version,
            commit=str(code_version().commit),
            measured_at=datetime.now(UTC),
            receipts=len(documents),
            cost_usd=_spent(runner.ledger, {r.key for r in requests}),
            readings=tuple(outputs),
        ),
        measurement,
    )
    save_questions(found, questions)
    kinds = Counter(q.kind for q in found)
    console.print(
        f"[green]read[/green] {len(documents)} receipts of gold set B; {len(found)} questions on "
        f"{len({q.doc_id for q in found})} receipts: "
        + ", ".join(f"{kind.value} {kinds[kind]}" for kind in kinds)
    )
    console.print("no metric is computed before the adjudication is complete")


@app.command("adjudicate")
def llm_adjudicate(  # noqa: PLR0913, PLR0917 - one option per setting
    reviewer: Reviewer = "emre",
    data_dir: DataDir = Path("data"),
    questions: QuestionsPath = GOLD_B_DIR / "questions.json",
    log: AdjudicationPath = ADJUDICATION,
    port: Annotated[int, typer.Option(help="Local port; 0 picks a free one.")] = 8773,
    browser: Annotated[bool, typer.Option(help="Open the page in the browser.")] = True,
    language: Annotated[
        PageLanguage, typer.Option(help="Language of the page; answers are recorded as typed.")
    ] = PageLanguage.TURKISH,
) -> None:
    """Serve the adjudication of gold set B on 127.0.0.1 until interrupted (Ctrl+C).

    Every answer is saved at once, so the page can be closed and opened again at any time. The
    page needs a clean commit, which every answer records; the answers so far are committed
    before the page is opened again.
    """
    if not questions.exists():
        console.print(f"[red]{questions} does not exist[/red]: run docfraud llm measure-b first")
        raise typer.Exit(1)
    require_clean_commit()
    asked = in_order(load_questions(questions))
    receipts = {q.doc_id for q in asked}
    images = {d.doc_id: d.image_path for d in findit_documents(data_dir) if d.doc_id in receipts}
    try:
        page = AdjudicationApp(
            asked,
            lambda doc_id: load_grayscale(images[doc_id]),
            log,
            reviewer=reviewer,
            commit=code_version().commit,
            language=language,
        )
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    server = make_server(page, port)
    url = f"http://{HOST}:{server.server_port}/"
    done = len(asked) - len(unanswered(asked, page.log.answers))
    console.print(
        f"adjudication of gold set B for [bold]{reviewer}[/bold] at {url}: {done} of "
        f"{len(asked)} questions done"
    )
    console.print(f"answers are saved to {log} (Ctrl+C to stop)")
    if browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("stopped; run the same command to continue")
    finally:
        server.server_close()


@app.command("evaluate-b")
def llm_evaluate_b(
    reviewer: Reviewer = "emre",
    measurement: MeasurementPath = GOLD_B_DIR / "measurement.json",
    questions: QuestionsPath = GOLD_B_DIR / "questions.json",
    log: AdjudicationPath = ADJUDICATION,
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = GOLD_B_REPORT,
) -> None:
    """Decide per text source whether the LLM reader replaces parser v2 (ADR-0025, decision 5).

    Runs once, from a clean commit, after every question of the adjudication has a final answer;
    gold set B is evaluated with the answers in place and its annotation is left unchanged.
    """
    if output.exists():
        console.print(f"[red]{output} exists[/red]: the evaluation of gold set B runs once")
        raise typer.Exit(1)
    if not measurement.exists() or not questions.exists():
        console.print("[red]gold set B has not been read[/red]: run docfraud llm measure-b first")
        raise typer.Exit(1)
    asked = load_questions(questions)
    try:
        answers = load_log(log, reviewer).answers
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    left = unanswered(asked, answers)
    if left:
        console.print(
            f"[red]{len(left)} of {len(asked)} questions have no final answer[/red]: finish "
            "the adjudication first (docfraud llm adjudicate)"
        )
        raise typer.Exit(1)
    require_clean_commit()
    measured = load_measurement(measurement)
    manifest = load_manifest(GOLD_B)
    original = {d: a.fields for d, a in load_annotations(GOLD_B.parent, manifest).items()}
    read: dict[tuple[str, TextSource], dict[str, ReceiptFields]] = defaultdict(dict)
    for reading in measured.readings:
        read[reading.reader, reading.source][reading.doc_id] = reading.extraction.fields
    if any(set(read[reader, source]) != set(original) for reader in READERS for source in TRACKS):
        console.print("[red]the reading does not cover the receipts of gold set B[/red]")
        raise typer.Exit(1)
    gold = adjudicated_gold(original, asked, answers)
    scores = {
        label: {
            reader: [score_document(d, gold[d], read[reader, source][d]) for d in sorted(gold)]
            for reader in READERS
        }
        for source, label in TRACKS.items()
    }
    decisions = [
        decide_reader(
            label,
            [s.rule_inputs_exact for s in by_reader[PARSER]],
            [s.rule_inputs_exact for s in by_reader[LLM]],
        )
        for label, by_reader in scores.items()
    ]
    authentic = frozenset(manifest.authentic)
    alarms = {
        "gold set B, adjudicated": dict.fromkeys(TRACKS.values(), false_alarms(gold, authentic))
    }
    alarms |= {
        reader: {label: false_alarms(read[reader, s], authentic) for s, label in TRACKS.items()}
        for reader in READERS
    }
    llm_readings = {
        label: [r for r in measured.readings if r.reader == LLM and r.source is source]
        for source, label in TRACKS.items()
    }
    rejections = {label: sum(len(r.rejections) for r in got) for label, got in llm_readings.items()}
    latencies = {label: latency([r.seconds for r in got]) for label, got in llm_readings.items()}
    names = [GOLD, *(reading_name(r, s) for r in READERS for s in TextSource)]
    version = measured.version
    header = (
        "# Gold set B: the LLM reader against parser v2\n\nGenerated by `docfraud llm "
        "evaluate-b`; do not edit by hand.\n\n"
        f"The frozen version (development round {version.round}, model "
        f"`{version.settings.model}`, thinking {version.settings.thinking}, effort "
        f"{version.settings.effort}, instruction `{version.instruction}` with SHA-256 "
        f"`{version.instruction_sha256[:12]}`, frozen at commit `{version.commit[:12]}`) read the "
        f"{measured.receipts} receipts of gold set B ({len(manifest.authentic)} authentic, "
        f"{len(manifest.forged)} forged) once from the transcripts (T1) and from our OCR (T2), "
        f"at commit `{measured.commit[:12]}` on {measured.measured_at:%Y-%m-%d}; parser v2 read "
        f"the same rows. It cost {measured.cost_usd:.4f} USD, "
        f"{measured.cost_usd / measured.receipts:.4f} USD per receipt for both text sources. "
        f"The gold set is the annotation with the owner's final answers to {len(asked)} "
        "questions in place ([ADR-0025](../../docs/decisions/0025-llm-extraction-"
        "preregistration.md), decision 4; [ADR-0026](../../docs/decisions/0026-adjudication-"
        "of-items-and-tax-rows.md)); the annotation itself is unchanged. Evaluated at commit "
        f"`{str(code_version().commit)[:12]}`.\n\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_gold_b(
            header,
            decisions=decisions,
            scores=scores,
            counts=adjudication_counts(asked, answers),
            readings=names,
            false_alarms=alarms,
            rejections=rejections,
            latencies=latencies,
            forged=frozenset(manifest.forged),
            authentic=len(manifest.authentic),
        ),
        encoding="utf-8",
    )
    verdicts = "; ".join(f"{d.source}: {'LLM' if d.passes else 'parser v2'}" for d in decisions)
    console.print(f"[green]written[/green] {output}; {verdicts}")


@app.command("false-alarms")
def llm_false_alarms(  # noqa: PLR0913, PLR0917 - one option per input and output
    data_dir: DataDir = Path("data"),
    cache: Cache = OCR_CACHE,
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    frozen: Frozen = FROZEN,
    measurement: MeasurementPath = FALSE_ALARMS_DIR / "measurement.json",
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = FALSE_ALARMS_REPORT,
) -> None:
    """Read the training receipts outside both gold sets once and apply ADR-0027.

    The frozen version, through the batch service, and parser v2 read every receipt from both
    text sources; the report decides per text source whether the LLM reader replaces parser v2.
    A run that leaves a request unanswered keeps nothing; running it again sends only what is
    missing.
    """
    if output.exists():
        console.print(f"[red]{output} exists[/red]: the test of ADR-0027 runs once")
        raise typer.Exit(1)
    require_clean_commit()
    version = _frozen(frozen)
    require_unchanged(version.commit, READER_CODE, "the reader changed after it was frozen")
    require_unchanged(FIXED_AT, RULES_CODE, "the accounting rules changed after ADR-0027")
    excluded = _gold_a_manifest().doc_ids | load_manifest(GOLD_B).doc_ids
    found = [d for d in findit_documents(data_dir) if d.split is Split.TRAIN]
    documents = sorted((d for d in found if d.doc_id not in excluded), key=lambda d: d.doc_id)
    allowed = {d.doc_id for d in found} - excluded
    check_documents([d.doc_id for d in documents], allowed, "the test of ADR-0027")
    if any(d.label is None for d in documents):
        console.print("[red]every receipt must be labelled authentic or forged[/red]")
        raise typer.Exit(1)
    forged = {d.doc_id for d in documents if d.label is Label.FORGED}
    ocr = cached_ocr(cache)
    reads = [
        (document.doc_id, source, read_document(document, ocr, source))
        for document in documents
        for source in TRACKS
    ]
    targets = [target(doc_id, source, reading.rows) for doc_id, source, reading in reads]
    runner = _runner(store, ledger)
    text = Path(version.instruction).read_text(encoding="utf-8")
    llm = runner.run(version.settings, text, targets)
    requests = runner.requests(version.settings, text, targets)
    missing = runner.store.missing(requests)
    if missing:
        console.print(
            f"[red]{len(missing)} of {len(requests)} requests were not answered[/red]; nothing "
            "is kept. Run the command again to send them."
        )
        raise typer.Exit(1)
    outputs = [
        ReaderOutput(
            doc_id=doc_id,
            reader=LLM,
            source=source,
            extraction=answered.extraction,
            rejections=answered.rejections,
            seconds=_seconds(runner.store, request),
        )
        for (doc_id, source, _), answered, request in zip(reads, llm, requests, strict=True)
    ] + [
        ReaderOutput(
            doc_id=doc_id,
            reader=PARSER,
            source=source,
            extraction=Extraction(fields=reading.fields, sources=reading.sources),
        )
        for doc_id, source, reading in reads
    ]
    measured = Measurement(
        version=version,
        commit=str(code_version().commit),
        measured_at=datetime.now(UTC),
        receipts=len(documents),
        cost_usd=_spent(runner.ledger, {r.key for r in requests}),
        readings=tuple(outputs),
    )
    save_measurement(measured, measurement)
    labels = {d.doc_id: d.doc_id in forged for d in documents}
    results = {track: judge(track, source, labels, outputs) for source, track in TRACKS.items()}
    cost = measured.cost_usd
    header = (
        "# The LLM reader judged by the accounting rules' false alarms\n\nGenerated by `docfraud "
        "llm false-alarms`; do not edit by hand.\n\n"
        f"The frozen version (development round {version.round}, model "
        f"`{version.settings.model}`, thinking {version.settings.thinking}, effort "
        f"{version.settings.effort}, instruction `{version.instruction}` with SHA-256 "
        f"`{version.instruction_sha256[:12]}`, frozen at commit `{version.commit[:12]}`) and "
        f"parser v2 read the {len(documents)} receipts of the training split outside gold sets A "
        f"and B ({len(documents) - len(forged)} authentic, {len(forged)} forged) once from the "
        f"transcripts (T1) and from our OCR (T2), at commit `{measured.commit[:12]}` on "
        f"{measured.measured_at:%Y-%m-%d}. The LLM's requests cost {cost:.4f} USD, "
        f"{cost / len(documents):.4f} USD per receipt for both text sources. The rule was fixed "
        "in [ADR-0027](../../docs/decisions/0027-llm-reader-judged-by-false-alarms.md) at commit "
        f"`{FIXED_AT[:12]}`, before the LLM read any of these receipts; on gold set B the LLM "
        "reader did not pass ([report](gold-b.md)).\n\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(
        render_false_alarms(
            header, results, authentic=len(documents) - len(forged), forged=len(forged)
        ),
        encoding="utf-8",
    )
    verdicts = "; ".join(
        f"{track}: {'LLM' if r.decision.passes else 'parser v2'}" for track, r in results.items()
    )
    console.print(f"[green]written[/green] {output}; {verdicts}")


@app.command("read-validation")
def llm_read_validation(  # noqa: PLR0913, PLR0917 - one option per input and output
    data_dir: DataDir = Path("data"),
    cache: Cache = OCR_CACHE,
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    frozen: Frozen = FROZEN,
    record: Annotated[Path, typer.Option(help="Where to record the reading.")] = VALIDATION_READING,
) -> None:
    """Read the validation receipts once with the frozen version, for Phase 5 (ADR-0027).

    The frozen version reads every validation receipt from both text sources through the batch
    service, and the answers are kept in the store, from which Phase 5 takes the fields. Nothing
    is measured: the record names the version, the commit and the cost, and no field, label or
    score of a validation receipt is shown. A run that leaves a request unanswered records
    nothing; running it again sends only what is missing.
    """
    if record.exists():
        console.print(f"[red]{record} exists[/red]: the validation split is read once")
        raise typer.Exit(1)
    require_clean_commit()
    version = _frozen(frozen)
    require_unchanged(version.commit, READER_CODE, "the reader changed after it was frozen")
    documents = sorted(select(findit_documents(data_dir), Split.VAL), key=lambda d: d.doc_id)
    targets = _targets(documents, cached_ocr(cache))
    runner = _runner(store, ledger)
    text = Path(version.instruction).read_text(encoding="utf-8")
    runner.run(version.settings, text, targets)
    requests = runner.requests(version.settings, text, targets)
    missing = runner.store.missing(requests)
    if missing:
        console.print(
            f"[red]{len(missing)} of {len(requests)} requests were not answered[/red]; nothing "
            "is recorded. Run the command again to send them."
        )
        raise typer.Exit(1)
    reader = FrozenReader(version, text, runner.store)
    readings = [reader.read(t) for t in targets]  # as Phase 5 will take them
    reading = SplitReading(
        split=Split.VAL,
        version=version,
        commit=str(code_version().commit),
        read_at=datetime.now(UTC),
        receipts=len(documents),
        requests=len(requests),
        cost_usd=_spent(runner.ledger, {r.key for r in requests}),
        without_answer=sum(any(r.field == "answer" for r in got.rejections) for got in readings),
    )
    save_split_reading(reading, record)
    console.print(
        f"[green]read[/green] {reading.receipts} validation receipts from both text sources: "
        f"{reading.requests} requests, {reading.cost_usd:.4f} USD, {reading.without_answer} "
        "without an answer"
    )
    console.print("nothing is measured; Phase 5 takes the fields from the store")


def _targets(documents: Sequence[Document], ocr: CachedOcr) -> list[Target]:
    """What the reader reads of each receipt: its rows from both text sources, in order."""
    return [
        target(document.doc_id, source, read_document(document, ocr, source).rows)
        for document in documents
        for source in TRACKS
    ]


@app.command("read-test")
def llm_read_test(  # noqa: PLR0913, PLR0917 - one option per input and output
    data_dir: DataDir = Path("data"),
    cache: Cache = OCR_CACHE,
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    frozen: Frozen = FROZEN,
    runs: Annotated[Path, typer.Option(help="Where reported run records are kept.")] = (
        REPORTED_RUNS
    ),
) -> None:
    """Read the test receipts once with the frozen version, for Phase 7 (ADR-0034).

    Runs only at the freeze tag, from a clean commit, and only if today's code builds every
    request with which the frozen version read the validation split exactly as it was stored:
    the reader's code changed after it was frozen (Phase 6b), so this, not a check that no file
    changed, shows that it reads as it did. The answers go to the store, from which the test
    inputs take the fields; the run record, which records the unlock, keeps aggregates only. A
    run that leaves a request unanswered records nothing; running it again sends only what is
    missing.
    """
    records = JsonRunStore(runs)
    if records.records(TEST_READING_RUN):
        console.print(f"[red]refusing to run again:[/red] a {TEST_READING_RUN} record exists")
        raise typer.Exit(1)
    require_clean_commit()
    require_freeze_tag()
    version = _frozen(frozen)
    text = Path(version.instruction).read_text(encoding="utf-8")
    documents = findit_documents(data_dir)
    ocr = cached_ocr(cache)
    runner = _runner(store, ledger)
    validation = sorted(select(documents, Split.VAL), key=lambda d: d.doc_id)
    stored = runner.requests(version.settings, text, _targets(validation, ocr))
    changed = runner.store.missing(stored)
    if changed:
        console.print(
            f"[red]{len(changed)} of {len(stored)} requests of the validation reading are not "
            "the stored ones[/red]: the reader no longer reads as it was frozen"
        )
        raise typer.Exit(1)
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        TEST_READING_RUN,
        store=records,
        config={"decision": "ADR-0034", "frozen": version.model_dump(mode="json")},
        seeds={},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        test = sorted(
            select(documents, Split.TEST, unlock_test=True, reason=TEST_UNLOCK),
            key=lambda d: d.doc_id,
        )
        targets = _targets(test, ocr)
        runner.run(version.settings, text, targets)
        requests = runner.requests(version.settings, text, targets)
        missing = runner.store.missing(requests)
        if missing:
            console.print(
                f"[red]{len(missing)} of {len(requests)} requests were not answered[/red]; "
                "nothing is recorded. Run the command again to send them."
            )
            raise typer.Exit(1)
        readings = [FrozenReader(version, text, runner.store).read(t) for t in targets]
        current.log("receipts", len(test))
        current.log("requests", len(requests))
        current.log("validation_requests_found", len(stored))
        current.log("cost_usd", f"{_spent(runner.ledger, {r.key for r in requests}):.4f}")
        current.log(
            "without_answer",
            sum(any(r.field == "answer" for r in got.rejections) for got in readings),
        )
    console.print(
        f"[green]read[/green] {len(test)} test receipts from both text sources: "
        f"{len(requests)} requests, {current.results['cost_usd']} USD"
    )


@app.command("spending")
def llm_spending(ledger: LedgerPath = LEDGER) -> None:
    """Show what the LLM reader has cost so far and what is left within the limit."""
    book = Ledger(ledger)
    spent = book.spent()
    console.print(
        f"spent {spent:.4f} USD on {len(book.entries())} responses; limit {book.limit} USD; "
        f"left {book.limit - spent:.4f} USD"
    )


def _gold_a_manifest() -> GoldManifest:
    return load_manifest(GOLD_A)


def _gold_a_targets(data_dir: Path, cache: Path) -> tuple[dict[str, GoldAnnotation], list[Target]]:
    """The labelled rows of gold set A from both text sources, after checking the receipts."""
    manifest = _gold_a_manifest()
    annotations = load_annotations(GOLD_A.parent, manifest)
    excluded = load_manifest(GOLD_B).doc_ids if GOLD_B.exists() else frozenset()
    documents = sorted(
        (d for d in findit_documents(data_dir) if d.doc_id in manifest.doc_ids),
        key=lambda d: d.doc_id,
    )
    allowed = {d.doc_id for d in documents if d.split is Split.TRAIN} - set(excluded)
    check_documents([d.doc_id for d in documents], allowed, "development on gold set A")
    ocr = cached_ocr(cache)
    targets = [
        target(document.doc_id, source, read_document(document, ocr, source).rows)
        for document in documents
        for source in TRACKS
    ]
    return annotations, targets


def _summary(
    name: str,
    targets: Sequence[Target],
    readings: Sequence[LlmReading] | None,
    annotations: dict[str, GoldAnnotation],
    cost: Decimal | None,
) -> ReaderRun:
    """Scores per text source; without readings, the parser reads the same rows."""
    scores: dict[str, list[DocumentScore]] = {label: [] for label in TRACKS.values()}
    rejections = dict.fromkeys(TRACKS.values(), 0)
    for n, item in enumerate(targets):
        label = TRACKS[item.source]
        fields = parse_receipt(item.rows) if readings is None else readings[n].fields
        if readings is not None:
            rejections[label] += len(readings[n].rejections)
        scores[label].append(score_document(item.doc_id, annotations[item.doc_id].fields, fields))
    return ReaderRun(name, scores, None if readings is None else rejections, cost)


def _false_alarms(
    targets: Sequence[Target],
    readings: Sequence[LlmReading],
    annotations: dict[str, GoldAnnotation],
    authentic: frozenset[str],
) -> dict[str, dict[str, int]]:
    """False alarms of the accounting rules on gold set A: gold fields, parser v2 and the LLM."""
    read: dict[tuple[str, str], dict[str, ReceiptFields]] = defaultdict(dict)
    for item, reading in zip(targets, readings, strict=True):
        label = TRACKS[item.source]
        read[PARSER, label][item.doc_id] = parse_receipt(item.rows)
        read[LLM, label][item.doc_id] = reading.fields
    gold = {doc_id: a.fields for doc_id, a in annotations.items()}
    return {"gold set A": dict.fromkeys(TRACKS.values(), false_alarms(gold, authentic))} | {
        reader: {label: false_alarms(read[reader, label], authentic) for label in TRACKS.values()}
        for reader in READERS
    }


def _per_receipt(targets: Sequence[Target], run: ReaderRun) -> list[tuple[bool, ...]]:
    """For each receipt, whether each text source has all rule inputs exact."""
    by_track = {
        label: {s.doc_id: s.rule_inputs_exact for s in run.scores[label]}
        for label in TRACKS.values()
    }
    receipts = list(dict.fromkeys(t.doc_id for t in targets))
    return [tuple(by_track[label][doc_id] for label in TRACKS.values()) for doc_id in receipts]


def _seconds(store: ResponseStore, request: Request) -> float | None:
    """How long the stored answer to a request took, if it was timed."""
    response = store.get(request.key)
    return None if response is None else response.seconds


def _spent(ledger: Ledger, keys: set[str]) -> Decimal:
    """What the responses to the given requests cost."""
    return sum(
        (Decimal(str(e["cost_usd"])) for e in ledger.entries() if e["key"] in keys), Decimal(0)
    )
