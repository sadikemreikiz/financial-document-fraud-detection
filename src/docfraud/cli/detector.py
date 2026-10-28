"""``docfraud detector``: the standalone LLM detector's development on the design fold (ADR-0031).

Every command reads design-fold receipts only, checked before any request is built; sends only
requests without a stored answer; and stops before the spending limit. ``develop`` runs one
version of the instruction for both rows (``--pilot``: five receipts, sent one by one, to check
the answer form); ``freeze`` keeps a version by the rule fixed in advance; ``latency`` times the
frozen version on 30 receipts sent one by one; ``stronger-model`` runs the frozen main row once on
Claude Opus 5.5 (ADR-0032).
"""

import hashlib
import json
from collections.abc import Sequence
from decimal import Decimal
from pathlib import Path
from typing import Annotated

import numpy as np
import typer

from docfraud.cli.common import (
    OCR_CACHE,
    DataDir,
    cached_ocr,
    console,
    findit_documents,
    require_clean_commit,
    require_unchanged,
)
from docfraud.cli.llm import LEDGER, STORE, Cache, LedgerPath, Store, _runner
from docfraud.data.models import Document, Label
from docfraud.data.normalize import load_grayscale
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS
from docfraud.evaluation.runs import code_version
from docfraud.llm.cost import cost
from docfraud.llm.detector import (
    SONNET,
    DetectorRow,
    page_image,
    read_judgement,
    request,
)
from docfraud.llm.detector_rounds import (
    DETECTOR_CODE,
    OPUS,
    VERSIONS,
    DevelopmentRound,
    FrozenDetector,
    StrongerModelCheck,
    clearly_better,
    kept_version,
    load_rounds,
    row_result,
    save_round,
)
from docfraud.llm.provider import Request, Response, Settings
from docfraud.llm.run import check_documents
from docfraud.reading.ocr import CachedOcr
from docfraud.reading.pipeline import TextSource, read_document

app = typer.Typer(help="The standalone LLM detector: development on the design fold.")

INSTRUCTIONS = Path("src/docfraud/llm/instructions")
ROUNDS = Path("reports/llm/detector")
PILOT_RECEIPTS = 5
LATENCY_RECEIPTS = 30
STRONGER_MODEL = "stronger-model.json"

Rounds = Annotated[Path, typer.Option("--rounds", help="Where development rounds are kept.")]


def instruction_path(version: int) -> Path:
    """Where version ``version`` of the detector's instruction is kept."""
    return INSTRUCTIONS / f"detector-v{version}.md"


def _design_fold(data_dir: Path) -> list[Document]:
    """The design-fold receipts in identifier order, checked before any request is built."""
    documents = sorted(
        (d for d in findit_documents(data_dir) if d.doc_id in DESIGN_FOLD_IDS),
        key=lambda d: d.doc_id,
    )
    check_documents([d.doc_id for d in documents], DESIGN_FOLD_IDS, "detector development")
    return documents


def requests(
    documents: Sequence[Document],
    ocr: CachedOcr,
    settings: Settings,
    instruction: str,
    *,
    repeat: int = 0,
) -> dict[DetectorRow, list[Request]]:
    """Both rows' requests for every receipt, in order."""
    found: dict[DetectorRow, list[Request]] = {row: [] for row in DetectorRow}
    for document in documents:
        page = page_image(load_grayscale(document.image_path))
        ocr_rows = read_document(document, ocr, TextSource.OCR).rows
        transcript = read_document(document, ocr, TextSource.TRANSCRIPT).rows
        for row, rows, image in (
            (DetectorRow.IMAGE_AND_OCR, ocr_rows, page),
            (DetectorRow.TRANSCRIPT, transcript, None),
        ):
            built = request(row, settings, instruction, rows, image)
            found[row].append(built.model_copy(update={"repeat": repeat}))
    return found


def _spent(responses: Sequence[Response | None], model: str) -> Decimal:
    return sum(
        (cost(r.usage, model, batch=r.batch) for r in responses if r is not None), Decimal(0)
    )


@app.command("develop")
def detector_develop(  # noqa: PLR0913, PLR0917 - one option per input and output
    version: Annotated[int, typer.Option(help="The version of the instruction, 1 or 2.")],
    pilot: Annotated[bool, typer.Option(help="Five receipts, one by one, to check the form.")] = (
        False
    ),
    data_dir: DataDir = Path("data"),
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    cache: Cache = OCR_CACHE,
    rounds: Rounds = ROUNDS,
) -> None:
    """Run one version of the instruction on the design fold, for both rows (ADR-0031, 2).

    A full run needs a clean commit, which its record names, and writes the round's record.
    """
    if version not in VERSIONS:
        console.print(f"[red]development allows versions {VERSIONS} only[/red]")
        raise typer.Exit(1)
    if (rounds / "frozen.json").exists():
        console.print("[red]the detector is frozen[/red]: development has ended")
        raise typer.Exit(1)
    if version > 1 and 1 not in load_rounds(rounds):
        console.print("[red]run version 1 on the whole design fold first[/red]")
        raise typer.Exit(1)
    if not pilot:
        require_clean_commit()
    path = instruction_path(version)
    text = path.read_text(encoding="utf-8")
    documents = _design_fold(data_dir)[: PILOT_RECEIPTS if pilot else None]
    built = requests(documents, cached_ocr(cache), SONNET, text)
    runner = _runner(store, ledger)
    answered = {row: runner.answer(built[row], batch=not pilot) for row in DetectorRow}
    readings = {row: [read_judgement(r) for r in answered[row]] for row in DetectorRow}
    if pilot:
        for row in DetectorRow:
            for document, reading in zip(documents, readings[row], strict=True):
                shown = reading.failure or f"{reading.suspicion}: {'; '.join(reading.reasons)}"
                console.print(f"{row.value} · {document.doc_id} · {shown}")
        return
    forged = [d.label is Label.FORGED for d in documents]
    record = DevelopmentRound(
        version=version,
        settings=SONNET,
        instruction=str(path),
        instruction_sha256=hashlib.sha256(text.encode()).hexdigest(),
        commit=code_version().commit,
        rows={
            row.value: row_result(
                forged, readings[row], f"{_spent(answered[row], SONNET.model):.4f}"
            )
            for row in DetectorRow
        },
    )
    save_round(record, rounds / f"round-v{version}.json")
    for name, result in record.rows.items():
        console.print(
            f"{name}: AP {result.average_precision:.3f}, ROC-AUC {result.roc_auc:.3f}, "
            f"{result.failures} failures, {result.cost_usd} USD"
        )


def _frozen(rounds: Path) -> tuple[FrozenDetector, str]:
    """The frozen version and its instruction, both checked unchanged since the freeze.

    The instruction is checked by its hash, the code that builds its requests by git.

    Raises:
        typer.Exit: If nothing is frozen, or something changed after it was frozen.
    """
    path = rounds / "frozen.json"
    if not path.exists():
        console.print(f"[red]{path} does not exist[/red]: freeze a version first")
        raise typer.Exit(1)
    frozen = FrozenDetector.model_validate_json(path.read_text(encoding="utf-8"))
    text = Path(frozen.instruction).read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != frozen.instruction_sha256:
        console.print(f"[red]{frozen.instruction} changed after it was frozen[/red]")
        raise typer.Exit(1)
    require_unchanged(frozen.commit, DETECTOR_CODE, "the detector changed after it was frozen")
    return frozen, text


@app.command("freeze")
def detector_freeze(rounds: Rounds = ROUNDS) -> None:
    """Freeze the version the rule keeps, at a clean commit (ADR-0031, section 2)."""
    path = rounds / "frozen.json"
    if path.exists():
        console.print(f"[red]{path} exists[/red]: the detector is frozen once")
        raise typer.Exit(1)
    require_clean_commit()
    developed = load_rounds(rounds)
    try:
        version = kept_version(developed)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    kept = developed[version]
    text = Path(kept.instruction).read_text(encoding="utf-8")
    if hashlib.sha256(text.encode()).hexdigest() != kept.instruction_sha256:
        console.print(f"[red]{kept.instruction} changed after version {version} ran[/red]")
        raise typer.Exit(1)
    frozen = FrozenDetector(
        version=version,
        settings=kept.settings,
        instruction=kept.instruction,
        instruction_sha256=kept.instruction_sha256,
        commit=str(code_version().commit),
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(frozen.model_dump_json(indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]frozen[/green] version {version} at {frozen.commit[:7]}")


@app.command("latency")
def detector_latency(
    data_dir: DataDir = Path("data"),
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    cache: Cache = OCR_CACHE,
    rounds: Rounds = ROUNDS,
) -> None:
    """Time the frozen version on 30 design-fold receipts sent one by one (ADR-0031, 2).

    The requests are second readings, so that none is answered from the batched run.
    """
    frozen, text = _frozen(rounds)
    documents = _design_fold(data_dir)[:LATENCY_RECEIPTS]
    built = requests(documents, cached_ocr(cache), frozen.settings, text, repeat=1)
    runner = _runner(store, ledger)
    found: dict[str, object] = {
        "version": frozen.version,
        "commit": code_version().commit,
        "receipts": len(documents),
    }
    for row in DetectorRow:
        answered = runner.answer(built[row], batch=False)
        seconds = [r.seconds for r in answered if r is not None and r.seconds is not None]
        found[row.value] = {
            "timed": len(seconds),
            "median_seconds": float(np.median(seconds)) if seconds else None,
            "p95_seconds": float(np.percentile(seconds, 95)) if seconds else None,
            "cost_usd": f"{_spent(answered, frozen.settings.model):.4f}",
        }
    path = rounds / "latency.json"
    path.write_text(json.dumps(found, indent=2) + "\n", encoding="utf-8")
    console.print(f"[green]written[/green] {path}")


@app.command("stronger-model")
def detector_stronger_model(
    data_dir: DataDir = Path("data"),
    store: Store = STORE,
    ledger: LedgerPath = LEDGER,
    cache: Cache = OCR_CACHE,
    rounds: Rounds = ROUNDS,
) -> None:
    """Run the frozen main row once on Claude Opus 5.5, over the design fold (ADR-0032).

    Every setting but the model is the frozen detector's. The record compares the result with
    the frozen main row's on the same receipts and states, by the rule fixed in advance, whether
    Opus also runs on the validation and test splits.
    """
    path = rounds / STRONGER_MODEL
    if path.exists():
        console.print(f"[red]{path} exists[/red]: the stronger model is checked once")
        raise typer.Exit(1)
    frozen, text = _frozen(rounds)
    require_clean_commit()
    frozen_result = load_rounds(rounds)[frozen.version].main_row()
    documents = _design_fold(data_dir)
    built = requests(documents, cached_ocr(cache), OPUS, text)[DetectorRow.IMAGE_AND_OCR]
    answered = _runner(store, ledger).answer(built)
    result = row_result(
        [d.label is Label.FORGED for d in documents],
        [read_judgement(r) for r in answered],
        f"{_spent(answered, OPUS.model):.4f}",
    )
    record = StrongerModelCheck(
        settings=OPUS,
        instruction=frozen.instruction,
        instruction_sha256=frozen.instruction_sha256,
        commit=code_version().commit,
        result=result,
        frozen_result=frozen_result,
        row_added=clearly_better(result, frozen_result),
    )
    path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
    for name, shown in ((OPUS.model, result), (frozen.settings.model, frozen_result)):
        console.print(
            f"{name}: AP {shown.average_precision:.3f}, ROC-AUC {shown.roc_auc:.3f}, "
            f"{shown.failures} failures, {shown.cost_usd} USD"
        )
    added = "runs" if record.row_added else "does not run"
    console.print(f"Opus {added} on the validation and test splits")
