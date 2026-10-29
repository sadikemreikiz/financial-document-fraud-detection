"""``docfraud demo``: the public demo of ADR-0035.

``cards`` keeps the system's outputs for the receipts the demo shows in a run record; ``assets``
exports their images and annotations for the owner's private repository; ``build`` writes the
page from both, and is what the deployment runs.
"""

import shutil
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Annotated, Any

import typer

from docfraud.cli.agent import AGENT
from docfraud.cli.common import (
    REPORTED_RUNS,
    DataDir,
    DatasetName,
    console,
    findit_documents,
    require_clean_commit,
)
from docfraud.cli.fusion import (
    TEST_INPUTS,
    TRAINING_INPUTS,
    VALIDATION_INPUTS,
    Runs,
    Training,
    Validation,
    _labels,
    _latest,
    _recorded_check,
    _recorded_table,
)
from docfraud.data.download import SOURCES, sha256_of
from docfraud.data.models import Split
from docfraud.data.splits import select
from docfraud.decision import BUDGETS, ReviewPolicy
from docfraud.demo.assets import AssetMismatchError, export, image_name, load
from docfraud.demo.cards import RUN_NAME, budget_name, demo_cards, shown_digests
from docfraud.demo.gallery import CAUGHT, FALSE_ALARM, MISSED
from docfraud.demo.page import Records, error_notes, render_page
from docfraud.evaluation.runs import JsonRunStore, code_version, run
from docfraud.forensics.learned.rounds import SEEDS
from docfraud.fusion.errors import RUN_NAME as ERRORS_RUN
from docfraud.fusion.errors import frozen_assessments
from docfraud.fusion.final import RESAVED_RUN
from docfraud.fusion.final import RUN_NAME as TEST_CHECK_RUN
from docfraud.fusion.rehearsal import TEST_INPUTS_RUN
from docfraud.llm.agent_final import RUN_NAME as AGENT_RUN
from docfraud.llm.detector_final import CLAIMS, DESCRIPTIVE, RUN_NAMES

app = typer.Typer(help="The public demo (ADR-0035).")

DEMO_UNLOCK = "ADR-0035: the receipts of the public demo, after the single evaluation"

PHASE_7_PAGE = Path("reports/README.md")
"""The Phase 7 page, whose table holds the error notes the owner checked."""

SITE = Path("_site")

Assets = Annotated[
    Path, typer.Option(help="The directory of the receipts' images and annotations.")
]


@app.command("cards")
def cards_command(
    data_dir: DataDir = Path("data"),
    runs: Runs = REPORTED_RUNS,
    training: Training = TRAINING_INPUTS,
    validation: Validation = VALIDATION_INPUTS,
    test: Annotated[Path, typer.Option(help="The test inputs table.")] = TEST_INPUTS,
) -> None:
    """Record the demo's cards from T2's frozen assessments of the test split (ADR-0035).

    The assessments are computed as the error analysis computed them, with the policies Phase 5's
    check recorded; the run stops unless they give that analysis's record: as many receipts sent
    to review, and the same misses with the same probabilities.
    """
    store = JsonRunStore(runs)
    require_clean_commit()
    check, training_table = _recorded_check(store, training, validation)
    measured = _recorded_table(store, TEST_INPUTS_RUN, test)
    errors = _latest(store, ERRORS_RUN)
    documents = findit_documents(data_dir)
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    stated = check.results["policy"]["ocr"]
    policies = [
        ReviewPolicy(stated[name]["budget"], stated[name]["threshold"])
        for name in (f"{b * 100:.0f} %" for b in BUDGETS)
    ]
    source = SOURCES[DatasetName.FINDIT2.value]
    with run(
        RUN_NAME,
        store=store,
        config={
            "decision": "ADR-0035",
            "validation_check": check.run_id,
            "error_analysis": errors.run_id,
        },
        seeds={},
        dataset_sha256={source.name: source.sha256},
    ) as current:
        receipts = sorted(
            select(documents, Split.TEST, unlock_test=True, reason=DEMO_UNLOCK),
            key=lambda d: d.doc_id,
        )
        rows = measured.rows("ocr", SEEDS[0])
        if [r.receipt for r in rows] != [d.doc_id for d in receipts]:
            console.print(f"[red]the rows of {test} are not the test receipts[/red]")
            raise typer.Exit(1)
        policy, assessed = frozen_assessments(
            training_table.rows("ocr", SEEDS[0]), _labels(train), rows, check.results
        )
        if policy != policies[0]:
            console.print(f"[red]{policy} is not the check's policy of {budget_name(policy)}[/red]")
            raise typer.Exit(1)
        results = demo_cards(assessed, receipts, policies, lambda d: sha256_of(d.image_path))
        problem = disagreement(results, errors.results)
        if problem:
            console.print(f"[red]{problem}[/red]")
            raise typer.Exit(1)
        for key, value in results.items():
            current.log(key, value)
    shown = sum(len(cards) for cards in results["gallery"].values())
    console.print(f"[green]recorded[/green] {shown} cards in {RUN_NAME}")


def disagreement(results: Mapping[str, Any], recorded: Mapping[str, Any]) -> str | None:
    """How the demo's outcomes differ from the error analysis's record, if they do."""
    outcomes = results["outcomes"]
    misses = {e["receipt"]: e["probability"] for e in recorded["errors"] if e["kind"] == "miss"}
    if outcomes[CAUGHT] + outcomes[FALSE_ALARM] != recorded["reviewed"]:
        return "the receipts sent to review differ from the error analysis's record"
    if outcomes[MISSED] != len(misses):
        return "the misses differ from the error analysis's record"
    if any(misses.get(c["receipt"]) != c["probability"] for c in results["gallery"][MISSED]):
        return "a miss shown differs from the error analysis's record"
    return None


@app.command("assets")
def assets_command(
    output: Assets,
    data_dir: DataDir = Path("data"),
    runs: Runs = REPORTED_RUNS,
) -> None:
    """Export the shown receipts' images and annotations for the private repository (ADR-0035, 6).

    The images are copied as the dataset distributes them, and must have the SHA-256 the
    ``demo-cards`` record pinned.
    """
    pinned = shown_digests(_latest(JsonRunStore(runs), RUN_NAME).results)
    documents = {
        d.doc_id: d
        for d in select(
            findit_documents(data_dir), Split.TEST, unlock_test=True, reason=DEMO_UNLOCK
        )
    }
    digests = export([documents[r] for r in pinned], output)
    differ = sorted(r for r in pinned if digests[r] != pinned[r])
    if differ:
        console.print(f"[red]images differ from the demo's record[/red]: {', '.join(differ)}")
        raise typer.Exit(1)
    console.print(f"[green]exported[/green] {len(digests)} receipts to {output}")


@app.command("build")
def build_command(
    assets: Assets,
    output: Annotated[Path, typer.Option(help="Where the page is written.")] = SITE,
    runs: Runs = REPORTED_RUNS,
) -> None:
    """Write the demo's page from the run records and the receipts' assets (ADR-0035, 3 to 6).

    Every image must have the SHA-256 the ``demo-cards`` record pinned; the build stops otherwise.
    """
    store = JsonRunStore(runs)
    cards = _latest(store, RUN_NAME).results
    pinned = shown_digests(cards)
    try:
        annotations = load(assets, pinned)
    except AssetMismatchError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    notes = error_notes(PHASE_7_PAGE.read_text(encoding="utf-8"))
    missing = [c["receipt"] for c in cards["gallery"][MISSED] if c["receipt"] not in notes]
    if missing:
        console.print(f"[red]no error note on {PHASE_7_PAGE} for[/red] {', '.join(missing)}")
        raise typer.Exit(1)
    case_files = {
        c["receipt"]: (AGENT / "test" / f"{c['receipt']}.md").read_text(encoding="utf-8")
        for c in cards["gallery"][CAUGHT]
    }
    version = code_version()
    built = datetime.now(UTC).date().isoformat()
    records = Records(
        check=_latest(store, TEST_CHECK_RUN).results,
        resaved=_latest(store, RESAVED_RUN).results,
        detector=_latest(store, RUN_NAMES[CLAIMS]).results,
        detector_descriptive=_latest(store, RUN_NAMES[DESCRIPTIVE]).results,
        agent=_latest(store, AGENT_RUN).results,
        errors=_latest(store, ERRORS_RUN).results,
        cards=cards,
        commit=version.commit or "unknown",
        built=built if version.dirty is False else f"{built} (with uncommitted changes)",
    )
    page = render_page(records, annotations, case_files, notes)
    receipts = output / "receipts"
    receipts.mkdir(parents=True, exist_ok=True)
    for receipt in pinned:
        shutil.copyfile(assets / image_name(receipt), receipts / image_name(receipt))
    (output / "index.html").write_text(page, encoding="utf-8")
    console.print(f"[green]written[/green] {output / 'index.html'} with {len(pinned)} receipts")
