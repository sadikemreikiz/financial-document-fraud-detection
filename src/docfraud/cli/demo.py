"""``docfraud demo``: the public demo of ADR-0035.

``cards`` keeps the system's outputs for the receipts the demo shows in a run record.
"""

from collections.abc import Mapping
from pathlib import Path
from typing import Annotated, Any

import typer

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
from docfraud.demo.cards import RUN_NAME, budget_name, demo_cards
from docfraud.demo.gallery import CAUGHT, FALSE_ALARM, MISSED
from docfraud.evaluation.runs import JsonRunStore, run
from docfraud.forensics.learned.rounds import SEEDS
from docfraud.fusion.errors import RUN_NAME as ERRORS_RUN
from docfraud.fusion.errors import frozen_assessments
from docfraud.fusion.rehearsal import TEST_INPUTS_RUN

app = typer.Typer(help="The public demo (ADR-0035).")

DEMO_UNLOCK = "ADR-0035: the receipts of the public demo, after the single evaluation"


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
