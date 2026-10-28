"""``docfraud final``: the reports of Phase 7, rendered from the test split's run records.

Every report is generated from its run record, as for the earlier phases; a report whose record
does not exist yet is skipped and named.
"""

from collections.abc import Callable
from pathlib import Path
from typing import Annotated

import typer

from docfraud.cli.common import REPORTED_RUNS, console
from docfraud.evaluation.runs import JsonRunStore, RunRecord
from docfraud.forensics.final import TEST_RUN as LOCALISATION_RUN
from docfraud.forensics.final_report import render_report as render_localisation
from docfraud.fusion.errors import RUN_NAME as ERRORS_RUN
from docfraud.fusion.final import RESAVED_RUN
from docfraud.fusion.final import RUN_NAME as TEST_CHECK_RUN
from docfraud.fusion.final_report import render_errors, render_test_report, render_track_r
from docfraud.llm.agent_final import RUN_NAME as AGENT_RUN
from docfraud.llm.agent_validation_report import render_report as render_agent
from docfraud.llm.detector_final import CLAIMS, DESCRIPTIVE, RUN_NAMES
from docfraud.llm.detector_final_report import render_report as render_detector

app = typer.Typer(help="The reports of Phase 7, from the test split's run records.")

REPORTS = Path("reports")

type Render = Callable[[JsonRunStore, RunRecord], str]


def _detector(store: JsonRunStore, claims: RunRecord) -> str:
    descriptive = store.records(RUN_NAMES[DESCRIPTIVE])
    return render_detector(
        claims.results, descriptive[-1].results if descriptive else None, claims.code.commit
    )


def _agent(_: JsonRunStore, record: RunRecord) -> str:
    return render_agent(
        record.results,
        record.config,
        record.code.commit,
        split="test",
        run_name=AGENT_RUN,
        command="docfraud final reports",
    )


RENDERED: dict[str, tuple[str, Render]] = {
    TEST_CHECK_RUN: (
        "fusion/test-check.md",
        lambda _, r: render_test_report(r.results, r.code.commit),
    ),
    RESAVED_RUN: ("fusion/track-r.md", lambda _, r: render_track_r(r.results, r.code.commit)),
    ERRORS_RUN: ("fusion/test-errors.md", lambda _, r: render_errors(r.results, r.code.commit)),
    LOCALISATION_RUN: (
        "forensics/test-localisation.md",
        lambda _, r: render_localisation(r.results, r.code.commit),
    ),
    RUN_NAMES[CLAIMS]: ("llm/detector/test.md", _detector),
    AGENT_RUN: ("llm/agent/test.md", _agent),
}
"""Each run of Phase 7, the report it renders, under ``reports``, and how."""


@app.command("reports")
def final_reports(
    runs: Annotated[Path, typer.Option(help="Where reported run records are kept.")] = (
        REPORTED_RUNS
    ),
    output: Annotated[Path, typer.Option(help="The reports directory.")] = REPORTS,
) -> None:
    """Render every report of Phase 7 whose run record exists, from the latest record."""
    store = JsonRunStore(runs)
    for name, (path, render) in RENDERED.items():
        found = store.records(name)
        if not found:
            console.print(f"no {name} record yet: {path} not written")
            continue
        target = output / path
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_text(render(store, found[-1]), encoding="utf-8")
        console.print(f"[green]written[/green] {target}")
