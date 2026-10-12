"""`docfraud observation`: observe the real forgeries of the design fold (ADR-0014, step 4.2)."""

import webbrowser
from pathlib import Path
from typing import Annotated

import typer

from docfraud.cli.common import DataDir, console, findit_documents, require_clean_commit
from docfraud.data.observation import (
    ASSISTANT,
    export_crops,
    load_log,
    observation_plan,
    seal,
    seal_matches,
)
from docfraud.data.observation_report import build_report, render_report
from docfraud.data.observation_server import HOST, ObservationApp, make_server
from docfraud.data.observation_text import PageLanguage
from docfraud.evaluation.runs import code_version

OBSERVATION_LOGS = Path("artifacts/observations")
"""Logs in progress. They stay out of version control until both observers have finished."""

OBSERVATION_RECORDS = Path("annotations/findit2/forgery-observations")
"""Finished logs and the assistant's seal, under version control."""

app = typer.Typer(
    help="Observe the real forgeries of the design fold (docs/forgery-observation-guide.md).",
    no_args_is_help=True,
)

Observer = Annotated[str, typer.Option(help="Observer name, lower case (e.g. emre).")]

OBSERVATION_REPORT = Path("reports/forensics/observations.md")


@app.command("page")
def observation_page(  # noqa: PLR0913, PLR0917 - one option per setting
    observer: Observer,
    data_dir: DataDir = Path("data"),
    logs: Annotated[Path, typer.Option(help="Directory of the logs in progress.")] = (
        OBSERVATION_LOGS
    ),
    port: Annotated[int, typer.Option(help="Local port; 0 picks a free one.")] = 8770,
    browser: Annotated[bool, typer.Option(help="Open the page in the browser.")] = True,
    language: Annotated[
        PageLanguage, typer.Option(help="Language of the page; answers are recorded in English.")
    ] = PageLanguage.ENGLISH,
) -> None:
    """Serve the observation page on 127.0.0.1 until interrupted (Ctrl+C).

    Every answer is saved at once, so the page can be closed and opened again at any time. The
    page needs a clean commit, which every answer records as the form version.
    """
    require_clean_commit()
    log_path = logs / f"{observer}.json"
    observation = ObservationApp(
        findit_documents(data_dir), log_path, observer, code_version().commit, language
    )
    server = make_server(observation, port)
    url = f"http://{HOST}:{server.server_port}/"
    console.print(f"observation page for [bold]{observer}[/bold] at {url} (Ctrl+C to stop)")
    console.print(f"answers are saved to {log_path}")
    if browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("stopped; run the same command to continue")
    finally:
        server.server_close()


@app.command("crops")
def observation_crops(
    data_dir: DataDir = Path("data"),
    output: Annotated[Path, typer.Option(help="Where to write the crops.")] = (
        OBSERVATION_LOGS / f"{ASSISTANT}-crops"
    ),
) -> None:
    """Write the crops the assistant observes: native resolution, in the order of the plan."""
    require_clean_commit()
    documents = findit_documents(data_dir)
    files = export_crops(documents, observation_plan(documents), output)
    console.print(f"[green]written[/green] {len(files)} crops to {output}")


@app.command("check")
def observation_check(
    observer: Observer,
    data_dir: DataDir = Path("data"),
    logs: Annotated[Path, typer.Option(help="Directory of the logs.")] = OBSERVATION_LOGS,
) -> None:
    """Validate an observer's log against the plan and show how much of it is done."""
    log = load_log(logs / f"{observer}.json", observer)
    plan = observation_plan(findit_documents(data_dir))
    planned = {item.key for item in plan.regions}
    problems = sorted(set(log.regions) - planned) + sorted(set(log.receipts) - set(plan.receipts))
    if observer != ASSISTANT:
        problems += [
            f"{key} (full form expected: {key in plan.full_form})"
            for key, region in sorted(log.regions.items())
            if region.detailed != (key in plan.full_form)
        ]
    if problems:
        console.print(f"[red]not as planned[/red]: {', '.join(problems)}")
        raise typer.Exit(1)
    progress = f"{len(log.regions)} of {len(plan.regions)} regions"
    if observer != ASSISTANT:
        full = sum(r.detailed for r in log.regions.values())
        progress += (
            f" ({full} of {len(plan.full_form)} full forms), "
            f"{len(log.receipts)} of {len(plan.receipts)} receipts"
        )
    console.print(f"[green]valid[/green] log of {observer}: {progress}")


@app.command("seal")
def observation_seal(
    observer: Observer = ASSISTANT,
    logs: Annotated[Path, typer.Option(help="Directory of the logs.")] = OBSERVATION_LOGS,
    records: Annotated[Path, typer.Option(help="Where the seal is kept.")] = (OBSERVATION_RECORDS),
    verify: Annotated[bool, typer.Option(help="Check the log against an existing seal.")] = False,
) -> None:
    """Seal an observer's log by its SHA-256, or check that it still matches its seal."""
    log_path = logs / f"{observer}.json"
    seal_path = records / f"{observer}.sha256"
    if verify:
        if not seal_matches(log_path, seal_path):
            console.print(f"[red]changed since it was sealed[/red]: {log_path}")
            raise typer.Exit(1)
        console.print(f"[green]matches its seal[/green]: {log_path}")
        return
    if seal_path.exists():
        console.print(f"[red]refusing to overwrite[/red] {seal_path}")
        raise typer.Exit(1)
    load_log(log_path, observer)
    digest = seal(log_path, seal_path)
    console.print(f"[green]sealed[/green] {log_path}: {digest} in {seal_path}")


@app.command("report")
def observation_report(
    owner: Observer = "emre",
    data_dir: DataDir = Path("data"),
    records: Annotated[Path, typer.Option(help="Finished logs and the seal.")] = (
        OBSERVATION_RECORDS
    ),
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = OBSERVATION_REPORT,
) -> None:
    """Analyse both finished logs as the guide fixes it and write the Markdown report.

    The assistant's log must still match its seal.
    """
    if not seal_matches(records / f"{ASSISTANT}.json", records / f"{ASSISTANT}.sha256"):
        console.print("[red]the assistant's log no longer matches its seal[/red]")
        raise typer.Exit(1)
    documents = findit_documents(data_dir)
    tables = build_report(
        documents,
        observation_plan(documents),
        load_log(records / f"{ASSISTANT}.json", ASSISTANT),
        load_log(records / f"{owner}.json", owner),
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(tables), encoding="utf-8")
    console.print(f"[green]written[/green] {output}")
