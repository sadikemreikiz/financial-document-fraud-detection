"""`docfraud marks`: box the benign marks of the design fold (ADR-0014, step 4.2, decision 4)."""

import webbrowser
from pathlib import Path
from typing import Annotated

import typer

from docfraud.cli.common import DataDir, console, findit_documents, require_clean_commit
from docfraud.data.benign_marks import (
    MAX_ROUNDS,
    ReceiptMarks,
    check_problems,
    export_views,
    final_marks,
    image_sizes,
    load_check_log,
    load_mark_log,
    load_rounds,
    mark_problems,
    marks_plan,
    round_path,
    round_receipts,
)
from docfraud.data.benign_marks_report import build_report, render_report
from docfraud.data.benign_marks_server import MarkCheckApp
from docfraud.data.local_page import HOST, make_server
from docfraud.data.models import Document
from docfraud.data.observation import ASSISTANT
from docfraud.data.observation_text import PageLanguage
from docfraud.evaluation.runs import code_version

MARK_WORK = Path("artifacts/benign-marks")
"""Marks and checks in progress, and the assistant's views; outside version control."""

MARK_RECORDS = Path("annotations/findit2/benign-marks")
"""Finished marks and checks of every round, under version control."""

MARK_REPORT = Path("reports/forensics/benign-marks.md")

app = typer.Typer(
    help="Box the benign marks of the design fold (docs/benign-marks-guide.md).",
    no_args_is_help=True,
)

Round = Annotated[
    int, typer.Option("--round", min=1, max=MAX_ROUNDS, help="Round of marking and checking.")
]
Checker = Annotated[str, typer.Option(help="Name of the person who checks, lower case.")]
Records = Annotated[Path, typer.Option(help="Finished marks and checks of earlier rounds.")]


def _round_receipts(
    documents: tuple[Document, ...], round_number: int, records: Path, checker: str
) -> tuple[str, ...]:
    """The receipts of a round, from the finished checks of the rounds before it."""
    _, checks = load_rounds(records, ASSISTANT, checker)
    if len(checks) < round_number - 1:
        console.print(f"[red]round {round_number - 1} is not checked yet[/red] in {records}")
        raise typer.Exit(1)
    return round_receipts(marks_plan(documents), checks[: round_number - 1])


@app.command("views")
def marks_views(  # noqa: PLR0913, PLR0917 - one option per setting
    round_number: Round = 1,
    data_dir: DataDir = Path("data"),
    records: Records = MARK_RECORDS,
    checker: Checker = "emre",
    marked: Annotated[
        bool, typer.Option(help="Draw the assistant's marks of this round from --work.")
    ] = False,
    work: Annotated[Path, typer.Option(help="Directory of work in progress.")] = MARK_WORK,
) -> None:
    """Write the views the assistant marks from, or checks its marks on (--marked).

    Receipts are numbered in identifier order, the order in which the assistant marks them.
    """
    documents = findit_documents(data_dir)
    receipts = sorted(_round_receipts(documents, round_number, records, checker))
    marks: dict[str, ReceiptMarks] | None = None
    output = work / f"views-{round_number}"
    if marked:
        marks = load_mark_log(round_path(work, ASSISTANT, round_number)).receipts
        receipts = [d for d in receipts if d in marks]
        output = work / f"views-{round_number}-marked"
    files = export_views(documents, receipts, output, marks)
    console.print(f"[green]written[/green] {len(files)} views of {len(receipts)} receipts")
    console.print(f"to {output}")


@app.command("check")
def marks_check(
    round_number: Round = 1,
    data_dir: DataDir = Path("data"),
    directory: Annotated[Path, typer.Option(help="Where the logs of this round are.")] = (
        MARK_WORK
    ),
    records: Records = MARK_RECORDS,
    checker: Checker = "emre",
) -> None:
    """Validate the marks and the checks of a round, as far as they exist, and show progress."""
    documents = findit_documents(data_dir)
    expected = _round_receipts(documents, round_number, records, checker)
    pages = image_sizes(documents, expected)
    problems: list[str] = []
    progress: list[str] = []
    marks_path = round_path(directory, ASSISTANT, round_number)
    if marks_path.exists():
        log = load_mark_log(marks_path)
        problems += mark_problems(log, round_number, expected, pages)
        boxes = sum(len(r.marks) for r in log.receipts.values())
        progress.append(f"marks for {len(log.receipts)} of {len(expected)} receipts, {boxes} boxes")
    checks_path = round_path(directory, checker, round_number)
    if checks_path.exists():
        checks = load_check_log(checks_path, checker, round_number)
        problems += check_problems(checks, expected, pages)
        progress.append(f"checks of {len(checks.receipts)} of {len(expected)} receipts")
    if problems:
        console.print(f"[red]not as planned[/red]: {'; '.join(problems)}")
        raise typer.Exit(1)
    if not progress:
        console.print(f"[red]no log of round {round_number}[/red] in {directory}")
        raise typer.Exit(1)
    console.print(f"[green]valid[/green] round {round_number}: {', '.join(progress)}")


@app.command("page")
def marks_page(  # noqa: PLR0913, PLR0917 - one option per setting
    checker: Checker,
    round_number: Round = 1,
    data_dir: DataDir = Path("data"),
    records: Records = MARK_RECORDS,
    work: Annotated[Path, typer.Option(help="Where the checks are saved.")] = MARK_WORK,
    port: Annotated[int, typer.Option(help="Local port; 0 picks a free one.")] = 8771,
    browser: Annotated[bool, typer.Option(help="Open the page in the browser.")] = True,
    language: Annotated[
        PageLanguage, typer.Option(help="Language of the page; answers are recorded in English.")
    ] = PageLanguage.ENGLISH,
) -> None:
    """Serve the check page on 127.0.0.1 until interrupted (Ctrl+C).

    The page shows the committed marks of the round, and every check is saved at once, so the
    page can be closed and opened again at any time. It needs a clean commit, which every check
    records.
    """
    require_clean_commit()
    documents = findit_documents(data_dir)
    marks = load_mark_log(round_path(records, ASSISTANT, round_number))
    expected = _round_receipts(documents, round_number, records, checker)
    problems = mark_problems(marks, round_number, expected, image_sizes(documents, expected))
    if problems:
        console.print(f"[red]the marks are not as planned[/red]: {'; '.join(problems)}")
        raise typer.Exit(1)
    log_path = round_path(work, checker, round_number)
    check = MarkCheckApp(
        documents, marks, log_path, checker, commit=code_version().commit, language=language
    )
    server = make_server(check, port)
    url = f"http://{HOST}:{server.server_port}/"
    console.print(f"check page for [bold]{checker}[/bold], round {round_number}, at {url}")
    console.print(f"checks are saved to {log_path} (Ctrl+C to stop)")
    if browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("stopped; run the same command to continue")
    finally:
        server.server_close()


@app.command("report")
def marks_report(
    data_dir: DataDir = Path("data"),
    records: Records = MARK_RECORDS,
    checker: Checker = "emre",
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = MARK_REPORT,
) -> None:
    """Analyse the finished rounds as the guide fixes it and write the Markdown report.

    Every round must be boxed and checked, and the procedure finished: every receipt accepted,
    or the last round held.
    """
    documents = findit_documents(data_dir)
    plan = marks_plan(documents)
    mark_logs, check_logs = load_rounds(records, ASSISTANT, checker)
    try:
        final = final_marks(plan, mark_logs, check_logs)
    except ValueError as error:
        console.print(f"[red]the rounds are not finished[/red]: {error}")
        raise typer.Exit(1) from error
    tables = build_report(documents, plan, final, check_logs, image_sizes(documents, plan))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_report(tables), encoding="utf-8")
    console.print(f"[green]written[/green] {output}")
