"""`docfraud synthesis`: what the synthetic forgery engine of ADR-0021 needs, and its review."""

import itertools
import webbrowser
from collections import Counter
from collections.abc import Iterable
from pathlib import Path
from typing import Annotated

import typer

from docfraud.cli import common
from docfraud.cli.common import (
    CV_FOLDS_PATH,
    OCR_CACHE,
    DataDir,
    cached_ocr,
    console,
    findit_documents,
    require_clean_commit,
)
from docfraud.data.local_page import HOST, make_server
from docfraud.data.observation import RegionObservation
from docfraud.data.observation import load_log as load_observations
from docfraud.data.observation_text import PageLanguage
from docfraud.evaluation.design_fold import DESIGN_FOLD
from docfraud.evaluation.folds import load_folds
from docfraud.evaluation.runs import code_version
from docfraud.markdown import render_table
from docfraud.synthesis.engine import Stream
from docfraud.synthesis.fonts import font_files
from docfraud.synthesis.review import (
    BREAK,
    MAX_ROUNDS,
    load_log,
    log_path,
    outcome,
    review_sample,
    review_tables,
)
from docfraud.synthesis.review_server import ReviewApp
from docfraud.synthesis.rounds import development_set, round_engine

FONTS_DIR = Path("artifacts/fonts")
REVIEW_RECORDS = Path("annotations/findit2/realism-review")
REVIEW_REPORT = Path("reports/synthesis/realism-review.md")
OBSERVATIONS = Path("annotations/findit2/forgery-observations/emre.json")

app = typer.Typer(help="The synthetic forgery engine.", no_args_is_help=True)

Reviewer = Annotated[str, typer.Option(help="Name of the person who reviews, lower case.")]
ReviewRound = Annotated[int, typer.Option("--round", min=1, max=MAX_ROUNDS, help="Review round.")]


@app.command("download-fonts")
def download_fonts(
    directory: Annotated[Path, typer.Option(help="Where the fonts are kept.")] = FONTS_DIR,
) -> None:
    """Download the fonts that imitations are typed with and verify their checksums.

    They are under the SIL Open Font License and are not stored in this repository.
    """
    for font in font_files():
        common.download_with_progress(font, directory)
    console.print(f"[green]{len(font_files())} fonts[/green] in {directory}")


@app.command("review")
def review_page(  # noqa: PLR0913, PLR0917 - one option per setting
    reviewer: Reviewer = "emre",
    round_number: ReviewRound = 1,
    data_dir: DataDir = Path("data"),
    records: Annotated[Path, typer.Option(help="Where the answers are saved.")] = REVIEW_RECORDS,
    port: Annotated[int, typer.Option(help="Local port; 0 picks a free one.")] = 8772,
    browser: Annotated[bool, typer.Option(help="Open the page in the browser.")] = True,
    language: Annotated[
        PageLanguage, typer.Option(help="Language of the page; answers are recorded in English.")
    ] = PageLanguage.TURKISH,
) -> None:
    """Serve the realism review on 127.0.0.1 until interrupted (Ctrl+C).

    It shows the 40 synthetic edits of the round, drawn from the development set of the engine
    at this commit, which must be clean; every answer is saved at once.
    """
    require_clean_commit()
    documents = findit_documents(data_dir)
    engine = round_engine(
        documents, load_folds(CV_FOLDS_PATH), DESIGN_FOLD, cached_ocr(OCR_CACHE), FONTS_DIR
    )
    items = review_sample([f.record for f in development_set(engine)], round_number)
    path = log_path(records, reviewer, round_number)
    review = ReviewApp(
        items,
        lambda base: engine.forge(base, 0, Stream.DEVELOPMENT).page,
        path,
        reviewer=reviewer,
        round_number=round_number,
        commit=code_version().commit,
        language=language,
    )
    server = make_server(review, port)
    url = f"http://{HOST}:{server.server_port}/"
    console.print(f"realism review for [bold]{reviewer}[/bold], round {round_number}, at {url}")
    console.print(f"answers are saved to {path} (Ctrl+C to stop)")
    if browser:
        webbrowser.open(url)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        console.print("stopped; run the same command to continue")
    finally:
        server.server_close()


def _visibility_seconds(observed: Iterable[RegionObservation]) -> list[float]:
    """Seconds spent on each real edit that asked visibility alone (step 4.2).

    The time of an answer is measured from the answer before it, whatever that asked.
    """
    ordered = sorted(observed, key=lambda a: a.observed_at)
    return [
        (later.observed_at - earlier.observed_at).total_seconds()
        for earlier, later in itertools.pairwise(ordered)
        if not later.detailed and (later.observed_at - earlier.observed_at).total_seconds() <= BREAK
    ]


@app.command("review-report")
def review_report(
    reviewer: Reviewer = "emre",
    round_number: ReviewRound = 1,
    data_dir: DataDir = Path("data"),
    records: Annotated[Path, typer.Option(help="Where the answers are.")] = REVIEW_RECORDS,
    output: Annotated[Path, typer.Option(help="Where to write the report.")] = REVIEW_REPORT,
) -> None:
    """Count a finished round of the realism review and write its report."""
    documents = findit_documents(data_dir)
    engine = round_engine(
        documents, load_folds(CV_FOLDS_PATH), DESIGN_FOLD, cached_ocr(OCR_CACHE), FONTS_DIR
    )
    recorded = [f.record for f in development_set(engine)]
    items = review_sample(recorded, round_number)
    log = load_log(log_path(records, reviewer, round_number), reviewer, round_number)
    try:
        result = outcome(log, items, recorded)
    except ValueError as error:
        console.print(f"[red]{error}[/red]")
        raise typer.Exit(1) from error
    observed = load_observations(OBSERVATIONS, reviewer).regions
    real = Counter(a.visibility for a in observed.values() if a.visibility is not None)
    real_seconds = _visibility_seconds(observed.values())
    verdict = (
        "**exceeded**: the synthetic edits are clearly more visible than the real ones"
        if result.exceeded
        else "**not exceeded**: the synthetic edits are not clearly more visible than the real ones"
    )
    header = (
        "# Realism review of the synthetic forgery engine\n\n"
        "> Generated by `docfraud synthesis review-report`. Do not edit by hand.\n\n"
        f"Round {round_number}, reviewer {reviewer}, {len(items)} synthetic edits on design-fold "
        "receipts ([ADR-0021](../../docs/decisions/0021-synthetic-forgery-engine.md), "
        "decision 5). "
        f"Found at 100 %: {result.found} of {len(items)}; the criterion is {verdict}.\n\n"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    tables = review_tables(result, real, items, log, recorded, real_seconds=real_seconds)
    output.write_text(header + "\n".join(render_table(t) for t in tables), encoding="utf-8")
    console.print(f"[green]written[/green] {output}")
