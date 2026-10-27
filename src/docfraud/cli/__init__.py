"""Command-line interface for docfraud: one command group per module."""

from typing import Annotated

import typer

from docfraud import __version__
from docfraud.cli import (
    baseline,
    copy_move,
    data,
    detector,
    forensics,
    fusion,
    llm,
    marks,
    observation,
    reading,
    rules,
    synthesis,
)

app = typer.Typer(
    name="docfraud",
    help="Explainable multimodal detection of forged financial documents.",
    no_args_is_help=True,
    add_completion=False,
)
app.add_typer(data.app, name="data")
app.add_typer(reading.app, name="reading")
app.add_typer(baseline.app, name="baseline")
app.add_typer(rules.app, name="rules")
app.add_typer(observation.app, name="observation")
app.add_typer(marks.app, name="marks")
app.add_typer(forensics.app, name="forensics")
app.add_typer(synthesis.app, name="synthesis")
app.add_typer(llm.app, name="llm")
app.add_typer(copy_move.app, name="copy-move")
app.add_typer(fusion.app, name="fusion")
app.add_typer(detector.app, name="detector")


def _print_version(value: bool) -> None:
    if value:
        typer.echo(f"docfraud {__version__}")
        raise typer.Exit


@app.callback()
def main(
    version: Annotated[
        bool,
        typer.Option(
            "--version",
            callback=_print_version,
            is_eager=True,
            help="Show the installed version and exit.",
        ),
    ] = False,
) -> None:
    """Detect forged financial documents and explain why."""
