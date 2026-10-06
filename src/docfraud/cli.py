"""Command-line interface for docfraud."""

from typing import Annotated

import typer

from docfraud import __version__

app = typer.Typer(
    name="docfraud",
    help="Explainable multimodal detection of forged financial documents.",
    no_args_is_help=True,
    add_completion=False,
)


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
