"""`docfraud forensics`: the external model of system 6, TruFor (ADR-0019)."""

from pathlib import Path
from typing import Annotated

import typer

from docfraud.cli import common
from docfraud.cli.common import console
from docfraud.data.download import safe_extract
from docfraud.forensics.trufor import CITATION, TRUFOR_CODE, TRUFOR_WEIGHTS

TRUFOR_DIR = Path("artifacts/trufor")

app = typer.Typer(help="External models of the pixel-forensics systems.", no_args_is_help=True)


@app.command("download-trufor")
def download_trufor(
    directory: Annotated[
        Path, typer.Option(help="Where the archives are kept and unpacked.")
    ] = TRUFOR_DIR,
) -> None:
    """Download TruFor's code and weights, verify their checksums and unpack them.

    Both stay outside version control: TruFor's licence allows informational and nonprofit use
    only and asks that its authors be credited wherever results are made public.
    """
    for source in (TRUFOR_CODE, TRUFOR_WEIGHTS):
        archive = common.download_with_progress(source, directory / "downloads")
        files = safe_extract(archive, directory)
        console.print(f"[green]unpacked[/green] {len(files)} files into {directory}")
    console.print(f"TruFor is for informational and nonprofit use; credit: {CITATION}")
