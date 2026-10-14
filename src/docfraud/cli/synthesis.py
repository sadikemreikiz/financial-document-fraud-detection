"""`docfraud synthesis`: what the synthetic forgery engine of ADR-0021 needs."""

from pathlib import Path
from typing import Annotated

import typer

from docfraud.cli import common
from docfraud.cli.common import console
from docfraud.synthesis.fonts import font_files

FONTS_DIR = Path("artifacts/fonts")

app = typer.Typer(help="The synthetic forgery engine.", no_args_is_help=True)


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
