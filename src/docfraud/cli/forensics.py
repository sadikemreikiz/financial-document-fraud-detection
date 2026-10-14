"""`docfraud forensics`: external models and weights of the pixel-forensics systems.

TruFor, system 6 (ADR-0019), and the ImageNet weights that the encoder of systems 2 to 4 starts
from (ADR-0022).
"""

from pathlib import Path
from typing import Annotated

import typer

from docfraud.cli import common
from docfraud.cli.common import console
from docfraud.data.download import safe_extract
from docfraud.forensics.learned.pretrained import IMAGENET_WEIGHTS
from docfraud.forensics.trufor import CITATION, TRUFOR_CODE, TRUFOR_WEIGHTS

TRUFOR_DIR = Path("artifacts/trufor")
PRETRAINED_DIR = Path("artifacts/learned/pretrained")

app = typer.Typer(
    help="External models and weights of the pixel-forensics systems.", no_args_is_help=True
)


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


@app.command("download-resnet")
def download_resnet(
    directory: Annotated[Path, typer.Option(help="Where the weights are kept.")] = PRETRAINED_DIR,
) -> None:
    """Download the ImageNet weights of ResNet-18 that systems 2 to 4 start from and verify them.

    The file stays outside version control. ImageNet's terms allow non-commercial research and
    education, so the models trained from it are for non-commercial research.
    """
    common.download_with_progress(IMAGENET_WEIGHTS, directory)
    console.print(f"ResNet-18 with ImageNet weights; credit: {IMAGENET_WEIGHTS.citation}")
