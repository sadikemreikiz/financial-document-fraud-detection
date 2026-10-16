"""`docfraud forensics`: the pixel-forensics systems' external models and learned rounds.

TruFor, system 6 (ADR-0019), the ImageNet weights that the encoder of systems 2 to 4 starts from,
and the training and measurement of their development round (ADR-0022).
"""

from pathlib import Path
from typing import TYPE_CHECKING, Annotated

import typer
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TimeRemainingColumn

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
from docfraud.cli.synthesis import FONTS_DIR
from docfraud.data.download import safe_extract
from docfraud.data.models import Split
from docfraud.evaluation.design_fold import DESIGN_FOLD, DESIGN_FOLD_IDS
from docfraud.evaluation.folds import load_folds
from docfraud.evaluation.runs import code_version, environment
from docfraud.forensics.learned.pages import PageCache
from docfraud.forensics.learned.pretrained import IMAGENET_WEIGHTS, read_weights
from docfraud.forensics.learned.sampler import CachedReceipts, RoundForger, System, round_material
from docfraud.forensics.trufor import CITATION, TRUFOR_CODE, TRUFOR_WEIGHTS
from docfraud.synthesis.engine import Stream
from docfraud.synthesis.rounds import round_engine
from docfraud.synthesis.statistics import round_statistics

TRUFOR_DIR = Path("artifacts/trufor")
PRETRAINED_DIR = Path("artifacts/learned/pretrained")
PAGES_DIR = Path("artifacts/learned/pages")
ROUNDS_DIR = Path("artifacts/learned/rounds")

app = typer.Typer(
    help="External models and weights of the pixel-forensics systems, and the development "
    "round of the learned ones.",
    no_args_is_help=True,
)

if TYPE_CHECKING:
    from docfraud.forensics.learned.rounds import RoundSetup

RoundsDir = Annotated[Path, typer.Option(help="Where the runs of the rounds are kept.")]


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


def _development_setup(data_dir: Path) -> RoundSetup:  # pragma: no cover - needs the dataset
    """What every run of the development round is trained from, with the page cache built."""
    from docfraud.forensics.learned.rounds import RoundSetup  # noqa: PLC0415 - imports PyTorch
    from docfraud.forensics.learned.training import initial_network  # noqa: PLC0415

    documents = findit_documents(data_dir)
    folds = load_folds(CV_FOLDS_PATH)
    ocr = cached_ocr(OCR_CACHE)
    pages = PageCache(PAGES_DIR)
    decoded = pages.build([d for d in documents if d.split is Split.TRAIN])
    console.print(f"page cache: {decoded} pages decoded into {pages.directory}")
    material = round_material(documents, folds, DESIGN_FOLD, pages, ocr)
    forger = RoundForger(
        round_statistics(documents, folds, DESIGN_FOLD, ocr),
        documents,
        folds,
        CachedReceipts(pages, material.words),
        FONTS_DIR,
    )
    encoder = read_weights(PRETRAINED_DIR / IMAGENET_WEIGHTS.filename)
    return RoundSetup(
        material,
        pages,
        forger,
        lambda held_out, seed, prior: initial_network(encoder, held_out, seed, prior),
    )


@app.command("repeat-check")
def repeat_check_command(  # pragma: no cover - needs the dataset, fonts and pretrained weights
    data_dir: DataDir = Path("data"), directory: RoundsDir = ROUNDS_DIR
) -> None:
    """Train the first 500 steps of system 4, seed 0, twice and compare the weights.

    ADR-0022 (decision 4) asks for this check before the development round, because PyTorch
    cannot make training bit for bit repeatable on the Apple GPU. About ten minutes; an
    interrupted check resumes, and a finished one is only compared again.
    """
    require_clean_commit()
    from docfraud.forensics.learned.rounds import (  # noqa: PLC0415 - imports PyTorch
        REPEAT_RUN,
        REPEAT_STEPS,
        repeat_check,
        repeat_directory,
    )

    setup = _development_setup(data_dir)
    system, seed = REPEAT_RUN
    columns = (BarColumn(), MofNCompleteColumn(), TimeRemainingColumn())
    with Progress("{task.description}", *columns, console=console) as progress:
        task = progress.add_task(
            f"system {int(system)}, seed {seed}, twice", total=2 * REPEAT_STEPS
        )
        result = repeat_check(
            setup,
            directory,
            code=code_version(),
            environment=environment(),
            on_step=lambda _done, _loss: progress.advance(task),
        )
    if result.identical:
        console.print(f"[green]identical[/green]: all {result.tensors} tensors bit for bit")
    else:
        console.print(
            f"[yellow]{len(result.differing)} of {result.tensors} tensors differ[/yellow], by up "
            f"to {result.largest_difference:.2e}; losses by up to "
            f"{result.largest_loss_difference:.2e}"
        )
    console.print(f"written: {repeat_directory(directory, DESIGN_FOLD)}")


@app.command("train-round")
def train_round_command(  # pragma: no cover - needs the dataset, fonts and pretrained weights
    data_dir: DataDir = Path("data"), directory: RoundsDir = ROUNDS_DIR
) -> None:
    """Train the nine runs of the development round of systems 2 to 4 (ADR-0022, decision 5).

    Round 2 holds out the design fold. Seed 0 of every system comes first. Runs resume from their
    last checkpoint and finished runs are skipped; the round stops at the first run that fails,
    and continues only at the commit it started from. About an hour per run on the reference
    machine, which must stay awake, on power and with its lid open.
    """
    require_clean_commit()
    from docfraud.forensics.learned.rounds import (  # noqa: PLC0415 - imports PyTorch
        stored_record,
        train_round,
    )
    from docfraud.forensics.learned.training import Settings  # noqa: PLC0415

    setup = _development_setup(data_dir)
    steps = Settings().steps
    columns = (BarColumn(), MofNCompleteColumn(), TimeRemainingColumn())
    with Progress("{task.description}", *columns, console=console) as progress:
        task = progress.add_task("", total=steps, visible=False)

        def started(system: System, seed: int) -> None:
            record = stored_record(directory, DESIGN_FOLD, system, seed)
            done = 0 if record is None else record.steps_done
            state = "finished" if record is not None and record.status != "running" else "training"
            progress.update(
                task,
                description=f"system {int(system)}, seed {seed}: {state}",
                completed=done,
                visible=True,
            )

        def stepped(done: int, loss: float) -> None:
            progress.update(task, completed=done)
            if done % 100 == 0:
                progress.console.print(f"  step {done:>5}: loss {loss:.4f}")

        outcome = train_round(
            setup,
            directory,
            code=code_version(),
            environment=environment(),
            on_run=started,
            on_step=stepped,
        )
    for record in outcome.records:
        console.print(
            f"system {record.system}, seed {record.seed}: {record.status}, "
            f"{record.steps_done} steps, {sum(record.seconds) / 3600:.2f} h"
        )
    if outcome.failed is not None:
        system, seed, reason = outcome.failed
        console.print(f"[red]system {int(system)}, seed {seed} {reason}[/red]: the round stops")
        raise typer.Exit(1)
    console.print(f"[green]round {DESIGN_FOLD} trained[/green]: {directory}")


@app.command("measure-round")
def measure_round_command(  # pragma: no cover - needs the dataset, the fonts and the trained runs
    data_dir: DataDir = Path("data"), directory: RoundsDir = ROUNDS_DIR
) -> None:
    """Measure every run of the finished development round of systems 2 to 4.

    Each network is run on the validation split, for thresholds only, then on the design fold and
    the synthetic development set, every receipt with its cached OCR words; the measurements of
    every receipt are written to the run's results.json, and runs measured by the same weights
    and in the current form before are skipped.
    """
    require_clean_commit()
    from docfraud.forensics.learned.inference import LearnedModel  # noqa: PLC0415 - PyTorch
    from docfraud.forensics.learned.rounds import (  # noqa: PLC0415 - imports PyTorch
        development_items,
        measure_round,
    )

    documents = findit_documents(data_dir)
    folds = load_folds(CV_FOLDS_PATH)
    ocr = cached_ocr(OCR_CACHE)
    engine = round_engine(documents, folds, DESIGN_FOLD, ocr, FONTS_DIR)
    items = development_items(documents, PageCache(PAGES_DIR), engine, ocr)
    total = (
        sum(d.split is Split.VAL for d in documents)
        + sum(d.doc_id in DESIGN_FOLD_IDS for d in documents)
        + len(engine.pools[Stream.DEVELOPMENT])
    )
    columns = (BarColumn(), MofNCompleteColumn(), TimeRemainingColumn())
    with Progress("{task.description}", *columns, console=console) as progress:
        task = progress.add_task("", total=total, visible=False)

        def started(system: System, seed: int) -> None:
            progress.update(
                task, description=f"system {int(system)}, seed {seed}", completed=0, visible=True
            )

        results = measure_round(
            directory,
            DESIGN_FOLD,
            items,
            model=LearnedModel.from_file,
            code=code_version(),
            environment=environment(),
            on_run=started,
            on_progress=lambda _group, _key: progress.advance(task),
        )
    for result in results:
        console.print(
            f"system {result.system}, seed {result.seed}: {len(result.measured)} receipts "
            f"measured by {result.model}"
        )
    console.print(f"[green]round {DESIGN_FOLD} measured[/green]: {directory}")
