"""The runs of a round of systems 2 to 4: training them, then measuring them (ADR-0022, decision 5).

A round trains three systems with seeds 0, 1 and 2, seed 0 of each system first, so that a
failed training shows after about three hours (:data:`RUNS`). Every run keeps its checkpoint,
weights, record and results in a directory of its own (:func:`run_directory`). The round stops at
the first run that fails by the rule of decision 5, since a failure repeats all of its runs, and
all of a round's runs are trained by one version of the code: a round started at one commit is
continued only at that commit.

Before the runs, the first steps of one of them are trained twice and compared
(:func:`repeat_check`), since PyTorch cannot make training bit for bit repeatable on the Apple
GPU. No network runs on a receipt before every run of the round has finished. Each run is then
measured (:mod:`~docfraud.forensics.learned.results`) on the receipts its round assigns: for the
development round, the validation split for thresholds, the design fold and the synthetic
development set (:func:`development_items`), each with its OCR word boxes, which set the text
height of the scores of ADR-0023. A run measured by the same weights before is skipped.

The clean measurement (ADR-0020) runs the rounds of the four clean folds the same way, after the
systems are frozen: :func:`train_rounds` trains all 36 runs from one commit, and
:func:`measure_rounds` measures them only once every one has finished with the settings of
ADR-0022, on the validation split and the round's held-out fold (:func:`clean_round_items`).
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import asdict, dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from pydantic import BaseModel, ConfigDict
from torch import nn

from docfraud.data.models import BBox, Document, Split
from docfraud.data.normalize import load_grayscale
from docfraud.evaluation.design_fold import (
    CLEAN_FOLDS,
    DESIGN_FOLD,
    DESIGN_FOLD_IDS,
    require_design_fold,
)
from docfraud.evaluation.folds import FoldAssignment
from docfraud.evaluation.runs import CodeVersion, Environment
from docfraud.forensics.learned.pages import PageCache
from docfraud.forensics.learned.results import (
    Analyser,
    Group,
    Item,
    Progress,
    RunResults,
    measure_run,
    real_item,
    stored_results,
    synthetic_item,
    write_results,
)
from docfraud.forensics.learned.sampler import (
    PLANS,
    Forger,
    Purpose,
    RoundMaterial,
    System,
    TileSampler,
    forged_share,
)
from docfraud.forensics.learned.training import (
    CHECKPOINT,
    MODEL,
    RECORD,
    Run,
    RunRecord,
    Settings,
    failure,
    train,
)
from docfraud.reading.ocr import OcrEngine
from docfraud.synthesis.engine import Engine
from docfraud.synthesis.rounds import development_forgeries

SEEDS = (0, 1, 2)

RUNS: tuple[tuple[System, int], ...] = tuple((system, seed) for seed in SEEDS for system in System)
"""The runs of a round in the order they are trained: every system's seed 0 first."""

TRIAL = Settings(steps=20, warmup=2, checkpoint_every=10, failure_window=10)
"""Runs of a few steps, to try the commands before the clean measurement; it never measures them."""

type NetworkFactory = Callable[[int, int, float], nn.Module]
"""The initial network of a run from its held-out fold, its seed and its initial output bias."""


def run_directory(root: Path, held_out: int, system: System, seed: int) -> Path:
    """Where one run keeps its checkpoint, weights, record and results."""
    return root / f"round-{held_out}" / f"system-{int(system)}" / f"seed-{seed}"


@dataclass(frozen=True, slots=True)
class RoundSetup:
    """What every run of a round is trained from."""

    material: RoundMaterial
    pages: PageCache
    forger: Forger  # the round's frozen engine on the training stream
    network: NetworkFactory


@dataclass(frozen=True, slots=True)
class RoundOutcome:
    """The records of a round's runs so far, and the run that failed, if one did."""

    records: tuple[RunRecord, ...]
    failed: tuple[System, int, str] | None  # system, seed and how it failed


def train_round(  # noqa: PLR0913 - the round, where it goes and what its records state
    setup: RoundSetup,
    root: Path,
    *,
    code: CodeVersion,
    environment: Environment,
    runs: Sequence[tuple[System, int]] = RUNS,
    settings: Settings | None = None,
    device: torch.device | None = None,
    on_run: Callable[[System, int], None] | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> RoundOutcome:
    """Train the runs of a round in order, resuming where it stopped, until one fails.

    Raises:
        ValueError: If the round is unfinished and one of its runs was trained by other code.
    """
    held_out = setup.material.held_out
    _require_one_code(root, held_out, runs, code)
    records: list[RunRecord] = []
    for system, seed in runs:
        if on_run is not None:
            on_run(system, seed)
        sampler = _sampler(setup, system, seed)
        prior = forged_share(sampler)
        record = train(
            Run(held_out, int(system), seed, prior, code, environment),
            setup.network(held_out, seed, prior),
            sampler,
            run_directory(root, held_out, system, seed),
            settings=settings,
            device=device,
            on_step=on_step,
        )
        records.append(record)
        if (reason := failure(record)) is not None:
            return RoundOutcome(tuple(records), (system, seed, reason))
    return RoundOutcome(tuple(records), None)


def train_rounds(  # noqa: PLR0913 - the rounds, where they go and what their records state
    setup: Callable[[int], RoundSetup],
    root: Path,
    *,
    code: CodeVersion,
    environment: Environment,
    rounds: Sequence[int] = CLEAN_FOLDS,
    runs: Sequence[tuple[System, int]] = RUNS,
    settings: Settings | None = None,
    device: torch.device | None = None,
    on_round: Callable[[int], None] | None = None,
    on_run: Callable[[System, int], None] | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> tuple[RoundOutcome, ...]:
    """Train the runs of several rounds in order, all from one commit, until one fails.

    The setup of a round, made from its held-out fold, is made only when the round's turn comes.
    Unlike a single round, which a later commit may measure once it has finished, the rounds of
    the clean measurement come from one commit: a run that other code trained stops all of them.

    Raises:
        ValueError: If a run of the rounds was trained by other code, or a setup does not hold
            out its round's fold.
    """
    others = {
        record.code
        for held_out in rounds
        for system, seed in runs
        if (record := stored_record(root, held_out, system, seed)) is not None
        and record.code != code
    }
    if others:
        msg = f"rounds {list(rounds)} hold runs trained by {sorted(map(str, others))}, not {code}"
        raise ValueError(msg)
    outcomes: list[RoundOutcome] = []
    for held_out in rounds:
        if on_round is not None:
            on_round(held_out)
        made = setup(held_out)
        if made.material.held_out != held_out:
            msg = f"the setup of round {held_out} holds out fold {made.material.held_out}"
            raise ValueError(msg)
        outcome = train_round(
            made,
            root,
            code=code,
            environment=environment,
            runs=runs,
            settings=settings,
            device=device,
            on_run=on_run,
            on_step=on_step,
        )
        outcomes.append(outcome)
        if outcome.failed is not None:
            break
    return tuple(outcomes)


def _sampler(setup: RoundSetup, system: System, seed: int) -> TileSampler:
    """A run's tiles; only systems that train on synthetic forgeries get the forger."""
    uses_forger = any(purpose is Purpose.SYNTHETIC for purpose, _ in PLANS[system])
    return TileSampler(
        setup.material, setup.pages, system, seed, setup.forger if uses_forger else None
    )


def stored_record(root: Path, held_out: int, system: System, seed: int) -> RunRecord | None:
    """The record a run has written so far, if it has started."""
    path = run_directory(root, held_out, system, seed) / RECORD
    if not path.exists():
        return None
    return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))


def _require_one_code(
    root: Path, held_out: int, runs: Sequence[tuple[System, int]], code: CodeVersion
) -> None:
    """Refuse to continue an unfinished round with code other than that of its runs."""
    stored = [stored_record(root, held_out, system, seed) for system, seed in runs]
    if all(r is not None and r.status == "finished" for r in stored):
        return
    others = {r.code for r in stored if r is not None and r.code != code}
    if others:
        msg = f"round {held_out} was started by {sorted(map(str, others))}; continue it there"
        raise ValueError(msg)


def finished_records(
    root: Path, held_out: int, runs: Sequence[tuple[System, int]] = RUNS
) -> list[RunRecord]:
    """The records of a round whose runs have all finished without failing.

    Raises:
        ValueError: If a run has not started, is still running or failed.
    """
    records = []
    for system, seed in runs:
        record = stored_record(root, held_out, system, seed)
        if record is None or record.status == "running":
            msg = f"system {int(system)}, seed {seed} of round {held_out} has not finished"
            raise ValueError(msg)
        if (reason := failure(record)) is not None:
            msg = f"system {int(system)}, seed {seed} of round {held_out} failed: {reason}"
            raise ValueError(msg)
        records.append(record)
    return records


REPEAT_RUN = (System.BOTH, 0)
"""The run whose first steps are trained twice: the system that draws every kind of tile."""

REPEAT_STEPS = 500
"""Steps of the repeat check (ADR-0022, decision 4)."""

REPEATABILITY = "repeatability.json"
"""The file of the repeat check's result."""


class Repeatability(BaseModel):
    """Whether the first steps of one run, trained twice, gave the same weights and losses."""

    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")

    held_out: int
    system: int
    seed: int
    steps: int
    device: str
    code: CodeVersion
    tensors: int
    differing: tuple[str, ...]  # the tensors that are not the same bit for bit
    largest_difference: float  # the largest absolute difference between two weights
    largest_loss_difference: float  # the largest difference between the losses of a step

    @property
    def identical(self) -> bool:
        """Whether every tensor agrees bit for bit."""
        return not self.differing


def repeat_directory(root: Path, held_out: int) -> Path:
    """Where the repeat check keeps its two trainings and its result."""
    return root / f"round-{held_out}" / "repeat"


def repeat_check(  # noqa: PLR0913 - the round, where it goes and what its records state
    setup: RoundSetup,
    root: Path,
    *,
    code: CodeVersion,
    environment: Environment,
    run: tuple[System, int] = REPEAT_RUN,
    steps: int = REPEAT_STEPS,
    settings: Settings | None = None,
    device: torch.device | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> Repeatability:
    """Train the first steps of a run twice from scratch and compare the two (ADR-0022, 4).

    Each training keeps its checkpoint, so that an interrupted check resumes and a finished one
    is only compared again. A check whose result is kept is returned as it is, by whatever code
    asks: measuring a finished round again at a later commit does not train anything.

    Raises:
        ValueError: If the steps are not fewer than a run's, so that no checkpoint would be kept.
    """
    settings = settings or Settings()
    if not 0 < steps < settings.steps:
        msg = f"the repeat check takes fewer steps than a run's {settings.steps}, not {steps}"
        raise ValueError(msg)
    held_out = setup.material.held_out
    directory = repeat_directory(root, held_out)
    system, seed = run
    if (kept := _kept_repeatability(directory)) is not None and (
        kept.held_out,
        kept.system,
        kept.seed,
        kept.steps,
    ) == (held_out, int(system), seed, steps):
        return kept
    states: list[dict[str, torch.Tensor]] = []
    records: list[RunRecord] = []
    for name in ("first", "second"):
        sampler = _sampler(setup, system, seed)
        prior = forged_share(sampler)
        records.append(
            train(
                Run(held_out, int(system), seed, prior, code, environment),
                setup.network(held_out, seed, prior),
                sampler,
                directory / name,
                settings=settings,
                device=device,
                until=steps,
                on_step=on_step,
            )
        )
        checkpoint = torch.load(
            directory / name / CHECKPOINT, map_location="cpu", weights_only=True
        )
        states.append(checkpoint["model"])
    first, second = states
    differing = tuple(name for name in first if not torch.equal(first[name], second[name]))
    result = Repeatability(
        held_out=held_out,
        system=int(system),
        seed=seed,
        steps=steps,
        device=records[0].device,
        code=code,
        tensors=len(first),
        differing=differing,
        largest_difference=max(
            (float((first[n].double() - second[n].double()).abs().max()) for n in differing),
            default=0.0,
        ),
        largest_loss_difference=max(
            abs(a - b) for a, b in zip(records[0].losses, records[1].losses, strict=True)
        ),
    )
    (directory / REPEATABILITY).write_text(
        result.model_dump_json(indent=1) + "\n", encoding="utf-8"
    )
    return result


def _kept_repeatability(directory: Path) -> Repeatability | None:
    path = directory / REPEATABILITY
    if not path.exists():
        return None
    return Repeatability.model_validate_json(path.read_text(encoding="utf-8"))


@dataclass(frozen=True, slots=True)
class RoundItems:
    """The receipts a round's networks are measured on, each made when it is needed."""

    validation: Callable[[], Iterable[Item]]
    measured: Callable[[], Iterable[Item]]


def measure_round(  # noqa: PLR0913 - the round, its receipts and what the results state
    root: Path,
    held_out: int,
    items: RoundItems,
    *,
    model: Callable[[Path], Analyser],
    code: CodeVersion,
    environment: Environment,
    runs: Sequence[tuple[System, int]] = RUNS,
    on_run: Callable[[System, int], None] | None = None,
    on_progress: Progress | None = None,
) -> list[RunResults]:
    """Measure every run of a finished round, skipping those measured by the same weights.

    Raises:
        ValueError: If a run of the round has not finished or failed.
    """
    finished_records(root, held_out, runs)
    measured = []
    for system, seed in runs:
        directory = run_directory(root, held_out, system, seed)
        network = model(directory / MODEL)
        stored = stored_results(directory)
        if stored is not None and stored.model == network.identifier:
            measured.append(stored)
            continue
        if on_run is not None:
            on_run(system, seed)
        results = measure_run(
            network,
            items.validation(),
            items.measured(),
            held_out=held_out,
            system=int(system),
            seed=seed,
            code=code,
            environment=environment,
            on_progress=on_progress,
        )
        write_results(directory, results)
        measured.append(results)
    return measured


def require_trained(
    root: Path,
    rounds: Sequence[int] = CLEAN_FOLDS,
    runs: Sequence[tuple[System, int]] = RUNS,
    settings: Settings | None = None,
) -> None:
    """Accept rounds whose runs have all finished, without failing, with the settings of ADR-0022.

    The clean measurement predicts nothing on a clean fold, with any system, before this holds,
    and the short runs of a trial never pass it.

    Raises:
        ValueError: If a run has not finished, failed or was trained with other settings.
    """
    wanted = asdict(settings or Settings())
    for held_out in rounds:
        for record in finished_records(root, held_out, runs):
            if record.settings != wanted:
                msg = (
                    f"system {record.system}, seed {record.seed} of round {held_out} was trained "
                    f"with {record.settings}, not {wanted}"
                )
                raise ValueError(msg)


def measure_rounds(  # noqa: PLR0913 - the rounds, their receipts and what the results state
    root: Path,
    items: Callable[[int], RoundItems],
    *,
    model: Callable[[Path], Analyser],
    code: CodeVersion,
    environment: Environment,
    rounds: Sequence[int] = CLEAN_FOLDS,
    runs: Sequence[tuple[System, int]] = RUNS,
    settings: Settings | None = None,
    on_run: Callable[[int, System, int], None] | None = None,
    on_progress: Progress | None = None,
) -> list[RunResults]:
    """Measure every run of several rounds, once all of them have finished.

    No network runs on a receipt before :func:`require_trained` accepts the rounds.

    Raises:
        ValueError: If a run has not finished, failed or was trained with other settings.
    """
    require_trained(root, rounds, runs, settings)
    measured: list[RunResults] = []
    for held_out in rounds:

        def started(system: System, seed: int, held_out: int = held_out) -> None:
            if on_run is not None:
                on_run(held_out, system, seed)

        measured.extend(
            measure_round(
                root,
                held_out,
                items(held_out),
                model=model,
                code=code,
                environment=environment,
                runs=runs,
                on_run=started,
                on_progress=on_progress,
            )
        )
    return measured


def _words(ocr: OcrEngine, page: npt.NDArray[np.uint8]) -> list[BBox]:
    return [token.box for token in ocr.recognize(page).tokens]


def validation_items(documents: Iterable[Document], ocr: OcrEngine) -> Iterator[Item]:
    """The receipts of the validation split, in identifier order, each loaded when needed."""
    for document in sorted((d for d in documents if d.split is Split.VAL), key=lambda d: d.doc_id):
        page = load_grayscale(document.image_path)
        yield real_item(document, Group.VALIDATION, page, _words(ocr, page))


def design_items(documents: Iterable[Document], pages: PageCache, ocr: OcrEngine) -> Iterator[Item]:
    """The receipts of the design fold, in identifier order, from the page cache."""
    chosen = sorted((d for d in documents if d.doc_id in DESIGN_FOLD_IDS), key=lambda d: d.doc_id)
    require_design_fold([d.doc_id for d in chosen])
    for document in chosen:
        page = pages.page(document.doc_id)
        yield real_item(document, Group.HELD_OUT, page, _words(ocr, page))


def development_items(
    documents: Sequence[Document], pages: PageCache, engine: Engine, ocr: OcrEngine
) -> RoundItems:
    """What the development round is measured on, in this order.

    The validation split, then the design fold, then the synthetic development set made by the
    round's engine. Every receipt carries its OCR word boxes, a synthetic forgery those of its
    base.

    Raises:
        ValueError: If the engine is not of the round that holds out the design fold.
    """
    if engine.held_out != DESIGN_FOLD:
        msg = f"the development round holds out fold {DESIGN_FOLD}, not {engine.held_out}"
        raise ValueError(msg)

    def measured() -> Iterator[Item]:
        yield from design_items(documents, pages, ocr)
        for forgery in development_forgeries(engine):
            yield synthetic_item(forgery, _words(ocr, pages.page(forgery.record.base)))

    return RoundItems(lambda: validation_items(documents, ocr), measured)


def clean_round_items(
    documents: Sequence[Document],
    folds: FoldAssignment,
    held_out: int,
    pages: PageCache,
    ocr: OcrEngine,
) -> RoundItems:
    """What a round of the clean measurement is measured on, in this order.

    The validation split, then the round's held-out fold in identifier order, from the page
    cache, every receipt with its OCR word boxes. No synthetic forgery is measured (ADR-0021,
    decision 6). A trial of the commands on the development round takes the design fold this way.

    Raises:
        ValueError: If the fold holds no receipt.
    """
    ids = folds.documents(held_out)
    if not ids:
        msg = f"fold {held_out} holds no receipt"
        raise ValueError(msg)
    chosen = sorted((d for d in documents if d.doc_id in ids), key=lambda d: d.doc_id)

    def measured() -> Iterator[Item]:
        for document in chosen:
            page = pages.page(document.doc_id)
            yield real_item(document, Group.HELD_OUT, page, _words(ocr, page))

    return RoundItems(lambda: validation_items(documents, ocr), measured)
