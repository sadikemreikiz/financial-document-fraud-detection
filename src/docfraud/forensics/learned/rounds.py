"""The runs of a round of systems 2 to 4: training them, then measuring them (ADR-0022, decision 5).

A round trains three systems with seeds 0, 1 and 2, seed 0 of each system first, so that a
failed training shows after about three hours (:data:`RUNS`). Every run keeps its checkpoint,
weights, record and results in a directory of its own (:func:`run_directory`). The round stops at
the first run that fails by the rule of decision 5, since a failure repeats all of its runs, and
all of a round's runs are trained by one version of the code: a round started at one commit is
continued only at that commit.

No network runs on a receipt before every run of the round has finished. Each run is then
measured (:mod:`~docfraud.forensics.learned.results`) on the receipts its round assigns: for the
development round, the validation split for thresholds, the design fold and the synthetic
development set (:func:`development_items`). A run measured by the same weights before is skipped.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from pathlib import Path

import torch
from torch import nn

from docfraud.data.models import Document, Split
from docfraud.data.normalize import load_grayscale
from docfraud.evaluation.design_fold import DESIGN_FOLD, DESIGN_FOLD_IDS, require_design_fold
from docfraud.evaluation.runs import CodeVersion, Environment
from docfraud.forensics.learned.pages import PageCache
from docfraud.forensics.learned.results import (
    RESULTS,
    Analyser,
    Group,
    Item,
    Progress,
    RunResults,
    measure_run,
    read_results,
    real_item,
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
    MODEL,
    RECORD,
    Run,
    RunRecord,
    Settings,
    failure,
    train,
)
from docfraud.synthesis.engine import Engine
from docfraud.synthesis.rounds import development_forgeries

SEEDS = (0, 1, 2)

RUNS: tuple[tuple[System, int], ...] = tuple((system, seed) for seed in SEEDS for system in System)
"""The runs of a round in the order they are trained: every system's seed 0 first."""

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
        uses_forger = any(purpose is Purpose.SYNTHETIC for purpose, _ in PLANS[system])
        sampler = TileSampler(
            setup.material, setup.pages, system, seed, setup.forger if uses_forger else None
        )
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
        if (directory / RESULTS).exists():
            stored = read_results(directory)
            if stored.model == network.identifier:
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


def validation_items(documents: Iterable[Document]) -> Iterator[Item]:
    """The receipts of the validation split, in identifier order, each loaded when needed."""
    for document in sorted((d for d in documents if d.split is Split.VAL), key=lambda d: d.doc_id):
        yield real_item(document, Group.VALIDATION, load_grayscale(document.image_path))


def design_items(documents: Iterable[Document], pages: PageCache) -> Iterator[Item]:
    """The receipts of the design fold, in identifier order, from the page cache."""
    chosen = sorted((d for d in documents if d.doc_id in DESIGN_FOLD_IDS), key=lambda d: d.doc_id)
    require_design_fold([d.doc_id for d in chosen])
    for document in chosen:
        yield real_item(document, Group.HELD_OUT, pages.page(document.doc_id))


def development_items(
    documents: Sequence[Document], pages: PageCache, engine: Engine
) -> RoundItems:
    """What the development round is measured on, in this order.

    The validation split, then the design fold, then the synthetic development set made by the
    round's engine.

    Raises:
        ValueError: If the engine is not of the round that holds out the design fold.
    """
    if engine.held_out != DESIGN_FOLD:
        msg = f"the development round holds out fold {DESIGN_FOLD}, not {engine.held_out}"
        raise ValueError(msg)

    def measured() -> Iterator[Item]:
        yield from design_items(documents, pages)
        yield from (synthetic_item(f) for f in development_forgeries(engine))

    return RoundItems(lambda: validation_items(documents), measured)
