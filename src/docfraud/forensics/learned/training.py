"""Training a run of systems 2 to 4: one round, one system, one seed (ADR-0022, decision 4).

A run takes a fixed number of steps, with AdamW, a linear warm-up and a cosine decay of the
learning rate to zero (:func:`learning_rate`), in single precision, on the Apple GPU when there is
one. The loss is binary cross-entropy per pixel, averaged over the pixels that count
(:func:`masked_loss`). Batches come from any source of numbered batches, prepared in background
processes (:class:`StepData`); because a step's tiles depend only on its number, a run resumed
from a checkpoint sees exactly the tiles it would have seen. Every ``checkpoint_every`` steps the
model and the optimiser are saved with the record of the run so far; a finished run is skipped.

The record (:class:`RunRecord`) holds the commit, the machine, the settings, and for every step
the loss, the share of forged pixels and the time, from which :func:`failure` applies the rule of
ADR-0022 (decision 5): a run diverges if its loss is ever not finite, and does not learn if its
mean loss over the last steps is not below that of the best constant prediction on the same tiles.
"""

import math
import time
from collections.abc import Callable, Mapping
from dataclasses import asdict, dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal, Protocol

import numpy as np
import torch
from pydantic import BaseModel, ConfigDict
from torch import Tensor, nn
from torch.nn import functional
from torch.utils.data import DataLoader, Dataset

from docfraud.evaluation.runs import CodeVersion, Environment
from docfraud.forensics.learned.network import UNet, load_encoder
from docfraud.forensics.learned.sampler import Batch, Purpose, generator
from docfraud.forensics.learned.tiles import FORGED, IGNORED

CHECKPOINT = "checkpoint.pt"
MODEL = "model.pt"
RECORD = "record.json"


@dataclass(frozen=True, slots=True)
class Settings:
    """The training settings of ADR-0022 (decision 4), the same for every run.

    None is tuned; the values are conventions stated with the results.
    """

    steps: int = 6000
    peak_rate: float = 3e-4
    warmup: int = 300
    beta1: float = 0.9
    beta2: float = 0.999
    weight_decay: float = 0.01
    checkpoint_every: int = 500
    failure_window: int = 500  # steps whose mean loss decides whether a run learned
    workers: int = 2  # background processes preparing batches

    def __post_init__(self) -> None:
        """Check that the steps fit together."""
        if not 0 < self.warmup < self.steps or self.checkpoint_every < 1:
            msg = f"inconsistent settings: {self}"
            raise ValueError(msg)


def learning_rate(step: int, settings: Settings) -> float:
    """The learning rate of a step: a linear warm-up, then a cosine decay to zero at the end."""
    if step < settings.warmup:
        return settings.peak_rate * (step + 1) / settings.warmup
    progress = (step - settings.warmup) / (settings.steps - settings.warmup)
    return settings.peak_rate * 0.5 * (1 + math.cos(math.pi * progress))


def masked_loss(logits: Tensor, labels: Tensor) -> Tensor:
    """Binary cross-entropy per pixel, averaged over the pixels that are not ignored.

    Args:
        logits: ``(n, 1, height, width)`` outputs of the network.
        labels: ``(n, height, width)`` labels ``CLEAN``, ``FORGED`` or ``IGNORED``, as ``uint8``.

    Raises:
        ValueError: If no pixel counts.
    """
    counted = labels != IGNORED
    if not bool(counted.any()):
        msg = "every pixel of the batch is ignored"
        raise ValueError(msg)
    target = (labels == FORGED).to(logits.dtype)
    weights = counted.to(logits.dtype)
    losses = functional.binary_cross_entropy_with_logits(logits[:, 0], target, reduction="none")
    return (losses * weights).sum() / weights.sum()


class Batches(Protocol):
    """Anything that gives the batch of a step, such as a :class:`TileSampler`."""

    def batch(self, step: int) -> Batch:
        """The tiles of a step and the labels of their pixels."""
        ...


class StepData(Dataset[tuple[Tensor, Tensor]]):
    """The batches of a run as a dataset indexed by step, for a loader with workers."""

    def __init__(self, batches: Batches) -> None:
        """Serve the batches of ``batches``."""
        self.batches = batches

    def __getitem__(self, step: int) -> tuple[Tensor, Tensor]:
        """Pages ``(n, 1, height, width)`` and labels ``(n, height, width)``, both ``uint8``."""
        batch = self.batches.batch(step)
        return torch.from_numpy(batch.pages[:, None]), torch.from_numpy(batch.labels)


class RunRecord(BaseModel):
    """What a run was made from and what it did, step by step."""

    # a loss that is not finite is written as such, so that a diverged run's record reads back
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")

    held_out: int
    system: int
    seed: int
    settings: dict[str, float | int]
    prior: float  # the share of forged pixels that set the initial output bias
    code: CodeVersion
    environment: Environment
    device: str
    started: datetime
    status: Literal["running", "finished", "diverged"]
    losses: list[float]  # the loss of every step done
    forged_shares: list[float]  # the share of forged pixels among those counted, per step
    seconds: list[float]  # the time of every step, waiting for its batch included
    resumed_at: list[int]  # the steps at which the run was resumed

    @property
    def steps_done(self) -> int:
        """How many steps the run has taken."""
        return len(self.losses)


def binary_entropy(share: float) -> float:
    """The cross-entropy of the best constant prediction where a share of the pixels is forged."""
    if not 0 < share < 1:
        return 0.0
    return -(share * math.log(share) + (1 - share) * math.log(1 - share))


def failure(record: RunRecord) -> str | None:
    """Whether a finished run failed by the rule of ADR-0022 (decision 5), and how.

    ``"diverged"`` if its loss was ever not finite; ``"does not learn"`` if its mean loss over the
    last ``failure_window`` steps is not below the loss of the best constant prediction on the
    same tiles, whose value is the mean share of forged pixels among them; otherwise ``None``.

    Raises:
        ValueError: If the run neither finished nor diverged.
    """
    if record.status == "diverged":
        return "diverged"
    if record.status != "finished":
        msg = f"the run is still {record.status}"
        raise ValueError(msg)
    window = int(record.settings["failure_window"])
    loss = float(np.mean(record.losses[-window:]))
    constant = binary_entropy(float(np.mean(record.forged_shares[-window:])))
    return None if loss < constant else "does not learn"


def weights_generator(held_out: int, seed: int) -> torch.Generator:
    """The generator of a run's initial decoder weights, the same for the three systems."""
    state = int(generator(held_out, seed, Purpose.WEIGHTS).integers(2**63))
    return torch.Generator().manual_seed(state)


def initial_network(encoder: Mapping[str, Tensor], held_out: int, seed: int, prior: float) -> UNet:
    """The network a run starts from: ImageNet's encoder, the decoder drawn from the run's seed."""
    network = UNet()
    load_encoder(network, encoder)
    network.initialise(weights_generator(held_out, seed), prior)
    return network


def default_device() -> torch.device:
    """The Apple GPU if there is one, else the processor."""
    return torch.device("mps" if torch.backends.mps.is_available() else "cpu")


@dataclass(frozen=True, slots=True)
class Run:
    """Which run is trained, and what its record should state about it."""

    held_out: int
    system: int
    seed: int
    prior: float
    code: CodeVersion
    environment: Environment


def train(  # noqa: PLR0913 - the run, its network and data, and where it goes
    run: Run,
    network: nn.Module,
    batches: Batches,
    directory: Path,
    *,
    settings: Settings | None = None,
    device: torch.device | None = None,
    until: int | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> RunRecord:
    """Train a run, resuming from its checkpoint, and return its record.

    A run whose record says it finished, or diverged, is not trained again. The final weights
    are saved in ``MODEL`` and the checkpoint is then removed. A checkpoint is resumed only by
    the code that wrote it, so that no run mixes two versions of the code.

    Args:
        run: The run and what its record states about it.
        network: The initial network; replaced by the checkpoint's weights when resuming.
        batches: The batches of the run's steps.
        directory: Where the checkpoint, the weights and the record of this run are kept.
        settings: Training settings; those of ADR-0022 by default.
        device: Where to train; the Apple GPU if there is one by default.
        until: Stop after this many steps, with a checkpoint, as if interrupted.
        on_step: Called with ``(steps_done, loss)`` after each step.

    Raises:
        ValueError: If the checkpoint belongs to another run, other settings or other code.
    """
    settings = settings or Settings()
    device = device or default_device()
    record_path = directory / RECORD
    if record_path.exists():
        record = RunRecord.model_validate_json(record_path.read_text(encoding="utf-8"))
        _check_same_run(record, run, settings)
        if record.status != "running":
            return record
    network = network.to(device).train()
    optimiser = torch.optim.AdamW(
        network.parameters(),
        lr=settings.peak_rate,
        betas=(settings.beta1, settings.beta2),
        weight_decay=settings.weight_decay,
    )
    record = _start_or_resume(run, network, optimiser, directory, settings=settings, device=device)
    end = settings.steps if until is None else min(until, settings.steps)
    losses, shares, seconds = list(record.losses), list(record.forged_shares), list(record.seconds)
    status: Literal["running", "finished", "diverged"] = "running"
    last = time.perf_counter()
    for step, (page_batch, label_batch) in _loader(batches, record.steps_done, end, settings):
        for group in optimiser.param_groups:
            group["lr"] = learning_rate(step, settings)
        labels = label_batch.to(device)
        loss = masked_loss(network(page_batch.to(device)), labels)
        value = float(loss.item())
        losses.append(value)
        counted = int((labels != IGNORED).sum())
        shares.append(int((labels == FORGED).sum()) / counted)
        if not math.isfinite(value):
            seconds.append(time.perf_counter() - last)
            status = "diverged"
            break
        optimiser.zero_grad(set_to_none=True)
        torch.autograd.backward(loss)
        optimiser.step()
        _synchronise(device)
        now = time.perf_counter()
        seconds.append(now - last)  # waiting for the batch included
        last = now
        done = step + 1
        if on_step is not None:
            on_step(done, value)
        if done % settings.checkpoint_every == 0 and done < settings.steps:
            record = record.model_copy(
                update={"losses": losses, "forged_shares": shares, "seconds": seconds}
            )
            _save_checkpoint(directory, network, optimiser, record)
    if status == "running" and len(losses) == settings.steps:
        status = "finished"
    record = record.model_copy(
        update={"losses": losses, "forged_shares": shares, "seconds": seconds, "status": status}
    )
    if status == "running":
        _save_checkpoint(directory, network, optimiser, record)
        return record
    if status == "finished":
        _save_atomically(directory / MODEL, _cpu_state(network))
    _write_record(directory, record)
    (directory / CHECKPOINT).unlink(missing_ok=True)
    return record


def _loader(
    batches: Batches, start: int, end: int, settings: Settings
) -> enumerate[tuple[Tensor, Tensor]]:
    """The batches of steps ``start`` to ``end - 1``, numbered from ``start``."""
    workers = settings.workers
    loader = DataLoader(
        StepData(batches),
        batch_size=None,
        sampler=range(start, end),
        num_workers=workers,
        multiprocessing_context="spawn" if workers else None,
        prefetch_factor=2 if workers else None,
    )
    return enumerate(loader, start=start)


def _start_or_resume(  # noqa: PLR0913 - everything a run starts from
    run: Run,
    network: nn.Module,
    optimiser: torch.optim.Optimizer,
    directory: Path,
    *,
    settings: Settings,
    device: torch.device,
) -> RunRecord:
    """The record of a fresh run, or the checkpoint's record with its weights loaded."""
    checkpoint_path = directory / CHECKPOINT
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
        record = RunRecord.model_validate_json(checkpoint["record"])
        _check_same_run(record, run, settings)
        if record.code != run.code:
            msg = f"the run was started at {record.code}; resume it there, not at {run.code}"
            raise ValueError(msg)
        network.load_state_dict(checkpoint["model"])
        optimiser.load_state_dict(checkpoint["optimiser"])
        return record.model_copy(update={"resumed_at": [*record.resumed_at, record.steps_done]})
    directory.mkdir(parents=True, exist_ok=True)
    return RunRecord(
        held_out=run.held_out,
        system=run.system,
        seed=run.seed,
        settings=asdict(settings),
        prior=run.prior,
        code=run.code,
        environment=run.environment,
        device=device.type,
        started=datetime.now(UTC),
        status="running",
        losses=[],
        forged_shares=[],
        seconds=[],
        resumed_at=[],
    )


def _check_same_run(record: RunRecord, run: Run, settings: Settings) -> None:
    """Refuse to continue a run with another identity, settings or initial bias."""
    found = (record.held_out, record.system, record.seed, record.settings, record.prior)
    wanted = (run.held_out, run.system, run.seed, asdict(settings), run.prior)
    if found != wanted:
        msg = f"the run kept here is {found}, not {wanted}"
        raise ValueError(msg)


def _save_checkpoint(
    directory: Path, network: nn.Module, optimiser: torch.optim.Optimizer, record: RunRecord
) -> None:
    _save_atomically(
        directory / CHECKPOINT,
        {
            "model": _cpu_state(network),
            "optimiser": optimiser.state_dict(),
            "record": record.model_dump_json(),
        },
    )
    _write_record(directory, record)


def _cpu_state(network: nn.Module) -> dict[str, Tensor]:
    return {name: tensor.detach().cpu() for name, tensor in network.state_dict().items()}


def _save_atomically(path: Path, content: object) -> None:
    temporary = path.with_suffix(".tmp")
    torch.save(content, temporary)
    temporary.replace(path)


def _write_record(directory: Path, record: RunRecord) -> None:
    temporary = directory / f"{RECORD}.tmp"
    temporary.write_text(record.model_dump_json(indent=1) + "\n", encoding="utf-8")
    temporary.replace(directory / RECORD)


def _synchronise(device: torch.device) -> None:
    """Wait for the device, so that a step's time is the time it took."""
    if device.type == "mps":  # pragma: no cover - CI has no Apple GPU
        torch.mps.synchronize()
