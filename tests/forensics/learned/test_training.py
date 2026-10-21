import math
from dataclasses import replace
from datetime import UTC, datetime
from itertools import pairwise
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn
from torch.nn import functional

from docfraud.evaluation.runs import CodeVersion, Environment
from docfraud.forensics.learned.network import ResNet18Encoder
from docfraud.forensics.learned.sampler import Batch
from docfraud.forensics.learned.tiles import CLEAN, FORGED, IGNORED
from docfraud.forensics.learned.training import (
    CHECKPOINT,
    MODEL,
    RECORD,
    Run,
    RunRecord,
    Settings,
    StepData,
    binary_entropy,
    failure,
    initial_network,
    learning_rate,
    masked_loss,
    train,
    weights_generator,
)

FORGED_SHARE = 32 / (4 * 15 * 16)  # two squares of 16 px among four tiles less their top rows
SETTINGS = Settings(
    steps=40, peak_rate=0.05, warmup=5, checkpoint_every=10, failure_window=10, workers=0
)
CPU = torch.device("cpu")
ENVIRONMENT = Environment(
    python="3.14",
    docfraud="test",
    packages={},
    platform="test",
    machine="test",
    processor="test",
    cpu_count=1,
    memory_bytes=None,
)


def run(seed: int = 0, prior: float = 0.05) -> Run:
    return Run(2, 3, seed, prior, CodeVersion(commit="abc", dirty=False), ENVIRONMENT)


class Tiny(nn.Module):
    """One convolution over the grey value: enough to learn that bright squares are forged.

    Its bias starts at the logit of the share of forged pixels, as the real network's does.
    """

    def __init__(self, seed: int = 0) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        with torch.no_grad():
            self.conv.bias.fill_(math.log(FORGED_SHARE / (1 - FORGED_SHARE)))  # type: ignore[union-attr]

    def forward(self, pages: Tensor) -> Tensor:
        logits: Tensor = self.conv(pages.float() / 255)
        return logits


class Squares:
    """Batches of 16 px tiles in which bright 4 px squares are forged and the top row ignored."""

    def __init__(self) -> None:
        self.asked: list[int] = []

    def batch(self, step: int) -> Batch:
        self.asked.append(step)
        rng = np.random.default_rng(step)
        pages = rng.integers(0, 100, size=(4, 16, 16), dtype=np.uint8)
        labels = np.full((4, 16, 16), CLEAN, dtype=np.uint8)
        for tile in range(2):
            y, x = (int(v) for v in rng.integers(1, 12, size=2))
            pages[tile, y : y + 4, x : x + 4] += 150
            labels[tile, y : y + 4, x : x + 4] = FORGED
        labels[:, 0] = IGNORED
        return Batch(pages, labels, ())


def weights(network: nn.Module) -> dict[str, Tensor]:
    return {k: v.detach().clone() for k, v in network.state_dict().items()}


def test_the_learning_rate_warms_up_then_decays_to_zero() -> None:
    settings = Settings()
    rates = [learning_rate(step, settings) for step in range(settings.steps)]
    assert rates[0] == pytest.approx(3e-4 / 300)
    assert rates[299] == pytest.approx(3e-4)
    assert rates[300] == pytest.approx(3e-4)
    assert rates[3150] == pytest.approx(1.5e-4)
    assert 0 < rates[-1] < 1e-9
    assert all(a <= b for a, b in pairwise(rates[:300]))
    assert all(a >= b for a, b in pairwise(rates[300:]))


@pytest.mark.parametrize("values", [{"warmup": 0}, {"warmup": 6000}, {"checkpoint_every": 0}])
def test_inconsistent_settings_are_refused(values: dict[str, int]) -> None:
    with pytest.raises(ValueError, match="inconsistent"):
        Settings(**values)


def test_the_loss_averages_over_the_pixels_that_count() -> None:
    logits = torch.randn(2, 1, 3, 3)
    labels = torch.tensor([[[0, 1, 2]] * 3] * 2, dtype=torch.uint8)
    counted = labels != IGNORED
    target = (labels == FORGED).float()
    expected = functional.binary_cross_entropy_with_logits(logits[:, 0][counted], target[counted])
    assert masked_loss(logits, labels).item() == pytest.approx(expected.item())
    changed = logits.clone()
    changed[:, 0][~counted] = 100.0
    assert masked_loss(changed, labels).item() == pytest.approx(expected.item())


def test_a_batch_without_counted_pixels_is_refused() -> None:
    labels = torch.full((1, 2, 2), IGNORED, dtype=torch.uint8)
    with pytest.raises(ValueError, match="ignored"):
        masked_loss(torch.zeros(1, 1, 2, 2), labels)


def test_step_data_adds_the_channel_axis() -> None:
    pages, labels = StepData(Squares())[3]
    assert pages.shape == (4, 1, 16, 16)
    assert labels.shape == (4, 16, 16)
    assert pages.dtype == labels.dtype == torch.uint8


def test_a_run_learns_and_keeps_its_weights_and_record(tmp_path: Path) -> None:
    record = train(run(), Tiny(), Squares(), tmp_path, settings=SETTINGS, device=CPU)
    assert record.status == "finished"
    assert record.steps_done == len(record.forged_shares) == len(record.seconds) == 40
    assert np.mean(record.losses[-10:]) < np.mean(record.losses[:10])
    assert failure(record) is None
    assert (tmp_path / MODEL).is_file()
    assert not (tmp_path / CHECKPOINT).exists()
    stored = RunRecord.model_validate_json((tmp_path / RECORD).read_text(encoding="utf-8"))
    assert stored == record
    assert stored.settings["steps"] == 40
    assert record.forged_shares[0] == pytest.approx(FORGED_SHARE)
    model = torch.load(tmp_path / MODEL, weights_only=True)
    assert model.keys() == Tiny().state_dict().keys()


def test_a_resumed_run_ends_where_an_uninterrupted_one_does(tmp_path: Path) -> None:
    whole = Tiny()
    straight = train(run(), whole, Squares(), tmp_path / "straight", settings=SETTINGS, device=CPU)
    first = train(
        run(), Tiny(), Squares(), tmp_path / "resumed", until=15, settings=SETTINGS, device=CPU
    )
    assert first.status == "running"
    assert first.steps_done == 15
    assert (tmp_path / "resumed" / CHECKPOINT).is_file()
    data = Squares()
    resumed_network = Tiny(seed=99)  # replaced by the checkpoint's weights
    resumed = train(
        run(), resumed_network, data, tmp_path / "resumed", settings=SETTINGS, device=CPU
    )
    assert data.asked == list(range(15, 40))
    assert resumed.resumed_at == [15]
    assert resumed.losses == straight.losses
    for name, tensor in weights(whole).items():
        assert torch.equal(tensor, weights(resumed_network)[name]), name


def test_a_checkpoint_is_kept_every_so_many_steps(tmp_path: Path) -> None:
    train(run(), Tiny(), Squares(), tmp_path, until=25, settings=SETTINGS, device=CPU)
    record = RunRecord.model_validate_json((tmp_path / RECORD).read_text(encoding="utf-8"))
    assert record.steps_done == 25
    checkpoint = torch.load(tmp_path / CHECKPOINT, weights_only=True)
    assert RunRecord.model_validate_json(checkpoint["record"]).steps_done == 25


def test_a_finished_run_is_not_trained_again(tmp_path: Path) -> None:
    done = train(run(), Tiny(), Squares(), tmp_path, settings=SETTINGS, device=CPU)
    data = Squares()
    again = train(run(), Tiny(), data, tmp_path, settings=SETTINGS, device=CPU)
    assert again == done
    assert data.asked == []


@pytest.mark.parametrize("other", [run(seed=1), run(prior=0.1)])
def test_another_run_cannot_continue_this_one(tmp_path: Path, other: Run) -> None:
    train(run(), Tiny(), Squares(), tmp_path, until=12, settings=SETTINGS, device=CPU)
    with pytest.raises(ValueError, match="run kept here"):
        train(other, Tiny(), Squares(), tmp_path, settings=SETTINGS, device=CPU)
    (tmp_path / RECORD).unlink()
    with pytest.raises(ValueError, match="run kept here"):
        train(other, Tiny(), Squares(), tmp_path, settings=SETTINGS, device=CPU)


def test_a_checkpoint_is_resumed_only_by_the_code_that_wrote_it(tmp_path: Path) -> None:
    train(run(), Tiny(), Squares(), tmp_path, until=12, settings=SETTINGS, device=CPU)
    later = replace(run(), code=CodeVersion(commit="def", dirty=False))
    with pytest.raises(ValueError, match="resume it there"):
        train(later, Tiny(), Squares(), tmp_path, settings=SETTINGS, device=CPU)


def test_every_step_is_reported_with_its_loss(tmp_path: Path) -> None:
    steps: list[tuple[int, float]] = []
    record = train(
        run(),
        Tiny(),
        Squares(),
        tmp_path,
        until=12,
        settings=SETTINGS,
        device=CPU,
        on_step=lambda done, loss: steps.append((done, loss)),
    )
    assert steps == list(zip(range(1, 13), record.losses, strict=True))


def test_a_run_whose_loss_is_not_finite_diverges(tmp_path: Path) -> None:
    network = Tiny()
    with torch.no_grad():
        network.conv.bias.fill_(math.nan)  # type: ignore[union-attr]
    record = train(run(), network, Squares(), tmp_path, settings=SETTINGS, device=CPU)
    assert record.status == "diverged"
    assert record.steps_done == 1
    assert failure(record) == "diverged"
    assert not (tmp_path / MODEL).exists()
    assert not (tmp_path / CHECKPOINT).exists()
    stored = RunRecord.model_validate_json((tmp_path / RECORD).read_text(encoding="utf-8"))
    assert stored.status == "diverged"
    assert math.isnan(stored.losses[0])
    again = train(run(), Tiny(), Squares(), tmp_path, settings=SETTINGS, device=CPU)
    assert again.status == "diverged"  # read back, not trained again


def record(losses: list[float], shares: list[float], status: str = "finished") -> RunRecord:
    return RunRecord(
        held_out=2,
        system=3,
        seed=0,
        settings={"failure_window": 2},
        prior=0.1,
        code=CodeVersion(commit=None, dirty=None),
        environment=ENVIRONMENT,
        device="cpu",
        started=datetime(2026, 10, 14, tzinfo=UTC),
        status=status,  # type: ignore[arg-type]
        losses=losses,
        forged_shares=shares,
        seconds=[0.0] * len(losses),
        resumed_at=[],
    )


def test_a_run_that_does_no_better_than_a_constant_does_not_learn() -> None:
    constant = binary_entropy(0.1)
    assert failure(record([9.0, constant - 0.01, constant - 0.01], [0.5, 0.1, 0.1])) is None
    assert failure(record([0.0, constant, constant], [0.5, 0.1, 0.1])) == "does not learn"


def test_only_ended_runs_are_judged() -> None:
    with pytest.raises(ValueError, match="still running"):
        failure(record([1.0], [0.1], status="running"))


def test_binary_entropy_of_a_share() -> None:
    assert binary_entropy(0.5) == pytest.approx(math.log(2))
    assert binary_entropy(0.0) == binary_entropy(1.0) == 0.0


def test_runs_with_one_seed_start_from_one_decoder() -> None:
    encoder = ResNet18Encoder().state_dict()
    first = initial_network(encoder, 2, 0, prior=0.01)
    second = initial_network(encoder, 2, 0, prior=0.02)
    other = initial_network(encoder, 2, 1, prior=0.01)
    for name, tensor in first.state_dict().items():
        if name == "head.bias":
            assert not torch.equal(tensor, second.state_dict()[name])
        else:
            assert torch.equal(tensor, second.state_dict()[name]), name
        if name.startswith("encoder."):
            assert torch.equal(tensor, encoder[name.removeprefix("encoder.")])
    assert not torch.equal(first.state_dict()["head.weight"], other.state_dict()["head.weight"])


def test_the_weights_generator_depends_on_round_and_seed() -> None:
    draws = {
        (held_out, seed): float(torch.rand(1, generator=weights_generator(held_out, seed)))
        for held_out in range(5)
        for seed in range(3)
    }
    assert len(set(draws.values())) == 15
    assert float(torch.rand(1, generator=weights_generator(2, 0))) == draws[2, 0]
