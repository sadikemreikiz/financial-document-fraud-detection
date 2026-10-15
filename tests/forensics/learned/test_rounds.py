import hashlib
import math
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
import torch
from torch import Tensor, nn

from docfraud.data.models import Document, Entity, Split, Technique
from docfraud.evaluation.design_fold import DESIGN_FOLD
from docfraud.evaluation.folds import FoldAssignment
from docfraud.evaluation.runs import CodeVersion
from docfraud.forensics import trufor
from docfraud.forensics.learned.inference import LearnedOutput
from docfraud.forensics.learned.pages import PageCache
from docfraud.forensics.learned.results import RESULTS, Group, read_results
from docfraud.forensics.learned.rounds import (
    REPEATABILITY,
    RUNS,
    Repeatability,
    RoundOutcome,
    RoundSetup,
    development_items,
    finished_records,
    measure_round,
    repeat_check,
    repeat_directory,
    run_directory,
    stored_record,
    train_round,
)
from docfraud.forensics.learned.sampler import (
    CachedReceipts,
    RoundForger,
    System,
    TileSampler,
    forged_share,
    round_material,
)
from docfraud.forensics.learned.training import MODEL, RECORD, RunRecord, Settings
from docfraud.synthesis.engine import Engine, Forgery, Stream
from docfraud.synthesis.statistics import RoundStatistics, Template
from tests.forensics.learned.test_training import ENVIRONMENT
from tests.synthesis.pages import documents, font, receipt
from tests.synthesis.test_statistics import FakeOcr

CODE = CodeVersion(commit="abc", dirty=False)
SETTINGS = Settings(steps=4, warmup=1, checkpoint_every=2, failure_window=2, workers=0)
CPU = torch.device("cpu")
TWO_RUNS = ((System.SYNTHETIC, 0), (System.REAL, 0))


class Tiny(nn.Module):
    """One convolution over the grey value, its weights drawn from the run's seed."""

    def __init__(self, seed: int, prior: float) -> None:
        super().__init__()
        torch.manual_seed(seed)
        self.conv = nn.Conv2d(1, 1, 3, padding=1)
        with torch.no_grad():
            self.conv.bias.fill_(math.log(prior / (1 - prior)))  # type: ignore[union-attr]

    def forward(self, pages: Tensor) -> Tensor:
        logits: Tensor = self.conv(pages.float() / 255)
        return logits


def statistics() -> RoundStatistics:
    entities = (Entity.PRODUCT, Entity.TOTAL_PAYMENT, Entity.METADATA, Entity.OTHER)
    templates = tuple(Template((Technique.CPI,), entity, 2) for entity in entities)
    return RoundStatistics(DESIGN_FOLD, templates, (1, 2), {}, {})


class CountingForger:
    """The round's forger, counting the forgeries it makes."""

    def __init__(self, forger: RoundForger) -> None:
        self.forger = forger
        self.made = 0

    def forge(self, base: str, index: int) -> Forgery:
        self.made += 1
        return self.forger.forge(base, index)


@dataclass(frozen=True)
class Toy:
    documents: list[Document]
    folds: FoldAssignment
    pages: PageCache
    setup: RoundSetup
    forger: CountingForger


@pytest.fixture
def toy(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Toy:
    docs, folds = documents(tmp_path)
    validation = [
        d.model_copy(update={"doc_id": f"V{d.doc_id}", "split": Split.VAL})
        for d in docs
        if folds.folds[d.doc_id] in {0, 1}
    ]
    pages = PageCache(tmp_path / "pages")
    pages.build(docs)
    material = round_material(docs, folds, DESIGN_FOLD, pages, FakeOcr())
    monkeypatch.setattr("docfraud.forensics.learned.sampler.directory_fonts", lambda _: font)
    forger = CountingForger(
        RoundForger(
            statistics(), docs, folds, CachedReceipts(pages, material.words), tmp_path / "fonts"
        )
    )
    setup = RoundSetup(material, pages, forger, lambda _held_out, seed, prior: Tiny(seed, prior))
    return Toy([*docs, *validation], folds, pages, setup, forger)


def train(
    toy: Toy,
    root: Path,
    runs: tuple[tuple[System, int], ...] = TWO_RUNS,
    *,
    code: CodeVersion = CODE,
    on_run: Callable[[System, int], None] | None = None,
    on_step: Callable[[int, float], None] | None = None,
) -> RoundOutcome:
    return train_round(
        toy.setup,
        root,
        code=code,
        environment=ENVIRONMENT,
        runs=runs,
        settings=SETTINGS,
        device=CPU,
        on_run=on_run,
        on_step=on_step,
    )


@pytest.fixture
def learning(monkeypatch: pytest.MonkeyPatch) -> None:
    """Runs of a few steps count as learned unless they diverge; the rule has its own tests."""
    monkeypatch.setattr(
        "docfraud.forensics.learned.rounds.failure",
        lambda record: "diverged" if record.status == "diverged" else None,
    )


def test_every_system_runs_seed_0_first() -> None:
    assert RUNS[:3] == ((System.SYNTHETIC, 0), (System.REAL, 0), (System.BOTH, 0))
    assert len(RUNS) == len(set(RUNS)) == 9
    assert [seed for _, seed in RUNS] == [0, 0, 0, 1, 1, 1, 2, 2, 2]
    assert run_directory(Path("r"), 2, System.BOTH, 1) == Path("r/round-2/system-4/seed-1")


@pytest.mark.usefixtures("learning")
def test_a_round_trains_its_runs_in_order_each_in_its_directory(toy: Toy, tmp_path: Path) -> None:
    started: list[tuple[System, int]] = []
    steps: list[int] = []
    outcome = train(
        toy,
        tmp_path / "rounds",
        on_run=lambda s, k: started.append((s, k)),
        on_step=lambda d, _: steps.append(d),
    )
    assert outcome.failed is None
    assert started == list(TWO_RUNS)
    assert steps == [1, 2, 3, 4] * 2
    for (system, seed), record in zip(TWO_RUNS, outcome.records, strict=True):
        directory = run_directory(tmp_path / "rounds", DESIGN_FOLD, system, seed)
        assert (directory / MODEL).is_file()
        assert stored_record(tmp_path / "rounds", DESIGN_FOLD, system, seed) == record
        assert (record.held_out, record.system, record.seed, record.status) == (
            DESIGN_FOLD,
            int(system),
            seed,
            "finished",
        )
        sampler = TileSampler(toy.setup.material, toy.pages, system, seed, toy.forger)
        assert record.prior == forged_share(sampler)
    assert stored_record(tmp_path / "rounds", DESIGN_FOLD, System.BOTH, 0) is None


@pytest.mark.usefixtures("learning")
def test_a_system_trained_on_real_forgeries_makes_no_synthetic_one(
    toy: Toy, tmp_path: Path
) -> None:
    train(toy, tmp_path, runs=((System.REAL, 0),))
    assert toy.forger.made == 0
    train(toy, tmp_path, runs=((System.REAL, 0), (System.SYNTHETIC, 0)))
    assert toy.forger.made > 0


def test_the_round_stops_at_the_first_run_that_fails(toy: Toy, tmp_path: Path) -> None:
    def diverging(_held_out: int, seed: int, prior: float) -> nn.Module:
        network = Tiny(seed, prior)
        with torch.no_grad():
            network.conv.bias.fill_(math.nan)  # type: ignore[union-attr]
        return network

    broken = RoundSetup(toy.setup.material, toy.pages, toy.forger, diverging)
    outcome = train_round(
        broken,
        tmp_path,
        code=CODE,
        environment=ENVIRONMENT,
        runs=TWO_RUNS,
        settings=SETTINGS,
        device=CPU,
    )
    assert outcome.failed == (System.SYNTHETIC, 0, "diverged")
    assert len(outcome.records) == 1
    assert stored_record(tmp_path, DESIGN_FOLD, System.REAL, 0) is None
    with pytest.raises(ValueError, match="failed: diverged"):
        finished_records(tmp_path, DESIGN_FOLD, TWO_RUNS)


class Constant(nn.Module):
    """Predicts its initial bias everywhere and cannot learn: its one weight has no effect."""

    def __init__(self, prior: float) -> None:
        super().__init__()
        self.logit = math.log(prior / (1 - prior))
        self.idle = nn.Parameter(torch.zeros(1))

    def forward(self, pages: Tensor) -> Tensor:
        return torch.full_like(pages, self.logit, dtype=torch.float32) + 0 * self.idle


def test_a_run_that_does_not_learn_stops_the_round(toy: Toy, tmp_path: Path) -> None:
    constant = RoundSetup(
        toy.setup.material, toy.pages, toy.forger, lambda _held_out, _seed, prior: Constant(prior)
    )
    outcome = train_round(
        constant,
        tmp_path,
        code=CODE,
        environment=ENVIRONMENT,
        runs=TWO_RUNS,
        settings=SETTINGS,
        device=CPU,
    )
    assert outcome.failed == (System.SYNTHETIC, 0, "does not learn")
    assert len(outcome.records) == 1


@pytest.mark.usefixtures("learning")
def test_an_unfinished_round_continues_only_at_its_commit(toy: Toy, tmp_path: Path) -> None:
    train(toy, tmp_path, runs=TWO_RUNS[:1])
    later = CodeVersion(commit="def", dirty=False)
    with pytest.raises(ValueError, match="continue it there"):
        train(toy, tmp_path, code=later)
    assert stored_record(tmp_path, DESIGN_FOLD, System.REAL, 0) is None
    train(toy, tmp_path)  # finished at its commit
    outcome = train(toy, tmp_path, code=later)  # nothing left to train
    assert [r.code for r in outcome.records] == [CODE, CODE]


@pytest.mark.usefixtures("learning")
def test_only_a_round_whose_runs_all_finished_is_measured(toy: Toy, tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="has not finished"):
        finished_records(tmp_path, DESIGN_FOLD, TWO_RUNS)
    train(toy, tmp_path, runs=TWO_RUNS[:1])
    with pytest.raises(ValueError, match="system 3, seed 0 of round 2 has not finished"):
        finished_records(tmp_path, DESIGN_FOLD, TWO_RUNS)
    train(toy, tmp_path)
    assert [r.system for r in finished_records(tmp_path, DESIGN_FOLD, TWO_RUNS)] == [2, 3]


class WeightsDigest:
    """Maps from each pixel's grey value, named after the weights file it was made from."""

    analysed = 0

    def __init__(self, path: Path) -> None:
        self.identifier = hashlib.sha256(path.read_bytes()).hexdigest()[:12]

    def analyse(self, page: npt.NDArray[np.uint8]) -> LearnedOutput:
        WeightsDigest.analysed += 1
        logits = ((page.astype(np.float32) - 128) / 16).astype(np.float32)
        return LearnedOutput(logits, trufor.tiles(*page.shape), 0.1)


def development_engine(toy: Toy) -> Engine:
    return Engine(statistics(), toy.documents, toy.folds, receipt, font)


@pytest.mark.usefixtures("learning")
def test_a_finished_round_is_measured_once_per_weights(toy: Toy, tmp_path: Path) -> None:
    train(toy, tmp_path)
    items = development_items(toy.documents, toy.pages, development_engine(toy))
    WeightsDigest.analysed = 0
    seen: list[tuple[Group, str]] = []
    measured_runs: list[tuple[System, int]] = []
    results = measure_round(
        tmp_path,
        DESIGN_FOLD,
        items,
        model=WeightsDigest,
        code=CODE,
        environment=ENVIRONMENT,
        runs=TWO_RUNS,
        on_run=lambda s, k: measured_runs.append((s, k)),
        on_progress=lambda group, key: seen.append((group, key)),
    )
    assert measured_runs == list(TWO_RUNS)
    validation = sorted(d.doc_id for d in toy.documents if d.split is Split.VAL)
    design = sorted(d.doc_id for d in toy.documents if toy.folds.folds.get(d.doc_id) == DESIGN_FOLD)
    bases = development_engine(toy).pools[Stream.DEVELOPMENT]
    expected = [
        *((Group.VALIDATION, k) for k in validation),
        *((Group.HELD_OUT, k) for k in design),
        *((Group.SYNTHETIC, k) for k in bases),
    ]
    assert seen == expected * 2
    for (system, seed), result in zip(TWO_RUNS, results, strict=True):
        directory = run_directory(tmp_path, DESIGN_FOLD, system, seed)
        assert read_results(directory) == result
        assert (result.system, result.seed) == (int(system), seed)
        assert [r.key for r in result.validation] == validation
        assert [(r.group, r.key) for r in result.measured] == expected[len(validation) :]
        assert all(r.edits for r in result.measured if r.group is Group.SYNTHETIC)
    analysed = WeightsDigest.analysed
    again = measure_round(
        tmp_path,
        DESIGN_FOLD,
        items,
        model=WeightsDigest,
        code=CODE,
        environment=ENVIRONMENT,
        runs=TWO_RUNS,
    )
    assert again == results
    assert WeightsDigest.analysed == analysed  # nothing measured twice
    directory = run_directory(tmp_path, DESIGN_FOLD, *TWO_RUNS[0])
    (directory / MODEL).write_bytes(b"other weights")
    measure_round(
        tmp_path,
        DESIGN_FOLD,
        items,
        model=WeightsDigest,
        code=CODE,
        environment=ENVIRONMENT,
        runs=TWO_RUNS,
    )
    assert WeightsDigest.analysed == analysed + len(expected)  # only the changed run again
    (directory / RESULTS).write_text('{"written by": "earlier code"}', encoding="utf-8")
    measure_round(
        tmp_path,
        DESIGN_FOLD,
        items,
        model=WeightsDigest,
        code=CODE,
        environment=ENVIRONMENT,
        runs=TWO_RUNS,
    )
    assert WeightsDigest.analysed == analysed + 2 * len(expected)  # an earlier form, measured again
    assert read_results(directory).model == WeightsDigest(directory / MODEL).identifier
    assert (directory / RESULTS).is_file()
    assert (directory / RECORD).is_file()


def test_only_the_round_holding_out_the_design_fold_has_development_items(toy: Toy) -> None:
    other = Engine(
        RoundStatistics(1, statistics().templates, (1, 2), {}, {}),
        toy.documents,
        toy.folds,
        receipt,
        font,
    )
    with pytest.raises(ValueError, match="holds out fold 2"):
        development_items(toy.documents, toy.pages, other)


def test_the_repeat_check_trains_one_run_twice_and_compares_the_weights(
    toy: Toy, tmp_path: Path
) -> None:
    result = repeat_check(
        toy.setup,
        tmp_path,
        code=CODE,
        environment=ENVIRONMENT,
        steps=3,
        settings=SETTINGS,
        device=CPU,
    )
    directory = repeat_directory(tmp_path, DESIGN_FOLD)
    assert (result.system, result.seed, result.steps, result.device) == (4, 0, 3, "cpu")
    assert result.identical  # on the processor, the same seed gives the same weights
    assert result.differing == ()
    assert (result.largest_difference, result.largest_loss_difference) == (0.0, 0.0)
    assert result.tensors == len(Tiny(0, 0.5).state_dict())
    stored = Repeatability.model_validate_json(
        (directory / REPEATABILITY).read_text(encoding="utf-8")
    )
    assert stored == result
    for name in ("first", "second"):
        record = RunRecord.model_validate_json(
            (directory / name / RECORD).read_text(encoding="utf-8")
        )
        assert (record.status, record.steps_done) == ("running", 3)
    again = repeat_check(
        toy.setup,
        tmp_path,
        code=CODE,
        environment=ENVIRONMENT,
        steps=3,
        settings=SETTINGS,
        device=CPU,
    )
    assert again == result  # compared again, not trained again


def test_the_repeat_check_reports_weights_that_differ(toy: Toy, tmp_path: Path) -> None:
    calls = 0

    def drifting(_held_out: int, seed: int, prior: float) -> nn.Module:
        nonlocal calls
        calls += 1
        network = Tiny(seed, prior)
        if calls == 2:  # the second training starts from other weights
            with torch.no_grad():
                network.conv.weight.add_(0.5)
        return network

    setup = RoundSetup(toy.setup.material, toy.pages, toy.forger, drifting)
    result = repeat_check(
        setup, tmp_path, code=CODE, environment=ENVIRONMENT, steps=3, settings=SETTINGS, device=CPU
    )
    assert not result.identical
    assert "conv.weight" in result.differing
    assert result.largest_difference > 0.4
    assert result.largest_loss_difference > 0


def test_the_repeat_check_needs_fewer_steps_than_a_run(toy: Toy, tmp_path: Path) -> None:
    for steps in (0, SETTINGS.steps):
        with pytest.raises(ValueError, match="fewer steps"):
            repeat_check(
                toy.setup,
                tmp_path,
                code=CODE,
                environment=ENVIRONMENT,
                steps=steps,
                settings=SETTINGS,
                device=CPU,
            )
