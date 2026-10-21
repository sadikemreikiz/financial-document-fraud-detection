import pickle
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from docfraud.data.models import BBox, Document, Entity, Label, Technique
from docfraud.evaluation.folds import FoldAssignment
from docfraud.forensics.learned.pages import PageCache
from docfraud.forensics.learned.sampler import (
    ATTEMPTS,
    MAX_STEPS,
    NEGATIVES,
    POSITIVES,
    CachedReceipts,
    Kind,
    RealForgery,
    RoundForger,
    RoundMaterial,
    System,
    TileSampler,
    TileSource,
    forged_share,
    forgery_index,
    reachable,
    round_material,
)
from docfraud.forensics.learned.tiles import CLEAN, FORGED, IGNORED, TILE, cut, flip, labels
from docfraud.synthesis.engine import Forgery
from docfraud.synthesis.statistics import RoundStatistics, Template
from tests.synthesis.pages import documents, font
from tests.synthesis.test_statistics import FakeOcr

HELD_OUT = 1


@dataclass(frozen=True)
class Setup:
    documents: list[Document]
    folds: FoldAssignment
    pages: PageCache
    material: RoundMaterial
    forger: RoundForger


def statistics() -> RoundStatistics:
    """A round whose real edits were all copies inside the receipt, so sources are on the page."""
    entities = (Entity.PRODUCT, Entity.TOTAL_PAYMENT, Entity.METADATA, Entity.OTHER)
    templates = tuple(Template((Technique.CPI,), entity, 2) for entity in entities)
    return RoundStatistics(HELD_OUT, templates, (1, 2), {}, {})


@pytest.fixture
def setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Setup:
    docs, folds = documents(tmp_path)
    pages = PageCache(tmp_path / "pages")
    pages.build(docs)
    material = round_material(docs, folds, HELD_OUT, pages, FakeOcr())
    monkeypatch.setattr("docfraud.forensics.learned.sampler.directory_fonts", lambda _: font)
    receipts = CachedReceipts(pages, material.words)
    forger = RoundForger(statistics(), docs, folds, receipts, tmp_path / "fonts")
    return Setup(docs, folds, pages, material, forger)


def sampler(setup: Setup, system: System, seed: int = 0) -> TileSampler:
    return TileSampler(setup.material, setup.pages, system, seed, setup.forger)


def unflipped(tile: npt.NDArray[np.uint8], source: TileSource) -> npt.NDArray[np.uint8]:
    return flip(tile, vertical=source.vertical, horizontal=source.horizontal)


def test_the_material_is_the_training_folds_of_the_round(setup: Setup) -> None:
    material = setup.material
    folds = setup.folds.folds
    assert material.held_out == HELD_OUT
    assert len(material.forged) == 4
    assert len(material.authentic) == 8
    assert all(folds[f.doc_id] != HELD_OUT for f in material.forged)
    assert all(folds[doc_id] != HELD_OUT for doc_id in material.authentic)
    assert material.words.keys() == set(material.authentic)
    assert all(material.words[doc_id] for doc_id in material.authentic)
    assert material.left_out == ()
    assert all(f.modified and not f.sources for f in material.forged)


def test_a_forged_receipt_without_regions_is_left_out(setup: Setup) -> None:
    docs = [
        d.model_copy(update={"regions": (), "has_region_annotations": False})
        if d.label is Label.FORGED and d.doc_id == "X00"
        else d
        for d in setup.documents
    ]
    material = round_material(docs, setup.folds, HELD_OUT, setup.pages, FakeOcr())
    assert material.left_out == ("X00",)
    assert len(material.forged) == 3


@pytest.mark.parametrize(
    ("system", "kinds"),
    [
        (System.SYNTHETIC, [Kind.SYNTHETIC] * 4),
        (System.REAL, [Kind.REAL] * 4),
        (System.BOTH, [Kind.REAL, Kind.REAL, Kind.SYNTHETIC, Kind.SYNTHETIC]),
    ],
)
def test_a_batch_follows_the_plan_of_its_system(
    setup: Setup, system: System, kinds: list[Kind]
) -> None:
    batch = sampler(setup, system).batch(0)
    tiles = POSITIVES + NEGATIVES
    assert batch.pages.shape == batch.labels.shape == (tiles, TILE, TILE)
    assert batch.pages.dtype == batch.labels.dtype == np.uint8
    clean = [Kind.WORD, Kind.WORD, Kind.UNIFORM, Kind.UNIFORM]
    assert [s.kind for s in batch.sources] == kinds + clean


def assert_same_tiles(first: TileSampler, second: TileSampler, step: int, slots: slice) -> None:
    one, other = first.batch(step), second.batch(step)
    assert one.sources[slots] == other.sources[slots]
    assert np.array_equal(one.pages[slots], other.pages[slots])
    assert np.array_equal(one.labels[slots], other.labels[slots])


def test_a_step_always_gives_the_same_tiles(setup: Setup) -> None:
    for system in System:
        for step in (0, 7):
            assert_same_tiles(sampler(setup, system), sampler(setup, system), step, slice(None))


def test_other_steps_and_seeds_draw_other_tiles(setup: Setup) -> None:
    run = sampler(setup, System.BOTH)
    assert run.batch(0).sources != run.batch(1).sources
    assert run.batch(0).sources != sampler(setup, System.BOTH, seed=1).batch(0).sources


def test_systems_with_one_seed_share_their_clean_tiles(setup: Setup) -> None:
    clean = slice(POSITIVES, None)
    for step in range(5):
        assert_same_tiles(
            sampler(setup, System.REAL), sampler(setup, System.SYNTHETIC), step, clean
        )
        assert_same_tiles(sampler(setup, System.REAL), sampler(setup, System.BOTH), step, clean)


def test_system_4_shares_the_first_edited_tiles_of_systems_2_and_3(setup: Setup) -> None:
    for step in range(5):
        both = sampler(setup, System.BOTH).batch(step)
        real = sampler(setup, System.REAL).batch(step)
        synthetic = sampler(setup, System.SYNTHETIC).batch(step)
        assert both.sources[:2] == real.sources[:2]
        assert both.sources[2:4] == synthetic.sources[:2]
        assert np.array_equal(both.pages[:2], real.pages[:2])
        assert np.array_equal(both.pages[2:4], synthetic.pages[:2])
        assert np.array_equal(both.labels[2:4], synthetic.labels[:2])


def test_real_tiles_show_a_region_with_its_labels(setup: Setup) -> None:
    run = sampler(setup, System.REAL)
    forged = {f.doc_id: f for f in setup.material.forged}
    for step in range(10):
        batch = run.batch(step)
        for tile, label, source in zip(batch.pages, batch.labels, batch.sources, strict=True):
            if source.kind is not Kind.REAL:
                continue
            page = setup.pages.page(source.doc_id)
            assert np.array_equal(unflipped(tile, source), cut(page, source.corner))
            forgery = forged[source.doc_id]
            expected = labels(source.corner, forgery.modified, forgery.sources)
            assert np.array_equal(unflipped(label, source), expected)
            assert (label == FORGED).any()


def test_copy_sources_of_real_forgeries_are_ignored(setup: Setup) -> None:
    forgery = setup.material.forged[0]
    source = BBox(x=40, y=40, width=50, height=20)
    material = replace(
        setup.material, forged=(RealForgery(forgery.doc_id, forgery.modified, (source,)),)
    )
    batch = TileSampler(material, setup.pages, System.REAL, 0).batch(0)
    assert all(s.doc_id == forgery.doc_id for s in batch.sources[:POSITIVES])
    assert all((label == IGNORED).sum() == source.area for label in batch.labels[:POSITIVES])


def test_synthetic_tiles_show_a_fresh_forgery_with_its_labels(setup: Setup) -> None:
    run = sampler(setup, System.SYNTHETIC)
    indexes = []
    ignored = 0
    for step in range(10):
        batch = run.batch(step)
        for tile, label, source in zip(batch.pages, batch.labels, batch.sources, strict=True):
            if source.kind is not Kind.SYNTHETIC:
                continue
            assert source.forgery is not None
            indexes.append(source.forgery)
            forgery = setup.forger.forge(source.doc_id, source.forgery)
            assert np.array_equal(unflipped(tile, source), cut(forgery.page, source.corner))
            regions = forgery.record.regions
            copied = [box for r in regions if r.source_document is None for box in r.sources]
            expected = labels(source.corner, [r.box for r in regions], copied)
            assert np.array_equal(unflipped(label, source), expected)
            assert (label == FORGED).any()
            ignored += int((label == IGNORED).sum())
    assert len(set(indexes)) == len(indexes) == 40
    assert ignored > 0


class Unlucky:
    """A forger whose first attempts place no edit."""

    def __init__(self, forger: RoundForger, failures: int) -> None:
        self.forger = forger
        self.failures = failures

    def forge(self, base: str, index: int) -> Forgery:
        forgery = self.forger.forge(base, index)
        if index % ATTEMPTS < self.failures:
            record = forgery.record.model_copy(update={"regions": ()})
            return Forgery(forgery.page, forgery.changed, forgery.written, record)
        return forgery


def test_a_forgery_without_an_edit_is_skipped(setup: Setup) -> None:
    run = TileSampler(setup.material, setup.pages, System.SYNTHETIC, 0, Unlucky(setup.forger, 1))
    for source in run.batch(0).sources[:POSITIVES]:
        assert source.forgery is not None
        assert source.forgery % ATTEMPTS == 1


def test_a_run_stops_when_no_forgery_has_an_edit(setup: Setup) -> None:
    forger = Unlucky(setup.forger, ATTEMPTS)
    with pytest.raises(RuntimeError, match="no usable forgery"):
        TileSampler(setup.material, setup.pages, System.SYNTHETIC, 0, forger).batch(0)


def test_clean_tiles_are_clean_and_words_lie_in_word_tiles(setup: Setup) -> None:
    run = sampler(setup, System.REAL)
    for step in range(10):
        batch = run.batch(step)
        for label, source in zip(batch.labels[POSITIVES:], batch.sources[POSITIVES:], strict=True):
            assert source.doc_id in setup.material.authentic
            assert (label == CLEAN).all()
            if source.kind is Kind.WORD:
                y, x = source.corner
                window = BBox(x=x, y=y, width=TILE, height=TILE)
                assert any(w.box.overlaps(window) for w in setup.material.words[source.doc_id])


def test_tiles_are_flipped_both_ways(setup: Setup) -> None:
    sources = [s for step in range(20) for s in sampler(setup, System.REAL).batch(step).sources]
    assert {s.vertical for s in sources} == {s.horizontal for s in sources} == {False, True}


def test_forgery_indexes_differ_for_every_seed_step_slot_and_attempt() -> None:
    drawn = [
        forgery_index(seed, step, slot, attempt)
        for seed in range(3)
        for step in (0, 1, 2, MAX_STEPS - 1)
        for slot in range(POSITIVES)
        for attempt in range(ATTEMPTS)
    ]
    assert len(set(drawn)) == len(drawn)


@pytest.mark.parametrize(
    ("seed", "step", "slot", "attempt"),
    [(-1, 0, 0, 0), (0, -1, 0, 0), (0, MAX_STEPS, 0, 0), (0, 0, POSITIVES, 0), (0, 0, 0, ATTEMPTS)],
)
def test_forgery_indexes_have_limits(seed: int, step: int, slot: int, attempt: int) -> None:
    with pytest.raises(ValueError, match=r"no forgery|not negative"):
        forgery_index(seed, step, slot, attempt)


def test_a_sampler_needs_what_its_system_trains_on(setup: Setup) -> None:
    with pytest.raises(ValueError, match="needs a forger"):
        TileSampler(setup.material, setup.pages, System.BOTH, 0)
    no_real = replace(setup.material, forged=())
    with pytest.raises(ValueError, match="no real forgery"):
        TileSampler(no_real, setup.pages, System.REAL, 0)
    TileSampler(no_real, setup.pages, System.SYNTHETIC, 0, setup.forger)
    no_words = replace(setup.material, words=dict.fromkeys(setup.material.authentic, ()))
    with pytest.raises(ValueError, match="OCR words"):
        TileSampler(no_words, setup.pages, System.REAL, 0)
    with pytest.raises(ValueError, match="not negative"):
        TileSampler(setup.material, setup.pages, System.REAL, -1)


@pytest.mark.parametrize("step", [-1, MAX_STEPS])
def test_steps_have_limits(setup: Setup, step: int) -> None:
    with pytest.raises(ValueError, match="no step"):
        sampler(setup, System.REAL).batch(step)


def test_boxes_no_tile_reaches_are_left_out(setup: Setup) -> None:
    height, width = setup.pages.page(setup.material.authentic[0]).shape
    edge = width - (width - TILE) // 8 * 8 - TILE  # pixels beyond the last aligned tile
    assert edge > 0
    assert not reachable(BBox(x=width - edge, y=0, width=edge, height=5), (height, width))
    assert reachable(BBox(x=0, y=0, width=5, height=5), (height, width))
    forgery = setup.material.forged[0]
    beyond = BBox(x=width - edge, y=10, width=edge, height=5)
    material = replace(
        setup.material, forged=(RealForgery(forgery.doc_id, (*forgery.modified, beyond), ()),)
    )
    assert TileSampler(material, setup.pages, System.REAL, 0).unreachable == 1


def test_the_forged_share_counts_the_pixels_the_loss_counts(setup: Setup) -> None:
    run = sampler(setup, System.BOTH)
    batches = [run.batch(step).labels for step in range(3)]
    forged = sum(int((b == FORGED).sum()) for b in batches)
    counted = sum(int((b != IGNORED).sum()) for b in batches)
    assert forged_share(run, steps=3) == forged / counted
    assert 0 < forged / counted < 1


def test_a_share_without_forged_pixels_is_refused(
    setup: Setup, monkeypatch: pytest.MonkeyPatch
) -> None:
    run = sampler(setup, System.REAL)
    clean = run.batch(0)
    monkeypatch.setattr(run, "batch", lambda _: replace(clean, labels=np.zeros_like(clean.labels)))
    with pytest.raises(ValueError, match="forged"):
        forged_share(run, steps=2)


def test_a_sampler_travels_to_another_process_without_its_engine(setup: Setup) -> None:
    run = sampler(setup, System.SYNTHETIC)
    first = run.batch(3)
    copy = pickle.loads(pickle.dumps(run))  # noqa: S301 - our own object
    assert isinstance(copy.forger, RoundForger)
    assert copy.forger._engine is None
    second = copy.batch(3)
    assert first.sources == second.sources
    assert np.array_equal(first.pages, second.pages)


def test_cached_receipts_map_pages_and_carry_words(setup: Setup) -> None:
    doc_id = setup.material.authentic[0]
    receipt = CachedReceipts(setup.pages, setup.material.words)(doc_id)
    assert isinstance(receipt.page, np.memmap)
    assert receipt.words == setup.material.words[doc_id]
