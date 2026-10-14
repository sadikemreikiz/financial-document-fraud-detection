"""The tiles of every training step of systems 2 to 4 (ADR-0022, decisions 3 and 4).

A batch holds ``POSITIVES`` tiles with an edit, then ``NEGATIVES`` tiles without one. A tile with
an edit comes from a real forged receipt of the round's training folds (system 3), from a fresh
synthetic forgery of one of its authentic receipts (system 2), or half from each (system 4): a
receipt is drawn uniformly, then one of its modified regions, and the tile is drawn around that
region (:mod:`~docfraud.forensics.learned.tiles`). Of the tiles without an edit, all from authentic
receipts drawn uniformly, the first ``AROUND_WORDS`` lie around an OCR word drawn uniformly and the
others anywhere on the page. Every tile is flipped vertically and horizontally, each with
probability one half.

Each slot of a batch has its own generator, seeded by the round, the run's seed, the purpose of
the draw, the step and the slot (:func:`generator`). The tiles of a step therefore depend only on
the round, the system, the seed and the step, so that a resumed run sees the same tiles, and the
three systems with the same seed see the same tiles without an edit. System 4 takes the first two
slots of the real and of the synthetic draws, so its tiles with an edit are those that system 3
and system 2 draw first at the same step: a difference between the systems then comes from their
training data rather than from which tiles each happened to draw (common random numbers).

Synthetic forgeries come from the round's frozen engine (ADR-0021), on the training stream; the
index of each (:func:`forgery_index`) encodes the seed, the step, the slot and the attempt, so
that no forgery is used twice and each seed draws its own. A forgery in which no edit could be
placed is skipped for the next attempt.
"""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from enum import IntEnum, StrEnum
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from docfraud.data.models import BBox, Document, Label, RegionRole
from docfraud.evaluation.folds import FoldAssignment
from docfraud.forensics.learned.pages import PageCache
from docfraud.forensics.learned.tiles import (
    FORGED,
    IGNORED,
    axis_positions,
    cut,
    flip,
    labels,
    position_around,
    uniform_position,
)
from docfraud.reading.ocr import OcrEngine
from docfraud.synthesis import layout
from docfraud.synthesis.engine import Engine, Forgery, Receipt, Stream, directory_fonts
from docfraud.synthesis.statistics import RoundStatistics, training_documents

SAMPLER_SEED = 20261015
"""First element of the seed of every generator of this module."""

POSITIVES = 4
"""Tiles with an edit in a batch."""

NEGATIVES = 4
"""Tiles without an edit in a batch."""

AROUND_WORDS = 2
"""Tiles without an edit that lie around an OCR word; the others lie anywhere."""

SHARED = 2
"""Tiles of each kind of forgery in a batch of system 4."""

PRIOR_STEPS = 100
"""Steps whose tiles estimate the share of forged pixels that sets the initial output bias."""

MAX_STEPS = 1_000_000
"""Steps a run may have; part of the arithmetic of :func:`forgery_index`."""

ATTEMPTS = 16
"""Synthetic forgeries tried for one tile before the run stops."""


class System(IntEnum):
    """The learned systems of ADR-0014, by what their tiles with an edit come from."""

    SYNTHETIC = 2
    REAL = 3
    BOTH = 4


class Purpose(IntEnum):
    """What a generator draws; part of its seed."""

    WEIGHTS = 0  # the initial weights of the decoder
    CLEAN = 1  # tiles without an edit
    REAL = 2  # tiles of real forgeries
    SYNTHETIC = 3  # tiles of synthetic forgeries


class Kind(StrEnum):
    """Where a tile comes from."""

    REAL = "real"
    SYNTHETIC = "synthetic"
    WORD = "word"  # an authentic receipt, around an OCR word
    UNIFORM = "uniform"  # an authentic receipt, anywhere


PLANS: dict[System, tuple[tuple[Purpose, int], ...]] = {
    System.SYNTHETIC: tuple((Purpose.SYNTHETIC, slot) for slot in range(POSITIVES)),
    System.REAL: tuple((Purpose.REAL, slot) for slot in range(POSITIVES)),
    System.BOTH: (
        *((Purpose.REAL, slot) for slot in range(SHARED)),
        *((Purpose.SYNTHETIC, slot) for slot in range(SHARED)),
    ),
}
"""The draws behind the tiles with an edit of each system, in batch order."""


def generator(held_out: int, seed: int, purpose: Purpose, *path: int) -> np.random.Generator:
    """The generator of one purpose of a run, for example of one slot of one step."""
    return np.random.default_rng([SAMPLER_SEED, held_out, seed, int(purpose), *path])


def forgery_index(seed: int, step: int, slot: int, attempt: int) -> int:
    """The engine's index of a synthetic forgery, different for every seed, step, slot, attempt.

    Raises:
        ValueError: If a part is negative or beyond its limit.
    """
    if not (0 <= step < MAX_STEPS and 0 <= slot < POSITIVES and 0 <= attempt < ATTEMPTS):
        msg = f"no forgery for step {step}, slot {slot}, attempt {attempt}"
        raise ValueError(msg)
    if seed < 0:
        msg = f"seeds are not negative, not {seed}"
        raise ValueError(msg)
    return ((seed * MAX_STEPS + step) * POSITIVES + slot) * ATTEMPTS + attempt


@dataclass(frozen=True, slots=True)
class RealForgery:
    """A real forged receipt as training sees it."""

    doc_id: str
    modified: tuple[BBox, ...]  # the annotated modified rectangles
    sources: tuple[BBox, ...]  # the annotated copy sources, left out of the loss


@dataclass(frozen=True, slots=True)
class RoundMaterial:
    """What the tiles of one cross-validation round are drawn from."""

    held_out: int
    forged: tuple[RealForgery, ...]  # forged training receipts with a modified region
    authentic: tuple[str, ...]  # authentic training receipts
    words: Mapping[str, tuple[layout.Word, ...]]  # the OCR words of every authentic receipt
    left_out: tuple[str, ...]  # forged training receipts without a modified region


def round_material(
    documents: Sequence[Document],
    folds: FoldAssignment,
    held_out: int,
    pages: PageCache,
    ocr: OcrEngine,
) -> RoundMaterial:
    """The training material of the round that holds out one fold.

    Raises:
        ValueError: If the fold does not exist.
    """
    training = sorted(training_documents(documents, folds, held_out), key=lambda d: d.doc_id)
    forged: list[RealForgery] = []
    left_out: list[str] = []
    for document in (d for d in training if d.label is Label.FORGED):
        modified = tuple(r.box for r in document.regions if r.role is RegionRole.MODIFIED)
        if not document.has_region_annotations or not modified:
            left_out.append(document.doc_id)
            continue
        sources = tuple(r.box for r in document.regions if r.role is RegionRole.SOURCE)
        forged.append(RealForgery(document.doc_id, modified, sources))
    authentic = tuple(d.doc_id for d in training if d.label is Label.AUTHENTIC)
    words: dict[str, tuple[layout.Word, ...]] = {}
    for doc_id in authentic:
        page = pages.page(doc_id)
        words[doc_id] = layout.words(page, ocr.recognize(page))
    return RoundMaterial(held_out, tuple(forged), authentic, words, tuple(left_out))


@dataclass(frozen=True, slots=True)
class CachedReceipts:
    """Receipts as the synthetic engine sees them, with pages from the page cache."""

    pages: PageCache
    words: Mapping[str, tuple[layout.Word, ...]]

    def __call__(self, doc_id: str) -> Receipt:
        """The receipt with its mapped page and its OCR words."""
        return Receipt(doc_id, self.pages.page(doc_id), self.words[doc_id])


class Forger(Protocol):
    """Anything that makes synthetic forgeries of a round's training stream."""

    def forge(self, base: str, index: int) -> Forgery:
        """Forgery ``index`` of an authentic receipt."""
        ...


class RoundForger:
    """The frozen engine of a round, made in each process when it is first used.

    The engine reads its fonts through a cache that cannot be sent to another process, so a
    process that receives the forger makes its own engine.
    """

    def __init__(
        self,
        statistics: RoundStatistics,
        documents: Sequence[Document],
        folds: FoldAssignment,
        receipts: CachedReceipts,
        fonts: Path,
    ) -> None:
        """Keep what the engine is made of."""
        self.statistics = statistics
        self.documents = tuple(documents)
        self.folds = folds
        self.receipts = receipts
        self.fonts = fonts
        self._engine: Engine | None = None

    def forge(self, base: str, index: int) -> Forgery:
        """Forgery ``index`` of a base receipt, on the training stream."""
        if self._engine is None:
            self._engine = Engine(
                self.statistics,
                self.documents,
                self.folds,
                self.receipts,
                directory_fonts(self.fonts),
            )
        return self._engine.forge(base, index, Stream.TRAINING)

    def __getstate__(self) -> dict[str, Any]:
        """Everything but the engine."""
        state = dict(self.__dict__)
        state["_engine"] = None
        return state


@dataclass(frozen=True, slots=True)
class TileSource:
    """Where a tile of a batch comes from."""

    kind: Kind
    doc_id: str  # the receipt; for a synthetic forgery, its base
    corner: tuple[int, int]  # top-left pixel (y, x) on the page
    vertical: bool  # flipped top to bottom
    horizontal: bool  # flipped left to right
    forgery: int | None = None  # the engine's index of a synthetic forgery


@dataclass(frozen=True, slots=True)
class Batch:
    """The tiles of one training step and the labels of their pixels."""

    pages: npt.NDArray[np.uint8]  # (tiles, TILE, TILE)
    labels: npt.NDArray[np.uint8]  # (tiles, TILE, TILE): CLEAN, FORGED or IGNORED
    sources: tuple[TileSource, ...]


type _Tile = tuple[npt.NDArray[np.uint8], npt.NDArray[np.uint8], TileSource]


def reachable(box: BBox, shape: tuple[int, int]) -> bool:
    """Whether an aligned tile can be drawn around a box on a page of this shape."""
    height, width = shape
    return bool(
        axis_positions(box.y, min(box.y2, height), height).size
        and axis_positions(box.x, min(box.x2, width), width).size
    )


class TileSampler:
    """The batches of one run: a round, a system and a seed.

    Boxes no aligned tile can reach, in the last few pixels of a page, are left out when the
    sampler is made (:attr:`unreachable`).

    Raises:
        ValueError: If the system needs forgeries the round or the arguments do not provide.
    """

    def __init__(
        self,
        material: RoundMaterial,
        pages: PageCache,
        system: System,
        seed: int,
        forger: Forger | None = None,
    ) -> None:
        """Prepare the draws of a run."""
        purposes = {purpose for purpose, _ in PLANS[system]}
        if Purpose.SYNTHETIC in purposes and forger is None:
            msg = f"system {int(system)} trains on synthetic forgeries and needs a forger"
            raise ValueError(msg)
        if seed < 0:
            msg = f"seeds are not negative, not {seed}"
            raise ValueError(msg)
        self.material = material
        self.pages = pages
        self.system = system
        self.seed = seed
        self.forger = forger
        self.unreachable = 0
        self._forged: list[tuple[RealForgery, tuple[BBox, ...]]] = []
        for forgery in material.forged:
            shape = self._shape(forgery.doc_id)
            kept = tuple(box for box in forgery.modified if reachable(box, shape))
            self.unreachable += len(forgery.modified) - len(kept)
            if kept:
                self._forged.append((forgery, kept))
        self._words: dict[str, tuple[BBox, ...]] = {}
        for doc_id in material.authentic:
            shape = self._shape(doc_id)
            boxes = tuple(w.box for w in material.words[doc_id])
            kept = tuple(box for box in boxes if reachable(box, shape))
            self.unreachable += len(boxes) - len(kept)
            if kept:
                self._words[doc_id] = kept
        self._worded = tuple(sorted(self._words))
        if Purpose.REAL in purposes and not self._forged:
            msg = f"round {material.held_out} has no real forgery to train system {int(system)}"
            raise ValueError(msg)
        if not material.authentic or not self._worded:
            msg = f"round {material.held_out} has no authentic receipt with OCR words"
            raise ValueError(msg)

    def _shape(self, doc_id: str) -> tuple[int, int]:
        height, width = self.pages.page(doc_id).shape
        return height, width

    def batch(self, step: int) -> Batch:
        """The tiles of a step: those with an edit first, then those without.

        Raises:
            ValueError: If the step is negative or beyond ``MAX_STEPS``.
        """
        if not 0 <= step < MAX_STEPS:
            msg = f"there is no step {step}"
            raise ValueError(msg)
        tiles = [self._edited(purpose, step, slot) for purpose, slot in PLANS[self.system]]
        tiles += [self._clean(step, slot) for slot in range(NEGATIVES)]
        return Batch(
            np.stack([page for page, _, _ in tiles]),
            np.stack([label for _, label, _ in tiles]),
            tuple(source for _, _, source in tiles),
        )

    def _edited(self, purpose: Purpose, step: int, slot: int) -> _Tile:
        rng = self._generator(purpose, step, slot)
        if purpose is Purpose.REAL:
            forgery, boxes = self._forged[int(rng.integers(len(self._forged)))]
            box = boxes[int(rng.integers(len(boxes)))]
            page = self.pages.page(forgery.doc_id)
            corner = position_around(box, (page.shape[0], page.shape[1]), rng)
            if corner is None:  # pragma: no cover - unreachable boxes were left out
                msg = f"no tile around {box} of {forgery.doc_id}"
                raise RuntimeError(msg)
            source = (Kind.REAL, forgery.doc_id, corner, None)
            return self._finish(
                page, labels(corner, forgery.modified, forgery.sources), source, rng
            )
        return self._synthetic(rng, step, slot)

    def _synthetic(self, rng: np.random.Generator, step: int, slot: int) -> _Tile:
        if self.forger is None:  # pragma: no cover - checked when the sampler is made
            msg = "no forger"
            raise RuntimeError(msg)
        authentic = self.material.authentic
        for attempt in range(ATTEMPTS):
            base = authentic[int(rng.integers(len(authentic)))]
            index = forgery_index(self.seed, step, slot, attempt)
            forgery = self.forger.forge(base, index)
            regions = forgery.record.regions
            if not regions:
                continue
            region = regions[int(rng.integers(len(regions)))]
            height, width = forgery.page.shape
            corner = position_around(region.box, (height, width), rng)
            if corner is None:
                continue
            modified = [r.box for r in regions]
            copied = [box for r in regions if r.source_document is None for box in r.sources]
            source = (Kind.SYNTHETIC, base, corner, index)
            return self._finish(forgery.page, labels(corner, modified, copied), source, rng)
        msg = f"no usable forgery in {ATTEMPTS} attempts at step {step}, slot {slot}"
        raise RuntimeError(msg)

    def _clean(self, step: int, slot: int) -> _Tile:
        rng = self._generator(Purpose.CLEAN, step, slot)
        corner: tuple[int, int] | None
        if slot < AROUND_WORDS:
            doc_id = self._worded[int(rng.integers(len(self._worded)))]
            boxes = self._words[doc_id]
            page = self.pages.page(doc_id)
            box = boxes[int(rng.integers(len(boxes)))]
            corner = position_around(box, (page.shape[0], page.shape[1]), rng)
            kind = Kind.WORD
        else:
            authentic = self.material.authentic
            doc_id = authentic[int(rng.integers(len(authentic)))]
            page = self.pages.page(doc_id)
            corner = uniform_position((page.shape[0], page.shape[1]), rng)
            kind = Kind.UNIFORM
        if corner is None:  # pragma: no cover - unreachable words were left out
            msg = f"no tile around a word of {doc_id}"
            raise RuntimeError(msg)
        return self._finish(page, labels(corner, (), ()), (kind, doc_id, corner, None), rng)

    def _finish(
        self,
        page: npt.NDArray[np.uint8],
        label: npt.NDArray[np.uint8],
        drawn: tuple[Kind, str, tuple[int, int], int | None],
        rng: np.random.Generator,
    ) -> _Tile:
        vertical, horizontal = (bool(v) for v in rng.integers(2, size=2))
        kind, doc_id, corner, forgery = drawn
        return (
            flip(cut(page, corner), vertical=vertical, horizontal=horizontal),
            flip(label, vertical=vertical, horizontal=horizontal),
            TileSource(kind, doc_id, corner, vertical, horizontal, forgery),
        )

    def _generator(self, purpose: Purpose, step: int, slot: int) -> np.random.Generator:
        return generator(self.material.held_out, self.seed, purpose, step, slot)


def forged_share(sampler: TileSampler, steps: int = PRIOR_STEPS) -> float:
    """The share of forged pixels, among those the loss counts, in the tiles of the first steps.

    Raises:
        ValueError: If no pixel or every pixel is forged.
    """
    forged = counted = 0
    for step in range(steps):
        batch_labels = sampler.batch(step).labels
        forged += int((batch_labels == FORGED).sum())
        counted += int((batch_labels != IGNORED).sum())
    if not 0 < forged < counted:
        msg = f"{forged} of {counted} pixels are forged in the first {steps} steps"
        raise ValueError(msg)
    return forged / counted
