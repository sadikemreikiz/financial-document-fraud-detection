"""The synthetic forgery engine of ADR-0021: edits of printed text on authentic receipts.

A forgery is made from one authentic base receipt by one generator, seeded by the round, the
base and an index, so that it can be made again instead of stored and does not depend on which
other forgeries are made. Its regions follow templates of the round's real regions
(:mod:`docfraud.synthesis.statistics`); every variant of how an edit looks is equally likely
(decision 3). The engine writes into a copy of the normalised page and keeps two masks: the
pixels whose value changed, and the area it wrote, which also holds written pixels that kept
their value. No pixel outside the written area changes. A region is planned completely before
anything is written, so it is either applied whole or not at all.

The engine refuses bases that the stream may not use (decision 4): in training, only authentic
receipts of the round's training folds; in the development stream, only authentic receipts of
the design fold, in the round that holds it out. Copies from another receipt (CPO) take their
donor from the same pool, so that a forgery a person may view contains no pixel of a clean fold.
"""

import math
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from enum import IntEnum
from functools import lru_cache
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict

from docfraud.data.models import BBox, Document, Entity, Label, Split, Technique
from docfraud.evaluation.design_fold import DESIGN_FOLD, DESIGN_FOLD_IDS
from docfraud.evaluation.folds import FoldAssignment
from docfraud.evaluation.localisation import receipt_key
from docfraud.synthesis.fonts import FONTS, font_path
from docfraud.synthesis.layout import Character, Kind, Word
from docfraud.synthesis.statistics import RoundStatistics, Template

ENGINE_VERSION = 1
"""Version of the engine; changes whenever the forgeries for a seed may change."""

ENGINE_SEED = 0
"""First element of every forgery's seed."""

TRIES = 20
"""Templates tried for a region before it is given up."""

WORDS_PER_TEMPLATE = 5
"""Words of a template's entity tried before the template is set aside."""

COVER_MARGIN = 1
"""Pixels of paper beyond the ink that covering the old characters includes."""

PASTE_MARGIN = 2
"""Pixels of paper around a copied character that a rectangle paste takes with it."""

KEEP_CLEAR = 2
"""Pixels around an edit and its source that later regions of the same forgery do not touch."""

TONE_STEP = 0.25
"""Share of the contrast between paper and ink by which a darker or lighter ink differs."""

SCALE = 0.05
"""Largest relative change of size of a slightly scaled paste."""

ROTATION = 1.5
"""Largest angle, in degrees, of a slightly rotated paste."""

SEARCH_STEPS = 6
"""Steps of its own size in each direction searched for blank paper to copy over an edit."""

SUPERSAMPLING = 4
"""Factor by which pen strokes are drawn finer before their coverage is averaged."""

HARD_PEN = 0.5
"""Coverage from which a pixel of a hard pen stroke is fully inked."""

_ALPHABETS = {
    Kind.DIGIT: "0123456789",
    Kind.UPPER: "ABCDEFGHIJKLMNOPQRSTUVWXYZ",
    Kind.LOWER: "abcdefghijklmnopqrstuvwxyz",
}
_REFERENCE = {Kind.DIGIT: "0", Kind.UPPER: "H", Kind.LOWER: "x"}

type FontSource = Callable[[str, bool, int], ImageFont.FreeTypeFont]
"""A font of a family, regular (``False``) or bold (``True``), at a size in pixels."""

type Mask = npt.NDArray[np.bool_]
type Grey = npt.NDArray[np.float64]


class Stream(IntEnum):
    """Which forgeries a generator makes; the streams never share a seed."""

    TRAINING = 0
    DEVELOPMENT = 1  # the held-out synthetic set of the development round (decision 6)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class SyntheticRegion(_Frozen):
    """One edit of a synthetic forgery."""

    technique: Technique
    entity: Entity
    box: BBox  # encloses the area this edit wrote
    sources: tuple[BBox, ...] = ()  # the copied characters, one box each (CPI, CPO)
    source_document: str | None = None  # the donor receipt of a CPO
    span: int  # characters of the word the edit spans, separators included
    before: str  # the printed characters of the word before the edit
    after: str | None  # after it; None when a free-hand edit leaves no known character
    variants: dict[str, str]


class ForgeryRecord(_Frozen):
    """What a synthetic forgery is made of; enough to make it again."""

    engine_version: int
    base: str
    held_out: int
    stream: Stream
    index: int
    regions: tuple[SyntheticRegion, ...]
    failures: dict[str, int]  # templates that could not be placed, by reason


@dataclass(frozen=True, slots=True)
class Receipt:
    """A receipt as the engine sees it: its normalised page and its words."""

    doc_id: str
    page: npt.NDArray[np.uint8]
    words: tuple[Word, ...]


type ReceiptSource = Callable[[str], Receipt]


@dataclass(frozen=True, slots=True)
class Forgery:
    """A synthetic forgery: the edited page, its two masks and its record."""

    page: npt.NDArray[np.uint8]
    changed: Mask  # pixels whose value differs from the base
    written: Mask  # pixels the engine wrote, changed or not
    record: ForgeryRecord


class RefusedBaseError(PermissionError):
    """Raised when a receipt may not serve as the base of a forgery of a stream."""


def directory_fonts(directory: Path) -> FontSource:
    """Fonts read from the directory where ``docfraud synthesis download-fonts`` put them."""

    @lru_cache(maxsize=256)
    def font(family: str, bold: bool, size: int) -> ImageFont.FreeTypeFont:
        return ImageFont.truetype(font_path(directory, family, bold=bold), size)

    return font


@dataclass(frozen=True, slots=True)
class _Layer:
    """Content to mix into the page over a box, with a weight per pixel."""

    box: BBox
    content: Grey
    alpha: Grey


@dataclass
class _Plan:
    """Everything one region will write, decided before anything is written."""

    layers: list[_Layer]
    after: str | None
    variants: dict[str, str] = field(default_factory=dict)
    sources: tuple[BBox, ...] = ()
    source_document: str | None = None


class _Canvas:
    """The page being edited, the area written, and what later regions must keep clear of."""

    def __init__(self, base: Receipt) -> None:
        self.base = base
        self.page = base.page.copy()
        self.written = np.zeros(base.page.shape, dtype=bool)
        self.clear = np.zeros(base.page.shape, dtype=bool)

    @property
    def shape(self) -> tuple[int, int]:
        height, width = self.page.shape
        return height, width

    def free(self, box: BBox) -> bool:
        """Whether a box lies inside the page and touches no earlier edit or source."""
        return _inside(box, self.shape) and not self.clear[box.y : box.y2, box.x : box.x2].any()

    def apply(self, layers: Sequence[_Layer]) -> Mask:
        """Write the layers in order; return the pixels they wrote."""
        region = np.zeros(self.shape, dtype=bool)
        for layer in layers:
            box = layer.box
            view = self.page[box.y : box.y2, box.x : box.x2]
            written = layer.alpha > 0
            mixed = layer.alpha * layer.content + (1 - layer.alpha) * view
            view[written] = np.clip(np.rint(mixed[written]), 0, 255).astype(np.uint8)
            region[box.y : box.y2, box.x : box.x2] |= written
        self.written |= region
        return region

    def keep_clear(self, box: BBox) -> None:
        grown = _expand(box, KEEP_CLEAR, self.shape)
        self.clear[grown.y : grown.y2, grown.x : grown.x2] = True


class Engine:
    """Makes synthetic forgeries for one cross-validation round.

    Args:
        statistics: The round's real forgeries (:func:`round_statistics`).
        documents: The dataset's documents, to know which receipts may be bases.
        folds: The cross-validation folds.
        receipts: Loads a receipt as the engine sees it, by identifier.
        fonts: The fonts imitations are typed with.
    """

    def __init__(
        self,
        statistics: RoundStatistics,
        documents: Sequence[Document],
        folds: FoldAssignment,
        receipts: ReceiptSource,
        fonts: FontSource,
    ) -> None:
        """Fix the round and the receipts each stream may use."""
        self.statistics = statistics
        self.held_out = statistics.held_out
        self.receipts = receipts
        self.fonts = fonts
        authentic = {
            d.doc_id
            for d in documents
            if d.split is Split.TRAIN and d.label is Label.AUTHENTIC and d.doc_id in folds.folds
        }
        held = {doc_id for doc_id in authentic if folds.folds[doc_id] == self.held_out}
        development = held & DESIGN_FOLD_IDS if self.held_out == DESIGN_FOLD else set()
        self.pools: dict[Stream, tuple[str, ...]] = {
            Stream.TRAINING: tuple(sorted(authentic - held)),
            Stream.DEVELOPMENT: tuple(sorted(development)),
        }

    def forge(self, base: str, index: int, stream: Stream = Stream.TRAINING) -> Forgery:
        """Make forgery ``index`` of a base receipt.

        Its record lists no region if no template could be placed; the page is then unchanged.

        Raises:
            RefusedBaseError: If the stream may not use the receipt as a base.
        """
        if base not in self.pools[stream]:
            msg = f"{base} may not be a base of stream {stream.name} in round {self.held_out}"
            raise RefusedBaseError(msg)
        rng = np.random.default_rng(
            [ENGINE_SEED, self.held_out, receipt_key(base), int(stream), index]
        )
        canvas = _Canvas(self.receipts(base))
        regions: list[SyntheticRegion] = []
        failures: Counter[str] = Counter()
        templates = self.statistics.templates
        for _ in range(int(rng.choice(self.statistics.regions_per_receipt))):
            for _ in range(TRIES):
                template = templates[int(rng.integers(len(templates)))]
                technique = template.techniques[int(rng.integers(len(template.techniques)))]
                outcome = self._region(canvas, template, technique, stream, rng)
                if isinstance(outcome, str):
                    failures[outcome] += 1
                    continue
                regions.append(outcome)
                break
            else:
                failures["region given up"] += 1
        changed = canvas.page != canvas.base.page
        if (changed & ~canvas.written).any():  # pragma: no cover - excluded by construction
            msg = "a pixel outside the written area changed"
            raise RuntimeError(msg)
        record = ForgeryRecord(
            engine_version=ENGINE_VERSION,
            base=base,
            held_out=self.held_out,
            stream=stream,
            index=index,
            regions=tuple(regions),
            failures=dict(failures),
        )
        return Forgery(canvas.page, changed, canvas.written.copy(), record)

    def _region(
        self,
        canvas: _Canvas,
        template: Template,
        technique: Technique,
        stream: Stream,
        rng: np.random.Generator,
    ) -> SyntheticRegion | str:
        """Place one region from a template, or say why it could not be placed.

        The template's entity is kept: up to ``WORDS_PER_TEMPLATE`` of its words are tried, so
        that entities whose words are harder to edit are not drawn less often than in the
        round's real regions.
        """
        candidates = [
            w
            for w in canvas.base.words
            if w.entity is template.entity and w.characters and w.editable
        ]
        if not candidates:
            return "no word of the entity"
        reason = ""
        for pick in rng.permutation(len(candidates))[:WORDS_PER_TEMPLATE]:
            word = candidates[int(pick)]
            outcome = self._place(canvas, word, template, rng, technique=technique, stream=stream)
            if not isinstance(outcome, str):
                return outcome
            reason = outcome
        return reason

    def _place(  # noqa: PLR0913 - one word, its template and how to edit it
        self,
        canvas: _Canvas,
        word: Word,
        template: Template,
        rng: np.random.Generator,
        *,
        technique: Technique,
        stream: Stream,
    ) -> SyntheticRegion | str:
        """Place one region on one word, or say why it could not be placed."""
        length = min(template.characters, len(word.characters))
        starts = [
            s
            for s in range(len(word.characters) - length + 1)
            if any(i in word.editable for i in range(s, s + length))
        ]
        start = starts[int(rng.integers(len(starts)))]
        positions = tuple(i for i in range(start, start + length) if i in word.editable)
        cover = self._cover(canvas, word, positions, rng)
        if cover is None:
            return "overlaps an earlier edit"
        plan: _Plan | str
        if technique in {Technique.CPI, Technique.CPO}:
            plan = self._copy(canvas, word, positions, rng, technique=technique, stream=stream)
        elif technique is Technique.IMI:
            plan = self._imitate(canvas, word, positions, rng)
        elif technique is Technique.CUT:
            kept = "".join(c.text for i, c in enumerate(word.characters) if i not in positions)
            plan = _Plan([], kept)
        else:
            plan = self._strokes(word, positions, rng)
        if isinstance(plan, str):
            return plan
        layers = plan.layers if technique is Technique.PIX else [*cover[0], *plan.layers]
        variants = plan.variants if technique is Technique.PIX else cover[1] | plan.variants
        region = canvas.apply(layers)
        if not region.any():  # a free-hand stroke too short to cover any pixel; nothing changed
            return "wrote no pixel"
        box = _mask_box(region)
        canvas.keep_clear(box)
        if plan.source_document is None:
            for source in plan.sources:
                canvas.keep_clear(source)
        return SyntheticRegion(
            technique=technique,
            entity=template.entity,
            box=box,
            sources=plan.sources,
            source_document=plan.source_document,
            span=length,
            before="".join(c.text for c in word.characters),
            after=plan.after,
            variants=variants,
        )

    def _cover(
        self, canvas: _Canvas, word: Word, positions: Sequence[int], rng: np.random.Generator
    ) -> tuple[list[_Layer], dict[str, str]] | None:
        """The layers that cover the old characters, or ``None`` if one would touch an edit.

        Each character is covered on its own, so that separators between them stay.
        """
        kind = _choice(rng, ("texture", "white", "flat"))
        edge = (
            int(_choice(rng, ("1", "2", "3"))) if _choice(rng, ("sharp", "soft")) == "soft" else 0
        )
        layers: list[_Layer] = []
        fallback = False
        for i in positions:
            box = _expand(word.characters[i].box, COVER_MARGIN + edge, canvas.shape)
            if not canvas.free(box):
                return None
            content: Grey | None = None
            if kind == "texture":
                content = _blank_paper(canvas.base.page, box, word.threshold)
                fallback = fallback or content is None
            if content is None:
                level = 255.0 if kind == "white" else word.paper
                content = np.full((box.height, box.width), level)
            layers.append(_Layer(box, content, _ramp(box.height, box.width, edge)))
        if fallback:
            kind = "texture (flat where no blank paper was nearby)"
        return layers, {"cover": kind, "cover edge": f"{edge} px" if edge else "sharp"}

    def _copy(  # noqa: PLR0913 - the target, the generator and how the copy is made
        self,
        canvas: _Canvas,
        word: Word,
        positions: Sequence[int],
        rng: np.random.Generator,
        *,
        technique: Technique,
        stream: Stream,
    ) -> _Plan | str:
        """Copy characters from elsewhere on the receipt (CPI) or from another (CPO)."""
        donor = canvas.base
        if technique is Technique.CPO:
            others = [doc_id for doc_id in self.pools[stream] if doc_id != canvas.base.doc_id]
            if not others:
                return "no other receipt to copy from"
            donor = self.receipts(others[int(rng.integers(len(others)))])
        row = _choice(rng, ("same row", "other row")) if technique is Technique.CPI else ""
        chosen: list[tuple[Character, Word]] = []
        for i in positions:
            pool = _sources(donor, canvas, word, word.characters[i], technique=technique, row=row)
            if not pool:
                return "no character to copy"
            chosen.append(pool[int(rng.integers(len(pool)))])
        mode = _choice(rng, ("rectangle", "ink only"))
        shift = _shift(rng)
        slight = _choice(rng, ("none", "slight")) == "slight"
        scale = 1 + float(rng.uniform(-SCALE, SCALE)) if slight else 1.0
        angle = float(rng.uniform(-ROTATION, ROTATION)) if slight else 0.0
        edge = int(_choice(rng, ("1", "2"))) if _choice(rng, ("sharp", "soft")) == "soft" else 0
        layers: list[_Layer] = []
        new = [c.text for c in word.characters]
        for i, (character, source_word) in zip(positions, chosen, strict=True):
            glyph = _expand(character.box, PASTE_MARGIN, donor.page.shape)
            patch = donor.page[glyph.y : glyph.y2, glyph.x : glyph.x2].astype(np.float64)
            if slight:
                patch = _affine(patch, scale, angle, _border(patch))
            target = word.characters[i]
            dx = target.cell.x + target.cell.width / 2 - (character.box.x + character.box.width / 2)
            box = _placed(
                glyph.x + round(dx) + shift[0],
                glyph.y + word.baseline - source_word.baseline + shift[1],
                patch.shape,
                canvas.shape,
            )
            if box is None or not canvas.free(box):
                return "paste outside the page or over an earlier edit"
            if mode == "rectangle":
                alpha = _ramp(box.height, box.width, edge)
            else:  # the ink alone: the pixels of the copy that are ink on its own receipt
                alpha = (patch < source_word.threshold).astype(np.float64)
            layers.append(_Layer(box, patch, alpha))
            new[i] = character.text
        variants = {
            "paste": mode,
            "position": f"shifted {shift[0]:+d}, {shift[1]:+d} px" if any(shift) else "aligned",
            "scale and rotation": f"{scale:.3f}, {angle:+.2f} degrees" if slight else "none",
            "paste edge": f"{edge} px" if edge and mode == "rectangle" else "sharp",
        }
        if technique is Technique.CPI:
            variants["source row"] = row
        return _Plan(
            layers,
            "".join(new),
            variants,
            tuple(c.box for c, _ in chosen),
            donor.doc_id if technique is Technique.CPO else None,
        )

    def _imitate(
        self, canvas: _Canvas, word: Word, positions: Sequence[int], rng: np.random.Generator
    ) -> _Plan | str:
        """Type new characters with a text tool (IMI)."""
        family = _choice(rng, tuple(FONTS))
        bold = _choice(rng, ("regular", "bold")) == "bold"
        antialias = _choice(rng, ("antialiased", "not antialiased")) == "antialiased"
        tone_name = _choice(rng, ("same", "darker", "lighter"))
        shift = _shift(rng)
        step = TONE_STEP * (word.paper - word.ink)
        tone = {"same": word.ink, "darker": max(0.0, word.ink - step), "lighter": word.ink + step}
        layers: list[_Layer] = []
        new = [c.text for c in word.characters]
        for i in positions:
            target = word.characters[i]
            text = _replacement(target, rng)
            font = self._sized(family, bold, target.kind, _text_height(word, target.kind))
            coverage, baseline = _render(text, font, antialias=antialias)
            x = round(target.cell.x + target.cell.width / 2 - coverage.shape[1] / 2) + shift[0]
            box = _placed(x, word.baseline - baseline + shift[1], coverage.shape, canvas.shape)
            if box is None or not canvas.free(box):
                return "typed text outside the page or over an earlier edit"
            layers.append(_Layer(box, np.full(coverage.shape, tone[tone_name]), coverage))
            new[i] = text
        variants = {
            "font": f"{family} {'bold' if bold else 'regular'}",
            "rendering": "antialiased" if antialias else "not antialiased",
            "ink": tone_name,
            "position": f"shifted {shift[0]:+d}, {shift[1]:+d} px" if any(shift) else "aligned",
        }
        return _Plan(layers, "".join(new), variants)

    def _sized(self, family: str, bold: bool, kind: Kind, height: float) -> ImageFont.FreeTypeFont:
        """The font at the size at which its reference glyph has the given height in pixels."""
        reference = self.fonts(family, bold, 100)
        _, top, _, bottom = reference.getbbox(_REFERENCE[kind], anchor="ls")
        return self.fonts(family, bold, max(4, round(100 * height / max(1.0, bottom - top))))

    def _strokes(self, word: Word, positions: Sequence[int], rng: np.random.Generator) -> _Plan:
        """Draw or erase short pen strokes inside the characters (PIX)."""
        mode = _choice(rng, ("draw", "erase"))
        pen = int(_choice(rng, ("1", "2", "3")))
        soft = _choice(rng, ("hard", "soft")) == "soft"
        strokes = int(_choice(rng, ("1", "2")))
        tone = word.ink if mode == "draw" else word.paper
        layers = []
        for i in positions:
            box = word.characters[i].box
            alpha = _pen(box, pen, strokes, rng, soft=soft)
            layers.append(_Layer(box, np.full(alpha.shape, tone), alpha))
        variants = {
            "stroke": mode,
            "pen": f"{pen} px, {'soft' if soft else 'hard'}",
            "strokes": str(strokes),
        }
        return _Plan(layers, None, variants)


def _choice(rng: np.random.Generator, options: Sequence[str]) -> str:
    """One option, each equally likely."""
    return options[int(rng.integers(len(options)))]


def _shift(rng: np.random.Generator) -> tuple[int, int]:
    """No shift, or a shift of 1 or 2 px in one of four directions, each half of the cases."""
    if _choice(rng, ("aligned", "shifted")) == "aligned":
        return (0, 0)
    size = int(rng.integers(1, 3))
    return ((size, 0), (-size, 0), (0, size), (0, -size))[int(rng.integers(4))]


def _placed(x: int, y: int, size: tuple[int, ...], shape: tuple[int, int]) -> BBox | None:
    """The box of ``size`` (height, width) at ``x``, ``y``, or ``None`` if it leaves the page."""
    height, width = size[0], size[1]
    if x < 0 or y < 0 or x + width > shape[1] or y + height > shape[0]:
        return None
    return BBox(x=x, y=y, width=width, height=height)


def _inside(box: BBox, shape: tuple[int, int]) -> bool:
    height, width = shape
    return box.x >= 0 and box.y >= 0 and box.x2 <= width and box.y2 <= height


def _expand(box: BBox, margin: int, shape: tuple[int, ...]) -> BBox:
    """A box grown by ``margin`` on every side and clipped to the page."""
    height, width = shape[0], shape[1]
    x, y = max(0, box.x - margin), max(0, box.y - margin)
    return BBox(
        x=x, y=y, width=min(width, box.x2 + margin) - x, height=min(height, box.y2 + margin) - y
    )


def _union(boxes: Sequence[BBox]) -> BBox:
    x, y = min(b.x for b in boxes), min(b.y for b in boxes)
    return BBox(x=x, y=y, width=max(b.x2 for b in boxes) - x, height=max(b.y2 for b in boxes) - y)


def _mask_box(mask: Mask) -> BBox:
    rows, columns = np.flatnonzero(mask.any(axis=1)), np.flatnonzero(mask.any(axis=0))
    return BBox(
        x=int(columns[0]),
        y=int(rows[0]),
        width=int(columns[-1] - columns[0]) + 1,
        height=int(rows[-1] - rows[0]) + 1,
    )


def _ramp(height: int, width: int, edge: int) -> Grey:
    """Weights that rise from the border of a box to 1 over ``edge`` pixels (1 everywhere if 0)."""
    if edge == 0:
        return np.ones((height, width))
    rows = np.minimum(np.arange(height), np.arange(height)[::-1])
    columns = np.minimum(np.arange(width), np.arange(width)[::-1])
    return np.minimum(1.0, (np.minimum.outer(rows, columns) + 1) / (edge + 1))


def _border(patch: Grey) -> float:
    """The median of a patch's outermost pixels: its paper."""
    return float(np.median(np.concatenate([patch[0], patch[-1], patch[:, 0], patch[:, -1]])))


def _blank_paper(page: npt.NDArray[np.uint8], box: BBox, threshold: float) -> Grey | None:
    """A patch of the base page of the size of ``box``, near it, without ink."""
    height, width = page.shape
    for step in range(1, SEARCH_STEPS + 1):
        for dx, dy in ((-step, 0), (step, 0), (0, -step), (0, step)):
            candidate = _placed(
                box.x + dx * (box.width + 2),
                box.y + dy * (box.height + 2),
                (box.height, box.width),
                (height, width),
            )
            if candidate is None:
                continue
            patch = page[candidate.y : candidate.y2, candidate.x : candidate.x2]
            if (patch >= threshold).all():
                return patch.astype(np.float64)
    return None


def _sources(  # noqa: PLR0913 - where to copy from, onto what, and how
    donor: Receipt,
    canvas: _Canvas,
    word: Word,
    old: Character,
    *,
    technique: Technique,
    row: str,
) -> list[tuple[Character, Word]]:
    """Characters that may be copied over ``old``: of its class, different, cleanly separated.

    Only characters located by the gaps of ink are copied, so that a copy holds one whole glyph.
    On the base receipt (CPI) the target word and earlier edits are left out, and the row option
    is honoured where the receipt allows it.
    """
    found: list[tuple[Character, Word, bool]] = []
    for other in donor.words:
        if other.method != "gaps" or (technique is Technique.CPI and other is word):
            continue
        same_row = abs(other.baseline - word.baseline) < word.characters[0].cell.height / 2
        for character in other.characters:
            if character.kind is not old.kind or character.text == old.text:
                continue
            if technique is Technique.CPI and not canvas.free(
                _expand(character.box, PASTE_MARGIN, canvas.shape)
            ):
                continue
            found.append((character, other, same_row))
    if technique is Technique.CPI:
        preferred = [(c, w) for c, w, same in found if same == (row == "same row")]
        if preferred:
            return preferred
    return [(c, w) for c, w, _ in found]


def _replacement(old: Character, rng: np.random.Generator) -> str:
    """Another character of the same class."""
    options = [c for c in _ALPHABETS[old.kind] if c != old.text]
    return options[int(rng.integers(len(options)))]


def _text_height(word: Word, kind: Kind) -> float:
    """The height of the word's printed characters of a class, or of all of them."""
    same = [c.box.height for c in word.characters if c.kind is kind]
    return float(np.median(same or [c.box.height for c in word.characters]))


def _render(text: str, font: ImageFont.FreeTypeFont, *, antialias: bool) -> tuple[Grey, int]:
    """Coverage of one typed character, between 0 and 1, and the row of its baseline in it.

    The coverage is cropped to the character's ink; the baseline row may lie below it.
    """
    left, top, right, bottom = (int(v) for v in font.getbbox(text, anchor="ls"))
    image = Image.new("L", (right - left + 2, bottom - top + 2), 0)
    draw = ImageDraw.Draw(image)
    draw.fontmode = "L" if antialias else "1"
    draw.text((1 - left, 1 - top), text, fill=255, font=font, anchor="ls")
    coverage = np.asarray(image, dtype=np.float64) / 255
    rows, columns = np.flatnonzero(coverage.any(axis=1)), np.flatnonzero(coverage.any(axis=0))
    cropped = coverage[rows[0] : rows[-1] + 1, columns[0] : columns[-1] + 1]
    return cropped, (1 - top) - int(rows[0])


def _pen(box: BBox, pen: int, strokes: int, rng: np.random.Generator, *, soft: bool) -> Grey:
    """Weights of short strokes of a pen, kept inside a character's box."""
    image = Image.new("L", (box.width * SUPERSAMPLING, box.height * SUPERSAMPLING), 0)
    draw = ImageDraw.Draw(image)
    for _ in range(strokes):
        ends = rng.uniform([0, 0], [box.width, box.height], size=(2, 2)) * SUPERSAMPLING
        draw.line([(float(x), float(y)) for x, y in ends], fill=255, width=pen * SUPERSAMPLING)
    fine = np.asarray(image, dtype=np.float64) / 255
    coverage = fine.reshape(box.height, SUPERSAMPLING, box.width, SUPERSAMPLING).mean(axis=(1, 3))
    return coverage if soft else (coverage >= HARD_PEN).astype(np.float64)


def _affine(patch: Grey, scale: float, angle: float, paper: float) -> Grey:
    """The patch scaled and rotated about its centre, keeping its size; corners get paper."""
    height, width = patch.shape
    theta = math.radians(angle)
    a, b = math.cos(theta) / scale, math.sin(theta) / scale
    cx, cy = width / 2, height / 2
    coefficients = (a, b, cx - a * cx - b * cy, -b, a, cy + b * cx - a * cy)
    image = Image.fromarray(patch.astype(np.float32), mode="F")
    moved = image.transform(
        (width, height),
        Image.Transform.AFFINE,
        coefficients,
        resample=Image.Resampling.BICUBIC,
        fillcolor=paper,
    )
    return np.clip(np.asarray(moved, dtype=np.float64), 0, 255)
