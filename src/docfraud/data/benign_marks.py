"""Where the benign marks of the design fold are (ADR-0014, step 4.2, decision 4).

Find-it-again flags the receipts that carry handwriting, highlighting or stamps ("handwritten
annotation") or digitally added text ("digital annotation"), but not where these marks are. A
pixel detector must not take them for edits, and its false alarms during development can only be
read if one knows whether they fall on a mark. On the flagged receipts of the design fold the
assistant therefore draws a box around every mark, from views that reach it at native resolution
(:func:`export_views`), and the project owner checks every receipt on the page served by
:mod:`docfraud.data.benign_marks_server`. A receipt the owner does not accept is marked again and
checked again in the next round, for at most ``MAX_ROUNDS`` rounds
(``docs/benign-marks-guide.md``).

Both see the normalised grayscale image that the models see. Every receipt passes
:func:`~docfraud.evaluation.design_fold.require_design_fold` before anything of it is shown.
"""

import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Self

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw, ImageFont
from pydantic import BaseModel, ConfigDict, Field, model_validator

from docfraud.data.models import BBox, Document
from docfraud.data.observation import (
    ASSISTANT,
    ASSISTANT_TILE,
    Display,
    Presence,
    observed_pixels,
    png_bytes,
)
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS

MARKS_VERSION = 1
"""Version of the mark and check formats; a file written for another version is refused."""

ORDER_SEED = 0
"""Seed of the order in which the check page shows the receipts."""

MAX_ROUNDS = 3
"""Rounds of marking and checking; a receipt not accepted by then is reported as disputed."""

TILE_GRID = 100
"""Spacing in pixels of the coordinate grid on the assistant's native-resolution tiles."""

OVERVIEW_GRID_STEPS = (100, 250, 500, 1000)
"""Grid spacings, in page pixels, from which a reduced overview takes the finest that stays
legible (``MIN_GRID_SPACING``)."""

MIN_GRID_SPACING = 40
"""Smallest distance between two grid lines in a view, in view pixels."""

_GRID_COLOUR = (0, 150, 255)
_MARK_COLOUR = (220, 0, 0)
_OUTLINE = 2
_FONT_SIZE = 13


class MarkKind(StrEnum):
    """What a benign mark is."""

    HANDWRITING = "handwriting"  # handwritten words, numbers or initials
    SIGNATURE = "signature"
    STAMP = "stamp"  # an inked stamp impression
    HIGHLIGHT = "highlight"  # a highlighter stroke
    PEN_MARK = "pen_mark"  # tick, circle, underline, strike-through, arrow or line
    DIGITAL_NOTE = "digital_note"  # text or shapes added digitally
    OTHER = "other"  # described in the mark's note


HAND_KINDS = frozenset(MarkKind) - {MarkKind.DIGITAL_NOTE, MarkKind.OTHER}
"""Kinds that the dataset's "handwritten annotation" flag covers: handwriting, highlighting and
stamps."""


class Verdict(StrEnum):
    """The owner's check of one receipt's boxes."""

    CORRECT = "correct"  # every mark has a box and every box is on a mark
    MISSING = "missing"  # a mark has no box
    WRONG_BOX = "wrong_box"  # a box is on no mark or clearly misplaced
    MISSING_AND_WRONG = "missing_and_wrong"


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Mark(_Frozen):
    """One benign mark: its box in page pixels, its kind and whether it touches print."""

    box: BBox
    kind: MarkKind
    touches_print: Presence
    note: str = ""

    @model_validator(mode="after")
    def _other_is_described(self) -> Self:
        if self.kind is MarkKind.OTHER and not self.note.strip():
            msg = "a mark of kind 'other' needs a note"
            raise ValueError(msg)
        return self


class ReceiptMarks(_Frozen):
    """Every mark on one receipt; none if no mark was found, which the note then says."""

    marks: tuple[Mark, ...] = ()
    note: str = ""


class MarkLog(_Frozen):
    """The marks of one round, for every receipt of that round."""

    version: int = MARKS_VERSION
    annotator: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    round: int = Field(ge=1, le=MAX_ROUNDS)
    commit: str | None  # of the guide and code the marks follow
    marked_at: datetime
    receipts: dict[str, ReceiptMarks]

    @model_validator(mode="after")
    def _current_version(self) -> Self:
        _check_version(self.version)
        return self


class Point(_Frozen):
    """A place on the page in page pixels, where the owner points to a problem."""

    x: int = Field(ge=0)
    y: int = Field(ge=0)


class ReceiptCheck(_Frozen):
    """The owner's check of one receipt."""

    verdict: Verdict
    points: tuple[Point, ...] = ()
    note: str = ""
    checked_at: datetime
    commit: str | None
    display: Display | None = None
    language: str | None = None  # of the page the check was made on; None: English

    @model_validator(mode="after")
    def _points_show_problems(self) -> Self:
        if self.verdict is Verdict.CORRECT and self.points:
            msg = "points mark problems; a correct receipt has none"
            raise ValueError(msg)
        return self


class CheckLog(_Frozen):
    """The owner's checks of one round, possibly unfinished."""

    version: int = MARKS_VERSION
    checker: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    round: int = Field(ge=1, le=MAX_ROUNDS)
    receipts: dict[str, ReceiptCheck] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _current_version(self) -> Self:
        _check_version(self.version)
        if self.checker == ASSISTANT:
            msg = "the assistant does not check its own marks"
            raise ValueError(msg)
        return self


def _check_version(version: int) -> None:
    if version != MARKS_VERSION:
        msg = f"file of format version {version}, expected {MARKS_VERSION}"
        raise ValueError(msg)


def marks_plan(documents: Iterable[Document]) -> tuple[str, ...]:
    """The flagged receipts of the design fold, in the seeded order of the check page.

    A receipt is flagged if the dataset marks it as carrying a handwritten or a digital
    annotation, whatever its label.
    """
    flagged = sorted(
        d.doc_id
        for d in documents
        if d.doc_id in DESIGN_FOLD_IDS and (d.handwritten_annotation or d.digital_annotation)
    )
    order = np.random.default_rng(ORDER_SEED).permutation(len(flagged))
    return tuple(flagged[i] for i in order)


def round_receipts(plan: Sequence[str], checks: Sequence[CheckLog]) -> tuple[str, ...]:
    """The receipts of the round after ``checks``, in plan order.

    Round 1 takes every planned receipt; each later round takes the receipts whose check in the
    round before was not "correct".

    Raises:
        ValueError: If the checks are not rounds 1, 2, ... in order, or a check does not cover
            exactly the receipts of its round.
    """
    receipts = tuple(plan)
    for number, check in enumerate(checks, start=1):
        if check.round != number:
            msg = f"check of round {check.round} found where round {number} was expected"
            raise ValueError(msg)
        if set(check.receipts) != set(receipts):
            msg = f"the check of round {number} must cover its {len(receipts)} receipts"
            raise ValueError(msg)
        receipts = tuple(d for d in receipts if check.receipts[d].verdict is not Verdict.CORRECT)
    return receipts


def image_sizes(documents: Iterable[Document], receipts: Iterable[str]) -> dict[str, BBox]:
    """The page of every receipt as a box at the origin, read from the image header only."""
    by_id = {d.doc_id: d for d in documents}
    sizes = {}
    for doc_id in receipts:
        with Image.open(by_id[doc_id].image_path) as image:
            sizes[doc_id] = BBox(x=0, y=0, width=image.width, height=image.height)
    return sizes


def mark_problems(
    log: MarkLog, round_number: int, expected: Sequence[str], pages: Mapping[str, BBox]
) -> list[str]:
    """What keeps a mark log from serving as round ``round_number``.

    Returns:
        One line per problem (another round, other receipts than the round's, boxes off the
        page); empty if there is none.
    """
    problems = (
        [f"marks of round {log.round}, not {round_number}"] if log.round != round_number else []
    )
    problems += _coverage_problems(set(log.receipts), set(expected), complete=True)
    for doc_id, receipt in sorted(log.receipts.items()):
        page = pages.get(doc_id)
        for number, mark in enumerate(receipt.marks, start=1):
            if page and not mark.box.fits_within(page.width, page.height):
                problems.append(f"{doc_id}: box {number} lies outside the page")
    return problems


def check_problems(log: CheckLog, expected: Sequence[str], pages: Mapping[str, BBox]) -> list[str]:
    """What is wrong with a possibly unfinished check log: strange receipts, points off the page.

    Returns:
        One line per problem; empty if there is none. Receipts not yet checked are no problem.
    """
    problems = _coverage_problems(set(log.receipts), set(expected), complete=False)
    for doc_id, check in sorted(log.receipts.items()):
        page = pages.get(doc_id)
        if page and any(p.x >= page.width or p.y >= page.height for p in check.points):
            problems.append(f"{doc_id}: a point lies outside the page")
    return problems


def _coverage_problems(found: set[str], expected: set[str], *, complete: bool) -> list[str]:
    problems = [f"{doc_id}: not a receipt of this round" for doc_id in sorted(found - expected)]
    if complete:
        problems += [f"{doc_id}: missing" for doc_id in sorted(expected - found)]
    return problems


@dataclass(frozen=True, slots=True)
class FinalMarks:
    """The outcome of all rounds: the latest marks of every receipt and when they were accepted.

    Attributes:
        marks: The marks of the last round in which each receipt was marked.
        accepted_in: The round in which the owner accepted each receipt, or ``None`` if the
            receipt was still not accepted after the last round (disputed).
        rounds: Number of rounds held.
    """

    marks: Mapping[str, ReceiptMarks]
    accepted_in: Mapping[str, int | None]
    rounds: int


def final_marks(
    plan: Sequence[str], mark_logs: Sequence[MarkLog], check_logs: Sequence[CheckLog]
) -> FinalMarks:
    """Combine finished rounds into the marks that later steps use.

    Raises:
        ValueError: If a round is unchecked, a log does not cover its round's receipts, or the
            procedure is unfinished: receipts remain unaccepted and another round is allowed.
    """
    if not mark_logs or len(mark_logs) != len(check_logs):
        msg = "every round needs its marks and a finished check"
        raise ValueError(msg)
    marks: dict[str, ReceiptMarks] = {}
    accepted_in: dict[str, int | None] = dict.fromkeys(plan)
    for number, log in enumerate(mark_logs, start=1):
        expected = round_receipts(plan, check_logs[: number - 1])
        if log.round != number or set(log.receipts) != set(expected):
            msg = f"the marks of round {number} must cover its {len(expected)} receipts"
            raise ValueError(msg)
        marks |= log.receipts
    remaining = round_receipts(plan, check_logs)  # also checks every check log
    for number, check in enumerate(check_logs, start=1):
        for doc_id, receipt_check in check.receipts.items():
            if receipt_check.verdict is Verdict.CORRECT:
                accepted_in[doc_id] = number
    if remaining and len(mark_logs) < MAX_ROUNDS:
        msg = f"{len(remaining)} receipts are not accepted yet; round {len(mark_logs) + 1} is due"
        raise ValueError(msg)
    return FinalMarks(marks, accepted_in, len(mark_logs))


def round_path(directory: Path, name: str, round_number: int) -> Path:
    """Where the marks or checks of one person and round are kept."""
    return directory / f"{name}-{round_number}.json"


def load_mark_log(path: Path) -> MarkLog:
    """Read the marks of one round."""
    return MarkLog.model_validate_json(path.read_text(encoding="utf-8"))


def load_check_log(path: Path, checker: str, round_number: int) -> CheckLog:
    """Read the checks of one round, or start an empty log if none exists yet.

    Raises:
        ValueError: If the log belongs to another checker or round.
    """
    if not path.exists():
        return CheckLog(checker=checker, round=round_number)
    log = CheckLog.model_validate_json(path.read_text(encoding="utf-8"))
    if (log.checker, log.round) != (checker, round_number):
        found = f"round {log.round} of {log.checker!r}"
        msg = f"{path} is {found}, not round {round_number} of {checker!r}"
        raise ValueError(msg)
    return log


def load_rounds(
    directory: Path, annotator: str, checker: str
) -> tuple[list[MarkLog], list[CheckLog]]:
    """The mark and check logs of rounds 1, 2, ... found in a directory, in round order."""
    mark_logs: list[MarkLog] = []
    check_logs: list[CheckLog] = []
    for number in range(1, MAX_ROUNDS + 1):
        if round_path(directory, annotator, number).exists():
            mark_logs.append(load_mark_log(round_path(directory, annotator, number)))
        if round_path(directory, checker, number).exists():
            path = round_path(directory, checker, number)
            check_logs.append(load_check_log(path, checker, number))
    return mark_logs, check_logs


def export_views(
    documents: Iterable[Document],
    receipts: Sequence[str],
    out_dir: Path,
    marks: Mapping[str, ReceiptMarks] | None = None,
) -> list[Path]:
    """Write what the assistant looks at to mark receipts, or to check its marks.

    For every receipt, numbered in the order given:

    - ``overview``: the whole page reduced so that its longer side is at most ``ASSISTANT_TILE``
      pixels, without anything drawn on it. It serves to find marks; it is the only reduced
      view, and nothing is measured on it.
    - ``overview-grid``: the same with a coordinate grid labelled in page pixels.
    - ``tile-<row>-<column>``: the page at native resolution in tiles of at most
      ``ASSISTANT_TILE`` pixels a side, with a grid every ``TILE_GRID`` page pixels. Boxes are
      read from these.

    With ``marks``, every box is drawn and numbered on the grid views, the file names end in
    ``-marked``, and the plain overview is left out.

    Returns:
        The written files.

    Raises:
        NotInDesignFoldError: If a receipt lies outside the design fold.
    """
    by_id = {d.doc_id: d for d in documents}
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for number, doc_id in enumerate(receipts, start=1):
        page = observed_pixels(by_id[doc_id])
        height, width = page.shape
        drawn = marks[doc_id].marks if marks is not None else None
        suffix = "-marked" if drawn is not None else ""
        scale = min(1.0, ASSISTANT_TILE / max(height, width))
        views: list[tuple[str, npt.NDArray[np.uint8]]] = []
        if drawn is None:
            views.append(("overview", _reduce(page, scale)))
        step = next(
            (s for s in OVERVIEW_GRID_STEPS if s * scale >= MIN_GRID_SPACING),
            OVERVIEW_GRID_STEPS[-1],
        )
        views.append(("overview-grid", _draw(_reduce(page, scale), (0, 0), scale, step, drawn)))
        for top in range(0, height, ASSISTANT_TILE):
            for left in range(0, width, ASSISTANT_TILE):
                tile = page[top : top + ASSISTANT_TILE, left : left + ASSISTANT_TILE]
                name = f"tile-{top // ASSISTANT_TILE}-{left // ASSISTANT_TILE}"
                views.append((name, _draw(tile, (left, top), 1.0, TILE_GRID, drawn)))
        for name, pixels in views:
            path = out_dir / f"{number:02d}-{doc_id}-{name}{suffix}.png"
            path.write_bytes(png_bytes(pixels))
            written.append(path)
    return written


def _reduce(page: npt.NDArray[np.uint8], scale: float) -> npt.NDArray[np.uint8]:
    if scale == 1:
        return page
    height, width = page.shape
    size = (max(1, round(width * scale)), max(1, round(height * scale)))
    return np.asarray(Image.fromarray(page).resize(size, Image.Resampling.LANCZOS))


def _draw(
    pixels: npt.NDArray[np.uint8],
    origin: tuple[int, int],
    scale: float,
    step: int,
    marks: Sequence[Mark] | None,
) -> npt.NDArray[np.uint8]:
    """A grid every ``step`` page pixels and the numbered boxes of ``marks`` on a view.

    The view shows the page from ``origin`` (x, y) on, ``scale`` view pixels per page pixel.
    """
    x0, y0 = origin
    image = Image.fromarray(pixels).convert("RGB")
    draw = ImageDraw.Draw(image)
    font = ImageFont.load_default(size=_FONT_SIZE)
    view_height, view_width = pixels.shape
    page_right, page_bottom = x0 + view_width / scale, y0 + view_height / scale
    for gx in range(math.ceil(x0 / step) * step, math.ceil(page_right), step):
        vx = round((gx - x0) * scale)
        draw.line([(vx, 0), (vx, view_height - 1)], fill=_GRID_COLOUR, width=1)
        _label(draw, (vx + 3, 2), str(gx), _GRID_COLOUR, font)
    for gy in range(math.ceil(y0 / step) * step, math.ceil(page_bottom), step):
        vy = round((gy - y0) * scale)
        draw.line([(0, vy), (view_width - 1, vy)], fill=_GRID_COLOUR, width=1)
        _label(draw, (3, vy + 2), str(gy), _GRID_COLOUR, font)
    for number, mark in enumerate(marks or (), start=1):
        box = mark.box
        corners = (
            round((box.x - x0) * scale) - _OUTLINE,
            round((box.y - y0) * scale) - _OUTLINE,
            round((box.x2 - x0) * scale) - 1 + _OUTLINE,
            round((box.y2 - y0) * scale) - 1 + _OUTLINE,
        )
        draw.rectangle(corners, outline=_MARK_COLOUR, width=_OUTLINE)
        _label(draw, (corners[0], corners[1] - _FONT_SIZE - 4), str(number), _MARK_COLOUR, font)
    return np.asarray(image)


def _label(
    draw: ImageDraw.ImageDraw,
    xy: tuple[int, int],
    text: str,
    colour: tuple[int, int, int],
    font: ImageFont.FreeTypeFont | ImageFont.ImageFont,
) -> None:
    """Text with a white halo, legible on print and on paper."""
    draw.text(xy, text, fill=colour, font=font, stroke_width=2, stroke_fill=(255, 255, 255))
