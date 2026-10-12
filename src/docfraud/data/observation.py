"""Observing how the forgers of the design fold worked (ADR-0014, step 4.2).

Before the synthetic forgery engine and the reference methods are designed, two observers record
what each annotated edit of the design fold looks like, on a form that was fixed before any
receipt was viewed for this purpose (``docs/forgery-observation-guide.md``). The assistant
answers the region fields that need neither a view of the whole receipt nor a judgement of what a
person sees at 100 %, for every region, from crops at native resolution (:func:`export_crops`),
because larger images reach it downscaled. The project owner answers, on the page served by
:mod:`docfraud.data.observation_server`, what only a person can: visibility at 100 % for every
region and the questions about whole receipts; and the full form for a seeded sample of regions,
which measures how far the two observers agree. Neither observer sees the other's answers: the
assistant's log is sealed by its SHA-256 (:func:`seal`) before the owner starts.

Observers see the normalised grayscale image that the models see, and never the technique,
entity or software labels: an observer who knows that an edit is labelled copy-paste looks for a
pasted patch. Every receipt passes :func:`~docfraud.evaluation.design_fold.require_design_fold`
before anything of it is shown.
"""

import hashlib
import io
from collections.abc import Iterable, Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Literal, Self

import numpy as np
import numpy.typing as npt
from PIL import Image, ImageDraw
from pydantic import BaseModel, ConfigDict, Field, model_validator

from docfraud.data.models import BBox, Document, Label
from docfraud.data.normalize import load_grayscale
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS, require_design_fold

FORM_VERSION = 1
"""Version of the observation form; a log written for another version is refused."""

OBSERVATION_SEED = 0
"""Seed of the order in which regions and receipts are shown (decision 3c)."""

CROP_MARGIN = 200
"""Rows shown above and below an annotated region, and columns beside it in the assistant's
close-up, in pixels."""

ASSISTANT_TILE = 1500
"""Longest side of an image shown to the assistant. Images above about 2000 px reach it
downscaled, which erases one-pixel detail; 1500 px leaves a margin."""

ASSISTANT = "claude"
"""Observer name of the assistant, which answers the region fields only (decision 3a)."""

AGREEMENT_REGIONS = 12
"""Regions on which the owner answers the full form, one from each of as many receipts."""

AGREEMENT_SEED = 20261012
"""Seed of the agreement sample; it depends on the region identifiers only, never on answers."""

DETAIL_FIELDS = (
    "change",
    "source",
    "background",
    "edge",
    "alignment",
    "font",
    "tone",
    "sharpness",
    "scaled_or_rotated",
    "box_fit",
)
"""Fields 1 to 9 and 11: the assistant answers them for every region, the owner for the
agreement sample."""

_OUTLINE = 2
_OUTLINE_COLOUR = (220, 0, 0)


class Visibility(StrEnum):
    """Field 10: whether the edit can be seen at 100 % (owner only)."""

    AT_100 = "at_100"
    ONLY_ZOOMED = "only_zoomed"
    NOT_FOUND = "not_found"
    UNCLEAR = "unclear"


class Change(StrEnum):
    """Field 1: what the edit changed."""

    SINGLE_CHARACTERS = "single_characters"
    WHOLE_VALUE = "whole_value"
    LINE_OR_LARGER = "line_or_larger"
    CONTENT_ADDED = "content_added"
    DELETED = "deleted"
    UNCLEAR = "unclear"


class Source(StrEnum):
    """Field 2: where pasted content came from."""

    SAME_LINE = "same_line"
    OTHER_LINE = "other_line"
    OTHER_RECEIPT = "other_receipt"
    NOT_VISIBLE = "not_visible"
    NOT_APPLICABLE = "not_applicable"
    UNCLEAR = "unclear"


class Background(StrEnum):
    """Field 3: the paper behind the edit compared with the paper around it."""

    MATCHES = "matches"
    WHITE = "white"
    FLAT = "flat"
    DARKER = "darker"
    UNCLEAR = "unclear"


class Edge(StrEnum):
    """Field 4: the border between the edit and its surroundings."""

    SHARP_RECTANGLE = "sharp_rectangle"
    SOFT = "soft"
    NOT_VISIBLE = "not_visible"
    UNCLEAR = "unclear"


class Alignment(StrEnum):
    """Field 5: the edit against the baseline and spacing of its line."""

    SHIFTED = "shifted"
    ALIGNED = "aligned"
    UNCLEAR = "unclear"


class Font(StrEnum):
    """Field 6: the font and stroke weight of new text against the print around it."""

    SAME = "same"
    OTHER_FONT = "other_font"
    OTHER_WEIGHT = "other_weight"
    NOT_APPLICABLE = "not_applicable"
    UNCLEAR = "unclear"


class Tone(StrEnum):
    """Field 7: how dark new text is against the print around it."""

    SAME = "same"
    DARKER = "darker"
    LIGHTER = "lighter"
    NOT_APPLICABLE = "not_applicable"
    UNCLEAR = "unclear"


class Sharpness(StrEnum):
    """Field 8: the sharpness of the edit against its neighbours."""

    SAME = "same"
    BLURRIER = "blurrier"
    SHARPER = "sharper"
    UNCLEAR = "unclear"


class Presence(StrEnum):
    """Fields 9, 12 and 13: yes, no or unclear."""

    YES = "yes"
    NO = "no"
    UNCLEAR = "unclear"


class BoxFit(StrEnum):
    """Field 11: the annotated rectangle against the edit."""

    TIGHT = "tight"
    MUCH_LARGER = "much_larger"
    MISSES_PART = "misses_part"
    UNCLEAR = "unclear"


@dataclass(frozen=True, slots=True)
class FormField:
    """One question of the form, as the page shows it."""

    number: int
    name: str
    question: str
    choices: Mapping[str, str]  # value -> label
    scope: Literal["region", "receipt"]
    assistant: bool  # also answered by the assistant
    detail: str | None = None  # free-text attribute that the answer "yes" requires


FORM: tuple[FormField, ...] = (
    FormField(
        10,
        "visibility",
        "Can you see the edit at 100 %?",
        {
            Visibility.AT_100: "Yes, at 100 %",
            Visibility.ONLY_ZOOMED: "Only when zoomed in",
            Visibility.NOT_FOUND: "Not found",
            Visibility.UNCLEAR: "Unclear",
        },
        "region",
        assistant=False,
    ),
    FormField(
        1,
        "change",
        "What did the edit change?",
        {
            Change.SINGLE_CHARACTERS: "Single characters",
            Change.WHOLE_VALUE: "A whole value (number, word or date)",
            Change.LINE_OR_LARGER: "A line or more",
            Change.CONTENT_ADDED: "New content added",
            Change.DELETED: "Deleted, nothing in its place",
            Change.UNCLEAR: "Unclear",
        },
        "region",
        assistant=True,
    ),
    FormField(
        2,
        "source",
        "Where does pasted content come from?",
        {
            Source.SAME_LINE: "The same line",
            Source.OTHER_LINE: "Another line of this receipt",
            Source.OTHER_RECEIPT: "Another receipt",
            Source.NOT_VISIBLE: "Pasted, source not visible",
            Source.NOT_APPLICABLE: "Not applicable: nothing pasted",
            Source.UNCLEAR: "Unclear",
        },
        "region",
        assistant=True,
    ),
    FormField(
        3,
        "background",
        "How does the paper behind the edit compare with the paper around it?",
        {
            Background.MATCHES: "Matches",
            Background.WHITE: "Different: pure white",
            Background.FLAT: "Different: flat, no paper texture",
            Background.DARKER: "Different: darker",
            Background.UNCLEAR: "Unclear",
        },
        "region",
        assistant=True,
    ),
    FormField(
        4,
        "edge",
        "What does the border of the edit look like?",
        {
            Edge.SHARP_RECTANGLE: "A sharp rectangular edge",
            Edge.SOFT: "A soft transition",
            Edge.NOT_VISIBLE: "No edge visible",
            Edge.UNCLEAR: "Unclear",
        },
        "region",
        assistant=True,
    ),
    FormField(
        5,
        "alignment",
        "Does the edit sit on the baseline and spacing of its line?",
        {
            Alignment.SHIFTED: "Shifted",
            Alignment.ALIGNED: "Aligned",
            Alignment.UNCLEAR: "Unclear",
        },
        "region",
        assistant=True,
    ),
    FormField(
        6,
        "font",
        "How do the font and stroke weight of new text compare with the print around it?",
        {
            Font.SAME: "Same font and weight",
            Font.OTHER_FONT: "Different font",
            Font.OTHER_WEIGHT: "Same font, different weight",
            Font.NOT_APPLICABLE: "Not applicable: no new text",
            Font.UNCLEAR: "Unclear",
        },
        "region",
        assistant=True,
    ),
    FormField(
        7,
        "tone",
        "How dark is new text compared with the print around it?",
        {
            Tone.SAME: "The same",
            Tone.DARKER: "Darker",
            Tone.LIGHTER: "Lighter",
            Tone.NOT_APPLICABLE: "Not applicable: no new text",
            Tone.UNCLEAR: "Unclear",
        },
        "region",
        assistant=True,
    ),
    FormField(
        8,
        "sharpness",
        "How sharp is the edit compared with its neighbours?",
        {
            Sharpness.SAME: "The same",
            Sharpness.BLURRIER: "Blurrier",
            Sharpness.SHARPER: "Sharper",
            Sharpness.UNCLEAR: "Unclear",
        },
        "region",
        assistant=True,
    ),
    FormField(
        9,
        "scaled_or_rotated",
        "Was the content scaled or rotated?",
        {Presence.YES: "Yes", Presence.NO: "No", Presence.UNCLEAR: "Unclear"},
        "region",
        assistant=True,
    ),
    FormField(
        11,
        "box_fit",
        "How does the annotated rectangle fit the edit?",
        {
            BoxFit.TIGHT: "Tight",
            BoxFit.MUCH_LARGER: "Much larger: the edit covers less than half",
            BoxFit.MISSES_PART: "Misses part of the edit",
            BoxFit.UNCLEAR: "Unclear",
        },
        "region",
        assistant=True,
    ),
    FormField(
        12,
        "unannotated_edit",
        "Did you clearly see an edit that no rectangle marks? (Not a full search.)",
        {Presence.YES: "Yes", Presence.NO: "No", Presence.UNCLEAR: "Unclear"},
        "receipt",
        assistant=False,
        detail="unannotated_edit_where",
    ),
    FormField(
        13,
        "global_trace",
        "Does the whole receipt differ in a way that is not local (tone, sharpness, border)?",
        {Presence.YES: "Yes", Presence.NO: "No", Presence.UNCLEAR: "Unclear"},
        "receipt",
        assistant=False,
        detail="global_trace_what",
    ),
)
"""The 13 fields in the order the page asks them: field 10 first, before the rectangle is
shown (decision 3b)."""


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Display(_Frozen):
    """The screen an answer was given on, as the browser reports it."""

    device_pixel_ratio: float = Field(gt=0)
    screen_width: int = Field(gt=0)
    screen_height: int = Field(gt=0)
    window_width: int = Field(gt=0)
    window_height: int = Field(gt=0)


class RegionObservation(_Frozen):
    """The answers for one annotated region: visibility, the region fields, or both."""

    visibility: Visibility | None = None
    change: Change | None = None
    source: Source | None = None
    background: Background | None = None
    edge: Edge | None = None
    alignment: Alignment | None = None
    font: Font | None = None
    tone: Tone | None = None
    sharpness: Sharpness | None = None
    scaled_or_rotated: Presence | None = None
    box_fit: BoxFit | None = None
    note: str = ""
    observed_at: datetime
    commit: str | None
    display: Display | None = None

    @property
    def detailed(self) -> bool:
        """Whether fields 1 to 9 and 11 are answered."""
        return self.change is not None

    @model_validator(mode="after")
    def _details_together(self) -> Self:
        given = [getattr(self, name) is not None for name in DETAIL_FIELDS]
        if any(given) and not all(given):
            msg = "fields 1 to 9 and 11 are answered together or not at all"
            raise ValueError(msg)
        if not any(given) and self.visibility is None:
            msg = "an observation needs visibility, the region fields, or both"
            raise ValueError(msg)
        return self


class ReceiptObservation(_Frozen):
    """The answers for one forged receipt as a whole (owner only)."""

    unannotated_edit: Presence
    unannotated_edit_where: str = ""
    global_trace: Presence
    global_trace_what: str = ""
    note: str = ""
    observed_at: datetime
    commit: str | None
    display: Display | None = None

    @model_validator(mode="after")
    def _yes_is_described(self) -> Self:
        for answer, detail in (
            (self.unannotated_edit, self.unannotated_edit_where),
            (self.global_trace, self.global_trace_what),
        ):
            if answer is Presence.YES and not detail.strip():
                msg = "an answer 'yes' needs a description"
                raise ValueError(msg)
        return self


class ObservationLog(_Frozen):
    """All answers of one observer."""

    form_version: int = FORM_VERSION
    observer: str = Field(pattern=r"^[a-z][a-z0-9-]*$")
    regions: dict[str, RegionObservation] = Field(default_factory=dict)
    receipts: dict[str, ReceiptObservation] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _matches_the_observer(self) -> Self:
        if self.form_version != FORM_VERSION:
            msg = f"log of form version {self.form_version}, expected {FORM_VERSION}"
            raise ValueError(msg)
        regions = self.regions.values()
        if self.observer == ASSISTANT:
            if self.receipts or any(r.visibility or not r.detailed for r in regions):
                msg = "the assistant answers the region fields only, for every region"
                raise ValueError(msg)
        elif any(r.visibility is None for r in regions):
            msg = "every region of the owner's log needs an answer on visibility"
            raise ValueError(msg)
        return self


def load_log(path: Path, observer: str) -> ObservationLog:
    """Read an observer's log, or start an empty one if none exists yet.

    Raises:
        ValueError: If the log belongs to another observer or another form version.
    """
    if not path.exists():
        return ObservationLog(observer=observer)
    log = ObservationLog.model_validate_json(path.read_text(encoding="utf-8"))
    if log.observer != observer:
        msg = f"{path} is the log of {log.observer!r}, not {observer!r}"
        raise ValueError(msg)
    return log


def save_log(log: ObservationLog, path: Path) -> None:
    """Write a log so that an interruption never leaves a half-written file."""
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(path.name + ".partial")
    partial.write_text(log.model_dump_json(indent=2) + "\n", encoding="utf-8")
    partial.replace(path)


def seal(log_path: Path, seal_path: Path) -> str:
    """Record the SHA-256 of a log, in the format of ``shasum -a 256 -c``.

    Committing the seal before the other observer starts shows later that the log was not
    changed after the other answers became visible.

    Returns:
        The hexadecimal digest.
    """
    digest = hashlib.sha256(log_path.read_bytes()).hexdigest()
    seal_path.parent.mkdir(parents=True, exist_ok=True)
    seal_path.write_text(f"{digest}  {log_path.name}\n", encoding="utf-8")
    return digest


def seal_matches(log_path: Path, seal_path: Path) -> bool:
    """Whether a log is still exactly the file that was sealed."""
    recorded = seal_path.read_text(encoding="utf-8").split()[0]
    return hashlib.sha256(log_path.read_bytes()).hexdigest() == recorded


@dataclass(frozen=True, slots=True)
class RegionItem:
    """One annotated modified region of the design fold."""

    key: str
    doc_id: str
    box: BBox


@dataclass(frozen=True, slots=True)
class ObservationPlan:
    """What is observed, in the order it is shown, and where the owner answers the full form."""

    regions: tuple[RegionItem, ...]
    receipts: tuple[str, ...]
    full_form: frozenset[str]


def observation_plan(documents: Iterable[Document]) -> ObservationPlan:
    """The modified regions and forged receipts of the design fold, each in a seeded order.

    Regions are shuffled across receipts, so that the edits of one forger do not follow each
    other. A region is named by its receipt and its position among the receipt's modified
    regions, for example ``X51005230617-0``. The agreement sample takes one region from each of
    ``AGREEMENT_REGIONS`` receipts, so that receipts with many edits do not dominate it.
    """
    forged = sorted(
        (d for d in documents if d.doc_id in DESIGN_FOLD_IDS and d.label is Label.FORGED),
        key=lambda d: d.doc_id,
    )
    regions = [
        RegionItem(f"{d.doc_id}-{i}", d.doc_id, region.box)
        for d in forged
        for i, region in enumerate(d.modified_regions)
    ]
    rng = np.random.default_rng(OBSERVATION_SEED)
    region_order = rng.permutation(len(regions))
    receipt_order = rng.permutation(len(forged))
    return ObservationPlan(
        regions=tuple(regions[i] for i in region_order),
        receipts=tuple(forged[i].doc_id for i in receipt_order),
        full_form=_agreement_sample(regions),
    )


def _agreement_sample(regions: list[RegionItem]) -> frozenset[str]:
    by_receipt: dict[str, list[str]] = {}
    for item in regions:
        by_receipt.setdefault(item.doc_id, []).append(item.key)
    receipts = sorted(by_receipt)
    rng = np.random.default_rng(AGREEMENT_SEED)
    size = min(AGREEMENT_REGIONS, len(receipts))
    chosen = sorted(int(i) for i in rng.choice(len(receipts), size=size, replace=False))
    keys = [by_receipt[receipts[i]] for i in chosen]
    return frozenset(options[int(rng.integers(len(options)))] for options in keys)


def crop_rows(box: BBox, height: int, margin: int = CROP_MARGIN) -> tuple[int, int]:
    """The rows a region crop shows: the region and ``margin`` rows above and below it.

    Returns:
        First row and the row after the last, inside an image of ``height`` rows.
    """
    return max(0, box.y - margin), min(height, box.y2 + margin)


def observed_pixels(document: Document) -> npt.NDArray[np.uint8]:
    """The image an observer sees: the normalised grayscale page, after the viewing lock.

    Raises:
        NotInDesignFoldError: If the receipt is outside the design fold.
    """
    require_design_fold(document.doc_id)
    return load_grayscale(document.image_path)


def png_bytes(pixels: npt.NDArray[np.uint8]) -> bytes:
    """Lossless PNG encoding of a grayscale or RGB array."""
    buffer = io.BytesIO()
    Image.fromarray(pixels).save(buffer, format="PNG")
    return buffer.getvalue()


def export_crops(documents: Iterable[Document], plan: ObservationPlan, out_dir: Path) -> list[Path]:
    """Write the crops the assistant observes, at native resolution and in plan order.

    For every region two views are written: a close-up of the region with ``CROP_MARGIN`` pixels
    on every side, and the strip the owner's page shows (the full receipt width, ``CROP_MARGIN``
    rows above and below). Both are cut into tiles of at most ``ASSISTANT_TILE`` pixels a side,
    so that nothing reaches the assistant downscaled. A red outline is drawn just outside the
    region; the receipt's own pixels are left unchanged.

    Returns:
        The written files.

    Raises:
        NotInDesignFoldError: If a planned region lies outside the design fold.
    """
    by_id = {d.doc_id: d for d in documents}
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for number, item in enumerate(plan.regions, start=1):
        page = observed_pixels(by_id[item.doc_id])
        height, width = page.shape
        top, bottom = crop_rows(item.box, height)
        left, right = max(0, item.box.x - CROP_MARGIN), min(width, item.box.x2 + CROP_MARGIN)
        views = {
            "close-up": (page[top:bottom, left:right], left, top),
            "strip": (page[top:bottom], 0, top),
        }
        for view, (pixels, x0, y0) in views.items():
            marked = _outline(pixels, item.box, x0, y0)
            for index, tile in enumerate(_tiles(marked)):
                path = out_dir / f"{number:02d}-{item.key}-{view}-{index}.png"
                path.write_bytes(png_bytes(tile))
                written.append(path)
    return written


def _outline(pixels: npt.NDArray[np.uint8], box: BBox, x0: int, y0: int) -> npt.NDArray[np.uint8]:
    image = Image.fromarray(pixels).convert("RGB")
    x, y = box.x - x0, box.y - y0
    shape = (
        x - _OUTLINE,
        y - _OUTLINE,
        x + box.width - 1 + _OUTLINE,
        y + box.height - 1 + _OUTLINE,
    )
    ImageDraw.Draw(image).rectangle(shape, outline=_OUTLINE_COLOUR, width=_OUTLINE)
    return np.asarray(image)


def _tiles(
    pixels: npt.NDArray[np.uint8], size: int = ASSISTANT_TILE
) -> list[npt.NDArray[np.uint8]]:
    height, width = pixels.shape[:2]
    return [
        pixels[top : top + size, left : left + size]
        for top in range(0, height, size)
        for left in range(0, width, size)
    ]
