"""What the maps of a trained network show, receipt by receipt (ADR-0022, decision 5).

The maps of one run fill gigabytes, so they are not kept: each receipt is measured while its map
is in memory, and its measurements are kept instead (:class:`RunResults`), from which reports are
made. A quantity that a report needs later means running the network again. Maps are measured in
double precision (:attr:`~docfraud.forensics.learned.inference.LearnedOutput.probabilities`) by
the rules of ADR-0015 (:mod:`docfraud.evaluation.localisation`).

A run's receipts come in three groups (:class:`Group`). The validation split serves thresholds
only (ADR-0022, decision 3): its document scores are kept for the threshold of document F1, and
the maps of its forged receipts choose the threshold of pixel F1 and IoU (ADR-0015, section 5);
nothing else is measured on it. The held-out fold and the synthetic development set are measured
in full: document scores, the lowest value and the peak of the map, the top region, the cuts
between windows, and on forged receipts the pixel metrics; on synthetic forgeries also each edit,
by its highest map value and its average precision against the clean pixels of the receipt.

The document scores of ADR-0023 need the receipt's text height, from its OCR word boxes, which
every item carries (a synthetic forgery those of its base). On the held-out fold the results also
keep the peak of each word box at the four scales of that record's diagnostic.

The clean measurement measures the maps of systems 1 and 6 the same way: anything that gives a
map with its windows, time and document scores (:class:`MapOutput`) is measured by these rules.
"""

from collections.abc import Callable, Iterable, Iterator, Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import StrEnum
from pathlib import Path
from typing import Protocol

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, ValidationError

from docfraud.data.models import BBox, Document, Label, Technique
from docfraud.evaluation.localisation import (
    annotated_rectangles,
    choose_pixel_threshold,
    edit_average_precisions,
    f1_at_thresholds,
    pixel_average_precision,
    pixel_f1_iou,
    positive_share,
    threshold_candidates,
    threshold_sample,
    top_region,
)
from docfraud.evaluation.runs import CodeVersion, Environment
from docfraud.forensics.learned.text_scale import WordScale, on_page, text_height, word_peaks
from docfraud.forensics.trufor import cut_check
from docfraud.synthesis.engine import Forgery

RESULTS = "results.json"
"""The file of a run's results, next to its weights and record."""


class Group(StrEnum):
    """Which receipts of a round a result belongs to."""

    HELD_OUT = "held-out"  # the fold the round holds out; the design fold in round 2
    SYNTHETIC = "synthetic"  # the synthetic development set, one forgery per authentic receipt
    VALIDATION = "validation"  # the validation split, for thresholds only


@dataclass(frozen=True, slots=True)
class Edit:
    """One edit of a synthetic forgery."""

    technique: Technique
    box: BBox  # encloses the area the engine wrote


@dataclass(frozen=True, slots=True)
class Item:
    """A receipt to measure: its page and what is known to be modified on it."""

    key: str  # the receipt's identifier; for a synthetic forgery, that of its base
    group: Group
    page: npt.NDArray[np.uint8]
    forged: bool
    modified: tuple[BBox, ...] = ()  # the modified rectangles; none if not annotated
    sources: tuple[BBox, ...] = ()  # copy sources on this page, ignored by the pixel metrics
    edits: tuple[Edit, ...] = ()  # the edits of a synthetic forgery
    words: tuple[BBox, ...] = ()  # the OCR word boxes; of its base for a synthetic forgery


def real_item(
    document: Document, group: Group, page: npt.NDArray[np.uint8], words: Sequence[BBox] = ()
) -> Item:
    """A receipt of the dataset with its annotated rectangles and its OCR word boxes."""
    modified, sources = annotated_rectangles(document)
    return Item(
        document.doc_id,
        group,
        page,
        document.label is Label.FORGED,
        tuple(modified),
        tuple(sources),
        words=tuple(words),
    )


def synthetic_item(forgery: Forgery, words: Sequence[BBox] = ()) -> Item:
    """A synthetic forgery, keyed by its base; the sources of copies from the same page count.

    Its words are those of its base, as in the report's random baseline.
    """
    regions = forgery.record.regions
    return Item(
        forgery.record.base,
        Group.SYNTHETIC,
        forgery.page,
        forged=True,
        modified=tuple(r.box for r in regions),
        sources=tuple(box for r in regions if r.source_document is None for box in r.sources),
        edits=tuple(Edit(r.technique, r.box) for r in regions),
        words=tuple(words),
    )


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid", ser_json_inf_nan="constants")


class ScoredReceipt(_Frozen):
    """A receipt's document scores and how it was analysed."""

    key: str
    group: Group
    forged: bool
    height: int
    width: int
    windows: int
    seconds: float  # time the network took on the machine that ran it
    word_count: int  # OCR word boxes on the page
    text_height: float  # the median height of those boxes, or that of the receipt's kind
    scores: dict[str, float]  # by name, in the order of ``inference.SCORE_ORDER``


class PixelScores(_Frozen):
    """The pixel metrics of a forged receipt (ADR-0015, section 5)."""

    average_precision: float
    chance: float  # the share of modified pixels among those counted
    f1: float  # at the run's pixel threshold
    iou: float


class EditResult(_Frozen):
    """How the map shows one edit of a synthetic forgery."""

    technique: Technique
    box: BBox
    peak: float  # the highest map value inside the edit's rectangle
    average_precision: float  # the edit's pixels against the clean pixels of the receipt
    chance: float  # the share of the edit's pixels among those


class MeasuredReceipt(ScoredReceipt):
    """Everything measured on a receipt of the held-out fold or the synthetic set."""

    lowest: float  # the lowest map value; a map whose values are all equal has no evidence
    peak: tuple[int, int]  # row and column of the first highest map value, in row-major order
    top_region: BBox
    cuts: tuple[float, float] | None  # receipts in several windows: see ``trufor.cut_check``
    modified: tuple[BBox, ...]
    sources: tuple[BBox, ...]
    pixels: PixelScores | None  # forged receipts with a modified rectangle
    edits: tuple[EditResult, ...]
    word_peaks: dict[WordScale, tuple[float, ...]] | None  # held-out: ADR-0023 diagnostic


class PixelThreshold(_Frozen):
    """The threshold of pixel F1 and IoU, chosen on the forged validation receipts."""

    receipts: int
    candidates: tuple[float, ...]
    mean_f1: tuple[float, ...]  # over the receipts, at each candidate
    value: float


class RunResults(_Frozen):
    """The measurements of one run's network."""

    held_out: int
    system: int
    seed: int
    model: str  # the network's identifier: weights, PyTorch version and device
    code: CodeVersion
    environment: Environment
    created: datetime
    threshold: PixelThreshold
    validation: tuple[ScoredReceipt, ...]
    measured: tuple[MeasuredReceipt, ...]


class MapOutput(Protocol):
    """A system's outputs on one page, such as a ``LearnedOutput``."""

    @property
    def windows(self) -> tuple[BBox, ...]:
        """The windows the page was analysed in."""
        ...

    @property
    def seconds(self) -> float:
        """Time the analysis took on the machine that ran it."""
        ...

    @property
    def values(self) -> npt.NDArray[np.float64]:
        """The map in double precision, one value per pixel, higher meaning more suspicious."""
        ...

    def scores(
        self, text_height: float, values: npt.NDArray[np.float64] | None = None, /
    ) -> dict[str, float]:
        """The document scores, by name, from the receipt's text height and the map if known."""
        ...


class Analyser(Protocol):
    """A system that analyses whole pages, such as a ``LearnedModel``."""

    @property
    def identifier(self) -> str:
        """Weights, version and device; changes whenever outputs may change."""
        ...

    def analyse(self, page: npt.NDArray[np.uint8]) -> MapOutput:
        """The system's map of a page."""
        ...


type Progress = Callable[[Group, str], None]


@dataclass(frozen=True, slots=True)
class HeldMap:
    """The output of a forged validation receipt, held until the pixel threshold is chosen."""

    key: str
    modified: tuple[BBox, ...]
    sources: tuple[BBox, ...]
    output: MapOutput  # a learned output holds its logits in single precision


def measure_run(  # noqa: PLR0913 - the network, its receipts and what the results state
    model: Analyser,
    validation: Iterable[Item],
    measured: Iterable[Item],
    *,
    held_out: int,
    system: int,
    seed: int,
    code: CodeVersion,
    environment: Environment,
    on_progress: Progress | None = None,
    peaks: bool = True,
) -> RunResults:
    """Measure a run's network: the validation receipts first, for the thresholds, then the rest.

    The items are taken one at a time, so that a generator can load each page when it is needed.
    The outputs of the forged validation receipts are held until the pixel threshold is chosen.
    Without ``peaks``, the word peaks of ADR-0023's diagnostic, which concerns the learned models,
    are not measured.

    Raises:
        ValueError: If an item is in the wrong group, a key repeats within a group, or no forged
            validation receipt has a modified rectangle.
    """
    scored: list[ScoredReceipt] = []
    held: list[HeldMap] = []
    for item in _unique(validation, (Group.VALIDATION,)):
        output = model.analyse(item.page)
        scored.append(_scored(item, output))
        if item.modified:
            held.append(HeldMap(item.key, item.modified, item.sources, output))
        if on_progress is not None:
            on_progress(item.group, item.key)
    threshold = choose_threshold(held)
    held.clear()  # free the maps before the larger groups
    results = []
    for item in _unique(measured, (Group.HELD_OUT, Group.SYNTHETIC)):
        output = model.analyse(item.page)
        results.append(measure_receipt(item, output, threshold.value, peaks=peaks))
        if on_progress is not None:
            on_progress(item.group, item.key)
    return RunResults(
        held_out=held_out,
        system=system,
        seed=seed,
        model=model.identifier,
        code=code,
        environment=environment,
        created=datetime.now(UTC),
        threshold=threshold,
        validation=tuple(scored),
        measured=tuple(results),
    )


def _unique(items: Iterable[Item], groups: Sequence[Group]) -> Iterator[Item]:
    """The items, refusing one of another group or a key seen before in its group."""
    seen: set[tuple[Group, str]] = set()
    for item in items:
        if item.group not in groups:
            msg = f"{item.key} belongs to {item.group}, not to {', '.join(groups)}"
            raise ValueError(msg)
        if (item.group, item.key) in seen:
            msg = f"{item.key} appears twice in {item.group}"
            raise ValueError(msg)
        seen.add((item.group, item.key))
        yield item


def choose_threshold(held: Sequence[HeldMap]) -> PixelThreshold:
    """The pixel threshold of ADR-0015 (section 5) from the maps of forged receipts.

    Raises:
        ValueError: If there is no receipt.
    """
    if not held:
        msg = "no forged validation receipt with a modified rectangle to choose a threshold"
        raise ValueError(msg)
    samples = [threshold_sample(h.output.values, h.key) for h in held]
    candidates = threshold_candidates(samples)
    f1 = [f1_at_thresholds(h.output.values, candidates, h.modified, h.sources) for h in held]
    return PixelThreshold(
        receipts=len(held),
        candidates=tuple(float(c) for c in candidates),
        mean_f1=tuple(float(m) for m in np.mean(f1, axis=0)),
        value=choose_pixel_threshold(candidates, f1),
    )


def measure_receipt(
    item: Item, output: MapOutput, threshold: float, *, peaks: bool = True
) -> MeasuredReceipt:
    """Everything a report needs from one map of the held-out fold or the synthetic set."""
    values = output.values
    pixels = None
    if item.modified:
        f1, iou = pixel_f1_iou(values, threshold, item.modified, item.sources)
        pixels = PixelScores(
            average_precision=pixel_average_precision(values, item.modified, item.sources),
            chance=positive_share(values.shape, item.modified, item.sources),
            f1=f1,
            iou=iou,
        )
    edits: tuple[EditResult, ...] = ()
    if item.edits:
        boxes = [edit.box for edit in item.edits]
        precisions = edit_average_precisions(values, boxes, item.sources)
        edits = tuple(
            EditResult(
                technique=edit.technique,
                box=edit.box,
                peak=float(values[edit.box.y : edit.box.y2, edit.box.x : edit.box.x2].max()),
                average_precision=precision.average_precision,
                chance=precision.chance,
            )
            for edit, precision in zip(item.edits, precisions, strict=True)
        )
    return MeasuredReceipt(
        **_scored(item, output, values).model_dump(),
        lowest=float(values.min()),
        peak=_first_maximum(values),
        top_region=top_region(values),
        cuts=cut_check(values) if len(output.windows) > 1 else None,
        modified=item.modified,
        sources=item.sources,
        pixels=pixels,
        edits=edits,
        word_peaks=word_peaks(values, item.words)
        if peaks and item.group is Group.HELD_OUT
        else None,
    )


def _first_maximum(values: npt.NDArray[np.float64]) -> tuple[int, int]:
    row, column = np.unravel_index(int(np.argmax(values)), values.shape)
    return int(row), int(column)


def _scored(
    item: Item, output: MapOutput, values: npt.NDArray[np.float64] | None = None
) -> ScoredReceipt:
    height, width = item.page.shape
    text = text_height(item.words, height, width)
    return ScoredReceipt(
        key=item.key,
        group=item.group,
        forged=item.forged,
        height=height,
        width=width,
        windows=len(output.windows),
        seconds=output.seconds,
        word_count=len(on_page(item.words, height, width)),
        text_height=text,
        scores=output.scores(text, values),
    )


def write_results(directory: Path, results: RunResults) -> Path:
    """Write a run's results next to its weights, replacing the file only when complete."""
    path = directory / RESULTS
    temporary = directory / f"{RESULTS}.tmp"
    temporary.write_text(results.model_dump_json(indent=1) + "\n", encoding="utf-8")
    temporary.replace(path)
    return path


def read_results(directory: Path) -> RunResults:
    """The results written by :func:`write_results`."""
    return RunResults.model_validate_json((directory / RESULTS).read_text(encoding="utf-8"))


def stored_results(directory: Path) -> RunResults | None:
    """The results kept in a directory, unless none are or this code cannot read their form.

    Results are made again from the system, so results of an earlier form are measured again
    rather than read.
    """
    if not (directory / RESULTS).exists():
        return None
    try:
        return read_results(directory)
    except ValidationError:
        return None
