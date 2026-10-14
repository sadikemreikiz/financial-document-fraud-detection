"""Development round of system 6, TruFor (ADR-0019), on the design fold.

Writes reports/forensics/system-6-development.md, the evidence behind the choices recorded in
docs/system-6-development.md: how the receipts were tiled, which tiles the highest tile score
comes from, which document score passes the size check of ADR-0015 and ADR-0017 under the rules
of the record, how often the top region of TruFor's map hits an edit against its random
baseline, and whether it falls on a benign mark. Only the design fold is read; every
measurement is computed by code.

TruFor runs on the Apple GPU, whose results differ slightly from the CPU's, so this report is
not part of ``make reports``. The outputs are cached (``CachedTruFor``) and the report
regenerates from the cache. PyTorch is supplied by ``uv run --with``.

Usage: ``make trufor``
"""

import time
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass, replace
from importlib.metadata import version
from pathlib import Path

import numpy as np
import numpy.typing as npt

from docfraud.data.benign_marks import Mark, final_marks, load_rounds, marks_plan
from docfraud.data.findit import load_findit2
from docfraud.data.models import BBox, Document, Label
from docfraud.data.normalize import load_grayscale
from docfraud.data.observation import ASSISTANT
from docfraud.evaluation.bootstrap import Estimate
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS, require_design_fold
from docfraud.evaluation.detection import average_precision
from docfraud.evaluation.localisation import (
    annotated_rectangles,
    baseline_hit_rate,
    hits,
    receipt_rng,
    size_dependence,
    top_region,
)
from docfraud.evaluation.mark_alarms import MarkCheck, check_marks, mark_tables
from docfraud.evaluation.runs import code_version
from docfraud.forensics.trufor import (
    CITATION,
    OVERLAP,
    TILE,
    TRUFOR_COMMIT,
    CachedTruFor,
    TruForModel,
    TruForOutput,
    analyse,
    axis_windows,
    owners,
    sigmoid,
)
from docfraud.markdown import Cell, Table, render_table
from docfraud.reading.ocr import CachedOcr, RapidOcrEngine

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "forensics" / "system-6-development.md"
DATA = ROOT / "data" / "raw" / "findit2"
OCR_CACHE = ROOT / "artifacts" / "ocr"
TRUFOR = ROOT / "artifacts" / "trufor"
TRUFOR_CACHE = TRUFOR / "outputs"
MARK_RECORDS = ROOT / "annotations" / "findit2" / "benign-marks"
LARGE_SIDE = 6000
INK, BLANK_SHARE = 128, 0.001  # a tile is blank if fewer than 0.1 % of its pixels are below 128
NEAR_CUT = 8  # pixels, one JPEG block
VERSIONED = ("torch", "torchvision", "timm", "yacs", "numpy")

HIGHEST_TILE = "TruFor's integrity score, highest over tiles"
CONTRAST = "highest map value minus the receipt's 99th percentile"
WHOLE = "TruFor's integrity score of the whole receipt"
SCORES: dict[str, Callable[[TruForOutput], float]] = {
    HIGHEST_TILE: lambda o: o.highest_tile_score(),
    CONTRAST: lambda o: float(o.localisation.max() - np.percentile(o.localisation, 99)),
    WHOLE: lambda o: o.integrity_score(),
}
"""Every document score tried for system 6, in the order they were considered."""

FIXED_ORDER = (HIGHEST_TILE, CONTRAST)
"""The order fixed before the run (record, section 1): the first whose size check passes."""

AFTER_RESULTS = WHOLE
"""Added after the results of the fixed order (record, section 3): chosen if its size check
passes, whatever its average precision; otherwise the outcome of the fixed order stands."""


@dataclass(frozen=True, slots=True)
class Measured:
    """TruFor's outputs on one receipt and what the rules of ADR-0015 make of them."""

    doc_id: str
    label: Label
    size: str  # how the receipt was analysed
    pixels: int
    windows: int
    seconds: float
    evidence: bool  # whether the map is not constant (ADR-0017)
    scores: dict[str, float]
    hit: bool | None  # forged receipts with a modified rectangle only
    baseline: float | None
    share: float  # area of the top region over the area of the page
    tiles: tuple[tuple[bool, float], ...]  # tiled receipts: whether each tile is blank, its score
    pooling_error: float  # one window: pooled logit minus TruFor's own logit; NaN if tiled
    cuts: tuple[float, float] | None  # tiled receipts: see cut_check


def cut_check(values: npt.NDArray[np.float32]) -> tuple[float, float]:
    """Where the map's peak lies against the cuts between tiles, and how the map jumps there.

    Returns:
        The distance in pixels from the peak to the nearest cut, and the 99th percentile of the
        jumps between neighbouring pixels across the cuts over that of the jumps inside tiles.
    """
    height, width = values.shape
    rows = np.flatnonzero(np.diff(owners(axis_windows(height), height)))  # last row before a cut
    columns = np.flatnonzero(np.diff(owners(axis_windows(width), width)))
    peak_y, peak_x = np.unravel_index(int(np.argmax(values)), values.shape)
    distance = min(
        [abs(peak_y - (r + 0.5)) for r in rows] + [abs(peak_x - (c + 0.5)) for c in columns]
    )
    vertical = np.abs(np.diff(values, axis=0))
    horizontal = np.abs(np.diff(values, axis=1))
    across = np.concatenate([vertical[rows].ravel(), horizontal[:, columns].ravel()])
    inside = np.concatenate(
        [np.delete(vertical, rows, axis=0).ravel(), np.delete(horizontal, columns, axis=1).ravel()]
    )
    return float(distance), float(np.percentile(across, 99) / np.percentile(inside, 99))


def size_class(height: int, width: int, windows: int) -> str:
    """One pass, tiled, or a page scan, which is always tiled."""
    if max(height, width) > LARGE_SIDE:
        return "page scan, tiled"
    return "one pass" if windows == 1 else "tiled"


SIZES = ("one pass", "tiled", "page scan, tiled")


def measure(
    documents: Sequence[Document],
    trufor: CachedTruFor,
    ocr: CachedOcr,
    marks: Mapping[str, Sequence[Mark]],
) -> tuple[list[Measured], list[MarkCheck]]:
    """Every check on every receipt, in identifier order; ``marks`` holds the flagged ones."""
    measured: list[Measured] = []
    mark_checks: list[MarkCheck] = []
    ordered = sorted(documents, key=lambda d: d.doc_id)
    for count, document in enumerate(ordered, start=1):
        require_design_fold(document.doc_id)
        page = load_grayscale(document.image_path)
        output = trufor.analyse(page)
        height, width = page.shape
        whole = BBox(x=0, y=0, width=width, height=height)
        region = top_region(output.localisation)
        evidence = bool(np.ptp(output.localisation) > 0)
        scores = {name: score(output) for name, score in SCORES.items()}
        blank = [
            float((page[w.y : w.y2, w.x : w.x2] < INK).mean()) < BLANK_SHARE for w in output.windows
        ]
        label = document.label or Label.AUTHENTIC  # every training receipt has a label
        modified, sources = annotated_rectangles(document)
        words = [token.box for token in ocr.recognize(page).tokens]
        hit = baseline = None
        if label is Label.FORGED and modified:
            hit = hits(region, modified + sources)
            baseline = baseline_hit_rate(
                region, whole, words, modified + sources, receipt_rng(document.doc_id)
            )
        measured.append(
            Measured(
                document.doc_id,
                label,
                size_class(height, width, len(output.windows)),
                height * width,
                len(output.windows),
                output.seconds,
                evidence,
                scores,
                hit,
                baseline,
                region.width * region.height / (width * height),
                tuple(zip(blank, map(sigmoid, output.logits), strict=True))
                if len(output.windows) > 1
                else (),
                output.pooled_logit - output.logits[0] if len(output.windows) == 1 else np.nan,
                cut_check(output.localisation) if len(output.windows) > 1 else None,
            )
        )
        if document.doc_id in marks:
            mark_checks.append(
                check_marks(
                    document.doc_id,
                    label,
                    region,
                    whole,
                    words,
                    marks[document.doc_id],
                    evidence=evidence,
                    score=scores[WHOLE],  # replaced by the chosen score once it is known
                )
            )
        print(f"{count}/{len(ordered)} {document.doc_id}", flush=True)  # noqa: T201
    return measured, mark_checks


def repeatability(model: TruForModel, document: Document) -> Table:
    """The same receipt analysed twice on the same device, without the cache."""
    page = load_grayscale(document.image_path)
    first, second = analyse(page, model), analyse(page, model)
    logits = max(abs(a - b) for a, b in zip(first.logits, second.logits, strict=True))
    return Table(
        "Repeatability",
        ["Receipt", "Largest map difference", "Largest logit difference"],
        [
            [
                document.doc_id,
                f"{float(np.abs(first.localisation - second.localisation).max()):.2e}",
                f"{logits:.2e}",
            ]
        ],
        f"The first design-fold receipt, analysed twice on {model.device.type} in one process. "
        "On the CPU the map of a synthetic page differs from the Apple GPU's by up to 4e-4, "
        "which is why the report is not regenerated by `make reports`.",
    )


def run_table(measured: Sequence[Measured]) -> Table:
    """How the receipts were analysed and how long it took."""
    rows: list[list[Cell]] = []
    for size in SIZES:
        chosen = [m for m in measured if m.size == size]
        if not chosen:
            continue
        windows = [m.windows for m in chosen]
        seconds = [m.seconds for m in chosen]
        rows.append(
            [
                size,
                len(chosen),
                sum(m.label is Label.FORGED for m in chosen),
                f"{min(windows)} to {max(windows)}",
                f"{float(np.median(seconds)):.1f}",
                f"{max(seconds):.1f}",
            ]
        )
    return Table(
        "How the receipts were analysed",
        ["Receipts", "Count", "Forged", "Windows", "Median seconds", "Most seconds"],
        rows,
        f"One pass: at most {TILE} x {TILE} pixels. Otherwise tiles of {TILE} pixels per side "
        f"that overlap by at least {OVERLAP} and start at multiples of 8; page scans have a "
        f"longer side above {LARGE_SIDE} px. Seconds: the whole receipt, on the machine that "
        "filled the cache.",
    )


def tile_table(measured: Sequence[Measured]) -> Table:
    """Whether the highest tile score of a tiled receipt comes from a tile without print."""
    rows: list[list[Cell]] = []
    for size in SIZES[1:]:
        chosen = [m for m in measured if m.size == size]
        if not chosen:
            continue
        blank = [[s for b, s in m.tiles if b] for m in chosen]
        inked = [[s for b, s in m.tiles if not b] for m in chosen]
        rows.append(
            [
                size,
                len(chosen),
                sum(len(m.tiles) for m in chosen),
                sum(map(len, blank)),
                sum(max(m.tiles, key=lambda tile: tile[1])[0] for m in chosen),
                float(np.median([max(s) for s in blank if s])) if any(blank) else None,
                float(np.median([max(s) for s in inked if s])),
            ]
        )
    return Table(
        "Where the highest tile score comes from",
        [
            "Tiled receipts",
            "Count",
            "Tiles",
            "Blank tiles",
            "Receipts whose highest tile is blank",
            "Median highest score of a blank tile",
            "Median highest score of a tile with print",
        ],
        rows,
        f"Blank: fewer than {100 * BLANK_SHARE:.1f} % of the tile's pixels are darker than "
        f"{INK}. Descriptive, found after the results of the fixed order; it is why the record "
        "adds the integrity score of the whole receipt (section 3).",
    )


def cut_table(measured: Sequence[Measured]) -> Table:
    """The map of tiled receipts at the cuts between tiles."""
    rows: list[list[Cell]] = []
    for size in SIZES[1:]:
        cuts = [m.cuts for m in measured if m.size == size and m.cuts is not None]
        if cuts:
            rows.append(
                [
                    size,
                    len(cuts),
                    sum(distance <= NEAR_CUT for distance, _ in cuts),
                    sum(ratio > 1 for _, ratio in cuts),
                    float(np.median([ratio for _, ratio in cuts])),
                    max(ratio for _, ratio in cuts),
                ]
            )
    return Table(
        "The cuts between tiles",
        [
            "Tiled receipts",
            "Count",
            f"Peak of the map within {NEAR_CUT} px of a cut",
            "Jumps larger across cuts than inside tiles",
            "Median ratio",
            "Largest ratio",
        ],
        rows,
        "A cut lies between two pixels that come from different tiles. Jumps: absolute "
        "differences of the map between neighbouring pixels; the ratio compares their 99th "
        "percentiles across the cuts and inside the tiles. Each tile sees another context, so "
        "some jump at the cuts is expected; values put in the wrong place, a defect of the "
        "stitching, are excluded by the tests. Descriptive, added after the results of the fixed "
        "order.",
    )


def pooling_table(measured: Sequence[Measured]) -> Table:
    """How closely the pooled logit reproduces TruFor's own on receipts of one window."""
    errors = [abs(m.pooling_error) for m in measured if np.isfinite(m.pooling_error)]
    return Table(
        "The whole-receipt score against TruFor's own",
        ["Receipts of one window", "Largest difference of the logits"],
        [[len(errors), f"{max(errors):.1e}"]],
        "For a receipt of one window, the logit of TruFor's last layers for the statistics "
        "pooled from its maps, against the logit TruFor computes itself: the same computation, "
        "in double instead of single precision. The score of such a receipt uses TruFor's own.",
    )


def _interval(estimate: Estimate) -> str:
    return f"{estimate.value:+.2f} ({estimate.low:+.2f} to {estimate.high:+.2f})"


def _passes(estimate: Estimate) -> bool:
    return estimate.low <= 0 <= estimate.high


def size_checks(measured: Sequence[Measured]) -> dict[str, tuple[Estimate, Estimate]]:
    """For each score: the size check on authentic receipts with evidence and on all of them."""
    authentic = [m for m in measured if m.label is Label.AUTHENTIC]
    with_evidence = [m for m in authentic if m.evidence]
    return {
        name: (
            size_dependence(
                [m.scores[name] for m in with_evidence], [m.pixels for m in with_evidence]
            ),
            size_dependence([m.scores[name] for m in authentic], [m.pixels for m in authentic]),
        )
        for name in SCORES
    }


def fixed_order_score(checks: Mapping[str, tuple[Estimate, Estimate]]) -> str | None:
    """The first score of the fixed order whose size check passes on receipts with evidence."""
    return next((name for name in FIXED_ORDER if _passes(checks[name][0])), None)


def chosen_score(checks: Mapping[str, tuple[Estimate, Estimate]]) -> str | None:
    """The score added after the results if its size check passes, else the fixed order's."""
    return AFTER_RESULTS if _passes(checks[AFTER_RESULTS][0]) else fixed_order_score(checks)


def score_table(
    measured: Sequence[Measured], checks: Mapping[str, tuple[Estimate, Estimate]]
) -> Table:
    """Every document score with its average precision and size checks."""
    labels = np.array([m.label is Label.FORGED for m in measured])
    authentic = [m for m in measured if m.label is Label.AUTHENTIC]
    with_evidence = sum(m.evidence for m in authentic)
    rows: list[list[Cell]] = [
        [
            f"{position}. {name}" + (" (added after 1 and 2)" if name == AFTER_RESULTS else ""),
            average_precision(labels, np.array([m.scores[name] for m in measured])),
            _interval(checks[name][0]),
            _interval(checks[name][1]),
            _passes(checks[name][0]),
        ]
        for position, name in enumerate(SCORES, start=1)
    ]
    return Table(
        "Document scores",
        [
            "Document score",
            f"Average precision (chance {labels.mean():.3f})",
            f"Size check, {with_evidence} authentic with evidence",
            f"Size check, {len(authentic)} authentic",
            "Size check passes",
        ],
        rows,
        "Size check: the Spearman rank correlation of document score and pixel count with its "
        "95 % interval (bootstrap over receipts, 10,000 resamples, seed 0), as ADR-0015 "
        "(section 6) fixes it; it passes if the interval includes zero on the receipts with "
        "evidence (ADR-0017). Scores 1 and 2 are the order fixed before the run; score 3 was "
        "added after their results and replaces the outcome of that order if it passes "
        "(record, section 3). No choice depends on average precision. Design fold: optimistic.",
    )


def by_size_table(measured: Sequence[Measured]) -> Table:
    """The median of each score by label and by how the receipt was analysed."""
    rows: list[list[Cell]] = []
    for label in (Label.AUTHENTIC, Label.FORGED):
        for size in SIZES:
            chosen = [m for m in measured if m.label is label and m.size == size]
            if chosen:
                rows.append(
                    [
                        f"{label.value}, {size}",
                        len(chosen),
                        *(float(np.median([m.scores[name] for m in chosen])) for name in SCORES),
                    ]
                )
    return Table(
        "Scores by size",
        ["Receipts", "Count", *(f"Median {name}" for name in SCORES)],
        rows,
        "Descriptive: whether tiling, which only large receipts undergo, moves the scores.",
    )


def localisation_table(measured: Sequence[Measured]) -> Table:
    """How often the top region of the map hits an edit, by how the receipt was analysed."""
    rows: list[list[Cell]] = []
    for size in (*SIZES, "all"):
        chosen = [m for m in measured if m.hit is not None and size in {m.size, "all"}]
        if chosen:
            rows.append(
                [
                    size,
                    len(chosen),
                    sum(bool(m.hit) for m in chosen),
                    float(np.mean([m.baseline for m in chosen if m.baseline is not None])),
                    f"{100 * float(np.median([m.share for m in chosen])):.2f} %",
                ]
            )
    return Table(
        "Top region of the map on forged receipts",
        ["Receipts", "Count", "Top region hits an edit", "Mean random baseline", "Median area"],
        rows,
        "Top region, hit and random baseline as ADR-0015 fixes them: the hit counts a modified "
        "rectangle or a copy source; the baseline is the share of 1,000 boxes of the same size, "
        "centred on OCR text, that hit. Median area: of the top region, as a share of the page. "
        "Design fold: optimistic.",
    )


def receipt_table(measured: Sequence[Measured]) -> Table:
    """Each forged receipt: its top region and every score, ranked among all receipts."""
    ranks = {
        name: {
            m.doc_id: 1 + sum(o.scores[name] > m.scores[name] for o in measured) for m in measured
        }
        for name in SCORES
    }
    rows: list[list[Cell]] = [
        [
            m.doc_id,
            m.size,
            "hit" if m.hit else "miss",
            m.baseline,
            *(f"{m.scores[name]:.3f} ({ranks[name][m.doc_id]})" for name in SCORES),
        ]
        for m in measured
        if m.hit is not None
    ]
    return Table(
        "Per forged receipt",
        ["Receipt", "Analysed", "Top region", "Baseline", *(f"{name} (rank)" for name in SCORES)],
        rows,
        f"Rank 1 is the highest score among all {len(measured)} design-fold receipts.",
    )


def header(model: TruForModel, checks: Mapping[str, tuple[Estimate, Estimate]]) -> str:
    """Title, provenance and the outcome of the rules of the record."""
    commit = code_version()
    state = "with uncommitted changes" if commit.dirty is not False else "clean"
    versions = ", ".join(f"{name} {version(name)}" for name in VERSIONED)
    fixed, chosen = fixed_order_score(checks), chosen_score(checks)
    outcome = (
        f"In the order fixed before the run, the first document score whose size check passes "
        f"is {fixed}. "
        if fixed
        else "In the order fixed before the run, no score passes. "
    ) + (
        f"The score added after those results passes its size check and is chosen: **{chosen}**."
        if chosen == AFTER_RESULTS
        else f"The score added after those results fails its size check, so the outcome of the "
        f"fixed order stands: **{chosen or 'none; the development round stops'}**."
    )
    return (
        "# System 6 development: design fold\n\n"
        "> Generated by `experiments/system6_development.py` (`make trufor`). Do not edit by "
        "hand. Not part of `make reports`: TruFor runs on the Apple GPU.\n\n"
        "The evidence behind the choices for system 6, TruFor "
        "([ADR-0019](../../docs/decisions/0019-system-6-trufor.md)), recorded in "
        "[system-6-development.md](../../docs/system-6-development.md).\n\n"
        f"TruFor is the work of its authors, used under its licence for nonprofit research: "
        f"{CITATION}. Code at commit `{TRUFOR_COMMIT[:7]}` of "
        "[its repository](https://github.com/grip-unina/TruFor), with its published weights.\n\n"
        f"- Model: `{model.identifier}`; {versions}.\n"
        f"- Code of this project: commit `{(commit.commit or 'unknown')[:7]}`, {state}.\n\n"
        f"{outcome}\n\n"
    )


def main() -> None:
    """Analyse the design fold, or read its cached outputs, and write the report."""
    documents = [d for d in load_findit2(DATA) if d.doc_id in DESIGN_FOLD_IDS]
    final = final_marks(marks_plan(documents), *load_rounds(MARK_RECORDS, ASSISTANT, "emre"))
    marks = {doc_id: receipt.marks for doc_id, receipt in final.marks.items()}
    model = TruForModel(TRUFOR)
    start = time.perf_counter()
    measured, mark_checks = measure(
        documents,
        CachedTruFor(model, TRUFOR_CACHE),
        CachedOcr(RapidOcrEngine(), OCR_CACHE),
        marks,
    )
    print(f"measured in {time.perf_counter() - start:.0f} s")  # noqa: T201
    checks = size_checks(measured)
    name = chosen_score(checks) or WHOLE  # the mark tables rank by the chosen score
    by_id = {m.doc_id: m for m in measured}
    authentic_scores = [m.scores[name] for m in measured if m.label is Label.AUTHENTIC]
    ranked = [replace(c, score=by_id[c.doc_id].scores[name]) for c in mark_checks]
    first = min(documents, key=lambda d: d.doc_id)
    tables = [
        run_table(measured),
        repeatability(model, first),
        tile_table(measured),
        cut_table(measured),
        pooling_table(measured),
        score_table(measured, checks),
        by_size_table(measured),
        localisation_table(measured),
        receipt_table(measured),
        *mark_tables(ranked, authentic_scores),
    ]
    body = "\n".join(render_table(t) for t in tables)
    OUTPUT.write_text(header(model, checks) + body, encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
