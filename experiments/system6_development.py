"""Development round of system 6, TruFor (ADR-0019), on the design fold.

Writes reports/forensics/system-6-development.md, the evidence behind the choices recorded in
docs/system-6-development.md: how the receipts were tiled, which document score passes the size
check of ADR-0015 and ADR-0017 in the order fixed before the run, how often the top region of
TruFor's map hits an edit against its random baseline, and whether it falls on a benign mark.
Only the design fold is read; every measurement is computed by code.

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
VERSIONED = ("torch", "torchvision", "timm", "yacs", "numpy")

INTEGRITY = "TruFor's integrity score, highest over tiles"
CONTRAST = "highest map value minus the receipt's 99th percentile"
SCORES: dict[str, Callable[[TruForOutput], float]] = {
    INTEGRITY: lambda o: o.integrity_score(),
    CONTRAST: lambda o: float(o.localisation.max() - np.percentile(o.localisation, 99)),
}
"""The document scores of system 6, in the order fixed before the run (ADR-0019): the first
whose size check passes is chosen; if none passes, the development round stops."""


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
                    score=scores[INTEGRITY],  # replaced by the chosen score once it is known
                )
            )
        print(f"{count}/{len(ordered)} {document.doc_id}: {len(output.windows)} windows")  # noqa: T201
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


def chosen_score(checks: Mapping[str, tuple[Estimate, Estimate]]) -> str | None:
    """The first score in the fixed order whose size check passes on receipts with evidence."""
    return next((name for name in SCORES if _passes(checks[name][0])), None)


def score_table(
    measured: Sequence[Measured], checks: Mapping[str, tuple[Estimate, Estimate]]
) -> Table:
    """Both document scores with their average precision and size checks."""
    labels = np.array([m.label is Label.FORGED for m in measured])
    authentic = [m for m in measured if m.label is Label.AUTHENTIC]
    with_evidence = sum(m.evidence for m in authentic)
    rows: list[list[Cell]] = [
        [
            f"{position}. {name}",
            average_precision(labels, np.array([m.scores[name] for m in measured])),
            _interval(checks[name][0]),
            _interval(checks[name][1]),
            _passes(checks[name][0]),
        ]
        for position, name in enumerate(SCORES, start=1)
    ]
    return Table(
        "Document scores, in the order fixed before the run",
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
        "evidence (ADR-0017). The first score that passes is chosen, whatever its average "
        "precision. Design fold: optimistic.",
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
    """Each forged receipt: its top region and both scores, ranked among all receipts."""
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


def header(model: TruForModel, chosen: str | None) -> str:
    """Title, provenance and the outcome of the fixed order."""
    commit = code_version()
    state = "with uncommitted changes" if commit.dirty is not False else "clean"
    versions = ", ".join(f"{name} {version(name)}" for name in VERSIONED)
    outcome = (
        f"In the order fixed before the run, the first document score whose size check passes "
        f"is **{chosen}**."
        if chosen
        else "**No document score passes the size check**: the development round stops here, "
        "as fixed before the run, and the next step is discussed."
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
    chosen = chosen_score(checks)
    name = chosen or INTEGRITY  # the mark tables rank by the chosen score
    by_id = {m.doc_id: m for m in measured}
    authentic_scores = [m.scores[name] for m in measured if m.label is Label.AUTHENTIC]
    ranked = [replace(c, score=by_id[c.doc_id].scores[name]) for c in mark_checks]
    first = min(documents, key=lambda d: d.doc_id)
    tables = [
        run_table(measured),
        repeatability(model, first),
        score_table(measured, checks),
        by_size_table(measured),
        localisation_table(measured),
        receipt_table(measured),
        *mark_tables(ranked, authentic_scores),
    ]
    body = "\n".join(render_table(t) for t in tables)
    OUTPUT.write_text(header(model, chosen) + body, encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
