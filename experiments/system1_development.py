"""Development evidence for system 1, the JPEG-lattice map (ADR-0016), on the design fold.

Writes reports/forensics/system-1-development.md, the evidence behind the choices recorded in
docs/system-1-development.md: whether system 1, which sees no annotation, finds the lattice that
the comparison study estimated outside the annotated rectangles, how often the top region of two
map variants hits an edit (ADR-0015) against its random baseline, the document scores tried, and
whether the top region of the frozen map falls on a benign mark. Only the design fold is read;
every measurement is computed by code.

Usage: ``uv run python experiments/system1_development.py``
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
from scipy import ndimage

from docfraud.data.benign_marks import Mark, final_marks, load_rounds, marks_plan
from docfraud.data.findit import load_findit2
from docfraud.data.models import BBox, Document, Label
from docfraud.data.normalize import load_grayscale
from docfraud.data.observation import ASSISTANT
from docfraud.evaluation.bootstrap import Estimate as Interval
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS, require_design_fold
from docfraud.evaluation.detection import average_precision
from docfraud.evaluation.localisation import (
    BASELINE_SEED,
    annotated_rectangles,
    baseline_hit_rate,
    hits,
    receipt_key,
    receipt_rng,
    size_dependence,
    top_region,
)
from docfraud.evaluation.mark_alarms import MarkCheck, check_marks, mark_tables
from docfraud.forensics.lattice_map import LatticeEvidence, lattice_evidence
from docfraud.forensics.quantisation import MIN_BLOCKS, block_fits, receipt_lattice
from docfraud.markdown import Cell, Table, render_table
from docfraud.reading.ocr import CachedOcr, RapidOcrEngine

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "forensics" / "system-1-development.md"
DATA = ROOT / "data" / "raw" / "findit2"
OCR_CACHE = ROOT / "artifacts" / "ocr"
MARK_RECORDS = ROOT / "annotations" / "findit2" / "benign-marks"
ALIGNMENT_STREAM = 3  # as in experiments/jpeg_trace_comparison.py
LARGE_SIDE = 6000
CHOSEN, SMOOTHED = "per block (chosen)", "mean over 3 x 3 blocks"
VARIANTS = (CHOSEN, SMOOTHED)


@dataclass(frozen=True, slots=True)
class Estimate:
    """The lattice of one receipt as the study and as system 1 estimate it."""

    label: Label
    study: bool  # detectable when estimated outside the annotated rectangles
    system: bool  # detectable when estimated from every block
    same_offset: bool
    step_agreement: float  # share of coefficients used by either with equal steps; NaN if none


@dataclass(frozen=True, slots=True)
class Localisation:
    """The top region of one map variant on one forged receipt."""

    doc_id: str
    hit: bool
    baseline: float
    share: float  # area of the top region over the area of the page


def smoothed(evidence: LatticeEvidence, page: npt.NDArray[np.uint8]) -> LatticeEvidence:
    """The compared variant: each block gets the mean fit of the measurable blocks around it.

    The mean runs over the 3 x 3 blocks centred on the block and needs ``MIN_BLOCKS`` of them
    to be measurable; otherwise the block has no evidence.
    """
    if not evidence.lattice.detectable:
        return evidence
    fits = block_fits(page, evidence.lattice)
    known = np.isfinite(fits)
    totals = ndimage.convolve(np.where(known, fits, 0.0), np.ones((3, 3)), mode="constant")
    counts = ndimage.convolve(known.astype(float), np.ones((3, 3)), mode="constant")
    with np.errstate(invalid="ignore", divide="ignore"):
        means = totals / counts
    measured = counts >= MIN_BLOCKS
    suspicion = np.where(measured, 1 - means, 0.0)
    return LatticeEvidence(evidence.lattice, suspicion, measured, evidence.shape)


def _highest(evidence: LatticeEvidence) -> float:
    values = evidence.suspicion[evidence.measured]
    return float(values.max()) if values.size else 0.0


def _minus_median(evidence: LatticeEvidence) -> float:
    values = evidence.suspicion[evidence.measured]
    return float(values.max() - np.median(values)) if values.size else 0.0


def _ratio(evidence: LatticeEvidence) -> float:
    values = evidence.suspicion[evidence.measured]
    if not values.size:
        return 0.0
    median, top = np.median(values), np.percentile(values, 99)
    return float((values.max() - median) / (top - median)) if top > median else 0.0


CHOSEN_SCORE = "highest block minus the receipt's 99th percentile (chosen)"
COVERAGE_ONLY = "1 with evidence, 0 without (no information on edits)"
SCORES: dict[str, Callable[[LatticeEvidence, npt.NDArray[np.uint8]], float]] = {
    "highest block suspicion": lambda e, _: _highest(e),
    "highest mean over 3 x 3 blocks": lambda e, page: _highest(smoothed(e, page)),
    "highest block minus the receipt's median": lambda e, _: _minus_median(e),
    CHOSEN_SCORE: lambda e, _: e.document_score(),
    "(highest - median) / (99th percentile - median)": lambda e, _: _ratio(e),
    COVERAGE_ONLY: lambda e, _: float(e.measured.any()),
}
"""Every document score tried for system 1, in the order they were considered."""


@dataclass(frozen=True, slots=True)
class Scored:
    """The candidate document scores of one receipt."""

    label: Label
    large: bool
    pixels: int
    evidence: bool  # whether the map of system 1 is not constant
    scores: dict[str, float]


def measure(
    documents: Sequence[Document], ocr: CachedOcr, marks: Mapping[str, Sequence[Mark]]
) -> tuple[list[Estimate], dict[str, list[Localisation]], list[Scored], list[MarkCheck]]:
    """Every check on every receipt, in identifier order; ``marks`` holds the flagged ones."""
    estimates: list[Estimate] = []
    localisations: dict[str, list[Localisation]] = {v: [] for v in VARIANTS}
    scored: list[Scored] = []
    mark_checks: list[MarkCheck] = []
    for document in sorted(documents, key=lambda d: d.doc_id):
        require_design_fold(document.doc_id)
        page = load_grayscale(document.image_path)
        modified, sources = annotated_rectangles(document)
        rectangles = modified + sources
        study = receipt_lattice(
            page,
            rectangles,
            np.random.default_rng([BASELINE_SEED, receipt_key(document.doc_id), ALIGNMENT_STREAM]),
        )
        evidence = lattice_evidence(page)
        system = evidence.lattice
        used = (study.steps > 0) | (system.steps > 0)
        agreement = (
            float((study.steps[used] == system.steps[used]).mean()) if used.any() else np.nan
        )
        label = document.label or Label.AUTHENTIC  # every training receipt has a label
        estimates.append(
            Estimate(
                label,
                study.detectable,
                system.detectable,
                study.offset == system.offset,
                agreement,
            )
        )
        height, width = page.shape
        scored.append(
            Scored(
                label,
                max(height, width) > LARGE_SIDE,
                height * width,
                bool(evidence.measured.any()),
                {name: score(evidence, page) for name, score in SCORES.items()},
            )
        )
        whole = BBox(x=0, y=0, width=width, height=height)
        if document.doc_id in marks:
            mark_checks.append(
                check_marks(
                    document.doc_id,
                    label,
                    top_region(evidence.pixel_map()),
                    whole,
                    [token.box for token in ocr.recognize(page).tokens],
                    marks[document.doc_id],
                    evidence=bool(evidence.measured.any()),
                    score=evidence.document_score(),
                )
            )
        if label is not Label.FORGED or not modified or not system.detectable:
            continue
        words = [token.box for token in ocr.recognize(page).tokens]
        for variant, made in ((CHOSEN, evidence), (SMOOTHED, smoothed(evidence, page))):
            region = top_region(made.pixel_map())
            baseline = baseline_hit_rate(
                region, whole, words, rectangles, receipt_rng(document.doc_id)
            )
            localisations[variant].append(
                Localisation(
                    document.doc_id,
                    hits(region, rectangles),
                    baseline,
                    region.width * region.height / (width * height),
                )
            )
    return estimates, localisations, scored, mark_checks


def estimate_table(estimates: Sequence[Estimate]) -> Table:
    """Whether system 1 finds the lattice of the comparison study."""
    groups: list[tuple[str, Callable[[Estimate], bool]]] = [
        ("authentic", lambda e: e.label is Label.AUTHENTIC),
        ("forged", lambda e: e.label is Label.FORGED),
        ("all", lambda _: True),
    ]
    rows: list[list[Cell]] = []
    for name, member in groups:
        chosen = [e for e in estimates if member(e)]
        both = [e for e in chosen if e.study and e.system]
        agreements = [e.step_agreement for e in both if np.isfinite(e.step_agreement)]
        rows.append(
            [
                name,
                len(chosen),
                sum(e.study for e in chosen),
                sum(e.system for e in chosen),
                f"{sum(e.same_offset for e in both)} of {len(both)}",
                min(agreements) if agreements else None,
            ]
        )
    return Table(
        "The lattice without annotations",
        [
            "Receipts",
            "Count",
            "Lattice in the study",
            "Lattice in system 1",
            "Same alignment where both",
            "Lowest step agreement where both",
        ],
        rows,
        "Study: estimated outside the annotated rectangles on 4,000 sampled block positions, as "
        "in the JPEG trace comparison. System 1: alignments scored on every block and nothing "
        "left out, as it must on a receipt it has not seen. Step agreement: the share of "
        "coefficients used by either estimate whose steps are equal.",
    )


def localisation_table(localisations: dict[str, list[Localisation]]) -> Table:
    """How often the top region of each variant hits an edit."""
    rows: list[list[Cell]] = []
    for variant in VARIANTS:
        chosen = localisations[variant]
        rows.append(
            [
                variant,
                len(chosen),
                sum(r.hit for r in chosen),
                float(np.mean([r.baseline for r in chosen])) if chosen else None,
                f"{100 * float(np.median([r.share for r in chosen])):.2f} %" if chosen else None,
            ]
        )
    return Table(
        "Top region of the map on forged receipts with a lattice",
        ["Map", "Receipts", "Top region hits an edit", "Mean random baseline", "Median area share"],
        rows,
        "Top region, hit and random baseline as ADR-0015 fixes them: the hit counts a modified "
        "rectangle or a copy source; the baseline is the share of 1,000 boxes of the same size, "
        "centred on OCR text, that hit. Receipts without a lattice have a map without evidence, "
        "whose top region is the whole page; they are left out here. Design fold: optimistic.",
    )


def receipt_table(localisations: dict[str, list[Localisation]]) -> Table:
    """Hit and baseline of each variant per receipt."""
    by_receipt = {v: {r.doc_id: r for r in localisations[v]} for v in VARIANTS}
    rows: list[list[Cell]] = [
        [
            doc_id,
            *(
                f"{'hit' if by_receipt[v][doc_id].hit else 'miss'}, "
                f"{by_receipt[v][doc_id].baseline:.3f}"
                for v in VARIANTS
            ),
        ]
        for doc_id in sorted(by_receipt[CHOSEN])
    ]
    return Table(
        "Per receipt",
        ["Receipt", *(f"{v}: top region, baseline" for v in VARIANTS)],
        rows,
    )


def evidence_table(scored: Sequence[Scored]) -> Table:
    """Receipts on which system 1 has evidence, by label and size."""
    rows: list[list[Cell]] = []
    for label in (Label.AUTHENTIC, Label.FORGED):
        for size, large in (("longer side above 6000 px", True), ("smaller", False)):
            chosen = [s for s in scored if s.label is label and s.large is large]
            rows.append([f"{label.value}, {size}", len(chosen), sum(s.evidence for s in chosen)])
    rows.append(["all", len(scored), sum(s.evidence for s in scored)])
    return Table(
        "Receipts with evidence",
        ["Receipts", "Count", "With evidence"],
        rows,
        "Evidence: the map of system 1 is not constant, that is, the receipt has a detectable "
        "lattice and at least one measurable block (ADR-0017).",
    )


def _interval(estimate: Interval) -> str:
    return f"{estimate.value:+.2f} ({estimate.low:+.2f} to {estimate.high:+.2f})"


def score_table(scored: Sequence[Scored]) -> Table:
    """Every document score tried, with its average precision and both size checks."""
    labels = np.array([s.label is Label.FORGED for s in scored])
    authentic = [s for s in scored if s.label is Label.AUTHENTIC]
    with_evidence = [s for s in authentic if s.evidence]
    rows: list[list[Cell]] = []
    for name in SCORES:
        values = np.array([s.scores[name] for s in scored])
        everywhere = size_dependence(
            [s.scores[name] for s in authentic], [s.pixels for s in authentic]
        )
        rows.append(
            [
                name,
                average_precision(labels, values),
                _interval(everywhere),
                _interval(
                    size_dependence(
                        [s.scores[name] for s in with_evidence], [s.pixels for s in with_evidence]
                    )
                )
                if name != COVERAGE_ONLY
                else None,
            ]
        )
    return Table(
        "Document scores tried",
        [
            "Document score",
            f"Average precision (chance {labels.mean():.3f})",
            f"Size check, {len(authentic)} authentic",
            f"Size check, {len(with_evidence)} authentic with evidence",
        ],
        rows,
        "Size check: the Spearman rank correlation of document score and pixel count with its 95 % "
        "interval (bootstrap over receipts, 10,000 resamples, seed 0), as ADR-0015 (section 6) "
        "fixes it; ADR-0017 runs it on the receipts with evidence. Receipts without evidence "
        "score 0. The last row carries no information on edits and is constant on the receipts "
        "with evidence. Design fold: optimistic, and every row was tried on the same 19 forged "
        "receipts.",
    )


def main() -> None:
    """Measure the design fold and write the report."""
    documents = [d for d in load_findit2(DATA) if d.doc_id in DESIGN_FOLD_IDS]
    final = final_marks(marks_plan(documents), *load_rounds(MARK_RECORDS, ASSISTANT, "emre"))
    marks = {doc_id: receipt.marks for doc_id, receipt in final.marks.items()}
    estimates, localisations, scored, mark_checks = measure(
        documents, CachedOcr(RapidOcrEngine(), OCR_CACHE), marks
    )
    authentic_scores = [s.scores[CHOSEN_SCORE] for s in scored if s.label is Label.AUTHENTIC]
    tables = [
        estimate_table(estimates),
        localisation_table(localisations),
        receipt_table(localisations),
        evidence_table(scored),
        score_table(scored),
        *mark_tables(mark_checks, authentic_scores),
    ]
    header = (
        "# System 1 development: design fold\n\n"
        "> Generated by `experiments/system1_development.py`. Do not edit by hand.\n\n"
        "The evidence behind the choices for system 1, the JPEG-lattice map of "
        "[ADR-0016](../../docs/decisions/0016-system-1-jpeg-trace.md), recorded in "
        "[system-1-development.md](../../docs/system-1-development.md).\n\n"
    )
    OUTPUT.write_text(header + "\n".join(render_table(t) for t in tables), encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
