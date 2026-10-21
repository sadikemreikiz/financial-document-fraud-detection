"""Which measurement of the JPEG trace the classical map of system 1 uses (step 4.3, decision 3).

Writes reports/forensics/jpeg-trace-comparison.md as docs/jpeg-trace-comparison.md fixes it: the
grid agreement of the invisible-traces study and the lattice fit, each compared on the modified
regions of the design fold with the same control boxes, and the choice between them by the rule
fixed in advance. Only the design fold is read; every measurement is computed by code.

Usage: ``uv run python experiments/jpeg_trace_comparison.py``
"""

from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from scipy import stats

from docfraud.data.findit import load_findit2
from docfraud.data.models import BBox, Document, Label
from docfraud.data.normalize import load_grayscale
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS, require_design_fold
from docfraud.evaluation.localisation import BASELINE_SEED, annotated_rectangles, receipt_key
from docfraud.forensics.quantisation import block_fits, box_fit, receipt_lattice
from docfraud.forensics.traces import (
    CONTROLS,
    Comparison,
    Jumps,
    compare,
    control_boxes,
    grid_agreement,
    paper_mask,
    receipt_grid,
    region_rng,
)
from docfraud.markdown import Cell, Table, render_table
from docfraud.reading.ocr import CachedOcr, RapidOcrEngine

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "forensics" / "jpeg-trace-comparison.md"
DATA = ROOT / "data" / "raw" / "findit2"
OCR_CACHE = ROOT / "artifacts" / "ocr"
LARGE_SIDE = 6000
MIN_MEASURED_CONTROLS = 100
CHANCE = 0.05
GRID, LATTICE = "grid agreement", "lattice fit"
MEASUREMENTS = (GRID, LATTICE)
ALIGNMENT_STREAM = 3  # generator stream of the alignment sample; the controls use stream 2


@dataclass(frozen=True, slots=True)
class Presence:
    """Whether one receipt carries each JPEG trace."""

    label: Label
    large: bool
    grid: bool
    lattice: bool


@dataclass(frozen=True, slots=True)
class RegionRow:
    """One modified region and its comparison on each measurement (or why it has none)."""

    key: str
    doc_id: str
    techniques: str
    results: dict[str, Comparison | str]  # measurement -> comparison, or the reason for none


def _compare(value: float | None, controls: Sequence[float | None]) -> Comparison | str:
    if value is None or not np.isfinite(value):
        return "not defined on the region"
    measured = [c for c in controls if c is not None and np.isfinite(c)]
    if len(measured) < MIN_MEASURED_CONTROLS:
        return f"fewer than {MIN_MEASURED_CONTROLS} measured controls"
    return compare(value, measured)


def measure(
    documents: Sequence[Document], ocr: CachedOcr
) -> tuple[list[Presence], list[RegionRow]]:
    """Both measurements on every receipt and region, in identifier order."""
    presence: list[Presence] = []
    regions: list[RegionRow] = []
    for document in sorted(documents, key=lambda d: d.doc_id):
        require_design_fold(document.doc_id)
        page = load_grayscale(document.image_path)
        height, width = page.shape
        whole = BBox(x=0, y=0, width=width, height=height)
        modified, sources = annotated_rectangles(document)
        rectangles = modified + sources
        jumps = Jumps.of(page)
        grid = receipt_grid(jumps, paper_mask(page), rectangles)
        alignment_rng = np.random.default_rng(
            [BASELINE_SEED, receipt_key(document.doc_id), ALIGNMENT_STREAM]
        )
        lattice = receipt_lattice(page, rectangles, alignment_rng)
        label = document.label or Label.AUTHENTIC  # every training receipt has a label
        large = max(height, width) > LARGE_SIDE
        presence.append(Presence(label, large, grid.detectable, lattice.detectable))
        if label is not Label.FORGED or not modified:
            continue
        fits = block_fits(page, lattice)
        words = [token.box for token in ocr.recognize(page).tokens]
        for index, region in enumerate(modified):
            techniques = ", ".join(
                sorted(t.value for t in document.modified_regions[index].techniques)
            )
            key = f"{document.doc_id}-{index}"
            controls = control_boxes(
                region, whole, words, rectangles, region_rng(document.doc_id, index)
            )
            if len(controls) < CONTROLS:
                reason = f"fewer than {CONTROLS} control boxes"
                regions.append(
                    RegionRow(key, document.doc_id, techniques, dict.fromkeys(MEASUREMENTS, reason))
                )
                continue
            results: dict[str, Comparison | str] = {
                GRID: _compare(
                    grid_agreement(jumps, grid, region),
                    [grid_agreement(jumps, grid, c) for c in controls],
                )
                if grid.detectable
                else "no detectable grid on the receipt",
                LATTICE: _compare(
                    box_fit(fits, lattice, region),
                    [box_fit(fits, lattice, c) for c in controls],
                )
                if lattice.detectable
                else "no detectable lattice on the receipt",
            }
            regions.append(RegionRow(key, document.doc_id, techniques, results))
    return presence, regions


def _member(label: Label, large: bool) -> Callable[[Presence], bool]:
    return lambda p: p.label is label and p.large is large


def presence_table(presence: Sequence[Presence]) -> Table:
    """Receipts carrying each trace, by label and size."""
    groups: list[tuple[str, Callable[[Presence], bool]]] = []
    for label in (Label.AUTHENTIC, Label.FORGED):
        for size, large in (("longer side above 6000 px", True), ("smaller", False)):
            groups.append((f"{label.value}, {size}", _member(label, large)))
    groups.append(("all", lambda _: True))
    rows: list[list[Cell]] = []
    for name, member in groups:
        chosen = [p for p in presence if member(p)]
        if not chosen:
            continue
        rows.append(
            [
                name,
                len(chosen),
                sum(p.grid for p in chosen),
                sum(p.lattice for p in chosen),
                sum(p.grid and p.lattice for p in chosen),
                sum(not p.grid and not p.lattice for p in chosen),
            ]
        )
    return Table(
        "JPEG traces of the receipts",
        ["Receipts", "Count", "Detectable grid", "Detectable lattice", "Both", "Neither"],
        rows,
        "Grid: as in the invisible-traces study. Lattice: one of the 64 block alignments fits "
        "the quantisation lattice at least 0.25 better than every other. Both are estimated "
        "outside the annotated rectangles.",
    )


def _comparisons(chosen: Sequence[RegionRow], measurement: str) -> list[Comparison]:
    return [c for r in chosen if isinstance(c := r.results[measurement], Comparison)]


def excess(regions: Sequence[RegionRow], measurement: str) -> float:
    """Regions standing out beyond the 5 % expected by chance among those compared."""
    comparisons = _comparisons(regions, measurement)
    return sum(c.stands_out is not None for c in comparisons) - CHANCE * len(comparisons)


def summary_table(regions: Sequence[RegionRow]) -> Table:
    """Regions standing out on each measurement, against chance."""
    rows: list[list[Cell]] = []
    for measurement in MEASUREMENTS:
        comparisons = _comparisons(regions, measurement)
        reasons = Counter(c for r in regions if isinstance(c := r.results[measurement], str))
        sides = Counter(c.stands_out for c in comparisons)
        out, measured = sides["below"] + sides["above"], len(comparisons)
        compared = [r for r in regions if isinstance(r.results[measurement], Comparison)]
        receipts = {r.doc_id for r in compared}
        receipts_out = {
            r.doc_id
            for r in compared
            if isinstance(c := r.results[measurement], Comparison) and c.stands_out
        }
        p = (
            float(stats.binomtest(out, measured, CHANCE, alternative="greater").pvalue)
            if measured
            else None
        )
        rows.append(
            [
                measurement,
                len(regions),
                measured,
                sides["below"],
                sides["above"],
                sum(c.uninformative for c in comparisons),
                f"{CHANCE * measured:.2f}",
                f"{excess(regions, measurement):.2f}",
                p,
                f"{len(receipts_out)} of {len(receipts)}",
                "; ".join(f"{reason}: {n}" for reason, n in sorted(reasons.items())) or None,
            ]
        )
    return Table(
        "Modified regions against their controls",
        [
            "Measurement",
            "Regions",
            "Compared",
            "Below the controls",
            "Above the controls",
            "Uninformative",
            "Expected by chance",
            "Beyond chance",
            "P(at least as many by chance)",
            "Receipts with a region standing out",
            "Not compared",
        ],
        rows,
        "A region stands out when its value lies below the 2.5th or above the 97.5th percentile "
        "of its 200 controls, which happens to 5 % of regions by chance. Beyond chance: regions "
        "standing out minus those expected by chance. The probability is a one-sided exact "
        "binomial test; regions of one receipt are not independent, so it is a guide, not a "
        "test. Uninformative: the region and every control share one value.",
    )


def technique_table(regions: Sequence[RegionRow]) -> Table:
    """The same counts by technique label (descriptive)."""
    techniques = sorted({t for r in regions for t in r.techniques.split(", ") if t})
    rows: list[list[Cell]] = []
    for technique in techniques:
        chosen = [r for r in regions if technique in r.techniques.split(", ")]
        row: list[Cell] = [technique, len(chosen)]
        for measurement in MEASUREMENTS:
            comparisons = _comparisons(chosen, measurement)
            sides = Counter(c.stands_out for c in comparisons)
            row.append(f"{sides['below']} below, {sides['above']} above, of {len(comparisons)}")
        rows.append(row)
    return Table(
        "By technique label",
        ["Technique", "Regions", *MEASUREMENTS],
        rows,
        "A region with several techniques counts under each. Descriptive: the groups are small.",
    )


def region_table(regions: Sequence[RegionRow]) -> Table:
    """Each region's percentile among its controls."""

    def cell(result: Comparison | str) -> str:
        if isinstance(result, str):
            return "\N{EM DASH}"
        if result.uninformative:
            return "all equal"
        side = f" ({result.stands_out})" if result.stands_out else ""
        return f"{result.percentile:.1f}{side}"

    rows: list[list[Cell]] = [
        [r.key, r.techniques, *(cell(r.results[m]) for m in MEASUREMENTS)] for r in regions
    ]
    return Table(
        "Per region",
        ["Region", "Techniques", *(f"{m}, percentile" for m in MEASUREMENTS)],
        rows,
        "Percentile: the share of the region's controls with a lower value, ties counting half. "
        "\N{EM DASH}: not compared (see the summary).",
    )


def choice(regions: Sequence[RegionRow]) -> str:
    """The measurement system 1 uses, by the rule of the plan."""
    grid, lattice = excess(regions, GRID), excess(regions, LATTICE)
    chosen = LATTICE if lattice > grid else GRID
    return (
        f"## Choice\n\nBeyond chance: {lattice:.2f} regions for the lattice fit, {grid:.2f} for "
        f"the grid agreement. By the rule fixed in the plan (the lattice fit only if it is "
        f"larger; otherwise, also on a tie, the grid agreement), system 1 uses the "
        f"**{chosen}**.\n"
    )


def main() -> None:
    """Measure the design fold and write the report."""
    documents = [d for d in load_findit2(DATA) if d.doc_id in DESIGN_FOLD_IDS]
    presence, regions = measure(documents, CachedOcr(RapidOcrEngine(), OCR_CACHE))
    tables = [
        presence_table(presence),
        summary_table(regions),
        technique_table(regions),
        region_table(regions),
    ]
    header = (
        "# JPEG trace comparison: design fold\n\n"
        "> Generated by `experiments/jpeg_trace_comparison.py`. Do not edit by hand.\n\n"
        "Which measurement of the JPEG trace the classical map of system 1 uses (step 4.3, "
        "decision 3), as the [study plan](../../docs/jpeg-trace-comparison.md) fixed before "
        "the lattice fit was computed on any forged receipt.\n\n"
    )
    body = "\n".join(render_table(t) for t in tables)
    OUTPUT.write_text(header + choice(regions) + "\n" + body, encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
