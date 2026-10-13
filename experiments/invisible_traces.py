"""Invisible traces of the real edits of the design fold (step 4.3, decision 2).

Writes reports/forensics/invisible-traces.md as docs/invisible-traces-study.md fixes it: whether
the receipts of the design fold carry a JPEG grid, and whether the noise level and the grid
agreement of each modified region stand out against control boxes of the same size on the
receipt's unedited text. Only the design fold is read; every measurement is computed by code.

Usage: ``uv run python experiments/invisible_traces.py``
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
from docfraud.evaluation.localisation import annotated_rectangles
from docfraud.forensics.traces import (
    CONTROLS,
    Comparison,
    Jumps,
    compare,
    control_boxes,
    grid_agreement,
    noise_level,
    paper_mask,
    receipt_grid,
    region_rng,
    residual,
)
from docfraud.markdown import Cell, Table, render_table
from docfraud.reading.ocr import CachedOcr, RapidOcrEngine

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "forensics" / "invisible-traces.md"
DATA = ROOT / "data" / "raw" / "findit2"
OCR_CACHE = ROOT / "artifacts" / "ocr"
LARGE_SIDE = 6000
MIN_MEASURED_CONTROLS = 100
CHANCE = 0.05
MEASUREMENTS = ("noise level", "grid agreement")


@dataclass(frozen=True, slots=True)
class GridRow:
    """The grid of one receipt."""

    label: Label
    large: bool
    column_ratio: float
    row_ratio: float
    detectable: bool


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


def measure(documents: Sequence[Document], ocr: CachedOcr) -> tuple[list[GridRow], list[RegionRow]]:
    """Every measurement of the study, receipt by receipt in identifier order."""
    grids: list[GridRow] = []
    regions: list[RegionRow] = []
    for document in sorted(documents, key=lambda d: d.doc_id):
        require_design_fold(document.doc_id)
        page = load_grayscale(document.image_path)
        height, width = page.shape
        whole = BBox(x=0, y=0, width=width, height=height)
        modified, sources = annotated_rectangles(document)
        rectangles = modified + sources
        paper, jumps = paper_mask(page), Jumps.of(page)
        grid = receipt_grid(jumps, paper, rectangles)
        label = document.label or Label.AUTHENTIC  # every training receipt has a label
        grids.append(
            GridRow(
                label,
                max(height, width) > LARGE_SIDE,
                grid.columns.ratio,
                grid.rows.ratio,
                grid.detectable,
            )
        )
        if label is not Label.FORGED or not modified:
            continue
        noise = residual(page)
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
                "noise level": _compare(
                    noise_level(noise, paper, region),
                    [noise_level(noise, paper, c) for c in controls],
                )
            }
            if grid.detectable:
                results["grid agreement"] = _compare(
                    grid_agreement(jumps, grid, region),
                    [grid_agreement(jumps, grid, c) for c in controls],
                )
            else:
                results["grid agreement"] = "no detectable grid on the receipt"
            regions.append(RegionRow(key, document.doc_id, techniques, results))
    return grids, regions


def _member(label: Label, large: bool) -> Callable[[GridRow], bool]:
    return lambda g: g.label is label and g.large is large


def grid_table(grids: Sequence[GridRow]) -> Table:
    """Receipts with a detectable grid, by label and size."""
    rows: list[list[Cell]] = []
    groups: list[tuple[str, Callable[[GridRow], bool]]] = []
    for label in (Label.AUTHENTIC, Label.FORGED):
        for size, large in (("longer side above 6000 px", True), ("smaller", False)):
            groups.append((f"{label.value}, {size}", _member(label, large)))
    groups.append(("all", lambda _: True))
    for name, member in groups:
        chosen = [g for g in grids if member(g)]
        if not chosen:
            continue
        rows.append(
            [
                name,
                len(chosen),
                sum(g.detectable for g in chosen),
                float(np.median([g.column_ratio for g in chosen])),
                float(np.median([g.row_ratio for g in chosen])),
            ]
        )
    return Table(
        "JPEG grid of the receipts",
        [
            "Receipts",
            "Count",
            "With a detectable grid",
            "Median ratio, columns",
            "Median ratio, rows",
        ],
        rows,
        "Ratio: the contrast of the excess-jump profile at period 8 over the larger contrast at "
        "periods 7 and 9, measured on paper outside the annotated rectangles. A grid is "
        "detectable when the ratio is at least 2 in both directions.",
    )


def _counts(chosen: Sequence[RegionRow], measurement: str) -> tuple[list[Comparison], Counter[str]]:
    compared = [r.results[measurement] for r in chosen]
    comparisons = [c for c in compared if isinstance(c, Comparison)]
    reasons = Counter(c for c in compared if isinstance(c, str))
    return comparisons, reasons


def summary_table(regions: Sequence[RegionRow]) -> Table:
    """Regions standing out on each measurement, against chance."""
    rows: list[list[Cell]] = []
    for measurement in MEASUREMENTS:
        comparisons, reasons = _counts(regions, measurement)
        sides = Counter(c.stands_out for c in comparisons)
        out = sides["below"] + sides["above"]
        measured = len(comparisons)
        compared = [(r.doc_id, r.results[measurement]) for r in regions]
        receipts = {d for d, c in compared if isinstance(c, Comparison)}
        receipts_out = {d for d, c in compared if isinstance(c, Comparison) and c.stands_out}
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
                f"{CHANCE * measured:.1f}",
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
            "Expected by chance",
            "P(at least as many by chance)",
            "Receipts with a region standing out",
            "Not compared",
        ],
        rows,
        "A region stands out when its value lies below the 2.5th or above the 97.5th percentile "
        "of its 200 controls, which happens to 5 % of regions by chance. The probability is a "
        "one-sided exact binomial test; regions of one receipt are not independent, so it is a "
        "guide, not a test.",
    )


def technique_table(regions: Sequence[RegionRow]) -> Table:
    """The same counts by technique label (descriptive)."""
    techniques = sorted({t for r in regions for t in r.techniques.split(", ") if t})
    rows: list[list[Cell]] = []
    for technique in techniques:
        chosen = [r for r in regions if technique in r.techniques.split(", ")]
        row: list[Cell] = [technique, len(chosen)]
        for measurement in MEASUREMENTS:
            comparisons, _ = _counts(chosen, measurement)
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


def main() -> None:
    """Measure the design fold and write the report."""
    documents = [d for d in load_findit2(DATA) if d.doc_id in DESIGN_FOLD_IDS]
    grids, regions = measure(documents, CachedOcr(RapidOcrEngine(), OCR_CACHE))
    tables = [
        grid_table(grids),
        summary_table(regions),
        technique_table(regions),
        region_table(regions),
    ]
    header = (
        "# Invisible traces: design fold\n\n"
        "> Generated by `experiments/invisible_traces.py`. Do not edit by hand.\n\n"
        "Whether the real edits of the design fold leave traces that code can measure (step 4.3, "
        "decision 2), as the [study plan](../../docs/invisible-traces-study.md) fixed before "
        "any measurement was computed.\n\n"
    )
    OUTPUT.write_text(header + "\n".join(render_table(t) for t in tables), encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
