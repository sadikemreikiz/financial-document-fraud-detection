"""The single measurement of copy-move on the clean folds (ADR-0029 section 4, ADR-0030).

Copy-move is frozen and has no fitted parameters, so every receipt of the four clean folds is
scored directly. The measurement follows the rules of Phase 4 as ADR-0030 adjusts them for a
pair of regions:

- **Detection.** Threshold-free metrics of the document score over the clean folds, with
  stratified bootstrap intervals: average precision (chance: the share of forged receipts),
  ROC-AUC and recall at the review budgets. F1 needs a threshold chosen on the validation
  split, which ADR-0029 (section 1) keeps for one check of the whole frozen system, so it is not
  reported here.
- **Localisation guard.** On every forged receipt with region annotations, the pair hits if
  either region shares a pixel with a modified rectangle or a copy source; its baseline places
  one random box per region (:func:`~docfraud.evaluation.localisation.any_box_baseline`). A
  receipt without a pair has no region: it counts as a miss with a baseline of 0, as a report
  without a located finding counts as a miss (ADR-0029, section 12). The guard is also given on
  the receipts with a pair, without large rectangles and without earlier contact.
- **Copy sources.** On the forged receipts with an annotated copy source: copy-source
  localisation (either region overlaps a source) and the pair hit rate (one region overlaps a
  modified rectangle and the other a source). The annotations do not say which source belongs
  to which copy, so the pair hit rate pairs any modified rectangle with any source of the same
  receipt.

Only aggregates enter the run record and the report; the scores of single clean-fold receipts
are kept out of both (ADR-0014, decision 1).
"""

from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np
import numpy.typing as npt

from docfraud.data.models import BBox, Document, Label, Technique
from docfraud.detectors.copy_move import (
    BLOCK_DIVISOR,
    INK_CONTRAST,
    MAX_GROUP,
    MIN_BLOCK,
    UNSATURATED_SHARE,
    CopyMovePair,
    copy_move_evidence,
)
from docfraud.evaluation.bootstrap import Estimate, ratio, stratified
from docfraud.evaluation.design_fold import EARLIER_CONTACTS
from docfraud.evaluation.detection import ranking_metrics
from docfraud.evaluation.localisation import (
    BASELINE_DRAWS,
    Guard,
    annotated_rectangles,
    any_box_baseline,
    has_large_rectangle,
    localisation_guard,
    overlapping,
    random_boxes,
    receipt_rng,
)
from docfraud.markdown import Cell, Table, render_table

RUN_NAME = "copy-move-clean"
CONFIG: dict[str, Any] = {
    "detector": "docfraud.detectors.copy_move",
    "block_divisor": BLOCK_DIVISOR,
    "min_block": MIN_BLOCK,
    "ink_contrast": INK_CONTRAST,
    "unsaturated_share": UNSATURATED_SHARE,
    "max_group": MAX_GROUP,
}
NAMED_TECHNIQUES = (Technique.CPI, Technique.CPO, Technique.IMI, Technique.CUT, Technique.PIX)
METRIC_NAMES = {
    "average_precision": "Average precision",
    "roc_auc": "ROC-AUC",
    "recall_at_10pct_review": "Recall at 10 % review",
    "recall_at_20pct_review": "Recall at 20 % review",
}

type Page = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Measured:
    """Copy-move on one clean-fold receipt, with what it is judged against."""

    forged: bool
    score: float
    techniques: frozenset[Technique]  # of the modified regions
    annotated: bool  # forged, with region annotations
    sourced: bool  # with an annotated copy source
    large: bool  # a modified rectangle covers more than 5 % of the page (ADR-0015, section 4)
    earlier: bool  # viewed before ADR-0014 (decision 1)
    paired: bool
    hit: bool  # either region on a modified rectangle or a copy source
    baseline: float
    source_hit: bool  # either region on a copy source
    source_baseline: float
    pair_hit: bool  # one region on a modified rectangle, the other on a copy source
    pair_baseline: float


def matches_both(pair: CopyMovePair, modified: Sequence[BBox], sources: Sequence[BBox]) -> bool:
    """Whether one region of the pair overlaps a modified rectangle and the other a source."""

    def on(region: BBox, boxes: Sequence[BBox]) -> bool:
        return any(region.overlaps(b) for b in boxes)

    first, second = pair.first, pair.second
    return (on(first, modified) and on(second, sources)) or (
        on(first, sources) and on(second, modified)
    )


def both_baseline(  # noqa: PLR0913 - the pair, its page and what it is judged against
    pair: CopyMovePair,
    page: BBox,
    word_boxes: Sequence[BBox],
    modified: Sequence[BBox],
    sources: Sequence[BBox],
    *,
    rng: np.random.Generator,
) -> float:
    """The chance of :func:`matches_both` for two independent random boxes of the pair's sizes."""
    x1, y1 = random_boxes(pair.first, page, word_boxes, rng, BASELINE_DRAWS)
    x2, y2 = random_boxes(pair.second, page, word_boxes, rng, BASELINE_DRAWS)
    first_modified = overlapping(x1, y1, pair.first, modified)
    first_source = overlapping(x1, y1, pair.first, sources)
    second_modified = overlapping(x2, y2, pair.second, modified)
    second_source = overlapping(x2, y2, pair.second, sources)
    both = (first_modified & second_source) | (first_source & second_modified)
    return float(both.mean())


def measure(document: Document, page: Page, word_boxes: Sequence[BBox]) -> Measured:
    """Run the frozen detector on one receipt and judge its pair."""
    evidence = copy_move_evidence(page, word_boxes)
    pair = evidence.pair
    modified, sources = annotated_rectangles(document)
    whole = BBox(x=0, y=0, width=page.shape[1], height=page.shape[0])
    forged = document.label is Label.FORGED
    annotated = forged and bool(modified)
    hit = source_hit = pair_hit = False
    baseline = source_baseline = pair_baseline = 0.0
    if pair is not None and annotated:
        regions = [pair.first, pair.second]
        rng = receipt_rng(document.doc_id)
        hit = any(r.overlaps(b) for r in regions for b in [*modified, *sources])
        baseline = any_box_baseline(regions, whole, word_boxes, [*modified, *sources], rng)
        if sources:
            source_hit = any(r.overlaps(b) for r in regions for b in sources)
            source_baseline = any_box_baseline(regions, whole, word_boxes, sources, rng)
            pair_hit = matches_both(pair, modified, sources)
            pair_baseline = both_baseline(pair, whole, word_boxes, modified, sources, rng=rng)
    return Measured(
        forged=forged,
        score=evidence.document_score,
        techniques=frozenset(t for r in document.modified_regions for t in r.techniques),
        annotated=annotated,
        sourced=forged and bool(sources),
        large=has_large_rectangle(document, whole),
        earlier=document.doc_id in EARLIER_CONTACTS,
        paired=pair is not None,
        hit=hit,
        baseline=baseline,
        source_hit=source_hit,
        source_baseline=source_baseline,
        pair_hit=pair_hit,
        pair_baseline=pair_baseline,
    )


def _estimate(e: Estimate) -> dict[str, float]:
    return {"value": e.value, "low": e.low, "high": e.high}


def _on(
    metric: Callable[[npt.NDArray[np.bool_], npt.NDArray[np.float64]], float],
    y: npt.NDArray[np.bool_],
    s: npt.NDArray[np.float64],
) -> Callable[[npt.NDArray[np.intp]], float]:
    return lambda draw: metric(y[draw], s[draw])


def _detection(chosen: Sequence[Measured]) -> dict[str, Any]:
    y = np.array([m.forged for m in chosen])
    s = np.array([m.score for m in chosen], dtype=np.float64)
    metrics: dict[str, Any] = {
        name: _estimate(stratified(_on(metric, y, s), y))
        for name, metric in ranking_metrics().items()
    }
    return metrics | {"receipts": len(chosen), "forged": int(y.sum()), "chance": float(y.mean())}


def _guard(chosen: Sequence[Measured], hit: Callable[[Measured], bool]) -> dict[str, Any]:
    guard: Guard = localisation_guard([hit(m) for m in chosen], [m.baseline for m in chosen])
    return {
        "receipts": guard.receipts,
        "hit_rate": _estimate(guard.hit_rate),
        "baseline": guard.baseline,
        "detects": guard.detects,
    }


def _rate(chosen: Sequence[Measured], hit: Callable[[Measured], bool], base: str) -> dict[str, Any]:
    estimate = ratio([float(hit(m)) for m in chosen], [1.0] * len(chosen))
    baselines = [getattr(m, base) for m in chosen]
    return {
        "receipts": len(chosen),
        "rate": _estimate(estimate),
        "baseline": float(np.mean(baselines)),
        "above_baseline": estimate.low > float(np.mean(baselines)),
    }


def results(measured: Sequence[Measured]) -> dict[str, Any]:
    """Every aggregate the report shows, ready for the run record.

    Raises:
        ValueError: If no forged receipt has region annotations or an annotated copy source.
    """
    annotated = [m for m in measured if m.annotated]
    sourced = [m for m in measured if m.sourced]
    if not annotated or not sourced:
        msg = "the measurement needs forged receipts with regions and with copy sources"
        raise ValueError(msg)
    authentic = [m for m in measured if not m.forged]
    with_copy = [m for m in measured if not m.forged or Technique.CPI in m.techniques]
    return {
        "detection": {
            "all": _detection(measured),
            "copy_paste_inside": _detection(with_copy),
            "without_earlier_contact": _detection([m for m in measured if not m.earlier]),
        },
        "pairs": {
            "authentic": sum(m.paired for m in authentic),
            "authentic_receipts": len(authentic),
            "forged": sum(m.paired for m in measured if m.forged),
            "forged_receipts": len(measured) - len(authentic),
        },
        "guard": {
            "all": _guard(annotated, lambda m: m.hit),
            "with_a_pair": _guard([m for m in annotated if m.paired], lambda m: m.hit),
            "without_large_rectangles": _guard(
                [m for m in annotated if not m.large], lambda m: m.hit
            ),
            "without_earlier_contact": _guard(
                [m for m in annotated if not m.earlier], lambda m: m.hit
            ),
        },
        "copy_sources": {
            "either_region": _rate(sourced, lambda m: m.source_hit, "source_baseline"),
            "pair": _rate(sourced, lambda m: m.pair_hit, "pair_baseline"),
        },
        "by_technique": {
            t.value: {
                "receipts": sum(t in m.techniques for m in annotated),
                "hits": sum(t in m.techniques and m.hit for m in annotated),
            }
            for t in NAMED_TECHNIQUES
        },
    }


def _interval(e: Mapping[str, float]) -> str:
    return f"{e['value']:.3f} ({e['low']:.3f} to {e['high']:.3f})"


def render_report(record_results: Mapping[str, Any], commit: str | None) -> str:
    """The report of the clean measurement, from its run record."""
    detection = record_results["detection"]
    ceilings = ", ".join(
        f"{min(1.0, 0.10 * d['receipts'] / d['forged']):.3f} for {label}"
        for label, d in (
            ("all receipts", detection["all"]),
            ("copy-paste inside against authentic", detection["copy_paste_inside"]),
        )
    )
    detection_table = Table(
        "Detection on the clean folds",
        ("Receipts", "Receipts (forged)", "Chance", *METRIC_NAMES.values()),
        [
            [
                label,
                f"{d['receipts']} ({d['forged']})",
                d["chance"],
                *(_interval(d[name]) for name in METRIC_NAMES),
            ]
            for label, d in (
                ("all", detection["all"]),
                ("copy-paste inside against authentic", detection["copy_paste_inside"]),
                ("without earlier contact", detection["without_earlier_contact"]),
            )
        ],
        note="The document score pooled over the four clean folds, with stratified bootstrap "
        "intervals (10,000 resamples, seed 0). Chance: the share of forged receipts for average "
        "precision, 0.5 for ROC-AUC, the budget for recall. Without earlier contact: the "
        "receipts viewed before ADR-0014 are left out (decision 1). A budget limits recall: at "
        f"10 % it cannot exceed {ceilings}. F1 needs a threshold chosen on the validation split "
        "and is not reported (ADR-0029, section 1).",
    )
    pairs = record_results["pairs"]
    guard_rows: list[list[Cell]] = []
    for label, key in (
        ("all", "all"),
        ("with a pair", "with_a_pair"),
        ("without large rectangles", "without_large_rectangles"),
        ("without earlier contact", "without_earlier_contact"),
    ):
        g = record_results["guard"][key]
        verdict = "points at forgeries" if g["detects"] else "probable shortcut"
        guard_rows.append([label, g["receipts"], _interval(g["hit_rate"]), g["baseline"], verdict])
    guard_table = Table(
        "Localisation guard of the pair",
        (
            "Forged receipts with regions",
            "Count",
            "Hit rate (95 % interval)",
            "Random baseline",
            "Verdict",
        ),
        guard_rows,
        note="The pair hits if either region shares a pixel with a modified rectangle or a copy "
        "source (ADR-0015, section 3; ADR-0030). Baseline: the share of 1,000 draws in which a "
        "random box of either region's size, centred on OCR text, hits. A receipt without a pair "
        "is a miss with a baseline of 0. Verdict: the pair points at forgeries if the lower "
        "bound of its hit rate exceeds the baseline (ADR-0014, decision 4). Large rectangles "
        "cover more than 5 % of the page (ADR-0015, section 4). Receipts with a pair: "
        f"{pairs['forged']} of {pairs['forged_receipts']} forged and {pairs['authentic']} of "
        f"{pairs['authentic_receipts']} authentic.",
    )
    sources = record_results["copy_sources"]
    source_table = Table(
        "Copy sources",
        ("Measure", "Receipts", "Rate (95 % interval)", "Random baseline", "Above baseline"),
        [
            [
                label,
                s["receipts"],
                _interval(s["rate"]),
                s["baseline"],
                "yes" if s["above_baseline"] else "no",
            ]
            for label, s in (
                ("copy-source localisation: either region on a source", sources["either_region"]),
                ("pair hit rate: one region on an edit, the other on a source", sources["pair"]),
            )
        ],
        note="Forged receipts with an annotated copy source; sources are only partly annotated "
        "(data card). The annotations do not link a source to its copy, so the pair hit rate "
        "pairs any modified rectangle with any source of the same receipt. Baselines from "
        "independent random boxes of the regions' sizes; above baseline: the lower bound of "
        "the rate exceeds it.",
    )
    technique_table = Table(
        "Guard hits by forgery technique",
        ("Technique", "Forged receipts with regions", "Pair hits"),
        [
            [name.upper(), t["receipts"], t["hits"]]
            for name, t in record_results["by_technique"].items()
        ],
        note="A receipt counts under every technique its modified regions use. Descriptive; "
        "groups are small (evaluation protocol, section 7).",
    )
    header = (
        "# Copy-move on the clean folds\n\n"
        "> Generated from the run record `copy-move-clean` by `docfraud copy-move report`. Do "
        "not edit by hand.\n\n"
        "The single measurement of the frozen copy-move detector "
        f"([ADR-0029](../../docs/decisions/0029-fusion-preregistration.md), section 4, as "
        f"[ADR-0030](../../docs/decisions/0030-copy-move-reports-a-pair.md) amends it), at "
        f"commit `{(commit or 'unknown')[:7]}`. Development: "
        "[record](../../docs/copy-move-development.md).\n\n"
    )
    tables = (detection_table, guard_table, source_table, technique_table)
    return header + "\n".join(render_table(t) for t in tables)
