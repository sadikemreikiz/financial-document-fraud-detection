"""Phase 7: where the pixel systems and copy-move point on the test split (ADR-0034, 1 and 5).

The maps measured are those fusion reads: system 2, the model trained on the whole training
split at seed 0, and system 1, the JPEG-lattice map. Each map is measured as Phase 4 measured
its held-out folds (:func:`~docfraud.forensics.learned.results.measure_receipt`): pixel average
precision, and per-receipt F1 and IoU at a pixel threshold chosen, before the test split is
unlocked, on the forged validation receipts by the rule of ADR-0015 (section 5); and the top
region against its random baseline, over the edits and copy sources and over the copy sources
alone. The copy-move detector is measured as Phase 5 measured it on the clean folds
(:mod:`docfraud.detectors.copy_move_measurement`): its pair against the edits and against the
copy sources. All of it is descriptive.
"""

from collections.abc import Iterable, Mapping, Sequence
from typing import Any

from docfraud.data.models import BBox
from docfraud.evaluation.bootstrap import DEFAULT_RESAMPLES, ratio
from docfraud.evaluation.localisation import (
    baseline_hit_rate,
    hits,
    localisation_guard,
    receipt_rng,
)
from docfraud.forensics.learned.results import (
    Analyser,
    HeldMap,
    Item,
    MeasuredReceipt,
    PixelThreshold,
    choose_threshold,
    measure_receipt,
)

THRESHOLDS_RUN = "forensics-pixel-thresholds"
TEST_RUN = "forensics-test-localisation"

SYSTEMS = {2: "system 2, synthetic forgeries", 1: "system 1, JPEG lattice"}
"""The maps fusion reads, by their number in Phase 4."""

LOCALISATION_CODE = (
    "src/docfraud/data/normalize.py",
    "src/docfraud/detectors/copy_move.py",
    "src/docfraud/detectors/copy_move_measurement.py",
    "src/docfraud/evaluation/localisation.py",
    "src/docfraud/forensics",
    "src/docfraud/reading/ocr.py",
)
"""The code that makes and measures the maps; unchanged from the thresholds to the test."""


def pixel_threshold(analyser: Analyser, items: Iterable[Item]) -> PixelThreshold:
    """The pixel threshold of ADR-0015 (section 5) from the forged receipts with edits.

    Raises:
        ValueError: If no item is a forged receipt with a modified rectangle.
    """
    return choose_threshold(
        [
            HeldMap(item.key, item.modified, item.sources, analyser.analyse(item.page))
            for item in items
            if item.forged and item.modified
        ]
    )


def measured(analyser: Analyser, items: Iterable[Item], threshold: float) -> list[MeasuredReceipt]:
    """Every receipt's map measured at the frozen pixel threshold, one map at a time."""
    return [
        measure_receipt(item, analyser.analyse(item.page), threshold, peaks=False) for item in items
    ]


def _mean(values: Sequence[float], resamples: int) -> dict[str, float] | None:
    if not values:
        return None
    e = ratio(list(values), [1.0] * len(values), resamples=resamples)
    return {"value": e.value, "low": e.low, "high": e.high}


def _guard(
    receipts: Sequence[MeasuredReceipt],
    targets: Mapping[str, Sequence[BBox]],
    words: Mapping[str, Sequence[BBox]],
    resamples: int,
) -> dict[str, Any] | None:
    """The top region against its random baseline on the receipts with targets."""
    chosen = [r for r in receipts if targets.get(r.key)]
    if not chosen:
        return None
    found = [hits(r.top_region, targets[r.key]) for r in chosen]
    baselines = [
        baseline_hit_rate(
            r.top_region,
            BBox(x=0, y=0, width=r.width, height=r.height),
            words[r.key],
            targets[r.key],
            receipt_rng(r.key),
        )
        for r in chosen
    ]
    guard = localisation_guard(found, baselines, resamples=resamples)
    e = guard.hit_rate
    return {
        "receipts": guard.receipts,
        "hit_rate": {"value": e.value, "low": e.low, "high": e.high},
        "baseline": guard.baseline,
        "detects": guard.detects,
    }


def map_results(
    receipts: Sequence[MeasuredReceipt],
    words: Mapping[str, Sequence[BBox]],
    threshold: float,
    *,
    resamples: int = DEFAULT_RESAMPLES,
) -> dict[str, Any]:
    """A system's localisation on the measured receipts, as Phase 4 reported it.

    ``words`` are each receipt's OCR word boxes, on which the random baseline places its boxes.
    """
    pixels = [r.pixels for r in receipts if r.pixels is not None]
    forged = [r for r in receipts if r.forged]
    edits = {r.key: [*r.modified, *r.sources] for r in forged if r.modified}
    sources = {r.key: list(r.sources) for r in forged if r.sources}
    return {
        "threshold": threshold,
        "forged_with_regions": len(pixels),
        "pixel_average_precision": _mean([p.average_precision for p in pixels], resamples),
        "chance": sum(p.chance for p in pixels) / len(pixels) if pixels else None,
        "f1": _mean([p.f1 for p in pixels], resamples),
        "iou": _mean([p.iou for p in pixels], resamples),
        "guard": _guard(forged, edits, words, resamples),
        "sources": _guard(forged, sources, words, resamples),
    }
