from typing import Any

import numpy as np
import pytest

from docfraud.fusion.development import (
    PRACTICAL_VARIANTS,
    SINGLE_SOURCES,
    VARIANTS,
    contribution_groups,
    first_regions,
    group_contributions,
    metrics,
    rq3,
    score,
    track_results,
    variant_inputs,
)
from docfraud.fusion.development_report import render_report
from docfraud.fusion.inputs import LINK_1, LINK_2
from docfraud.fusion.training_inputs import ReceiptInputs
from tests.fusion.test_training_inputs import table

N = 40


def setup() -> tuple[dict[int, list[ReceiptInputs]], np.ndarray, np.ndarray]:
    built = table(n=N, seeds=(0, 1, 2))
    rows_by_seed = {seed: built.rows("transcript", seed) for seed in (0, 1, 2)}
    y = np.array([i % 4 == 0 for i in range(N)])
    return rows_by_seed, y, np.arange(N) % 5


def test_leaving_a_source_out_recomputes_the_links_from_the_others() -> None:
    rows_by_seed, _, _ = setup()
    row = rows_by_seed[0][0]  # forged: the rule fails and system 2 lies elsewhere
    by_name = {v.name: v for v in VARIANTS}
    assert variant_inputs(row, by_name["fusion"])[-2:] == [row.inputs[LINK_1], row.inputs[LINK_2]]
    without_pixel = variant_inputs(row, by_name["fusion without pixel evidence"])
    assert without_pixel[-2:] == [0.0, 0.0]  # no copy-move pair, so no region to link
    assert len(variant_inputs(row, by_name["fusion without the accounting rules"])) == 5


def test_the_track_results_hold_every_row_comparison_seed_and_subset() -> None:
    rows_by_seed, y, fold_of = setup()
    subsets = {"without fold 0": fold_of != 0}
    results, scored = track_results(rows_by_seed, y, fold_of, subsets=subsets, resamples=50)
    assert list(results["table"]) == [v.name for v in VARIANTS]
    assert results["table"]["accounting rules"]["average_precision"]["value"] == pytest.approx(1.0)
    singles = {v.name for v in SINGLE_SOURCES}
    assert set(results["comparisons"]) == singles | {"score-level baseline"}
    assert set(results["seeds"]) == {"0", "1", "2"}
    assert results["subsets"] == {"without fold 0": int((fold_of != 0).sum())}
    assert set(results["sensitivity"]["without fold 0"]) == {v.name for v in VARIANTS}
    assert np.isfinite(results["calibration_slope"])
    assert np.isfinite(results["calibration_slope_unpenalised"])
    assert isinstance(results["fusion_beats_every_single_source"], bool)
    assert set(scored) == {v.name for v in VARIANTS}


def test_rq3_and_rq4_come_from_each_receipts_fold_model() -> None:
    rows_by_seed, y, fold_of = setup()
    _, scored = track_results(rows_by_seed, y, fold_of, subsets={}, resamples=50)
    rows = rows_by_seed[0]
    contributions = group_contributions(rows, scored["fusion"], fold_of)
    assert contributions.shape == (N, len(contribution_groups()))
    groups = {"technique": {"CPI": y.copy(), "none": np.zeros(N, dtype=bool)}}
    found = rq3(contributions, scored, [r.receipt for r in rows], groups)
    cpi = found["technique"]["CPI"]
    assert "none" not in found["technique"]
    assert cpi["receipts"] == int(y.sum())
    assert sum(cpi["largest"].values()) == int(y.sum())
    assert cpi["caught"]["accounting rules"] == pytest.approx(4.0)  # 10 % of 40 receipts
    regions = first_regions(rows, scored["fusion"], fold_of)
    assert len(regions) == N
    practical = {
        v.name: metrics(y, score(rows, v, y, fold_of).probabilities, 50) for v in PRACTICAL_VARIANTS
    }
    assert set(practical) == {"file metadata alone", "fusion of track T3"}


def report_record() -> dict[str, Any]:
    """The results of a run of `docfraud fusion training-oof` on a small made-up table."""
    rows_by_seed, y, fold_of = setup()
    results, scored = track_results(
        rows_by_seed, y, fold_of, subsets={"all": np.ones(N, bool)}, resamples=50
    )
    contributions = group_contributions(rows_by_seed[0], scored["fusion"], fold_of)
    groups = {"technique": {"CPI": y.copy()}}
    rq = rq3(contributions, scored, [r.receipt for r in rows_by_seed[0]], groups)
    estimate = {"value": 0.5, "low": 0.4, "high": 0.6}
    rq4 = {
        "receipts": 10,
        "located": 9,
        "hit_rate": estimate,
        "baseline": 0.1,
        "detects": True,
        "in_review": 4,
        "in_review_hits": 3,
        "by_technique": {"CPI": {"receipts": 8, "hits": 5}},
        "by_entity": {"total_payment": {"receipts": 6, "hits": 4}},
    }
    return {
        "transcript": results,
        "ocr": results,
        "T3": {"fusion of track T2": results["table"]["fusion"]},
        "rq3": {"transcript": rq, "ocr": rq},
        "rq4": {"transcript": rq4, "ocr": rq4},
    }


def test_the_report_renders_every_section_from_the_results() -> None:
    text = render_report(report_record(), "0123456789")
    for title in (
        "Ablation table, T1 (transcripts)",
        "Comparisons of section 8, T2 (OCR)",
        "Fusion by system 2 seed",
        "Slope without the penalty",
        "Track T3",
        "RQ3 by technique, T1 (transcripts)",
        "RQ4: the report's first region",
        "RQ4 by entity",
    ):
        assert title in text
    assert "`0123456`" in text
    assert "points at forgeries" in text
