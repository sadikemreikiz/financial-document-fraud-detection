from dataclasses import replace
from pathlib import Path

import numpy as np
import pytest

from docfraud.data.models import BBox, Document, Label, Region, RegionRole, Split, Technique
from docfraud.detectors.copy_move import CopyMovePair
from docfraud.detectors.copy_move_measurement import (
    Measured,
    both_baseline,
    matches_both,
    measure,
    render_report,
    results,
)
from docfraud.evaluation.localisation import receipt_rng
from tests.detectors.test_copy_move import ink_patch, noisy_page

EDIT, SOURCE, ELSEWHERE = (
    BBox(x=180, y=100, width=20, height=30),
    BBox(x=30, y=20, width=20, height=30),
    BBox(x=120, y=60, width=10, height=10),
)
PAGE = BBox(x=0, y=0, width=240, height=160)
WORDS = [BBox(x=10, y=10 + 30 * k, width=220, height=24) for k in range(5)]


def pair(first: BBox, second: BBox) -> CopyMovePair:
    return CopyMovePair(first, second, blocks=1, area=first.area)


def receipt(label: Label, regions: tuple[Region, ...] = ()) -> Document:
    return Document(
        dataset="toy",
        doc_id=f"X{label.value}",
        source_id=f"X{label.value}",
        split=Split.TRAIN,
        label=label,
        image_path=Path("x.png"),
        regions=regions,
        has_region_annotations=bool(regions),
    )


def test_a_pair_matches_both_when_one_region_is_on_the_edit_and_the_other_on_its_source() -> None:
    assert matches_both(pair(SOURCE, EDIT), [EDIT], [SOURCE])
    assert matches_both(pair(EDIT, SOURCE), [EDIT], [SOURCE])
    assert not matches_both(pair(EDIT, EDIT), [EDIT], [SOURCE])
    assert not matches_both(pair(SOURCE, ELSEWHERE), [EDIT], [SOURCE])


def test_the_baseline_of_both_needs_one_box_on_each_and_is_reproducible() -> None:
    small = pair(BBox(x=0, y=0, width=2, height=2), BBox(x=0, y=0, width=2, height=2))
    everywhere = [PAGE]
    assert both_baseline(small, PAGE, WORDS, everywhere, everywhere, rng=receipt_rng("X1")) == 1.0
    assert both_baseline(small, PAGE, WORDS, everywhere, [], rng=receipt_rng("X1")) == 0.0
    half = [BBox(x=0, y=0, width=120, height=160)]
    other = [BBox(x=120, y=0, width=120, height=160)]
    first = both_baseline(small, PAGE, WORDS, half, other, rng=receipt_rng("X1"))
    assert 0.4 < first < 0.6  # about 2 x 0.5 x 0.5
    assert both_baseline(small, PAGE, WORDS, half, other, rng=receipt_rng("X1")) == first


def test_a_copied_patch_is_found_with_its_source_and_judged_against_chance() -> None:
    page = noisy_page()
    page[SOURCE.y : SOURCE.y2, SOURCE.x : SOURCE.x2] = ink_patch(30, 20)
    page[EDIT.y : EDIT.y2, EDIT.x : EDIT.x2] = page[SOURCE.y : SOURCE.y2, SOURCE.x : SOURCE.x2]
    regions = (
        Region(box=EDIT, role=RegionRole.MODIFIED, techniques=frozenset({Technique.CPI})),
        Region(box=SOURCE, role=RegionRole.SOURCE),
    )
    found = measure(receipt(Label.FORGED, regions), page, WORDS)
    assert (found.forged, found.annotated, found.sourced, found.paired) == (True,) * 4
    assert (found.hit, found.source_hit, found.pair_hit) == (True,) * 3
    assert found.techniques == {Technique.CPI}
    assert 0 < found.baseline < 1
    assert 0 < found.pair_baseline < found.baseline
    assert found.score == pytest.approx(600 / 24**2)
    authentic = measure(receipt(Label.AUTHENTIC), page, WORDS)
    assert (authentic.paired, authentic.annotated, authentic.hit) == (True, False, False)
    assert authentic.baseline == 0.0


def test_a_receipt_without_an_annotated_source_has_no_source_measures() -> None:
    page = noisy_page()
    page[SOURCE.y : SOURCE.y2, SOURCE.x : SOURCE.x2] = ink_patch(30, 20)
    page[EDIT.y : EDIT.y2, EDIT.x : EDIT.x2] = page[SOURCE.y : SOURCE.y2, SOURCE.x : SOURCE.x2]
    edit = Region(box=EDIT, role=RegionRole.MODIFIED, techniques=frozenset({Technique.CPI}))
    found = measure(receipt(Label.FORGED, (edit,)), page, WORDS)
    assert (found.hit, found.sourced, found.source_hit, found.pair_hit) == (
        True,
        False,
        False,
        False,
    )
    assert (found.source_baseline, found.pair_baseline) == (0.0, 0.0)


def measured(*, forged: bool, score: float, hit: bool = False, paired: bool = True) -> Measured:
    return Measured(
        forged=forged,
        score=score,
        techniques=frozenset({Technique.CPI}) if forged else frozenset(),
        annotated=forged,
        sourced=forged,
        large=False,
        earlier=False,
        paired=paired,
        hit=hit,
        baseline=0.1 if paired and forged else 0.0,
        source_hit=hit,
        source_baseline=0.05 if paired and forged else 0.0,
        pair_hit=hit,
        pair_baseline=0.01 if paired and forged else 0.0,
    )


def toy_results() -> list[Measured]:
    forged = [measured(forged=True, score=1.0 - k / 10, hit=k < 6) for k in range(8)]
    forged.append(measured(forged=True, score=0.0, paired=False))
    authentic = [measured(forged=False, score=k / 100) for k in range(30)]
    return forged + authentic


def test_results_hold_detection_guard_sources_and_techniques() -> None:
    summary = results(toy_results())
    assert summary["detection"]["all"]["receipts"] == 39
    assert summary["detection"]["all"]["chance"] == pytest.approx(9 / 39)
    ap = summary["detection"]["all"]["average_precision"]["value"]
    assert ap == pytest.approx(8 / 9 + 1 / 39)  # the receipt without a pair ties at the bottom
    guard = summary["guard"]["all"]
    assert guard["receipts"] == 9
    assert guard["hit_rate"]["value"] == pytest.approx(6 / 9)
    assert guard["baseline"] == pytest.approx(0.8 / 9)  # the receipt without a pair adds 0
    assert summary["guard"]["with_a_pair"]["receipts"] == 8
    assert summary["copy_sources"]["pair"]["rate"]["value"] == pytest.approx(6 / 9)
    assert summary["by_technique"]["cpi"] == {"receipts": 9, "hits": 6}
    assert summary["pairs"] == {
        "authentic": 30,
        "authentic_receipts": 30,
        "forged": 8,
        "forged_receipts": 9,
    }


def test_results_need_forged_receipts_with_regions_and_sources() -> None:
    authentic = [measured(forged=False, score=0.1)] * 3
    with pytest.raises(ValueError, match="copy sources"):
        results(authentic)
    no_source = [replace(m, sourced=False) for m in toy_results()]
    with pytest.raises(ValueError, match="copy sources"):
        results(no_source)


def test_the_report_states_every_table_and_its_verdicts() -> None:
    text = render_report(results(toy_results()), "0123456789abcdef")
    for title in (
        "Detection on the clean folds",
        "Localisation guard of the pair",
        "Copy sources",
        "Guard hits by forgery technique",
    ):
        assert title in text
    assert "points at forgeries" in text
    assert "`0123456`" in text
    assert "`unknown`" in render_report(results(toy_results()), None)
    assert np.isfinite(results(toy_results())["detection"]["all"]["roc_auc"]["value"])
