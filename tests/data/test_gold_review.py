import json
import re
from pathlib import Path

import pytest

from docfraud.data.gold import GoldAnnotation, GoldManifest
from docfraud.data.gold_review import (
    ReceiptReview,
    ReviewResult,
    UnitVerdict,
    draw_review_sample,
    image_url,
    render_review_page,
    render_review_report,
    review_units,
    score_review,
)
from docfraud.data.models import Split
from docfraud.reading.fields import ReceiptFields

MANIFEST = GoldManifest(
    dataset="d",
    split=Split.TRAIN,
    seed=0,
    authentic=tuple(f"A{i:02}" for i in range(20)),
    forged=tuple(f"F{i:02}" for i in range(5)),
)

FIELDS = ReceiptFields.model_validate(
    {
        "merchant": "SHOP",
        "dates": ["2018-01-02"],
        "items": [{"description": "TEA", "quantity": "2", "unit_price": "1.50", "amount": "3.00"}],
        "amounts": {"total_due": ["3.00", "3.00"], "tax": ["0.17"]},
        "tax_summary": [{"code": "SR", "rate_percent": "6", "taxable_amount": "2.83"}],
        "item_count": 1,
        "quantity_total": "2",
    }
)


def annotation(doc_id: str, fields: ReceiptFields = FIELDS) -> GoldAnnotation:
    return GoldAnnotation(doc_id=doc_id, annotator="t", fields=fields, notes="note")


def test_sample_is_stratified_seeded_and_sorted() -> None:
    sample = draw_review_sample(MANIFEST, n_authentic=4, n_forged=1, seed=3)
    assert len(sample) == 5
    assert sum(s.startswith("F") for s in sample) == 1
    assert list(sample) == sorted(sample)
    reordered = MANIFEST.model_copy(
        update={"authentic": MANIFEST.authentic[::-1], "forged": MANIFEST.forged[::-1]}
    )
    assert draw_review_sample(reordered, n_authentic=4, n_forged=1, seed=3) == sample
    assert draw_review_sample(MANIFEST, n_authentic=4, n_forged=1, seed=4) != sample


def test_sample_needs_large_enough_strata() -> None:
    with pytest.raises(ValueError, match="cannot draw 6 receipts from a stratum of 5"):
        draw_review_sample(MANIFEST, n_forged=6)


def test_every_recorded_value_is_one_unit() -> None:
    units = review_units(FIELDS)
    assert [u.unit_id for u in units] == [
        "merchant",
        "date:0",
        "item:0",
        "amount:tax:0",
        "amount:total_due:0",
        "amount:total_due:1",
        "tax:0",
        "item_count",
        "quantity_total",
    ]
    assert units[2].text == "TEA | qty 2 | unit price 1.50 | amount 3.00"
    assert units[6].text == "code SR | rate 6% | taxable 2.83 | tax \N{EM DASH}"
    assert review_units(ReceiptFields()) == ()


def full_review(doc_id: str, *, reject: str = "", missed: tuple[str, ...] = ()) -> ReceiptReview:
    return ReceiptReview(
        doc_id=doc_id,
        verdicts=tuple(
            UnitVerdict(unit_id=u.unit_id, accepted=u.unit_id != reject, comment="bad")
            for u in review_units(FIELDS)
        ),
        missed=missed,
    )


def test_score_counts_rejections_and_missed_values() -> None:
    result = ReviewResult(
        reviewer="owner",
        reviews=(
            full_review("A01", reject="merchant", missed=("rounding 0.00",)),
            full_review("F00"),
        ),
    )
    scores = score_review(
        result, {"A01": annotation("A01"), "F00": annotation("F00")}, ["F00", "A01"]
    )
    assert [(s.doc_id, s.units, s.accepted, s.missed) for s in scores] == [
        ("A01", 9, 8, 1),
        ("F00", 9, 9, 0),
    ]
    assert scores[0].rate == pytest.approx(0.8)
    report = render_review_report(result, scores, {"A01/merchant": "kept"})
    assert "| A01 | merchant | bad | kept |" in report
    assert "| A01 | missed:rounding 0.00 | rounding 0.00 | unresolved |" in report
    assert "17 of 18 units accepted, 1 missed values; agreement 0.895" in report


def test_empty_receipt_agrees_trivially() -> None:
    empty = ReceiptFields()
    result = ReviewResult(reviewer="o", reviews=(ReceiptReview(doc_id="A01", verdicts=()),))
    (score,) = score_review(result, {"A01": annotation("A01", empty)}, ["A01"])
    assert score.rate == 1.0


def test_review_must_cover_the_sample() -> None:
    result = ReviewResult(reviewer="o", reviews=(full_review("A01"),))
    with pytest.raises(ValueError, match="expected the sample"):
        score_review(result, {"A01": annotation("A01")}, ["A01", "A02"])
    twice = ReviewResult(reviewer="o", reviews=(full_review("A01"), full_review("A01")))
    with pytest.raises(ValueError, match="expected the sample"):
        score_review(twice, {"A01": annotation("A01")}, ["A01"])


def test_verdicts_must_match_units() -> None:
    partial = ReceiptReview(doc_id="A01", verdicts=full_review("A01").verdicts[1:])
    with pytest.raises(ValueError, match="A01: verdicts do not match"):
        score_review(
            ReviewResult(reviewer="o", reviews=(partial,)), {"A01": annotation("A01")}, ["A01"]
        )
    verdicts = full_review("A01").verdicts
    doubled = ReceiptReview(doc_id="A01", verdicts=(*verdicts, verdicts[0]))
    with pytest.raises(ValueError, match="A01: verdicts do not match"):
        score_review(
            ReviewResult(reviewer="o", reviews=(doubled,)), {"A01": annotation("A01")}, ["A01"]
        )


def test_page_embeds_the_units_safely() -> None:
    hostile = FIELDS.model_copy(update={"merchant": "</script><b>x"})
    page = render_review_page(["A01"], {"A01": annotation("A01", hostile)}, {"A01": "img/a.png"})
    assert "__DATA__" not in page
    assert "</script><b>" not in page
    payload = re.search(r'<script id="data" type="application/json">(.*?)</script>', page, re.S)
    assert payload is not None
    (receipt,) = json.loads(payload.group(1))
    assert receipt["image"] == "img/a.png"
    assert receipt["units"][0] == {"id": "merchant", "field": "merchant", "text": "</script><b>x"}


def test_image_url_is_relative_to_the_page(tmp_path: Path) -> None:
    page_dir = tmp_path / "artifacts" / "review"
    image = tmp_path / "data" / "train" / "X1.png"
    assert image_url(page_dir, image) == "../../data/train/X1.png"
