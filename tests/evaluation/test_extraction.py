from decimal import Decimal

import pytest

from docfraud.evaluation.extraction import (
    METRICS,
    DocumentScore,
    estimate,
    item_matches,
    maximum_matching,
    render_extraction_report,
    score_document,
    similarity,
    tax_row_matches,
)
from docfraud.reading.fields import AmountRole, LineItem, ReceiptFields, TaxSummaryRow


def fields(**values: object) -> ReceiptFields:
    return ReceiptFields.model_validate(values)


def item(amount: str | None, quantity: str | None = None, unit: str | None = None) -> LineItem:
    return LineItem.model_validate({"amount": amount, "quantity": quantity, "unit_price": unit})


def test_items_match_on_amount_and_on_quantity_and_price_where_both_have_them() -> None:
    gold = item("3.00", "2", "1.50")
    assert item_matches(gold, item("3.00"))
    assert item_matches(gold, item("3.00", "2.0", "1.5"))
    assert not item_matches(gold, item("3.00", "3"))
    assert not item_matches(gold, item("3.01", "2", "1.50"))
    assert not item_matches(item(None), item(None))


def test_tax_rows_match_on_rate_taxable_and_tax_but_not_code() -> None:
    gold = TaxSummaryRow.model_validate({"code": "SR", "rate_percent": "6", "taxable_amount": "10"})
    assert tax_row_matches(gold, gold.model_copy(update={"code": "S"}))
    assert not tax_row_matches(gold, gold.model_copy(update={"tax": Decimal("0.60")}))


def test_matching_is_maximal_and_independent_of_order() -> None:
    # A greedy first-fit would pair gold 0 with predicted 0 and leave gold 1 unmatched.
    gold = [item("1.00"), item("1.00", "2")]
    predicted = [item("1.00", "2"), item("1.00", "1")]
    assert maximum_matching(gold, predicted, item_matches) == [(0, 1), (1, 0)]
    assert len(maximum_matching(gold, predicted[::-1], item_matches)) == 2
    assert maximum_matching([], predicted, item_matches) == []


@pytest.mark.parametrize(
    ("a", "b", "expected"),
    [("TEH (B)", "teh  (b)", 1.0), ("TEA", "TEH", 2 / 3), ("", "", 1.0), ("ABC", "", 0.0)],
)
def test_similarity_is_normalised_edit_similarity(a: str, b: str, expected: float) -> None:
    assert similarity(a, b) == pytest.approx(expected)


GOLD = fields(
    merchant="Shop  A",
    dates=["2018-01-02"],
    items=[
        {"description": "TEA", "amount": "3.00", "quantity": "2"},
        {"description": "BUN", "amount": "1.00"},
    ],
    amounts={"total_due": ["4.00", "4.00"], "tax": ["0.23"], "tendered": ["5.00"]},
    tax_summary=[{"code": "SR", "rate_percent": "6", "taxable_amount": "3.77", "tax": "0.23"}],
    quantity_total="3",
)


def test_score_counts_amounts_as_a_multiset() -> None:
    predicted = fields(
        merchant="SHOP A",
        dates=["2018-01-02"],
        items=[{"description": "TEH", "amount": "3.00"}, {"description": "X", "amount": "9.00"}],
        amounts={"total_due": ["4.00"], "tax": ["0.23"], "change": ["1.00"], "tendered": ["5"]},
        quantity_total="3.0",
    )
    score = score_document("D1", GOLD, predicted)
    assert (score.amounts.matched, score.amounts.predicted, score.amounts.gold) == (3, 4, 4)
    assert score.missed_amounts == ((AmountRole.TOTAL_DUE, Decimal("4.00")),)
    assert score.extra_amounts == ((AmountRole.CHANGE, Decimal("1.00")),)
    assert not score.amounts_exact
    assert (score.items.matched, score.items.predicted, score.items.gold) == (1, 2, 2)
    assert score.description_similarities == pytest.approx((2 / 3,))
    assert (score.tax_rows.matched, score.tax_rows.predicted, score.tax_rows.gold) == (0, 0, 1)
    assert score.merchant
    assert score.dates
    assert score.item_count
    assert score.quantity_total


def test_a_perfect_prediction_is_exact() -> None:
    score = score_document("D1", GOLD, GOLD)
    assert score.amounts_exact
    assert score.amounts.matched == score.amounts.gold == 4
    assert score.tax_rows.matched == 1


def scored(doc_id: str, predicted: ReceiptFields) -> DocumentScore:
    return score_document(doc_id, GOLD, predicted)


def test_metrics_are_ratios_of_document_sums() -> None:
    scores = [scored("A", GOLD), scored("B", fields())]
    values = {m.name: e.value for m in METRICS if (e := estimate(m, scores)) is not None}
    assert values["Amounts: precision"] == 1.0
    assert values["Amounts: recall"] == 0.5
    assert values["Amounts: F1"] == pytest.approx(2 * 4 / (4 + 8))
    assert values["Documents with all amounts exact"] == 0.5
    assert estimate(METRICS[0], [scored("B", fields())]) is None  # nothing predicted


def test_report_compares_two_sources_on_the_same_documents() -> None:
    t1 = [scored("A", GOLD), scored("F", GOLD)]
    t2 = [scored("A", fields(amounts={"total_due": ["4.00"]})), scored("F", GOLD)]
    report = render_extraction_report({"T1": t1, "T2": t2}, frozenset({"F"}), "# Head\n\n")
    assert report.startswith("# Head\n\n### Headline metrics")
    assert "| Amounts: recall | 1.000 [1.000, 1.000] | 0.625 [" in report
    assert "### Difference T2 - T1" in report
    assert "| authentic (1) | 1.000 [1.000, 1.000] | 0.400 [" in report
    assert "| total_due | 4 | 1.000 | 0 | 0.750 | 0 |" in report
    assert (
        "| A | no | 1 / 4 / 1 | tax 0.23, tendered 5.00, total_due 4.00 | \N{EM DASH} |" in report
    )
    with pytest.raises(ValueError, match="same documents"):
        render_extraction_report({"T1": t1, "T2": t2[::-1]}, frozenset(), "")
