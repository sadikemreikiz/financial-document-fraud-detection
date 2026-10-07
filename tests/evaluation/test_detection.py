import numpy as np
import pytest
from sklearn import metrics as sk

from docfraud.evaluation.detection import (
    Counts,
    as_arrays,
    average_precision,
    brier,
    choose_threshold,
    compare,
    evaluate,
    expected_calibration_error,
    recall_at_budget,
    roc_auc,
)


def random_case(seed: int, n: int = 200, levels: int = 7) -> tuple[np.ndarray, np.ndarray]:
    """Labels and scores with many ties, as accounting rules produce them."""
    rng = np.random.default_rng(seed)
    y = rng.random(n) < 0.2
    y[:2] = [True, False]  # both classes always present
    s = np.round(rng.random(n) * (levels - 1) + 1.5 * y, 0) / levels
    return y, s


@pytest.mark.parametrize("seed", range(5))
def test_threshold_metrics_match_scikit_learn(seed: int) -> None:
    y, s = random_case(seed)
    predicted = s >= np.median(s)
    counts = Counts.of(y, predicted)
    assert counts.f1 == pytest.approx(sk.f1_score(y, predicted))
    assert counts.precision == pytest.approx(sk.precision_score(y, predicted, zero_division=0))
    assert counts.recall == pytest.approx(sk.recall_score(y, predicted))
    assert counts.mcc == pytest.approx(sk.matthews_corrcoef(y, predicted))
    assert counts.balanced_accuracy == pytest.approx(sk.balanced_accuracy_score(y, predicted))


@pytest.mark.parametrize("seed", range(5))
def test_ranking_metrics_match_scikit_learn_with_ties(seed: int) -> None:
    y, s = random_case(seed)
    assert average_precision(y, s) == pytest.approx(sk.average_precision_score(y, s))
    assert roc_auc(y, s) == pytest.approx(sk.roc_auc_score(y, s))


def test_ranking_metrics_ignore_the_order_of_tied_documents() -> None:
    y = np.array([True, False, True, False])
    s = np.array([0.5, 0.5, 0.5, 0.1])
    flipped = np.array([False, True, True, False])
    assert average_precision(y, s) == average_precision(flipped, s)
    assert roc_auc(y, s) == pytest.approx(0.75)


def test_brier_matches_scikit_learn() -> None:
    y, s = random_case(0)
    assert brier(y, s) == pytest.approx(sk.brier_score_loss(y, s))


def test_expected_calibration_error_on_equal_mass_bins() -> None:
    y = np.array([False, False, True, True])
    p = np.array([0.0, 0.5, 0.5, 1.0])
    # two bins of two: |0 - 0.25| and |1 - 0.75|, each weighted by half
    assert expected_calibration_error(y, p, bins=2) == pytest.approx(0.25)
    assert expected_calibration_error(y, y.astype(float), bins=2) == 0.0


def test_recall_at_budget_shares_the_cut_off_among_tied_documents() -> None:
    y = np.array([True, False, True, False, False, False, False, False, False, False])
    s = np.array([0.9, 0.5, 0.5, 0.5, 0.5, 0.1, 0.1, 0.1, 0.1, 0.1])
    # 2 of 10 reviewed: the 0.9 receipt, then 1 of the 4 tied at 0.5 (1 forged among them)
    assert recall_at_budget(y, s, 0.2) == pytest.approx((1 + 1 / 4) / 2)
    assert recall_at_budget(y, s, 1.0) == pytest.approx(1.0)


def test_threshold_maximises_f1() -> None:
    y = np.array([True, False, True, False])
    s = np.array([0.9, 0.8, 0.7, 0.1])
    # F1 at 0.9: 2/3, at 0.8: 1/2, at 0.7: 4/5, at 0.1: 2/3
    assert choose_threshold(y, s) == pytest.approx(0.7)


def test_threshold_ties_go_to_the_higher_threshold() -> None:
    y = np.array([True, False, False, True, False, False])
    s = np.array([0.9, 0.8, 0.8, 0.6, 0.2, 0.2])
    # F1 at 0.9: 2/3, at 0.8: 2/5, at 0.6: 2/3, at 0.2: 1/2; 0.9 and 0.6 tie
    assert choose_threshold(y, s) == pytest.approx(0.9)


def test_evaluate_reports_every_metric_with_an_enclosing_interval() -> None:
    y, s = random_case(1)
    estimates = evaluate(y, s, threshold=0.5, probabilities=True, resamples=300)
    assert set(estimates) == {
        "f1",
        "precision",
        "recall",
        "mcc",
        "balanced_accuracy",
        "average_precision",
        "roc_auc",
        "recall_at_10pct_review",
        "recall_at_20pct_review",
        "brier",
        "expected_calibration_error",
    }
    for estimate in estimates.values():
        assert estimate.low <= estimate.value <= estimate.high
    assert estimates["f1"].value == pytest.approx(sk.f1_score(y, s >= 0.5))


def test_scores_are_only_calibrated_when_they_are_probabilities() -> None:
    y, s = random_case(2)
    assert "brier" not in evaluate(y, s * 10, threshold=5.0, resamples=10)
    with pytest.raises(ValueError, match="probabilities in"):
        evaluate(y, s * 10, threshold=5.0, probabilities=True, resamples=10)


def test_identical_systems_do_not_differ() -> None:
    y, s = random_case(3)
    differences = compare(y, s, s, 0.5, 0.5, resamples=200)
    assert all((d.value, d.low, d.high) == (0.0, 0.0, 0.0) for d in differences.values())


def test_a_perfect_candidate_beats_a_random_baseline() -> None:
    y, _ = random_case(4)
    differences = compare(y, np.random.default_rng(0).random(y.size), y * 1.0, 0.5, 0.5)
    assert differences["f1"].excludes_zero
    assert differences["roc_auc"].value > 0


@pytest.mark.parametrize(
    ("labels", "scores", "message"),
    [
        ([True, False], [0.1], "one entry per document"),
        ([], [], "one entry per document"),
        ([True, False], [0.1, float("nan")], "finite"),
        ([True, True], [0.1, 0.2], "both classes"),
    ],
)
def test_invalid_inputs_are_rejected(labels: list[bool], scores: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        as_arrays(labels, scores)
