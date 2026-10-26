import numpy as np
import pytest

from docfraud.decision import BUDGETS, Decision, ReviewPolicy, outcome


def test_the_threshold_reviews_the_budget_of_the_receipts_it_is_set_on() -> None:
    p = np.random.default_rng(0).random(190)
    policy = ReviewPolicy.set_on(p)
    assert policy.budget == BUDGETS[0]
    assert policy.threshold == np.sort(p)[::-1][18]  # the 19th highest
    assert policy.reviewed(p).sum() == 19
    assert ReviewPolicy.set_on(p, BUDGETS[1]).reviewed(p).sum() == 38
    assert policy.decide(policy.threshold) is Decision.REVIEW
    assert policy.decide(policy.threshold - 1e-9) is Decision.ACCEPT


def test_a_budget_must_review_some_but_not_every_receipt() -> None:
    with pytest.raises(ValueError, match="reviews 0"):
        ReviewPolicy.set_on([0.1, 0.2, 0.3], 0.1)
    with pytest.raises(ValueError, match="reviews 3"):
        ReviewPolicy.set_on([0.1, 0.2, 0.3], 1.0)


def test_the_outcome_counts_the_reviewed_receipts() -> None:
    policy = ReviewPolicy(budget=0.1, threshold=0.5)
    found = outcome(policy, [0.9, 0.6, 0.4, 0.2, 0.7], [True, False, True, False, True])
    assert (found.receipts, found.reviewed, found.forged, found.forged_reviewed) == (5, 3, 3, 2)
    assert found.review_share == pytest.approx(0.6)
    assert found.recall == pytest.approx(2 / 3)
    assert found.precision == pytest.approx(2 / 3)
    nothing = outcome(policy, [0.1, 0.2], [False, False])
    assert (nothing.recall, nothing.precision) == (0.0, 0.0)
