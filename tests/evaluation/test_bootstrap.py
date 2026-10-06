import pytest

from docfraud.evaluation.bootstrap import Estimate, paired_ratio_difference, ratio


def test_ratio_is_a_ratio_of_sums_with_an_enclosing_interval() -> None:
    estimate = ratio([9, 10, 18], [10, 10, 20], resamples=2000)
    assert estimate.value == pytest.approx(37 / 40)
    assert estimate.low <= estimate.value <= estimate.high
    assert estimate.level == 0.95


def test_ratio_is_deterministic_for_a_seed() -> None:
    args = ([1, 2, 3, 4], [4, 4, 4, 4])
    assert ratio(*args, seed=3) == ratio(*args, seed=3)


def test_identical_systems_differ_by_exactly_zero() -> None:
    diff = paired_ratio_difference([5, 6, 7], [5, 6, 7], [10, 10, 10], resamples=500)
    assert (diff.value, diff.low, diff.high) == (0.0, 0.0, 0.0)
    assert not diff.excludes_zero


def test_a_consistently_better_system_is_significant() -> None:
    worse = [5] * 30
    better = [9] * 30
    diff = paired_ratio_difference(worse, better, [10] * 30, resamples=1000)
    assert diff.value == pytest.approx(0.4)
    assert diff.excludes_zero


def test_documents_without_items_are_tolerated() -> None:
    estimate = ratio([0, 3], [0, 4], resamples=1000)
    assert estimate.value == pytest.approx(0.75)


@pytest.mark.parametrize(
    ("numerator", "denominator", "message"),
    [
        ([1, 2], [3], "one entry per document"),
        ([], [], "one entry per document"),
        ([0], [0], "positive"),
    ],
)
def test_invalid_input(numerator: list[float], denominator: list[float], message: str) -> None:
    with pytest.raises(ValueError, match=message):
        ratio(numerator, denominator)


def test_excludes_zero() -> None:
    assert Estimate(0.1, 0.01, 0.2).excludes_zero
    assert Estimate(-0.1, -0.2, -0.01).excludes_zero
    assert not Estimate(0.0, -0.1, 0.1).excludes_zero
