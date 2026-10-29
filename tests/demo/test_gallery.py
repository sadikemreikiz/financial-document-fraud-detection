"""The rule that chooses the demo's receipts, on made-up probabilities."""

import pytest

from docfraud.demo.gallery import (
    CAUGHT,
    FALSE_ALARM,
    GENUINE,
    MISSED,
    choose,
    evenly_spaced,
    outcome,
    outcomes,
    ranked,
)


def test_each_receipt_has_one_outcome_under_the_policy() -> None:
    assert outcome(forged=True, reviewed=True) == CAUGHT
    assert outcome(forged=True, reviewed=False) == MISSED
    assert outcome(forged=False, reviewed=False) == GENUINE
    assert outcome(forged=False, reviewed=True) == FALSE_ALARM


@pytest.mark.parametrize(
    ("size", "count", "ranks"),
    [(24, 4, [0, 8, 15, 23]), (11, 3, [0, 5, 10]), (183, 3, [0, 91, 182]), (2, 2, [0, 1])],
)
def test_ranks_are_spread_evenly_with_both_ends(size: int, count: int, ranks: list[int]) -> None:
    assert evenly_spaced(size, count) == ranks


@pytest.mark.parametrize(("size", "count"), [(5, 1), (2, 3), (0, 2)])
def test_too_few_or_too_many_ranks_are_refused(size: int, count: int) -> None:
    with pytest.raises(ValueError, match="both ends"):
        evenly_spaced(size, count)


def test_receipts_rank_by_probability_and_ties_by_identifier() -> None:
    probabilities = {"B": 0.5, "A": 0.5, "C": 0.9, "D": 0.1}
    assert ranked(probabilities, probabilities) == ["C", "A", "B", "D"]


def test_the_gallery_takes_both_ends_of_each_outcome_it_shows() -> None:
    probabilities = {f"F{i}": 0.99 - i / 100 for i in range(6)} | {
        f"G{i}": 0.5 - i / 100 for i in range(5)
    }
    forged = {r: r.startswith("F") for r in probabilities}
    reviewed = {r: probabilities[r] > 0.955 for r in probabilities}
    found = outcomes(probabilities, forged, reviewed)
    assert found[CAUGHT] == ["F0", "F1", "F2", "F3"]
    assert found[MISSED] == ["F4", "F5"]
    assert found[FALSE_ALARM] == []
    with pytest.raises(ValueError, match="both ends"):
        choose(found)  # three misses are shown, and there are two
    found[MISSED].append("F6")
    assert choose(found) == {CAUGHT: [0, 1, 2, 3], MISSED: [0, 1, 2], GENUINE: [0, 2, 4]}
