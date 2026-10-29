"""The demo's commands, on made-up records."""

from typing import Any

from docfraud.cli.demo import disagreement
from docfraud.demo.gallery import CAUGHT, FALSE_ALARM, GENUINE, MISSED


def _results(*, caught: int = 2, missed: tuple[tuple[str, float], ...] = ()) -> dict[str, Any]:
    return {
        "outcomes": {CAUGHT: caught, MISSED: len(missed), GENUINE: 5, FALSE_ALARM: 0},
        "gallery": {MISSED: [{"receipt": r, "probability": p} for r, p in missed]},
    }


RECORDED = {
    "reviewed": 2,
    "errors": [
        {"receipt": "M1", "kind": "miss", "probability": 0.3},
        {"receipt": "M2", "kind": "miss", "probability": 0.2},
    ],
}


def test_the_cards_agree_with_the_error_analysis() -> None:
    assert disagreement(_results(missed=(("M1", 0.3), ("M2", 0.2))), RECORDED) is None


def test_a_difference_from_the_error_analysis_is_named() -> None:
    assert "sent to review" in str(disagreement(_results(caught=3), RECORDED))
    assert "misses differ" in str(disagreement(_results(missed=(("M1", 0.3),)), RECORDED))
    found = disagreement(_results(missed=(("M1", 0.3), ("M2", 0.25))), RECORDED)
    assert "a miss shown" in str(found)
