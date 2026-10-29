"""Which receipts the demo shows: a stated rule over the test split's outcomes (ADR-0035, 1).

Each receipt has an outcome under the review policy: a forged receipt sent to review is caught,
one accepted is missed, a genuine receipt accepted is genuine, and one sent to review is a false
alarm. Within each outcome the receipts are ranked by calibrated probability, highest first, ties
broken by identifier, and evenly spaced ranks are taken with both ends included (rounded half to
even, as Python rounds), so that the hardest cases are shown by construction and no image is
looked at to choose.
"""

from collections.abc import Iterable, Mapping

CAUGHT = "caught"
MISSED = "missed"
GENUINE = "genuine"
FALSE_ALARM = "false alarm"

OUTCOMES = (CAUGHT, MISSED, GENUINE, FALSE_ALARM)

SHOWN: dict[str, int] = {CAUGHT: 4, MISSED: 3, GENUINE: 3}
"""How many receipts of each outcome are shown. The test split has no false alarm to show."""


def outcome(forged: bool, reviewed: bool) -> str:
    """The outcome of one receipt under the review policy."""
    if forged:
        return CAUGHT if reviewed else MISSED
    return FALSE_ALARM if reviewed else GENUINE


def evenly_spaced(size: int, count: int) -> list[int]:
    """``count`` ranks among ``size``, spread evenly, the first and the last included.

    Raises:
        ValueError: If fewer than two ranks are asked for, or more than there are.
    """
    if not 2 <= count <= size:  # noqa: PLR2004 - both ends need two ranks
        msg = f"cannot spread {count} ranks over {size} receipts with both ends included"
        raise ValueError(msg)
    return [round(i * (size - 1) / (count - 1)) for i in range(count)]


def ranked(probabilities: Mapping[str, float], receipts: Iterable[str]) -> list[str]:
    """Receipts by calibrated probability, highest first, ties broken by identifier."""
    return sorted(receipts, key=lambda receipt: (-probabilities[receipt], receipt))


def outcomes(
    probabilities: Mapping[str, float],
    forged: Mapping[str, bool],
    reviewed: Mapping[str, bool],
) -> dict[str, list[str]]:
    """Every receipt in its outcome, each outcome ranked."""
    found: dict[str, list[str]] = {name: [] for name in OUTCOMES}
    for receipt in probabilities:
        found[outcome(forged[receipt], reviewed[receipt])].append(receipt)
    return {name: ranked(probabilities, members) for name, members in found.items()}


def choose(ranked_outcomes: Mapping[str, list[str]]) -> dict[str, list[int]]:
    """The ranks shown in each outcome that the gallery shows."""
    return {name: evenly_spaced(len(ranked_outcomes[name]), count) for name, count in SHOWN.items()}
