import pytest

from docfraud.evaluation.mcnemar import exact_mcnemar


@pytest.mark.parametrize(
    ("first", "second", "p"),
    [
        (0, 0, 1.0),
        (3, 3, 1.0),
        (5, 0, 0.0625),
        (6, 0, 0.03125),
        (8, 1, 0.0390625),
        (1, 8, 0.0390625),
    ],
)
def test_exact_two_sided_p_values(first: int, second: int, p: float) -> None:
    assert exact_mcnemar(first, second) == pytest.approx(p)


def test_counts_cannot_be_negative() -> None:
    with pytest.raises(ValueError, match="negative"):
        exact_mcnemar(-1, 2)
