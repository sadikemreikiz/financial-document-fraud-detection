"""The exact McNemar test for two systems judged right or wrong on the same documents.

Only the documents on which the systems disagree carry information. Under the hypothesis that
both systems are equally good, each of them is the better one on such a document with
probability one half, so the count of one side is binomial (ADR-0025, decision 5).
"""

from math import comb


def exact_mcnemar(first_only: int, second_only: int) -> float:
    """Two-sided p-value of the exact McNemar test.

    Args:
        first_only: Documents on which only the first system is right.
        second_only: Documents on which only the second system is right.

    Raises:
        ValueError: If a count is negative.
    """
    if first_only < 0 or second_only < 0:
        msg = "counts must not be negative"
        raise ValueError(msg)
    n = first_only + second_only
    if n == 0:
        return 1.0
    tail = sum(comb(n, k) for k in range(max(first_only, second_only), n + 1))
    return min(1.0, 2 * tail / (1 << n))
