import numpy as np
import pytest
from scipy import stats

from docfraud.fusion.inputs import PRACTICAL, SYSTEM_1_INPUT
from docfraud.fusion.shift import (
    VARIES_BY,
    input_shift,
    ks_distance,
    permutation_test,
    summary,
)
from docfraud.fusion.shift_report import render_report
from docfraud.fusion.training_inputs import InputsTable
from tests.fusion.test_validation import N_TRAIN, N_VALIDATION, SEEDS, tables


def test_the_ks_distance_is_scipys_and_the_difference_of_shares_for_0_or_1() -> None:
    rng = np.random.default_rng(0)
    a, b = rng.normal(size=80), rng.normal(0.3, size=50)
    assert ks_distance(a, b) == pytest.approx(stats.ks_2samp(a, b).statistic)
    ones, other = rng.random(90) < 0.3, rng.random(40) < 0.6
    assert ks_distance(ones.astype(float), other.astype(float)) == pytest.approx(
        abs(ones.mean() - other.mean())
    )
    assert ks_distance(a, a.copy()) == 0.0


def test_the_permutation_test_reads_its_distance_as_ks_does_also_with_ties() -> None:
    rng = np.random.default_rng(1)
    for _ in range(20):
        a = np.round(rng.exponential(size=60), 1) * (rng.random(60) < 0.5)
        b = np.round(rng.exponential(1.3, size=25), 1) * (rng.random(25) < 0.4)
        assert permutation_test(a, b, 50).distance == pytest.approx(ks_distance(a, b))


def test_the_permutation_p_value_is_small_for_a_shift_and_one_without() -> None:
    rng = np.random.default_rng(2)
    a, b = rng.normal(size=200), rng.normal(size=100)
    clear = permutation_test(a, b + 2, 999)
    assert clear.p_value == pytest.approx(1 / 1000)
    same = permutation_test(a, a[:100].copy(), 999)
    assert 0.05 < same.p_value <= 1
    assert permutation_test(np.zeros(30), np.zeros(10), 99).p_value == 1.0
    asymptotic = 1.36 * np.sqrt((200 + 100) / (200 * 100))
    assert clear.critical == pytest.approx(asymptotic, rel=0.2)
    assert permutation_test(a, b, 999, seed=3) == permutation_test(a, b, 999, seed=3)


def test_every_input_is_compared_at_the_level_it_varies_on() -> None:
    training, _, validation, _ = tables()
    found = input_shift(training, validation, seeds=SEEDS, permutations=99)
    scopes = [(c["input"], c["track"], c["seed"]) for c in found["comparisons"]]
    assert len(scopes) == 21
    assert [s for s in scopes if s[0] == "rules"] == [
        ("rules", "transcript", None),
        ("rules", "ocr", None),
    ]
    assert [s for s in scopes if s[0] == "system 2"] == [("system 2", None, s) for s in SEEDS]
    assert len([s for s in scopes if s[0] == "link 1"]) == 6
    assert {s[0] for s in scopes} == set(PRACTICAL) == set(VARIES_BY)
    assert found["receipts"] == {"training": N_TRAIN, "validation": N_VALIDATION}
    assert found["below_0_05"] == sum(c["p_value"] < 0.05 for c in found["comparisons"])
    rules = found["comparisons"][0]
    assert set(rules["training"]) == {"mean", "zeros", "median", "q90"}
    assert rules["distance"] > rules["critical"]  # the toy scorers differ by construction


def test_an_input_that_should_not_vary_must_not() -> None:
    training, _, validation, _ = tables()
    rows = [
        r.model_copy(update={"inputs": r.inputs | {SYSTEM_1_INPUT: 0.5}}) if r.track == "ocr" else r
        for r in validation.receipts
    ]
    with pytest.raises(ValueError, match="system 1 differs"):
        input_shift(training, InputsTable(receipts=tuple(rows)), seeds=SEEDS, permutations=9)
    shuffled = [r for r in validation.receipts if r.seed != 2] + list(
        reversed([r for r in validation.receipts if r.seed == 2])
    )
    with pytest.raises(ValueError, match="hold different receipts"):
        input_shift(training, InputsTable(receipts=tuple(shuffled)), seeds=SEEDS, permutations=9)
    with pytest.raises(ValueError, match="no rows for transcript, seed 3"):
        input_shift(training, validation, seeds=(*SEEDS, 3), permutations=9)


def test_a_summary_states_the_mean_zeros_median_and_90th_percentile() -> None:
    assert summary(np.array([0.0, 0.0, 1.0, 3.0])) == pytest.approx(
        {"mean": 1.0, "zeros": 0.5, "median": 0.5, "q90": 2.4}
    )


def test_the_report_states_every_comparison_and_the_chance_count() -> None:
    training, _, validation, _ = tables()
    text = render_report(
        input_shift(training, validation, seeds=SEEDS, permutations=99), "abc1234x"
    )
    assert "Every fusion input, training → validation" in text
    assert "| rules | T1 (transcripts) | all |" in text
    assert "| system 2 | both | 1 |" in text
    assert "of 21 comparisons have p < 0.05; about 1 would by chance" in text
    assert "`abc1234`" in text
