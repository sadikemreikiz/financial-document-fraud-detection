from pathlib import Path

import numpy as np
import pytest

from docfraud.detectors.evaluation import cross_validated, flag_all, held_out, rule_counts
from docfraud.detectors.report import render_rules_report
from docfraud.detectors.rules import RULES, Outcome, Status
from docfraud.detectors.scoring import FEATURES, Scorer, any_fail, features, out_of_fold
from docfraud.evaluation.runs import JsonRunStore, run

PASS, FAIL, NA = Outcome(Status.PASS), Outcome(Status.FAIL), Outcome.na("")


def outcomes(failing: str | None, missing: str | None = None) -> dict[str, Outcome]:
    return {r: FAIL if r == failing else NA if r == missing else PASS for r in RULES}


def dataset(n: int = 60) -> tuple[list[dict[str, Outcome]], np.ndarray, np.ndarray]:
    """Forged receipts fail change_given; authentic ones mostly pass, a few miss a total."""
    y = np.arange(n) % 5 == 0
    outs = [outcomes("change_given" if f else None, "totals_reconcile" if i % 7 == 0 else None)
            for i, f in enumerate(y)]  # fmt: skip
    return outs, y, np.arange(n) % 3  # every fold holds forged receipts


def test_features_have_one_column_per_rule_and_non_passing_outcome() -> None:
    x = features([outcomes("line_amounts", "tax_rates")])
    assert x.shape == (1, 2 * len(RULES)) == (1, len(FEATURES))
    assert dict(zip(FEATURES, x[0], strict=True))["line_amounts:fail"] == 1.0
    assert dict(zip(FEATURES, x[0], strict=True))["tax_rates:not_applicable"] == 1.0
    assert x.sum() == 2.0
    assert features([]).shape == (0, len(FEATURES))


def test_any_fail_is_a_binary_score() -> None:
    assert any_fail([outcomes(None), outcomes("tax_rates")]).tolist() == [0.0, 1.0]


def test_the_scorer_learns_which_failure_matters() -> None:
    outs, y, folds = dataset()
    scorer = Scorer.fit(features(outs), y)
    assert scorer.weights()["change_given:fail"] > 1
    oof = out_of_fold(features(outs), y, folds)
    assert ((oof > 0.5) == y).all()


def test_rule_counts_by_label() -> None:
    outs, y, _ = dataset(10)
    counts = rule_counts(outs, y)
    assert counts["change_given"]["forged"] == {"applicable": 2, "fail": 2}
    assert counts["totals_reconcile"]["authentic"] == {"applicable": 7, "fail": 0}  # 7 of 8


def test_flagging_everything_is_the_f1_floor() -> None:
    y = np.array([True] + [False] * 5)
    result = flag_all(y, [str(i) for i in range(6)])
    assert result["metrics"]["f1"]["value"] == pytest.approx(2 / 7)
    assert result["metrics"]["roc_auc"]["value"] == pytest.approx(0.5)


def test_cross_validated_and_held_out_results() -> None:
    outs, y, folds = dataset()
    ids = [str(i) for i in range(len(y))]
    cv = cross_validated(outs, y, folds, ids)
    assert set(cv) == {"threshold", "learned", "any_fail", "flag_all", "weights", "rules"}
    assert cv["learned"]["metrics"]["f1"]["value"] == pytest.approx(1.0)
    assert "brier" in cv["learned"]["metrics"]
    held = held_out(outs, y, folds, outs[:20], y[:20], ids[:20])
    assert held["learned"]["n_documents"] == 20
    assert held["threshold"] == pytest.approx(cv["threshold"])


def test_report_renders_before_and_after_validation(tmp_path: Path) -> None:
    outs, y, folds = dataset()
    ids = [str(i) for i in range(len(y))]
    store = JsonRunStore(tmp_path)
    compared = ["f1", "average_precision", "roc_auc"]
    compared += ["recall_at_10pct_review", "recall_at_20pct_review"]
    versus = {k: {"value": 0.1, "low": 0.0, "high": 0.2, "excludes_zero": True} for k in compared}
    track = cross_validated(outs, y, folds, ids) | {"versus_bow": versus}
    gold = {"any_fail": flag_all(y, ids), "rules": rule_counts(outs, y)}
    with run("rules-cv", store=store, config={}, seeds={}, dataset_sha256={}) as cv:
        cv.log("transcript", track)
        cv.log("ocr", track)
        cv.log("gold_fields", gold)
    with run("bow-lr-cv", store=store, config={}, seeds={}, dataset_sha256={}) as bow:
        for key, value in track["learned"].items():
            bow.log(key, value)
    assert cv.record is not None
    assert bow.record is not None
    before = render_rules_report(cv.record, bow.record, None)
    assert "Not measured yet" in before
    assert "| Any rule fails | 48 / 48 | 12 / 12 |" in before
    assert "+10.0 [+0.0, +20.0] *" in before
    with run("rules-validation", store=store, config={}, seeds={}, dataset_sha256={}) as val:
        val.log("bow", track["learned"])
        val.log("transcript", track)
        val.log("ocr", track)
    assert val.record is not None
    after = render_rules_report(cv.record, bow.record, val.record)
    assert "Validation split, measured once (unbiased)" in after
