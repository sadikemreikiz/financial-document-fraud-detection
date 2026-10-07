"""Committed reports must match what their committed sources render today.

These need no dataset: the baseline and rules reports render from the run records in
reports/runs, the gold verification reports from the recorded owner reviews. Reports that need
the dataset are checked with `make check-reports`.
"""

from pathlib import Path

import pytest
from typer.testing import CliRunner

from docfraud.baselines.report import render_report
from docfraud.cli import app
from docfraud.detectors.report import render_rules_report
from docfraud.evaluation.runs import JsonRunStore, RunRecord

ROOT = Path(__file__).resolve().parents[1]
RUNS = JsonRunStore(ROOT / "reports" / "runs")
ANNOTATIONS = ROOT / "annotations" / "findit2"


def latest(name: str) -> RunRecord:
    records = RUNS.records(name)
    assert records, f"no {name} run in reports/runs"
    return records[-1]


def test_baseline_report_matches_its_run_records() -> None:
    expected = render_report(latest("bow-lr-cv"), latest("bow-lr-test"))
    committed = (ROOT / "reports" / "baselines" / "bow-lr.md").read_text(encoding="utf-8")
    assert committed == expected, "run `docfraud baseline report`"


def test_rules_report_matches_its_run_records() -> None:
    expected = render_rules_report(
        latest("rules-cv"),
        latest("bow-lr-cv"),
        latest("rules-validation"),
        latest("rules-breakdown"),
    )
    committed = (ROOT / "reports" / "rules" / "accounting-rules.md").read_text(encoding="utf-8")
    assert committed == expected, "run `docfraud rules report`"


@pytest.mark.parametrize(
    ("review", "options", "report"),
    [
        ("gold-review.json", [], "gold-verification.md"),
        (
            "gold-review-b.json",
            [
                "--manifest",
                str(ANNOTATIONS / "gold-fields-b" / "manifest.json"),
                "--authentic",
                "4",
                "--forged",
                "1",
                "--resolutions",
                str(ANNOTATIONS / "gold-review-b-resolutions.json"),
            ],
            "gold-verification-b.md",
        ),
    ],
)
def test_gold_verification_reports_match_the_recorded_reviews(
    review: str, options: list[str], report: str, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.chdir(ROOT)
    output = tmp_path / report
    args = [
        "reading",
        "score-review",
        str(ANNOTATIONS / review),
        "--record",
        str(ANNOTATIONS / review),
    ]
    result = CliRunner().invoke(app, [*args, *options, "--output", str(output)])
    assert result.exit_code == 0, result.output
    committed = (ROOT / "reports" / "reading" / report).read_text(encoding="utf-8")
    assert committed == output.read_text(encoding="utf-8"), "run the score-review command"
