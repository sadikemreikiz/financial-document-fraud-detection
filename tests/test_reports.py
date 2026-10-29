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
from docfraud.cli.final import RENDERED
from docfraud.detectors.copy_move_measurement import RUN_NAME as COPY_MOVE_RUN
from docfraud.detectors.copy_move_measurement import render_report as render_copy_move
from docfraud.detectors.report import render_rules_report
from docfraud.evaluation.runs import JsonRunStore, RunRecord
from docfraud.fusion.development_report import render_report as render_training
from docfraud.fusion.shift_report import render_report as render_shift
from docfraud.fusion.validation_report import render_report as render_validation
from docfraud.llm.agent_validation_report import render_report as render_agent
from docfraud.llm.detector_validation_report import render_report as render_detector

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


def test_copy_move_report_matches_its_run_record() -> None:
    record = latest(COPY_MOVE_RUN)
    expected = render_copy_move(record.results, record.code.commit)
    committed = (ROOT / "reports" / "fusion" / "copy-move-clean.md").read_text(encoding="utf-8")
    assert committed == expected, "run `docfraud copy-move report`"


def test_training_report_matches_its_run_record() -> None:
    record = latest("fusion-training-oof")
    expected = render_training(record.results, record.code.commit)
    committed = (ROOT / "reports" / "fusion" / "training-oof.md").read_text(encoding="utf-8")
    assert committed == expected, "run `docfraud fusion training-report`"


def test_validation_report_matches_its_run_record() -> None:
    record = latest("fusion-validation-check")
    expected = render_validation(record.results, record.code.commit)
    committed = (ROOT / "reports" / "fusion" / "validation-check.md").read_text(encoding="utf-8")
    assert committed == expected, "run `docfraud fusion validation-report`"


def test_detector_validation_report_matches_its_run_record() -> None:
    record = latest("llm-detector-validation")
    expected = render_detector(record.results, record.code.commit)
    committed = (ROOT / "reports" / "llm" / "detector" / "validation.md").read_text("utf-8")
    assert committed == expected, "run `docfraud detector validation-report`"


def test_agent_validation_report_matches_its_run_record() -> None:
    record = latest("llm-agent-validation")
    expected = render_agent(record.results, record.config, record.code.commit)
    committed = (ROOT / "reports" / "llm" / "agent" / "validation.md").read_text("utf-8")
    assert committed == expected, "run `docfraud agent validation-report`"


def test_input_shift_report_matches_its_run_record() -> None:
    record = latest("fusion-input-shift")
    expected = render_shift(record.results, record.code.commit)
    committed = (ROOT / "reports" / "fusion" / "input-shift.md").read_text(encoding="utf-8")
    assert committed == expected, "run `docfraud fusion input-shift-report`"


@pytest.mark.parametrize(("name", "path"), [(name, path) for name, (path, _) in RENDERED.items()])
def test_phase_7_report_matches_its_run_record(name: str, path: str) -> None:
    _, render = RENDERED[name]
    committed = (ROOT / "reports" / path).read_text(encoding="utf-8")
    assert committed == render(RUNS, latest(name)), "run `docfraud final reports`"


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
