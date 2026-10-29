"""The reports of Phase 7, rendered from made-up run records of every test run."""

import json
from pathlib import Path
from typing import Any

import pytest
from typer.testing import CliRunner

from docfraud.cli import app
from docfraud.data.models import BBox, Label
from docfraud.decision import Decision
from docfraud.evaluation import runs as runs_module
from docfraud.evaluation.runs import CodeVersion, JsonRunStore, run
from docfraud.forensics.final import map_results, measured
from docfraud.fusion.errors import error_analysis
from docfraud.fusion.final import frozen_check, resaved_check, subset
from docfraud.fusion.validation import validation_check
from docfraud.llm.agent import SONNET_AGENT
from docfraud.llm.agent_rounds import FrozenAgent
from docfraud.llm.detector_final import CLAIMS, DESCRIPTIVE, detector_test, part_rows
from tests.forensics.test_final import WORDS, Spot, item
from tests.fusion.test_errors import POLICY, decided
from tests.fusion.test_validation import SEEDS, labels, pages, tables
from tests.llm.test_agent_rounds import document
from tests.llm.test_agent_tools import BOX, assessment
from tests.llm.test_agent_validation import results as agent_results
from tests.llm.test_detector_final import _validation_run

RESAMPLES = 20
runner = CliRunner()


def _fusion() -> tuple[dict[str, Any], dict[str, Any]]:
    """The test check and track R on the made-up tables, as their records keep them."""
    training, y_train, validation, documents = tables()
    check = json.loads(
        json.dumps(
            validation_check(
                training,
                y_train,
                validation,
                documents,
                pages(documents),
                seeds=SEEDS,
                resamples=RESAMPLES,
            )
        )
    )
    arguments: dict[str, Any] = {"recorded": check, "seeds": SEEDS, "resamples": RESAMPLES}
    found = frozen_check(training, y_train, validation, documents, pages(documents), **arguments)
    kept_table, kept = subset(validation, documents, frozenset({"V00"}))
    found["sensitivity"] = frozen_check(
        training, y_train, kept_table, kept, pages(kept), **arguments
    )
    track_r = resaved_check(
        training,
        y_train,
        validation,
        {95: validation, 75: validation},
        labels(documents),
        recorded=check["ocr"],
        resamples=RESAMPLES,
    )
    return found, track_r


def _localisation() -> dict[str, Any]:
    receipts = measured(Spot(), [item("R1", forged=True), item("R2", forged=False)], 0.5)
    words = {r.key: list(WORDS) for r in receipts}
    rate = {"value": 0.5, "low": 0.2, "high": 0.8}
    guard = {"receipts": 4, "hit_rate": rate, "baseline": 0.1, "detects": True}
    sourced = {"receipts": 3, "rate": rate, "baseline": 0.05, "above_baseline": True}
    found = map_results(receipts, words, 0.5, resamples=RESAMPLES)
    return {
        "2": found,
        "1": found,
        "copy_move": {
            "guard": {"all": guard},
            "copy_sources": {"either_region": sourced, "pair": sourced},
        },
    }


def _errors() -> dict[str, Any]:
    documents = [
        document("R1", Label.FORGED, BOX),
        document("R2", Label.FORGED, BBox(x=60, y=40, width=10, height=10)),
        document("R3", Label.AUTHENTIC),
    ]
    accepted = decided(assessment(), Decision.ACCEPT)
    reviewed = decided(assessment(), Decision.REVIEW)
    assessed = {"R1": accepted, "R2": accepted, "R3": reviewed}
    return error_analysis(documents, assessed, POLICY)


def _record(store: JsonRunStore, name: str, results: dict[str, Any], **config: Any) -> None:
    with run(name, store=store, config=config, seeds={}, dataset_sha256={}) as current:
        for key, value in json.loads(json.dumps(results)).items():
            current.log(key, value)


def test_every_report_of_phase_7_renders_from_its_record(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    clean = CodeVersion(commit="f" * 40, dirty=False)
    monkeypatch.setattr(runs_module, "code_version", lambda repo=None: clean)  # noqa: ARG005
    store = JsonRunStore(tmp_path / "runs")
    command = ["final", "reports", "--runs", str(tmp_path / "runs")]
    command += ["--output", str(tmp_path / "reports")]
    nothing = runner.invoke(app, command)
    assert nothing.exit_code == 0
    assert "no fusion-test-check record yet" in nothing.output
    fusion, track_r = _fusion()
    _record(store, "fusion-test-check", fusion)
    _record(store, "fusion-resaved-inputs", {}, fix="phase-7-fix-1")
    (inputs,) = store.records("fusion-resaved-inputs")
    _record(store, "fusion-resaved-check", track_r, resaved_inputs=inputs.run_id)
    _record(store, "fusion-test-errors", _errors())
    _record(store, "forensics-test-localisation", _localisation())
    given, recorded = _validation_run()
    for part, name in (
        (CLAIMS, "llm-detector-test"),
        (DESCRIPTIVE, "llm-detector-test-descriptive"),
    ):
        rows = part_rows(given["rows"], part)
        found = detector_test(**(given | {"rows": rows}), recorded=recorded, resamples=RESAMPLES)
        _record(store, name, found)
    frozen = FrozenAgent(
        version=2,
        settings=SONNET_AGENT,
        token_budget=34_000,
        instruction="agent-v2.md",
        instruction_sha256="0" * 64,
        commit="b" * 40,
    )
    agent = agent_results(tmp_path) | {"policy": {"budget": 0.1, "threshold": 0.7}}
    _record(store, "llm-agent-test", agent, frozen=frozen.model_dump(mode="json"))
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    reports = tmp_path / "reports"
    test_check = (reports / "fusion" / "test-check.md").read_text("utf-8")
    assert test_check.startswith("# The frozen system on the test split\n")
    assert "ADR-0013's success criterion, test" in test_check
    assert "### The pre-named tests, T2 (OCR)" in test_check
    assert "## Without the eight receipts that had copies" in test_check
    assert "#### ADR-0013's success criterion, test without copies" in test_check
    track = (reports / "fusion" / "track-r.md").read_text("utf-8")
    assert "saved as JPEG at quality 95" in track
    assert "| quality 75 | 0.000 (0.000 to 0.000) |" in track  # the same inputs cost nothing
    assert "made after the one fix of Phase 7, at the tag `phase-7-fix-1`" in track
    errors = (reports / "fusion" / "test-errors.md").read_text("utf-8")
    assert "| R3 | false alarm | pixel evidence |" in errors
    assert "found at the edit, too weak 1, found elsewhere 1" in errors
    localisation = (reports / "forensics" / "test-localisation.md").read_text("utf-8")
    assert "| system 2, synthetic forgeries | 0.500 | 1 |" in localisation
    detector = (reports / "llm" / "detector" / "test.md").read_text("utf-8")
    assert "**RQ6, the ninth pre-named test.**" in detector
    assert "**The paper's result.**" in detector
    assert "image and OCR text, claude-opus-5-5" in detector  # the descriptive part
    agent_report = (reports / "llm" / "agent" / "test.md").read_text("utf-8")
    assert agent_report.startswith("# The investigation agent on the test split\n")
    assert "Phase 5's check, applied unchanged" in agent_report
    assert "### The agent on the test split" in agent_report
    assert "[`test/`](test/)" in agent_report
