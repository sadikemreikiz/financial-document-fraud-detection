import hashlib
import json
import shutil
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from decimal import Decimal
from functools import partial
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
import typer
from PIL import Image
from typer.testing import CliRunner

from docfraud import __version__
from docfraud.cli import agent as cli_agent
from docfraud.cli import app
from docfraud.cli import baseline as cli_baseline
from docfraud.cli import common as cli_common
from docfraud.cli import copy_move as cli_copy_move
from docfraud.cli import data as cli_data
from docfraud.cli import detector as cli_detector
from docfraud.cli import forensics as cli_forensics
from docfraud.cli import fusion as cli_fusion
from docfraud.cli import llm as cli_llm
from docfraud.cli import marks as cli_marks
from docfraud.cli import observation as cli_observation
from docfraud.cli import reading as cli_reading
from docfraud.cli import rules as cli_rules
from docfraud.cli import synthesis as cli_synthesis
from docfraud.data.benign_marks import ReceiptMarks, load_mark_log, marks_plan, round_path
from docfraud.data.download import SOURCES, PinnedFile
from docfraud.data.gold import GoldManifest, load_manifest
from docfraud.data.models import Document, Entity, Label, Split, Technique
from docfraud.data.observation import (
    ASSISTANT,
    ObservationLog,
    ReceiptObservation,
    Visibility,
    save_log,
    seal,
)
from docfraud.data.splits import TestSetLockedError
from docfraud.evaluation import runs as runs_module
from docfraud.evaluation.bootstrap import Estimate
from docfraud.evaluation.design_fold import CLEAN_FOLDS, DESIGN_FOLD
from docfraud.evaluation.folds import assign_folds, load_folds, save_folds
from docfraud.evaluation.runs import CodeVersion, JsonRunStore, run
from docfraud.forensics.learned.pretrained import IMAGENET_WEIGHTS
from docfraud.forensics.trufor import TRUFOR_CODE, TRUFOR_WEIGHTS
from docfraud.fusion.training_inputs import InputsTable, load_table, save_table, summary
from docfraud.fusion.validation import validation_check
from docfraud.llm.adjudication import (
    AdjudicationLog,
    Listing,
    first_answer,
    load_questions,
)
from docfraud.llm.adjudication import save_log as save_adjudication
from docfraud.llm.agent import SONNET_AGENT
from docfraud.llm.agent_rounds import FrozenAgent
from docfraud.llm.agent_rounds import load_rounds as load_agent_rounds
from docfraud.llm.agent_tools import ReadField
from docfraud.llm.agent_validation import agent_validation
from docfraud.llm.cost import LIMIT_USD, Ledger
from docfraud.llm.decision import ModelStep
from docfraud.llm.detector import SONNET
from docfraud.llm.detector_rounds import OPUS, FrozenDetector, StrongerModelCheck, load_rounds
from docfraud.llm.detector_validation import detector_validation, validation_rows
from docfraud.llm.false_alarms import FIXED_AT
from docfraud.llm.frozen import load_split_reading
from docfraud.llm.gold_b import load_measurement
from docfraud.llm.run import COMPARISON_SETTINGS, ForbiddenDocumentsError, Runner
from docfraud.llm.store import ResponseStore
from docfraud.reading.fields import Extraction, FieldSources, ReceiptFields
from docfraud.reading.layout import Row, Segment
from docfraud.reading.ocr import CachedOcr, OcrResult
from docfraud.reading.pipeline import read_document
from docfraud.synthesis.engine import Engine
from docfraud.synthesis.fonts import font_files
from docfraud.synthesis.review import ReviewLog, review_sample
from docfraud.synthesis.review import log_path as review_path
from docfraud.synthesis.review import save_log as save_review
from docfraud.synthesis.rounds import development_set
from docfraud.synthesis.statistics import RoundStatistics, Template
from tests.baselines.test_bow import corpus
from tests.data.test_benign_marks import HAND, PLAIN, check_log, mark, mark_log, marks_documents
from tests.data.test_observation import observation_documents, receipt_answers
from tests.detectors.test_copy_move_measurement import measured
from tests.evaluation.test_folds import documents as fold_documents
from tests.fixtures.findit import load_toy
from tests.fixtures.llm import FakeAgent, FakeDetector, FakeProvider
from tests.fusion.test_development import report_record
from tests.fusion.test_validation import N_TRAIN, pages, tables
from tests.llm.test_agent_rounds import ELSEWHERE
from tests.llm.test_agent_rounds import document as agent_document
from tests.llm.test_agent_tools import BOX as AGENT_BOX
from tests.llm.test_agent_tools import case as agent_case
from tests.llm.test_detector import development
from tests.synthesis.pages import documents as synthesis_documents
from tests.synthesis.pages import font as synthesis_font
from tests.synthesis.pages import receipt as synthesis_receipt

runner = CliRunner()


def test_version_flag_prints_installed_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"docfraud {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output


@pytest.fixture
def toy_source(findit_root: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Package the miniature dataset as a pinned archive under <tmp>/data/raw."""
    raw = tmp_path / "data" / "raw"
    raw.mkdir(parents=True)
    archive = Path(shutil.make_archive(str(raw / "findit2"), "zip", findit_root.parent, "findit2"))
    payload = archive.read_bytes()
    shutil.rmtree(findit_root)
    source = PinnedFile(
        name="findit2",
        url="https://example.org/findit2.zip",
        filename=archive.name,
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        homepage="https://example.org",
        citation="Toy citation",
    )
    monkeypatch.setitem(SOURCES, "findit2", source)
    monkeypatch.setattr(cli_common, "load_findit2", load_toy)
    monkeypatch.setattr(cli_data, "DUPLICATE_GROUPS", ())
    return tmp_path / "data"


def test_data_download_uses_the_pinned_source(
    toy_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_download(source: PinnedFile, directory: Path, **_: object) -> Path:
        calls.append((source.url, directory))
        return directory / source.filename

    monkeypatch.setattr(cli_common, "download", fake_download)
    result = runner.invoke(app, ["data", "download", "--data-dir", str(toy_source)])
    assert result.exit_code == 0, result.output
    assert calls == [("https://example.org/findit2.zip", toy_source / "raw")]
    assert "verified" in result.output


def test_data_prepare_extracts_and_validates(toy_source: Path) -> None:
    result = runner.invoke(app, ["data", "prepare", "--data-dir", str(toy_source)])
    assert result.exit_code == 0, result.output
    assert "6 documents (3 forged)" in result.output
    assert (toy_source / "raw" / "findit2" / "train.txt").is_file()


def test_download_fonts_fetches_every_pinned_font(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[str] = []

    def fake_download(source: PinnedFile, directory: Path, **_: object) -> Path:
        calls.append(source.filename)
        return directory / source.filename

    monkeypatch.setattr(cli_common, "download", fake_download)
    result = runner.invoke(app, ["synthesis", "download-fonts", "--directory", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert calls == [font.filename for font in font_files()]
    assert len(calls) == 6


def test_download_trufor_verifies_and_unpacks_code_and_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_download(source: PinnedFile, directory: Path, **_: object) -> Path:
        calls.append((source.url, directory))
        directory.mkdir(parents=True, exist_ok=True)
        archive = directory / source.filename
        with zipfile.ZipFile(archive, "w") as zf:
            zf.writestr(f"{source.name}/file.txt", source.name)
        return archive

    monkeypatch.setattr(cli_common, "download", fake_download)
    result = runner.invoke(app, ["forensics", "download-trufor", "--directory", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert [url for url, _ in calls] == [TRUFOR_CODE.url, TRUFOR_WEIGHTS.url]
    assert {directory for _, directory in calls} == {tmp_path / "downloads"}
    assert (tmp_path / "trufor-code" / "file.txt").read_text() == "trufor-code"
    assert (tmp_path / "trufor-weights" / "file.txt").is_file()
    assert "Guillaro" in result.output


def test_download_resnet_fetches_the_pinned_weights(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[PinnedFile, Path]] = []

    def fake_download(source: PinnedFile, directory: Path, **_: object) -> Path:
        calls.append((source, directory))
        return directory / source.filename

    monkeypatch.setattr(cli_common, "download", fake_download)
    result = runner.invoke(app, ["forensics", "download-resnet", "--directory", str(tmp_path)])
    assert result.exit_code == 0, result.output
    assert calls == [(IMAGENET_WEIGHTS, tmp_path)]
    assert "He" in result.output


def test_a_trial_of_the_clean_rounds_never_runs_where_the_development_round_is_kept(
    tmp_path: Path,
) -> None:
    assert cli_forensics.clean_rounds(cli_forensics.ROUNDS_DIR, trial=False) == CLEAN_FOLDS
    assert cli_forensics.clean_rounds(tmp_path, trial=True) == (DESIGN_FOLD,)
    with pytest.raises(typer.BadParameter, match="keeps the development round"):
        cli_forensics.clean_rounds(cli_forensics.ROUNDS_DIR, trial=True)
    for command in ("train-clean", "measure-clean"):
        result = runner.invoke(app, ["forensics", command, "--trial"])
        assert result.exit_code == 2, result.output  # refused before anything is read


def test_data_audit_writes_report(toy_source: Path, tmp_path: Path) -> None:
    runner.invoke(app, ["data", "prepare", "--data-dir", str(toy_source)])
    output = tmp_path / "audit.md"
    result = runner.invoke(
        app, ["data", "audit", "--data-dir", str(toy_source), "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    assert "Toy citation" in output.read_text(encoding="utf-8")


def test_unknown_dataset_is_rejected() -> None:
    result = runner.invoke(app, ["data", "prepare", "nope"])
    assert result.exit_code != 0


def test_data_draw_gold_writes_a_manifest(toy_source: Path, tmp_path: Path) -> None:
    runner.invoke(app, ["data", "prepare", "--data-dir", str(toy_source)])
    output = tmp_path / "gold.json"
    args = ["data", "draw-gold", "--data-dir", str(toy_source), "--authentic", "1"]
    args += ["--forged", "1", "--output", str(output)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    manifest = load_manifest(output)
    assert (len(manifest.authentic), len(manifest.forged)) == (1, 1)
    again = runner.invoke(app, args)
    assert again.exit_code == 1
    assert "refusing to overwrite" in again.output
    assert load_manifest(output) == manifest


def test_data_draw_gold_draws_around_earlier_sets(toy_source: Path, tmp_path: Path) -> None:
    runner.invoke(app, ["data", "prepare", "--data-dir", str(toy_source)])
    first, second = tmp_path / "a.json", tmp_path / "b.json"
    base = ["data", "draw-gold", "--data-dir", str(toy_source)]
    drawn = runner.invoke(app, [*base, "--authentic", "1", "--forged", "0", "--output", str(first)])
    assert drawn.exit_code == 0, drawn.output
    later = ["--authentic", "0", "--forged", "1", "--exclude", str(first), "--output", str(second)]
    result = runner.invoke(app, [*base, *later])
    assert result.exit_code == 0, result.output
    assert not load_manifest(first).doc_ids & load_manifest(second).doc_ids


def test_data_draw_folds_writes_folds_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(cli_common, "load_findit2", lambda _: fold_documents())
    output = tmp_path / "folds.json"
    args = ["data", "draw-folds", "--output", str(output)]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    assert "(seed 0)" in result.output
    folds = load_folds(output)
    assert folds.n_folds == 5
    again = runner.invoke(app, [*args, "--seed", "1"])
    assert again.exit_code == 1
    assert "refusing to overwrite" in again.output
    assert load_folds(output) == folds


class FakeOcr:
    identifier = "fake-ocr"

    def recognize(self, image: npt.NDArray[np.uint8]) -> OcrResult:
        return OcrResult(
            engine=self.identifier, width=image.shape[1], height=image.shape[0], tokens=()
        )


def test_reading_ocr_caches_unlocked_splits_only(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(app, ["data", "prepare", "--data-dir", str(toy_source)])
    monkeypatch.setattr(cli_common, "RapidOcrEngine", FakeOcr)
    cache = tmp_path / "ocr"
    result = runner.invoke(
        app, ["reading", "ocr", "--data-dir", str(toy_source), "--cache", str(cache)]
    )
    assert result.exit_code == 0, result.output
    assert "cached 4 pages" in result.output
    locked = runner.invoke(
        app,
        ["reading", "ocr", "--split", "test", "--data-dir", str(toy_source), "--cache", str(cache)],
    )
    assert locked.exit_code != 0
    assert isinstance(locked.exception, TestSetLockedError)


def test_reading_check_gold_lists_suspicious_authentic_annotations(tmp_path: Path) -> None:
    manifest = GoldManifest(
        dataset="d", split=Split.TRAIN, seed=0, authentic=("XA", "XB"), forged=("XF",)
    )
    path = tmp_path / "manifest.json"
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    amounts = {"total_due": ["5.00"], "tendered": ["10.00"], "change": ["4.00"]}
    bad = json.dumps({"doc_id": "XA", "annotator": "t", "fields": {"amounts": amounts}})
    (tmp_path / "XA.json").write_text(bad, encoding="utf-8")
    result = runner.invoke(app, ["reading", "check-gold", "--manifest", str(path)])
    assert result.exit_code == 0, result.output
    assert "1 of 3 gold receipts annotated" in result.output
    assert "re-check XA: Tendered - due = change" in result.output
    assert "missing: XB, XF" in result.output


def write_gold(tmp_path: Path, doc_ids: tuple[str, ...]) -> Path:
    manifest = GoldManifest(
        dataset="findit2", split=Split.TRAIN, seed=0, authentic=doc_ids, forged=()
    )
    path = tmp_path / "gold" / "manifest.json"
    path.parent.mkdir()
    path.write_text(manifest.model_dump_json(), encoding="utf-8")
    amounts = {"total_due": ["5.00"]}
    for doc_id in doc_ids:
        record = {"doc_id": doc_id, "annotator": "t", "fields": {"amounts": amounts}}
        (path.parent / f"{doc_id}.json").write_text(json.dumps(record), encoding="utf-8")
    return path


def test_reading_review_gold_writes_a_page_for_the_sample(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(app, ["data", "prepare", "--data-dir", str(toy_source)])
    train = [d.doc_id for d in load_toy(toy_source / "raw" / "findit2") if d.split is Split.TRAIN]
    manifest = write_gold(tmp_path, tuple(train))
    monkeypatch.setattr(cli_reading, "draw_review_sample", lambda *_, **__: (train[0],))
    out = tmp_path / "review"
    args = ["reading", "review-gold", "--data-dir", str(toy_source), "--manifest", str(manifest)]
    result = runner.invoke(app, [*args, "--output", str(out)])
    assert result.exit_code == 0, result.output
    page = (out / "index.html").read_text(encoding="utf-8")
    assert f'"doc_id": "{train[0]}"' in page
    assert f"../data/raw/findit2/train/{train[0]}" in page


def test_reading_review_gold_needs_annotated_sample(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_gold(tmp_path, ("XA",))
    monkeypatch.setattr(cli_reading, "draw_review_sample", lambda *_, **__: ("XA", "XB"))
    result = runner.invoke(app, ["reading", "review-gold", "--manifest", str(manifest)])
    assert result.exit_code == 1
    assert "not annotated yet: XB" in result.output


def test_reading_score_review_keeps_the_review_and_writes_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_gold(tmp_path, ("XA",))
    monkeypatch.setattr(cli_reading, "draw_review_sample", lambda *_, **__: ("XA",))
    review = {
        "reviewer": "owner",
        "reviews": [
            {"doc_id": "XA", "verdicts": [{"unit_id": "amount:total_due:0", "accepted": True}]}
        ],
    }
    downloaded = tmp_path / "gold-review.json"
    downloaded.write_text(json.dumps(review), encoding="utf-8")
    record, report = tmp_path / "kept.json", tmp_path / "reports" / "verification.md"
    args = ["reading", "score-review", str(downloaded), "--manifest", str(manifest)]
    result = runner.invoke(app, [*args, "--record", str(record), "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert "1 of 1 accepted" in result.output
    assert json.loads(record.read_text(encoding="utf-8"))["reviewer"] == "owner"
    assert "agreement 1.000" in report.read_text(encoding="utf-8")


def test_reading_score_review_reports_resolutions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_gold(tmp_path, ("XA",))
    monkeypatch.setattr(cli_reading, "draw_review_sample", lambda *_, **__: ("XA",))
    verdict = {"unit_id": "amount:total_due:0", "accepted": False, "comment": "5.50?"}
    review = {"reviewer": "owner", "reviews": [{"doc_id": "XA", "verdicts": [verdict]}]}
    downloaded = tmp_path / "gold-review.json"
    downloaded.write_text(json.dumps(review), encoding="utf-8")
    resolutions = tmp_path / "resolutions.json"
    resolutions.write_text(json.dumps({"XA/amount:total_due:0": "kept"}), encoding="utf-8")
    report = tmp_path / "verification.md"
    args = ["reading", "score-review", str(downloaded), "--manifest", str(manifest)]
    args += ["--record", str(downloaded), "--resolutions", str(resolutions)]
    result = runner.invoke(app, [*args, "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert "| XA | amount:total_due:0 | 5.50? | kept |" in report.read_text(encoding="utf-8")


def test_reading_evaluate_gold_scores_both_text_sources(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(app, ["data", "prepare", "--data-dir", str(toy_source)])
    monkeypatch.setattr(cli_common, "RapidOcrEngine", FakeOcr)
    train = [d.doc_id for d in load_toy(toy_source / "raw" / "findit2") if d.split is Split.TRAIN]
    manifest = write_gold(tmp_path, tuple(train))
    out, report = tmp_path / "predictions", tmp_path / "evaluation.md"
    args = ["reading", "evaluate-gold", "--data-dir", str(toy_source), "--manifest", str(manifest)]
    args += ["--cache", str(tmp_path / "ocr"), "--predictions", str(out)]
    result = runner.invoke(app, [*args, "--output", str(report)])
    assert result.exit_code == 0, result.output
    text = report.read_text(encoding="utf-8")
    assert "| Metric | T1 transcript | T2 OCR |" in text
    assert "OCR engine: `fake-ocr`" in text
    assert len(list((out / "transcript").glob("*.json"))) == len(train)
    assert len(list((out / "ocr").glob("*.json"))) == len(train)


def test_reading_evaluate_gold_states_the_note(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(app, ["data", "prepare", "--data-dir", str(toy_source)])
    monkeypatch.setattr(cli_common, "RapidOcrEngine", FakeOcr)
    train = [d.doc_id for d in load_toy(toy_source / "raw" / "findit2") if d.split is Split.TRAIN]
    manifest, report = write_gold(tmp_path, tuple(train)), tmp_path / "evaluation.md"
    args = ["reading", "evaluate-gold", "--data-dir", str(toy_source), "--manifest", str(manifest)]
    args += ["--cache", str(tmp_path / "ocr"), "--predictions", str(tmp_path / "predictions")]
    args += ["--title", "Parser v2 on gold set B", "--note", "Frozen before annotation."]
    result = runner.invoke(app, [*args, "--output", str(report)])
    assert result.exit_code == 0, result.output
    assert report.read_text(encoding="utf-8").startswith(
        "# Parser v2 on gold set B\n\nGenerated by `docfraud reading evaluate-gold`; "
        "do not edit by hand.\n\nFrozen before annotation.\n\n"
    )


def _keep_predictions(directory: Path, doc_ids: tuple[str, ...], total_due: str) -> None:
    for source in ("transcript", "ocr"):
        (directory / source).mkdir(parents=True)
        for doc_id in doc_ids:
            fields = {"amounts": {"total_due": [total_due]}}
            (directory / source / f"{doc_id}.json").write_text(json.dumps(fields), "utf-8")


def test_reading_compare_gold_compares_kept_predictions_per_text_source(tmp_path: Path) -> None:
    manifest = write_gold(tmp_path, ("XA",))
    _keep_predictions(tmp_path / "v1", ("XA",), "9.99")
    _keep_predictions(tmp_path / "v2", ("XA",), "5.00")
    report = tmp_path / "comparison.md"
    args = ["reading", "compare-gold", "--manifest", str(manifest), "--output", str(report)]
    args += ["--baseline", str(tmp_path / "v1"), "--candidate", str(tmp_path / "v2")]
    args += ["--baseline-name", "parser v1", "--candidate-name", "parser v2"]
    result = runner.invoke(app, args)
    assert result.exit_code == 0, result.output
    text = report.read_text(encoding="utf-8")
    assert text.startswith("# Comparison of two extractors on the gold set\n")
    assert "## T1 transcript" in text
    assert "## T2 OCR" in text
    assert "| Metric | parser v1 | parser v2 |" in text
    assert "### Difference parser v2 - parser v1" in text
    assert "| Amounts: recall | 0.000 [0.000, 0.000] | 1.000 [1.000, 1.000] |" in text


def test_reading_compare_gold_needs_every_prediction(tmp_path: Path) -> None:
    manifest = write_gold(tmp_path, ("XA",))
    _keep_predictions(tmp_path / "v1", ("XA",), "5.50")
    (tmp_path / "v2").mkdir()
    args = ["reading", "compare-gold", "--manifest", str(manifest)]
    args += ["--output", str(tmp_path / "comparison.md")]
    args += ["--baseline", str(tmp_path / "v1"), "--candidate", str(tmp_path / "v2")]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "no prediction" in result.output


def test_reading_compare_gold_needs_distinct_names(tmp_path: Path) -> None:
    args = ["reading", "compare-gold", "--output", str(tmp_path / "comparison.md")]
    args += ["--baseline", str(tmp_path), "--candidate", str(tmp_path)]
    result = runner.invoke(app, [*args, "--baseline-name", "v", "--candidate-name", "v"])
    assert result.exit_code == 1
    assert "different names" in result.output


def test_reading_evaluate_gold_needs_every_annotation(tmp_path: Path) -> None:
    manifest = write_gold(tmp_path, ("XA",))
    (manifest.parent / "XA.json").unlink()
    result = runner.invoke(app, ["reading", "evaluate-gold", "--manifest", str(manifest)])
    assert result.exit_code == 1
    assert "not annotated yet: XA" in result.output


def baseline_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    docs = corpus(tmp_path / "corpus")
    monkeypatch.setattr(cli_common, "load_findit2", lambda _: docs)
    folds = tmp_path / "folds.json"
    save_folds(assign_folds(docs), folds)
    return ["--runs", str(tmp_path / "runs"), "--folds", str(folds)]


def test_baseline_cross_validation_is_recorded_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = baseline_setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["baseline", "bow-lr", *options])
    assert result.exit_code == 0, result.output
    assert "cross-validated F1 1.000" in result.output
    (record,) = JsonRunStore(tmp_path / "runs").records("bow-lr-cv")
    assert record.test_unlocks == ()
    assert record.results["counts"]["fp"] == 0
    report = tmp_path / "bow-lr.md"
    args = ["baseline", "report", "--runs", str(tmp_path / "runs"), "--output", str(report)]
    assert runner.invoke(app, args).exit_code == 0
    assert "Not run yet" in report.read_text(encoding="utf-8")


def test_baseline_test_run_happens_once_from_a_clean_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = baseline_setup(tmp_path, monkeypatch)
    command = ["baseline", "bow-lr", *options, "--unlock-test"]
    monkeypatch.setattr(cli_common, "code_version", lambda: CodeVersion(commit="a", dirty=True))
    dirty = runner.invoke(app, command)
    assert dirty.exit_code == 1
    assert "commit all changes first" in dirty.output
    monkeypatch.setattr(cli_common, "code_version", lambda: CodeVersion(commit="a", dirty=False))
    first = runner.invoke(app, command)
    assert first.exit_code == 0, first.output
    assert "reproduction failed" in first.output  # perfect toy scores are far above 17.78
    (record,) = JsonRunStore(tmp_path / "runs").records("bow-lr-test")
    assert record.test_unlocks == (cli_baseline.BOW_TEST_REASON,)
    assert record.results["n_documents"] == 12
    again = runner.invoke(app, command)
    assert again.exit_code == 1
    assert "refusing to run again" in again.output


def test_baseline_report_needs_a_cross_validation_run(tmp_path: Path) -> None:
    args = ["baseline", "report", "--runs", str(tmp_path), "--output", str(tmp_path / "r.md")]
    result = runner.invoke(app, args)
    assert result.exit_code == 1
    assert "no bow-lr-cv run" in result.output


def rules_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Training and validation receipts whose forged ones print a wrong change."""
    docs = [
        d.model_copy(update={"split": Split.VAL}) if d.split is Split.TEST else d
        for d in corpus(tmp_path / "corpus")
    ]
    gold_ids = tuple(d.doc_id for d in docs[:4])
    manifest = write_gold(tmp_path, gold_ids)

    def fields(document: Any, *_: Any) -> SimpleNamespace:
        change = "9.00" if document.label is Label.FORGED else "8.00"
        amounts = {"total_due": ["12.00"], "tendered": ["20.00"], "change": [change]}
        return SimpleNamespace(fields=ReceiptFields.model_validate({"amounts": amounts}))

    monkeypatch.setattr(cli_common, "load_findit2", lambda _: tuple(docs))
    monkeypatch.setattr(cli_common, "RapidOcrEngine", FakeOcr)
    monkeypatch.setattr(cli_rules, "read_document", fields)
    folds = tmp_path / "folds.json"
    save_folds(assign_folds(docs), folds)
    runs = tmp_path / "runs"
    options = ["--runs", str(runs), "--folds", str(folds), "--cache", str(tmp_path / "ocr")]
    assert (
        runner.invoke(
            app, ["baseline", "bow-lr", "--runs", str(runs), "--folds", str(folds)]
        ).exit_code
        == 0
    )
    return [*options, "--gold", str(manifest)]


def test_rules_are_cross_validated_against_the_baseline_and_reported(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = rules_setup(tmp_path, monkeypatch)
    result = runner.invoke(app, ["rules", "evaluate", *options])
    assert result.exit_code == 0, result.output
    (record,) = JsonRunStore(tmp_path / "runs").records("rules-cv")
    assert record.results["transcript"]["any_fail"]["counts"]["fp"] == 0
    assert record.results["gold_fields"]["any_fail"]["n_documents"] == 4
    report = tmp_path / "rules.md"
    args = ["rules", "report", "--runs", str(tmp_path / "runs"), "--output", str(report)]
    assert runner.invoke(app, args).exit_code == 0
    assert "Not measured yet" in report.read_text(encoding="utf-8")


def test_rules_validation_runs_once_from_a_clean_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = rules_setup(tmp_path, monkeypatch)
    command = ["rules", "evaluate", *options, "--validation"]
    monkeypatch.setattr(cli_common, "code_version", lambda: CodeVersion(commit="a", dirty=True))
    assert "commit all changes first" in runner.invoke(app, command).output
    monkeypatch.setattr(cli_common, "code_version", lambda: CodeVersion(commit="a", dirty=False))
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    (record,) = JsonRunStore(tmp_path / "runs").records("rules-validation")
    assert record.test_unlocks == ()
    assert record.results["ocr"]["learned"]["n_documents"] == 12
    again = runner.invoke(app, command)
    assert again.exit_code == 1
    assert "refusing to run again" in again.output


def test_copy_move_is_measured_once_on_the_clean_folds_from_a_clean_commit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    docs = [d for d in corpus(tmp_path / "corpus") if d.split is Split.TRAIN]
    for document in docs:
        Image.new("L", (40, 30), 200).save(document.image_path)
    folds = tmp_path / "folds.json"
    save_folds(assign_folds(docs), folds)
    assignment = load_folds(folds).folds
    seen: list[str] = []

    def fake_measure(document: Any, *_: Any) -> Any:
        seen.append(document.doc_id)
        forged = document.label is Label.FORGED
        return measured(forged=forged, score=1.0 if forged else 0.1, hit=forged)

    monkeypatch.setattr(cli_common, "load_findit2", lambda _: tuple(docs))
    monkeypatch.setattr(cli_common, "RapidOcrEngine", FakeOcr)
    monkeypatch.setattr(cli_copy_move, "measure", fake_measure)
    report = tmp_path / "copy-move.md"
    runs = tmp_path / "runs"
    options = ["--runs", str(runs), "--folds", str(folds), "--cache", str(tmp_path / "ocr")]
    command = ["copy-move", "measure", *options, "--output", str(report)]
    monkeypatch.setattr(cli_common, "code_version", lambda: CodeVersion(commit="a", dirty=True))
    assert "commit all changes first" in runner.invoke(app, command).output
    monkeypatch.setattr(cli_common, "code_version", lambda: CodeVersion(commit="a", dirty=False))
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert seen
    assert all(assignment[doc_id] in CLEAN_FOLDS for doc_id in seen)
    assert {d.doc_id for d in docs if assignment[d.doc_id] in CLEAN_FOLDS} == set(seen)
    (record,) = JsonRunStore(runs).records("copy-move-clean")
    assert record.results["guard"]["all"]["detects"]
    assert "Localisation guard of the pair" in report.read_text(encoding="utf-8")
    again = runner.invoke(app, command)
    assert again.exit_code == 1
    assert "refusing to run again" in again.output
    rendered = tmp_path / "again.md"
    args = ["copy-move", "report", "--runs", str(runs), "--output", str(rendered)]
    assert runner.invoke(app, args).exit_code == 0
    assert rendered.read_text(encoding="utf-8") == report.read_text(encoding="utf-8")
    empty = ["copy-move", "report", "--runs", str(tmp_path / "none"), "--output", str(rendered)]
    assert "no copy-move-clean record" in runner.invoke(app, empty).output


def test_the_training_report_renders_from_its_latest_run_record(tmp_path: Path) -> None:
    runs = tmp_path / "runs"
    store = JsonRunStore(runs)
    with run("fusion-training-oof", store=store, config={}, seeds={}, dataset_sha256={}) as current:
        for key, value in report_record().items():
            current.log(key, value)
    output = tmp_path / "training-oof.md"
    args = ["fusion", "training-report", "--runs", str(runs), "--output", str(output)]
    assert runner.invoke(app, args).exit_code == 0
    assert "RQ4: the report's first region" in output.read_text(encoding="utf-8")
    empty = ["fusion", "training-report", "--runs", str(tmp_path / "none"), "--output", str(output)]
    assert "no fusion-training-oof record" in runner.invoke(app, empty).output


@dataclass
class Recorded:
    """Both inputs tables on disk, their run records, and the commands that read them."""

    tmp: Path
    runs: Path
    store: JsonRunStore
    paths: dict[str, Path]
    training: Any
    training_run: Any

    def check(self, **options: Path) -> Any:
        """Run the validation check, with other tables or runs where given."""
        given = self.paths | options
        return runner.invoke(
            app,
            [
                *("fusion", "validation-check", "--runs", str(given.get("runs", self.runs))),
                *("--training", str(given["training"]), "--validation", str(given["validation"])),
                *("--output", str(self.tmp / "check.md")),
            ],
        )

    def shift(self, output: Path) -> Any:
        """Run the comparison of the inputs."""
        tables_given = ["--training", str(self.paths["training"])]
        tables_given += ["--validation", str(self.paths["validation"])]
        command = ["fusion", "input-shift", "--runs", str(self.runs), *tables_given]
        return runner.invoke(app, [*command, "--output", str(output)])


def said(result: Any) -> str:
    """A command's output with the console's line wrapping undone."""
    return " ".join(result.output.split())


@pytest.fixture
def recorded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Recorded:
    """Both inputs tables with the run records that made them, from a clean commit."""
    training, y_train, validation, checked = tables()
    train = [
        Document(
            dataset="toy",
            doc_id=f"R{i:02d}",
            source_id=f"R{i:02d}",
            split=Split.TRAIN,
            label=Label.FORGED if forged else Label.AUTHENTIC,
            image_path=tmp_path / f"R{i:02d}.png",
        )
        for i, forged in enumerate(y_train)
    ]
    assert len(train) == N_TRAIN
    paths = {"training": tmp_path / "training.json", "validation": tmp_path / "validation.json"}
    save_table(training, paths["training"])
    save_table(validation, paths["validation"])
    clean = CodeVersion(commit="a" * 40, dirty=False)
    monkeypatch.setattr(runs_module, "code_version", lambda repo=None: clean)  # noqa: ARG005
    runs = tmp_path / "runs"
    store = JsonRunStore(runs)
    record = {"store": store, "config": {}, "seeds": {}, "dataset_sha256": {}}
    with run("fusion-training-inputs", **record) as current:  # type: ignore[arg-type]
        current.log("summary", summary(training))
    (training_run,) = store.records("fusion-training-inputs")
    with run(
        "fusion-validation-inputs",
        store=store,
        config={"training_inputs": training_run.run_id},
        seeds={},
        dataset_sha256={},
    ) as current:
        current.log("sha256", hashlib.sha256(paths["validation"].read_bytes()).hexdigest())
    monkeypatch.setattr(cli_common, "load_findit2", lambda _: (*train, *checked))
    monkeypatch.setattr(cli_common, "code_version", lambda: clean)
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: [])
    monkeypatch.setattr(cli_fusion, "_pages", lambda documents: pages(list(documents)))
    monkeypatch.setattr(cli_fusion, "validation_check", partial(validation_check, resamples=50))
    return Recorded(tmp_path, runs, store, paths, training, training_run)


def test_the_test_inputs_are_made_once_by_code_that_reproduced_the_validation_inputs(
    recorded: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp, store = recorded.tmp, recorded.store
    model = tmp / "model.pt"
    model.write_bytes(b"weights")
    monkeypatch.setattr(cli_fusion, "_whole_training_models", lambda _: {0: model})
    validation = load_table(recorded.paths["validation"])
    rebuilt = {"table": InputsTable(receipts=validation.receipts[1:])}

    def split_inputs(*_: Any) -> Any:
        return cli_fusion._Made(rebuilt["table"], {}, {"0": "m"}, {"page and OCR": [0.5, 1.5]})

    monkeypatch.setattr(cli_fusion, "_split_inputs", split_inputs)
    runs = ["--runs", str(recorded.runs)]
    rehearse = ["fusion", "rehearse-inputs", *runs, "--output", str(tmp / "again.json")]
    rehearse += ["--validation", str(recorded.paths["validation"])]
    final = ["fusion", "test-inputs", *runs, "--output", str(tmp / "test.json")]
    assert "no fusion-inputs-rehearsal record" in said(runner.invoke(app, final))
    other = tmp / "other.json"
    other.write_text(recorded.paths["validation"].read_text("utf-8") + "\n", encoding="utf-8")
    elsewhere = [*rehearse[: rehearse.index("--validation")], "--validation", str(other)]
    assert "is not the table" in said(runner.invoke(app, elsewhere))
    assert "not reproduced" in said(runner.invoke(app, rehearse))
    assert "did not reproduce the validation inputs" in said(runner.invoke(app, final))
    rebuilt["table"] = validation
    assert "the validation inputs: reproduced" in said(runner.invoke(app, rehearse))
    failed, passed = store.records("fusion-inputs-rehearsal")
    assert failed.results["differences"]["rows_missing"] == 1
    assert passed.results["reproduced"]
    assert "has not read the test split" in said(runner.invoke(app, final))
    with run("llm-test-reading", store=store, config={}, seeds={}, dataset_sha256={}):
        pass
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: ["src/docfraud/linking/links.py"])
    assert "the source changed after phase-7-freeze" in said(runner.invoke(app, final))
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: [])
    result = runner.invoke(app, final)
    assert result.exit_code == 0, result.output
    (made,) = store.records("fusion-test-inputs")
    assert made.test_unlocks == (cli_common.TEST_UNLOCK,)
    assert made.config["rehearsal"] == passed.run_id
    assert made.results["efficiency"]["seconds"]["page and OCR"]["median"] == 1.0
    assert load_table(tmp / "test.json") == validation
    assert "refusing to run again" in said(runner.invoke(app, final))


def test_the_validation_check_refuses_tables_code_and_an_order_it_cannot_trust(
    recorded: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    assert "no fusion-validation-inputs record" in said(recorded.check(runs=recorded.tmp / "x"))
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: ["src/docfraud/fusion/model.py"])
    assert "the code changed since the validation inputs were made" in said(recorded.check())
    assert "the code changed" in said(recorded.shift(recorded.tmp / "shift.md"))
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: [])
    other = recorded.tmp / "other.json"
    save_table(table_of_seeds(recorded.training, (0, 1)), other)
    assert "is not the table" in said(recorded.check(training=other))
    other.write_text(recorded.paths["validation"].read_text(encoding="utf-8") + "\n", "utf-8")
    assert "is not the table" in said(recorded.check(validation=other))
    assert "compare the inputs of both splits first" in said(recorded.check())
    assert not recorded.store.records("fusion-validation-check")


def _validation_receipts(tmp: Path) -> list[Document]:
    """The toy validation receipts with a page and a transcript each; a few say the opposite."""
    pages_dir, transcripts = tmp / "pages", tmp / "transcripts"
    pages_dir.mkdir()
    transcripts.mkdir()
    found = []
    for shade, document in enumerate(tables()[3], start=200):
        image, text = pages_dir / f"{document.doc_id}.png", transcripts / f"{document.doc_id}.txt"
        Image.new("L", (40, 30), shade).save(image)
        says_edited = (document.label is Label.FORGED) != (shade % 7 == 0)
        text.write_text(
            f"RECEIPT {document.doc_id}\n{'EDITED' if says_edited else 'PRINTED'}\n", "utf-8"
        )
        found.append(document.model_copy(update={"image_path": image, "transcript_path": text}))
    return found


def test_the_detector_runs_once_on_the_validation_split_against_phase_5s_fusion(
    recorded: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp, store = recorded.tmp, recorded.store
    assert recorded.shift(tmp / "shift.md").exit_code == 0
    assert recorded.check().exit_code == 0
    validation_receipts = _validation_receipts(tmp)
    monkeypatch.setattr(cli_common, "load_findit2", lambda _: tuple(validation_receipts))
    monkeypatch.setattr(cli_common, "RapidOcrEngine", FakeOcr)
    monkeypatch.setattr(
        cli_detector, "detector_validation", partial(detector_validation, resamples=20)
    )
    instruction = tmp / "detector-v1.md"
    instruction.write_text(Path("src/docfraud/llm/instructions/detector-v1.md").read_text("utf-8"))
    sha = hashlib.sha256(instruction.read_text("utf-8").encode()).hexdigest()
    rounds = tmp / "rounds"
    rounds.mkdir()
    named = {"instruction": str(instruction), "instruction_sha256": sha, "commit": "a" * 40}
    frozen = FrozenDetector(version=1, settings=SONNET, **named)
    (rounds / "frozen.json").write_text(frozen.model_dump_json(), encoding="utf-8")
    output = tmp / "validation.md"

    def validate(
        validation: Path = recorded.paths["validation"], runs: Path = recorded.runs
    ) -> Any:
        command = ["detector", "validation", "--runs", str(runs), "--rounds", str(rounds)]
        command += ["--validation", str(validation), "--cache", str(tmp / "ocr")]
        return runner.invoke(app, [*command, "--output", str(output)])

    def answering(provider: FakeProvider) -> None:
        books = (ResponseStore(tmp / "store"), Ledger(tmp / "ledger.jsonl"))
        monkeypatch.setattr(cli_detector, "_runner", lambda *_: Runner(provider, *books))

    answering(FakeDetector(fail=frozenset({0})))
    assert "check the stronger model on the design fold first" in said(validate())
    row = development(1, 0.5).main_row()
    checked = StrongerModelCheck(
        settings=OPUS, **named, result=row, frozen_result=row, row_added=True
    )
    (rounds / "stronger-model.json").write_text(checked.model_dump_json(), encoding="utf-8")
    other = tmp / "other.json"
    other.write_text(recorded.paths["validation"].read_text("utf-8") + "\n", encoding="utf-8")
    assert "is not the table" in said(validate(other))
    assert "Phase 5's check comes first" in said(validate(runs=tmp / "no-runs"))
    assert "of 120 requests were not answered" in said(validate())
    assert not store.records("llm-detector-validation")
    answering(FakeDetector())
    result = validate()
    assert result.exit_code == 0, result.output
    (made,) = store.records("llm-detector-validation")
    assert made.config["validation_check"] == store.records("fusion-validation-check")[0].run_id
    assert list(made.results["rows"]) == [r.name for r in validation_rows(SONNET, stronger=True)]
    assert made.results["rows"]["transcript, claude-sonnet-5-5"]["fitted"]["platt"] is not None
    spent = [json.loads(line) for line in (tmp / "ledger.jsonl").read_text().splitlines()]
    assert sum(entry["model"] == OPUS.model for entry in spent) == 30
    assert "**RQ6, as a check.**" in output.read_text(encoding="utf-8")
    assert "refusing to run again" in said(validate())
    again = tmp / "again.md"
    rendered = ["detector", "validation-report", "--runs", str(recorded.runs)]
    assert runner.invoke(app, [*rendered, "--output", str(again)]).exit_code == 0
    assert again.read_text(encoding="utf-8") == output.read_text(encoding="utf-8")
    missing = runner.invoke(app, ["detector", "validation-report", "--runs", str(tmp / "none")])
    assert "no llm-detector-validation record" in said(missing)


def test_the_inputs_are_compared_then_the_validation_check_runs_once(recorded: Recorded) -> None:
    store, shift_report = recorded.store, recorded.tmp / "shift.md"
    shifted = recorded.shift(shift_report)
    assert shifted.exit_code == 0, shifted.output
    (compared,) = store.records("fusion-input-shift")
    (inputs_run,) = store.records("fusion-validation-inputs")
    assert compared.config["validation_inputs"] == inputs_run.run_id
    assert len(compared.results["comparisons"]) == 21
    assert "Every fusion input, training → validation" in shift_report.read_text("utf-8")
    result = recorded.check()
    assert result.exit_code == 0, result.output
    (made,) = store.records("fusion-validation-check")
    assert made.config["training_inputs"] == recorded.training_run.run_id
    assert made.test_unlocks == ()
    assert set(made.results) >= {"transcript", "ocr", "policy", "grounding", "rq4"}
    text = (recorded.tmp / "check.md").read_text(encoding="utf-8")
    assert "What Phase 7 applies to the test split" in text
    for late in (recorded.check(), recorded.shift(recorded.tmp / "late.md")):
        assert late.exit_code == 1
        assert "refusing to run again" in said(late)
    rendered = recorded.tmp / "again.md"
    for command, made_text in (("validation-report", text), ("input-shift-report", None)):
        args = ["fusion", command, "--runs", str(recorded.runs), "--output", str(rendered)]
        assert runner.invoke(app, args).exit_code == 0
        expected = made_text or shift_report.read_text(encoding="utf-8")
        assert rendered.read_text(encoding="utf-8") == expected


def test_validation_receipts_are_read_without_their_labels_and_annotations() -> None:
    forged = next(d for d in tables()[3] if d.label is Label.FORGED)
    assert (bool(forged.regions), forged.editing_software) == (True, "gimp")
    plain = cli_fusion.unlabelled(forged)
    assert (plain.label, plain.regions, plain.editing_software) == (None, (), None)
    assert (plain.has_region_annotations, plain.handwritten_annotation) == (False, False)
    assert plain.digital_annotation is False
    assert (plain.doc_id, plain.image_path, plain.split) == (
        forged.doc_id,
        forged.image_path,
        forged.split,
    )
    assert Document.model_validate(plain.model_dump()) == plain


def table_of_seeds(table: Any, seeds: Sequence[int]) -> Any:
    """The same table with the rows of some seeds only."""
    return table.model_copy(
        update={"receipts": tuple(r for r in table.receipts if r.seed in seeds)}
    )


def test_rules_need_the_baseline_runs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    options = rules_setup(tmp_path, monkeypatch)
    for record in (tmp_path / "runs").iterdir():
        shutil.rmtree(record)
    result = runner.invoke(app, ["rules", "evaluate", *options])
    assert result.exit_code == 1
    assert "run the baseline first" in result.output
    report = ["rules", "report", "--runs", str(tmp_path / "runs")]
    assert "need rules-cv and bow-lr-cv" in runner.invoke(app, report).output


def test_rules_breakdown_groups_recorded_decisions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = rules_setup(tmp_path, monkeypatch)
    runs = ["--runs", str(tmp_path / "runs")]
    missing = runner.invoke(app, ["rules", "breakdown", *runs])
    assert missing.exit_code == 1
    assert "no rules-cv run" in missing.output
    assert runner.invoke(app, ["rules", "evaluate", *options]).exit_code == 0
    result = runner.invoke(app, ["rules", "breakdown", *runs])
    assert result.exit_code == 0, result.output
    (record,) = JsonRunStore(tmp_path / "runs").records("rules-breakdown")
    assert set(record.results) == {"training"}
    flagged = record.results["training"]["transcript"]["any_fail"]["forged"]["software"]
    assert flagged == {"not recorded": {"receipts": 12.0, "found": 12.0}}
    report = tmp_path / "rules.md"
    assert runner.invoke(app, ["rules", "report", *runs, "--output", str(report)]).exit_code == 0
    assert (
        "Training, cross-validated: forged receipts found, by editing software"
        in report.read_text(encoding="utf-8")
    )


@pytest.fixture
def observation_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    documents = observation_documents(tmp_path / "d")
    monkeypatch.setattr(cli_observation, "findit_documents", lambda *_, **__: documents)
    monkeypatch.setattr(cli_observation, "require_clean_commit", lambda: None)
    monkeypatch.setattr(
        cli_observation, "code_version", lambda: CodeVersion(commit="abc1234", dirty=False)
    )
    return tmp_path


def test_observation_page_serves_until_interrupted(
    observation_cli: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []

    class FakeServer:
        server_port = 9999
        closed = False

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            FakeServer.closed = True

    monkeypatch.setattr(cli_observation, "make_server", lambda *_: FakeServer())
    monkeypatch.setattr("webbrowser.open", opened.append)
    logs = observation_cli / "logs"
    result = runner.invoke(app, ["observation", "page", "--observer", "emre", "--logs", str(logs)])
    assert result.exit_code == 0, result.output
    assert opened == ["http://127.0.0.1:9999/"]
    assert "run the same command to continue" in result.output
    assert FakeServer.closed


def test_observation_crops_are_written_for_the_assistant(observation_cli: Path) -> None:
    output = observation_cli / "crops"
    result = runner.invoke(app, ["observation", "crops", "--output", str(output)])
    assert result.exit_code == 0, result.output
    assert len(list(output.glob("*-close-up-*.png"))) == 3


def test_observation_check_reports_progress_and_strays(observation_cli: Path) -> None:
    logs = observation_cli / "logs"
    result = runner.invoke(app, ["observation", "check", "--observer", "emre", "--logs", str(logs)])
    assert result.exit_code == 0, result.output
    assert "0 of 3 regions (0 of 2 full forms), 0 of 2 receipts" in result.output
    stray = ObservationLog(
        observer="emre",
        receipts={"X51005268472": ReceiptObservation.model_validate(receipt_answers())},
    )
    save_log(stray, logs / "emre.json")
    result = runner.invoke(app, ["observation", "check", "--observer", "emre", "--logs", str(logs)])
    assert result.exit_code == 1
    assert "X51005268472" in result.output


def test_observation_seal_is_written_once_and_verified(observation_cli: Path) -> None:
    logs, records = observation_cli / "logs", observation_cli / "records"
    save_log(ObservationLog(observer=ASSISTANT), logs / "claude.json")
    options = ["--logs", str(logs), "--records", str(records)]
    assert runner.invoke(app, ["observation", "seal", *options]).exit_code == 0
    assert runner.invoke(app, ["observation", "seal", *options]).exit_code == 1
    assert runner.invoke(app, ["observation", "seal", *options, "--verify"]).exit_code == 0
    (logs / "claude.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(app, ["observation", "seal", *options, "--verify"])
    assert result.exit_code == 1
    assert "changed since it was sealed" in result.output


def test_observation_report_needs_an_intact_seal(observation_cli: Path) -> None:
    from tests.data.test_observation_report import observed  # noqa: PLC0415

    _, _, assistant, owner = observed(observation_cli / "o")
    records, output = observation_cli / "records", observation_cli / "report.md"
    save_log(assistant, records / "claude.json")
    save_log(owner, records / "emre.json")
    seal(records / "claude.json", records / "claude.sha256")
    options = ["--records", str(records), "--output", str(output)]
    result = runner.invoke(app, ["observation", "report", *options])
    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8").startswith("# Forgery observations")
    (records / "claude.json").write_text("{}", encoding="utf-8")
    result = runner.invoke(app, ["observation", "report", *options])
    assert result.exit_code == 1
    assert "no longer matches its seal" in result.output


@pytest.fixture
def marks_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    documents = marks_documents(tmp_path / "d")
    monkeypatch.setattr(cli_marks, "findit_documents", lambda *_, **__: documents)
    monkeypatch.setattr(cli_marks, "require_clean_commit", lambda: None)
    monkeypatch.setattr(
        cli_marks, "code_version", lambda: CodeVersion(commit="abc1234", dirty=False)
    )
    plan = marks_plan(documents)
    receipts = {doc_id: ReceiptMarks() for doc_id in plan}
    receipts[HAND] = ReceiptMarks(marks=(mark(),))
    save_log(mark_log(receipts), round_path(tmp_path / "work", ASSISTANT, 1))
    return tmp_path


def marks_options(root: Path, *extra: str) -> list[str]:
    return ["--work", str(root / "work"), "--records", str(root / "records"), *extra]


def test_marks_views_are_written_plain_and_marked(marks_cli: Path) -> None:
    result = runner.invoke(app, ["marks", "views", *marks_options(marks_cli)])
    assert result.exit_code == 0, result.output
    assert len(list((marks_cli / "work" / "views-1").glob("*-overview.png"))) == 3
    result = runner.invoke(app, ["marks", "views", "--marked", *marks_options(marks_cli)])
    assert result.exit_code == 0, result.output
    assert len(list((marks_cli / "work" / "views-1-marked").glob("*-marked.png"))) == 6


def test_marks_check_validates_a_round_and_shows_progress(marks_cli: Path) -> None:
    options = ["--directory", str(marks_cli / "work"), "--records", str(marks_cli / "records")]
    result = runner.invoke(app, ["marks", "check", *options])
    assert result.exit_code == 0, result.output
    assert "marks for 3 of 3 receipts, 1 boxes" in result.output
    save_log(check_log({PLAIN: "correct"}), round_path(marks_cli / "work", "emre", 1))
    result = runner.invoke(app, ["marks", "check", *options])
    assert result.exit_code == 1
    assert f"{PLAIN}: not a receipt of this round" in result.output
    result = runner.invoke(app, ["marks", "check", "--round", "2", *options])
    assert result.exit_code == 1
    assert "round 1 is not checked yet" in result.output
    empty = ["--directory", str(marks_cli / "nothing"), "--records", str(marks_cli / "records")]
    result = runner.invoke(app, ["marks", "check", *empty])
    assert result.exit_code == 1
    assert "no log of round 1" in result.output


def test_marks_page_serves_the_committed_marks_until_interrupted(
    marks_cli: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []

    class FakeServer:
        server_port = 9998
        closed = False

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            FakeServer.closed = True

    monkeypatch.setattr(cli_marks, "make_server", lambda *_: FakeServer())
    monkeypatch.setattr("webbrowser.open", opened.append)
    records = marks_cli / "records"
    command = ["marks", "page", "--checker", "emre", *marks_options(marks_cli)]
    records.mkdir()
    shutil.copy(round_path(marks_cli / "work", ASSISTANT, 1), round_path(records, ASSISTANT, 1))
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    assert opened == ["http://127.0.0.1:9998/"]
    assert FakeServer.closed
    broken = mark_log({HAND: ReceiptMarks()})  # misses the other receipts of the round
    save_log(broken, round_path(records, ASSISTANT, 1))
    result = runner.invoke(app, command)
    assert result.exit_code == 1
    assert "the marks are not as planned" in result.output


def test_marks_report_needs_finished_rounds(marks_cli: Path) -> None:
    records, output = marks_cli / "records", marks_cli / "report.md"
    records.mkdir()
    shutil.copy(round_path(marks_cli / "work", ASSISTANT, 1), round_path(records, ASSISTANT, 1))
    options = ["--records", str(records), "--output", str(output)]
    result = runner.invoke(app, ["marks", "report", *options])
    assert result.exit_code == 1
    assert "the rounds are not finished" in result.output
    plan = load_mark_log(round_path(records, ASSISTANT, 1)).receipts
    save_log(check_log(dict.fromkeys(plan, "correct")), round_path(records, "emre", 1))
    result = runner.invoke(app, ["marks", "report", *options])
    assert result.exit_code == 0, result.output
    assert output.read_text(encoding="utf-8").startswith("# Benign marks: design fold")


@pytest.fixture
def synthesis_cli(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> tuple[Path, Engine]:
    """The review commands on drawn receipts: an engine of the design-fold round."""
    (tmp_path / "d").mkdir()
    documents, folds = synthesis_documents(tmp_path / "d", per_fold=12)
    statistics = RoundStatistics(
        DESIGN_FOLD,
        tuple(Template((Technique.CPI,), e, 1) for e in (Entity.PRODUCT, Entity.TOTAL_PAYMENT)),
        (4,),
        {},
        {},
    )
    engine = Engine(statistics, documents, folds, synthesis_receipt, synthesis_font)
    monkeypatch.setattr(cli_synthesis, "findit_documents", lambda *_, **__: documents)
    monkeypatch.setattr(cli_synthesis, "load_folds", lambda _: folds)
    monkeypatch.setattr(cli_synthesis, "cached_ocr", lambda _: None)
    monkeypatch.setattr(cli_synthesis, "round_engine", lambda *_: engine)
    monkeypatch.setattr(cli_synthesis, "require_clean_commit", lambda: None)
    monkeypatch.setattr(
        cli_synthesis, "code_version", lambda: CodeVersion(commit="abc1234", dirty=False)
    )
    return tmp_path, engine


def test_review_page_serves_until_interrupted(
    synthesis_cli: tuple[Path, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    opened: list[str] = []

    class FakeServer:
        server_port = 9997
        closed = False

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            FakeServer.closed = True

    monkeypatch.setattr(cli_synthesis, "make_server", lambda *_: FakeServer())
    monkeypatch.setattr("webbrowser.open", opened.append)
    records = synthesis_cli[0] / "review"
    result = runner.invoke(app, ["synthesis", "review", "--records", str(records)])
    assert result.exit_code == 0, result.output
    assert opened == ["http://127.0.0.1:9997/"]
    assert FakeServer.closed


def test_review_report_counts_a_finished_round(
    synthesis_cli: tuple[Path, Engine], monkeypatch: pytest.MonkeyPatch
) -> None:
    directory, engine = synthesis_cli
    records = directory / "review"
    output = directory / "report.md"
    arguments = ["synthesis", "review-report", "--records", str(records), "--output", str(output)]
    unfinished = runner.invoke(app, arguments)
    assert unfinished.exit_code == 1
    assert "not finished" in unfinished.output
    items = review_sample([f.record for f in development_set(engine)], 1)
    answers = {
        item.key: {
            "visibility": "not_found",
            "plausible": "yes",
            "display": {
                "device_pixel_ratio": 2.0,
                "screen_width": 1470,
                "screen_height": 956,
                "window_width": 1400,
                "window_height": 900,
            },
            "answered_at": "2026-10-15T10:00:00Z",
            "commit": "abc1234",
            "language": "tr",
        }
        for item in items
    }
    save_review(
        ReviewLog.model_validate({"reviewer": "emre", "round": 1, "answers": answers}),
        review_path(records, "emre", 1),
    )
    start = datetime(2026, 10, 12, 20, 0, tzinfo=UTC)
    observed = SimpleNamespace(
        regions={
            str(n): SimpleNamespace(
                visibility=Visibility.AT_100,
                observed_at=start + timedelta(seconds=8 * n),
                detailed=False,
            )
            for n in range(3)
        }
    )
    monkeypatch.setattr(cli_synthesis, "load_observations", lambda *_: observed)
    result = runner.invoke(app, arguments)
    assert result.exit_code == 0, result.output
    report = output.read_text()
    assert "Found at 100 %: 0 of 40; the criterion is **not exceeded**" in report
    assert "| real, step 4.2, visibility only | 2 | 8.000 |" in report


def flat(text: str) -> str:
    """Console output with line wrapping undone."""
    return " ".join(text.split())


def _llm_setup(toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    runner.invoke(app, ["data", "prepare", "--data-dir", str(toy_source)])
    monkeypatch.setattr(cli_common, "RapidOcrEngine", FakeOcr)
    monkeypatch.setattr(cli_llm, "ClaudeProvider", FakeProvider)
    train = [d.doc_id for d in load_toy(toy_source / "raw" / "findit2") if d.split is Split.TRAIN]
    monkeypatch.setattr(cli_llm, "GOLD_A", write_gold(tmp_path, tuple(train)))
    monkeypatch.setattr(cli_llm, "GOLD_B", tmp_path / "no-gold-b.json")
    instruction = tmp_path / "extraction-v1.md"
    instruction.write_text("Read the receipt.", encoding="utf-8")
    return [
        "--instruction",
        str(instruction),
        "--data-dir",
        str(toy_source),
        "--cache",
        str(tmp_path / "ocr"),
        "--store",
        str(tmp_path / "store"),
        "--ledger",
        str(tmp_path / "ledger.jsonl"),
    ]


def test_llm_compare_models_then_a_development_round(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = _llm_setup(toy_source, tmp_path, monkeypatch)
    report, choice = tmp_path / "comparison.md", tmp_path / "choice.json"
    compare = ["llm", "compare-models", *common, "--output", str(report), "--choice", str(choice)]
    result = runner.invoke(app, compare)
    assert result.exit_code == 0, result.output
    text = report.read_text(encoding="utf-8")
    assert "| claude-haiku-5-5 | claude-sonnet-5-5 |" in text  # one step: Sonnet is no better
    assert "claude-opus-5-5 |" not in text  # so Opus is never read
    assert text.endswith("**Chosen model:** `claude-haiku-5-5`.\n")
    assert json.loads(choice.read_text(encoding="utf-8"))["model"] == "claude-haiku-5-5"

    rounds = tmp_path / "rounds"
    develop = ["llm", "develop", "--round", "1", *common, "--choice", str(choice)]
    develop += ["--reports", str(rounds), "--effort", "low"]
    result = runner.invoke(app, develop)
    assert result.exit_code == 0, result.output
    round_report = (rounds / "round-01.md").read_text(encoding="utf-8")
    assert "thinking disabled, effort low" in round_report
    assert "| gold set A | 0 | 0 |" in round_report
    assert "| LLM | 0 | 0 |" in round_report
    again = runner.invoke(app, develop)
    assert again.exit_code == 1
    assert "never rewritten" in again.output
    eleventh = runner.invoke(app, [*develop[:3], "11", *develop[4:]])
    assert eleventh.exit_code == 1
    assert "numbered 1 to 10" in eleventh.output
    spending = runner.invoke(app, ["llm", "spending", "--ledger", str(tmp_path / "ledger.jsonl")])
    assert spending.exit_code == 0
    assert f"limit {LIMIT_USD} USD" in spending.output


def test_llm_development_refuses_receipts_of_gold_set_b(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = _llm_setup(toy_source, tmp_path, monkeypatch)
    train = [d.doc_id for d in load_toy(toy_source / "raw" / "findit2") if d.split is Split.TRAIN]
    gold_b = tmp_path / "gold-b" / "manifest.json"
    gold_b.parent.mkdir()
    gold_b.write_text(
        GoldManifest(
            dataset="findit2", split=Split.TRAIN, seed=1, authentic=(train[0],), forged=()
        ).model_dump_json(),
        encoding="utf-8",
    )
    monkeypatch.setattr(cli_llm, "GOLD_B", gold_b)
    output = tmp_path / "comparison.md"
    result = runner.invoke(app, ["llm", "compare-models", *common, "--output", str(output)])
    assert isinstance(result.exception, ForbiddenDocumentsError)
    assert not output.exists()


def test_llm_compare_models_climbs_while_the_larger_model_is_clearly_better(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = _llm_setup(toy_source, tmp_path, monkeypatch)

    def better(smaller: str, larger: str, *_: object) -> ModelStep:
        return ModelStep(smaller, larger, Estimate(0.3, 0.1, 0.5))

    monkeypatch.setattr(cli_llm, "model_step", better)
    report, choice = tmp_path / "comparison.md", tmp_path / "choice.json"
    compare = ["llm", "compare-models", *common, "--output", str(report), "--choice", str(choice)]
    assert runner.invoke(app, compare).exit_code == 0
    text = report.read_text(encoding="utf-8")
    assert "| claude-sonnet-5-5 | claude-opus-5-5 |" in text
    assert text.endswith("**Chosen model:** `claude-opus-5-5`.\n")
    rounds = tmp_path / "rounds"
    develop = ["llm", "develop", "--round", "2", *common, "--choice", str(choice)]
    result = runner.invoke(app, [*develop, "--reports", str(rounds), "--thinking", "adaptive"])
    assert result.exit_code == 0, result.output
    assert "Model `claude-opus-5-5`, thinking adaptive, effort low" in (
        rounds / "round-02.md"
    ).read_text(encoding="utf-8")


def test_llm_freeze_the_best_round_then_read_gold_set_a_again(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = _llm_setup(toy_source, tmp_path, monkeypatch)
    choice, rounds, frozen = tmp_path / "choice.json", tmp_path / "rounds", tmp_path / "frozen.json"
    runner.invoke(
        app,
        [
            "llm",
            "compare-models",
            *common,
            "--output",
            str(tmp_path / "c.md"),
            "--choice",
            str(choice),
        ],
    )
    develop = ["llm", "develop", *common, "--choice", str(choice), "--reports", str(rounds)]
    assert runner.invoke(app, [*develop, "--round", "1"]).exit_code == 0
    assert runner.invoke(app, [*develop, "--round", "2", "--effort", "low"]).exit_code == 0
    assert (
        json.loads((rounds / "round-02.json").read_text(encoding="utf-8"))["settings"]["effort"]
        == "low"
    )

    stability = ["llm", "stability", *common[2:], "--frozen", str(frozen)]
    stability += ["--output", str(tmp_path / "stability.md")]
    early = runner.invoke(app, stability)
    assert early.exit_code == 1
    assert "freeze a development round first" in flat(early.output)

    monkeypatch.setattr(cli_llm, "require_clean_commit", lambda: None)
    monkeypatch.setattr(cli_llm, "code_version", lambda: CodeVersion(commit="abc123", dirty=False))
    freeze = ["llm", "freeze", "--reports", str(rounds), "--frozen", str(frozen)]
    result = runner.invoke(app, freeze)
    assert result.exit_code == 0, result.output
    version = json.loads(frozen.read_text(encoding="utf-8"))
    assert (version["round"], version["commit"]) == (1, "abc123")  # equal rounds: the earlier
    again = runner.invoke(app, freeze)
    assert again.exit_code == 1
    assert "frozen once" in flat(again.output)

    result = runner.invoke(app, stability)
    assert result.exit_code == 0, result.output
    assert "| T1 transcript |" in (tmp_path / "stability.md").read_text(encoding="utf-8")

    Path(version["instruction"]).write_text("Read the receipt carefully.", encoding="utf-8")
    changed = runner.invoke(app, stability)
    assert changed.exit_code == 1
    assert "changed after it was frozen" in flat(changed.output)


def test_llm_freeze_refuses_an_instruction_changed_after_its_round(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    common = _llm_setup(toy_source, tmp_path, monkeypatch)
    choice, rounds = tmp_path / "choice.json", tmp_path / "rounds"
    runner.invoke(
        app,
        [
            "llm",
            "compare-models",
            *common,
            "--output",
            str(tmp_path / "c.md"),
            "--choice",
            str(choice),
        ],
    )
    runner.invoke(
        app,
        [
            "llm",
            "develop",
            "--round",
            "1",
            *common,
            "--choice",
            str(choice),
            "--reports",
            str(rounds),
        ],
    )
    Path(common[1]).write_text("Something else.", encoding="utf-8")
    monkeypatch.setattr(cli_llm, "require_clean_commit", lambda: None)
    result = runner.invoke(
        app, ["llm", "freeze", "--reports", str(rounds), "--frozen", str(tmp_path / "f.json")]
    )
    assert result.exit_code == 1
    assert "changed after round 1" in flat(result.output)


def _write_gold_set(directory: Path, authentic: tuple[str, ...], forged: tuple[str, ...]) -> Path:
    directory.mkdir(parents=True)
    manifest = GoldManifest(
        dataset="findit2", split=Split.TRAIN, seed=0, authentic=authentic, forged=forged
    )
    (directory / "manifest.json").write_text(manifest.model_dump_json(), encoding="utf-8")
    for doc_id in (*authentic, *forged):
        record = {"doc_id": doc_id, "annotator": "t", "fields": {"amounts": {"total_due": ["5"]}}}
        (directory / f"{doc_id}.json").write_text(json.dumps(record), encoding="utf-8")
    return directory / "manifest.json"


def _gold_b_setup(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], Path]:
    """Gold set A is F2; gold set B is A1 (authentic) and F1 (forged); round 1 is frozen."""
    common = _llm_setup(toy_source, tmp_path, monkeypatch)
    monkeypatch.setattr(cli_llm, "GOLD_A", _write_gold_set(tmp_path / "gold-a", (), ("F2",)))
    monkeypatch.setattr(cli_llm, "GOLD_B", _write_gold_set(tmp_path / "gold-b", ("A1",), ("F1",)))
    monkeypatch.setattr(cli_llm, "require_clean_commit", lambda: None)
    monkeypatch.setattr(cli_llm, "code_version", lambda: CodeVersion(commit="def456", dirty=False))
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: [])
    instruction = Path(common[1])
    frozen = tmp_path / "frozen.json"
    frozen.write_text(
        json.dumps(
            {
                "round": 1,
                "settings": COMPARISON_SETTINGS["claude-haiku-5-5"].model_dump(mode="json"),
                "instruction": str(instruction),
                "instruction_sha256": hashlib.sha256(instruction.read_bytes()).hexdigest(),
                "commit": "abc123",
            }
        ),
        encoding="utf-8",
    )
    paths = ["--measurement", str(tmp_path / "b" / "m.json")]
    paths += ["--questions", str(tmp_path / "b" / "q.json")]
    return [*common[2:], "--frozen", str(frozen), *paths], tmp_path / "b"


def test_llm_gold_set_b_is_read_once_then_served_for_adjudication(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measure, directory = _gold_b_setup(toy_source, tmp_path, monkeypatch)
    log = tmp_path / "adjudication.json"
    adjudicate = ["llm", "adjudicate", "--questions", str(directory / "q.json"), "--log", str(log)]
    early = runner.invoke(app, [*adjudicate, "--no-browser", "--data-dir", measure[1]])
    assert early.exit_code == 1
    assert "run docfraud llm measure-b first" in flat(early.output)

    monkeypatch.setattr(cli_llm, "ClaudeProvider", lambda **_: FakeProvider(fail=frozenset({0})))
    failed = runner.invoke(app, ["llm", "measure-b", *measure])
    assert failed.exit_code == 1
    assert "1 of 4 requests were not answered" in flat(failed.output)
    assert not (directory / "m.json").exists()  # nothing is kept

    sent: list[FakeProvider] = []

    def provider(**_: object) -> FakeProvider:
        sent.append(FakeProvider())
        return sent[-1]

    monkeypatch.setattr(cli_llm, "ClaudeProvider", provider)
    result = runner.invoke(app, ["llm", "measure-b", *measure])
    assert result.exit_code == 0, result.output
    assert sent[0].calls == [(1, False)]  # the one left unanswered, sent on its own
    assert "read 2 receipts of gold set B; 2 questions on 2 receipts" in flat(result.output)
    assert "no metric is computed" in result.output
    again = runner.invoke(app, ["llm", "measure-b", *measure])
    assert again.exit_code == 1
    assert "read once" in flat(again.output)

    served: list[Any] = []

    class FakeServer:
        server_port = 9996

        def serve_forever(self) -> None:
            raise KeyboardInterrupt

        def server_close(self) -> None:
            pass

    def make_server(page: Any, _: int) -> FakeServer:
        served.append(page)
        return FakeServer()

    monkeypatch.setattr(cli_llm, "make_server", make_server)
    opened: list[str] = []
    monkeypatch.setattr("webbrowser.open", opened.append)
    assistant = runner.invoke(app, [*adjudicate, "--reviewer", "claude", "--data-dir", measure[1]])
    assert assistant.exit_code == 1
    assert "only the owner adjudicates" in flat(assistant.output)
    result = runner.invoke(app, [*adjudicate, "--data-dir", measure[1]])
    assert result.exit_code == 0, result.output
    assert "0 of 2 questions done" in flat(result.output)
    assert opened == ["http://127.0.0.1:9996/"]
    assert served[0].handle("GET", "/image/A1.png").status == 200


def test_llm_gold_set_b_is_evaluated_once_after_every_answer(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measure, directory = _gold_b_setup(toy_source, tmp_path, monkeypatch)
    log, report = tmp_path / "adjudication.json", tmp_path / "gold-b.md"
    evaluate = ["llm", "evaluate-b", "--measurement", str(directory / "m.json")]
    evaluate += ["--questions", str(directory / "q.json"), "--log", str(log)]
    evaluate += ["--output", str(report)]
    unread = runner.invoke(app, evaluate)
    assert unread.exit_code == 1
    assert "gold set B has not been read" in flat(unread.output)
    assert runner.invoke(app, ["llm", "measure-b", *measure]).exit_code == 0
    unfinished = runner.invoke(app, evaluate)
    assert unfinished.exit_code == 1
    assert "2 of 2 questions have no final answer" in flat(unfinished.output)

    asked = load_questions(directory / "q.json")
    printed = Listing(amounts=(Decimal(5),))  # the owner reads what the gold set says
    answers = {q.key: first_answer(q, printed, datetime.now(UTC), "def456") for q in asked}
    save_adjudication(AdjudicationLog(reviewer="emre", answers=answers), log)
    other = runner.invoke(app, [*evaluate, "--reviewer", "someone"])
    assert other.exit_code == 1
    assert "belongs to emre" in flat(other.output)
    kept = (directory / "m.json").read_text(encoding="utf-8")
    partial = json.loads(kept)
    partial["readings"] = partial["readings"][1:]
    (directory / "m.json").write_text(json.dumps(partial), encoding="utf-8")
    short = runner.invoke(app, evaluate)
    assert short.exit_code == 1
    assert "does not cover the receipts" in flat(short.output)
    (directory / "m.json").write_text(kept, encoding="utf-8")
    result = runner.invoke(app, evaluate)
    assert result.exit_code == 0, result.output
    assert "T1 transcript: parser v2; T2 OCR: parser v2" in flat(result.output)
    text = report.read_text(encoding="utf-8")
    assert "read the 2 receipts of gold set B (1 authentic, 1 forged) once" in text
    assert "| T1 transcript | 0 of 2 | 0 of 2 | +0.000 [" in text
    assert "| Amounts of one role | 2 | 2 | 0 |" in text
    assert "| T1 transcript | 0 | 2 | 1.50 | 1.50 |" in text  # every answer timed
    assert "## T2 OCR" in text
    once = runner.invoke(app, evaluate)
    assert once.exit_code == 1
    assert "runs once" in flat(once.output)


def test_llm_gold_set_b_is_read_only_by_the_frozen_reader(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    measure, directory = _gold_b_setup(toy_source, tmp_path, monkeypatch)
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: ["src/docfraud/llm/reader.py"])
    changed = runner.invoke(app, ["llm", "measure-b", *measure])
    assert changed.exit_code == 1
    assert "the reader changed after it was frozen at abc123: src/docfraud/llm/reader.py" in flat(
        changed.output
    )
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: None)
    unknown = runner.invoke(app, ["llm", "measure-b", *measure])
    assert unknown.exit_code == 1
    assert "cannot compare with commit abc123" in flat(unknown.output)

    monkeypatch.setattr(cli_common, "changed_since", lambda *_: [])
    (tmp_path / "gold-b" / "F1.json").rename(tmp_path / "F1.json")
    unannotated = runner.invoke(app, ["llm", "measure-b", *measure])
    assert unannotated.exit_code == 1
    assert "not fully annotated" in flat(unannotated.output)
    (tmp_path / "F1.json").rename(tmp_path / "gold-b" / "F1.json")
    monkeypatch.setattr(cli_llm, "GOLD_A", _write_gold_set(tmp_path / "overlap", ("A1",), ()))
    result = runner.invoke(app, ["llm", "measure-b", *measure])
    assert isinstance(result.exception, ForbiddenDocumentsError)
    assert not directory.exists()


def _false_alarms_setup(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], Path]:
    """Gold set A is F2 and gold set B is empty, so the test reads A1 (authentic) and F1."""
    measure, _ = _gold_b_setup(toy_source, tmp_path, monkeypatch)
    monkeypatch.setattr(cli_llm, "GOLD_B", _write_gold_set(tmp_path / "empty-b", (), ()))
    report = tmp_path / "false-alarms.md"
    common = measure[: measure.index("--measurement")]
    return [*common, "--measurement", str(tmp_path / "fa" / "m.json"), "--output", str(report)], (
        report
    )


def test_llm_false_alarms_are_counted_once_outside_the_gold_sets(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, report = _false_alarms_setup(toy_source, tmp_path, monkeypatch)
    monkeypatch.setattr(cli_llm, "ClaudeProvider", lambda **_: FakeProvider(fail=frozenset({0})))
    failed = runner.invoke(app, ["llm", "false-alarms", *arguments])
    assert failed.exit_code == 1
    assert "1 of 4 requests were not answered" in flat(failed.output)
    assert not report.exists()
    assert not (tmp_path / "fa").exists()  # nothing is kept

    monkeypatch.setattr(cli_llm, "ClaudeProvider", FakeProvider)
    result = runner.invoke(app, ["llm", "false-alarms", *arguments])
    assert result.exit_code == 0, result.output
    assert "T1 transcript: parser v2; T2 OCR: parser v2" in flat(result.output)
    text = report.read_text(encoding="utf-8")
    assert (
        "read the 2 receipts of the training split outside gold sets A and B (1 authentic, "
        "1 forged)" in text
    )
    assert "| T1 transcript | 0 of 1 | 0 of 1 | +0.000 [+0.000, +0.000] | 0 | 0 | 1.0000 |" in text
    assert "### Rules that fail, T2 OCR" in text
    assert load_measurement(tmp_path / "fa" / "m.json").receipts == 2
    once = runner.invoke(app, ["llm", "false-alarms", *arguments])
    assert once.exit_code == 1
    assert "runs once" in flat(once.output)


def test_llm_false_alarms_need_unchanged_accounting_rules(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, report = _false_alarms_setup(toy_source, tmp_path, monkeypatch)

    def changed(commit: str, paths: Sequence[str]) -> list[str]:
        return list(paths) if commit == FIXED_AT else []

    monkeypatch.setattr(cli_common, "changed_since", changed)
    result = runner.invoke(app, ["llm", "false-alarms", *arguments])
    assert result.exit_code == 1
    assert (
        "the accounting rules changed after ADR-0027 at 888aeb65c1d0: "
        "src/docfraud/detectors/rules.py" in flat(result.output)
    )
    assert not report.exists()


def _validation_setup(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[list[str], Path]:
    """The validation split of the toy dataset is A2; round 1 is frozen."""
    measure, _ = _gold_b_setup(toy_source, tmp_path, monkeypatch)
    record = tmp_path / "validation-reading.json"
    return [*measure[: measure.index("--measurement")], "--record", str(record)], record


def test_llm_the_validation_split_is_read_once_and_nothing_is_measured(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, record = _validation_setup(toy_source, tmp_path, monkeypatch)
    monkeypatch.setattr(cli_llm, "ClaudeProvider", lambda **_: FakeProvider(fail=frozenset({0})))
    failed = runner.invoke(app, ["llm", "read-validation", *arguments])
    assert failed.exit_code == 1
    assert "1 of 2 requests were not answered" in flat(failed.output)
    assert not record.exists()  # nothing is recorded

    monkeypatch.setattr(cli_llm, "ClaudeProvider", FakeProvider)
    result = runner.invoke(app, ["llm", "read-validation", *arguments])
    assert result.exit_code == 0, result.output
    assert (  # the pilot of each run paid one request at the full price
        "read 1 validation receipts from both text sources: 2 requests, 0.0004 USD, 0 without "
        "an answer" in flat(result.output)
    )
    reading = load_split_reading(record)
    assert (reading.split, reading.receipts, reading.commit) == (Split.VAL, 1, "def456")
    once = runner.invoke(app, ["llm", "read-validation", *arguments])
    assert once.exit_code == 1
    assert "the validation split is read once" in flat(once.output)


def test_llm_the_validation_split_is_read_only_by_the_frozen_reader(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, record = _validation_setup(toy_source, tmp_path, monkeypatch)
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: ["src/docfraud/llm/reader.py"])
    result = runner.invoke(app, ["llm", "read-validation", *arguments])
    assert result.exit_code == 1
    assert "the reader changed after it was frozen at abc123" in flat(result.output)
    assert not record.exists()


def test_llm_the_test_split_is_read_once_at_the_freeze_by_a_reader_shown_unchanged(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    arguments, _ = _validation_setup(toy_source, tmp_path, monkeypatch)
    clean = CodeVersion(commit="d" * 40, dirty=False)
    monkeypatch.setattr(runs_module, "code_version", lambda repo=None: clean)  # noqa: ARG005
    monkeypatch.setattr(cli_llm, "ClaudeProvider", FakeProvider)
    runs = tmp_path / "runs"
    given = arguments[: arguments.index("--record")]
    read_test = ["llm", "read-test", *given, "--runs", str(runs)]
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: None)
    assert "no tag phase-7-freeze" in flat(runner.invoke(app, read_test).output)
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: ["src/docfraud/llm/agent.py"])
    assert "the source changed after phase-7-freeze" in flat(runner.invoke(app, read_test).output)
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: [])
    unshown = flat(runner.invoke(app, read_test).output)
    assert "2 of 2 requests of the validation reading are not the stored ones" in unshown
    assert runner.invoke(app, ["llm", "read-validation", *arguments]).exit_code == 0
    reading = read_document

    def distinct(document: Document, *rest: Any) -> Any:  # the toy receipts read alike
        found = reading(document, *rest)
        if document.split is not Split.TEST:
            return found
        return replace(found, rows=(*found.rows, Row((Segment(document.doc_id),))))

    monkeypatch.setattr(cli_llm, "read_document", distinct)
    monkeypatch.setattr(cli_llm, "ClaudeProvider", lambda **_: FakeProvider(fail=frozenset({0})))
    assert "requests were not answered" in flat(runner.invoke(app, read_test).output)
    assert not JsonRunStore(runs).records("llm-test-reading")  # nothing is recorded
    monkeypatch.setattr(cli_llm, "ClaudeProvider", FakeProvider)
    result = runner.invoke(app, read_test)
    assert result.exit_code == 0, result.output
    (made,) = JsonRunStore(runs).records("llm-test-reading")
    assert made.test_unlocks == (cli_common.TEST_UNLOCK,)
    assert made.results["receipts"] >= 1
    assert made.results["requests"] == 2 * made.results["receipts"]  # both text sources
    assert made.results["validation_requests_found"] == 2
    assert "refusing to run again" in flat(runner.invoke(app, read_test).output)


def _detector_setup(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> list[str]:
    """Eight design-fold receipts, both instructions, a fake detector and a clean commit."""
    docs = [d for d in corpus(tmp_path / "corpus") if d.split is Split.TRAIN][:8]
    for shade, document in enumerate(docs, start=200):  # one page per receipt, one request each
        Image.new("L", (40, 30), shade).save(document.image_path)
    instructions = tmp_path / "instructions"
    instructions.mkdir()
    written = (Path("src/docfraud/llm/instructions/detector-v1.md")).read_text(encoding="utf-8")
    (instructions / "detector-v1.md").write_text(written, encoding="utf-8")
    (instructions / "detector-v2.md").write_text(written + "\nBe brief.\n", encoding="utf-8")
    answering = Runner(
        FakeDetector(), ResponseStore(tmp_path / "store"), Ledger(tmp_path / "ledger.jsonl")
    )
    clean = CodeVersion(commit="b" * 40, dirty=False)
    monkeypatch.setattr(cli_common, "load_findit2", lambda _: tuple(docs))
    monkeypatch.setattr(cli_common, "RapidOcrEngine", FakeOcr)
    monkeypatch.setattr(cli_detector, "DESIGN_FOLD_IDS", frozenset(d.doc_id for d in docs))
    monkeypatch.setattr(cli_detector, "INSTRUCTIONS", instructions)
    monkeypatch.setattr(cli_detector, "_runner", lambda *_: answering)
    monkeypatch.setattr(cli_detector, "code_version", lambda: clean)
    monkeypatch.setattr(cli_common, "code_version", lambda: clean)
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: [])
    return ["--rounds", str(tmp_path / "rounds"), "--cache", str(tmp_path / "ocr")]


def test_the_detector_is_developed_frozen_and_timed_on_the_design_fold_only(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = _detector_setup(tmp_path, monkeypatch)
    rounds = tmp_path / "rounds"
    clean = CodeVersion(commit="b" * 40, dirty=False)

    def develop(*more: str) -> Any:
        return runner.invoke(app, ["detector", "develop", *options, *more])

    assert "run version 1" in develop("--version", "2").output
    assert "versions (1, 2) only" in develop("--version", "3").output
    pilot = develop("--version", "1", "--pilot")
    assert pilot.exit_code == 0, pilot.output
    assert pilot.output.count("transcript ·") == 5
    monkeypatch.setattr(cli_common, "code_version", lambda: CodeVersion(commit="b", dirty=True))
    assert "commit all changes first" in develop("--version", "1").output
    monkeypatch.setattr(cli_common, "code_version", lambda: clean)
    for version in ("1", "2"):
        result = develop("--version", version)
        assert result.exit_code == 0, result.output
    developed = load_rounds(rounds)
    transcript = developed[1].rows["transcript"]
    assert (transcript.receipts, transcript.forged, transcript.failures) == (8, 2, 0)
    assert transcript.average_precision == pytest.approx(1.0)  # its text says EDITED
    assert developed[1].commit == clean.commit
    freeze = ["detector", "freeze", "--rounds", str(rounds)]
    assert runner.invoke(app, freeze).exit_code == 0
    frozen = FrozenDetector.model_validate_json((rounds / "frozen.json").read_text("utf-8"))
    assert frozen.version == 1  # version 2 is not clearly better
    assert "frozen once" in runner.invoke(app, freeze).output
    assert "development has ended" in develop("--version", "2").output
    timed = runner.invoke(app, ["detector", "latency", *options])
    assert timed.exit_code == 0, timed.output
    latency = json.loads((rounds / "latency.json").read_text(encoding="utf-8"))
    assert latency["image and OCR text"]["timed"] == 8
    assert latency["transcript"]["median_seconds"] == pytest.approx(1.5)


def test_the_frozen_main_row_is_checked_once_on_the_stronger_model(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    options = _detector_setup(tmp_path, monkeypatch)
    rounds = tmp_path / "rounds"
    stronger = ["detector", "stronger-model", *options]
    assert "freeze a version first" in flat(runner.invoke(app, stronger).output)
    assert runner.invoke(app, ["detector", "develop", "--version", "1", *options]).exit_code == 0
    assert runner.invoke(app, ["detector", "freeze", "--rounds", str(rounds)]).exit_code == 0
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: ["src/docfraud/llm/detector.py"])
    assert "the detector changed after it was frozen" in flat(runner.invoke(app, stronger).output)
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: [])
    checked = runner.invoke(app, stronger)
    assert checked.exit_code == 0, checked.output
    record = StrongerModelCheck.model_validate_json(
        (rounds / "stronger-model.json").read_text(encoding="utf-8")
    )
    assert record.settings == OPUS
    assert record.instruction.endswith("detector-v1.md")
    assert record.result.receipts == record.frozen_result.receipts == 8
    assert not record.row_added  # the same answers are not clearly better
    spent = [json.loads(line) for line in (tmp_path / "ledger.jsonl").read_text().splitlines()]
    assert sum(entry["model"] == OPUS.model for entry in spent) == 8  # the main row only
    assert "checked once" in flat(runner.invoke(app, stronger).output)


def test_the_detector_refuses_to_freeze_or_time_what_it_cannot_trust(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    rounds = tmp_path / "rounds"
    clean = CodeVersion(commit="c" * 40, dirty=False)
    monkeypatch.setattr(cli_common, "code_version", lambda: clean)
    monkeypatch.setattr(cli_detector, "code_version", lambda: clean)
    freeze = ["detector", "freeze", "--rounds", str(rounds)]
    latency = ["detector", "latency", "--rounds", str(rounds)]
    assert "version 1 has not run" in " ".join(runner.invoke(app, freeze).output.split())
    assert "freeze a version first" in " ".join(runner.invoke(app, latency).output.split())
    instruction = tmp_path / "detector-v1.md"
    instruction.write_text("first", encoding="utf-8")
    developed = development(1, 0.5).model_copy(
        update={"instruction": str(instruction)}  # its hash is not the file's: it changed
    )
    rounds.mkdir()
    (rounds / "round-v1.json").write_text(developed.model_dump_json(), encoding="utf-8")
    assert "changed after version 1 ran" in " ".join(runner.invoke(app, freeze).output.split())
    frozen = developed.model_dump(mode="json", exclude={"rows"})
    (rounds / "frozen.json").write_text(json.dumps(frozen), encoding="utf-8")
    assert "changed after it was frozen" in " ".join(runner.invoke(app, latency).output.split())


@pytest.mark.parametrize(
    ("computed", "kept"), [(False, "claude-haiku-5-5"), (True, "claude-sonnet-5-5")]
)
def test_the_agent_is_piloted_once_on_three_cases_and_keeps_a_model_by_its_rule(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    computed: bool,
    kept: str,
) -> None:
    pilot = ("R1", "R2", "R3")
    documents = [
        Document(
            dataset="toy",
            doc_id=r,
            source_id=r,
            split=Split.TRAIN,
            label=Label.AUTHENTIC if r == "R3" else Label.FORGED,
            image_path=tmp_path / f"{r}.png",
        )
        for r in pilot
    ]
    found = SimpleNamespace(assessments={})
    development = SimpleNamespace(pilot=pilot)
    monkeypatch.setattr(cli_agent, "_training", lambda *_: (documents, development, found))
    monkeypatch.setattr(cli_agent, "_reader", lambda *_: None)
    monkeypatch.setattr(
        cli_agent, "cases", lambda chosen, *_: {d.doc_id: agent_case(d.doc_id) for d in chosen}
    )
    monkeypatch.setattr(cli_agent, "DESIGN_FOLD_IDS", frozenset(pilot))
    books = (ResponseStore(tmp_path / "store"), Ledger(tmp_path / "ledger.jsonl"))
    monkeypatch.setattr(
        cli_agent, "_runner", lambda *_: Runner(FakeAgent(computed=computed), *books)
    )
    clean = CodeVersion(commit="d" * 40, dirty=False)
    monkeypatch.setattr(cli_agent, "code_version", lambda: clean)
    monkeypatch.setattr(cli_common, "code_version", lambda: CodeVersion(commit="d", dirty=True))
    output = tmp_path / "agent"
    command = ["agent", "pilot", "--output", str(output), "--cache", str(tmp_path / "ocr")]
    assert "commit all changes first" in said(runner.invoke(app, command))
    monkeypatch.setattr(cli_common, "code_version", lambda: clean)
    result = runner.invoke(app, command)
    assert result.exit_code == 0, result.output
    record = json.loads((output / "pilot.json").read_text(encoding="utf-8"))
    assert record["kept"]["model"] == kept
    assert record["cases"] == list(pilot)
    haiku = record["models"]["claude-haiku-5-5"]
    assert [c["receipt"] for c in haiku] == list(pilot)
    assert [c["forged"] for c in haiku] == [True, True, False]
    assert haiku[0]["tool_calls"][0] == {"tool": "assessment", "arguments": {}}
    assert bool(haiku[0]["grounding"]["violations"]) is computed
    assert record["token_budget"] == 8000  # twice 3,600 tokens, rounded up to a thousand
    assert len(list((output / "pilot").glob("*.md"))) == 6
    assert "Case file: R1" in (output / "pilot" / "claude-sonnet-5-5-R1.md").read_text("utf-8")
    assert "the pilot runs once" in said(runner.invoke(app, command))


def test_the_agent_is_developed_once_per_version_and_frozen_by_its_rule(  # noqa: PLR0915
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    documents = [
        agent_document("R1", Label.FORGED, AGENT_BOX),
        agent_document("R2", Label.FORGED, ELSEWHERE),
        agent_document("R3", Label.AUTHENTIC),
    ]
    receipts = tuple(d.doc_id for d in documents)
    found = SimpleNamespace(assessments={r: agent_case(r).assessment for r in receipts})
    development = SimpleNamespace(receipts=receipts)
    monkeypatch.setattr(cli_agent, "_training", lambda *_: (documents, development, found))
    monkeypatch.setattr(cli_agent, "_reader", lambda *_: None)
    monkeypatch.setattr(
        cli_agent, "cases", lambda chosen, *_: {d.doc_id: agent_case(d.doc_id) for d in chosen}
    )
    monkeypatch.setattr(cli_agent, "DESIGN_FOLD_IDS", frozenset(receipts))
    instructions = tmp_path / "instructions"
    instructions.mkdir()
    (instructions / "agent-v1.md").write_text("Draft the case file.", encoding="utf-8")
    monkeypatch.setattr(cli_agent, "INSTRUCTIONS", instructions)
    books = (ResponseStore(tmp_path / "store"), Ledger(tmp_path / "ledger.jsonl"))
    computed = Runner(FakeAgent(computed=True, silent=1), *books)
    monkeypatch.setattr(cli_agent, "_runner", lambda *_: computed)
    clean = CodeVersion(commit="d" * 40, dirty=False)
    monkeypatch.setattr(cli_agent, "code_version", lambda: clean)
    monkeypatch.setattr(cli_common, "code_version", lambda: clean)
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: [])
    output = tmp_path / "agent"
    options = ["--output", str(output), "--cache", str(tmp_path / "ocr")]
    develop = ["agent", "develop", *options, "--version"]
    freeze = ["agent", "freeze", "--output", str(output)]
    assert "versions (1, 2) only" in said(runner.invoke(app, [*develop, "3"]))
    assert "run version 1 on the development cases first" in said(
        runner.invoke(app, [*develop, "2"])
    )
    assert "run the pilot first" in said(runner.invoke(app, [*develop, "1"]))
    output.mkdir()
    pilot = {"kept": SONNET_AGENT.model_dump(mode="json"), "token_budget": 34_000}
    (output / "pilot.json").write_text(json.dumps(pilot), encoding="utf-8")
    assert "version 1 has not run" in said(runner.invoke(app, freeze))
    assert "1 of 3 cases were not answered (R1); nothing is recorded" in said(
        runner.invoke(app, [*develop, "1"])
    )
    assert not (output / "round-v1.json").exists()
    first = runner.invoke(app, [*develop, "1"])  # sends only the step that was not answered
    assert first.exit_code == 0, first.output
    (v1,) = load_agent_rounds(output).values()
    assert (v1.settings, v1.token_budget) == (SONNET_AGENT, 34_000)
    assert [c.receipt for c in v1.cases] == list(receipts)
    assert (v1.failures, v1.violations) == (0, 3)  # every case file states a computed number
    assert [c.first_location_hit for c in v1.cases] == [True, False, None]
    assert (v1.hits, v1.report_hits) == (1, 1)
    assert sorted(p.name for p in (output / "v1").iterdir()) == [f"{r}.md" for r in receipts]
    assert "each version runs once" in said(runner.invoke(app, [*develop, "1"]))
    assert "write version 2 first" in said(runner.invoke(app, [*develop, "2"]))
    (instructions / "agent-v2.md").write_text("Draft the case file. Quote.", encoding="utf-8")
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: ["src/docfraud/llm/agent.py"])
    assert "the agent's code changed after version 1 ran" in said(
        runner.invoke(app, [*develop, "2"])
    )
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: [])
    grounded = Runner(FakeAgent(), *books)
    monkeypatch.setattr(cli_agent, "_runner", lambda *_: grounded)
    second = runner.invoke(app, [*develop, "2"])
    assert second.exit_code == 0, second.output
    assert load_agent_rounds(output)[2].violations == 0
    (instructions / "agent-v2.md").write_text("Changed.", encoding="utf-8")
    assert "changed after version 2 ran" in said(runner.invoke(app, freeze))
    (instructions / "agent-v2.md").write_text("Draft the case file. Quote.", encoding="utf-8")
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: ["src/docfraud/llm/case_file.py"])
    assert "the agent's code changed after version 2 ran" in said(runner.invoke(app, freeze))
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: [])
    frozen = runner.invoke(app, freeze)
    assert frozen.exit_code == 0, frozen.output
    kept = FrozenAgent.model_validate_json((output / "frozen.json").read_text(encoding="utf-8"))
    assert (kept.version, kept.token_budget, kept.commit) == (2, 34_000, "d" * 40)
    assert kept.instruction.endswith("agent-v2.md")
    assert "the agent is frozen once" in said(runner.invoke(app, freeze))
    assert "development has ended" in said(runner.invoke(app, [*develop, "2"]))


def test_the_frozen_agent_runs_once_on_the_validation_queue_of_phase_5s_check(
    recorded: Recorded, monkeypatch: pytest.MonkeyPatch
) -> None:
    tmp = recorded.tmp
    assert recorded.shift(tmp / "shift.md").exit_code == 0
    assert recorded.check().exit_code == 0
    monkeypatch.setattr(cli_agent, "_reader", lambda *_: None)
    monkeypatch.setattr(
        cli_agent, "cases", lambda chosen, *_: {d.doc_id: agent_case(d.doc_id) for d in chosen}
    )
    monkeypatch.setattr(cli_agent, "_pages", lambda documents: pages(list(documents)))
    monkeypatch.setattr(cli_agent, "agent_validation", partial(agent_validation, resamples=20))
    books = (ResponseStore(tmp / "store"), Ledger(tmp / "ledger.jsonl"))
    unanswered = Runner(FakeAgent(silent=1), *books)
    monkeypatch.setattr(cli_agent, "_runner", lambda *_: unanswered)
    output, report = tmp / "agent", tmp / "agent" / "validation.md"
    monkeypatch.setattr(  # files written before the record is saved would make its tree dirty
        runs_module,
        "code_version",
        lambda repo=None: CodeVersion(commit="a" * 40, dirty=(output / "validation").exists()),  # noqa: ARG005
    )

    def validate(
        validation: Path = recorded.paths["validation"], runs: Path = recorded.runs
    ) -> Any:
        command = ["agent", "validation", "--runs", str(runs), "--output", str(output)]
        command += ["--training", str(recorded.paths["training"])]
        command += ["--validation", str(validation), "--cache", str(tmp / "ocr")]
        return runner.invoke(app, [*command, "--report", str(report)])

    assert "freeze the agent first" in said(validate())
    instruction = tmp / "agent-v2.md"
    instruction.write_text("Draft the case file.", encoding="utf-8")
    sha = hashlib.sha256(instruction.read_text("utf-8").encode()).hexdigest()
    frozen = FrozenAgent(
        version=2,
        settings=SONNET_AGENT,
        token_budget=34_000,
        instruction=str(instruction),
        instruction_sha256=sha,
        commit="a" * 40,
    )
    output.mkdir()
    (output / "frozen.json").write_text(frozen.model_dump_json(), encoding="utf-8")
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: ["src/docfraud/llm/agent.py"])
    assert "the agent changed after it was frozen" in said(validate())
    monkeypatch.setattr(cli_common, "changed_since", lambda *_: [])
    assert "Phase 5's check comes first" in said(validate(runs=tmp / "no-runs"))
    other = tmp / "other.json"
    other.write_text(recorded.paths["validation"].read_text("utf-8") + "\n", encoding="utf-8")
    assert "is not the table" in said(validate(other))
    assert "1 of 30 cases were not answered" in said(validate())
    assert not recorded.store.records("llm-agent-validation")
    result = validate()  # sends only the step that was not answered
    assert result.exit_code == 0, result.output
    (made,) = recorded.store.records("llm-agent-validation")
    assert made.code.dirty is False
    assert (
        made.config["validation_check"]
        == recorded.store.records("fusion-validation-check")[0].run_id
    )
    assert made.config["frozen"]["version"] == 2
    assert made.config["policy"]["budget"] == 0.1
    assert made.results["receipts"] == len(made.results["cases"]) == 30
    assert made.results["grounding"]["violations"] == []
    assert made.results["first_location"]["receipts"] == made.results["forged"] == 10
    assert len(list((output / "validation").glob("*.md"))) == 30
    assert "# The investigation agent on the validation split" in report.read_text("utf-8")
    assert "refusing to run again" in said(validate())
    again = tmp / "again.md"
    rendered = ["agent", "validation-report", "--runs", str(recorded.runs)]
    assert runner.invoke(app, [*rendered, "--report", str(again)]).exit_code == 0
    assert again.read_text(encoding="utf-8") == report.read_text(encoding="utf-8")


def test_the_agent_sees_each_receipt_through_the_systems_outputs(tmp_path: Path) -> None:
    image = tmp_path / "R1.png"
    Image.new("L", (40, 30), 200).save(image)
    document = Document(
        dataset="toy",
        doc_id="R1",
        source_id="R1",
        split=Split.TRAIN,
        label=Label.FORGED,
        image_path=image,
    )
    reading = SimpleNamespace(
        extraction=Extraction(fields=ReceiptFields(merchant="SHOP"), sources=FieldSources())
    )
    reader = SimpleNamespace(read=lambda _: reading)
    assessed = {"R1": agent_case().assessment}
    ocr = CachedOcr(FakeOcr(), tmp_path / "ocr")
    (found,) = cli_agent.cases([document], assessed, ocr, reader).values()  # type: ignore[arg-type]
    assert found.fields == (ReadField(field="merchant", value="SHOP"),)
    assert found.page.shape == (30, 40)
    assert found.assessment is assessed["R1"]
