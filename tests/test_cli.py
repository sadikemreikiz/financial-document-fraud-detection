import hashlib
import json
import shutil
import zipfile
from collections.abc import Sequence
from datetime import UTC, datetime, timedelta
from decimal import Decimal
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
from docfraud.cli import app
from docfraud.cli import baseline as cli_baseline
from docfraud.cli import common as cli_common
from docfraud.cli import copy_move as cli_copy_move
from docfraud.cli import data as cli_data
from docfraud.cli import forensics as cli_forensics
from docfraud.cli import llm as cli_llm
from docfraud.cli import marks as cli_marks
from docfraud.cli import observation as cli_observation
from docfraud.cli import reading as cli_reading
from docfraud.cli import rules as cli_rules
from docfraud.cli import synthesis as cli_synthesis
from docfraud.data.benign_marks import ReceiptMarks, load_mark_log, marks_plan, round_path
from docfraud.data.download import SOURCES, PinnedFile
from docfraud.data.gold import GoldManifest, load_manifest
from docfraud.data.models import Entity, Label, Split, Technique
from docfraud.data.observation import (
    ASSISTANT,
    ObservationLog,
    ReceiptObservation,
    Visibility,
    save_log,
    seal,
)
from docfraud.data.splits import TestSetLockedError
from docfraud.evaluation.bootstrap import Estimate
from docfraud.evaluation.design_fold import CLEAN_FOLDS, DESIGN_FOLD
from docfraud.evaluation.folds import assign_folds, load_folds, save_folds
from docfraud.evaluation.runs import CodeVersion, JsonRunStore, run
from docfraud.forensics.learned.pretrained import IMAGENET_WEIGHTS
from docfraud.forensics.trufor import TRUFOR_CODE, TRUFOR_WEIGHTS
from docfraud.llm.adjudication import (
    AdjudicationLog,
    Listing,
    first_answer,
    load_questions,
)
from docfraud.llm.adjudication import save_log as save_adjudication
from docfraud.llm.decision import ModelStep
from docfraud.llm.false_alarms import FIXED_AT
from docfraud.llm.frozen import load_split_reading
from docfraud.llm.gold_b import load_measurement
from docfraud.llm.run import COMPARISON_SETTINGS, ForbiddenDocumentsError
from docfraud.reading.fields import ReceiptFields
from docfraud.reading.ocr import OcrResult
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
from tests.fixtures.llm import FakeProvider
from tests.fusion.test_development import report_record
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
    assert "limit 12 USD" in spending.output


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
    monkeypatch.setattr(cli_llm, "changed_since", lambda *_: [])
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
    monkeypatch.setattr(cli_llm, "changed_since", lambda *_: ["src/docfraud/llm/reader.py"])
    changed = runner.invoke(app, ["llm", "measure-b", *measure])
    assert changed.exit_code == 1
    assert "the reader changed after it was frozen at abc123: src/docfraud/llm/reader.py" in flat(
        changed.output
    )
    monkeypatch.setattr(cli_llm, "changed_since", lambda *_: None)
    unknown = runner.invoke(app, ["llm", "measure-b", *measure])
    assert unknown.exit_code == 1
    assert "cannot compare with commit abc123" in flat(unknown.output)

    monkeypatch.setattr(cli_llm, "changed_since", lambda *_: [])
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

    monkeypatch.setattr(cli_llm, "changed_since", changed)
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
    monkeypatch.setattr(cli_llm, "changed_since", lambda *_: ["src/docfraud/llm/reader.py"])
    result = runner.invoke(app, ["llm", "read-validation", *arguments])
    assert result.exit_code == 1
    assert "the reader changed after it was frozen at abc123" in flat(result.output)
    assert not record.exists()
