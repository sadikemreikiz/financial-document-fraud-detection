import hashlib
import json
import shutil
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import numpy.typing as npt
import pytest
from typer.testing import CliRunner

from docfraud import __version__
from docfraud.cli import app
from docfraud.cli import baseline as cli_baseline
from docfraud.cli import common as cli_common
from docfraud.cli import data as cli_data
from docfraud.cli import reading as cli_reading
from docfraud.cli import rules as cli_rules
from docfraud.data.download import SOURCES, DatasetSource
from docfraud.data.gold import GoldManifest, load_manifest
from docfraud.data.models import Label, Split
from docfraud.data.splits import TestSetLockedError
from docfraud.evaluation.folds import assign_folds, load_folds, save_folds
from docfraud.evaluation.runs import CodeVersion, JsonRunStore
from docfraud.reading.fields import ReceiptFields
from docfraud.reading.ocr import OcrResult
from tests.baselines.test_bow import corpus
from tests.evaluation.test_folds import documents as fold_documents
from tests.fixtures.findit import load_toy

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
    source = DatasetSource(
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

    def fake_download(source: DatasetSource, directory: Path, **_: object) -> Path:
        calls.append((source.url, directory))
        return directory / source.filename

    monkeypatch.setattr(cli_data, "download", fake_download)
    result = runner.invoke(app, ["data", "download", "--data-dir", str(toy_source)])
    assert result.exit_code == 0, result.output
    assert calls == [("https://example.org/findit2.zip", toy_source / "raw")]
    assert "verified" in result.output


def test_data_prepare_extracts_and_validates(toy_source: Path) -> None:
    result = runner.invoke(app, ["data", "prepare", "--data-dir", str(toy_source)])
    assert result.exit_code == 0, result.output
    assert "6 documents (3 forged)" in result.output
    assert (toy_source / "raw" / "findit2" / "train.txt").is_file()


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
