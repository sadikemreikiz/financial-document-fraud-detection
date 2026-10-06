import functools
import hashlib
import json
import shutil
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest
from typer.testing import CliRunner

import docfraud.cli
from docfraud import __version__
from docfraud.cli import app
from docfraud.data.download import SOURCES, DatasetSource
from docfraud.data.gold import GoldManifest, draw_gold_set, load_manifest
from docfraud.data.models import Split
from docfraud.data.splits import TestSetLockedError
from docfraud.reading.ocr import OcrResult
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
    monkeypatch.setattr(docfraud.cli, "load_findit2", load_toy)
    monkeypatch.setattr(docfraud.cli, "DUPLICATE_GROUPS", ())
    return tmp_path / "data"


def test_data_download_uses_the_pinned_source(
    toy_source: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    calls: list[tuple[str, Path]] = []

    def fake_download(source: DatasetSource, directory: Path, **_: object) -> Path:
        calls.append((source.url, directory))
        return directory / source.filename

    monkeypatch.setattr(docfraud.cli, "download", fake_download)
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


def test_data_draw_gold_writes_a_manifest(
    toy_source: Path, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    runner.invoke(app, ["data", "prepare", "--data-dir", str(toy_source)])
    monkeypatch.setattr(
        docfraud.cli, "draw_gold_set", functools.partial(draw_gold_set, n_authentic=1, n_forged=1)
    )
    output = tmp_path / "gold.json"
    result = runner.invoke(
        app, ["data", "draw-gold", "--data-dir", str(toy_source), "--output", str(output)]
    )
    assert result.exit_code == 0, result.output
    manifest = load_manifest(output)
    assert (len(manifest.authentic), len(manifest.forged)) == (1, 1)


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
    monkeypatch.setattr(docfraud.cli, "RapidOcrEngine", FakeOcr)
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
    monkeypatch.setattr(docfraud.cli, "draw_review_sample", lambda _: (train[0],))
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
    monkeypatch.setattr(docfraud.cli, "draw_review_sample", lambda _: ("XA", "XB"))
    result = runner.invoke(app, ["reading", "review-gold", "--manifest", str(manifest)])
    assert result.exit_code == 1
    assert "not annotated yet: XB" in result.output


def test_reading_score_review_keeps_the_review_and_writes_a_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    manifest = write_gold(tmp_path, ("XA",))
    monkeypatch.setattr(docfraud.cli, "draw_review_sample", lambda _: ("XA",))
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
    monkeypatch.setattr(docfraud.cli, "draw_review_sample", lambda _: ("XA",))
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
    monkeypatch.setattr(docfraud.cli, "RapidOcrEngine", FakeOcr)
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


def test_reading_evaluate_gold_needs_every_annotation(tmp_path: Path) -> None:
    manifest = write_gold(tmp_path, ("XA",))
    (manifest.parent / "XA.json").unlink()
    result = runner.invoke(app, ["reading", "evaluate-gold", "--manifest", str(manifest)])
    assert result.exit_code == 1
    assert "not annotated yet: XA" in result.output
