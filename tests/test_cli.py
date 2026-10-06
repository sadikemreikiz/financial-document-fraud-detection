import functools
import hashlib
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
from docfraud.data.findit import load_findit2
from docfraud.data.splits import TestSetLockedError
from docfraud.reading.ocr import OcrResult

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
    loader = functools.partial(
        load_findit2, duplicate_resolutions={}, unannotated_forgeries=frozenset()
    )
    monkeypatch.setattr(docfraud.cli, "load_findit2", loader)
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
