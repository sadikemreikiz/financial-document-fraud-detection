import hashlib
import io
import zipfile
from pathlib import Path
from typing import BinaryIO

import pytest

from docfraud.data.download import (
    ChecksumMismatchError,
    Fetch,
    PinnedArchive,
    UnsafeArchiveError,
    _fetch_https,
    download,
    safe_extract,
    sha256_of,
    verify,
)

PAYLOAD = b"receipt-bytes" * 1000


def source(payload: bytes = PAYLOAD) -> PinnedArchive:
    return PinnedArchive(
        name="toy",
        url="https://example.org/toy.zip",
        filename="toy.zip",
        size_bytes=len(payload),
        sha256=hashlib.sha256(payload).hexdigest(),
        homepage="https://example.org",
        citation="Toy dataset",
    )


def serving(payload: bytes) -> tuple[list[str], Fetch]:
    calls: list[str] = []

    def fetch(url: str) -> BinaryIO:
        calls.append(url)
        return io.BytesIO(payload)

    return calls, fetch


def test_download_verifies_and_reports_progress(tmp_path: Path) -> None:
    calls, fetch = serving(PAYLOAD)
    progress: list[tuple[int, int]] = []
    path = download(
        source(), tmp_path, fetch=fetch, on_progress=lambda d, t: progress.append((d, t))
    )
    assert path.read_bytes() == PAYLOAD
    assert calls == ["https://example.org/toy.zip"]
    assert progress[-1] == (len(PAYLOAD), len(PAYLOAD))
    assert not list(tmp_path.glob("*.part"))


def test_existing_verified_file_is_not_downloaded_again(tmp_path: Path) -> None:
    (tmp_path / "toy.zip").write_bytes(PAYLOAD)
    calls, fetch = serving(PAYLOAD)
    download(source(), tmp_path, fetch=fetch)
    assert calls == []


def test_corrupted_download_leaves_no_file(tmp_path: Path) -> None:
    _, fetch = serving(PAYLOAD[:-1] + b"X")
    with pytest.raises(ChecksumMismatchError, match="expected"):
        download(source(), tmp_path, fetch=fetch)
    assert list(tmp_path.iterdir()) == []


@pytest.mark.parametrize("content", [PAYLOAD[:-1], PAYLOAD[:-1] + b"X"], ids=["size", "digest"])
def test_existing_mismatching_file_is_never_overwritten(tmp_path: Path, content: bytes) -> None:
    path = tmp_path / "toy.zip"
    path.write_bytes(content)
    with pytest.raises(ChecksumMismatchError, match="Delete it"):
        verify(source(), path)
    assert path.read_bytes() == content


def test_only_https_is_allowed() -> None:
    with pytest.raises(ValueError, match="non-HTTPS"):
        _fetch_https("http://example.org/toy.zip")


def test_sha256_of(tmp_path: Path) -> None:
    path = tmp_path / "f"
    path.write_bytes(PAYLOAD)
    assert sha256_of(path) == hashlib.sha256(PAYLOAD).hexdigest()


def make_zip(path: Path, members: dict[str, bytes]) -> Path:
    with zipfile.ZipFile(path, "w") as zf:
        for name, data in members.items():
            zf.writestr(name, data)
    return path


def test_safe_extract_skips_macos_metadata(tmp_path: Path) -> None:
    archive = make_zip(
        tmp_path / "a.zip",
        {
            "ds/train.txt": b"x",
            "ds/train/1.png": b"y",
            "__MACOSX/ds/._1.png": b"z",
            "ds/empty/": b"",
        },
    )
    files = safe_extract(archive, tmp_path / "out")
    assert sorted(p.relative_to(tmp_path / "out").as_posix() for p in files) == [
        "ds/train.txt",
        "ds/train/1.png",
    ]
    assert not (tmp_path / "out" / "__MACOSX").exists()


@pytest.mark.parametrize("name", ["../evil.txt", "ds/../../evil.txt", "/abs/evil.txt"])
def test_safe_extract_blocks_zip_slip(tmp_path: Path, name: str) -> None:
    archive = make_zip(tmp_path / "a.zip", {name: b"x"})
    with pytest.raises(UnsafeArchiveError):
        safe_extract(archive, tmp_path / "out")
    assert not (tmp_path / "evil.txt").exists()
