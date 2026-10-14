"""Download, verify and unpack archives from their original sources.

Datasets are never committed (ADR-0002), nor is external code whose licence differs from the
project's (ADR-0019). Instead each archive is pinned by URL, size and SHA-256, so that anyone
reproducing the project provably works on identical bytes.
"""

import hashlib
import shutil
import urllib.request
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import BinaryIO

CHUNK_SIZE = 1 << 20

type Fetch = Callable[[str], BinaryIO]
type Progress = Callable[[int, int], None]


@dataclass(frozen=True, slots=True)
class PinnedArchive:
    """An archive, such as a dataset, pinned to exact bytes."""

    name: str
    url: str
    filename: str
    size_bytes: int
    sha256: str
    homepage: str
    citation: str


FINDIT2 = PinnedArchive(
    name="findit2",
    url="https://l3i-share.univ-lr.fr/2023Finditagain/findit2.zip",
    filename="findit2.zip",
    size_bytes=673_974_679,
    sha256="6f8c6f134951956627edcc06c7e8fc298abb9e984073ab193174aee204d46261",
    homepage="https://l3i-share.univ-lr.fr/2023Finditagain/index.html",
    citation=(
        "B. Martínez Tornés et al., Receipt Dataset for Document Forgery Detection, "
        "ICDAR 2023, LNCS 14191, pp. 454-469, doi:10.1007/978-3-031-41682-8_28"
    ),
)

SOURCES: dict[str, PinnedArchive] = {FINDIT2.name: FINDIT2}


class ChecksumMismatchError(RuntimeError):
    """Raised when downloaded or existing bytes differ from the pinned checksum."""


class UnsafeArchiveError(RuntimeError):
    """Raised when an archive member would be written outside the target directory."""


def sha256_of(path: Path) -> str:
    """Return the hex SHA-256 digest of a file, reading it in chunks."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(CHUNK_SIZE):
            digest.update(chunk)
    return digest.hexdigest()


def _fetch_https(url: str) -> BinaryIO:
    if not url.startswith("https://"):
        msg = f"refusing to download over a non-HTTPS URL: {url}"
        raise ValueError(msg)
    response: BinaryIO = urllib.request.urlopen(url, timeout=60)  # noqa: S310 - scheme checked above
    return response


def download(
    source: PinnedArchive,
    directory: Path,
    *,
    fetch: Fetch = _fetch_https,
    on_progress: Progress | None = None,
) -> Path:
    """Download an archive into ``directory`` unless a verified copy already exists.

    The file is streamed to a temporary ``.part`` file, hashed while writing and only moved into
    place after size and checksum match, so an interrupted download never leaves a file that
    looks complete.

    Args:
        source: The pinned archive.
        directory: Where to store the archive.
        fetch: Opens a URL and returns a binary stream; injectable for testing.
        on_progress: Called with ``(bytes_done, bytes_total)`` after each chunk.

    Returns:
        Path to the verified archive.

    Raises:
        ChecksumMismatchError: If an existing file or the downloaded bytes do not match.
    """
    target = directory / source.filename
    if target.exists():
        verify(source, target)
        return target

    directory.mkdir(parents=True, exist_ok=True)
    partial = target.with_name(target.name + ".part")
    digest = hashlib.sha256()
    done = 0
    try:
        with fetch(source.url) as stream, partial.open("wb") as out:
            while chunk := stream.read(CHUNK_SIZE):
                out.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                if on_progress is not None:
                    on_progress(done, source.size_bytes)
        if done != source.size_bytes or digest.hexdigest() != source.sha256:
            msg = (
                f"{source.name}: downloaded {done} bytes with SHA-256 {digest.hexdigest()}, "
                f"expected {source.size_bytes} bytes with SHA-256 {source.sha256}"
            )
            raise ChecksumMismatchError(msg)
        partial.replace(target)
    finally:
        partial.unlink(missing_ok=True)
    return target


def verify(source: PinnedArchive, path: Path) -> None:
    """Check an existing archive against the pinned size and checksum.

    Raises:
        ChecksumMismatchError: If the file differs. It is never overwritten automatically,
            because a mismatch may mean the upstream data changed and deserves a look.
    """
    size = path.stat().st_size
    digest = sha256_of(path) if size == source.size_bytes else "not computed (size differs)"
    if size != source.size_bytes or digest != source.sha256:
        msg = (
            f"{path} does not match {source.name}: {size} bytes, SHA-256 {digest}; expected "
            f"{source.size_bytes} bytes, SHA-256 {source.sha256}. Delete it to download again."
        )
        raise ChecksumMismatchError(msg)


def safe_extract(
    archive: Path,
    destination: Path,
    *,
    exclude_prefixes: tuple[str, ...] = ("__MACOSX/",),
) -> list[Path]:
    """Extract a zip archive, refusing members that would escape ``destination``.

    Args:
        archive: The zip file.
        destination: Directory to extract into; created if needed.
        exclude_prefixes: Member name prefixes to skip, by default macOS resource forks.

    Returns:
        The extracted file paths, in archive order.

    Raises:
        UnsafeArchiveError: If a member path is absolute or traverses outside ``destination``.
    """
    root = destination.resolve()
    extracted: list[Path] = []
    with zipfile.ZipFile(archive) as zf:
        for member in zf.infolist():
            if member.is_dir() or member.filename.startswith(exclude_prefixes):
                continue
            target = (root / member.filename).resolve()
            if not target.is_relative_to(root):
                msg = f"archive member escapes the destination: {member.filename}"
                raise UnsafeArchiveError(msg)
            target.parent.mkdir(parents=True, exist_ok=True)
            with zf.open(member) as src, target.open("wb") as dst:
                shutil.copyfileobj(src, dst, CHUNK_SIZE)
            extracted.append(target)
    return extracted
