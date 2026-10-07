"""Run records: what produced a number, so that it can be traced and reproduced.

Every experiment writes one record (evaluation protocol, section 8): the code version, the
configuration and its hash, the dataset checksum, seeds, package versions, hardware, any unlock
of the test split, and the results. Reports are generated from records, never typed by hand.

Records go through the ``RunStore`` interface. ``JsonRunStore`` keeps each record as a JSON file;
another backend (an experiment tracker, for example) can be added behind the same interface.
"""

import hashlib
import json
import logging
import os
import platform
import re
import subprocess
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from datetime import UTC, datetime
from importlib.metadata import distributions
from pathlib import Path
from typing import Any, Protocol

from pydantic import BaseModel, ConfigDict, Field

from docfraud import __version__

SPLITS_LOGGER = "docfraud.data.splits"  # where select() logs every unlock of the test split
RECORD_NAME = "record.json"
RUN_NAME = re.compile(r"[a-z0-9][a-z0-9-]*")


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class CodeVersion(_Frozen):
    """The commit a run was made from. ``dirty`` means uncommitted changes were present."""

    commit: str | None
    dirty: bool | None


class Environment(_Frozen):
    """Software and hardware a run was made on."""

    python: str
    docfraud: str
    packages: dict[str, str]
    platform: str
    machine: str
    processor: str
    cpu_count: int | None
    memory_bytes: int | None


class RunRecord(_Frozen):
    """Everything needed to trace one result back to its inputs."""

    run_id: str
    name: str
    started: datetime
    finished: datetime
    code: CodeVersion
    environment: Environment
    config: dict[str, Any]
    config_hash: str
    dataset_sha256: dict[str, str] = Field(description="Archive checksum per dataset used")
    seeds: dict[str, int]
    test_unlocks: tuple[str, ...] = Field(description="Reasons given for each test-split unlock")
    results: dict[str, Any]


def config_hash(config: Mapping[str, Any]) -> str:
    """SHA-256 of the configuration in canonical JSON (sorted keys, no whitespace)."""
    canonical = json.dumps(config, sort_keys=True, separators=(",", ":"), default=str)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def code_version(repo: Path | None = None) -> CodeVersion:
    """The current commit and whether the working tree differs from it, if git is available."""
    cwd = repo or Path.cwd()
    try:
        commit = _git(cwd, "rev-parse", "HEAD")
        status = _git(cwd, "status", "--porcelain")
    except OSError, subprocess.CalledProcessError:
        return CodeVersion(commit=None, dirty=None)
    return CodeVersion(commit=commit, dirty=bool(status))


def _git(cwd: Path, *args: str) -> str:
    result = subprocess.run(  # noqa: S603 - fixed git arguments, no shell
        ["git", *args],  # noqa: S607 - git is looked up on PATH like any developer tool
        cwd=cwd,
        capture_output=True,
        text=True,
        check=True,
    )
    return result.stdout.strip()


def environment() -> Environment:
    """Python, package versions and hardware of the current process."""
    packages = {d.metadata["Name"].lower(): d.version for d in distributions()}
    return Environment(
        python=platform.python_version(),
        docfraud=__version__,
        packages=dict(sorted(packages.items())),
        platform=platform.platform(),
        machine=platform.machine(),
        processor=platform.processor(),
        cpu_count=os.cpu_count(),
        memory_bytes=_memory_bytes(),
    )


def _memory_bytes() -> int | None:
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except AttributeError, OSError, ValueError:
        return None


class _UnlockCollector(logging.Handler):
    def __init__(self) -> None:
        super().__init__(level=logging.WARNING)
        self.reasons: list[str] = []

    def emit(self, record: logging.LogRecord) -> None:
        if record.getMessage().startswith("test split unlocked: "):
            self.reasons.append(record.getMessage().removeprefix("test split unlocked: "))


class Run:
    """An experiment in progress. Results are added while it runs; ``record`` is set at the end."""

    def __init__(
        self, name: str, config: Mapping[str, Any], seeds: Mapping[str, int], started: datetime
    ) -> None:
        """Start a run; prefer the ``run`` context manager, which also records test unlocks."""
        self.name = name
        self.config = dict(config)
        self.seeds = dict(seeds)
        self.started = started
        self.results: dict[str, Any] = {}
        self.record: RunRecord | None = None

    def log(self, key: str, value: Any) -> None:
        """Add one result. Values must be JSON-serialisable; keys must be unique within a run.

        Raises:
            KeyError: If the key was already logged.
        """
        if key in self.results:
            msg = f"result {key!r} was already logged in run {self.name!r}"
            raise KeyError(msg)
        self.results[key] = value


class RunStore(Protocol):
    """Where run records are kept."""

    def save(self, record: RunRecord) -> Path:
        """Persist a record and return where it was stored."""
        ...


class JsonRunStore:
    """Keeps each run as ``<root>/<run_id>/record.json``."""

    def __init__(self, root: Path) -> None:
        """Store runs below ``root``, which is created when the first run is saved."""
        self.root = root

    def save(self, record: RunRecord) -> Path:
        """Write the record, refusing to overwrite an existing run.

        Raises:
            FileExistsError: If a run with the same identifier exists.
        """
        directory = self.root / record.run_id
        directory.mkdir(parents=True, exist_ok=False)
        path = directory / RECORD_NAME
        path.write_text(record.model_dump_json(indent=2) + "\n", encoding="utf-8")
        return path

    def load(self, run_id: str) -> RunRecord:
        """Read a stored record."""
        path = self.root / run_id / RECORD_NAME
        return RunRecord.model_validate_json(path.read_text(encoding="utf-8"))

    def records(self, name: str) -> list[RunRecord]:
        """All stored runs with this name, oldest first."""
        if not self.root.is_dir():
            return []
        found = [self.load(p.parent.name) for p in sorted(self.root.glob(f"*/{RECORD_NAME}"))]
        return [r for r in found if r.name == name]


@contextmanager
def run(  # noqa: PLR0913 - every argument is part of the record
    name: str,
    *,
    store: RunStore,
    config: Mapping[str, Any],
    seeds: Mapping[str, int],
    dataset_sha256: Mapping[str, str],
    repo: Path | None = None,
) -> Iterator[Run]:
    """Record one experiment; the record is saved only if the block completes without error.

    Test-split unlocks made inside the block (``docfraud.data.splits.select``) are captured with
    their reasons, so an unlock can never go unrecorded.

    Raises:
        ValueError: If the name is not lower-case letters, digits and hyphens.
    """
    if not RUN_NAME.fullmatch(name):
        msg = f"run name {name!r} must be lower-case letters, digits and hyphens"
        raise ValueError(msg)
    started = datetime.now(UTC)
    current = Run(name, config, seeds, started)
    collector = _UnlockCollector()
    splits_logger = logging.getLogger(SPLITS_LOGGER)
    splits_logger.addHandler(collector)
    try:
        yield current
    finally:
        splits_logger.removeHandler(collector)
    finished = datetime.now(UTC)
    record = RunRecord(
        run_id=f"{started:%Y%m%dT%H%M%S%fZ}-{name}",
        name=name,
        started=started,
        finished=finished,
        code=code_version(repo),
        environment=environment(),
        config=current.config,
        config_hash=config_hash(current.config),
        dataset_sha256=dict(dataset_sha256),
        seeds=current.seeds,
        test_unlocks=tuple(collector.reasons),
        results=current.results,
    )
    store.save(record)
    current.record = record
