import json
import subprocess
from pathlib import Path

import pytest

from docfraud.data.models import Document, Split
from docfraud.data.splits import select
from docfraud.evaluation.runs import JsonRunStore, code_version, config_hash, environment, run


def test_a_run_records_its_inputs_and_results(tmp_path: Path) -> None:
    store = JsonRunStore(tmp_path / "runs")
    with run(
        "smoke",
        store=store,
        config={"tolerance": 0.01},
        seeds={"folds": 0},
        dataset_sha256={"findit2": "abc"},
    ) as current:
        current.log("f1", 0.5)
    record = current.record
    assert record is not None
    assert record.results == {"f1": 0.5}
    assert record.config_hash == config_hash({"tolerance": 0.01})
    assert record.test_unlocks == ()
    assert record.finished >= record.started
    assert store.load(record.run_id) == record


def test_test_unlocks_inside_a_run_are_recorded(tmp_path: Path) -> None:
    doc = Document(dataset="d", doc_id="x", source_id="x", split=Split.TEST, image_path=Path("x"))
    with run("unlock", store=JsonRunStore(tmp_path), config={}, seeds={}, dataset_sha256={}) as r:
        select([doc], Split.TEST, unlock_test=True, reason="baseline reproduction")
    assert r.record is not None
    assert r.record.test_unlocks == ("baseline reproduction",)


def test_a_failed_run_leaves_no_record(tmp_path: Path) -> None:
    store = JsonRunStore(tmp_path)
    with (
        pytest.raises(RuntimeError),
        run("broken", store=store, config={}, seeds={}, dataset_sha256={}),
    ):
        raise RuntimeError
    assert not list(tmp_path.iterdir())


def test_results_cannot_be_overwritten(tmp_path: Path) -> None:
    with run("twice", store=JsonRunStore(tmp_path), config={}, seeds={}, dataset_sha256={}) as r:
        r.log("f1", 0.5)
        with pytest.raises(KeyError, match="already logged"):
            r.log("f1", 0.6)


def test_run_names_are_safe_directory_names(tmp_path: Path) -> None:
    store = JsonRunStore(tmp_path)
    with (
        pytest.raises(ValueError, match="lower-case"),
        run("../escape", store=store, config={}, seeds={}, dataset_sha256={}),
    ):
        pass


def test_a_store_never_overwrites_a_run(tmp_path: Path) -> None:
    store = JsonRunStore(tmp_path)
    with run("once", store=store, config={}, seeds={}, dataset_sha256={}) as r:
        pass
    assert r.record is not None
    with pytest.raises(FileExistsError):
        store.save(r.record)


def test_config_hash_ignores_key_order() -> None:
    assert config_hash({"a": 1, "b": [1, 2]}) == config_hash({"b": [1, 2], "a": 1})
    assert config_hash({"a": 1}) != config_hash({"a": 2})


def git(cwd: Path, *args: str) -> None:
    identity = ["-c", "user.name=t", "-c", "user.email=t@t", "-c", "commit.gpgsign=false"]
    subprocess.run(["git", *identity, *args], cwd=cwd, check=True)  # noqa: S603, S607


def test_code_version_reports_commit_and_dirty_state(tmp_path: Path) -> None:
    assert code_version(tmp_path).commit is None  # not a repository
    git(tmp_path, "init", "-q")
    (tmp_path / "f").write_text("1", encoding="utf-8")
    git(tmp_path, "add", "f")
    git(tmp_path, "commit", "-q", "-m", "init")
    clean = code_version(tmp_path)
    assert clean.commit is not None
    assert len(clean.commit) == 40
    assert clean.dirty is False
    (tmp_path / "f").write_text("2", encoding="utf-8")
    assert code_version(tmp_path).dirty is True


def test_environment_lists_installed_packages() -> None:
    env = environment()
    assert "numpy" in env.packages
    assert env.cpu_count
    json.dumps(env.model_dump())
