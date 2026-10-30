"""The demo built end to end from the real dataset: the assets, then the page (local only)."""

import hashlib
import re
from pathlib import Path

import pytest
from typer.testing import CliRunner

from docfraud.cli import app

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "findit2"

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not DATA.is_dir(), reason="run `docfraud data download && docfraud data prepare`"
    ),
]

runner = CliRunner()


def test_the_page_is_built_from_the_exported_receipts_unchanged(tmp_path: Path) -> None:
    assets, site = tmp_path / "assets", tmp_path / "site"
    exported = runner.invoke(app, ["demo", "assets", "--output", str(assets)])
    assert exported.exit_code == 0, exported.output
    built = runner.invoke(app, ["demo", "build", "--assets", str(assets), "--output", str(site)])
    assert built.exit_code == 0, built.output
    page = (site / "index.html").read_text(encoding="utf-8")
    shown = sorted(set(re.findall(r'src="receipts/(X\d+\.png)"', page)))
    assert len(shown) == 10  # ADR-0035 shows ten receipts, each in the overview and its card
    for name in shown:
        original = (DATA / "test" / name).read_bytes()
        assert (site / "receipts" / name).read_bytes() == original
        assert hashlib.sha256(original).hexdigest() in page
