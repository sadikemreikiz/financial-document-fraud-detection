from typer.testing import CliRunner

from docfraud import __version__
from docfraud.cli import app

runner = CliRunner()


def test_version_flag_prints_installed_version() -> None:
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert result.stdout.strip() == f"docfraud {__version__}"


def test_no_arguments_shows_help() -> None:
    result = runner.invoke(app, [])
    assert "Usage" in result.output
