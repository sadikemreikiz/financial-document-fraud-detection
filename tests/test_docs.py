"""Documentation integrity: every relative link in the Markdown files must resolve."""

import re
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
LINK = re.compile(r"\]\(([^)\s]+)\)")
MARKDOWN_FILES = sorted(
    path
    for path in ROOT.rglob("*.md")
    if not {".venv", "data", "artifacts", "runs"} & set(path.relative_to(ROOT).parts)
)


@pytest.mark.parametrize("markdown", MARKDOWN_FILES, ids=lambda p: str(p.relative_to(ROOT)))
def test_relative_links_resolve(markdown: Path) -> None:
    broken = [
        target
        for target in LINK.findall(markdown.read_text(encoding="utf-8"))
        if not target.startswith(("http://", "https://", "mailto:", "#"))
        and not (markdown.parent / target.split("#", 1)[0]).exists()
    ]
    assert not broken, f"broken links in {markdown.name}: {broken}"


def test_documentation_is_found() -> None:
    assert len(MARKDOWN_FILES) >= 10
