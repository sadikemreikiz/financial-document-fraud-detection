"""Checks on the real Find-it-again data. Run locally with ``pytest -m data``."""

from collections import Counter
from pathlib import Path

import pytest

from docfraud.data.audit import build_audit, inspect_files, render_audit
from docfraud.data.download import FINDIT2
from docfraud.data.findit import load_findit2
from docfraud.data.models import Document, Label, Split
from docfraud.data.normalize import read_transcript
from docfraud.data.splits import check_disjoint

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / "data" / "raw" / "findit2"
REPORT = ROOT / "reports" / "data" / "findit2-audit.md"

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not DATA.is_dir(), reason="run `docfraud data download && docfraud data prepare`"
    ),
]


@pytest.fixture(scope="module")
def documents() -> tuple[Document, ...]:
    return load_findit2(DATA)


def test_split_sizes_match_the_evaluation_protocol(documents: tuple[Document, ...]) -> None:
    sizes = Counter((d.split, d.label) for d in documents)
    assert {
        s: (sizes[s, Label.AUTHENTIC] + sizes[s, Label.FORGED], sizes[s, Label.FORGED])
        for s in Split
    } == {
        Split.TRAIN: (577, 95),
        Split.VAL: (192, 33),
        Split.TEST: (218, 35),
    }


def test_splits_are_disjoint(documents: tuple[Document, ...]) -> None:
    check_disjoint(documents)


def test_documented_corrections_took_effect(documents: tuple[Document, ...]) -> None:
    by_id = {d.doc_id: d for d in documents}
    assert by_id["X51006619709"].split is Split.TRAIN
    assert by_id["X51006619709"].label is Label.FORGED
    unlocalisable = [
        d.doc_id for d in documents if d.label is Label.FORGED and not d.modified_regions
    ]
    assert unlocalisable == ["X51008114260"]


def test_normalised_transcripts_carry_no_format_traces(documents: tuple[Document, ...]) -> None:
    for document in documents:
        assert document.transcript_path is not None
        text = read_transcript(document.transcript_path)
        assert "\r" not in text
        assert all(line == line.rstrip() for line in text.splitlines())


@pytest.mark.slow
def test_committed_audit_report_is_up_to_date(documents: tuple[Document, ...]) -> None:
    tables = build_audit([inspect_files(d) for d in documents])
    expected = render_audit(tables, source=FINDIT2.citation)
    assert REPORT.read_text(encoding="utf-8") == expected, "run `docfraud data audit`"
