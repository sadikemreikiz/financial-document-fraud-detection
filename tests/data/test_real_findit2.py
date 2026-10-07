"""Checks on the real Find-it-again data. Run locally with ``pytest -m data``."""

from collections import Counter
from pathlib import Path

import pytest

from docfraud.data.audit import build_audit, inspect_files, render_audit
from docfraud.data.download import FINDIT2
from docfraud.data.duplicates import find_candidates, resolve_duplicates, undocumented
from docfraud.data.findit import (
    DUPLICATE_GROUPS,
    REVIEWED_DISTINCT,
    SENSITIVITY_EXCLUSIONS,
    load_findit2,
)
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
        Split.TRAIN: (562, 94),
        Split.VAL: (190, 33),
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


@pytest.fixture(scope="module")
def raw_documents() -> tuple[Document, ...]:
    return load_findit2(DATA, duplicate_groups=())


def test_the_test_split_is_kept_whole(raw_documents: tuple[Document, ...]) -> None:
    published = {d.doc_id for d in raw_documents if d.split is Split.TEST}
    kept, _ = resolve_duplicates(raw_documents, DUPLICATE_GROUPS)
    assert {d.doc_id for d in kept if d.split is Split.TEST} == published


def test_sensitivity_exclusions_are_the_tested_copies(raw_documents: tuple[Document, ...]) -> None:
    split = {d.doc_id: d.split for d in raw_documents}
    members = {m for g in DUPLICATE_GROUPS for m in g.members if split[m] is Split.TEST}
    assert members == SENSITIVITY_EXCLUSIONS
    assert len(SENSITIVITY_EXCLUSIONS) == 8


@pytest.mark.slow
def test_every_duplicate_candidate_has_been_reviewed(raw_documents: tuple[Document, ...]) -> None:
    candidates = find_candidates(raw_documents)
    assert undocumented(candidates, DUPLICATE_GROUPS, REVIEWED_DISTINCT) == []


@pytest.mark.slow
def test_committed_audit_report_is_up_to_date(raw_documents: tuple[Document, ...]) -> None:
    documents, dropped = resolve_duplicates(raw_documents, DUPLICATE_GROUPS)
    tables = build_audit([inspect_files(d) for d in documents], DUPLICATE_GROUPS, dropped)
    expected = render_audit(tables, source=FINDIT2.citation)
    assert REPORT.read_text(encoding="utf-8") == expected, "run `docfraud data audit`"
