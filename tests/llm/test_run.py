import json
from decimal import Decimal
from pathlib import Path

import pytest

from docfraud.llm.cost import Ledger, SpendingLimitError
from docfraud.llm.provider import Settings, Thinking
from docfraud.llm.run import (
    COMPARISON_SETTINGS,
    ForbiddenDocumentsError,
    Runner,
    Target,
    check_documents,
    target,
    worst_case,
)
from docfraud.llm.store import ResponseStore
from docfraud.reading.layout import Row, Segment
from docfraud.reading.pipeline import TextSource
from tests.fixtures.llm import FakeProvider

SETTINGS = Settings(
    model="claude-haiku-5-5", max_tokens=4000, thinking=Thinking.DISABLED, effort="medium"
)


def targets(count: int) -> list[Target]:
    return [
        target(f"X{n}", TextSource.TRANSCRIPT, [Row((Segment(f"TOTAL {n}.00"),))])
        for n in range(count)
    ]


def make(tmp_path: Path, provider: FakeProvider, limit: Decimal = Decimal(12)) -> Runner:
    return Runner(
        provider, ResponseStore(tmp_path / "store"), Ledger(tmp_path / "ledger.jsonl", limit)
    )


def test_a_pilot_is_sent_one_by_one_and_the_rest_as_a_batch_once(tmp_path: Path) -> None:
    provider = FakeProvider()
    runner = make(tmp_path, provider)
    readings = runner.run(SETTINGS, "Read.", targets(12))
    assert provider.calls == [(5, False), (7, True)]
    assert all(r.rejections == () for r in readings)
    assert len(runner.ledger.entries()) == 12
    runner.run(SETTINGS, "Read.", targets(12))  # everything is stored now
    assert provider.calls == [(5, False), (7, True)]


def test_without_the_batch_every_request_is_sent_one_by_one_and_timed(tmp_path: Path) -> None:
    provider = FakeProvider()
    runner = make(tmp_path, provider)
    runner.run(SETTINGS, "Read.", targets(12), batch=False)
    assert provider.calls == [(5, False), (7, False)]
    timed = [runner.store.get(r.key) for r in runner.requests(SETTINGS, "Read.", targets(12))]
    assert all(r is not None and r.seconds == 1.5 for r in timed)
    assert runner.ledger.spent() == Decimal("0.0024")  # 12 x 0.0002 USD, no batch discount


def test_the_batch_is_refused_when_its_projection_passes_the_limit(tmp_path: Path) -> None:
    # Haiku: 1000 x 0.10 + 200 x 0.50 per million = 0.0002 USD a request, half in a batch.
    worst = worst_case(make(tmp_path, FakeProvider()).requests(SETTINGS, "Read.", targets(5)))
    runner = make(tmp_path, FakeProvider(), limit=worst + Decimal("0.0001"))
    with pytest.raises(SpendingLimitError):
        runner.run(SETTINGS, "Read.", targets(1000))
    assert runner.ledger.spent() == Decimal("0.0010")  # the pilot was paid, the batch never sent


def test_the_pilot_is_refused_when_its_worst_case_passes_the_limit(tmp_path: Path) -> None:
    provider = FakeProvider()
    with pytest.raises(SpendingLimitError):
        make(tmp_path, provider, limit=Decimal("0.0001")).run(SETTINGS, "Read.", targets(3))
    assert provider.calls == []


def test_failed_requests_cost_nothing_are_not_stored_and_read_as_unanswered(tmp_path: Path) -> None:
    provider = FakeProvider(fail=frozenset({1}))
    runner = make(tmp_path, provider)
    readings = runner.run(SETTINGS, "Read.", targets(3))
    assert [r.rejections[0].reason if r.rejections else None for r in readings] == [
        None,
        "not answered",
        None,
    ]
    runner.run(SETTINGS, "Read.", targets(3))
    assert provider.calls == [(3, False), (1, False)]  # only the failed request is sent again


def test_nothing_is_batched_when_no_pilot_request_was_answered(tmp_path: Path) -> None:
    provider = FakeProvider(fail=frozenset(range(5)))
    with pytest.raises(RuntimeError, match="no pilot request"):
        make(tmp_path, provider).run(SETTINGS, "Read.", targets(8))


def test_a_run_reads_only_the_receipts_its_purpose_allows() -> None:
    check_documents(["A1", "A2"], {"A1", "A2", "A3"}, "development")
    with pytest.raises(
        ForbiddenDocumentsError, match="development may not read 2 of these receipts: B1, B2"
    ):
        check_documents(["A1", "B1", "B2"], {"A1"}, "development")


def test_the_comparison_settings_follow_the_record() -> None:
    assert [(s.thinking, s.effort) for s in COMPARISON_SETTINGS.values()] == [
        (Thinking.DISABLED, "medium"),
        (Thinking.BETWEEN_TOOLS, "high"),
        (Thinking.ADAPTIVE, "low"),
    ]


def test_requests_carry_the_labelled_rows_and_the_answer_schema(tmp_path: Path) -> None:
    (request,) = make(tmp_path, FakeProvider()).requests(SETTINGS, "Read.", targets(1))
    assert request.user.startswith("r0: TOTAL ")
    assert json.dumps(request.answer_schema).count("additionalProperties") == 4
