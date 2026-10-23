import hashlib
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

import pytest

from docfraud.data.models import Split
from docfraud.llm.cost import Ledger
from docfraud.llm.frozen import (
    FrozenReader,
    SplitReading,
    UnreadReceiptError,
    load_split_reading,
    save_split_reading,
)
from docfraud.llm.record import FrozenVersion
from docfraud.llm.run import COMPARISON_SETTINGS, Runner, Target, target
from docfraud.llm.store import ResponseStore
from docfraud.reading.layout import Row, Segment
from docfraud.reading.pipeline import TextSource
from tests.fixtures.llm import FakeProvider

INSTRUCTION = "Read the receipt."
VERSION = FrozenVersion(
    round=4,
    settings=COMPARISON_SETTINGS["claude-sonnet-5-5"],
    instruction="extraction-v2.md",
    instruction_sha256=hashlib.sha256(INSTRUCTION.encode()).hexdigest(),
    commit="f80bc32",
)


def receipts() -> list[Target]:
    return [
        target(f"X{n}", source, [Row((Segment(f"TOTAL {n}.00"),))])
        for n in range(3)
        for source in TextSource
    ]


def test_the_reader_takes_the_answers_a_run_stored_without_sending_anything(
    tmp_path: Path,
) -> None:
    store = ResponseStore(tmp_path / "store")
    runner = Runner(FakeProvider(), store, Ledger(tmp_path / "ledger.jsonl"))
    paid = runner.run(VERSION.settings, INSTRUCTION, receipts())
    reader = FrozenReader(VERSION, INSTRUCTION, store)
    assert [reader.request(t) for t in receipts()] == runner.requests(
        VERSION.settings, INSTRUCTION, receipts()
    )
    assert [reader.read(t) for t in receipts()] == paid


def test_a_receipt_the_frozen_version_has_not_read_is_an_error(tmp_path: Path) -> None:
    store = ResponseStore(tmp_path / "store")
    Runner(FakeProvider(), store, Ledger(tmp_path / "ledger.jsonl")).run(
        VERSION.settings, INSTRUCTION, receipts()[:1]
    )
    reader = FrozenReader(VERSION, INSTRUCTION, store)
    assert reader.read(receipts()[1]).rejections == ()  # the same text, so the same request
    with pytest.raises(UnreadReceiptError, match="has not read X1 from the transcript"):
        reader.read(receipts()[2])
    changed = target("X0", TextSource.TRANSCRIPT, [Row((Segment("TOTAL 9.00"),))])
    with pytest.raises(UnreadReceiptError, match="has not read X0 from the transcript"):
        reader.read(changed)  # rows read differently are a different request


def test_the_reader_refuses_an_instruction_that_is_not_the_frozen_one(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match=r"extraction-v2\.md changed after it was frozen"):
        FrozenReader(VERSION, INSTRUCTION + " ", ResponseStore(tmp_path))


def test_the_record_of_a_split_reading_is_kept(tmp_path: Path) -> None:
    reading = SplitReading(
        split=Split.VAL,
        version=VERSION,
        commit="abc123",
        read_at=datetime(2026, 10, 23, 15, 0, tzinfo=UTC),
        receipts=190,
        requests=380,
        cost_usd=Decimal("1.3800"),
        without_answer=0,
    )
    path = tmp_path / "reports" / "validation-reading.json"
    save_split_reading(reading, path)
    assert load_split_reading(path) == reading
