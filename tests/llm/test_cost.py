from decimal import Decimal
from pathlib import Path

import pytest

from docfraud.llm.cost import LIMIT_USD, Ledger, SpendingLimitError, cost
from docfraud.llm.provider import Response, Usage

D = Decimal
USAGE = Usage(input_tokens=2000, output_tokens=400, cache_write_tokens=1000, cache_read_tokens=3000)


def test_every_kind_of_token_is_priced() -> None:
    # Sonnet 5.5: 2000 x 2.00 + 400 x 10.00 + 1000 x 2.50 + 3000 x 0.10 per million tokens.
    assert cost(USAGE, "claude-sonnet-5-5", batch=False) == D("0.0108")
    assert cost(USAGE, "claude-sonnet-5-5", batch=True) == D("0.0054")
    assert cost(USAGE, "claude-haiku-5-5", batch=False) == D("0.000555")


def test_a_model_without_a_price_is_never_called() -> None:
    with pytest.raises(ValueError, match="no price"):
        cost(USAGE, "claude-unknown", batch=False)


def response(key: str, *, batch: bool = False) -> Response:
    return Response(
        key=key,
        model="claude-sonnet-5-5",
        answer="{}",
        stop_reason="end_turn",
        usage=USAGE,
        batch=batch,
    )


def test_the_ledger_adds_up_what_was_spent(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl")
    assert ledger.spent() == 0
    assert ledger.record([], "claude-sonnet-5-5") == 0
    assert not ledger.path.exists()
    assert ledger.record([response("a"), response("b", batch=True)], "claude-sonnet-5-5") == D(
        "0.0162"
    )
    assert ledger.spent() == D("0.0162")
    assert [e["key"] for e in ledger.entries()] == ["a", "b"]


def test_work_stops_before_the_limit_is_passed(tmp_path: Path) -> None:
    ledger = Ledger(tmp_path / "ledger.jsonl", limit=D("0.02"))
    ledger.record([response("a")], "claude-sonnet-5-5")  # 0.0108 spent
    ledger.check(D("0.0092"))  # exactly reaches the limit
    with pytest.raises(SpendingLimitError, match="would pass the limit"):
        ledger.check(D("0.0093"))


def test_the_default_limit_is_the_owners_credit_less_room() -> None:
    assert D("24.50") == LIMIT_USD
