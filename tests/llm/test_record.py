import pytest

from docfraud.llm.provider import Settings, Thinking
from docfraud.llm.record import RoundRecord, best_round

SETTINGS = Settings(
    model="claude-haiku-5-5", max_tokens=8000, thinking=Thinking.DISABLED, effort="medium"
)


def record(number: int, t1: int, t2: int) -> RoundRecord:
    return RoundRecord(
        round=number,
        settings=SETTINGS,
        instruction=f"v{number}.md",
        instruction_sha256="0" * 64,
        receipts=50,
        exact={"T1 transcript": t1, "T2 OCR": t2},
    )


def test_the_round_with_most_exact_readings_is_frozen_and_ties_go_to_the_earlier() -> None:
    rounds = [record(1, 30, 20), record(2, 25, 30), record(3, 35, 20)]
    assert best_round(rounds).round == 2
    assert best_round(rounds).exact_total == 55
    assert best_round([record(4, 30, 30), record(5, 31, 29)]).round == 4


def test_nothing_is_frozen_without_a_round() -> None:
    with pytest.raises(ValueError, match="no development round"):
        best_round([])
