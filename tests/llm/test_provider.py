import pytest

from docfraud.llm.provider import Request, Response, Settings, Thinking, Usage, by_key

SETTINGS = Settings(
    model="claude-haiku-5-5", max_tokens=4000, thinking=Thinking.DISABLED, effort="medium"
)


def request(**change: object) -> Request:
    fields: dict[str, object] = {
        "settings": SETTINGS,
        "system": "Read the receipt.",
        "user": "r0: CASH 52.00",
        "answer_schema": {"type": "object", "properties": {"a": {"type": "string"}}},
    }
    return Request.model_validate(fields | change)


def test_the_key_is_stable_and_ignores_the_order_of_the_schema() -> None:
    reordered = request(answer_schema={"properties": {"a": {"type": "string"}}, "type": "object"})
    assert request().key == request().key == reordered.key
    assert len(request().key) == 64


@pytest.mark.parametrize(
    "change",
    [
        {"settings": SETTINGS.model_copy(update={"model": "claude-sonnet-5-5"})},
        {"settings": SETTINGS.model_copy(update={"effort": "low"})},
        {"settings": SETTINGS.model_copy(update={"max_tokens": 8000})},
        {"system": "Read the receipt carefully."},
        {"user": "r0: CASH 52.01"},
        {"answer_schema": {"type": "object"}},
    ],
)
def test_everything_that_can_change_the_answer_changes_the_key(change: dict[str, object]) -> None:
    assert request(**change).key != request().key


def test_responses_by_key() -> None:
    response = Response(
        key="k", model="m", answer="{}", stop_reason="end_turn", usage=Usage(), batch=False
    )
    assert by_key([response]) == {"k": response}
