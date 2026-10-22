from pathlib import Path

import pytest

from docfraud.llm.provider import Request, Response, Settings, Thinking, Usage
from docfraud.llm.store import ResponseStore, StoredResponseConflictError

SETTINGS = Settings(
    model="claude-haiku-5-5", max_tokens=4000, thinking=Thinking.DISABLED, effort="medium"
)


def request(user: str) -> Request:
    return Request(settings=SETTINGS, system="s", user=user, answer_schema={"type": "object"})


def response(key: str, answer: str = "{}") -> Response:
    return Response(
        key=key,
        model="claude-haiku-5-5",
        answer=answer,
        stop_reason="end_turn",
        usage=Usage(input_tokens=10, output_tokens=2),
        batch=True,
    )


def test_a_stored_response_comes_back_unchanged(tmp_path: Path) -> None:
    store = ResponseStore(tmp_path)
    stored = response(request("a").key)
    store.put(stored)
    store.put(stored)  # storing it again changes nothing
    assert store.get(stored.key) == stored
    assert store.path(stored.key).parent.name == stored.key[:2]
    assert store.get(request("b").key) is None


def test_a_different_response_never_replaces_a_stored_one(tmp_path: Path) -> None:
    store = ResponseStore(tmp_path)
    key = request("a").key
    store.put(response(key))
    with pytest.raises(StoredResponseConflictError):
        store.put(response(key, answer='{"x": 1}'))


def test_missing_requests_keep_their_order_without_repeats(tmp_path: Path) -> None:
    store = ResponseStore(tmp_path)
    a, b, c = request("a"), request("b"), request("c")
    store.put(response(b.key))
    assert store.missing([a, b, c, a]) == [a, c]
