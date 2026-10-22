from collections.abc import Iterator
from typing import Any, cast

import anthropic
import pytest
from anthropic.types import Message
from anthropic.types.messages import MessageBatch, MessageBatchIndividualResponse

from docfraud.llm.claude import ClaudeProvider, params
from docfraud.llm.provider import Request, Settings, Thinking

SETTINGS = Settings(
    model="claude-haiku-5-5", max_tokens=4000, thinking=Thinking.DISABLED, effort="medium"
)
SCHEMA = {"type": "object", "properties": {}, "required": [], "additionalProperties": False}


def request(user: str, settings: Settings = SETTINGS) -> Request:
    return Request(settings=settings, system="Read the receipt.", user=user, answer_schema=SCHEMA)


def message(
    text: str = '{"a": 1}', stop_reason: str = "end_turn", model: str = "claude-haiku-5-5"
) -> Message:
    return Message.model_validate(
        {
            "id": "msg_1",
            "type": "message",
            "role": "assistant",
            "model": model,
            "content": [{"type": "text", "text": text}],
            "stop_reason": stop_reason,
            "stop_sequence": None,
            "usage": {
                "input_tokens": 100,
                "output_tokens": 20,
                "cache_creation_input_tokens": 1500,
                "cache_read_input_tokens": 0,
                "output_tokens_details": {"thinking_tokens": 3},
            },
        }
    )


def batch(status: str) -> MessageBatch:
    return MessageBatch.model_validate(
        {
            "id": "batch_1",
            "type": "message_batch",
            "processing_status": status,
            "request_counts": {
                "processing": 0,
                "succeeded": 0,
                "errored": 0,
                "canceled": 0,
                "expired": 0,
            },
            "created_at": "2026-10-22T00:00:00Z",
            "expires_at": "2026-10-23T00:00:00Z",
            "archived_at": None,
            "cancel_initiated_at": None,
            "ended_at": None,
            "results_url": None,
        }
    )


class FakeBatches:
    def __init__(self, results: list[MessageBatchIndividualResponse]) -> None:
        self.results_list = results
        self.created: list[dict[str, Any]] = []
        self.polls = 0

    def create(self, *, requests: list[dict[str, Any]]) -> MessageBatch:
        self.created = requests
        return batch("in_progress")

    def retrieve(self, _batch_id: str) -> MessageBatch:
        self.polls += 1
        return batch("ended" if self.polls >= 2 else "in_progress")

    def results(self, _batch_id: str) -> Iterator[MessageBatchIndividualResponse]:
        return iter(self.results_list)


class FakeMessages:
    def __init__(
        self, replies: list[Message], results: list[MessageBatchIndividualResponse]
    ) -> None:
        self.replies = iter(replies)
        self.calls: list[dict[str, Any]] = []
        self.batches = FakeBatches(results)

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        return next(self.replies)


class FakeClient:
    def __init__(
        self,
        replies: list[Message] | None = None,
        results: list[MessageBatchIndividualResponse] | None = None,
    ) -> None:
        self.messages = FakeMessages(replies or [], results or [])


def provider(client: FakeClient, waits: list[float] | None = None) -> ClaudeProvider:
    sleeps = waits if waits is not None else []
    return ClaudeProvider(cast("anthropic.Anthropic", client), wait=sleeps.append, poll_seconds=7.0)


def test_the_request_caches_the_instructions_and_constrains_the_answer() -> None:
    sent = params(request("r0: CASH 52.00"))
    assert sent["system"] == [
        {"type": "text", "text": "Read the receipt.", "cache_control": {"type": "ephemeral"}}
    ]
    assert sent["messages"] == [{"role": "user", "content": "r0: CASH 52.00"}]
    assert sent["thinking"] == {"type": "disabled"}
    assert sent["output_config"] == {
        "effort": "medium",
        "format": {"type": "json_schema", "schema": SCHEMA},
    }
    assert (sent["model"], sent["max_tokens"]) == ("claude-haiku-5-5", 4000)


@pytest.mark.parametrize(
    ("thinking", "sent"),
    [
        (Thinking.DISABLED, "disabled"),
        (Thinking.BETWEEN_TOOLS, "between_tools"),
        (Thinking.ADAPTIVE, "adaptive"),
    ],
)
def test_thinking_settings(thinking: Thinking, sent: str) -> None:
    settings = SETTINGS.model_copy(update={"thinking": thinking})
    assert params(request("x", settings))["thinking"] == {"type": sent}


def test_one_request_at_a_time() -> None:
    client = FakeClient(replies=[message()])
    (response,) = provider(client).run([request("a")], batch=False)
    assert (response.key, response.answer, response.stop_reason) == (
        request("a").key,
        '{"a": 1}',
        "end_turn",
    )
    assert (
        response.usage.input_tokens,
        response.usage.cache_write_tokens,
        response.usage.thinking_tokens,
    ) == (
        100,
        1500,
        3,
    )
    assert (response.batch, response.request_id, response.model) == (
        False,
        "msg_1",
        "claude-haiku-5-5",
    )
    assert len(client.messages.calls) == 1


def test_a_refusal_is_kept_without_an_answer() -> None:
    (response,) = provider(FakeClient(replies=[message(stop_reason="refusal")])).run(
        [request("a")], batch=False
    )
    assert (response.answer, response.stop_reason) == (None, "refusal")


def result(key: str, outcome: dict[str, Any]) -> MessageBatchIndividualResponse:
    return MessageBatchIndividualResponse.model_validate({"custom_id": key, "result": outcome})


def test_a_batch_is_polled_until_it_ends_and_answers_come_back_in_order() -> None:
    a, b, c = request("a"), request("b"), request("c")
    results = [
        result(c.key, {"type": "expired"}),
        result(
            b.key,
            {
                "type": "errored",
                "error": {
                    "type": "error",
                    "error": {"type": "overloaded_error", "message": "busy"},
                },
            },
        ),
        result(a.key, {"type": "succeeded", "message": message().model_dump(mode="json")}),
    ]
    client, waits = FakeClient(results=results), list[float]()
    responses = provider(client, waits).run([a, b, c], batch=True)
    assert [r.key for r in responses] == [a.key, b.key, c.key]
    assert [(r.answer, r.error) for r in responses] == [
        ('{"a": 1}', None),
        (None, "overloaded_error"),
        (None, "expired"),
    ]
    assert all(r.batch for r in responses)
    assert responses[1].usage.output_tokens == 0
    assert responses[1].model == "claude-haiku-5-5"
    assert [r["custom_id"] for r in client.messages.batches.created] == [a.key, b.key, c.key]
    assert waits == [7.0, 7.0]


def test_a_batch_must_answer_every_request() -> None:
    a, b = request("a"), request("b")
    client = FakeClient(
        results=[result(a.key, {"type": "succeeded", "message": message().model_dump(mode="json")})]
    )
    with pytest.raises(RuntimeError, match="no result for 1 requests"):
        provider(client).run([a, b], batch=True)


def test_a_batch_cannot_repeat_a_request() -> None:
    with pytest.raises(ValueError, match="twice"):
        provider(FakeClient()).run([request("a"), request("a")], batch=True)


def test_nothing_to_answer_needs_no_client() -> None:
    assert ClaudeProvider(None).run([], batch=True) == []


def test_the_client_is_created_on_first_use(monkeypatch: pytest.MonkeyPatch) -> None:
    sentinel = object()
    made: list[object] = []

    def make() -> object:
        made.append(sentinel)
        return sentinel

    monkeypatch.setattr(anthropic, "Anthropic", make)
    lazy = ClaudeProvider()
    assert made == []
    first = lazy.client
    assert lazy.client is first
    assert made == [sentinel]
    assert cast("object", first) is sentinel
