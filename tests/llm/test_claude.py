import json
from collections.abc import Iterator
from pathlib import Path
from typing import Any, cast

import anthropic
import httpx2
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
    """Each poll takes the next status, or raises it; results may fail a few times first."""

    def __init__(
        self,
        results: list[MessageBatchIndividualResponse],
        statuses: list[str | Exception] | None = None,
        results_failures: int = 0,
    ) -> None:
        self.results_list = results
        self.statuses = iter(statuses if statuses is not None else ["in_progress", "ended"])
        self.results_failures = results_failures
        self.created: list[dict[str, Any]] = []
        self.creations = 0
        self.polls = 0

    def create(self, *, requests: list[dict[str, Any]]) -> MessageBatch:
        self.created = requests
        self.creations += 1
        return batch("in_progress")

    def retrieve(self, _batch_id: str) -> MessageBatch:
        self.polls += 1
        status = next(self.statuses)
        if isinstance(status, Exception):
            raise status
        return batch(status)

    def results(self, _batch_id: str) -> Iterator[MessageBatchIndividualResponse]:
        if self.results_failures:
            self.results_failures -= 1
            raise anthropic.APIConnectionError(request=SENT)
        return iter(self.results_list)


class FakeMessages:
    def __init__(
        self, replies: list[Message | Exception], results: list[MessageBatchIndividualResponse]
    ) -> None:
        self.replies = iter(replies)
        self.calls: list[dict[str, Any]] = []
        self.batches = FakeBatches(results)

    def create(self, **kwargs: Any) -> Message:
        self.calls.append(kwargs)
        reply = next(self.replies)
        if isinstance(reply, Exception):
            raise reply
        return reply


class FakeClient:
    def __init__(
        self,
        replies: list[Message | Exception] | None = None,
        results: list[MessageBatchIndividualResponse] | None = None,
        statuses: list[str | Exception] | None = None,
        results_failures: int = 0,
    ) -> None:
        self.messages = FakeMessages(replies or [], results or [])
        self.messages.batches = FakeBatches(results or [], statuses, results_failures)


def provider(
    client: FakeClient,
    waits: list[float] | None = None,
    *,
    pending: Path | None = None,
    patience: int = 120,
) -> ClaudeProvider:
    sleeps = waits if waits is not None else []
    ticks = iter(float(n) for n in range(0, 1000, 2))  # every request takes two seconds
    return ClaudeProvider(
        cast("anthropic.Anthropic", client),
        pending=pending,
        wait=sleeps.append,
        poll_seconds=7.0,
        clock=lambda: next(ticks),
        patience=patience,
    )


SENT = httpx2.Request("POST", "https://api.anthropic.com/v1/messages")


def status_error(kind: type[anthropic.APIStatusError], status: int, name: str) -> Exception:
    body = {"type": "error", "error": {"type": name, "message": name}}
    return kind(name, response=httpx2.Response(status, request=SENT), body=body)


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
    assert sent["inference_geo"] == "global"  # the list prices, whatever the workspace default


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
    assert response.seconds == 2.0
    assert len(client.messages.calls) == 1


def test_a_transient_failure_costs_nothing_and_keeps_the_other_answers() -> None:
    replies: list[Message | Exception] = [
        message(),
        status_error(anthropic.OverloadedError, 529, "overloaded_error"),
        anthropic.APIConnectionError(request=SENT),
        message(),
    ]
    answered = provider(FakeClient(replies=replies)).run([request(n) for n in "abcd"], batch=False)
    assert [r.error for r in answered] == [None, "overloaded_error", "APIConnectionError", None]
    assert [r.usage.output_tokens for r in answered] == [20, 0, 0, 20]
    assert [r.seconds for r in answered] == [2.0, 2.0, 2.0, 2.0]


def test_any_other_failure_stops_the_run() -> None:
    invalid = status_error(anthropic.BadRequestError, 400, "invalid_request_error")
    with pytest.raises(anthropic.BadRequestError):
        provider(FakeClient(replies=[invalid])).run([request("a")], batch=False)


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


def answered(*keys: str) -> list[MessageBatchIndividualResponse]:
    return [
        result(key, {"type": "succeeded", "message": message().model_dump(mode="json")})
        for key in keys
    ]


def test_a_batch_waits_out_a_lost_connection() -> None:
    a = request("a")
    lost = anthropic.APIConnectionError(request=SENT)
    client = FakeClient(
        results=answered(a.key), statuses=[lost, "in_progress", lost, "ended"], results_failures=1
    )
    waits: list[float] = []
    (response,) = provider(client, waits).run([a], batch=True)
    assert response.answer == '{"a": 1}'
    assert waits == [7.0] * 5  # two polls, two lost polls and one lost result download


def test_a_stopped_batch_is_collected_again_without_paying_twice(tmp_path: Path) -> None:
    a, b = request("a"), request("b")
    pending = tmp_path / "batches"
    lost = anthropic.APIConnectionError(request=SENT)
    first = FakeClient(results=answered(a.key, b.key), statuses=[lost, lost])
    with pytest.raises(anthropic.APIConnectionError):
        provider(first, pending=pending, patience=2).run([a, b], batch=True)
    (marker,) = pending.iterdir()
    assert json.loads(marker.read_text(encoding="utf-8")) == {"batch": "batch_1", "requests": 2}

    again = FakeClient(results=answered(a.key, b.key), statuses=["ended"])
    responses = provider(again, pending=pending).run([a, b], batch=True)
    assert [r.answer for r in responses] == ['{"a": 1}', '{"a": 1}']
    assert again.messages.batches.creations == 0  # the remembered batch, not a new one
    assert list(pending.iterdir()) == []  # forgotten once its results are in
