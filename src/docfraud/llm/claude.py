"""Claude through the Anthropic API (ADR-0024), one request at a time or through the batch service.

The client takes its credentials from the environment (``ANTHROPIC_API_KEY``) or from a profile
stored by ``ant auth login``; this code never handles a key. No server-side fallback to another
model is requested: every measured answer must come from the model the record names, so a
refusal is kept as a response without an answer. The fixed instructions are marked for prompt
caching, and the answer is constrained to the form's JSON schema.
"""

import time
from collections.abc import Callable, Sequence

import anthropic
from anthropic.types import Message, ThinkingConfigParam
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatchIndividualResponse

from docfraud.llm.provider import Request, Response, Thinking, Usage

POLL_SECONDS = 30.0


class ClaudeProvider:
    """Answers requests with Claude models."""

    identifier = "anthropic"

    def __init__(
        self,
        client: anthropic.Anthropic | None = None,
        *,
        wait: Callable[[float], None] = time.sleep,
        poll_seconds: float = POLL_SECONDS,
    ) -> None:
        """Use ``client``, or create one on first use; ``wait`` sleeps between batch polls."""
        self._client = client
        self._wait = wait
        self._poll_seconds = poll_seconds

    @property
    def client(self) -> anthropic.Anthropic:
        """The API client, created when first needed."""
        if self._client is None:
            self._client = anthropic.Anthropic()
        return self._client

    def run(self, requests: Sequence[Request], *, batch: bool) -> list[Response]:
        """Answer every request in order; the batch service costs half and takes longer.

        Raises:
            ValueError: If a batch holds the same request twice.
            RuntimeError: If the batch service returns no result for a request.
        """
        if not requests:
            return []
        if not batch:
            return [
                _from_message(r.key, self.client.messages.create(**params(r))) for r in requests
            ]
        keys = [request.key for request in requests]
        if len(set(keys)) != len(keys):
            msg = "a batch cannot hold the same request twice"
            raise ValueError(msg)
        job = self.client.messages.batches.create(
            requests=[{"custom_id": r.key, "params": params(r)} for r in requests]
        )
        while job.processing_status != "ended":
            self._wait(self._poll_seconds)
            job = self.client.messages.batches.retrieve(job.id)
        models = {request.key: request.settings.model for request in requests}
        found = {
            result.custom_id: _from_result(result, models[result.custom_id])
            for result in self.client.messages.batches.results(job.id)
        }
        missing = [key for key in keys if key not in found]
        if missing:
            msg = f"batch {job.id} returned no result for {len(missing)} requests"
            raise RuntimeError(msg)
        return [found[key] for key in keys]


def params(request: Request) -> MessageCreateParamsNonStreaming:
    """The Messages API parameters of a request."""
    settings = request.settings
    return {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": [
            {"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": request.user}],
        "thinking": _thinking(settings.thinking),
        "output_config": {
            "effort": settings.effort,
            "format": {"type": "json_schema", "schema": dict(request.answer_schema)},
        },
    }


def _thinking(thinking: Thinking) -> ThinkingConfigParam:
    match thinking:
        case Thinking.DISABLED:
            return {"type": "disabled"}
        case Thinking.BETWEEN_TOOLS:
            return {"type": "between_tools"}
        case Thinking.ADAPTIVE:
            return {"type": "adaptive"}


def _from_message(key: str, message: Message, *, batch: bool = False) -> Response:
    text = next((block.text for block in message.content if block.type == "text"), None)
    usage = message.usage
    details = usage.output_tokens_details
    return Response(
        key=key,
        model=message.model,
        answer=text if message.stop_reason == "end_turn" else None,
        stop_reason=message.stop_reason,
        usage=Usage(
            input_tokens=usage.input_tokens,
            output_tokens=usage.output_tokens,
            cache_write_tokens=usage.cache_creation_input_tokens or 0,
            cache_read_tokens=usage.cache_read_input_tokens or 0,
            thinking_tokens=details.thinking_tokens if details is not None else 0,
        ),
        batch=batch,
        request_id=message.id,
    )


def _from_result(result: MessageBatchIndividualResponse, model: str) -> Response:
    outcome = result.result
    if outcome.type == "succeeded":
        return _from_message(result.custom_id, outcome.message, batch=True)
    error = outcome.error.error.type if outcome.type == "errored" else outcome.type
    return Response(
        key=result.custom_id,
        model=model,
        answer=None,
        stop_reason=None,
        error=error,
        usage=Usage(),
        batch=True,
    )
