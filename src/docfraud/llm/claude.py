"""Claude through the Anthropic API (ADR-0024), one request at a time or through the batch service.

The client takes its credentials from the environment (``ANTHROPIC_API_KEY``) or from a profile
stored by ``ant auth login``; this code never handles a key. No server-side fallback to another
model is requested: every measured answer must come from the model the record names, so a
refusal is kept as a response without an answer. The fixed instructions are marked for prompt
caching, and the answer is constrained to the form's JSON schema. Inference is routed globally,
as the request says rather than a workspace default, so that every answer is billed at the list
prices the ledger uses (US-only inference costs 1.1 times as much).

A batch is remembered on disk from the moment it is submitted until its results are in. Polls
wait out transient failures, such as a lost connection, for up to an hour; if the run stops for
any reason, running it again collects the same batch instead of paying for a second one.
"""

import hashlib
import json
import time
from collections.abc import Callable, Sequence
from pathlib import Path

import anthropic
from anthropic.types import ContentBlockParam, Message, ThinkingConfigParam
from anthropic.types.message_create_params import MessageCreateParamsNonStreaming
from anthropic.types.messages import MessageBatchIndividualResponse

from docfraud.llm.provider import Request, Response, Thinking, Usage

POLL_SECONDS = 30.0
#: Transient failures in a row that a batch run waits out, one poll apart: an hour at 30 s.
PATIENCE = 120

#: Failures that the client still reports after its own retries and that are not billed: the
#: request is answered later by sending it again. Any other error stops the run.
TRANSIENT = (
    anthropic.APIConnectionError,
    anthropic.RateLimitError,
    anthropic.InternalServerError,
    anthropic.OverloadedError,
    anthropic.ServiceUnavailableError,
    anthropic.DeadlineExceededError,
)


class ClaudeProvider:
    """Answers requests with Claude models."""

    identifier = "anthropic"

    def __init__(  # noqa: PLR0913 - the client, where batches are remembered, and timing
        self,
        client: anthropic.Anthropic | None = None,
        *,
        pending: Path | None = None,
        wait: Callable[[float], None] = time.sleep,
        poll_seconds: float = POLL_SECONDS,
        clock: Callable[[], float] = time.perf_counter,
        patience: int = PATIENCE,
    ) -> None:
        """Use ``client``, or create one on first use.

        ``pending`` is the directory where a submitted batch is remembered until its results
        are in; without it, a stopped run cannot collect its batch. ``wait`` sleeps between
        batch polls, ``patience`` is how many transient failures in a row a batch run waits
        out, and ``clock`` times each request sent on its own.
        """
        self._client = client
        self._pending = pending
        self._wait = wait
        self._poll_seconds = poll_seconds
        self._clock = clock
        self._patience = patience

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
            return [self._one(request) for request in requests]
        keys = [request.key for request in requests]
        if len(set(keys)) != len(keys):
            msg = "a batch cannot hold the same request twice"
            raise ValueError(msg)
        batches = self.client.messages.batches
        marker = self._marker(keys)
        batch_id = _remembered(marker)
        if batch_id is None:
            job = batches.create(
                requests=[{"custom_id": r.key, "params": params(r)} for r in requests]
            )
            batch_id = job.id
            _remember(marker, batch_id, len(keys))
        else:
            job = self._patiently(lambda: batches.retrieve(batch_id))
        while job.processing_status != "ended":
            self._wait(self._poll_seconds)
            job = self._patiently(lambda: batches.retrieve(batch_id))
        models = {request.key: request.settings.model for request in requests}
        found = self._patiently(
            lambda: {
                result.custom_id: _from_result(result, models[result.custom_id])
                for result in batches.results(batch_id)
            }
        )
        missing = [key for key in keys if key not in found]
        if missing:
            msg = f"batch {batch_id} returned no result for {len(missing)} requests"
            raise RuntimeError(msg)
        if marker is not None:
            marker.unlink(missing_ok=True)
        return [found[key] for key in keys]

    def _marker(self, keys: Sequence[str]) -> Path | None:
        """Where the batch of these requests is remembered, if anywhere."""
        if self._pending is None:
            return None
        digest = hashlib.sha256("\n".join(keys).encode()).hexdigest()
        return self._pending / f"{digest}.json"

    def _patiently[T](self, call: Callable[[], T]) -> T:
        """Call, waiting out transient failures one poll apart, up to the patience."""
        failures = 0
        while True:
            try:
                return call()
            except TRANSIENT:
                failures += 1
                if failures >= self._patience:
                    raise
                self._wait(self._poll_seconds)

    def _one(self, request: Request) -> Response:
        """Send one request on its own and time it, retries by the client included.

        A transient failure comes back as a response without an answer and without cost, as a
        failed request of the batch service does, so the answers already paid for are kept.
        """
        client = self.client
        sent = self._clock()
        try:
            message = client.messages.create(**params(request))
        except TRANSIENT as error:
            status = error.type if isinstance(error, anthropic.APIStatusError) else None
            return Response(
                key=request.key,
                model=request.settings.model,
                answer=None,
                stop_reason=None,
                error=status or type(error).__name__,
                usage=Usage(),
                batch=False,
                seconds=self._clock() - sent,
            )
        seconds = self._clock() - sent
        return _from_message(request.key, message).model_copy(update={"seconds": seconds})


def _remembered(marker: Path | None) -> str | None:
    """The batch remembered at ``marker``, if one was submitted and not yet collected."""
    if marker is None or not marker.exists():
        return None
    batch_id: str = json.loads(marker.read_text(encoding="utf-8"))["batch"]
    return batch_id


def _remember(marker: Path | None, batch_id: str, requests: int) -> None:
    """Remember a submitted batch until its results are collected."""
    if marker is None:
        return
    marker.parent.mkdir(parents=True, exist_ok=True)
    marker.write_text(json.dumps({"batch": batch_id, "requests": requests}) + "\n", "utf-8")


def params(request: Request) -> MessageCreateParamsNonStreaming:
    """The Messages API parameters of a request."""
    settings = request.settings
    return {
        "model": settings.model,
        "max_tokens": settings.max_tokens,
        "system": [
            {"type": "text", "text": request.system, "cache_control": {"type": "ephemeral"}}
        ],
        "messages": [{"role": "user", "content": _content(request)}],
        "inference_geo": "global",
        "thinking": _thinking(settings.thinking),
        "output_config": {
            "effort": settings.effort,
            "format": {"type": "json_schema", "schema": dict(request.answer_schema)},
        },
    }


def _content(request: Request) -> str | list[ContentBlockParam]:
    """The user turn: the text alone, or the images followed by the text."""
    if not request.images:
        return request.user
    blocks: list[ContentBlockParam] = [
        {
            "type": "image",
            "source": {"type": "base64", "media_type": image.media_type, "data": image.data},
        }
        for image in request.images
    ]
    blocks.append({"type": "text", "text": request.user})
    return blocks


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
