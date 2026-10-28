"""Running an LLM reader on receipts: a small paid pilot, a projection, then the batch service.

Before anything is sent, requests already answered are taken from the store. Of the rest, a
pilot of a few requests is sent one by one, after checking its worst case against the spending
limit; its measured cost projects the cost of the remaining requests, which are sent only if
the projection, with a margin, stays within the limit (ADR-0024, decision 6). They go to the
batch service at half the price, or one by one when the time to each answer is measured.
Which receipts a run may read is checked before any request is built (ADR-0025, decision 3).
"""

import base64
import json
import math
from collections.abc import Collection, Sequence
from dataclasses import dataclass
from decimal import Decimal

from docfraud.llm.answer import answer_schema
from docfraud.llm.cost import BATCH_FACTOR, Ledger, cost
from docfraud.llm.labels import LabelledRows, label_rows
from docfraud.llm.provider import (
    IMAGE_PIXELS_PER_TOKEN,
    Block,
    Provider,
    Request,
    Response,
    Settings,
    Thinking,
    Usage,
)
from docfraud.llm.reader import EMPTY, LlmReading, Rejection, read_response
from docfraud.llm.store import ResponseStore
from docfraud.reading.layout import Row
from docfraud.reading.pipeline import TextSource

PILOT = 5
MARGIN = Decimal("1.5")
CHARS_PER_TOKEN = 2  # fewer characters per token than any receipt needs, so more tokens

#: The settings of the model comparison: thinking off where the model allows it, Claude Opus
#: 5.5 at its lowest effort, every model otherwise at its default effort (ADR-0025, decision 2).
COMPARISON_SETTINGS: dict[str, Settings] = {
    "claude-haiku-5-5": Settings(
        model="claude-haiku-5-5", max_tokens=8000, thinking=Thinking.DISABLED, effort="medium"
    ),
    "claude-sonnet-5-5": Settings(
        model="claude-sonnet-5-5", max_tokens=8000, thinking=Thinking.BETWEEN_TOOLS, effort="high"
    ),
    "claude-opus-5-5": Settings(
        model="claude-opus-5-5", max_tokens=8000, thinking=Thinking.ADAPTIVE, effort="low"
    ),
}


class ForbiddenDocumentsError(RuntimeError):
    """A run was asked to read receipts its purpose does not allow."""


def check_documents(doc_ids: Collection[str], allowed: Collection[str], purpose: str) -> None:
    """Stop a run that would read receipts outside the set its purpose allows.

    Raises:
        ForbiddenDocumentsError: If any receipt is outside ``allowed``.
    """
    outside = sorted(set(doc_ids) - set(allowed))
    if outside:
        shown = ", ".join(outside[:5]) + (" ..." if len(outside) > 5 else "")  # noqa: PLR2004
        msg = f"{purpose} may not read {len(outside)} of these receipts: {shown}"
        raise ForbiddenDocumentsError(msg)


@dataclass(frozen=True, slots=True)
class Target:
    """One receipt read from one text source, as the reader sees it."""

    doc_id: str
    source: TextSource
    rows: tuple[Row, ...]
    labelled: LabelledRows


def target(doc_id: str, source: TextSource, rows: Sequence[Row]) -> Target:
    """The labelled rows of one receipt and text source."""
    return Target(doc_id, source, tuple(rows), label_rows(rows))


class Runner:
    """Answers targets with one provider, paying each request at most once."""

    def __init__(
        self,
        provider: Provider,
        store: ResponseStore,
        ledger: Ledger,
        *,
        pilot: int = PILOT,
        margin: Decimal = MARGIN,
    ) -> None:
        """Answer with ``provider``, keep answers in ``store`` and spending in ``ledger``."""
        self.provider = provider
        self.store = store
        self.ledger = ledger
        self.pilot = pilot
        self.margin = margin

    def requests(
        self, settings: Settings, instruction: str, targets: Sequence[Target], *, repeat: int = 0
    ) -> list[Request]:
        """The request for each target; ``repeat`` 1 asks for a second, independent reading."""
        schema = answer_schema()
        return [
            Request(
                settings=settings,
                system=instruction,
                user=t.labelled.text,
                answer_schema=schema,
                repeat=repeat,
            )
            for t in targets
        ]

    def run(
        self,
        settings: Settings,
        instruction: str,
        targets: Sequence[Target],
        *,
        repeat: int = 0,
        batch: bool = True,
    ) -> list[LlmReading]:
        """Read every target, sending only requests that have no stored answer.

        With ``batch`` false, the requests after the pilot are sent one by one too, so that
        every answer is timed, at the full price.

        Raises:
            SpendingLimitError: Before the pilot or the batch, if it could pass the limit.
            RuntimeError: If no pilot request was answered, so nothing can be projected.
        """
        requests = self.requests(settings, instruction, targets, repeat=repeat)
        self.answer(requests, batch=batch)
        return [self._read(r, t) for r, t in zip(requests, targets, strict=True)]

    def answer(self, requests: Sequence[Request], *, batch: bool = True) -> list[Response | None]:
        """Answer every request that has no stored answer, and return each stored response.

        A pilot of the first requests is sent one by one, after checking its worst case against
        the limit; its measured cost projects the rest, which go to the batch service (or one
        by one, timed, without ``batch``) only if the projection stays within the limit.

        Raises:
            ValueError: If the requests name more than one model.
            SpendingLimitError: Before the pilot or the batch, if it could pass the limit.
            RuntimeError: If no pilot request was answered, so nothing can be projected.
        """
        models = {request.settings.model for request in requests}
        if len(models) > 1:
            msg = f"one model per run, not {sorted(models)}"
            raise ValueError(msg)
        missing = self.store.missing(requests)
        if missing:
            model = missing[0].settings.model
            pilot, rest = missing[: self.pilot], missing[self.pilot :]
            self.ledger.check(worst_case(pilot))
            paid = self._settle(self.provider.run(pilot, batch=False), model)
            if rest:
                if not paid:
                    msg = "no pilot request was answered; nothing is sent in a batch"
                    raise RuntimeError(msg)
                spent = sum((cost(r.usage, model, batch=False) for r in paid), Decimal(0))
                per_request = spent / len(paid) * (BATCH_FACTOR if batch else 1)
                self.ledger.check(per_request * len(rest) * self.margin)
                self._settle(self.provider.run(rest, batch=batch), model)
        return [self.store.get(request.key) for request in requests]

    def _settle(self, responses: Sequence[Response], model: str) -> list[Response]:
        """Record what was paid and store every response worth keeping.

        A failed request that cost nothing is not stored, so a later run sends it again.
        """
        paid = [response for response in responses if _billed(response.usage)]
        self.ledger.record(paid, model)
        for response in responses:
            if response.answer is not None or _billed(response.usage):
                self.store.put(response)
        return paid

    def _read(self, request: Request, target: Target) -> LlmReading:
        stored = self.store.get(request.key)
        if stored is None:
            return LlmReading(EMPTY, (Rejection("answer", "", "not answered"),))
        return read_response(stored, target.labelled, target.rows)


def _png_tokens(data: str) -> int:
    """The image tokens of a base64 PNG, from the width and height in its header."""
    header = base64.standard_b64decode(data[:32])
    width, height = int.from_bytes(header[16:20]), int.from_bytes(header[20:24])
    return -(-width * height // IMAGE_PIXELS_PER_TOKEN)


def _block_tokens(block: Block) -> int:
    """The most input tokens a content block of an earlier turn can take."""
    source = block.get("source")
    if block.get("type") == "image" and isinstance(source, dict):
        return _png_tokens(str(source["data"]))
    content = block.get("content")
    if isinstance(content, list | tuple):  # a tool result, possibly with images
        return sum(_block_tokens(b) for b in content if isinstance(b, dict))
    return math.ceil(len(json.dumps(block, ensure_ascii=False)) / CHARS_PER_TOKEN)


def _input_tokens(request: Request) -> int:
    """The most input tokens a request can take: its text, images, tools and earlier turns."""
    tools = json.dumps([t.model_dump(mode="json") for t in request.tools], ensure_ascii=False)
    return (
        math.ceil(len(request.system + request.user + tools) / CHARS_PER_TOKEN)
        + sum(image.tokens for image in request.images)
        + sum(_block_tokens(b) for turn in request.turns for b in turn.content)
    )


def worst_case(requests: Sequence[Request]) -> Decimal:
    """The most that requests can cost.

    Every input token, an image's, a tool's and an earlier turn's included, is priced as a
    cache write, the dearest input, and every request is assumed to use all of its output
    tokens.
    """
    return sum(
        (
            cost(
                Usage(cache_write_tokens=_input_tokens(r), output_tokens=r.settings.max_tokens),
                r.settings.model,
                batch=False,
            )
            for r in requests
        ),
        Decimal(0),
    )


def _billed(usage: Usage) -> bool:
    return any(
        (usage.input_tokens, usage.output_tokens, usage.cache_write_tokens, usage.cache_read_tokens)
    )
