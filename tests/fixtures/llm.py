"""A provider that answers without calling any API, for tests of the LLM reader."""

import json
from collections.abc import Sequence
from pathlib import Path

from docfraud.llm.answer import Answer
from docfraud.llm.provider import Request, Response, Usage

EMPTY_ANSWER = Answer(
    merchant_rows=[],
    dates=[],
    items=[],
    amounts=[],
    tax_summary=[],
    item_count=None,
    quantity_total=None,
)
USAGE = Usage(input_tokens=1000, output_tokens=200)
SECONDS = 1.5  # the time to every answer of a request sent on its own


class FakeProvider:
    """Answers every request with an empty form; requests at the positions in ``fail`` fail
    without cost, as an overloaded service would."""

    identifier = "fake"

    def __init__(self, fail: frozenset[int] = frozenset(), *, pending: Path | None = None) -> None:
        self.calls: list[tuple[int, bool]] = []
        self.fail = fail
        self.pending = pending  # where the real provider remembers its batches; unused here

    def run(self, requests: Sequence[Request], *, batch: bool) -> list[Response]:
        self.calls.append((len(requests), batch))
        return [
            Response(
                key=r.key,
                model=r.settings.model,
                answer=None if n in self.fail else EMPTY_ANSWER.model_dump_json(),
                stop_reason=None if n in self.fail else "end_turn",
                error="overloaded_error" if n in self.fail else None,
                usage=Usage() if n in self.fail else USAGE,
                batch=batch,
                seconds=None if batch else SECONDS,
            )
            for n, r in enumerate(requests)
        ]


class FakeDetector(FakeProvider):
    """Answers the detector's requests: a high score for receipts whose text says EDITED."""

    def run(self, requests: Sequence[Request], *, batch: bool) -> list[Response]:
        answered = super().run(requests, batch=batch)
        return [
            response
            if response.answer is None
            else response.model_copy(
                update={
                    "answer": json.dumps(
                        {
                            "suspicion": 90 if "EDITED" in request.user else 10,
                            "reasons": ["the total"],
                        }
                    )
                }
            )
            for request, response in zip(requests, answered, strict=True)
        ]
