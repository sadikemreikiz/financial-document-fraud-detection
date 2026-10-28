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


class FakeAgent:
    """Answers the agent's steps: the assessment and a crop of its first box, then the OCR
    rows, then a case file citing a number and the box it found.

    ``computed`` makes the case file state a number no tool gave; ``calls`` asks for that many
    crops in the first step instead; ``silent`` steps time out without cost.
    """

    identifier = "fake-agent"

    def __init__(self, *, computed: bool = False, calls: int = 0, silent: int = 0) -> None:
        self.computed = computed
        self.calls = calls
        self.silent = silent
        self.sent: list[Request] = []

    def run(self, requests: Sequence[Request], *, batch: bool) -> list[Response]:
        return [self._one(r, batch=batch) for r in requests]

    def _one(self, request: Request, *, batch: bool) -> Response:
        self.sent.append(request)
        base: dict[str, object] = {
            "key": request.key,
            "model": request.settings.model,
            "batch": batch,
            "seconds": SECONDS,
        }
        if self.silent:
            self.silent -= 1
            failed = {"answer": None, "stop_reason": None, "error": "overloaded_error"}
            return Response.model_validate(base | failed | {"usage": Usage()})
        step = len(request.turns) // 2
        uses: list[tuple[str, dict[str, int]]]
        if request.tool_choice == "auto" and step == 0:
            uses = [("assessment", {})] + [
                ("crop", {"x": 0, "y": 0, "width": 4, "height": 4})
            ] * max(self.calls, 1)
        elif request.tool_choice == "auto" and step == 1:
            uses = [("ocr_rows", {})]
        else:
            answer = json.dumps(self._case_file(request))
            return Response.model_validate(
                base | {"answer": answer, "stop_reason": "end_turn", "usage": USAGE}
            )
        content = tuple(
            {"type": "tool_use", "id": f"toolu_{step}_{k}", "name": name, "input": given}
            for k, (name, given) in enumerate(uses)
        )
        return Response.model_validate(
            base | {"answer": None, "stop_reason": "tool_use", "usage": USAGE, "content": content}
        )

    def _case_file(self, request: Request) -> dict[str, object]:
        result = request.turns[1].content[0]["content"]
        assessment = json.loads(result[0]["text"])  # type: ignore[index]
        probability = assessment["probability of forgery"]
        stated = probability + 0.5 if self.computed else probability
        box = next(
            b for item in assessment["evidence"] for f in item["findings"] for b in f["boxes"]
        )
        return {
            "observations": [
                {
                    "statement": f"The system gives the receipt {stated:.3f}.",
                    "citations": [{"tool": "assessment", "detail": "probability of forgery"}],
                    "location": box,
                }
            ],
            "questions": ["Does the customer's copy show the same total?"],
            "checks": ["Look at the total's characters."],
        }
