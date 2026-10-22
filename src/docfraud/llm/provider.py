"""The provider interface: what is asked of a model, what comes back, and who answers.

Callers depend on :class:`Provider` only (ADR-0006), so a local model or another vendor can be
measured on the same gold sets without changing the reader. A request is identified by a hash
of everything that can change its answer; the same request is never paid for twice
(:mod:`docfraud.llm.store`).
"""

import hashlib
import json
from collections.abc import Mapping, Sequence
from enum import StrEnum
from typing import Literal, Protocol

from pydantic import BaseModel, ConfigDict, Field


class Thinking(StrEnum):
    """How a model may reason before it answers, as its API names the setting."""

    DISABLED = "disabled"  # Claude Haiku 5.5, at effort high or below
    BETWEEN_TOOLS = "between_tools"  # Claude Sonnet 5.5: no thinking without tools
    ADAPTIVE = "adaptive"  # the model decides; Claude Opus 5.5 cannot switch it off


type Effort = Literal["low", "medium", "high"]


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Settings(_Frozen):
    """How a model is asked; part of a request's identity."""

    model: str
    max_tokens: int = Field(gt=0)
    thinking: Thinking
    effort: Effort


class Request(_Frozen):
    """One question to a model: fixed instructions, one receipt, and the form of the answer."""

    settings: Settings
    system: str
    user: str
    answer_schema: Mapping[str, object]

    @property
    def key(self) -> str:
        """A hash of everything that can change the answer, used to store and match it."""
        canonical = json.dumps(
            {
                "settings": self.settings.model_dump(mode="json"),
                "system": self.system,
                "user": self.user,
                "schema": self.answer_schema,
            },
            sort_keys=True,
            ensure_ascii=False,
        )
        return hashlib.sha256(canonical.encode()).hexdigest()


class Usage(_Frozen):
    """Tokens billed for one answer. Thinking tokens are part of the output tokens."""

    input_tokens: int = 0
    output_tokens: int = 0
    cache_write_tokens: int = 0
    cache_read_tokens: int = 0
    thinking_tokens: int = 0


class Response(_Frozen):
    """What a model returned for one request.

    ``answer`` is the structured answer as JSON text, or ``None`` when the model refused, ran
    out of tokens or the request failed; ``stop_reason`` or ``error`` says which. ``model`` is
    the model ID the provider reports, which can name a snapshot of the requested model.
    """

    key: str
    model: str
    answer: str | None
    stop_reason: str | None
    error: str | None = None
    usage: Usage
    batch: bool
    request_id: str | None = None


class Provider(Protocol):
    """Anything that answers requests."""

    @property
    def identifier(self) -> str:
        """Who answers, e.g. ``anthropic``."""
        ...

    def run(self, requests: Sequence[Request], *, batch: bool) -> list[Response]:
        """Answer every request, in the order given; ``batch`` asks for the batch service."""
        ...


def by_key(responses: Sequence[Response]) -> dict[str, Response]:
    """Responses indexed by their request's key."""
    return {response.key: response for response in responses}
