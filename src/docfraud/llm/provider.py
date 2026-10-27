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


IMAGE_PIXELS_PER_TOKEN = 750
"""About how many pixels one image token covers, as the provider estimates image tokens."""


class Image(_Frozen):
    """An image sent before the text of a request: PNG data in base64 and its size in pixels."""

    media_type: Literal["image/png"] = "image/png"
    data: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)

    @property
    def sha256(self) -> str:
        """A hash of the image data, which stands for the image in a request's key."""
        return hashlib.sha256(self.data.encode()).hexdigest()

    @property
    def tokens(self) -> int:
        """The input tokens the image costs, by the provider's estimate."""
        return -(-self.width * self.height // IMAGE_PIXELS_PER_TOKEN)


class Request(_Frozen):
    """One question to a model: fixed instructions, one receipt, and the form of the answer.

    A request may carry images, which the model sees before the text.
    """

    settings: Settings
    system: str
    user: str
    answer_schema: Mapping[str, object]
    repeat: int = Field(default=0, ge=0, description="0 for a first reading, 1 to read it again")
    images: tuple[Image, ...] = ()

    @property
    def key(self) -> str:
        """A hash of everything that can change the answer, used to store and match it.

        Images enter as their hashes and sizes, and only when there are any, so that a request
        without images keeps the key it had before images could be sent.
        """
        identity: dict[str, object] = {
            "settings": self.settings.model_dump(mode="json"),
            "system": self.system,
            "user": self.user,
            "schema": self.answer_schema,
            "repeat": self.repeat,
        }
        if self.images:
            identity["images"] = [
                {
                    "media_type": i.media_type,
                    "sha256": i.sha256,
                    "width": i.width,
                    "height": i.height,
                }
                for i in self.images
            ]
        canonical = json.dumps(identity, sort_keys=True, ensure_ascii=False)
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
    ``seconds`` is the time from sending the request to its answer when it was sent on its own;
    the batch service answers requests together and times none of them.
    """

    key: str
    model: str
    answer: str | None
    stop_reason: str | None
    error: str | None = None
    usage: Usage
    batch: bool
    request_id: str | None = None
    seconds: float | None = None


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
