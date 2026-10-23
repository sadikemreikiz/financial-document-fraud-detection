"""The frozen LLM reader's fields from Phase 5 on, taken from stored answers only.

Under ADR-0027 the LLM reader replaces parser v2 on both text sources, so every receipt's fields
come from the frozen version (``reports/llm/frozen.json``). Its answers are paid for once, by
the commands that say so: the training receipts were read in Phase 6a, the validation receipts
by ``docfraud llm read-validation``, and the test receipts are read in Phase 7.
:class:`FrozenReader` never sends a request: it rebuilds the frozen version's request for a
receipt, takes the stored answer and turns it into fields, and a receipt it has no answer for is
an error. Because the request is rebuilt from the receipt's rows, a change in how rows are read
shows up as an unread receipt, never as an answer to a different text.
"""

import hashlib
from dataclasses import dataclass
from datetime import datetime
from decimal import Decimal
from pathlib import Path

from pydantic import BaseModel, ConfigDict, Field

from docfraud.data.models import Split
from docfraud.llm.answer import answer_schema
from docfraud.llm.provider import Request
from docfraud.llm.reader import LlmReading, read_response
from docfraud.llm.record import FrozenVersion
from docfraud.llm.run import Target
from docfraud.llm.store import ResponseStore


class UnreadReceiptError(LookupError):
    """The frozen version has no stored answer for a receipt."""


@dataclass(frozen=True, slots=True)
class FrozenReader:
    """The frozen version's fields from the answers in ``store``; it never pays for a request."""

    version: FrozenVersion
    instruction: str  # the text of the frozen instruction
    store: ResponseStore

    def __post_init__(self) -> None:
        """Check that ``instruction`` is the text that was frozen.

        Raises:
            ValueError: If its hash differs from the frozen one.
        """
        if hashlib.sha256(self.instruction.encode()).hexdigest() != self.version.instruction_sha256:
            msg = f"{self.version.instruction} changed after it was frozen"
            raise ValueError(msg)

    def request(self, target: Target) -> Request:
        """The frozen version's first reading of one receipt and text source.

        It is the request :meth:`docfraud.llm.run.Runner.requests` builds for the same target.
        """
        return Request(
            settings=self.version.settings,
            system=self.instruction,
            user=target.labelled.text,
            answer_schema=answer_schema(),
        )

    def read(self, target: Target) -> LlmReading:
        """The fields of one receipt from one text source.

        Raises:
            UnreadReceiptError: If no answer to the frozen version's request is stored.
        """
        request = self.request(target)
        response = self.store.get(request.key)
        if response is None:
            msg = (
                f"the frozen version has not read {target.doc_id} from the {target.source.value} "
                f"(request {request.key[:12]})"
            )
            raise UnreadReceiptError(msg)
        return read_response(response, target.labelled, target.rows)


class SplitReading(BaseModel):
    """The one paid reading of a split by the frozen version: the version, the commit, the cost.

    It holds no field, label or score of the receipts: reading a split measures nothing on it.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")

    split: Split
    version: FrozenVersion
    commit: str
    read_at: datetime
    receipts: int = Field(ge=0)
    requests: int = Field(ge=0)  # one per receipt and text source
    cost_usd: Decimal
    without_answer: int = Field(ge=0)  # responses with no valid answer, so with empty fields


def save_split_reading(reading: SplitReading, path: Path) -> None:
    """Keep the record of a split's reading."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(reading.model_dump_json(indent=1) + "\n", encoding="utf-8")


def load_split_reading(path: Path) -> SplitReading:
    """The record kept by :func:`save_split_reading`."""
    return SplitReading.model_validate_json(path.read_text(encoding="utf-8"))
