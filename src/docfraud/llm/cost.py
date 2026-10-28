"""What answers cost, and the spending limit (ADR-0024, decision 6).

Every paid response is recorded in an append-only ledger. Before a call or a batch, the caller
projects its cost; work stops before the total for all LLM use would pass the limit, which
only the owner raises.
"""

import json
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import UTC, datetime
from decimal import Decimal
from pathlib import Path

from docfraud.llm.provider import Response, Usage

MILLION = Decimal(1_000_000)
BATCH_FACTOR = Decimal("0.5")
#: The owner's prepaid credit, less 0.50 USD of room for a difference between this ledger and
#: the provider's billing. The first 10 USD were bought under ADR-0024's limit of 15 EUR; 15 USD
#: more were bought on 2026-10-28 for the rest of Phase 6b and Phase 7, estimated at 10 to 16 USD
#: (ADR-0031, section 7). Only the owner raises it, by buying more credit.
LIMIT_USD = Decimal("24.50")


@dataclass(frozen=True, slots=True)
class Price:
    """US dollars per million tokens."""

    input: Decimal
    output: Decimal
    cache_write: Decimal
    cache_read: Decimal


#: Anthropic list prices for prompts up to 100,000 tokens and five-minute cache writes, checked
#: on 2026-10-22 at https://platform.claude.com/docs/en/about-claude/pricing. The batch service
#: halves every price.
PRICES: dict[str, Price] = {
    "claude-haiku-5-5": Price(Decimal("0.10"), Decimal("0.50"), Decimal("0.125"), Decimal("0.01")),
    "claude-sonnet-5-5": Price(Decimal("2.00"), Decimal("10.00"), Decimal("2.50"), Decimal("0.10")),
    "claude-opus-5-5": Price(Decimal("4.00"), Decimal("20.00"), Decimal("5.00"), Decimal("0.20")),
}


class SpendingLimitError(RuntimeError):
    """Going on would pass the spending limit; the owner decides."""


def cost(usage: Usage, model: str, *, batch: bool) -> Decimal:
    """The cost in US dollars of one response's tokens.

    Raises:
        ValueError: For a model without a known price, which is never called.
    """
    price = PRICES.get(model)
    if price is None:
        msg = f"no price is known for {model}; add it to PRICES before calling it"
        raise ValueError(msg)
    total = (
        usage.input_tokens * price.input
        + usage.output_tokens * price.output
        + usage.cache_write_tokens * price.cache_write
        + usage.cache_read_tokens * price.cache_read
    ) / MILLION
    return total * BATCH_FACTOR if batch else total


class Ledger:
    """An append-only record of spending, one JSON line per paid response."""

    def __init__(self, path: Path, limit: Decimal = LIMIT_USD) -> None:
        """Record spending in ``path`` and stop at ``limit`` US dollars."""
        self.path = path
        self.limit = limit

    def entries(self) -> list[dict[str, object]]:
        """Every recorded response, oldest first."""
        if not self.path.exists():
            return []
        lines = self.path.read_text(encoding="utf-8").splitlines()
        return [json.loads(line) for line in lines if line.strip()]

    def spent(self) -> Decimal:
        """Everything spent so far, in US dollars."""
        return sum((Decimal(str(e["cost_usd"])) for e in self.entries()), Decimal(0))

    def check(self, projected: Decimal) -> None:
        """Stop before work whose projected cost would take the total past the limit.

        Raises:
            SpendingLimitError: If spent plus ``projected`` exceeds the limit.
        """
        spent = self.spent()
        if spent + projected > self.limit:
            msg = (
                f"spent {spent:.4f} USD; the next step is projected at {projected:.4f} USD, "
                f"which would pass the limit of {self.limit} USD"
            )
            raise SpendingLimitError(msg)

    def record(self, responses: Sequence[Response], model: str) -> Decimal:
        """Append the cost of paid responses for ``model``; returns what they cost."""
        now = datetime.now(UTC).isoformat(timespec="seconds")
        total = Decimal(0)
        lines: list[str] = []
        for response in responses:
            amount = cost(response.usage, model, batch=response.batch)
            total += amount
            entry = {
                "time": now,
                "key": response.key,
                "model": response.model,
                "batch": response.batch,
                "usage": response.usage.model_dump(),
                "cost_usd": str(amount),
            }
            lines.append(json.dumps(entry, sort_keys=True))
        if lines:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            with self.path.open("a", encoding="utf-8") as ledger:
                ledger.write("\n".join(lines) + "\n")
        return total
