"""The form of an LLM reader's answer: labels and rows, never values.

The model fills this form for one receipt, constrained by its JSON schema (structured output).
Numbers and dates are named by their labels (:mod:`docfraud.llm.labels`); text fields by the
rows they stand on. The form holds no receipt text, so stored answers can be published and
scored again after the model is retired (ADR-0024). Turning an answer into fields is the
reader's job.
"""

from pydantic import BaseModel, ConfigDict, Field

from docfraud.reading.fields import AmountRole


class _Form(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class AmountAnswer(_Form):
    """One printed amount and its role in the receipt's arithmetic."""

    role: AmountRole
    label: str = Field(description="Label of the printed number, e.g. n12")


class ItemAnswer(_Form):
    """One purchased line."""

    rows: list[int] = Field(description="Rows on which the item's description is printed")
    quantity: str | None = Field(description="Label of the printed quantity, or null")
    unit_price: str | None = Field(description="Label of the printed unit price, or null")
    amount: str | None = Field(description="Label of the line amount, or null")


class TaxRowAnswer(_Form):
    """One row of the tax summary table."""

    rate_percent: str | None = Field(description="Label of the printed tax rate, or null")
    taxable_amount: str | None = Field(description="Label of the taxable amount, or null")
    tax: str | None = Field(description="Label of the tax amount, or null")


class Answer(_Form):
    """Everything an LLM reader reports for one receipt."""

    merchant_rows: list[int] = Field(description="Rows of the merchant's name, or empty")
    dates: list[str] = Field(description="Labels of the printed dates, in print order")
    items: list[ItemAnswer]
    amounts: list[AmountAnswer] = Field(
        description="Every printed receipt-level amount with its role, in print order"
    )
    tax_summary: list[TaxRowAnswer]
    item_count: str | None = Field(description="Label of the printed number of items, or null")
    quantity_total: str | None = Field(description="Label of the printed total quantity, or null")


def answer_schema() -> dict[str, object]:
    """The JSON schema of :class:`Answer`, in the form the structured-output API accepts.

    Every object is closed (``additionalProperties: false``) and every property required, with
    ``null`` where a value may be missing.
    """
    return Answer.model_json_schema()
