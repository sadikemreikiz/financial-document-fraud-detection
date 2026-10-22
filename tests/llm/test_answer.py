import json

import pytest
from pydantic import ValidationError

from docfraud.llm.answer import Answer, answer_schema
from docfraud.reading.fields import AmountRole

UNSUPPORTED = {"minimum", "maximum", "multipleOf", "minLength", "maxLength", "pattern", "minItems"}


def objects(node: object) -> list[dict[str, object]]:
    """Every JSON object schema inside a schema."""
    found: list[dict[str, object]] = []
    if isinstance(node, dict):
        if node.get("type") == "object":
            found.append(node)
        for value in node.values():
            found += objects(value)
    elif isinstance(node, list):
        for value in node:
            found += objects(value)
    return found


def keys(node: object) -> set[str]:
    if isinstance(node, dict):
        return set(node) | {k for value in node.values() for k in keys(value)}
    if isinstance(node, list):
        return {k for value in node for k in keys(value)}
    return set()


def test_the_schema_fits_the_structured_output_rules() -> None:
    schema = answer_schema()
    closed = objects(schema)
    assert len(closed) == 4  # the answer, an amount, an item, a tax row
    for node in closed:
        required, properties = node["required"], node["properties"]
        assert isinstance(required, list)
        assert isinstance(properties, dict)
        assert node["additionalProperties"] is False
        assert set(required) == set(properties)
    assert not keys(schema) & UNSUPPORTED
    json.dumps(schema)


def test_an_answer_names_labels_and_rows_only() -> None:
    answer = Answer.model_validate_json(
        json.dumps(
            {
                "merchant_rows": [0],
                "dates": ["d1"],
                "items": [{"rows": [4], "quantity": "n7", "unit_price": "n8", "amount": "n9"}],
                "amounts": [{"role": "tendered", "label": "n31"}],
                "tax_summary": [{"rate_percent": "n40", "taxable_amount": None, "tax": "n41"}],
                "item_count": None,
                "quantity_total": "n12",
            }
        )
    )
    assert answer.amounts[0].role is AmountRole.TENDERED
    assert answer.items[0].rows == [4]


@pytest.mark.parametrize(
    "change",
    [{"total": "32.70"}, {"amounts": [{"role": "cash", "label": "n1"}]}, {"items": [{"rows": []}]}],
    ids=["extra-field", "unknown-role", "missing-item-parts"],
)
def test_answers_outside_the_form_are_rejected(change: dict[str, object]) -> None:
    base: dict[str, object] = {
        "merchant_rows": [],
        "dates": [],
        "items": [],
        "amounts": [],
        "tax_summary": [],
        "item_count": None,
        "quantity_total": None,
    }
    with pytest.raises(ValidationError):
        Answer.model_validate(base | change)
