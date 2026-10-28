"""The investigation agent's tools: read-only views of one receipt (ADR-0031, section 5).

The agent sees a receipt only through four tools, all over the system's own outputs:

1. ``assessment``: the calibrated probability, the decision of the review policy and the
   evidence with its contributions, findings, reasons and boxes;
2. ``read_fields``: the fields the frozen reader read, each with the box it was read from;
3. ``ocr_rows``: the OCR text rows, each segment with its box;
4. ``crop``: a box of the normalised page at its own resolution, so that it can look at a
   region; a box too large for the provider's limits is made smaller, and the result says so.

None of them runs a detector again or changes a setting. Every output is JSON text (the crop
adds the image), with numbers rounded to four decimals, so that the grounding check can look up
every number and box a case file states in what the agent received.
"""

import json
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal
from typing import Any

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict

from docfraud.data.models import BBox
from docfraud.llm.detector import page_image
from docfraud.llm.provider import Block, Image, Tool
from docfraud.reading.fields import (
    FieldSources,
    ItemSources,
    ReceiptFields,
    Source,
    TaxRowSources,
)
from docfraud.reading.layout import Row
from docfraud.reporting.assessment import Assessment

DECIMALS = 4

_NO_INPUT: dict[str, object] = {"type": "object", "properties": {}, "additionalProperties": False}

ASSESSMENT = Tool(
    name="assessment",
    description="The system's assessment of the receipt: its calibrated probability of forgery, "
    "the review policy's decision and threshold, and the evidence, strongest push towards "
    "forgery first, each with its contribution to the log-odds and its findings: detector, "
    "kind, score, reason and boxes on the page.",
    input_schema=_NO_INPUT,
)
READ_FIELDS = Tool(
    name="read_fields",
    description="The fields the system read from the receipt's OCR text: merchant, dates, "
    "amounts by role, line items and tax rows, each value as printed with the box it was read "
    "from (null where it has none).",
    input_schema=_NO_INPUT,
)
OCR_ROWS = Tool(
    name="ocr_rows",
    description="The receipt's OCR text, row by row from the top, each segment with its box.",
    input_schema=_NO_INPUT,
)
CROP = Tool(
    name="crop",
    description="An image of one box of the receipt's page, in pixels of the page (x and y "
    "from the top left), at the page's own resolution. Use it to look at a region closely.",
    input_schema={
        "type": "object",
        "properties": {name: {"type": "integer"} for name in ("x", "y", "width", "height")},
        "required": ["x", "y", "width", "height"],
        "additionalProperties": False,
    },
)
TOOLS = (ASSESSMENT, READ_FIELDS, OCR_ROWS, CROP)


class _Frozen(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class ReadField(_Frozen):
    """One value the reader read, as printed, and where."""

    field: str
    value: str
    box: BBox | None = None


@dataclass(frozen=True, slots=True)
class Case:
    """Everything the agent's tools show of one receipt: the system's outputs and the page."""

    receipt: str
    assessment: Assessment
    fields: tuple[ReadField, ...]
    rows: tuple[Row, ...]
    page: npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class ToolOutput:
    """What one tool call returned: JSON text, the image of a crop, and whether it failed."""

    text: str
    image: Image | None = None
    error: bool = False

    def block(self, tool_use_id: str) -> Block:
        """The tool result as the next turn sends it."""
        content: list[Block] = [{"type": "text", "text": self.text}]
        if self.image is not None:
            content.append(
                {
                    "type": "image",
                    "source": {
                        "type": "base64",
                        "media_type": self.image.media_type,
                        "data": self.image.data,
                    },
                }
            )
        block: dict[str, object] = {
            "type": "tool_result",
            "tool_use_id": tool_use_id,
            "content": content,
        }
        if self.error:
            block["is_error"] = True
        return block


def read_fields(fields: ReceiptFields, sources: FieldSources) -> tuple[ReadField, ...]:
    """Every value of a reading with its box, in the order of the fields."""
    found: list[ReadField] = []

    def add(name: str, value: object, source: Source | None) -> None:
        if value is not None and value != "":
            box = source.box if source is not None else None
            found.append(ReadField(field=name, value=str(value), box=box))

    add("merchant", fields.merchant, sources.merchant[0] if sources.merchant else None)
    for k, date in enumerate(fields.dates):
        add("date", date, sources.dates[k] if k < len(sources.dates) else None)
    for role, values in fields.amounts.items():
        located = sources.amounts.get(role, ())
        for k, value in enumerate(values):
            add(role.value, value, located[k] if k < len(located) else None)
    for i, item in enumerate(fields.items, start=1):
        parts = sources.items[i - 1] if i <= len(sources.items) else ItemSources()
        add(f"item {i}: description", item.description, _first(parts.description))
        add(f"item {i}: quantity", item.quantity, parts.quantity)
        add(f"item {i}: unit price", item.unit_price, parts.unit_price)
        add(f"item {i}: amount", item.amount, parts.amount)
    for i, row in enumerate(fields.tax_summary, start=1):
        tax = sources.tax_summary[i - 1] if i <= len(sources.tax_summary) else TaxRowSources()
        add(f"tax row {i}: code", row.code, tax.code)
        add(f"tax row {i}: rate percent", row.rate_percent, tax.rate_percent)
        add(f"tax row {i}: taxable amount", row.taxable_amount, tax.taxable_amount)
        add(f"tax row {i}: tax", row.tax, tax.tax)
    add("item count", fields.item_count, sources.item_count)
    add("quantity total", fields.quantity_total, sources.quantity_total)
    return tuple(found)


def _first(sources: Sequence[Source]) -> Source | None:
    return sources[0] if sources else None


def _box(box: BBox | None) -> dict[str, int] | None:
    if box is None:
        return None
    return {"x": box.x, "y": box.y, "width": box.width, "height": box.height}


def _rounded(value: Any) -> Any:
    if isinstance(value, float):
        return round(value, DECIMALS)
    if isinstance(value, Decimal):
        return str(value)
    if isinstance(value, dict):
        return {k: _rounded(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_rounded(v) for v in value]
    return value


def _json(value: object) -> str:
    return json.dumps(_rounded(value), ensure_ascii=False)


def assessment_output(assessment: Assessment) -> str:
    """The ``assessment`` tool's output."""
    return _json(
        {
            "receipt": assessment.receipt,
            "probability of forgery": assessment.probability,
            "decision": assessment.decision.value,
            "review threshold": assessment.threshold,
            "review budget": assessment.budget,
            "evidence": [
                {
                    "inputs": list(item.inputs),
                    "contribution to the log-odds": item.contribution,
                    "findings": [
                        {
                            "detector": f.detector,
                            "kind": f.kind,
                            "score": f.score,
                            "reason": f.reason,
                            "boxes": [_box(b) for b in f.boxes],
                        }
                        for f in item.findings
                    ],
                }
                for item in assessment.evidence
            ],
        }
    )


def fields_output(fields: Sequence[ReadField]) -> str:
    """The ``read_fields`` tool's output."""
    return _json([{"field": f.field, "value": f.value, "box": _box(f.box)} for f in fields])


def rows_output(rows: Sequence[Row]) -> str:
    """The ``ocr_rows`` tool's output; rows are numbered from 0, as the reader numbers them."""
    return _json(
        [
            {
                "row": i,
                "segments": [{"text": s.text, "box": _box(s.box)} for s in row.segments],
            }
            for i, row in enumerate(rows)
        ]
    )


def crop_output(page: npt.NDArray[np.uint8], arguments: Mapping[str, object]) -> ToolOutput:
    """The ``crop`` tool's output: the box's image, or an error if it is not on the page."""
    try:
        x, y, width, height = (int(str(arguments[k])) for k in ("x", "y", "width", "height"))
    except KeyError, ValueError:
        return ToolOutput(
            _json({"error": "a crop needs integers x, y, width and height"}), error=True
        )
    rows, columns = page.shape
    if width < 1 or height < 1 or x < 0 or y < 0 or x + width > columns or y + height > rows:
        stated = {
            "error": "the box is not inside the page",
            "page": {"width": columns, "height": rows},
        }
        return ToolOutput(_json(stated), error=True)
    image = page_image(np.ascontiguousarray(page[y : y + height, x : x + width]))
    box = {"x": x, "y": y, "width": width, "height": height}
    return ToolOutput(_json({"crop": box, "scale": image.width / width}), image=image)


def call(case: Case, name: str, arguments: Mapping[str, object]) -> ToolOutput:
    """Run one tool on a case; an unknown tool is an error the model reads."""
    match name:
        case ASSESSMENT.name:
            return ToolOutput(assessment_output(case.assessment))
        case READ_FIELDS.name:
            return ToolOutput(fields_output(case.fields))
        case OCR_ROWS.name:
            return ToolOutput(rows_output(case.rows))
        case CROP.name:
            return crop_output(case.page, arguments)
        case _:
            return ToolOutput(_json({"error": f"there is no tool {name!r}"}), error=True)
