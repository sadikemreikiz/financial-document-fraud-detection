import base64
import io
import json
from decimal import Decimal

import numpy as np
from PIL import Image as PilImage

from docfraud.data.models import BBox
from docfraud.decision import Decision
from docfraud.detectors.finding import Finding
from docfraud.llm.agent_tools import (
    CROP,
    TOOLS,
    Case,
    ReadField,
    ToolOutput,
    call,
    read_fields,
)
from docfraud.llm.detector import LONG_EDGE
from docfraud.reading.fields import (
    AmountRole,
    FieldSources,
    ItemSources,
    LineItem,
    ReceiptFields,
    Source,
    TaxSummaryRow,
)
from docfraud.reading.layout import Row, Segment
from docfraud.reporting.assessment import Assessment, Evidence

BOX = BBox(x=10, y=20, width=30, height=12)
TOTAL_BOX = BBox(x=50, y=20, width=20, height=10)


def assessment(receipt: str = "R1") -> Assessment:
    region = Finding(
        detector="system 2",
        kind="suspicious region",
        score=0.958412,
        boxes=(BOX,),
        reason="system 2 scores the receipt 0.958; its most suspicious region is "
        "(10, 20, 30 x 12 px)",
    )
    return Assessment(
        receipt=receipt,
        track="T2 (OCR)",
        probability=0.912345,
        decision=Decision.REVIEW,
        budget=0.1,
        threshold=0.8416,
        evidence=(Evidence(inputs=("system 2",), contribution=2.129, findings=(region,)),),
    )


def case(receipt: str = "R1", page: np.ndarray | None = None) -> Case:
    rows = (
        Row((Segment("TOTAL", BBox(x=10, y=20, width=30, height=10)), Segment("12.30", TOTAL_BOX))),
        Row((Segment("CASH 20.00"),)),
    )
    fields = read_fields(
        ReceiptFields(merchant="SHOP", amounts={AmountRole.TOTAL_DUE: (Decimal("12.30"),)}),
        FieldSources(
            amounts={
                AmountRole.TOTAL_DUE: (
                    Source(row=0, start=6, end=11, segments=(1,), box=TOTAL_BOX),
                )
            }
        ),
    )
    shown = page if page is not None else np.arange(60 * 80, dtype=np.uint8).reshape(60, 80)
    return Case(receipt, assessment(receipt), fields, rows, shown)


def test_the_agent_has_four_read_only_tools_whose_schemas_the_api_accepts() -> None:
    assert [t.name for t in TOOLS] == ["assessment", "read_fields", "ocr_rows", "crop"]
    for tool in TOOLS:
        assert tool.input_schema["additionalProperties"] is False
        assert "minimum" not in json.dumps(tool.input_schema)  # strict schemas reject it
    assert CROP.input_schema["required"] == ["x", "y", "width", "height"]


def test_every_value_read_is_listed_with_its_box_where_it_has_one() -> None:
    fields = ReceiptFields(
        merchant="SHOP",
        dates=("2018-05-11",),
        items=(LineItem(description="TEA", quantity=Decimal(2), amount=Decimal("5.00")),),
        amounts={AmountRole.TOTAL_DUE: (Decimal("5.00"), Decimal("5.00"))},
        tax_summary=(TaxSummaryRow(code="SR", rate_percent=Decimal(6)),),
        item_count=1,
    )
    located = Source(row=1, start=0, end=4, segments=(0,), box=TOTAL_BOX)
    sources = FieldSources(
        amounts={AmountRole.TOTAL_DUE: (located,)}, items=(ItemSources(amount=located),)
    )
    found = read_fields(fields, sources)
    assert [f.field for f in found] == [
        "merchant",
        "date",
        "total_due",
        "total_due",
        "item 1: description",
        "item 1: quantity",
        "item 1: amount",
        "tax row 1: code",
        "tax row 1: rate percent",
        "item count",
    ]
    assert found[2] == ReadField(field="total_due", value="5.00", box=TOTAL_BOX)
    assert found[3].box is None  # its second printing has no source
    assert found[6].box == TOTAL_BOX
    assert read_fields(ReceiptFields(), FieldSources()) == ()


def test_the_text_tools_give_json_with_rounded_numbers_and_boxes() -> None:
    shown = case()
    given = json.loads(call(shown, "assessment", {}).text)
    assert given["probability of forgery"] == 0.9123
    assert given["decision"] == "review"
    finding = given["evidence"][0]["findings"][0]
    assert finding["score"] == 0.9584
    assert finding["boxes"] == [{"x": 10, "y": 20, "width": 30, "height": 12}]
    assert json.loads(call(shown, "read_fields", {}).text)[1] == {
        "field": "total_due",
        "value": "12.30",
        "box": {"x": 50, "y": 20, "width": 20, "height": 10},
    }
    rows = json.loads(call(shown, "ocr_rows", {}).text)
    assert [r["row"] for r in rows] == [0, 1]
    assert rows[1]["segments"] == [{"text": "CASH 20.00", "box": None}]


def test_a_crop_shows_the_box_at_the_pages_own_resolution() -> None:
    shown = case()
    output = call(shown, "crop", {"x": 10, "y": 20, "width": 30, "height": 12})
    assert not output.error
    assert json.loads(output.text) == {
        "crop": {"x": 10, "y": 20, "width": 30, "height": 12},
        "scale": 1.0,
    }
    assert output.image is not None
    decoded = PilImage.open(io.BytesIO(base64.standard_b64decode(output.image.data)))
    assert np.array_equal(np.asarray(decoded), shown.page[20:32, 10:40])
    block = output.block("toolu_1")
    assert block["tool_use_id"] == "toolu_1"
    content = block["content"]
    assert isinstance(content, list)
    assert [b["type"] for b in content] == ["text", "image"]
    assert "is_error" not in block


def test_a_box_off_the_page_or_an_unknown_tool_is_an_error_the_model_reads() -> None:
    shown = case()
    for given in (
        {"x": 70, "y": 0, "width": 20, "height": 5},
        {"x": 0, "y": 0, "width": 0, "height": 5},
        {"x": 0, "y": 0, "width": "wide", "height": 5},
        {"x": 0, "y": 0},
    ):
        assert call(shown, "crop", given).error
    unknown = call(shown, "detect_again", {})
    assert unknown.error
    assert unknown.block("toolu_2")["is_error"] is True
    assert ToolOutput("{}").block("toolu_3") == {
        "type": "tool_result",
        "tool_use_id": "toolu_3",
        "content": [{"type": "text", "text": "{}"}],
    }


def test_a_crop_too_large_for_the_provider_is_made_smaller_and_says_so() -> None:
    tall = case(page=np.zeros((LONG_EDGE * 2, 40), dtype=np.uint8))
    output = call(tall, "crop", {"x": 0, "y": 0, "width": 40, "height": LONG_EDGE * 2})
    assert output.image is not None
    assert output.image.height <= LONG_EDGE
    assert json.loads(output.text)["scale"] < 1
