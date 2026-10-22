"""Every value the accounting rules use can be named by a label (ADR-0025). Run with ``-m data``."""

import json
from collections.abc import Iterator
from pathlib import Path

import pytest

from docfraud.cli.common import cached_ocr, findit_documents
from docfraud.llm.labels import LabelKind, label_rows
from docfraud.reading.fields import ReceiptFields
from docfraud.reading.pipeline import TextSource, read_document

ROOT = Path(__file__).resolve().parents[2]
GOLD_A = ROOT / "annotations" / "findit2" / "gold-fields"
OCR_CACHE = ROOT / "artifacts" / "ocr"

pytestmark = [
    pytest.mark.data,
    pytest.mark.skipif(
        not (ROOT / "data" / "raw" / "findit2").is_dir() or not OCR_CACHE.is_dir(),
        reason="needs the dataset and the OCR cache",
    ),
]


def rule_inputs(fields: ReceiptFields) -> Iterator[object]:
    yield from (value for values in fields.amounts.values() for value in values)
    for item in fields.items:
        yield from (v for v in (item.amount, item.quantity, item.unit_price) if v is not None)
    for row in fields.tax_summary:
        yield from (v for v in (row.rate_percent, row.taxable_amount, row.tax) if v is not None)


@pytest.mark.parametrize(
    ("source", "at_least"), [(TextSource.TRANSCRIPT, 848), (TextSource.OCR, 824)]
)
def test_rule_inputs_of_gold_set_a_are_labelled(source: TextSource, at_least: int) -> None:
    manifest = json.loads((GOLD_A / "manifest.json").read_text(encoding="utf-8"))
    ids = set(manifest["authentic"]) | set(manifest["forged"])
    documents = [d for d in findit_documents(ROOT / "data") if d.doc_id in ids]
    ocr = cached_ocr(OCR_CACHE)
    labelled = found = 0
    for document in documents:
        gold = json.loads((GOLD_A / f"{document.doc_id}.json").read_text(encoding="utf-8"))
        fields = ReceiptFields.model_validate(gold["fields"])
        labels = label_rows(read_document(document, ocr, source).rows).labels.values()
        numbers = {label.number for label in labels if label.kind is LabelKind.NUMBER}
        needed = list(rule_inputs(fields))
        labelled += sum(value in numbers for value in needed)
        found += len(needed)
    assert len(documents) == 50
    assert found == 848
    assert labelled >= at_least
