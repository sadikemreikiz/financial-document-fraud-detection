"""Write gold annotations through the validating model.

Used while annotating the gold set by hand (docs/annotation-guidelines.md): each call validates
a record against GoldAnnotation before writing it, so malformed amounts never reach a file.

Usage: put ``experiments`` on ``sys.path``, import this module and call :func:`write`.
"""

import json
from pathlib import Path
from typing import Any

from docfraud.data.gold import GoldAnnotation, annotation_path, load_manifest

ROOT = Path(__file__).resolve().parents[1]
DIR = ROOT / "annotations/findit2/gold-fields"
MANIFEST = load_manifest(DIR / "manifest.json")


def item(
    description: str,
    quantity: str | None = None,
    unit_price: str | None = None,
    amount: str | None = None,
) -> dict[str, Any]:
    """One line item; decimals as strings."""
    return {
        "description": description,
        "quantity": quantity,
        "unit_price": unit_price,
        "amount": amount,
    }


def tax_row(
    code: str, rate: str, taxable: str | None = None, tax: str | None = None
) -> dict[str, Any]:
    """One tax summary row; decimals as strings."""
    return {"code": code, "rate_percent": rate, "taxable_amount": taxable, "tax": tax}


def write(  # noqa: PLR0913 - mirrors the annotation fields
    doc_id: str,
    *,
    merchant: str | None,
    dates: list[str],
    items: list[dict[str, Any]] | tuple[()] = (),
    amounts: dict[str, list[str]] | None = None,
    tax_summary: list[dict[str, Any]] | tuple[()] = (),
    item_count: int | None = None,
    quantity_total: str | None = None,
    notes: str = "",
) -> None:
    """Validate and write one gold annotation."""
    if doc_id not in MANIFEST.doc_ids:
        msg = f"{doc_id} is not a gold receipt"
        raise ValueError(msg)
    record = {
        "doc_id": doc_id,
        "annotator": "claude",
        "fields": {
            "merchant": merchant,
            "dates": list(dates),
            "items": list(items),
            "amounts": {k: list(v) for k, v in (amounts or {}).items()},
            "tax_summary": list(tax_summary),
            "item_count": item_count,
            "quantity_total": quantity_total,
        },
        "notes": notes,
    }
    annotation = GoldAnnotation.model_validate(record)
    path = annotation_path(DIR, doc_id)
    path.write_text(
        json.dumps(annotation.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8"
    )
    print("written", path.name)  # noqa: T201
