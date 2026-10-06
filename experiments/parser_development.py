"""Indirect measurement of the receipt parser on development receipts.

Writes reports/reading/parser-development.md. Development receipts are the training split minus
the gold set (docs/annotation-guidelines.md); gold receipts are never read here.

On authentic receipts the printed arithmetic holds, so a relation that fails on an authentic
receipt usually means the parser misread or mislabelled an amount. Pass rates on authentic
receipts are therefore a lower bound on parser quality, without any manual annotation. Forged
receipts are reported alongside, but they are not used to tune the parser.

Usage: ``uv run python experiments/parser_development.py [--show-failures N]``
"""

import argparse
from collections.abc import Callable
from dataclasses import dataclass
from decimal import Decimal
from pathlib import Path

from docfraud.data.findit import load_findit2
from docfraud.data.gold import development_documents, load_manifest
from docfraud.data.models import Document, Label
from docfraud.markdown import Cell, Table, render_table
from docfraud.reading.fields import AmountRole, ReceiptFields
from docfraud.reading.ocr import CachedOcr, RapidOcrEngine
from docfraud.reading.pipeline import Reading, TextSource, read_document

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "reading" / "parser-development.md"
CENT = Decimal("0.01")
R = AmountRole

type Check = Callable[[ReceiptFields], bool | None]  # None: not applicable


def close(a: Decimal, b: Decimal) -> bool:
    """Equal to the cent."""
    return abs(a - b) <= CENT


def items_sum(f: ReceiptFields) -> bool | None:
    """The item amounts add up to one of the printed totals."""
    amounts = [i.amount for i in f.items if i.amount is not None]
    roles = (R.SUBTOTAL, R.TOTAL_EXCLUDING_TAX, R.TOTAL_INCLUDING_TAX, R.TOTAL_DUE)
    totals = [f.first(r) for r in roles]
    present = [t for t in totals if t is not None]
    if not amounts or not present:
        return None
    return any(close(sum(amounts, Decimal(0)), t) for t in present)


def line_math(f: ReceiptFields) -> bool | None:
    """Quantity times unit price equals the line amount, for every complete line."""
    lines = [
        (i.quantity, i.unit_price, i.amount)
        for i in f.items
        if i.quantity is not None and i.unit_price is not None and i.amount is not None
    ]
    if not lines:
        return None
    return all(close(quantity * price, amount) for quantity, price, amount in lines)


def tax_math(f: ReceiptFields) -> bool | None:
    """Total excluding tax plus tax equals total including tax."""
    excl, tax, incl = f.first(R.TOTAL_EXCLUDING_TAX), f.first(R.TAX), f.first(R.TOTAL_INCLUDING_TAX)
    if excl is None or tax is None or incl is None:
        return None
    return close(excl + tax, incl)


def rounding_math(f: ReceiptFields) -> bool | None:
    """Total including tax plus rounding equals the amount due."""
    incl, rounding, due = f.first(R.TOTAL_INCLUDING_TAX), f.first(R.ROUNDING), f.first(R.TOTAL_DUE)
    if incl is None or rounding is None or due is None:
        return None
    return close(incl + rounding, due)


def change_math(f: ReceiptFields) -> bool | None:
    """Tendered minus the amount due equals the change."""
    due = f.first(R.TOTAL_DUE) or f.first(R.TOTAL_INCLUDING_TAX)
    tendered, change = f.first(R.TENDERED), f.first(R.CHANGE)
    if due is None or tendered is None or change is None:
        return None
    return close(tendered - due, change)


def repeats_agree(f: ReceiptFields) -> bool | None:
    """Every amount printed more than once has the same value each time."""
    repeated = [values for values in f.amounts.values() if len(values) > 1]
    if not repeated:
        return None
    return all(all(close(v, values[0]) for v in values) for values in repeated)


CHECKS: dict[str, Check] = {
    "Items add up to a printed total": items_sum,
    "Quantity x unit price = line amount": line_math,
    "Excl. tax + tax = incl. tax": tax_math,
    "Incl. tax + rounding = amount due": rounding_math,
    "Tendered - due = change": change_math,
    "Repeated amounts agree": repeats_agree,
}


COVERAGE: dict[str, Callable[[ReceiptFields], bool]] = {
    "amount due found": lambda f: f.first(R.TOTAL_DUE) is not None,
    "any item found": lambda f: bool(f.items),
    "a date found": lambda f: bool(f.dates),
}


@dataclass(frozen=True)
class Outcome:
    """Pass and applicability counts of one check on one group of receipts."""

    passed: int
    applicable: int

    def cell(self) -> str:
        """``rate (passed/applicable)``."""
        if not self.applicable:
            return "\N{EM DASH}"
        return f"{100 * self.passed / self.applicable:.1f} % ({self.passed}/{self.applicable})"


def outcome(readings: list[Reading], check: Check) -> Outcome:
    """Apply a check to every reading of a group."""
    results = [r for reading in readings if (r := check(reading.fields)) is not None]
    return Outcome(sum(results), len(results))


def coverage(readings: list[Reading], has: Callable[[ReceiptFields], bool]) -> str:
    """Share of receipts where the parser found something."""
    n = sum(has(r.fields) for r in readings)
    return f"{100 * n / len(readings):.1f} %"


def main() -> None:
    """Read every development receipt with both text sources and write the report."""
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--show-failures", type=int, default=0, help="print failing authentic receipts"
    )
    args = parser.parse_args()

    documents = load_findit2(ROOT / "data" / "raw" / "findit2")
    manifest = load_manifest(ROOT / "annotations" / "findit2" / "gold-fields" / "manifest.json")
    development = development_documents(documents, manifest)
    ocr = CachedOcr(RapidOcrEngine(), ROOT / "artifacts" / "ocr")

    tables = []
    for source in TextSource:
        readings: dict[Label, list[Reading]] = {Label.AUTHENTIC: [], Label.FORGED: []}
        failures: list[tuple[Document, Reading, str]] = []
        for document in development:
            if document.label is None:
                continue
            reading = read_document(document, ocr, source)
            readings[document.label].append(reading)
            if document.label is Label.AUTHENTIC:
                failures += [
                    (document, reading, name)
                    for name, check in CHECKS.items()
                    if check(reading.fields) is False
                ]
        authentic, forged = readings[Label.AUTHENTIC], readings[Label.FORGED]
        rows: list[list[Cell]] = [
            [name, outcome(authentic, check).cell(), outcome(forged, check).cell()]
            for name, check in CHECKS.items()
        ]
        rows += [
            [f"Coverage: {name}", coverage(authentic, has), coverage(forged, has)]
            for name, has in COVERAGE.items()
        ]
        tables.append(
            Table(
                f"Text source: {source.value} ({len(authentic)} authentic, {len(forged)} forged)",
                ["Relation", "Authentic: holds", "Forged: holds"],
                rows,
            )
        )
        for document, reading, name in failures[: args.show_failures]:
            print(f"\n--- {source.value} {document.doc_id}: {name}")  # noqa: T201
            print("\n".join(r.text for r in reading.rows))  # noqa: T201
            print(reading.fields.model_dump(exclude_defaults=True))  # noqa: T201

    header = (
        "# Parser development measurement\n\n"
        "> Generated by `experiments/parser_development.py`. Do not edit by hand.\n\n"
        f"Development receipts: the training split minus the gold set ({len(development)} "
        "receipts). A relation counts only where all its amounts were found. On authentic "
        "receipts the printed arithmetic holds, so failures there point to parsing errors.\n\n"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(header + "\n".join(render_table(t) for t in tables), encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
