"""Field extraction metrics against the gold set (docs/annotation-guidelines.md, section 7).

Every metric is a ratio of per-document sums (for example matched amounts over predicted
amounts), so that confidence intervals can resample whole documents
(:mod:`docfraud.evaluation.bootstrap`). Micro F1 is also such a ratio:
``2 * matched / (predicted + gold)``.
"""

import re
from collections import Counter
from collections.abc import Callable, Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from docfraud.evaluation.bootstrap import Estimate, paired_ratio_difference, ratio
from docfraud.markdown import Cell, Table, render_table
from docfraud.reading.fields import AmountRole, LineItem, ReceiptFields, TaxSummaryRow


@dataclass(frozen=True, slots=True)
class Matches:
    """Matched, predicted and gold counts of one kind of element on one document."""

    matched: int
    predicted: int
    gold: int


@dataclass(frozen=True, slots=True)
class DocumentScore:
    """Everything measured on one document."""

    doc_id: str
    amounts: Matches
    items: Matches
    tax_rows: Matches
    description_similarities: tuple[float, ...]
    merchant: bool
    dates: bool
    item_count: bool
    quantity_total: bool
    gold_amounts: tuple[tuple[AmountRole, Decimal], ...]
    missed_amounts: tuple[tuple[AmountRole, Decimal], ...]
    extra_amounts: tuple[tuple[AmountRole, Decimal], ...]

    @property
    def amounts_exact(self) -> bool:
        """Whether every amount was extracted with its role and nothing else was."""
        return not self.missed_amounts and not self.extra_amounts

    @property
    def rule_inputs_exact(self) -> bool:
        """Whether amounts, items and tax summary rows, which the rules read, are all exact.

        Every gold element is matched and nothing extra is read (ADR-0025, decision 5).
        """
        return (
            self.amounts_exact
            and self.items.matched == self.items.gold == self.items.predicted
            and self.tax_rows.matched == self.tax_rows.gold == self.tax_rows.predicted
        )


def _amount_pairs(fields: ReceiptFields) -> Counter[tuple[AmountRole, Decimal]]:
    return Counter((role, value) for role, values in fields.amounts.items() for value in values)


def _equal_where_both(a: Decimal | None, b: Decimal | None) -> bool:
    return a is None or b is None or a == b


def item_matches(gold: LineItem, predicted: LineItem) -> bool:
    """Equal amount; equal quantity and unit price where both sides have them."""
    return (
        gold.amount is not None
        and gold.amount == predicted.amount
        and _equal_where_both(gold.quantity, predicted.quantity)
        and _equal_where_both(gold.unit_price, predicted.unit_price)
    )


def tax_row_matches(gold: TaxSummaryRow, predicted: TaxSummaryRow) -> bool:
    """Equal rate, taxable amount and tax (the code is not compared)."""
    return (
        gold.rate_percent == predicted.rate_percent
        and gold.taxable_amount == predicted.taxable_amount
        and gold.tax == predicted.tax
    )


def maximum_matching[G, P](
    gold: Sequence[G], predicted: Sequence[P], matches: Callable[[G, P], bool]
) -> list[tuple[int, int]]:
    """A largest one-to-one matching, as ``(gold index, predicted index)`` pairs.

    Augmenting paths (Kuhn's algorithm) make the result independent of the order in which
    elements are printed or predicted; receipts are small enough for the cubic cost.
    """
    edges = [[p for p, q in enumerate(predicted) if matches(g, q)] for g in gold]
    owner: dict[int, int] = {}

    def augment(g: int, seen: set[int]) -> bool:
        for p in edges[g]:
            if p not in seen:
                seen.add(p)
                if p not in owner or augment(owner[p], seen):
                    owner[p] = g
                    return True
        return False

    for g in range(len(gold)):
        augment(g, set())
    return sorted((g, p) for p, g in owner.items())


def normalise_text(text: str | None) -> str:
    """Upper case with runs of whitespace collapsed, for text comparisons."""
    return re.sub(r"\s+", " ", (text or "").upper()).strip()


def similarity(a: str, b: str) -> float:
    """``1 - edit distance / longer length`` of the normalised strings (1.0 when both empty)."""
    a, b = normalise_text(a), normalise_text(b)
    if not a and not b:
        return 1.0
    previous = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        current = [i]
        for j, cb in enumerate(b, 1):
            current.append(min(previous[j] + 1, current[j - 1] + 1, previous[j - 1] + (ca != cb)))
        previous = current
    return 1 - previous[-1] / max(len(a), len(b))


def score_document(doc_id: str, gold: ReceiptFields, predicted: ReceiptFields) -> DocumentScore:
    """Compare one prediction with its gold annotation."""
    gold_amounts, predicted_amounts = _amount_pairs(gold), _amount_pairs(predicted)
    item_pairs = maximum_matching(gold.items, predicted.items, item_matches)
    tax_pairs = maximum_matching(gold.tax_summary, predicted.tax_summary, tax_row_matches)
    return DocumentScore(
        doc_id=doc_id,
        amounts=Matches(
            sum((gold_amounts & predicted_amounts).values()),
            predicted_amounts.total(),
            gold_amounts.total(),
        ),
        items=Matches(len(item_pairs), len(predicted.items), len(gold.items)),
        tax_rows=Matches(len(tax_pairs), len(predicted.tax_summary), len(gold.tax_summary)),
        description_similarities=tuple(
            similarity(gold.items[g].description, predicted.items[p].description)
            for g, p in item_pairs
        ),
        merchant=normalise_text(gold.merchant) == normalise_text(predicted.merchant),
        dates=gold.dates == predicted.dates,
        item_count=gold.item_count == predicted.item_count,
        quantity_total=gold.quantity_total == predicted.quantity_total,
        gold_amounts=tuple(sorted(gold_amounts.elements())),
        missed_amounts=tuple(sorted((gold_amounts - predicted_amounts).elements())),
        extra_amounts=tuple(sorted((predicted_amounts - gold_amounts).elements())),
    )


@dataclass(frozen=True, slots=True)
class Metric:
    """A metric as a ratio of per-document numerators and denominators."""

    name: str
    numerator: Callable[[DocumentScore], float]
    denominator: Callable[[DocumentScore], float]
    paired: bool = False  # same denominator on every text source, so T1 and T2 can be paired


def _f1(m: Callable[[DocumentScore], Matches]) -> tuple[Callable[..., float], Callable[..., float]]:
    return (lambda s: 2 * m(s).matched), (lambda s: m(s).predicted + m(s).gold)


def _prf(name: str, m: Callable[[DocumentScore], Matches]) -> list[Metric]:
    return [
        Metric(f"{name}: precision", lambda s: m(s).matched, lambda s: m(s).predicted),
        Metric(f"{name}: recall", lambda s: m(s).matched, lambda s: m(s).gold, paired=True),
        Metric(f"{name}: F1", *_f1(m)),
    ]


AMOUNTS = _prf("Amounts", lambda s: s.amounts)
ALL_AMOUNTS_EXACT = Metric(
    "Documents with all amounts exact", lambda s: s.amounts_exact, lambda _: 1, paired=True
)
#: The primary metric of the LLM pilot (ADR-0025); kept out of :data:`METRICS` so that the
#: reports of the parser regenerate unchanged.
RULE_INPUTS_EXACT = Metric(
    "Documents with all rule inputs exact", lambda s: s.rule_inputs_exact, lambda _: 1, paired=True
)
METRICS: tuple[Metric, ...] = (
    *AMOUNTS,
    *_prf("Items", lambda s: s.items),
    Metric(
        "Item descriptions: similarity of matched items",
        lambda s: sum(s.description_similarities),
        lambda s: len(s.description_similarities),
    ),
    *_prf("Tax summary rows", lambda s: s.tax_rows),
    Metric("Merchant: accuracy", lambda s: s.merchant, lambda _: 1, paired=True),
    Metric("Dates: accuracy", lambda s: s.dates, lambda _: 1, paired=True),
    Metric("Item count: accuracy", lambda s: s.item_count, lambda _: 1, paired=True),
    Metric("Quantity total: accuracy", lambda s: s.quantity_total, lambda _: 1, paired=True),
    ALL_AMOUNTS_EXACT,
)


def estimate(metric: Metric, scores: Sequence[DocumentScore]) -> Estimate | None:
    """Point estimate and document-level bootstrap interval, or ``None`` if undefined."""
    den = [float(metric.denominator(s)) for s in scores]
    if sum(den) <= 0:
        return None
    return ratio([float(metric.numerator(s)) for s in scores], den)


def _cell(e: Estimate | None) -> str:
    return "\N{EM DASH}" if e is None else f"{e.value:.3f} [{e.low:.3f}, {e.high:.3f}]"


def _pairs(values: Sequence[tuple[AmountRole, Decimal]]) -> str:
    return ", ".join(f"{role.value} {value}" for role, value in values) or "\N{EM DASH}"


def render_extraction_report(
    scores: Mapping[str, Sequence[DocumentScore]], forged: frozenset[str], header: str
) -> str:
    """The generated gold evaluation report for two text sources (first one is the baseline).

    Raises:
        ValueError: If the sources were not scored on the same documents in the same order.
    """
    (base, base_scores), (other, other_scores) = scores.items()
    if [s.doc_id for s in base_scores] != [s.doc_id for s in other_scores]:
        msg = "both text sources must be scored on the same documents"
        raise ValueError(msg)
    tables = [
        Table(
            "Headline metrics",
            ("Metric", base, other),
            [
                (m.name, _cell(estimate(m, base_scores)), _cell(estimate(m, other_scores)))
                for m in METRICS
            ],
            note="Value [95 % interval], bootstrap over documents (10,000 resamples). "
            "Precision, recall and F1 are micro-averaged over all elements.",
        ),
    ]
    paired = []
    for m in METRICS:
        denominator = [float(m.denominator(s)) for s in base_scores]
        if not m.paired or sum(denominator) <= 0:
            continue
        diff = paired_ratio_difference(
            [float(m.numerator(s)) for s in base_scores],
            [float(m.numerator(s)) for s in other_scores],
            denominator,
        )
        paired.append((m.name, _cell(diff), diff.excludes_zero))
    tables.append(
        Table(
            f"Difference {other} - {base}",
            ("Metric", "Difference [95 % interval]", "Interval excludes zero"),
            paired,
            note="Paired bootstrap over the same documents; only metrics whose denominator does "
            "not depend on the prediction can be paired.",
        )
    )
    by_label = []
    for label, keep in (("authentic", False), ("forged", True)):
        rows = []
        for source_scores in (base_scores, other_scores):
            group = [s for s in source_scores if (s.doc_id in forged) is keep]
            rows.append((estimate(AMOUNTS[2], group), estimate(ALL_AMOUNTS_EXACT, group)))
        n = len([s for s in base_scores if (s.doc_id in forged) is keep])
        by_label.append(
            (f"{label} ({n})", *(_cell(e) for pair in zip(*rows, strict=True) for e in pair))
        )
    tables.append(
        Table(
            "By label",
            (
                "Receipts",
                f"Amounts F1: {base}",
                f"Amounts F1: {other}",
                f"All amounts exact: {base}",
                f"All amounts exact: {other}",
            ),
            by_label,
            note="Forged receipts are read as printed, so the same metrics apply.",
        )
    )
    role_rows: list[list[Cell]] = []
    for role in AmountRole:
        gold_n = sum(r == role for s in base_scores for r, _ in s.gold_amounts)
        if not gold_n:
            continue
        cells: list[Cell] = [role.value, gold_n]
        for source_scores in (base_scores, other_scores):
            missed = sum(r == role for s in source_scores for r, _ in s.missed_amounts)
            extra = sum(r == role for s in source_scores for r, _ in s.extra_amounts)
            cells += [(gold_n - missed) / gold_n, extra]
        role_rows.append(cells)
    tables.append(
        Table(
            "Amounts by role",
            (
                "Role",
                "Gold",
                f"Recall: {base}",
                f"Wrong or extra: {base}",
                f"Recall: {other}",
                f"Wrong or extra: {other}",
            ),
            role_rows,
            note="'Wrong or extra' counts predicted (role, value) pairs without a gold "
            "counterpart: a misread value, a wrong role or an amount that is not in the gold.",
        )
    )
    for name, source_scores in scores.items():
        tables.append(
            Table(
                f"Per receipt: {name}",
                ("Receipt", "Forged", "Amounts matched / gold / predicted", "Missed", "Extra"),
                [
                    (
                        s.doc_id,
                        s.doc_id in forged,
                        f"{s.amounts.matched} / {s.amounts.gold} / {s.amounts.predicted}",
                        _pairs(s.missed_amounts),
                        _pairs(s.extra_amounts),
                    )
                    for s in source_scores
                ],
            )
        )
    return header + "\n".join(render_table(t) for t in tables)
