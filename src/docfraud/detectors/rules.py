"""Accounting rules: checks that a receipt's printed arithmetic adds up (ADR-0012).

A forger who edits one amount and forgets the others breaks the arithmetic. Each rule checks one
relation and reports pass, fail or not applicable (the receipt does not print what the rule
needs), with a sentence that states the numbers, so that a finding always shows which amounts
disagree.

The rules model how real receipts are built: discounts, service charge, tax added on top or
included in the prices, rounding, several payment or tax lines, delivery notes without a
subtotal. They never compensate for reading errors; those are measured separately.

Printed totals are linked by adjustments. Going from one printed total to the next, any of the
printed adjustments may apply: a discount reduces the amount, a service charge adds to it, tax
adds to it (or is taken out when a tax-inclusive subtotal is followed by the total excluding
tax), rounding adds its printed sign. Receipts print these in different orders and not every
adjustment applies between every pair of totals, so a relation holds when some combination of
the printed adjustments explains the difference; the explanation names the combination.
"""

from collections.abc import Callable, Sequence
from dataclasses import dataclass
from decimal import Decimal
from enum import StrEnum
from itertools import pairwise, product

from docfraud.reading.fields import AmountRole, ReceiptFields

R = AmountRole
CENT = Decimal("0.01")
HALF_CENT = Decimal("0.005")
ZERO = Decimal(0)

# Printed totals from the bottom of the arithmetic to the top.
TOTALS = (R.SUBTOTAL, R.TOTAL_EXCLUDING_TAX, R.TOTAL_INCLUDING_TAX, R.TOTAL_DUE)
# Roles that a receipt prints once in meaning; a repetition must repeat the same value. A second
# tax line of zero (GST and an unused service tax) is not a repetition.
ONCE_PER_RECEIPT = (*TOTALS, R.TAX, R.ROUNDING, R.CHANGE)


class Status(StrEnum):
    """Outcome of one rule on one receipt."""

    PASS = "pass"  # noqa: S105 - an outcome, not a password
    FAIL = "fail"
    NOT_APPLICABLE = "not_applicable"


@dataclass(frozen=True, slots=True)
class Outcome:
    """A rule's verdict and the arithmetic behind it, in words."""

    status: Status
    reason: str = ""

    @classmethod
    def na(cls, reason: str) -> Outcome:
        """The receipt does not print what the rule needs."""
        return cls(Status.NOT_APPLICABLE, reason)


type Rule = Callable[[ReceiptFields], Outcome]


def close(a: Decimal, b: Decimal) -> bool:
    """Equal to the cent."""
    return abs(a - b) <= CENT


@dataclass(frozen=True, slots=True)
class Adjustment:
    """A printed amount that may lie between two totals, and the signs it may take."""

    label: str
    value: Decimal
    signs: tuple[int, ...]


def adjustments(f: ReceiptFields) -> list[Adjustment]:
    """Every printed discount, service charge, tax and rounding amount, with its allowed signs.

    When no tax amount is printed, the taxes of the tax summary rows stand in for it.
    """
    signs = {
        R.DISCOUNT: ("discount", (0, -1)),
        R.SERVICE_CHARGE: ("service charge", (0, 1)),
        R.TAX: ("tax", (0, 1, -1)),
        R.ROUNDING: ("rounding", (0, 1)),
    }
    found = [
        Adjustment(label, abs(value) if role is R.DISCOUNT else value, allowed)
        for role, (label, allowed) in signs.items()
        for value in f.amounts.get(role, ())
    ]
    if R.TAX not in f.amounts:  # some receipts print the tax only in the summary table
        found += [
            Adjustment(f"tax ({row.code or 'summary'})", row.tax, signs[R.TAX][1])
            for row in f.tax_summary
            if row.tax is not None
        ]
    return [a for a in found if a.value != ZERO]


def explain(start: Decimal, target: Decimal, adj: Sequence[Adjustment]) -> str | None:
    """The simplest combination of adjustments that leads from ``start`` to ``target``, if any.

    Returns ``""`` when no adjustment is needed and ``None`` when nothing explains the gap.
    """
    best: tuple[int, ...] | None = None
    for signs in product(*(a.signs for a in adj)):
        total = start + sum((s * a.value for s, a in zip(signs, adj, strict=True)), ZERO)
        if close(total, target) and (best is None or sum(map(abs, signs)) < sum(map(abs, best))):
            best = signs
    if best is None:
        return None
    return "".join(
        f" {'+' if s > 0 else '-'} {a.label} {a.value}" for s, a in zip(best, adj, strict=True) if s
    )


def _label(role: AmountRole) -> str:
    return role.value.replace("_", " ")


def totals_reconcile(f: ReceiptFields) -> Outcome:
    """Each printed total follows from the one below it through the printed adjustments."""
    printed = [(role, value) for role in TOTALS if (value := f.first(role)) is not None]
    if len(printed) < 2:  # noqa: PLR2004 - a relation needs two totals
        return Outcome.na("fewer than two totals printed")
    adj = adjustments(f)
    steps = []
    for (low_role, low), (high_role, high) in pairwise(printed):
        how = explain(low, high, adj)
        if how is None:
            return Outcome(
                Status.FAIL,
                f"{_label(high_role)} {high} does not follow from {_label(low_role)} {low} "
                "with the printed discounts, service charge, tax and rounding",
            )
        steps.append(f"{_label(low_role)} {low}{how} = {_label(high_role)} {high}")
    return Outcome(Status.PASS, "; ".join(steps))


def items_reconcile(f: ReceiptFields) -> Outcome:
    """The line items add up to the lowest printed total, through the printed adjustments."""
    amounts = [i.amount for i in f.items if i.amount is not None]
    lowest = next(((r, v) for r in TOTALS if (v := f.first(r)) is not None), None)
    if not amounts or lowest is None:
        return Outcome.na("no line amounts or no total printed")
    role, total = lowest
    items = sum(amounts, ZERO)
    how = explain(items, total, adjustments(f))
    if how is None:
        return Outcome(
            Status.FAIL,
            f"the {len(amounts)} line amounts add up to {items}, which does not lead to "
            f"{_label(role)} {total}",
        )
    return Outcome(Status.PASS, f"line amounts {items}{how} = {_label(role)} {total}")


def line_amounts(f: ReceiptFields) -> Outcome:
    """Quantity times unit price equals the line amount, on every line that prints all three."""
    lines = [
        (i.description, i.quantity, i.unit_price, i.amount)
        for i in f.items
        if i.quantity is not None and i.unit_price is not None and i.amount is not None
    ]
    if not lines:
        return Outcome.na("no line prints quantity, unit price and amount")
    wrong = [
        f"{d or 'a line'}: {q} x {p} = {q * p}, printed {a}"
        for d, q, p, a in lines
        if not close(q * p, a)
    ]
    if wrong:
        return Outcome(Status.FAIL, "; ".join(wrong))
    return Outcome(Status.PASS, f"{len(lines)} lines multiply out")


def change_given(f: ReceiptFields) -> Outcome:
    """The change is the amount paid minus the amount due.

    A payment printed with a minus sign counts by its size. With several payment lines, the
    change may follow from one of them (a 'cash' line repeating the amount due next to the
    amount handed over) or from their sum (a split payment).
    """
    due = f.first(R.TOTAL_DUE) or f.first(R.TOTAL_INCLUDING_TAX)
    change = f.first(R.CHANGE)
    paid = [abs(v) for v in f.amounts.get(R.TENDERED, ())]
    if due is None or change is None or not paid:
        return Outcome.na("amount due, payment or change not printed")
    candidates = [*paid, sum(paid, ZERO)] if len(paid) > 1 else paid
    for amount in candidates:
        if close(amount - due, change):
            return Outcome(Status.PASS, f"paid {amount} - due {due} = change {change}")
    return Outcome(
        Status.FAIL,
        f"paid {' / '.join(map(str, paid))} - due {due} does not give the change {change}",
    )


def repeats_agree(f: ReceiptFields) -> Outcome:
    """A total, the tax, rounding or change printed more than once shows the same value each time.

    Zero tax lines are ignored: a receipt may print GST next to an unused service tax of 0.00.
    """
    repeated = {}
    for role in ONCE_PER_RECEIPT:
        values = f.amounts.get(role, ())
        if role is R.TAX:
            values = tuple(v for v in values if v != ZERO)
        if len(values) > 1:
            repeated[role] = values
    if not repeated:
        return Outcome.na("no total, tax, rounding or change is printed twice")
    differing = [
        f"{_label(role)} printed as {', '.join(map(str, values))}"
        for role, values in repeated.items()
        if not all(close(v, values[0]) for v in values)
    ]
    if differing:
        return Outcome(Status.FAIL, "; ".join(differing))
    return Outcome(Status.PASS, ", ".join(_label(r) for r in repeated) + " repeated consistently")


def tax_rates(f: ReceiptFields) -> Outcome:
    """Each tax summary row's tax is its taxable amount times its rate.

    Tills that compute tax per line and add the rounded line taxes can be off by up to half a
    cent per line, so the tolerance grows with the number of lines (at least one cent).
    """
    rows = [
        (r.code or "row", rate, taxable, tax)
        for r in f.tax_summary
        if (rate := r.rate_percent) is not None
        and (taxable := r.taxable_amount) is not None
        and (tax := r.tax) is not None
    ]
    if not rows:
        return Outcome.na("no tax summary row prints rate, taxable amount and tax")
    tolerance = max(CENT, HALF_CENT * len(f.items))
    wrong = [
        f"{code} {rate} % of {taxable} is {(taxable * rate / 100).quantize(CENT)}, printed {tax}"
        for code, rate, taxable, tax in rows
        if abs(taxable * rate / 100 - tax) > tolerance
    ]
    if wrong:
        return Outcome(Status.FAIL, "; ".join(wrong))
    return Outcome(Status.PASS, f"{len(rows)} tax rows match their rates")


RULES: dict[str, Rule] = {
    "totals_reconcile": totals_reconcile,
    "items_reconcile": items_reconcile,
    "line_amounts": line_amounts,
    "change_given": change_given,
    "repeats_agree": repeats_agree,
    "tax_rates": tax_rates,
}


def check(f: ReceiptFields) -> dict[str, Outcome]:
    """Apply every rule to one receipt."""
    return {name: rule(f) for name, rule in RULES.items()}
