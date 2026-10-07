"""Arithmetic relations that hold on every correctly read, authentic receipt.

They serve two purposes before the accounting rules of Phase 3 exist: measuring the parser
indirectly (a relation failing on an authentic receipt points to a reading error) and checking
gold annotations for typos. Each check returns ``None`` when the receipt does not print all the
amounts it needs.

Authentic receipts can still fail some relations, for example when zero-rated items are left out
of the printed total excluding tax, so pass rates are lower bounds, not accuracy.
"""

from collections.abc import Callable
from decimal import Decimal

from docfraud.reading.fields import AmountRole, ReceiptFields

R = AmountRole

CENT = Decimal("0.01")

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


UNIQUE_ROLES = (
    R.SUBTOTAL,
    R.TOTAL_EXCLUDING_TAX,
    R.TAX,
    R.TOTAL_INCLUDING_TAX,
    R.ROUNDING,
    R.TOTAL_DUE,
    R.CHANGE,
)


def repeats_agree(f: ReceiptFields) -> bool | None:
    """Every once-per-receipt amount printed more than once has the same value each time.

    Discounts, service charges and payments can legitimately occur several times with
    different values (item discounts, split payments), so they are excluded.
    """
    repeated = [f.amounts[r] for r in UNIQUE_ROLES if len(f.amounts.get(r, ())) > 1]
    if not repeated:
        return None
    return all(all(close(v, values[0]) for v in values) for values in repeated)


CHECKS: dict[str, Check] = {
    "Items add up to a printed total": items_sum,
    "Quantity x unit price = line amount": line_math,
    "Excl. tax + tax = incl. tax": tax_math,
    "Incl. tax + rounding = amount due": rounding_math,
    "Tendered - due = change": change_math,
    "Repeated once-per-receipt amounts agree": repeats_agree,
}
