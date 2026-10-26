"""The grounding check: is every fact a report states traceable to the evidence? (protocol, 5).

The check does not use the report's template. It reads the facts back from the rendered text,
every number and every box, and looks each up in the assessment the report was made from:

- a **box** ``(x, y, w x h px)`` is grounded if a finding of the assessment has exactly that box;
- a **number** is grounded if a value anywhere in the assessment, a probability, threshold,
  contribution, score, coordinate or a number inside a finding's reason or a name, gives the
  number when rounded to the decimals the report prints (a budget of 0.1 is also printed as 10).

The grounding rate is the share of facts that are grounded; the protocol's target is 100 %, and
every violation is listed. A template makes the rate 100 % by construction; this check is what
shows it. The check is lenient in one way: a number that happens to equal another value of the
assessment passes. It catches numbers the evidence does not contain at all, which is what an
invented fact looks like.
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from typing import Any

from docfraud.data.models import BBox
from docfraud.reporting.assessment import Assessment

_BOX = re.compile(r"\((\d+), (\d+), (\d+) x (\d+) px\)")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


@dataclass(frozen=True, slots=True)
class Grounding:
    """How many of a report's facts the assessment backs, and which it does not."""

    facts: int
    violations: tuple[str, ...]

    @property
    def rate(self) -> float:
        """The share of grounded facts; 1 for a report without facts."""
        return 1 - len(self.violations) / self.facts if self.facts else 1.0


def _values(item: Any) -> Iterator[float]:
    """Every number in a dumped assessment, including those written inside its strings."""
    if isinstance(item, bool):
        return
    if isinstance(item, int | float):
        yield float(item)
        yield float(item) * 100  # a share may be printed as a percentage
    elif isinstance(item, str):
        yield from (float(n) for n in _NUMBER.findall(item))
    elif isinstance(item, dict):
        for key, value in item.items():
            yield from _values(key)
            yield from _values(value)
    elif isinstance(item, list | tuple):
        for value in item:
            yield from _values(value)


def _decimals(number: str) -> int:
    return len(number.split(".")[1]) if "." in number else 0


def check_grounding(report: str, assessment: Assessment) -> Grounding:
    """Look every number and box of a rendered report up in its assessment."""
    boxes = {b for e in assessment.evidence for f in e.findings for b in f.boxes}
    violations: list[str] = []
    facts = 0
    for match in _BOX.finditer(report):
        facts += 1
        x, y, width, height = map(int, match.groups())
        if BBox(x=x, y=y, width=width, height=height) not in boxes:
            violations.append(match.group(0))
    allowed = list(_values(assessment.model_dump(mode="json")))
    for number in _NUMBER.findall(_BOX.sub(" ", report)):
        facts += 1
        places = _decimals(number)
        target = round(float(number), places)
        if not any(round(v, places) == target for v in allowed):
            violations.append(number)
    return Grounding(facts, tuple(violations))
