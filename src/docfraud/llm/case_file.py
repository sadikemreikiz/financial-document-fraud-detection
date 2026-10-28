"""The agent's case file: its fixed form, its template and its grounding check (ADR-0031, 5-6).

The agent answers in a fixed form: observations, each resting on at least one tool output and
located by a box where it has one; questions for the reviewer; and checks the reviewer could
make. The form has no place for a verdict or a probability of the agent's own. A template
renders it, as the audit report is rendered.

The grounding check does not use the template. It reads every number and box back from the
rendered text and looks each up in what the agent received, the receipt's identifier and its
tools' outputs:

- a **box** ``(x, y, w x h px)`` is grounded if an output holds exactly that box;
- a **number** is grounded if a value in an output, or a number written inside one of its
  strings, gives the number when rounded to the decimals the case file prints (a share is also
  allowed as a percentage).

A number the agent computed, a sum or a difference, is therefore a violation unless the receipt
or the system already states it, which is how the instruction's ban on arithmetic is checked.
"""

import json
import re
from collections.abc import Iterator, Sequence
from dataclasses import dataclass
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docfraud.data.models import BBox
from docfraud.llm.provider import Response

type ToolName = Literal["assessment", "read_fields", "ocr_rows", "crop"]

_BOX = re.compile(r"\((\d+), (\d+), (\d+) x (\d+) px\)")
_NUMBER = re.compile(r"[-+]?\d+(?:\.\d+)?")


class _Form(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Citation(_Form):
    """The tool output an observation rests on, and what in it."""

    tool: ToolName
    detail: str = Field(description="What in that output: a finding, a field, a row or a crop")


class Location(_Form):
    """A box on the page, in pixels from the top left."""

    x: int
    y: int
    width: int
    height: int

    def box(self) -> BBox:
        """The location as a box."""
        return BBox(x=self.x, y=self.y, width=self.width, height=self.height)


class Observation(_Form):
    """One thing the agent saw in the system's outputs or on the page."""

    statement: str
    citations: list[Citation] = Field(description="At least one tool output it rests on")
    location: Location | None = Field(description="Where on the page, or null if nowhere")


class CaseFile(_Form):
    """The case file of one receipt sent to review."""

    observations: list[Observation] = Field(description="The most important first")
    questions: list[str] = Field(description="Questions for the reviewer")
    checks: list[str] = Field(description="Checks the reviewer could make")


def case_file_schema() -> dict[str, object]:
    """The JSON schema of :class:`CaseFile`, in the form the structured-output API accepts."""
    return CaseFile.model_json_schema()


@dataclass(frozen=True, slots=True)
class Read:
    """A case file read from an answer, or why it does not take the fixed form."""

    case_file: CaseFile | None
    failure: str | None = None


def read_case_file(response: Response | None) -> Read:
    """The case file of an answer, checked against the fixed form.

    The form needs at least one observation, every observation resting on at least one tool
    output, and every location a box of positive size.
    """
    if response is None:
        return Read(None, "not answered")
    if response.answer is None:
        return Read(None, response.error or response.stop_reason or "no answer")
    try:
        found = CaseFile.model_validate(json.loads(response.answer))
    except (json.JSONDecodeError, ValidationError) as error:
        return Read(None, f"answer form: {type(error).__name__}")
    failure = _form_failure(found)
    return Read(None, failure) if failure is not None else Read(found)


def _form_failure(case_file: CaseFile) -> str | None:
    """What keeps a case file valid in JSON from the fixed form, if anything."""
    if not case_file.observations:
        return "no observation"
    if any(not o.citations for o in case_file.observations):
        return "an observation cites no tool output"
    if any(
        o.location is not None and (o.location.width < 1 or o.location.height < 1)
        for o in case_file.observations
    ):
        return "a location is not a box"
    return None


def box_text(box: BBox) -> str:
    """A box as case files state it, as audit reports do."""
    return f"({box.x}, {box.y}, {box.width} x {box.height} px)"


def render_case_file(receipt: str, case_file: CaseFile) -> str:
    """The case file in Markdown. Its fixed wording holds no number."""
    lines = [
        f"# Case file: {receipt}",
        "",
        "Drafted by a language model from the system's outputs, for the reviewer. It states no "
        "verdict; every number and box in it comes from what the model read.",
        "",
        "## Observations",
        "",
    ]
    for o in case_file.observations:
        lines.append(f"- {o.statement}")
        cited = "; ".join(f"{c.tool} ({c.detail})" for c in o.citations)
        lines.append(f"  - Rests on: {cited}")
        if o.location is not None:
            lines.append(f"  - Location: {box_text(o.location.box())}")
    for title, items in (
        ("Questions for the reviewer", case_file.questions),
        ("Checks the reviewer could make", case_file.checks),
    ):
        if items:
            lines += ["", f"## {title}", ""]
            lines += [f"- {item}" for item in items]
    return "\n".join(lines) + "\n"


def first_location(case_file: CaseFile) -> BBox | None:
    """The location of the first observation, which RQ4's measure counts; ``None`` is a miss."""
    location = case_file.observations[0].location
    return location.box() if location is not None else None


@dataclass(frozen=True, slots=True)
class Grounding:
    """How many of a case file's facts what the agent received backs, and which it does not."""

    facts: int
    violations: tuple[str, ...]

    @property
    def rate(self) -> float:
        """The share of grounded facts; 1 for a case file without facts."""
        return 1 - len(self.violations) / self.facts if self.facts else 1.0


def _numbers(item: Any) -> Iterator[float]:
    """Every number in a JSON value, including those written inside its strings."""
    if isinstance(item, bool):
        return
    if isinstance(item, int | float):
        yield float(item)
        yield float(item) * 100  # a share may be printed as a percentage
    elif isinstance(item, str):
        yield from (float(n) for n in _NUMBER.findall(item))
    elif isinstance(item, dict):
        for key, value in item.items():
            yield from _numbers(key)
            yield from _numbers(value)
    elif isinstance(item, list):
        for value in item:
            yield from _numbers(value)


def _boxes(item: Any) -> Iterator[BBox]:
    """Every box in a JSON value: an object with integer x, y, width and height."""
    if isinstance(item, dict):
        if all(isinstance(item.get(k), int) for k in ("x", "y", "width", "height")):
            yield BBox(x=item["x"], y=item["y"], width=item["width"], height=item["height"])
        for value in item.values():
            yield from _boxes(value)
    elif isinstance(item, list):
        for value in item:
            yield from _boxes(value)


def _parsed(text: str) -> Any:
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        return text


def _decimals(number: str) -> int:
    return len(number.split(".")[1]) if "." in number else 0


def check_case_file(rendered: str, received: Sequence[str]) -> Grounding:
    """Look every number and box of a rendered case file up in what the agent received.

    ``received`` holds the texts the agent was given: the receipt's identifier and every tool
    output, JSON where the tools write JSON.
    """
    parsed = [_parsed(text) for text in received]
    boxes = {b for item in parsed for b in _boxes(item)}
    allowed = [n for item in parsed for n in _numbers(item)]
    violations: list[str] = []
    facts = 0
    for match in _BOX.finditer(rendered):
        facts += 1
        x, y, width, height = map(int, match.groups())
        if BBox(x=x, y=y, width=width, height=height) not in boxes:
            violations.append(match.group(0))
    for number in _NUMBER.findall(_BOX.sub(" ", rendered)):
        facts += 1
        places = _decimals(number)
        target = round(float(number), places)
        if not any(round(v, places) == target for v in allowed):
            violations.append(number)
    return Grounding(facts, tuple(violations))
