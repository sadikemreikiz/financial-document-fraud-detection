"""The investigation agent: it drafts the case file of one receipt sent to review (ADR-0031, 5).

The agent receives the receipt's identifier and reads the receipt only through its tools
(:mod:`docfraud.llm.agent_tools`). Each step is one request that repeats the conversation so
far; the model calls tools, their results go back in the next step, and the model ends with a
case file in the fixed form (:mod:`docfraud.llm.case_file`). After :data:`MAX_TOOL_CALLS`
calls, or once the case has used its token budget, the next step offers no further call and
asks for the case file; a call beyond the limit is answered with an error, not run.

Every step goes through the runner, so it is paid for once, recorded in the ledger and stored:
an agent run again replays its stored steps for nothing and goes on where it stopped.
"""

from collections.abc import Mapping
from dataclasses import dataclass
from decimal import Decimal

from docfraud.llm.agent_tools import TOOLS, Case, ToolOutput, call
from docfraud.llm.case_file import (
    Grounding,
    Read,
    case_file_schema,
    check_case_file,
    read_case_file,
    render_case_file,
)
from docfraud.llm.cost import cost
from docfraud.llm.provider import Request, Response, Settings, Thinking, Turn
from docfraud.llm.run import Runner

MAX_TOOL_CALLS = 8
"""At most this many tool calls per case (ADR-0031, section 5)."""

HAIKU_AGENT = Settings(
    model="claude-haiku-5-5", max_tokens=8000, thinking=Thinking.ADAPTIVE, effort="medium"
)
SONNET_AGENT = HAIKU_AGENT.model_copy(update={"model": "claude-sonnet-5-5"})
"""The two models of the pilot, both at medium effort (ADR-0031, section 5)."""

LIMIT_REACHED = ToolOutput(
    '{"error": "the limit of tool calls is reached: write the case file"}', error=True
)


@dataclass(frozen=True, slots=True)
class ToolCall:
    """One call the agent made: the tool, its arguments and what it returned."""

    name: str
    arguments: Mapping[str, object]
    output: ToolOutput


@dataclass(frozen=True, slots=True)
class Investigation:
    """What the agent did on one case: its steps, its tool calls and its case file."""

    receipt: str
    settings: Settings
    read: Read
    calls: tuple[ToolCall, ...]
    responses: tuple[Response, ...]

    @property
    def received(self) -> tuple[str, ...]:
        """What the agent was given: the receipt's identifier and every tool output."""
        return (self.receipt, *(c.output.text for c in self.calls))

    @property
    def rendered(self) -> str | None:
        """The case file from its template, if the agent wrote one in the fixed form."""
        found = self.read.case_file
        return render_case_file(self.receipt, found) if found is not None else None

    def grounding(self) -> Grounding | None:
        """The grounding check of the rendered case file, if there is one."""
        rendered = self.rendered
        return check_case_file(rendered, self.received) if rendered is not None else None

    @property
    def cost_usd(self) -> Decimal:
        """What its steps cost."""
        return sum(
            (cost(r.usage, self.settings.model, batch=r.batch) for r in self.responses),
            Decimal(0),
        )

    @property
    def tokens(self) -> int:
        """Every token its steps used, input of every kind and output."""
        return sum(_tokens(r) for r in self.responses)

    @property
    def seconds(self) -> float | None:
        """The time its steps took, if every step was timed."""
        times = [r.seconds for r in self.responses]
        return sum(t for t in times if t is not None) if None not in times else None


def _tokens(response: Response) -> int:
    u = response.usage
    return u.input_tokens + u.output_tokens + u.cache_write_tokens + u.cache_read_tokens


def task(receipt: str) -> str:
    """The first user message of a case."""
    return f"Draft the case file of receipt {receipt}."


def investigate(
    case: Case,
    settings: Settings,
    instruction: str,
    runner: Runner,
    *,
    budget: int | None = None,
) -> Investigation:
    """Run the agent on one case until it writes its case file or stops.

    ``budget`` is the case's token budget; ``None`` sets none (the pilot measures it).
    """
    turns: list[Turn] = []
    calls: list[ToolCall] = []
    responses: list[Response] = []
    last = False  # whether this step must write the case file
    for _ in range(MAX_TOOL_CALLS + 2):  # every call in its own step, then the case file
        request = Request(
            settings=settings,
            system=instruction,
            user=task(case.receipt),
            answer_schema=case_file_schema(),
            tools=TOOLS,
            turns=tuple(turns),
            tool_choice="none" if last else "auto",
        )
        (response,) = runner.answer([request], batch=False)
        found = (case.receipt, settings)
        if response is None:
            return Investigation(*found, Read(None, "not answered"), tuple(calls), tuple(responses))
        responses.append(response)
        if response.stop_reason != "tool_use":
            read = read_case_file(response)
            return Investigation(*found, read, tuple(calls), tuple(responses))
        results = []
        for use in (b for b in response.content if b.get("type") == "tool_use"):
            arguments = use.get("input")
            given = arguments if isinstance(arguments, dict) else {}
            if len(calls) < MAX_TOOL_CALLS:
                output = call(case, str(use.get("name")), given)
                calls.append(ToolCall(str(use.get("name")), given, output))
            else:
                output = LIMIT_REACHED
            results.append(output.block(str(use.get("id"))))
        turns += [
            Turn(role="assistant", content=response.content),
            Turn(role="user", content=tuple(results)),
        ]
        spent = sum(_tokens(r) for r in responses)
        last = len(calls) >= MAX_TOOL_CALLS or (budget is not None and spent >= budget)
    return Investigation(
        case.receipt, settings, Read(None, "too many steps"), tuple(calls), tuple(responses)
    )
