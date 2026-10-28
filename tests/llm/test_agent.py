from pathlib import Path

from docfraud.llm.agent import HAIKU_AGENT, MAX_TOOL_CALLS, SONNET_AGENT, investigate, task
from docfraud.llm.cost import Ledger
from docfraud.llm.run import Runner
from docfraud.llm.store import ResponseStore
from tests.fixtures.llm import SECONDS, FakeAgent
from tests.llm.test_agent_tools import case

INSTRUCTION = "Draft the case file."


def runner(tmp_path: Path, agent: FakeAgent) -> Runner:
    return Runner(agent, ResponseStore(tmp_path / "store"), Ledger(tmp_path / "ledger.jsonl"))


def test_the_agent_calls_tools_step_by_step_and_writes_a_grounded_case_file(
    tmp_path: Path,
) -> None:
    agent = FakeAgent()
    done = investigate(case(), HAIKU_AGENT, INSTRUCTION, runner(tmp_path, agent))
    assert done.read.failure is None
    assert [c.name for c in done.calls] == ["assessment", "crop", "ocr_rows"]
    assert len(done.responses) == 3
    first, second, third = agent.sent
    assert (first.turns, first.user) == ((), task("R1"))
    assert [t.role for t in third.turns] == ["assistant", "user", "assistant", "user"]
    assert third.turns[:2] == second.turns  # the conversation only grows, unchanged
    grounding = done.grounding()
    assert grounding is not None
    assert grounding.violations == ()
    assert done.rendered is not None
    assert "(10, 20, 30 x 12 px)" in done.rendered
    assert done.received[0] == "R1"
    assert done.seconds == 3 * SECONDS
    assert done.tokens == 3 * 1200
    assert done.cost_usd > 0


def test_a_case_run_again_replays_its_stored_steps_for_nothing(tmp_path: Path) -> None:
    agent = FakeAgent()
    first = investigate(case(), HAIKU_AGENT, INSTRUCTION, runner(tmp_path, agent))
    again = investigate(case(), HAIKU_AGENT, INSTRUCTION, runner(tmp_path, agent))
    assert len(agent.sent) == 3
    assert again.read == first.read
    assert len(Ledger(tmp_path / "ledger.jsonl").entries()) == 3


def test_a_computed_number_fails_the_grounding_check(tmp_path: Path) -> None:
    done = investigate(
        case(), SONNET_AGENT, INSTRUCTION, runner(tmp_path, FakeAgent(computed=True))
    )
    grounding = done.grounding()
    assert grounding is not None
    assert grounding.violations == ("1.412",)


def test_calls_beyond_the_limit_are_refused_and_the_next_step_must_answer(tmp_path: Path) -> None:
    agent = FakeAgent(calls=MAX_TOOL_CALLS + 2)
    done = investigate(case(), HAIKU_AGENT, INSTRUCTION, runner(tmp_path, agent))
    assert len(done.calls) == MAX_TOOL_CALLS
    refused = [b for b in agent.sent[1].turns[1].content if b.get("is_error")]
    assert len(refused) == 3  # the assessment and seven crops ran; three crops did not
    assert agent.sent[1].tool_choice == "none"
    assert done.read.case_file is not None


def test_the_token_budget_ends_the_calls(tmp_path: Path) -> None:
    agent = FakeAgent()
    done = investigate(case(), HAIKU_AGENT, INSTRUCTION, runner(tmp_path, agent), budget=1000)
    assert agent.sent[1].tool_choice == "none"
    assert len(done.responses) == 2
    assert done.read.case_file is not None


def test_a_step_without_an_answer_stops_the_case_without_cost(tmp_path: Path) -> None:
    agent = FakeAgent(silent=1)
    done = investigate(case(), HAIKU_AGENT, INSTRUCTION, runner(tmp_path, agent))
    assert done.read.failure == "not answered"
    assert done.calls == ()
    assert done.rendered is None
    assert done.grounding() is None
    assert done.seconds == 0
    resumed = investigate(case(), HAIKU_AGENT, INSTRUCTION, runner(tmp_path, agent))
    assert resumed.read.case_file is not None
