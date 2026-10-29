"""The committed ``demo-cards`` record agrees with the records of the single evaluation."""

from pathlib import Path

from docfraud.cli.agent import AGENT
from docfraud.cli.common import REPORTED_RUNS
from docfraud.cli.demo import DEMO_UNLOCK
from docfraud.cli.fusion import _latest
from docfraud.demo.cards import RUN_NAME
from docfraud.demo.gallery import CAUGHT, FALSE_ALARM, GENUINE, MISSED, SHOWN, evenly_spaced
from docfraud.demo.page import error_notes
from docfraud.evaluation.runs import JsonRunStore
from docfraud.fusion.errors import RUN_NAME as ERRORS_RUN
from docfraud.fusion.final import RUN_NAME as TEST_CHECK_RUN
from docfraud.llm.agent_final import RUN_NAME as AGENT_RUN

STORE = JsonRunStore(REPORTED_RUNS)
CARDS = _latest(STORE, RUN_NAME)
RESULTS = CARDS.results


def test_the_outcomes_are_the_test_check_s_and_the_error_analysis_s() -> None:
    check = _latest(STORE, TEST_CHECK_RUN).results
    errors = _latest(STORE, ERRORS_RUN).results
    outcomes = RESULTS["outcomes"]
    policy = check["policy"]["ocr"]
    assert outcomes[CAUGHT] == policy["10 %"]["forged_reviewed"]
    reviewed = outcomes[CAUGHT] + outcomes[FALSE_ALARM]
    assert reviewed == policy["10 %"]["reviewed"] == errors["reviewed"]
    assert outcomes[MISSED] == errors["misses"]["count"]
    assert sum(outcomes.values()) == RESULTS["receipts"] == check["ocr"]["receipts"]
    alarms = RESULTS["alarms"]["20 %"]
    assert alarms["false_alarms"] == policy["20 %"]["reviewed"] - policy["20 %"]["forged_reviewed"]
    assert alarms["genuine"] == outcomes[GENUINE] + outcomes[FALSE_ALARM]


def test_the_policies_are_the_validation_check_s() -> None:
    check = STORE.load(CARDS.config["validation_check"]).results["policy"]["ocr"]
    assert RESULTS["policies"] == [
        {"budget": check[name]["budget"], "threshold": check[name]["threshold"]}
        for name in ("10 %", "20 %")
    ]


def test_the_rule_chose_the_shown_ranks() -> None:
    for group, count in SHOWN.items():
        shown = RESULTS["gallery"][group]
        assert [c["rank"] for c in shown] == evenly_spaced(RESULTS["outcomes"][group], count)
        probabilities = [c["probability"] for c in shown]
        assert probabilities == sorted(probabilities, reverse=True)


def test_the_misses_shown_are_the_error_analysis_s() -> None:
    errors = _latest(STORE, ERRORS_RUN).results["errors"]
    recorded = {e["receipt"]: e["probability"] for e in errors if e["kind"] == "miss"}
    for shown in RESULTS["gallery"][MISSED]:
        assert recorded[shown["receipt"]] == shown["probability"]
    notes = error_notes(Path("reports/README.md").read_text(encoding="utf-8"))
    assert {c["receipt"] for c in RESULTS["gallery"][MISSED]} <= set(notes)


def test_every_catch_shown_has_the_agent_s_case_file() -> None:
    cases = {c["receipt"] for c in _latest(STORE, AGENT_RUN).results["cases"]}
    for shown in RESULTS["gallery"][CAUGHT]:
        assert shown["receipt"] in cases
        assert (AGENT / "test" / f"{shown['receipt']}.md").exists()


def test_the_record_names_its_code_and_its_unlock() -> None:
    assert CARDS.code.dirty is False
    assert CARDS.test_unlocks == (DEMO_UNLOCK,)
