import copy
from pathlib import Path
from typing import Any

import pytest

from docfraud.data.models import BBox, Label
from docfraud.decision import Decision
from docfraud.llm.agent import SONNET_AGENT, investigate
from docfraud.llm.agent_rounds import FrozenAgent
from docfraud.llm.agent_validation import (
    agent_validation,
    first_location_hit,
    measured,
    review_queue,
)
from docfraud.llm.agent_validation_report import render_report
from docfraud.llm.case_file import CaseFile, Citation, Location, Observation
from tests.fixtures.llm import FakeAgent
from tests.fusion.test_validation import PAGE_BOX, WORDS, check_results, labels, tables
from tests.llm.test_agent import INSTRUCTION, runner
from tests.llm.test_agent_rounds import ELSEWHERE, document
from tests.llm.test_agent_tools import BOX, case

PAGE = (PAGE_BOX, WORDS)


def test_the_review_queue_is_the_one_phase_5s_check_recorded() -> None:
    training, y_train, validation, documents = tables()
    check = check_results()
    rows = (training.rows("ocr", 0), y_train, validation.rows("ocr", 0), labels(documents))
    queue = review_queue(*rows, check)
    stated = check["policy"]["ocr"]["10 %"]
    assert len(queue.assessments) == stated["reviewed"]  # ties on the toy receipts: all of them
    assert queue.policy.threshold == pytest.approx(stated["threshold"], abs=1e-12)
    order = [r.receipt for r in validation.rows("ocr", 0)]
    assert list(queue.assessments) == [r for r in order if r in queue.assessments]
    assert all(a.decision is Decision.REVIEW for a in queue.assessments.values())
    assert all(a.track == "T2 (OCR)" for a in queue.assessments.values())
    moved = copy.deepcopy(check)
    moved["policy"]["ocr"]["10 %"]["threshold"] += 0.01
    with pytest.raises(ValueError, match="review policy"):
        review_queue(*rows, moved)
    refitted = copy.deepcopy(check)
    refitted["ocr"]["fitted"]["fusion"]["intercept"] += 0.1
    with pytest.raises(ValueError, match="refitted fusion model"):
        review_queue(*rows, refitted)


def case_file(location: BBox | None) -> CaseFile:
    box = None if location is None else Location(**location.model_dump())
    observation = Observation(
        statement="The system flags a region.",
        citations=[Citation(tool="assessment", detail="a finding")],
        location=box,
    )
    return CaseFile(observations=[observation], questions=[], checks=[])


def test_the_first_location_is_judged_like_the_reports_first_region() -> None:
    forged = document("R1", Label.FORGED, BOX)
    hit = first_location_hit(case_file(BOX), forged, *PAGE)
    assert (hit.located, hit.hit) == (True, True)
    assert 0 < hit.baseline < 1
    missed = first_location_hit(case_file(ELSEWHERE), forged, *PAGE)
    assert (missed.located, missed.hit) == (True, False)
    for unlocated in (case_file(None), None):
        found = first_location_hit(unlocated, forged, *PAGE)
        assert (found.located, found.hit, found.baseline) == (False, False, 0.0)


def results(tmp_path: Path) -> dict[str, Any]:
    shown = case()
    grounded = investigate(shown, SONNET_AGENT, INSTRUCTION, runner(tmp_path / "a", FakeAgent()))
    computed = investigate(
        case("R2"), SONNET_AGENT, INSTRUCTION, runner(tmp_path / "b", FakeAgent(computed=True))
    )
    cases = [
        measured(grounded, document("R1", Label.FORGED, BOX), shown.assessment, PAGE),
        measured(computed, document("R2", Label.AUTHENTIC), shown.assessment, None),
    ]
    return agent_validation(cases, resamples=20)


def test_the_run_measures_grounding_cost_time_and_where_it_points(tmp_path: Path) -> None:
    found = results(tmp_path)
    assert (found["receipts"], found["forged"], found["failures"]) == (2, 1, {})
    grounding = found["grounding"]
    assert grounding["case_files"] == 2
    assert grounding["violations"] == ["R2: 1.412"]  # the fake's computed probability
    assert grounding["rate"] == 1 - 1 / grounding["facts"]
    assert found["cost_usd"]["total"] == sum(c["cost_usd"] for c in found["cases"])
    assert found["seconds"]["timed"] == 2
    assert found["seconds"]["median"] == found["cases"][0]["seconds"]
    for key in ("first_location", "report_first_region"):
        assert (found[key]["receipts"], found[key]["located"], found[key]["hits"]) == (1, 1, 1)
    assert found["cases"][1]["first_location"] is None  # authentic: nothing to point at
    with pytest.raises(ValueError, match="empty"):
        agent_validation([])


def test_the_report_lists_every_measure_violation_and_case(tmp_path: Path) -> None:
    frozen = FrozenAgent(
        version=2,
        settings=SONNET_AGENT,
        token_budget=34_000,
        instruction="agent-v2.md",
        instruction_sha256="0" * 64,
        commit="f" * 40,
    )
    config = {"frozen": frozen.model_dump(mode="json"), "policy": {"budget": 0.1, "threshold": 0.7}}
    text = render_report(results(tmp_path), config, "e" * 40)
    assert text.startswith("# The investigation agent on the validation split\n")
    assert "version 2 of its instruction, `claude-sonnet-5-5` at medium effort" in text
    assert "34,000 tokens per case, frozen at `fffffff`" in text
    assert "at commit `eeeeeee`, on the 2 validation receipts" in text
    assert "the 10 % budget" in text
    assert "| R2 | 1.412 |" in text
    assert "The check looks up numbers and boxes, not what a sentence claims." in text
    assert "### Where it points" in text
    assert "| R1 | forged | yes |" in text
