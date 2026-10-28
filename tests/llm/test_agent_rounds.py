from decimal import Decimal
from pathlib import Path

import pytest

from docfraud.data.models import BBox, Document, Label, Region, RegionRole, Split
from docfraud.llm.agent import SONNET_AGENT, investigate
from docfraud.llm.agent_rounds import (
    AgentRound,
    CaseResult,
    case_result,
    kept_version,
    load_rounds,
    save_round,
)
from tests.fixtures.llm import FakeAgent
from tests.llm.test_agent import INSTRUCTION, runner
from tests.llm.test_agent_tools import BOX, case

ELSEWHERE = BBox(x=60, y=40, width=10, height=10)
"""A box that shares no pixel with :data:`BOX`, where the fake agent and report point."""


def result(
    receipt: str = "R1",
    *,
    forged: bool = True,
    violations: int = 0,
    failure: str | None = None,
    hit: bool = True,
) -> CaseResult:
    return CaseResult(
        receipt=receipt,
        forged=forged,
        failure=failure,
        observations=None if failure else 2,
        facts=None if failure else 10,
        violations=None if failure else ("9",) * violations,
        first_location_hit=(hit and failure is None) if forged else None,
        report_hit=True if forged else None,
        tool_calls=(),
        steps=3,
        tokens=14_000,
        cost_usd="0.0250",
        seconds=12.5,
    )


def developed(version: int, *cases: CaseResult) -> AgentRound:
    return AgentRound(
        version=version,
        settings=SONNET_AGENT,
        token_budget=34_000,
        instruction=f"agent-v{version}.md",
        instruction_sha256="0" * 64,
        commit="c" * 40,
        cases=cases,
    )


def document(receipt: str, label: Label, *boxes: BBox) -> Document:
    return Document(
        dataset="toy",
        doc_id=receipt,
        source_id=receipt,
        split=Split.TRAIN,
        label=label,
        image_path=Path(f"{receipt}.png"),
        regions=tuple(Region(box=b, role=RegionRole.MODIFIED) for b in boxes),
        has_region_annotations=bool(boxes),
    )


def test_a_round_counts_failures_violations_and_where_it_points_on_the_forged_cases() -> None:
    found = developed(
        1,
        result("R1", violations=2),
        result("R2", hit=False),
        result("R3", failure="not answered"),
        result("R4", forged=False, violations=1),
    )
    assert (found.failures, found.violations) == (1, 3)
    assert (found.forged, found.hits, found.report_hits) == (3, 1, 3)
    assert found.cost_usd == Decimal("0.1000")


@pytest.mark.parametrize(
    ("second", "kept"),
    [
        ((result("R1", violations=1), result("R2", forged=False)), 2),
        ((result("R1", violations=2), result("R2", forged=False)), 1),  # not fewer violations
        ((result("R1"), result("R2", forged=False, failure="answer form")), 1),  # more failures
        ((result("R1", hit=False), result("R2", forged=False)), 1),  # points worse
    ],
)
def test_version_2_replaces_version_1_only_if_all_three_conditions_hold(
    second: tuple[CaseResult, ...], kept: int
) -> None:
    first = developed(1, result("R1", violations=2), result("R2", forged=False))
    assert kept_version({1: first}) == 1
    assert kept_version({1: first, 2: developed(2, *second)}) == kept


def test_the_rule_refuses_rounds_it_cannot_compare() -> None:
    first = developed(1, result("R1"))
    with pytest.raises(ValueError, match="version 1 has not run"):
        kept_version({2: developed(2, result("R1"))})
    with pytest.raises(ValueError, match="allows versions"):
        kept_version({1: first, 3: developed(3, result("R1"))})
    with pytest.raises(ValueError, match="different cases"):
        kept_version({1: first, 2: developed(2, result("R2"))})


@pytest.mark.parametrize(
    ("label", "boxes", "on_target"),
    [
        (Label.FORGED, (BOX,), True),
        (Label.FORGED, (ELSEWHERE,), False),
        (Label.FORGED, (), False),  # a forged receipt without annotations cannot be hit
        (Label.AUTHENTIC, (), None),
    ],
)
def test_a_case_records_where_the_agent_and_the_report_point(
    tmp_path: Path, label: Label, boxes: tuple[BBox, ...], on_target: bool | None
) -> None:
    shown = case()
    done = investigate(shown, SONNET_AGENT, INSTRUCTION, runner(tmp_path, FakeAgent()))
    found = case_result(done, document("R1", label, *boxes), shown.assessment)
    assert found.failure is None
    assert found.violations == ()
    assert found.first_location_hit is on_target  # the case file points at the report's box
    assert found.report_hit is on_target
    assert [c.tool for c in found.tool_calls] == ["assessment", "crop", "ocr_rows"]
    assert found.steps == len(done.responses)


def test_a_case_without_a_case_file_is_a_failure_and_a_miss(tmp_path: Path) -> None:
    shown = case()
    done = investigate(shown, SONNET_AGENT, INSTRUCTION, runner(tmp_path, FakeAgent(silent=1)))
    found = case_result(done, document("R1", Label.FORGED, BOX), shown.assessment)
    assert found.failure is not None
    assert (found.observations, found.facts, found.violations) == (None, None, None)
    assert found.first_location_hit is False
    assert found.report_hit is True


def test_rounds_are_saved_and_loaded_by_version(tmp_path: Path) -> None:
    rounds = {v: developed(v, result("R1", violations=v)) for v in (1, 2)}
    for version, found in rounds.items():
        save_round(found, tmp_path / f"round-v{version}.json")
    assert load_rounds(tmp_path) == rounds
