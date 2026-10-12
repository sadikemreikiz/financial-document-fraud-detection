from pathlib import Path

import pytest

from docfraud.data.models import Document
from docfraud.data.observation import (
    ASSISTANT,
    ObservationLog,
    ObservationPlan,
    ReceiptObservation,
    RegionObservation,
    observation_plan,
)
from docfraud.data.observation_report import (
    Resolution,
    build_report,
    cohen_kappa,
    final_answers,
    load_resolutions,
    render_report,
)
from docfraud.markdown import Table
from tests.data.test_observation import observation_documents, receipt_answers, region_answers


@pytest.mark.parametrize(
    ("first", "second", "expected"),
    [
        (["a", "b", "a"], ["a", "b", "a"], 1.0),
        (["a", "a", "b", "b"], ["a", "b", "b", "b"], 0.5),
        (["a", "b"], ["b", "a"], -1.0),
    ],
)
def test_cohen_kappa(first: list[str], second: list[str], expected: float) -> None:
    assert cohen_kappa(first, second) == pytest.approx(expected)


def test_cohen_kappa_is_undefined_without_variation_or_items() -> None:
    assert cohen_kappa(["a", "a"], ["a", "a"]) is None
    assert cohen_kappa([], []) is None
    with pytest.raises(ValueError, match="same items"):
        cohen_kappa(["a"], [])


def observed(
    tmp_path: Path,
) -> tuple[tuple[Document, ...], ObservationPlan, ObservationLog, ObservationLog]:
    documents = observation_documents(tmp_path / "d")
    plan = observation_plan(documents)
    assistant = ObservationLog(
        observer=ASSISTANT,
        regions={
            item.key: RegionObservation.model_validate(region_answers(visibility=None))
            for item in plan.regions
        },
    )
    sample = sorted(plan.full_form)
    owner_regions = {
        item.key: RegionObservation.model_validate(
            {"visibility": "not_found", "observed_at": "2026-10-12T20:00:00Z", "commit": "x"}
        )
        for item in plan.regions
    }
    owner_regions[sample[0]] = RegionObservation.model_validate(region_answers(tone="lighter"))
    owner_regions[sample[1]] = RegionObservation.model_validate(region_answers())
    owner = ObservationLog(
        observer="emre",
        regions=owner_regions,
        receipts={
            doc_id: ReceiptObservation.model_validate(
                receipt_answers(global_trace="yes", global_trace_what="lighter print")
            )
            for doc_id in plan.receipts
        },
    )
    return documents, plan, assistant, owner


def test_open_disagreements_keep_the_assistants_answer(tmp_path: Path) -> None:
    _, plan, assistant, owner = observed(tmp_path)
    key = sorted(plan.full_form)[0]
    final, still_open = final_answers(plan, assistant, owner, {})
    assert still_open == [(key, "tone")]
    assert final[key]["tone"] == "darker"
    resolved = {key: {"tone": Resolution(value="lighter", reason="looked again together")}}
    final, still_open = final_answers(plan, assistant, owner, resolved)
    assert (still_open, final[key]["tone"]) == ([], "lighter")


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("change", "whole_value", "nothing to resolve"),
        ("tone", "purple", "not a choice"),
    ],
)
def test_invalid_resolutions_are_refused(
    tmp_path: Path, field: str, value: str, message: str
) -> None:
    _, plan, assistant, owner = observed(tmp_path)
    key = sorted(plan.full_form)[0]
    resolutions = {key: {field: Resolution(value=value, reason="")}}
    with pytest.raises(ValueError, match=message):
        final_answers(plan, assistant, owner, resolutions)


def test_resolutions_outside_the_sample_are_refused(tmp_path: Path) -> None:
    _, plan, assistant, owner = observed(tmp_path)
    outside = next(item.key for item in plan.regions if item.key not in plan.full_form)
    resolutions = {outside: {"tone": Resolution(value="lighter", reason="")}}
    with pytest.raises(ValueError, match="outside the agreement sample"):
        final_answers(plan, assistant, owner, resolutions)


def test_the_report_has_every_table_of_the_fixed_analysis(tmp_path: Path) -> None:
    documents, plan, assistant, owner = observed(tmp_path)
    tables = {t.title: t for t in build_report(documents, plan, assistant, owner, {})}
    assert list(tables) == [
        "Observers and coverage",
        "Visibility at 100 % (field 10, owner)",
        "Region fields (fields 1 to 9 and 11)",
        "Agreement on the agreement sample",
        "Disagreements",
        "Whole receipts (fields 12 and 13, owner)",
        "Descriptions of the answers 'yes' on whole receipts",
        "Observations against the technique labels",
    ]
    visibility = rows_by(tables["Visibility at 100 % (field 10, owner)"], 1)
    # one region outside the sample, on a receipt with two regions of two receipts in all
    assert visibility["Not found"][2:] == [1, "33.3 %", "25.0 %"]
    assert visibility["Yes, at 100 %"][2:] == [2, "66.7 %", "75.0 %"]
    agreement = rows_by(tables["Agreement on the agreement sample"], 0)
    assert agreement["7. tone"][1:3] == [1, 2]
    assert agreement["1. change"][3] is None  # one and the same answer from both
    assert len(tables["Descriptions of the answers 'yes' on whole receipts"].rows) == 2
    labels = rows_by(tables["Observations against the technique labels"], 0)
    assert labels["cpi"][1:3] == [3, "at_100 2, not_found 1"]  # every toy region is CPI
    assert (
        "1 disagreements are not resolved yet"
        in tables["Region fields (fields 1 to 9 and 11)"].note
    )
    report = render_report(list(tables.values()))
    assert report.startswith("# Forgery observations: design fold")


def rows_by(table: Table, column: int) -> dict[str, list[object]]:
    return {str(row[column]): list(row) for row in table.rows}


def test_resolutions_are_read_from_json(tmp_path: Path) -> None:
    path = tmp_path / "resolutions.json"
    assert load_resolutions(path) == {}
    path.write_text('{"k": {"tone": {"value": "lighter", "reason": "r"}}}', encoding="utf-8")
    assert load_resolutions(path)["k"]["tone"] == Resolution(value="lighter", reason="r")
