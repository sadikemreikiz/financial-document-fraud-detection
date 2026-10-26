import inspect
from pathlib import Path
from types import SimpleNamespace
from typing import Any

import numpy as np
import pytest

from docfraud.data.models import BBox
from docfraud.detectors.finding import COPY_MOVE, LINKING, METADATA, SYSTEM_1, region_finding
from docfraud.detectors.rules import RULES, Outcome, Status
from docfraud.forensics.learned.inference import QUARTER_SQUARE
from docfraud.forensics.learned.results import Group
from docfraud.fusion import training_inputs as module
from docfraud.fusion.evidence import SystemTwo, page_findings, receipt_evidence
from docfraud.fusion.inputs import PRACTICAL
from docfraud.fusion.training_inputs import (
    InputsTable,
    load_table,
    save_table,
    stored_system_2,
    summary,
    training_inputs,
)
from docfraud.linking.values import LocatedValue
from docfraud.reading.fields import AmountRole, FieldSources, Source
from tests.detectors.test_copy_move import ink_patch, noisy_page

DUE_BOX = BBox(x=100, y=60, width=40, height=12)
SOURCES = FieldSources(
    amounts={AmountRole.TOTAL_DUE: (Source(row=0, start=0, end=5, segments=(0,), box=DUE_BOX),)}
)
NOT_APPLICABLE = {rule: Outcome.na("not printed") for rule in RULES}
FAIL = NOT_APPLICABLE | {
    "change_given": Outcome(Status.FAIL, "paid 20.00 - due 10.60 does not give 9.00")
}
PASS = NOT_APPLICABLE | {"change_given": Outcome(Status.PASS, "paid 20.00 - due 10.60 = 9.40")}
PAGE = (region_finding(SYSTEM_1, 0.0, None),)  # every page has a system 1 finding
WORDS = [BBox(x=10, y=10 + 30 * k, width=200, height=24) for k in range(4)]


def test_page_findings_report_system_1_always_and_copy_move_and_traits_when_found() -> None:
    page = noisy_page()
    plain = page_findings(page, WORDS, ())
    assert [f.detector for f in plain] == [SYSTEM_1]
    page[20:50, 30:50] = ink_patch(30, 20)
    page[100:130, 180:200] = page[20:50, 30:50]
    found = page_findings(page, WORDS, ("EXIF",))
    assert [f.detector for f in found] == [SYSTEM_1, COPY_MOVE, METADATA]


def test_receipt_evidence_links_system_2_to_a_value_a_failing_rule_reads() -> None:
    system_2 = SystemTwo(0.9, BBox(x=105, y=62, width=8, height=8))  # on the amount due
    inputs, findings = receipt_evidence(0.3, system_2, PAGE, FAIL, SOURCES)
    assert list(inputs) == list(PRACTICAL)
    assert (inputs["system 2"], inputs["link 1"], inputs["link 2"]) == (0.9, 1.0, 1.0)
    assert sum(f.detector == LINKING for f in findings) == 2
    elsewhere = SystemTwo(0.9, BBox(x=5, y=5, width=8, height=8))
    unlinked, _ = receipt_evidence(0.3, elsewhere, PAGE, FAIL, SOURCES)
    assert (unlinked["link 1"], unlinked["link 2"]) == (0.0, 0.0)


def test_evidence_cannot_take_a_document() -> None:
    assert "document" not in inspect.signature(receipt_evidence).parameters
    assert "document" not in inspect.signature(page_findings).parameters


def table(n: int = 20, seeds: tuple[int, ...] = (0, 1)) -> InputsTable:
    receipts = [f"R{i:02d}" for i in range(n)]
    forged = [i % 4 == 0 for i in range(n)]
    readings = {
        "transcript": [(FAIL if f else PASS, SOURCES) for f in forged],
        "ocr": [(PASS, SOURCES) for _ in forged],
    }
    system_2 = {
        seed: {r: SystemTwo(0.1 * seed, BBox(x=5, y=5, width=4, height=4)) for r in receipts}
        for seed in seeds
    }
    return training_inputs(
        receipts,
        forged,
        [i % 5 for i in range(n)],
        readings,
        page=[PAGE] * n,
        system_2=system_2,
    )


def test_every_receipt_gets_inputs_per_track_and_seed_with_the_rules_score_out_of_fold() -> None:
    built = table()
    assert len(built.receipts) == 20 * 2 * 2
    rows = built.rows("transcript", 0)
    assert [r.receipt for r in rows] == [f"R{i:02d}" for i in range(20)]
    forged_score = np.mean([r.inputs["rules"] for r in rows if r.receipt in {"R00", "R04"}])
    authentic_score = np.mean([r.inputs["rules"] for r in rows if r.receipt in {"R01", "R02"}])
    assert forged_score > authentic_score  # the rule fails on forged receipts only
    assert all(r.inputs["system 2"] == pytest.approx(0.1) for r in built.rows("ocr", 1))


def test_training_inputs_refuse_missing_outputs_and_mismatched_lengths() -> None:
    receipts = [f"R{i:02d}" for i in range(10)]
    forged = [i % 2 == 0 for i in range(10)]
    readings = {"ocr": [(PASS, SOURCES)] * 10}
    partial = {r: SystemTwo(0.1, DUE_BOX) for r in receipts[1:]}
    with pytest.raises(ValueError, match="no output for 1 receipts"):
        training_inputs(
            receipts,
            forged,
            [i % 5 for i in range(10)],
            readings,
            page=[PAGE] * 10,
            system_2={0: partial},
        )
    with pytest.raises(ValueError, match="one label"):
        training_inputs(
            ["R0"], [True, False], [0], {"ocr": [(PASS, SOURCES)]}, page=[PAGE], system_2={}
        )


def test_the_table_round_trips_and_summarises_without_receipts(tmp_path: Path) -> None:
    built = table(n=10, seeds=(0,))
    path = tmp_path / "inputs.json"
    save_table(built, path)
    assert load_table(path) == built
    assert built.receipts[0].values == (LocatedValue("total_due", DUE_BOX),)
    aggregates = summary(built)
    assert set(aggregates) == {"ocr, seed 0", "transcript, seed 0"}
    assert aggregates["ocr, seed 0"]["receipts"] == 10
    assert set(aggregates["ocr, seed 0"]["means"]) == set(PRACTICAL)
    assert "R00" not in str(aggregates)


def results(*keys: str) -> Any:
    measured = [
        SimpleNamespace(
            group=Group.HELD_OUT, key=k, scores={QUARTER_SQUARE: 0.5}, top_region=DUE_BOX
        )
        for k in keys
    ]
    other = {"scores": {QUARTER_SQUARE: 0.9}, "top_region": DUE_BOX}
    measured.append(SimpleNamespace(group=Group.VALIDATION, key="V", **other))
    return SimpleNamespace(measured=measured)


def test_system_2_comes_from_the_round_that_held_each_receipt_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    by_round = {f"round-{k}": results(f"R{k}") for k in range(5)}
    monkeypatch.setattr(module, "read_results", lambda directory: by_round[directory.parts[-3]])
    found = stored_system_2(tmp_path, 0)
    assert set(found) == {f"R{k}" for k in range(5)}
    assert found["R3"] == SystemTwo(0.5, DUE_BOX)
    by_round["round-4"] = results("R0")
    with pytest.raises(ValueError, match="two rounds"):
        stored_system_2(tmp_path, 0)
