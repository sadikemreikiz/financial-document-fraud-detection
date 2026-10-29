"""The demo's cards, from made-up assessments of made-up receipts."""

from typing import Any

import pytest

from docfraud.data.models import Document, Label
from docfraud.decision import Decision, ReviewPolicy
from docfraud.demo.cards import alarms, box, card, demo_cards
from docfraud.demo.gallery import CAUGHT, GENUINE, MISSED
from docfraud.reporting.assessment import Assessment, Evidence
from tests.fusion.test_errors import pushing
from tests.llm.test_agent_rounds import document
from tests.llm.test_agent_tools import BOX, assessment

POLICIES = (ReviewPolicy(0.1, 0.8), ReviewPolicy(0.2, 0.4))


def assessed(receipt: str, probability: float, *evidence: Evidence) -> Assessment:
    found = assessment(receipt)
    decision = POLICIES[0].decide(probability)
    return found.model_copy(
        update={"probability": probability, "decision": decision}
        | ({"evidence": evidence} if evidence else {})
    )


def marked(receipt: str, label: Label, *, handwritten: bool, digital: bool) -> Document:
    return document(receipt, label).model_copy(
        update={"handwritten_annotation": handwritten, "digital_annotation": digital}
    )


def test_a_card_keeps_the_strongest_evidence_each_way_and_the_report() -> None:
    evidence = (
        pushing(("copy-move",), 2.0),
        pushing(("system 2",), 1.0),
        pushing(("link 1",), 0.5),
        pushing(("rules",), 0.25),
        pushing(("link 2",), -0.1),
        pushing(("system 1", "system 1 evidence"), -0.9),
    )
    found = card(assessed("R1", 0.6, *evidence), POLICIES)
    assert [e["inputs"] for e in found["towards"]] == [["copy-move"], ["system 2"], ["link 1"]]
    assert [e["inputs"] for e in found["against"]] == [["system 1", "system 1 evidence"]]
    assert found["decisions"] == {"10 %": "accept", "20 %": "review"}
    assert found["first_region"] == [box(BOX)]
    assert len(found["overlay"]) == 4  # every located finding pushing towards forgery
    assert found["report"].startswith("# Audit report: R1")


def test_alarms_count_the_marks_the_dataset_records() -> None:
    documents = {
        "A": marked("A", Label.AUTHENTIC, handwritten=True, digital=True),
        "B": marked("B", Label.AUTHENTIC, handwritten=False, digital=True),
        "C": marked("C", Label.AUTHENTIC, handwritten=False, digital=False),
        "D": marked("D", Label.AUTHENTIC, handwritten=True, digital=False),
        "F": marked("F", Label.FORGED, handwritten=True, digital=False),
    }
    probabilities = {"A": 0.5, "B": 0.45, "C": 0.41, "D": 0.1, "F": 0.9}
    assert alarms(probabilities, documents, POLICIES[1]) == {
        "genuine": 4,
        "genuine_with_marks": 3,
        "false_alarms": 3,
        "with_marks": 2,
        "handwritten": 1,
        "digital_only": 1,
    }


def _test_split() -> tuple[dict[str, Assessment], list[Document]]:
    probabilities = (
        {f"C{i}": 0.95 - i / 100 for i in range(5)}
        | {f"M{i}": 0.7 - i / 10 for i in range(4)}
        | {f"G{i}": 0.3 - i / 100 for i in range(6)}
    )
    documents = [
        document(r, Label.FORGED if r[0] in "CM" else Label.AUTHENTIC) for r in probabilities
    ]
    return {r: assessed(r, p) for r, p in probabilities.items()}, documents


def test_the_record_shows_the_rule_s_receipts_with_their_digests() -> None:
    assessments, documents = _test_split()
    results: dict[str, Any] = demo_cards(
        assessments, documents, POLICIES, lambda d: f"sha-{d.doc_id}"
    )
    shown = {name: [c["receipt"] for c in cards] for name, cards in results["gallery"].items()}
    assert shown == {
        CAUGHT: ["C0", "C1", "C3", "C4"],
        MISSED: ["M0", "M2", "M3"],
        GENUINE: ["G0", "G2", "G5"],
    }
    assert results["outcomes"] == {CAUGHT: 5, MISSED: 4, GENUINE: 6, "false alarm": 0}
    assert (results["receipts"], results["forged"]) == (15, 9)
    first = results["gallery"][CAUGHT][0]
    assert (first["rank"], first["sha256"]) == (0, "sha-C0")
    assert first["card"]["decisions"]["10 %"] == Decision.REVIEW.value
    assert results["alarms"]["20 %"]["false_alarms"] == 0
    assert results["policies"][1] == {"budget": 0.2, "threshold": 0.4}


def test_the_record_needs_a_policy_and_every_receipt() -> None:
    assessments, documents = _test_split()
    with pytest.raises(ValueError, match="policy"):
        demo_cards(assessments, documents, (), lambda d: d.doc_id)
    with pytest.raises(ValueError, match="every test receipt"):
        demo_cards(assessments, documents[1:], POLICIES, lambda d: d.doc_id)
