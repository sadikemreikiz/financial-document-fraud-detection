import json

import numpy as np
import pytest

from docfraud.data.models import BBox, Label
from docfraud.decision import Decision, ReviewPolicy
from docfraud.fusion.errors import (
    ELSEWHERE,
    NOTHING_FOUND,
    NOTHING_PUSHES,
    TOO_WEAK,
    error,
    error_analysis,
    false_alarm_cause,
    frozen_assessments,
    miss_cause,
)
from docfraud.fusion.validation import validation_check
from docfraud.reporting.assessment import Assessment, Evidence
from tests.fusion.test_validation import SEEDS, pages, tables
from tests.llm.test_agent_rounds import ELSEWHERE as AWAY
from tests.llm.test_agent_rounds import document
from tests.llm.test_agent_tools import BOX, assessment

POLICY = ReviewPolicy(0.1, 0.8416)


def decided(found: Assessment, decision: Decision, *evidence: Evidence) -> Assessment:
    return found.model_copy(update={"decision": decision, "evidence": evidence or found.evidence})


def pushing(inputs: tuple[str, ...], contribution: float) -> Evidence:
    """Evidence of these inputs at :data:`BOX`, with this contribution."""
    finding = assessment().evidence[0].findings[0]
    return Evidence(inputs=inputs, contribution=contribution, findings=(finding,))


def test_a_miss_is_explained_by_the_first_rule_that_applies() -> None:
    accepted = decided(assessment(), Decision.ACCEPT)
    on_the_edit = document("R1", Label.FORGED, BOX)
    assert miss_cause(accepted, on_the_edit) == TOO_WEAK
    assert miss_cause(accepted, document("R1", Label.FORGED, AWAY)) == ELSEWHERE
    assert miss_cause(accepted, document("R1", Label.FORGED)) == ELSEWHERE  # nothing to judge by
    against = decided(accepted, Decision.ACCEPT, pushing(("rules",), -0.4))
    assert miss_cause(against, on_the_edit) == NOTHING_FOUND


def test_a_false_alarm_is_explained_by_the_group_pushing_most() -> None:
    sent = decided(
        assessment(),
        Decision.REVIEW,
        pushing(("link 1",), 0.3),
        pushing(("system 2",), 2.1),
        pushing(("rules",), 1.0),
    )
    assert false_alarm_cause(sent) == "pixel evidence"
    assert false_alarm_cause(decided(sent, Decision.REVIEW, pushing(("rules",), -1.0))) == (
        NOTHING_PUSHES
    )


def test_only_wrong_decisions_are_errors_and_each_shows_what_the_data_say() -> None:
    reviewed, accepted = (
        decided(assessment(), Decision.REVIEW),
        decided(assessment(), Decision.ACCEPT),
    )
    assert error(reviewed, document("R1", Label.FORGED, BOX)) is None
    assert error(accepted, document("R1", Label.AUTHENTIC)) is None
    miss = error(accepted, document("R1", Label.FORGED, BOX))
    assert miss is not None
    assert (miss["kind"], miss["cause"], miss["annotated"]) == ("miss", TOO_WEAK, True)
    assert miss["largest_contributions"][0]["group"] == "pixel evidence"
    marked = document("R2", Label.AUTHENTIC).model_copy(update={"handwritten_annotation": True})
    alarm = error(reviewed, marked)
    assert alarm is not None
    assert (alarm["kind"], alarm["cause"], alarm["handwritten_annotation"]) == (
        "false alarm",
        "pixel evidence",
        True,
    )


def test_the_analysis_counts_every_error_by_cause() -> None:
    documents = [
        document("R1", Label.FORGED, BOX),
        document("R2", Label.FORGED, BBox(x=60, y=40, width=10, height=10)),
        document("R3", Label.AUTHENTIC),
        document("R4", Label.AUTHENTIC),
    ]
    accepted, reviewed = (
        decided(assessment(), Decision.ACCEPT),
        decided(assessment(), Decision.REVIEW),
    )
    assessed = {"R1": accepted, "R2": accepted, "R3": reviewed, "R4": accepted}
    found = error_analysis(documents, assessed, POLICY)
    assert (found["receipts"], found["forged"], found["reviewed"]) == (4, 2, 1)
    assert found["misses"]["by_cause"] == {TOO_WEAK: 1, ELSEWHERE: 1}
    assert found["false_alarms"] == {
        "count": 1,
        "by_cause": {"pixel evidence": 1},
        "with_benign_marks": 0,
    }
    assert [e["receipt"] for e in found["errors"]] == ["R1", "R2", "R3"]
    json.dumps(found)  # a run record can keep it


def test_every_receipt_is_assessed_with_the_checks_record_applied() -> None:
    training, y_train, validation, documents = tables()
    check = validation_check(
        training, y_train, validation, documents, pages(documents), seeds=SEEDS, resamples=20
    )
    rows = (training.rows("ocr", 0), y_train, validation.rows("ocr", 0))
    policy, assessed = frozen_assessments(*rows, check)
    assert policy.threshold == check["policy"]["ocr"]["10 %"]["threshold"]
    assert list(assessed) == [d.doc_id for d in documents]
    reviewed = sum(a.decision is Decision.REVIEW for a in assessed.values())
    assert reviewed == check["policy"]["ocr"]["10 %"]["reviewed"]
    changed = json.loads(json.dumps(check))
    changed["ocr"]["fitted"]["fusion"]["intercept"] += 0.1
    with pytest.raises(ValueError, match="not the recorded one"):
        frozen_assessments(*rows, changed)
    assert np.isfinite([a.probability for a in assessed.values()]).all()
