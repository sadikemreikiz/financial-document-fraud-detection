import numpy as np
import pytest
from pydantic import ValidationError

from docfraud.data.models import BBox
from docfraud.detectors.copy_move import CopyMoveEvidence, CopyMovePair
from docfraud.detectors.finding import (
    COPY_MOVE,
    METADATA,
    RULES,
    SYSTEM_1,
    SYSTEM_2,
    Finding,
    copy_move_finding,
    lattice_finding,
    metadata_finding,
    region_finding,
    rule_findings,
)
from docfraud.detectors.rules import Outcome, Status
from docfraud.forensics.lattice_map import LatticeEvidence
from docfraud.forensics.quantisation import Lattice
from docfraud.linking.values import ITEM_AMOUNT, LocatedValue

DUE = LocatedValue("total_due", BBox(x=10, y=50, width=30, height=10))
CHANGE = LocatedValue("change", BBox(x=10, y=70, width=30, height=10))
ITEM = LocatedValue(ITEM_AMOUNT, BBox(x=10, y=20, width=30, height=10))


def test_a_failing_rule_is_located_at_the_values_it_reads() -> None:
    outcomes = {
        "change_given": Outcome(
            Status.FAIL, "paid 20.00 - due 10.60 does not give the change 9.00"
        ),
        "line_amounts": Outcome(Status.PASS, "2 lines multiply out"),
        "tax_rates": Outcome.na("no tax summary row"),
    }
    (finding,) = rule_findings(outcomes, [DUE, CHANGE, ITEM])
    assert finding.detector == RULES
    assert finding.boxes == (DUE.box, CHANGE.box)
    assert finding.values == ("change", "total_due")
    assert finding.reason.startswith("change_given: paid 20.00")
    assert finding.details == {"rule": "change_given"}


def test_a_pixel_system_reports_its_score_and_top_region() -> None:
    region = BBox(x=3, y=4, width=5, height=6)
    found = region_finding(SYSTEM_2, 0.8125, region)
    assert found.boxes == (region,)
    assert "0.812" in found.reason
    assert "(3, 4, 5 x 6 px)" in found.reason
    assert region_finding(SYSTEM_2, 0.1, None).boxes == ()


def evidence(suspicion: np.ndarray, measured: np.ndarray) -> LatticeEvidence:
    lattice = Lattice(offset=(0, 0), score=1.0, runner_up=0.0, steps=np.ones((8, 8), dtype=int))
    return LatticeEvidence(lattice, suspicion, measured, (16, 24))


def test_system_1_has_a_region_only_where_it_has_evidence() -> None:
    suspicion = np.zeros((2, 3))
    suspicion[1, 2] = 0.9
    found = lattice_finding(evidence(suspicion, np.ones((2, 3), dtype=bool)))
    assert found.detector == SYSTEM_1
    assert found.boxes == (BBox(x=16, y=8, width=8, height=8),)
    assert found.details == {"evidence": True}
    none = lattice_finding(evidence(np.zeros((2, 3)), np.zeros((2, 3), dtype=bool)))
    assert none.boxes == ()
    assert none.score == 0.0
    assert none.details == {"evidence": False}


def test_copy_move_reports_its_pair_or_nothing() -> None:
    first, second = BBox(x=1, y=1, width=4, height=4), BBox(x=20, y=30, width=4, height=4)
    pair = CopyMovePair(first, second, blocks=1, area=16)
    found = copy_move_finding(CopyMoveEvidence(text_height=8.0, block=4, pair=pair))
    assert found is not None
    assert found.detector == COPY_MOVE
    assert found.boxes == (first, second)
    assert found.score == pytest.approx(0.25)
    assert "16 px" in found.reason
    assert copy_move_finding(CopyMoveEvidence(text_height=8.0, block=4, pair=None)) is None


def test_file_metadata_has_no_location() -> None:
    found = metadata_finding(("alpha channel", "EXIF"))
    assert found is not None
    assert found.detector == METADATA
    assert found.boxes == ()
    assert "alpha channel, EXIF" in found.reason
    assert metadata_finding(()) is None


def test_findings_are_immutable_and_refuse_unknown_fields() -> None:
    found = region_finding(SYSTEM_2, 0.5, None)
    with pytest.raises(ValidationError):
        found.score = 1.0  # type: ignore[misc]
    with pytest.raises(ValidationError):
        Finding(detector="x", kind="y", score=0.0, reason="z", label="forged")  # type: ignore[call-arg]
