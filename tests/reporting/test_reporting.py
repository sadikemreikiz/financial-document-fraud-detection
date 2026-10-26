import re
from pathlib import Path

import numpy as np
import pytest

from docfraud.data.models import BBox, Document, Label, Region, RegionRole, Split
from docfraud.decision import Decision, ReviewPolicy
from docfraud.detectors.copy_move import CopyMoveEvidence, CopyMovePair
from docfraud.detectors.finding import (
    SYSTEM_1,
    SYSTEM_2,
    Finding,
    copy_move_finding,
    region_finding,
    rule_findings,
)
from docfraud.detectors.rules import Outcome, Status
from docfraud.fusion.inputs import FUSION, fusion_inputs
from docfraud.fusion.model import FusionModel, Platt
from docfraud.linking.links import link, link_findings
from docfraud.linking.values import LocatedValue
from docfraud.reporting.assessment import Assessment, assess
from docfraud.reporting.grounding import _values, check_grounding
from docfraud.reporting.localisation import first_region_hit
from docfraud.reporting.overlay import COLOURS, overlay
from docfraud.reporting.text import box_text, render_report

PAGE = BBox(x=0, y=0, width=200, height=120)
WORDS = [BBox(x=10, y=10 + 20 * k, width=180, height=14) for k in range(5)]
DUE = LocatedValue("total_due", BBox(x=120, y=50, width=40, height=12))
CHANGE = LocatedValue("change", BBox(x=120, y=70, width=40, height=12))
REGION = BBox(x=125, y=52, width=10, height=8)  # system 2's top region, on the amount due


def fitted() -> tuple[FusionModel, Platt, ReviewPolicy]:
    rng = np.random.default_rng(0)
    n = 300
    x = rng.random((n, len(FUSION)))
    x[:, 3] = x[:, 3] > 0.5  # system 1 evidence and the links are flags
    x[:, 5:] = x[:, 5:] > 0.7
    logit = 3 * x[:, 1] + 2 * x[:, 4] + 1.5 * x[:, 6] - 3.5
    y = rng.random(n) < 1 / (1 + np.exp(-logit))
    model = FusionModel.fit(x, y, FUSION)
    log_odds = model.log_odds(x)
    platt = Platt.fit(log_odds, y)
    policy = ReviewPolicy.set_on(platt.probabilities(log_odds))
    return model, platt, policy


def receipt_findings() -> tuple[Finding, ...]:
    pair = CopyMovePair(REGION, BBox(x=30, y=12, width=10, height=8), blocks=4, area=80)
    copy = copy_move_finding(CopyMoveEvidence(text_height=14.0, block=4, pair=pair))
    assert copy is not None
    detected = (
        region_finding(SYSTEM_2, 0.95, REGION),
        region_finding(SYSTEM_1, 0.0, None),
        copy,
        *rule_findings(
            {"change_given": Outcome(Status.FAIL, "paid 20.00 - due 10.60 does not give 9.00")},
            [DUE, CHANGE],
        ),
    )
    return (*detected, *link_findings(link(detected, [DUE, CHANGE])))


def assessment() -> Assessment:
    model, platt, policy = fitted()
    findings = receipt_findings()
    inputs = fusion_inputs(0.4, findings, link(findings, [DUE, CHANGE]))
    return assess("X1", "T2 OCR", inputs, findings, model=model, platt=platt, policy=policy)


def test_evidence_is_grouped_by_input_and_ordered_by_contribution() -> None:
    model, platt, policy = fitted()
    found = assessment()
    contributions = [e.contribution for e in found.evidence]
    assert contributions == sorted(contributions, reverse=True)
    assert [e.inputs for e in found.evidence if len(e.inputs) > 1] == [
        ("system 1", "system 1 evidence")
    ]
    inputs = fusion_inputs(0.4, receipt_findings(), link(receipt_findings(), [DUE, CHANGE]))
    x = np.array([[inputs[name] for name in FUSION]])
    assert sum(contributions) + model.intercept == pytest.approx(model.log_odds(x)[0])
    assert found.probability == pytest.approx(platt.probabilities(model.log_odds(x))[0])
    assert found.decision is policy.decide(found.probability)
    assert all(e.findings[0].kind == "input value" for e in found.evidence)


def test_the_first_region_is_that_of_the_strongest_located_push_towards_forgery() -> None:
    found = assessment()
    first = next(e for e in found.evidence if e.contribution > 0 and e.boxes)
    assert found.first_region() == first.boxes
    against = tuple(e.model_copy(update={"contribution": -1.0}) for e in found.evidence)
    quiet = found.model_copy(update={"evidence": against})
    assert quiet.first_region() == ()


def test_assessing_needs_the_models_inputs() -> None:
    model, platt, policy = fitted()
    with pytest.raises(ValueError, match="not the model's"):
        assess("X1", "T2", {"rules": 0.1}, (), model=model, platt=platt, policy=policy)


def test_the_report_states_the_decision_and_every_piece_of_evidence() -> None:
    found = assessment()
    text = render_report(found)
    assert text.startswith("# Audit report: X1")
    assert f"Probability of forgery: {found.probability:.3f}" in text
    assert "Rejecting it is the reviewer's decision." in text
    assert "## Evidence towards forgery, strongest first" in text
    assert box_text(REGION) in text
    assert "paid 20.00 - due 10.60 does not give 9.00" in text
    assert text.count("on the log-odds") == len(found.evidence)
    against = [e.contribution for e in found.evidence if e.contribution <= 0]
    listed = [float(x) for x in re.findall(r"\(([-+]\d+\.\d+) on the log-odds\)", text)]
    assert listed[-len(against) :] == sorted(round(c, 3) for c in against)


def test_a_report_from_the_template_is_fully_grounded_and_an_invented_fact_is_not() -> None:
    found = assessment()
    text = render_report(found)
    grounding = check_grounding(text, found)
    assert grounding.facts > 20
    assert grounding.violations == ()
    assert grounding.rate == 1.0
    invented = text + "\n- the change should have been 13.37 at (1, 2, 3 x 4 px)\n"
    bad = check_grounding(invented, found)
    assert set(bad.violations) == {"13.37", "(1, 2, 3 x 4 px)"}
    assert bad.rate == pytest.approx(1 - 2 / bad.facts)
    assert check_grounding("nothing", found).rate == 1.0


def test_a_report_without_evidence_has_no_evidence_section() -> None:
    text = render_report(assessment().model_copy(update={"evidence": ()}))
    assert "## Evidence" not in text
    assert list(_values(None)) == []
    assert list(_values(True)) == []


def test_the_overlay_draws_the_boxes_of_evidence_towards_forgery() -> None:
    found = assessment()
    page = np.full((PAGE.height, PAGE.width), 255, dtype=np.uint8)
    image = np.asarray(overlay(page, found))
    assert image.shape == (PAGE.height, PAGE.width, 3)
    drawn = {tuple(map(int, image[REGION.y, x])) for x in range(REGION.x, REGION.x2)}
    assert drawn & set(COLOURS.values())


def forged(region: BBox) -> Document:
    return Document(
        dataset="toy",
        doc_id="X1",
        source_id="X1",
        split=Split.TRAIN,
        label=Label.FORGED,
        image_path=Path("x.png"),
        regions=(Region(box=region, role=RegionRole.MODIFIED),),
        has_region_annotations=True,
    )


def test_the_first_region_is_judged_against_the_annotations_and_chance() -> None:
    found = assessment()
    first = found.first_region()
    hit = first_region_hit(found, forged(first[0]), PAGE, WORDS)
    assert (hit.located, hit.hit) == (True, True)
    assert 0 < hit.baseline < 1
    miss = first_region_hit(found, forged(BBox(x=0, y=110, width=5, height=5)), PAGE, WORDS)
    assert (miss.located, miss.hit) == (True, False)
    quiet = found.model_copy(update={"evidence": ()})
    nothing = first_region_hit(quiet, forged(first[0]), PAGE, WORDS)
    assert (nothing.located, nothing.hit, nothing.baseline) == (False, False, 0.0)
    assert found.decision in set(Decision)
