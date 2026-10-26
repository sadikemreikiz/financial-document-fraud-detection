from docfraud.data.models import BBox
from docfraud.detectors.finding import (
    COPY_MOVE,
    LINKING,
    SYSTEM_2,
    Finding,
    metadata_finding,
    region_finding,
    rule_findings,
)
from docfraud.detectors.rules import Outcome, Status
from docfraud.linking.links import (
    Links,
    failing_rules,
    link,
    link_findings,
    suspicious_regions,
)
from docfraud.linking.values import DATE, LocatedValue

DUE = LocatedValue("total_due", BBox(x=10, y=50, width=30, height=10))
CHANGE = LocatedValue("change", BBox(x=10, y=70, width=30, height=10))
DAY = LocatedValue(DATE, BBox(x=10, y=5, width=40, height=10))
VALUES = (DUE, CHANGE, DAY)
FAILING_CHANGE = {"change_given": Outcome(Status.FAIL, "paid 20.00 - due 10.60 is not 9.00")}
FAILING_TAX = {"tax_rates": Outcome(Status.FAIL, "SR 6 % of 10.00 is 0.60, printed 0.70")}


def on(box: BBox) -> Finding:
    """A system 2 finding whose top region overlaps ``box``."""
    return region_finding(SYSTEM_2, 0.9, BBox(x=box.x + 2, y=box.y + 2, width=5, height=5))


def test_a_region_on_a_value_that_a_failing_rule_reads_makes_both_links() -> None:
    findings = (on(DUE.box), *rule_findings(FAILING_CHANGE, VALUES))
    links = link(findings, VALUES)
    assert links.on_value == (DUE,)
    assert links.on_failing_value == (DUE,)
    assert (links.region_on_value, links.failing_value) == (True, True)


def test_a_region_on_a_value_no_failing_rule_reads_makes_link_1_only() -> None:
    links = link((on(DAY.box), *rule_findings(FAILING_CHANGE, VALUES)), VALUES)
    assert (links.region_on_value, links.failing_value) == (True, False)
    links = link((on(DUE.box), *rule_findings(FAILING_TAX, VALUES)), VALUES)
    assert (links.region_on_value, links.failing_value) == (True, False)


def test_only_pixel_findings_supply_suspicious_regions() -> None:
    rules = rule_findings(FAILING_CHANGE, VALUES)  # located at DUE and CHANGE themselves
    extra = metadata_finding(("EXIF",))
    assert extra is not None
    findings = (*rules, extra, region_finding(SYSTEM_2, 0.1, None))
    assert suspicious_regions(findings) == ()
    assert link(findings, VALUES) == Links((), ())
    assert failing_rules(findings) == ("change_given",)


def test_either_region_of_a_copy_move_pair_links() -> None:
    pair = Finding(
        detector=COPY_MOVE,
        kind="duplicated regions",
        score=0.5,
        boxes=(BBox(x=200, y=200, width=5, height=5), BBox(x=12, y=72, width=5, height=5)),
        reason="two identical regions",
    )
    assert link((pair,), VALUES).on_value == (CHANGE,)


def test_links_become_findings_at_their_values() -> None:
    findings = (on(DUE.box), *rule_findings(FAILING_CHANGE, VALUES))
    first, second = link_findings(link(findings, VALUES))
    assert (first.detector, second.detector) == (LINKING, LINKING)
    assert first.boxes == second.boxes == (DUE.box,)
    assert first.reason == "a suspicious region lies on: total_due"
    assert second.kind == "region on a value a failing rule reads"
    assert link_findings(Links((), ())) == ()
