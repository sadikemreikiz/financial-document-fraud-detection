import inspect

import pytest

from docfraud.data.models import BBox
from docfraud.detectors.copy_move import CopyMoveEvidence, CopyMovePair
from docfraud.detectors.finding import (
    SYSTEM_1,
    SYSTEM_2,
    Finding,
    copy_move_finding,
    metadata_finding,
    region_finding,
)
from docfraud.fusion.inputs import (
    BASELINE,
    FUSION,
    LINK_1,
    LINK_2,
    METADATA_INPUT,
    PRACTICAL,
    SINGLE_SOURCES,
    fusion_inputs,
)
from docfraud.linking.links import Links
from docfraud.linking.values import LocatedValue

REGION = BBox(x=1, y=2, width=3, height=4)
VALUE = LocatedValue("total_due", REGION)


def findings(*extra: Finding) -> tuple[Finding, ...]:
    return (
        region_finding(SYSTEM_2, 0.7, REGION),
        region_finding(SYSTEM_1, 0.0, None),
        *extra,
    )


def test_the_input_sets_nest_and_name_every_single_source() -> None:
    assert FUSION[: len(BASELINE)] == BASELINE
    assert (*FUSION, METADATA_INPUT) == PRACTICAL
    singles = {name for inputs in SINGLE_SOURCES.values() for name in inputs}
    assert set(FUSION) - {LINK_1, LINK_2} == singles


def test_inputs_come_from_the_rules_score_findings_and_links() -> None:
    pair = CopyMovePair(REGION, BBox(x=40, y=2, width=3, height=4), blocks=1, area=12)
    copy = copy_move_finding(CopyMoveEvidence(text_height=4.0, block=4, pair=pair))
    assert copy is not None
    inputs = fusion_inputs(0.3, findings(copy), Links((VALUE,), ()))
    assert list(inputs) == list(FUSION)
    assert inputs == {
        "rules": 0.3,
        "system 2": 0.7,
        "system 1": 0.0,
        "system 1 evidence": 0.0,
        "copy-move": 0.75,
        "link 1": 1.0,
        "link 2": 0.0,
    }
    assert fusion_inputs(0.3, findings(), Links((), ()))["copy-move"] == 0.0


def test_only_track_t3_sees_the_file_metadata() -> None:
    traits = metadata_finding(("EXIF",))
    assert traits is not None
    with_traits = findings(traits)
    assert METADATA_INPUT not in fusion_inputs(0.1, with_traits, Links((), ()))
    assert fusion_inputs(0.1, with_traits, Links((), ())) == fusion_inputs(
        0.1, findings(), Links((), ())
    )
    practical = fusion_inputs(0.1, with_traits, Links((), ()), practical=True)
    assert list(practical) == list(PRACTICAL)
    assert practical[METADATA_INPUT] == 1.0
    assert fusion_inputs(0.1, findings(), Links((), ()), practical=True)[METADATA_INPUT] == 0.0


def test_inputs_cannot_take_a_document() -> None:
    assert list(inspect.signature(fusion_inputs).parameters) == [
        "rules_score",
        "findings",
        "links",
        "practical",
    ]


def test_every_receipt_needs_one_finding_of_each_pixel_system() -> None:
    with pytest.raises(ValueError, match="every receipt"):
        fusion_inputs(0.1, (region_finding(SYSTEM_2, 0.7, REGION),), Links((), ()))
    with pytest.raises(ValueError, match="one system 2 finding"):
        fusion_inputs(0.1, findings(region_finding(SYSTEM_2, 0.1, None)), Links((), ()))
