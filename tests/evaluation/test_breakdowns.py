from pathlib import Path

import pytest

from docfraud.data.models import BBox, Document, Entity, Label, Region, RegionRole, Technique
from docfraud.evaluation.breakdowns import (
    breakdown,
    flagged_by_budget,
    flagged_by_threshold,
    rules_blind,
)


def doc(doc_id: str, label: Label, **extra: object) -> Document:
    return Document.model_validate(
        {"dataset": "t", "doc_id": doc_id, "source_id": doc_id, "label": label,
         "image_path": Path(f"{doc_id}.png"), **extra}
    )  # fmt: skip


def region(technique: Technique, entity: Entity) -> Region:
    return Region(
        box=BBox(x=0, y=0, width=1, height=1),
        role=RegionRole.MODIFIED,
        techniques=frozenset({technique}),
        entity=entity,
    )


def test_forged_receipts_count_in_every_group_they_belong_to() -> None:
    forged = doc(
        "F",
        Label.FORGED,
        regions=(
            region(Technique.CPI, Entity.TOTAL_PAYMENT),
            region(Technique.CUT, Entity.METADATA),
        ),
        has_region_annotations=True,
        editing_software="gimp",
    )
    missed = doc(
        "G",
        Label.FORGED,
        regions=(region(Technique.CPI, Entity.PRODUCT),),
        has_region_annotations=True,
    )
    result = breakdown([forged, missed], {"F": 1.0, "G": 0.0})
    assert result["forged"]["technique"]["cpi"] == {"receipts": 2, "found": 1.0}
    assert result["forged"]["technique"]["cut"] == {"receipts": 1, "found": 1.0}
    assert result["forged"]["entity"]["product"] == {"receipts": 1, "found": 0.0}
    assert result["forged"]["software"] == {
        "gimp": {"receipts": 1, "found": 1.0},
        "not recorded": {"receipts": 1, "found": 0.0},
    }


def test_authentic_receipts_are_grouped_by_benign_annotation() -> None:
    docs = [
        doc("A", Label.AUTHENTIC, handwritten_annotation=True),
        doc("B", Label.AUTHENTIC, digital_annotation=True),
        doc("C", Label.AUTHENTIC, handwritten_annotation=True, digital_annotation=True),
        doc("D", Label.AUTHENTIC),
    ]
    result = breakdown(docs, {"A": 1.0, "B": 0.0, "C": 0.5, "D": 0.0})
    assert result["authentic"] == {
        "digital": {"receipts": 1, "flagged": 0.0},
        "handwritten": {"receipts": 1, "flagged": 1.0},
        "handwritten and digital": {"receipts": 1, "flagged": 0.5},
        "none": {"receipts": 1, "flagged": 0.0},
    }


def test_unannotated_forgeries_have_their_own_group() -> None:
    result = breakdown([doc("F", Label.FORGED)], {"F": 1.0})
    assert result["forged"]["technique"] == {"not annotated": {"receipts": 1, "found": 1.0}}


def test_every_document_needs_a_decision() -> None:
    with pytest.raises(ValueError, match="missing decisions: \\['A'\\]"):
        breakdown([doc("A", Label.AUTHENTIC)], {})


def test_decisions_by_threshold_and_by_review_budget() -> None:
    scores = {"a": 0.9, "b": 0.5, "c": 0.5, "d": 0.1}
    assert flagged_by_threshold(scores, 0.5) == {"a": 1.0, "b": 1.0, "c": 1.0, "d": 0.0}
    # half of four receipts: "a", then one of the two tied at 0.5
    assert flagged_by_budget(scores, 0.5) == {"a": 1.0, "b": 0.5, "c": 0.5, "d": 0.0}


def test_only_changes_that_avoid_amounts_and_products_are_blind_to_the_rules() -> None:
    def forged(*entities: Entity) -> Document:
        regions = tuple(region(Technique.CPI, e) for e in entities)
        return doc("F", Label.FORGED, regions=regions, has_region_annotations=True)

    assert rules_blind(forged(Entity.METADATA))
    assert rules_blind(forged(Entity.COMPANY, Entity.OTHER))
    assert not rules_blind(forged(Entity.METADATA, Entity.TOTAL_PAYMENT))
    assert not rules_blind(forged(Entity.PRODUCT))
    assert not rules_blind(doc("A", Label.AUTHENTIC))  # nothing annotated changed
