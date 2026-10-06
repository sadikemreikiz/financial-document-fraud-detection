from pathlib import Path

import pytest
from pydantic import ValidationError

from docfraud.data.models import BBox, Document, Label, Region, RegionRole, Split


def box(**kwargs: int) -> BBox:
    return BBox(**({"x": 2, "y": 3, "width": 4, "height": 5} | kwargs))


def test_bbox_edges_and_area() -> None:
    b = box()
    assert (b.x2, b.y2, b.area) == (6, 8, 20)
    assert b.fits_within(6, 8)
    assert not b.fits_within(5, 8)


@pytest.mark.parametrize("bad", [{"x": -1}, {"width": 0}, {"height": -2}])
def test_bbox_rejects_invalid_geometry(bad: dict[str, int]) -> None:
    with pytest.raises(ValidationError):
        box(**bad)


def document(**kwargs: object) -> Document:
    base: dict[str, object] = {
        "dataset": "d",
        "doc_id": "1",
        "source_id": "s/1",
        "image_path": Path("x.png"),
    }
    return Document.model_validate(base | kwargs)


def test_document_uid_and_modified_regions() -> None:
    modified = Region(box=box(), role=RegionRole.MODIFIED)
    source = Region(box=box(x=9), role=RegionRole.SOURCE)
    doc = document(
        label=Label.FORGED,
        split=Split.TRAIN,
        regions=(modified, source),
        has_region_annotations=True,
    )
    assert doc.uid == "d/1"
    assert doc.modified_regions == (modified,)


def test_document_is_immutable() -> None:
    doc = document()
    with pytest.raises(ValidationError):
        doc.doc_id = "2"  # type: ignore[misc]


def test_regions_require_the_annotation_flag() -> None:
    with pytest.raises(ValidationError, match="has_region_annotations"):
        document(regions=(Region(box=box(), role=RegionRole.SOURCE),))


def test_authentic_documents_cannot_have_modified_regions() -> None:
    with pytest.raises(ValidationError, match="authentic"):
        document(
            label=Label.AUTHENTIC,
            regions=(Region(box=box(), role=RegionRole.MODIFIED),),
            has_region_annotations=True,
        )
