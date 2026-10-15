from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from docfraud.data.models import BBox, Document, Entity, Region, RegionRole, Technique
from docfraud.reading.ocr import OcrResult
from docfraud.synthesis.statistics import (
    character_width,
    round_statistics,
    training_documents,
)
from tests.synthesis.pages import documents, draw_receipt


class FakeOcr:
    """Reads every page as the drawn receipt."""

    identifier = "fake"

    def recognize(self, _image: npt.NDArray[np.uint8]) -> OcrResult:
        return draw_receipt()[1]


def _with_regions(document: Document, *regions: Region) -> Document:
    return document.model_copy(update={"regions": regions})


def test_a_round_reads_only_its_training_folds(tmp_path: Path) -> None:
    docs, folds = documents(tmp_path)
    assert {folds.folds[d.doc_id] for d in training_documents(docs, folds, 1)} == {0, 2, 3, 4}
    with pytest.raises(ValueError, match="no fold"):
        training_documents(docs, folds, 5)
    first = round_statistics(docs, folds, 1, FakeOcr())
    changed = [
        _with_regions(
            d,
            Region(
                box=BBox(x=0, y=0, width=99, height=9),
                role=RegionRole.MODIFIED,
                techniques=frozenset({Technique.IMI}),
                entity=Entity.COMPANY,
            ),
        )
        if folds.folds[d.doc_id] == 1
        else d
        for d in docs
    ]
    assert round_statistics(changed, folds, 1, FakeOcr()) == first  # the held-out fold is unread
    assert len(first.templates) == 4  # one forged receipt with one region per training fold
    assert first.regions_per_receipt == (1, 1, 1, 1)


def test_templates_measure_width_in_characters_and_leave_out_other(tmp_path: Path) -> None:
    docs, folds = documents(tmp_path, folds=2, per_fold=1)
    forged = docs[0]
    amount = next(t.box for t in draw_receipt()[1].tokens if t.text == "48.75")
    pitch = amount.width / 5
    regions = (
        Region(  # two characters of "48.75"
            box=BBox(x=amount.x, y=amount.y, width=round(2 * pitch), height=amount.height),
            role=RegionRole.MODIFIED,
            techniques=frozenset({Technique.CPI, Technique.OTHER}),
            entity=Entity.PRODUCT,
        ),
        Region(  # no word under it: the width is measured on the row
            box=BBox(x=amount.x2 + 40, y=amount.y, width=round(3 * pitch), height=amount.height),
            role=RegionRole.MODIFIED,
            techniques=frozenset({Technique.CUT}),
            entity=Entity.PRODUCT,
        ),
        Region(
            box=BBox(x=10, y=10, width=5, height=5),
            role=RegionRole.MODIFIED,
            techniques=frozenset({Technique.OTHER}),
            entity=Entity.OTHER,
        ),
        Region(box=BBox(x=10, y=10, width=5, height=5), role=RegionRole.MODIFIED),
        Region(
            box=BBox(x=10, y=10, width=5, height=5),
            role=RegionRole.MODIFIED,
            techniques=frozenset({Technique.PIX}),
        ),
        Region(box=BBox(x=1, y=1, width=5, height=5), role=RegionRole.SOURCE),
    )
    docs = [_with_regions(forged, *regions), *docs[1:]]
    statistics = round_statistics(docs, folds, 1, FakeOcr())
    assert [t.techniques for t in statistics.templates] == [(Technique.CPI,), (Technique.CUT,)]
    assert statistics.templates[0].characters == 2
    assert statistics.skipped == {"technique other only": 1, "no technique": 1, "no entity": 1}
    assert statistics.widths_measured_on == {"words": 1, "row": 1}
    assert statistics.regions_per_receipt == (5,)


def test_character_width_falls_back_to_the_row_and_the_receipt() -> None:
    _, ocr = draw_receipt()
    amount = next(t.box for t in ocr.tokens if t.text == "48.75")
    on_word = character_width(ocr, BBox(x=amount.x, y=amount.y, width=5, height=10))
    beside = character_width(ocr, BBox(x=amount.x2 + 40, y=amount.y, width=5, height=10))
    nowhere = character_width(ocr, BBox(x=400, y=2, width=5, height=2))
    assert on_word is not None
    assert on_word[1] == "words"
    assert beside is not None
    assert beside[1] == "row"
    assert nowhere is not None
    assert nowhere[1] == "receipt"
    empty = OcrResult(engine="t", width=10, height=10, tokens=())
    assert character_width(empty, BBox(x=0, y=0, width=5, height=5)) is None


def test_forged_receipts_without_regions_are_counted(tmp_path: Path) -> None:
    docs, folds = documents(tmp_path, folds=2, per_fold=1)
    docs = [_with_regions(docs[0]), *docs[1:]]
    statistics = round_statistics(docs, folds, 1, FakeOcr())
    assert statistics.skipped == {"forged receipt without regions": 1}
    assert statistics.templates == ()
