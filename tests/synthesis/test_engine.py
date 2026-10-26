from pathlib import Path

import numpy as np
import pytest

from docfraud.data.models import BBox, Document, Entity, Technique
from docfraud.evaluation.design_fold import DESIGN_FOLD, WHOLE_TRAINING
from docfraud.evaluation.folds import FoldAssignment
from docfraud.synthesis.engine import (
    ENGINE_VERSION,
    Engine,
    Forgery,
    RefusedBaseError,
    Stream,
    directory_fonts,
)
from docfraud.synthesis.layout import Kind, kind
from docfraud.synthesis.statistics import RoundStatistics, Template
from tests.synthesis.pages import documents, font, receipt

ENTITIES = (Entity.PRODUCT, Entity.TOTAL_PAYMENT, Entity.METADATA, Entity.COMPANY, Entity.OTHER)


def statistics(
    technique: Technique, held_out: int = 1, characters: int = 2, regions: int = 2
) -> RoundStatistics:
    """A round whose real regions all used one technique, on every entity."""
    templates = tuple(Template((technique,), entity, characters) for entity in ENTITIES)
    return RoundStatistics(held_out, templates, (regions,), {}, {})


@pytest.fixture
def setup(tmp_path: Path) -> tuple[list[Document], FoldAssignment]:
    return documents(tmp_path)


def engine(
    setup: tuple[list[Document], FoldAssignment], technique: Technique, **options: int
) -> Engine:
    docs, folds = setup
    return Engine(statistics(technique, **options), docs, folds, receipt, font)


def forgeries(made: Engine, count: int = 12) -> list[Forgery]:
    bases = made.pools[Stream.TRAINING]
    return [made.forge(bases[i % len(bases)], i) for i in range(count)]


def test_bases_are_the_authentic_receipts_of_the_training_folds(
    setup: tuple[list[Document], FoldAssignment],
) -> None:
    docs, folds = setup
    made = engine(setup, Technique.CPI)
    training = made.pools[Stream.TRAINING]
    assert training
    assert all(folds.folds[b] != 1 for b in training)
    forged = next(d.doc_id for d in docs if d.label and d.label.value == "forged")
    held_out = next(
        d.doc_id for d in docs if folds.folds[d.doc_id] == 1 and d.doc_id not in training
    )
    for refused in (forged, held_out):
        with pytest.raises(RefusedBaseError):
            made.forge(refused, 0)
    assert made.pools[Stream.DEVELOPMENT] == ()  # only the round that holds out the design fold
    with pytest.raises(RefusedBaseError):
        made.forge(training[0], 0, Stream.DEVELOPMENT)


def test_the_whole_training_round_uses_every_authentic_training_receipt(
    setup: tuple[list[Document], FoldAssignment],
) -> None:
    docs, folds = setup
    made = engine(setup, Technique.CPI, held_out=WHOLE_TRAINING)
    authentic = {
        d.doc_id
        for d in docs
        if d.label and d.label.value == "authentic" and d.doc_id in folds.folds
    }
    assert set(made.pools[Stream.TRAINING]) == authentic
    assert made.pools[Stream.DEVELOPMENT] == ()
    assert made.forge(made.pools[Stream.TRAINING][0], 0).record.held_out == WHOLE_TRAINING


def test_the_development_stream_uses_only_the_held_out_design_fold(
    setup: tuple[list[Document], FoldAssignment],
) -> None:
    _, folds = setup
    made = engine(setup, Technique.CPO, held_out=DESIGN_FOLD)
    development = made.pools[Stream.DEVELOPMENT]
    assert development
    assert all(folds.folds[b] == DESIGN_FOLD for b in development)
    assert not set(development) & set(made.pools[Stream.TRAINING])
    forgery = made.forge(development[0], 0, Stream.DEVELOPMENT)
    donors = {r.source_document for r in forgery.record.regions}
    assert donors
    assert donors <= set(development) - {development[0]}  # no pixel of another fold is copied


@pytest.mark.parametrize("technique", [t for t in Technique if t is not Technique.OTHER])
def test_no_pixel_outside_the_written_area_changes(
    setup: tuple[list[Document], FoldAssignment], technique: Technique
) -> None:
    for forgery in forgeries(engine(setup, technique)):
        base = receipt(forgery.record.base).page
        assert forgery.page.shape == base.shape
        np.testing.assert_array_equal(forgery.changed, forgery.page != base)
        assert not (forgery.changed & ~forgery.written).any()
        assert forgery.record.regions, forgery.record.failures
        for region in forgery.record.regions:
            assert region.technique is technique
            box = region.box
            assert forgery.written[box.y : box.y2, box.x : box.x2].any()
        written = np.zeros_like(forgery.written)
        for region in forgery.record.regions:
            written[region.box.y : region.box.y2, region.box.x : region.box.x2] = True
        assert not (forgery.written & ~written).any()  # the boxes enclose the written area


def test_a_forgery_is_made_again_from_its_seed_and_leaves_the_base_alone(
    setup: tuple[list[Document], FoldAssignment],
) -> None:
    made = engine(setup, Technique.CPI)
    base = made.pools[Stream.TRAINING][0]
    before = receipt(base).page.copy()
    first = made.forge(base, 7)
    made.forge(base, 8)
    again = made.forge(base, 7)
    np.testing.assert_array_equal(first.page, again.page)
    assert first.record == again.record
    np.testing.assert_array_equal(receipt(base).page, before)
    other = made.forge(base, 9)
    assert not np.array_equal(first.page, other.page) or first.record != other.record
    assert first.record.engine_version == ENGINE_VERSION
    assert (first.record.held_out, first.record.stream, first.record.index) == (
        1,
        Stream.TRAINING,
        7,
    )


@pytest.mark.parametrize("technique", [Technique.CPI, Technique.CPO, Technique.IMI])
def test_replaced_characters_keep_their_class_and_the_separators(
    setup: tuple[list[Document], FoldAssignment], technique: Technique
) -> None:
    for forgery in forgeries(engine(setup, technique, characters=5)):
        for region in forgery.record.regions:
            assert region.after is not None
            assert len(region.after) == len(region.before)
            assert region.after != region.before
            for old, new in zip(region.before, region.after, strict=True):
                assert kind(old) is kind(new)
                if kind(old) is Kind.SEPARATOR:
                    assert old == new


def test_a_deletion_leaves_no_ink_where_the_characters_were(
    setup: tuple[list[Document], FoldAssignment],
) -> None:
    for forgery in forgeries(engine(setup, Technique.CUT)):
        words = {w.text: w for w in receipt(forgery.record.base).words}
        for region in forgery.record.regions:
            assert region.after is not None
            assert len(region.after) < len(region.before)
            word = next(
                w for w in words.values() if "".join(c.text for c in w.characters) == region.before
            )
            for character in word.characters:
                box = character.box
                if box.overlaps(region.box) and box.x >= region.box.x and box.x2 <= region.box.x2:
                    covered = forgery.page[box.y : box.y2, box.x : box.x2]
                    assert (covered >= word.threshold).all()


def test_a_copy_inside_the_receipt_comes_from_elsewhere(
    setup: tuple[list[Document], FoldAssignment],
) -> None:
    for forgery in forgeries(engine(setup, Technique.CPI)):
        for region in forgery.record.regions:
            assert len(region.sources) == sum(
                a != b for a, b in zip(region.before, region.after or "", strict=True)
            )
            assert region.source_document is None
            assert not any(source.overlaps(region.box) for source in region.sources)
            assert "source row" in region.variants


def test_typed_text_has_the_height_of_the_receipts_text(
    setup: tuple[list[Document], FoldAssignment],
) -> None:
    for forgery in forgeries(engine(setup, Technique.IMI, characters=1)):
        for region in forgery.record.regions:
            word = next(
                w
                for w in receipt(forgery.record.base).words
                if "".join(c.text for c in w.characters) == region.before
                and w.box.overlaps(region.box)
            )
            heights = [c.box.height for c in word.characters if c.kind is not Kind.SEPARATOR]
            assert region.box.height <= max(heights) * 1.6 + 8  # the cover may reach a little


def test_free_hand_strokes_stay_inside_their_characters(
    setup: tuple[list[Document], FoldAssignment],
) -> None:
    for forgery in forgeries(engine(setup, Technique.PIX)):
        boxes = [c.box for w in receipt(forgery.record.base).words for c in w.characters]
        for region in forgery.record.regions:
            assert region.after is None
            ys, xs = np.nonzero(
                forgery.written[region.box.y : region.box.y2, region.box.x : region.box.x2]
            )
            for y, x in zip(ys + region.box.y, xs + region.box.x, strict=True):
                assert any(b.x <= x < b.x2 and b.y <= y < b.y2 for b in boxes)


def test_a_free_hand_stroke_that_covers_no_pixel_is_not_placed(
    setup: tuple[list[Document], FoldAssignment], monkeypatch: pytest.MonkeyPatch
) -> None:
    # a stroke whose ends almost meet covers no pixel enough for a hard pen; it once crashed
    monkeypatch.setattr(
        "docfraud.synthesis.engine._pen",
        lambda box, *_arguments, **_options: np.zeros((box.height, box.width)),
    )
    made = engine(setup, Technique.PIX)
    forgery = made.forge(made.pools[Stream.TRAINING][0], 0)
    assert forgery.record.regions == ()
    assert forgery.record.failures["wrote no pixel"] > 0
    assert not forgery.written.any()
    assert not forgery.changed.any()


def test_regions_do_not_overlap(setup: tuple[list[Document], FoldAssignment]) -> None:
    for forgery in forgeries(engine(setup, Technique.CPI, regions=4)):
        boxes = [r.box for r in forgery.record.regions]
        for i, first in enumerate(boxes):
            for second in boxes[i + 1 :]:
                assert not first.overlaps(second)


def test_templates_that_cannot_be_placed_are_counted(
    setup: tuple[list[Document], FoldAssignment],
) -> None:
    docs, folds = setup
    nothing = RoundStatistics(1, (Template((Technique.CPI,), Entity.COMPANY, 1),), (1,), {}, {})
    made = Engine(
        nothing,
        docs,
        folds,
        lambda doc_id: receipt(doc_id).__class__(doc_id, receipt(doc_id).page, ()),
        font,
    )
    forgery = made.forge(made.pools[Stream.TRAINING][0], 0)
    assert forgery.record.regions == ()
    assert forgery.record.failures == {"no word of the entity": 20, "region given up": 1}
    assert not forgery.changed.any()


def test_fonts_are_read_from_their_directory(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="download-fonts"):
        directory_fonts(tmp_path)("Lato", False, 20)


def test_boxes_are_unchanged_by_a_record_round_trip(
    setup: tuple[list[Document], FoldAssignment],
) -> None:
    forgery = forgeries(engine(setup, Technique.CPI), 1)[0]
    record = forgery.record
    assert type(record).model_validate_json(record.model_dump_json()) == record
    assert all(isinstance(r.box, BBox) for r in record.regions)
