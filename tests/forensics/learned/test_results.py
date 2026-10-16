import math
from pathlib import Path

import numpy as np
import numpy.typing as npt
import pytest

from docfraud.data.models import (
    BBox,
    Document,
    Entity,
    Label,
    Region,
    RegionRole,
    Split,
    Technique,
)
from docfraud.evaluation.localisation import (
    choose_pixel_threshold,
    edit_average_precisions,
    f1_at_thresholds,
    pixel_average_precision,
    pixel_f1_iou,
    positive_share,
    threshold_candidates,
    threshold_sample,
    top_region,
)
from docfraud.evaluation.runs import CodeVersion, Environment
from docfraud.forensics import trufor
from docfraud.forensics.learned.inference import HIGHEST, QUARTER_SQUARE, LearnedOutput
from docfraud.forensics.learned.results import (
    Edit,
    Group,
    HeldMap,
    Item,
    MeasuredReceipt,
    RunResults,
    ScoredReceipt,
    choose_threshold,
    measure_receipt,
    measure_run,
    read_results,
    real_item,
    synthetic_item,
    write_results,
)
from docfraud.forensics.learned.text_scale import (
    TEXT_HEIGHT_OTHER,
    WordScale,
    text_height,
    word_peaks,
)
from docfraud.synthesis.engine import (
    ENGINE_VERSION,
    Forgery,
    ForgeryRecord,
    Stream,
    SyntheticRegion,
)

CODE = CodeVersion(commit="abc", dirty=False)
ENVIRONMENT = Environment(
    python="3.14",
    docfraud="test",
    packages={},
    platform="test",
    machine="test",
    processor="test",
    cpu_count=1,
    memory_bytes=None,
)


def box(x: int, y: int, width: int, height: int) -> BBox:
    return BBox(x=x, y=y, width=width, height=height)


class PageLogits:
    """A logit from each pixel's grey value, in the windows inference would use."""

    identifier = "page-logits"

    def __init__(self) -> None:
        self.analysed: list[tuple[int, int]] = []

    def analyse(self, page: npt.NDArray[np.uint8]) -> LearnedOutput:
        self.analysed.append(page.shape)
        logits = ((page.astype(np.float32) - 128) / 16).astype(np.float32)
        return LearnedOutput(logits, trufor.tiles(*page.shape), 0.5)


def noise(height: int, width: int, seed: int = 0) -> npt.NDArray[np.uint8]:
    return np.random.default_rng(seed).integers(0, 200, size=(height, width), dtype=np.uint8)


def forged_page(seed: int = 0) -> npt.NDArray[np.uint8]:
    page = noise(60, 100, seed)
    page[10:20, 10:30] = 230  # the edit scores higher than most of the receipt
    return page


def item(
    key: str,
    group: Group = Group.HELD_OUT,
    *,
    forged: bool = True,
    seed: int = 0,
) -> Item:
    if not forged:
        return Item(key, group, noise(60, 100, seed), forged=False)
    return Item(
        key,
        group,
        forged_page(seed),
        forged=True,
        modified=(box(10, 10, 20, 10),),
        sources=(box(60, 30, 10, 10),),
    )


def synthetic(key: str) -> Item:
    return Item(
        key,
        Group.SYNTHETIC,
        forged_page(),
        forged=True,
        modified=(box(10, 10, 20, 10), box(50, 40, 10, 5)),
        edits=(Edit(Technique.CPI, box(10, 10, 20, 10)), Edit(Technique.PIX, box(50, 40, 10, 5))),
    )


def test_a_receipt_of_the_dataset_carries_its_annotated_rectangles() -> None:
    modified, source = box(10, 10, 20, 10), box(60, 30, 10, 10)
    document = Document(
        dataset="toy",
        doc_id="X1",
        source_id="X1",
        split=Split.TRAIN,
        label=Label.FORGED,
        image_path=Path("x.png"),
        regions=(
            Region(box=modified, role=RegionRole.MODIFIED),
            Region(box=source, role=RegionRole.SOURCE),
        ),
        has_region_annotations=True,
    )
    words = [box(5, 5, 20, 8), box(40, 5, 30, 9)]
    receipt = real_item(document, Group.HELD_OUT, noise(60, 100), words)
    assert (receipt.key, receipt.forged, receipt.modified, receipt.sources, receipt.edits) == (
        "X1",
        True,
        (modified,),
        (source,),
        (),
    )
    assert receipt.words == tuple(words)
    authentic = real_item(
        document.model_copy(update={"label": Label.AUTHENTIC, "regions": ()}),
        Group.VALIDATION,
        noise(60, 100),
    )
    assert (authentic.forged, authentic.modified, authentic.words) == (False, (), ())


def test_a_synthetic_forgery_is_keyed_by_its_base_and_counts_only_sources_on_its_page() -> None:
    copied, donor = box(70, 5, 8, 10), box(1, 1, 8, 10)

    def region(
        technique: Technique, at: BBox, sources: tuple[BBox, ...], other: str | None
    ) -> SyntheticRegion:
        return SyntheticRegion(
            technique=technique,
            entity=Entity.TOTAL_PAYMENT,
            box=at,
            sources=sources,
            source_document=other,
            span=1,
            before="1",
            after="2",
            variants={},
        )

    record = ForgeryRecord(
        engine_version=ENGINE_VERSION,
        base="X9",
        held_out=2,
        stream=Stream.DEVELOPMENT,
        index=0,
        regions=(
            region(Technique.CPI, box(10, 10, 8, 10), (copied,), None),
            region(Technique.CPO, box(30, 10, 8, 10), (donor,), "X7"),  # copied from X7's page
        ),
        failures={},
    )
    blank = np.zeros((60, 100), dtype=bool)
    words = [box(2, 2, 10, 7)]  # the OCR words of its base
    forgery = synthetic_item(Forgery(noise(60, 100), blank, blank, record), words)
    assert forgery.words == tuple(words)
    assert (forgery.key, forgery.group, forgery.forged) == ("X9", Group.SYNTHETIC, True)
    assert forgery.modified == (box(10, 10, 8, 10), box(30, 10, 8, 10))
    assert forgery.sources == (copied,)
    assert [e.technique for e in forgery.edits] == [Technique.CPI, Technique.CPO]


def test_a_forged_receipt_is_measured_by_the_rules_of_adr_0015() -> None:
    receipt = item("X1")
    output = PageLogits().analyse(receipt.page)
    values = output.probabilities
    result = measure_receipt(receipt, output, threshold=0.9)
    assert (result.word_count, result.text_height) == (0, TEXT_HEIGHT_OTHER)  # no OCR words
    assert result.scores == output.scores(TEXT_HEIGHT_OTHER)
    assert result.top_region == top_region(values)
    assert result.lowest == values.min()
    assert values[result.peak] == values.max() == result.scores[HIGHEST]
    assert result.peak == divmod(int(np.argmax(values)), 100)  # the first in row-major order
    assert result.cuts is None
    assert (result.height, result.width, result.windows, result.seconds) == (60, 100, 1, 0.5)
    assert result.pixels is not None
    assert result.pixels.average_precision == pixel_average_precision(
        values, receipt.modified, receipt.sources
    )
    assert result.pixels.chance == positive_share((60, 100), receipt.modified, receipt.sources)
    assert (result.pixels.f1, result.pixels.iou) == pixel_f1_iou(
        values, 0.9, receipt.modified, receipt.sources
    )
    assert result.pixels.average_precision > result.pixels.chance
    assert result.edits == ()


def test_the_scores_and_the_diagnostic_follow_the_receipts_ocr_words() -> None:
    words = (box(5, 2, 20, 12), box(40, 30, 30, 8), box(10, 45, 25, 10), box(150, 0, 10, 10))
    receipt = Item("X5", Group.HELD_OUT, noise(60, 100), forged=False, words=words)
    output = PageLogits().analyse(receipt.page)
    values = output.probabilities
    result = measure_receipt(receipt, output, threshold=0.5)
    assert result.word_count == 3  # the last box starts outside the page
    assert result.text_height == text_height(words, 60, 100) == 10.0
    assert result.scores == output.scores(10.0)
    assert result.scores[QUARTER_SQUARE] < result.scores[HIGHEST]
    assert result.word_peaks == word_peaks(values, words)
    assert result.word_peaks is not None
    assert [len(p) for p in result.word_peaks.values()] == [3, 3, 3, 3]
    assert result.word_peaks[WordScale.PIXEL][0] == values[2:14, 5:25].max()
    synthetic_forgery = measure_receipt(
        Item("X5", Group.SYNTHETIC, forged_page(), forged=True, words=words), output, 0.5
    )
    assert synthetic_forgery.word_peaks is None  # the diagnostic covers the held-out fold
    assert synthetic_forgery.text_height == 10.0


def test_an_authentic_receipt_has_no_pixel_metrics() -> None:
    result = measure_receipt(item("X2", forged=False), PageLogits().analyse(noise(60, 100)), 0.5)
    assert (result.forged, result.pixels, result.modified, result.edits) == (False, None, (), ())


def test_every_edit_of_a_synthetic_forgery_is_measured_against_the_clean_pixels() -> None:
    receipt = synthetic("X3")
    output = PageLogits().analyse(receipt.page)
    values = output.probabilities
    result = measure_receipt(receipt, output, threshold=0.9)
    expected = edit_average_precisions(values, [e.box for e in receipt.edits], receipt.sources)
    assert [e.technique for e in result.edits] == [Technique.CPI, Technique.PIX]
    for edit, measured, reference in zip(receipt.edits, result.edits, expected, strict=True):
        assert measured.box == edit.box
        assert measured.peak == values[edit.box.y : edit.box.y2, edit.box.x : edit.box.x2].max()
        assert (measured.average_precision, measured.chance) == (
            reference.average_precision,
            reference.chance,
        )
    assert result.edits[0].peak > result.edits[1].peak  # only the first edit was brightened


def test_a_receipt_in_several_windows_records_the_cuts(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(trufor, "TILE", 48)
    monkeypatch.setattr(trufor, "OVERLAP", 16)
    receipt = item("X4")
    output = PageLogits().analyse(receipt.page)
    assert len(output.windows) > 1
    result = measure_receipt(receipt, output, 0.5)
    assert result.cuts == trufor.cut_check(output.probabilities)


def test_the_pixel_threshold_follows_adr_0015_on_the_held_maps() -> None:
    model = PageLogits()
    held = []
    for seed in range(3):
        receipt = item(f"V{seed}", Group.VALIDATION, seed=seed)
        logits = model.analyse(receipt.page).logits
        held.append(HeldMap(receipt.key, receipt.modified, receipt.sources, logits))
    threshold = choose_threshold(held)
    maps = [LearnedOutput(h.logits, (), 0.0).probabilities for h in held]
    candidates = threshold_candidates(
        [threshold_sample(m, h.key) for m, h in zip(maps, held, strict=True)]
    )
    f1 = [
        f1_at_thresholds(m, candidates, h.modified, h.sources)
        for m, h in zip(maps, held, strict=True)
    ]
    assert threshold.receipts == 3
    assert threshold.candidates == tuple(candidates)
    assert threshold.mean_f1 == pytest.approx(tuple(np.mean(f1, axis=0)))
    assert threshold.value == choose_pixel_threshold(candidates, f1)
    with pytest.raises(ValueError, match="no forged validation receipt"):
        choose_threshold([])


def run(
    validation: list[Item], measured: list[Item], seen: list[tuple[Group, str]] | None = None
) -> RunResults:
    return measure_run(
        PageLogits(),
        iter(validation),
        iter(measured),
        held_out=2,
        system=4,
        seed=0,
        code=CODE,
        environment=ENVIRONMENT,
        on_progress=None if seen is None else lambda group, key: seen.append((group, key)),
    )


def test_a_run_scores_validation_for_thresholds_and_measures_the_rest() -> None:
    validation = [
        item("V1", Group.VALIDATION),
        item("V2", Group.VALIDATION, forged=False),
        item("V3", Group.VALIDATION, seed=1),
    ]
    measured = [item("D1"), item("D2", forged=False), synthetic("D2")]
    seen: list[tuple[Group, str]] = []
    results = run(validation, measured, seen)
    assert seen == [(i.group, i.key) for i in validation + measured]
    assert [r.key for r in results.validation] == ["V1", "V2", "V3"]
    assert not any(isinstance(r, MeasuredReceipt) for r in results.validation)
    assert all(type(r) is ScoredReceipt for r in results.validation)
    assert [(r.group, r.key) for r in results.measured] == [
        (Group.HELD_OUT, "D1"),
        (Group.HELD_OUT, "D2"),
        (Group.SYNTHETIC, "D2"),  # a synthetic forgery shares the key of its base
    ]
    assert results.threshold.receipts == 2  # the forged validation receipts with a rectangle
    assert (results.held_out, results.system, results.seed, results.model) == (
        2,
        4,
        0,
        "page-logits",
    )
    expected = measure_receipt(measured[0], PageLogits().analyse(measured[0].page), 0.0)
    assert results.measured[0].pixels is not None
    assert expected.pixels is not None
    assert results.measured[0].pixels.average_precision == expected.pixels.average_precision


def test_receipts_in_the_wrong_group_or_twice_are_refused() -> None:
    validation = [item("V1", Group.VALIDATION)]
    with pytest.raises(ValueError, match="belongs to held-out"):
        run([item("V1")], [])
    with pytest.raises(ValueError, match="belongs to validation"):
        run(validation, [item("D1", Group.VALIDATION)])
    with pytest.raises(ValueError, match="appears twice"):
        run(validation, [item("D1"), item("D1", forged=False)])
    with pytest.raises(ValueError, match="no forged validation receipt"):
        run([item("V2", Group.VALIDATION, forged=False)], [item("D1")])


def test_results_are_written_and_read_back_whole(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(trufor, "TILE", 48)
    monkeypatch.setattr(trufor, "OVERLAP", 16)
    results = run([item("V1", Group.VALIDATION)], [item("D1"), synthetic("D1")])
    flat = results.measured[0].model_copy(update={"cuts": (2.5, math.inf)})  # only cuts jump
    results = results.model_copy(update={"measured": (flat, *results.measured[1:])})
    path = write_results(tmp_path, results)
    assert path == tmp_path / "results.json"
    assert not list(tmp_path.glob("*.tmp"))
    assert read_results(tmp_path) == results
