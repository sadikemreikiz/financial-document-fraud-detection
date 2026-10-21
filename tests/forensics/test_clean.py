import math
from pathlib import Path

import numpy as np
import pytest

from docfraud.data.models import Document, Label
from docfraud.evaluation.design_fold import CLEAN_FOLDS
from docfraud.evaluation.localisation import hits
from docfraud.forensics.clean import (
    LATTICE,
    LATTICE_SCORE,
    PROBE,
    TRUFOR_SCORE,
    LatticeSystem,
    TruForSystem,
    measure_probe,
    measure_system,
    probe_identifier,
    probe_round,
    read_probe_results,
    system_directory,
)
from docfraud.forensics.lattice_map import lattice_evidence
from docfraud.forensics.learned.inference import HIGHEST
from docfraud.forensics.learned.results import Group, Item, measure_receipt, read_results
from docfraud.forensics.learned.rounds import RoundItems
from docfraud.forensics.probe import PROPERTIES, as_matrix, probe_scores
from docfraud.forensics.trufor import CachedTruFor
from tests.forensics.learned.test_results import CODE, ENVIRONMENT, PageLogits, item
from tests.forensics.test_quantisation import box, jpeg, scattered_text
from tests.forensics.test_trufor import FakeModel, noise
from tests.synthesis.pages import documents


def test_system_1_gives_its_map_and_scores_in_the_measured_form() -> None:
    page = jpeg(scattered_text(3))[0].copy()
    page[240:280, 160:240] = page[103:143, 61:141]  # pasted off the grid
    output = LatticeSystem().analyse(page)
    evidence = lattice_evidence(page)
    height, width = page.shape
    assert output.windows == (box(0, 0, width, height),)  # analysed at once
    assert np.array_equal(output.values, evidence.pixel_map())
    assert output.scores(40.0) == {
        HIGHEST: float(evidence.pixel_map().max()),
        LATTICE_SCORE: evidence.document_score(),
    }
    assert output.seconds >= 0
    patch = box(160, 240, 80, 40)
    receipt = Item("X1", Group.HELD_OUT, page, forged=True, modified=(patch,))
    measured = measure_receipt(receipt, output, threshold=0.5)
    assert hits(measured.top_region, [patch])
    assert measured.cuts is None
    assert LatticeSystem().identifier == LatticeSystem().identifier
    assert LatticeSystem().identifier.startswith("lattice-")


def test_system_6_gives_trufors_map_and_integrity_score(tmp_path: Path) -> None:
    trufor = CachedTruFor(FakeModel(), tmp_path)
    page = noise(60, 80)
    output = TruForSystem(trufor).analyse(page)
    stored = trufor.analyse(page)  # read from the cache
    assert np.array_equal(output.values, stored.localisation.astype(np.float64))
    assert output.values.dtype == np.float64
    assert output.windows == stored.windows
    assert output.seconds == stored.seconds
    assert output.scores(40.0) == {
        HIGHEST: float(stored.localisation.max()),
        TRUFOR_SCORE: stored.integrity_score(),
    }
    assert TruForSystem(trufor).identifier == "fake-1"


def round_items() -> RoundItems:
    return RoundItems(
        lambda: [item("V1", Group.VALIDATION), item("V2", Group.VALIDATION, forged=False)],
        lambda: [item("X1"), item("X2", forged=False, seed=1)],
    )


def test_a_system_without_training_is_measured_in_every_round_once(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    checked: list[tuple[int, ...]] = []
    monkeypatch.setattr(
        "docfraud.forensics.clean.require_trained",
        lambda _root, rounds, _runs, _settings: checked.append(tuple(rounds)),
    )
    model = PageLogits()
    started: list[int] = []
    results = measure_system(
        model,
        LATTICE,
        tmp_path,
        lambda _held_out: round_items(),
        code=CODE,
        environment=ENVIRONMENT,
        on_round=started.append,
    )
    assert checked == [CLEAN_FOLDS]
    assert started == list(CLEAN_FOLDS)
    for held_out, result in zip(CLEAN_FOLDS, results, strict=True):
        assert (result.held_out, result.system, result.seed) == (held_out, LATTICE, 0)
        assert [r.key for r in result.validation] == ["V1", "V2"]
        assert [r.key for r in result.measured] == ["X1", "X2"]
        assert all(r.word_peaks is None for r in result.measured)  # a learned diagnostic
        assert read_results(system_directory(tmp_path, held_out, LATTICE)) == result
    analysed = len(model.analysed)
    again = measure_system(
        model,
        LATTICE,
        tmp_path,
        lambda _held_out: round_items(),
        code=CODE,
        environment=ENVIRONMENT,
    )
    assert again == results
    assert len(model.analysed) == analysed  # measured by the same system before


def test_nothing_is_measured_before_the_learned_runs_have_finished(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    def unfinished(*_args: object) -> None:
        msg = "system 2, seed 0 of round 0 has not finished"
        raise ValueError(msg)

    monkeypatch.setattr("docfraud.forensics.clean.require_trained", unfinished)
    model = PageLogits()
    with pytest.raises(ValueError, match="has not finished"):
        measure_system(
            model, LATTICE, tmp_path, lambda _k: round_items(), code=CODE, environment=ENVIRONMENT
        )
    with pytest.raises(ValueError, match="has not finished"):
        measure_probe(
            [], documents(tmp_path)[1], lambda _d: {}, tmp_path, code=CODE, environment=ENVIRONMENT
        )
    assert model.analysed == []
    assert not any(tmp_path.glob("round-*"))


def test_the_probe_learns_from_the_other_folds_and_scores_the_held_out_one(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr("docfraud.forensics.clean.require_trained", lambda *_args: None)
    docs, folds = documents(tmp_path)
    rng = np.random.default_rng(0)
    properties = {d.doc_id: {name: float(rng.normal()) for name in PROPERTIES} for d in docs}
    first = min(folds.documents(1))
    properties[first][PROPERTIES[0]] = math.nan  # a property not defined on a receipt
    read: list[str] = []

    def found(document: Document) -> dict[str, float]:
        read.append(document.doc_id)
        return properties[document.doc_id]

    progressed: list[str] = []
    results = measure_probe(
        docs,
        folds,
        found,
        tmp_path / "rounds",
        code=CODE,
        environment=ENVIRONMENT,
        on_receipt=progressed.append,
    )
    assert sorted(read) == sorted(folds.folds)  # every receipt's properties read once
    assert progressed == read
    for held_out, result in zip(CLEAN_FOLDS, results, strict=True):
        training = sorted(d.doc_id for d in docs if folds.folds[d.doc_id] != held_out)
        predicted = sorted(folds.documents(held_out))
        by_id = {d.doc_id: d for d in docs}
        expected = probe_scores(
            as_matrix([properties[i] for i in training]),
            np.array([by_id[i].label is Label.FORGED for i in training]),
            as_matrix([properties[i] for i in predicted]),
        )
        assert (result.held_out, result.model) == (held_out, probe_identifier())
        assert (result.trained_on, result.trained_forged) == (len(training), 4)
        assert [r.key for r in result.receipts] == predicted
        assert [r.score for r in result.receipts] == list(expected)
        assert [r.forged for r in result.receipts] == [
            by_id[i].label is Label.FORGED for i in predicted
        ]
        stored = read_probe_results(tmp_path / "rounds", held_out)
        assert stored.model_dump(exclude={"receipts"}) == result.model_dump(exclude={"receipts"})
        assert [r.key for r in stored.receipts] == predicted
    kept = {r.key: r for r in read_probe_results(tmp_path / "rounds", 1).receipts}
    assert math.isnan(kept[first].properties[PROPERTIES[0]])  # NaN is written and read as such
    assert system_directory(tmp_path, 0, PROBE).name == "system-5"
    with pytest.raises(ValueError, match="fold 7 holds no receipt"):
        probe_round(docs, folds, 7, properties, code=CODE, environment=ENVIRONMENT)
