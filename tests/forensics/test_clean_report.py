from dataclasses import asdict, replace
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest

from docfraud.data.models import BBox, Document, Entity, Label, Region, RegionRole, Split, Technique
from docfraud.evaluation.design_fold import CLEAN_FOLDS
from docfraud.evaluation.detection import average_precision, choose_threshold
from docfraud.forensics.clean import LATTICE, PROBE, TRUFOR, ProbeReceipt, ProbeResults
from docfraud.forensics.clean_report import (
    MAP_SYSTEMS,
    SCORES,
    CleanData,
    breakdown_tables,
    comparisons,
    localised,
    pooled,
    probe_table,
    report,
    size_table,
)
from docfraud.forensics.learned.inference import HIGHEST
from docfraud.forensics.learned.results import (
    Group,
    MeasuredReceipt,
    PixelScores,
    PixelThreshold,
    RunResults,
    ScoredReceipt,
)
from docfraud.forensics.learned.sampler import System
from docfraud.forensics.learned.training import RunRecord, Settings
from docfraud.forensics.probe import PROPERTIES
from tests.forensics.learned.test_results import CODE, ENVIRONMENT

ROUNDS = (0, 1)
PER_FOLD = 12
FORGED = 4  # the first receipts of each fold
RESAMPLES = 40
EDIT = BBox(x=40, y=40, width=20, height=10)


def key(fold: int, number: int) -> str:
    return f"R{fold}{number:02d}"


def document(fold: int, number: int) -> Document:
    forged = number < FORGED
    entity = Entity.METADATA if number % 2 else Entity.TOTAL_PAYMENT
    regions = (
        (
            Region(
                box=EDIT,
                role=RegionRole.MODIFIED,
                techniques=frozenset({Technique.CPI}),
                entity=entity,
            ),
            Region(box=BBox(x=5, y=150, width=20, height=10), role=RegionRole.SOURCE),
        )
        if forged
        else ()
    )
    return Document(
        dataset="test",
        doc_id=key(fold, number),
        source_id=key(fold, number),
        split=Split.TRAIN,
        label=Label.FORGED if forged else Label.AUTHENTIC,
        image_path=Path("unused.png"),
        regions=regions,
        has_region_annotations=forged,
        handwritten_annotation=number == FORGED,
        editing_software="paint" if forged else None,
    )


def measured(doc: Document, score: str, value: float, found: bool) -> MeasuredReceipt:
    """A receipt whose top region lies on its edit when ``found``."""
    number = int(doc.doc_id[-2:])
    height, width = (170 + 5 * number, 100) if number != 9 else (7000, 100)  # a page scan per fold
    forged = doc.label is Label.FORGED
    modified = (EDIT,) if forged else ()
    sources = (BBox(x=5, y=150, width=20, height=10),) if forged else ()
    return MeasuredReceipt(
        key=doc.doc_id,
        group=Group.HELD_OUT,
        forged=forged,
        height=height,
        width=width,
        windows=1,
        seconds=0.5 + value,
        word_count=3,
        text_height=10.0,
        scores={HIGHEST: abs(value) + 1, score: value},  # every map holds evidence
        lowest=0.0,
        peak=(45, 50),
        top_region=EDIT if found else BBox(x=0, y=0, width=10, height=10),
        cuts=None,
        modified=modified,
        sources=sources,
        pixels=PixelScores(average_precision=0.5 + value / 10, chance=0.01, f1=0.2, iou=0.1)
        if forged
        else None,
        edits=(),
        word_peaks=None,
    )


def validation(score: str, shift: float) -> tuple[ScoredReceipt, ...]:
    """Validation scores whose F1 threshold is ``1 + shift``."""
    rows = []
    for number, (forged, value) in enumerate(
        ((True, 1.0), (True, 1.2), (False, 0.2), (False, 0.4))
    ):
        rows.append(
            ScoredReceipt(
                key=f"V{number}",
                group=Group.VALIDATION,
                forged=forged,
                height=10,
                width=10,
                windows=1,
                seconds=0.1,
                word_count=1,
                text_height=10.0,
                scores={HIGHEST: value, score: value + shift},
            )
        )
    return tuple(rows)


def results(system: int, held_out: int, seed: int, documents: list[Document]) -> RunResults:
    score = SCORES[system]
    rng = np.random.default_rng([system, held_out, seed])
    receipts = []
    for doc in documents:
        forged = doc.label is Label.FORGED
        strength = 0.3 * (system - 1)  # higher systems separate the classes better
        value = float(rng.normal(strength if forged else 0.0, 0.5))
        receipts.append(measured(doc, score, value, found=forged and system != LATTICE))
    return RunResults(
        held_out=held_out,
        system=system,
        seed=seed,
        model=f"model-{system}",
        code=CODE,
        environment=ENVIRONMENT,
        created=datetime(2026, 10, 19, tzinfo=UTC),
        threshold=PixelThreshold(receipts=2, candidates=(0.5,), mean_f1=(0.2,), value=0.5),
        validation=validation(score, 0.1 * held_out),
        measured=tuple(receipts),
        peak_memory=2**30,
    )


def record(held_out: int, system: System, seed: int) -> RunRecord:
    return RunRecord(
        held_out=held_out,
        system=int(system),
        seed=seed,
        settings=asdict(Settings()),
        prior=0.01,
        code=CODE,
        environment=ENVIRONMENT,
        device="cpu",
        started=datetime(2026, 10, 19, tzinfo=UTC),
        status="finished",
        losses=[0.01] * 600,
        forged_shares=[0.02] * 600,
        seconds=[0.5] * 600,
        resumed_at=[],
    )


def probe(held_out: int, documents: list[Document], separating: bool) -> ProbeResults:
    receipts = tuple(
        ProbeReceipt(
            key=d.doc_id,
            forged=d.label is Label.FORGED,
            score=(0.9 if d.label is Label.FORGED else 0.1) if separating else 0.5,
            properties={name: float(i) for i, name in enumerate(PROPERTIES)},
        )
        for d in documents
    )
    return ProbeResults(
        held_out=held_out,
        model="probe-test",
        code=CODE,
        environment=ENVIRONMENT,
        created=datetime(2026, 10, 19, tzinfo=UTC),
        trained_on=30,
        trained_forged=8,
        receipts=receipts,
    )


def clean_data(*, separating_probe: bool = False, rounds: tuple[int, ...] = ROUNDS) -> CleanData:
    folds = {k: [document(k, n) for n in range(PER_FOLD)] for k in rounds}
    learned = {
        (k, system, seed): results(int(system), k, seed, folds[k])
        for k in rounds
        for system in System
        for seed in (0, 1)
    }
    return CleanData(
        records={key: record(*key) for key in learned},
        learned=learned,
        untrained={
            (k, system): results(system, k, 0, folds[k])
            for k in rounds
            for system in (LATTICE, TRUFOR)
        },
        probe={k: probe(k, folds[k], separating_probe) for k in rounds},
        documents={d.doc_id: d for docs in folds.values() for d in docs},
        words={
            d.doc_id: [BBox(x=10, y=10, width=60, height=150)]
            for docs in folds.values()
            for d in docs
        },
        rounds=rounds,
        earlier=frozenset({key(0, 0), key(1, 5)}),
        resamples=RESAMPLES,
    )


def test_each_receipt_takes_the_threshold_of_its_rounds_validation() -> None:
    data = clean_data()
    found = pooled(data, int(System.BOTH))
    assert [r.key for r in found.receipts] == sorted(data.documents)
    assert set(found.thresholds[:PER_FOLD]) == {
        choose_threshold([True, True, False, False], [1.0, 1.2, 0.2, 0.4])
    }
    assert list(found.thresholds[PER_FOLD:]) == [pytest.approx(1.1)] * PER_FOLD  # shifted scores
    assert np.array_equal(found.margins >= 0, found.scores >= np.array(found.thresholds))
    assert found.rounds == (0,) * PER_FOLD + (1,) * PER_FOLD


def test_the_receipts_must_be_those_of_the_clean_folds() -> None:
    data = clean_data()
    short = replace(data, documents={**data.documents, "R099": document(0, 99)})
    with pytest.raises(ValueError, match="not those of the clean folds"):
        pooled(short, LATTICE)


def test_the_comparisons_pair_the_systems_on_the_same_receipts() -> None:
    data = clean_data()
    (name, base, other, difference, _), _ = comparisons(data)
    assert (name, base, other) == ("RQ5: system 4 against system 3", 3, 4)
    expected = average_precision(
        pooled(data, 4).labels, pooled(data, 4).scores
    ) - average_precision(pooled(data, 3).labels, pooled(data, 3).scores)
    assert difference.value == pytest.approx(expected)


def test_a_hit_is_counted_against_the_edit_and_its_source() -> None:
    data = clean_data()
    found = localised(data, pooled(data, int(System.REAL)).receipts)
    assert set(found) == {d for d, doc in data.documents.items() if doc.label is Label.FORGED}
    assert all(v.hit for v in found.values())
    missed = localised(data, pooled(data, LATTICE).receipts)
    assert not any(v.hit for v in missed.values())
    sources = localised(data, pooled(data, int(System.REAL)).receipts, targets="sources")
    assert set(sources) == set(found)
    assert not any(v.hit for v in sources.values())  # the top region lies on the edit


def test_the_probe_finds_a_trace_only_above_chance() -> None:
    assert probe_table(clean_data(separating_probe=True)).rows[0][-1] == "carries a global trace"
    assert probe_table(clean_data()).rows[0][-1] == "no global trace found"


def test_the_breakdowns_count_every_receipt_in_its_groups() -> None:
    forged, authentic = breakdown_tables(clean_data())
    techniques = [row for row in forged.rows if row[0] == "technique"]
    assert techniques[0][1:3] == ["cpi", FORGED * len(ROUNDS)]
    assert sum(row[1] for row in authentic.rows if isinstance(row[1], int)) == (
        PER_FOLD - FORGED
    ) * len(ROUNDS)


def test_the_size_check_needs_three_receipts_with_evidence() -> None:
    data = clean_data()
    table = size_table(data)
    assert len(table.rows) == len(MAP_SYSTEMS) * len(ROUNDS)
    assert all(row[3] == PER_FOLD - FORGED for row in table.rows)  # every authentic has evidence
    flat = replace(
        data,
        untrained={
            k: v.model_copy(
                update={
                    "measured": tuple(
                        r.model_copy(update={"scores": {**r.scores, HIGHEST: r.lowest}})
                        for r in v.measured
                    )
                }
            )
            for k, v in data.untrained.items()
        },
    )
    rows = [row for row in size_table(flat).rows if row[0] == "system 1, JPEG lattice"]
    assert all(row[3] == 0 and row[6] is None for row in rows)  # no evidence, no check


def test_the_report_holds_every_table_of_the_record() -> None:
    text = report(clean_data())
    for title in (
        "Primary metric: document-level average precision",
        "Comparisons of the learned models",
        "Localisation guard",
        "Secondary document metrics",
        "Calibration",
        "Pixel metrics",
        "Copy sources",
        "Seeds",
        "Forged receipts the accounting rules cannot see",
        "Forged receipts found at the 10 % review budget",
        "Authentic receipts flagged at the 10 % review budget",
        "Receipts with evidence",
        "Size check, descriptive",
        "Global-trace probe",
        "Properties of the probe on the clean folds",
        "Document scores against the probe",
        "Runs of systems 2 to 4",
        "Systems without a run",
        "Times and memory",
    ):
        assert f"### {title}\n" in text
    assert "RQ5: system 4 against system 3: average precision" in text
    assert "trained at commit `abc`" in text
    assert "Guillaro" in text  # TruFor's authors are credited
    assert text.count("not a probability") == 2 * (len(MAP_SYSTEMS) - 1)  # two columns
    assert f"system {PROBE}" in text


def test_only_a_trial_on_other_rounds_says_it_is_one() -> None:
    assert "**Trial.**" in report(clean_data())
    text = report(clean_data(rounds=CLEAN_FOLDS))
    assert "**Trial.**" not in text
    assert "folds 0, 1, 3, 4" in text
    assert "one of 4 checks excludes zero with a probability of about 19 %" in text
