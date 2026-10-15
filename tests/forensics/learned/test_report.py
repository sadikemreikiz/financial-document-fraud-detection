from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import UTC, datetime

import numpy as np
import pytest

from docfraud.data.benign_marks import Mark, MarkKind
from docfraud.data.models import BBox, Label, Technique
from docfraud.data.observation import Presence
from docfraud.evaluation.runs import CodeVersion
from docfraud.forensics.learned.inference import CONTRAST, HIGHEST
from docfraud.forensics.learned.report import (
    BLANK_SHARE,
    RoundData,
    RunKey,
    blank_table,
    chosen_score,
    curve_table,
    evidence,
    guard_table,
    ink_share,
    large,
    mark_table,
    report,
    score_table,
    size_class,
    technique_table,
)
from docfraud.forensics.learned.results import (
    EditResult,
    Group,
    MeasuredReceipt,
    PixelScores,
    PixelThreshold,
    RunResults,
)
from docfraud.forensics.learned.rounds import RUNS, Repeatability
from docfraud.forensics.learned.sampler import System
from docfraud.forensics.learned.training import RunRecord, Settings
from tests.forensics.learned.test_training import ENVIRONMENT

CODE = CodeVersion(commit="0123456789abcdef", dirty=False)
AUTHENTIC = 40
RESAMPLES = 500  # enough to separate the scores here, and quick


def box(x: int, y: int, width: int, height: int) -> BBox:
    return BBox(x=x, y=y, width=width, height=height)


def receipt(
    key: str,
    group: Group = Group.HELD_OUT,
    *,
    height: int = 200,
    highest: float = 0.5,
    contrast: float = 0.2,
    lowest: float = 0.01,
    top: BBox | None = None,
    modified: tuple[BBox, ...] = (),
    edits: tuple[EditResult, ...] = (),
    windows: int = 1,
) -> MeasuredReceipt:
    forged = bool(modified)
    return MeasuredReceipt(
        key=key,
        group=group,
        forged=forged,
        height=height,
        width=100,
        windows=windows,
        seconds=0.2,
        scores={HIGHEST: highest, CONTRAST: contrast},
        lowest=lowest,
        peak=(15, 15),
        top_region=top or box(10, 10, 20, 10),
        cuts=(3.0, 1.5) if windows > 1 else None,
        modified=modified,
        sources=(),
        pixels=PixelScores(average_precision=0.4, chance=0.01, f1=0.3, iou=0.2) if forged else None,
        edits=edits,
    )


def authentic_receipts(growing: bool) -> list[MeasuredReceipt]:
    """Authentic receipts of growing size; the highest score grows with them if asked to."""
    shuffled = np.random.default_rng(0).permutation(AUTHENTIC)
    return [
        receipt(
            f"A{n:02d}",
            height=200 + 10 * n,
            highest=0.5 + n / 100 if growing else 0.5 + float(shuffled[n]) / 100,
            contrast=float(shuffled[n]) / 100,
        )
        for n in range(AUTHENTIC)
    ]


def forged_receipts() -> list[MeasuredReceipt]:
    return [
        receipt("F0", highest=0.99, modified=(box(10, 10, 20, 10),)),  # hit by the top region
        receipt("F1", highest=0.98, modified=(box(60, 150, 20, 20),)),  # missed
        receipt("F2", highest=0.97, modified=(box(0, 0, 100, 150),)),  # large, hit
    ]


def synthetic_receipts() -> list[MeasuredReceipt]:
    def edit(technique: Technique, peak: float) -> EditResult:
        return EditResult(
            technique=technique,
            box=box(10, 10, 20, 10),
            peak=peak,
            average_precision=0.5,
            chance=0.02,
        )

    return [
        receipt(  # A00's highest is 0.5 + shuffled[0] / 100, at most 0.89
            "A00",
            Group.SYNTHETIC,
            modified=(box(10, 10, 20, 10),),
            edits=(edit(Technique.CPI, 0.95), edit(Technique.PIX, 0.10)),
        ),
    ]


def results(system: System, seed: int, held_out: list[MeasuredReceipt]) -> RunResults:
    return RunResults(
        held_out=2,
        system=int(system),
        seed=seed,
        model=f"learned-{int(system)}{seed}-torch2.14.1-mps",
        code=CODE,
        environment=ENVIRONMENT,
        created=datetime(2026, 10, 16, tzinfo=UTC),
        threshold=PixelThreshold(receipts=33, candidates=(0.5,), mean_f1=(0.3,), value=0.5),
        validation=(),
        measured=(*held_out, *synthetic_receipts()),
    )


def record(system: System, seed: int) -> RunRecord:
    steps = 1000
    return RunRecord(
        held_out=2,
        system=int(system),
        seed=seed,
        settings=asdict(Settings()),
        prior=0.0138,
        code=CODE,
        environment=ENVIRONMENT,
        device="mps",
        started=datetime(2026, 10, 15, tzinfo=UTC),
        status="finished",
        losses=list(np.linspace(0.2, 0.05, steps)),
        forged_shares=[0.02] * steps,
        seconds=[0.5] * steps,
        resumed_at=[],
    )


REPEAT = Repeatability(
    held_out=2,
    system=4,
    seed=0,
    steps=500,
    device="mps",
    code=CODE,
    tensors=122,
    differing=("decoder.0.weight",),
    largest_difference=3e-6,
    largest_loss_difference=1e-5,
)


def round_data(growing: bool = False, ink: float = 0.5) -> RoundData:
    held_out = [*authentic_receipts(growing), *forged_receipts()]
    words = {r.key: [box(5, 5, 40, 10)] for r in held_out}
    return RoundData(
        records={key: record(*key) for key in RUNS},
        results={key: results(*key, held_out) for key in RUNS},
        repeatability=REPEAT,
        words=words,
        flagged={"A01": Label.AUTHENTIC, "F1": Label.FORGED},
        marks={
            "A01": [Mark(box=box(5, 5, 30, 20), kind=MarkKind.STAMP, touches_print=Presence.NO)]
        },
        ink=lambda _key, _box: ink,
        resamples=RESAMPLES,
    )


def sized(runs: Mapping[RunKey, RunResults]) -> dict[RunKey, RunResults]:
    """The same results with both document scores growing with the size of the receipt."""
    return {
        key: r.model_copy(
            update={
                "measured": tuple(
                    m.model_copy(
                        update={"scores": dict.fromkeys((HIGHEST, CONTRAST), m.height / 1e4)}
                    )
                    for m in r.measured
                )
            }
        )
        for key, r in runs.items()
    }


def test_receipts_are_sorted_by_size_and_evidence() -> None:
    assert size_class(receipt("a")) == "one pass"
    assert size_class(receipt("b", windows=2)) == "tiled"
    assert size_class(receipt("c", height=7000, windows=24)) == "page scan"
    assert evidence(receipt("d"))
    assert not evidence(receipt("e", highest=0.3, lowest=0.3))
    assert large(receipt("f", modified=(box(0, 0, 100, 20),)))  # 10 % of the page
    assert not large(receipt("g", modified=(box(0, 0, 10, 10),)))


def test_the_document_score_follows_the_fixed_order() -> None:
    assert chosen_score(round_data().results, RESAMPLES) == HIGHEST
    assert (
        chosen_score(round_data(growing=True).results, RESAMPLES) == CONTRAST
    )  # the first grows with size
    growing = round_data(growing=True)
    both = sized(growing.results)
    assert chosen_score(both, RESAMPLES) is None
    without = {k: v for k, v in growing.results.items() if k != (System.REAL, 0)}
    with pytest.raises(KeyError):
        chosen_score(without, RESAMPLES)


def test_the_score_table_shows_every_run_and_each_systems_mean() -> None:
    table = score_table(round_data())
    labels = [row[0] for row in table.rows]
    assert labels[:4] == [
        "system 2, seed 0",
        "system 2, seed 1",
        "system 2, seed 2",
        "system 2, mean of 3 seeds",
    ]
    assert len(table.rows) == 12
    mean = table.rows[3]
    assert mean[1] == pytest.approx(table.rows[0][1])  # every seed has the same receipts here
    assert (mean[2], mean[3]) == (None, None)  # a size check is not averaged


def test_the_guard_counts_hits_with_and_without_large_rectangles() -> None:
    table = guard_table(round_data(), Group.HELD_OUT)
    row = table.rows[0]
    assert row[1:3] == [3, 2]  # F0 and the large F2 are hit
    assert row[6:8] == [2, 1]  # without F2
    synthetic = guard_table(round_data(), Group.SYNTHETIC)
    assert synthetic.title == "Top region on synthetic forgeries"
    assert synthetic.rows[0][1:3] == [1, 1]


def test_an_edit_is_seen_when_it_raises_the_map_above_its_base() -> None:
    table = technique_table(round_data())
    by_technique = {row[0]: row for row in table.rows}
    assert set(by_technique) == {"cpi", "pix"}
    assert by_technique["cpi"][1:4] == [1, "100 %", 0.5]
    assert by_technique["pix"][2] == "0 %"
    assert by_technique["cpi"][-1] == 0.02


def test_the_curve_divides_the_loss_by_that_of_the_best_constant() -> None:
    table = curve_table(round_data())
    assert table.columns == ["Run", "steps 1-500", "steps 501-1000"]
    first = table.rows[0][1]
    assert isinstance(first, float)
    assert first == pytest.approx(
        float(np.mean(np.linspace(0.2, 0.05, 1000)[:500]))
        / -(0.02 * np.log(0.02) + 0.98 * np.log(0.98))
    )


def test_marks_and_blank_paper_are_counted_per_run() -> None:
    marks = mark_table(round_data(), HIGHEST)
    assert marks.rows[0][1:3] == [1, 1]  # A01's top region overlaps its stamp
    assert marks.rows[0][4:6] == [1, 0]
    assert blank_table(round_data(ink=BLANK_SHARE / 2)).rows[0][1] == f"{AUTHENTIC} of {AUTHENTIC}"
    assert blank_table(round_data(ink=0.2)).rows[0][1] == f"0 of {AUTHENTIC}"
    page = np.full((10, 10), 255, dtype=np.uint8)
    page[0, :5] = 0
    assert ink_share(page, box(0, 0, 10, 2)) == 0.25


def test_the_report_states_its_outcome_and_every_table() -> None:
    text = report(round_data())
    assert f"the **{HIGHEST}**" in text
    assert "Trained at commit `0123456`; measured at commit `0123456`." in text
    dirty = CodeVersion(commit="fedcba9876543210", dirty=True)
    data = round_data()
    later = {k: r.model_copy(update={"code": dirty}) for k, r in data.results.items()}
    header = report(replace(data, results=later)).split("\n\n")[3]
    assert (
        header
        == "- Trained at commit `0123456`; measured at commit `fedcba9` with uncommitted changes."
    )
    mixed = dict(later)
    mixed[RUNS[0]] = data.results[RUNS[0]]
    header = report(replace(data, results=mixed)).split("\n\n")[3]
    assert header.endswith("commits `0123456`, `fedcba9` with uncommitted changes.")
    for title in (
        "Runs",
        "Learning curves",
        "Repeatability",
        "Document scores",
        "Top region on forged receipts",
        "Pixel metrics",
        "Document score on synthetic forgeries",
        "Top region on synthetic forgeries",
        "Synthetic edits by technique",
        "Top region on a benign mark",
        "Top region on blank paper",
        "The cuts between windows",
        "Times",
    ):
        assert f"### {title}\n" in text, title


def test_a_report_without_a_passing_score_says_development_stops() -> None:
    data = round_data()
    flat = sized(data.results)
    text = report(replace(data, results=flat))
    assert "development stops" in text
    assert "Document score on synthetic forgeries" not in text
