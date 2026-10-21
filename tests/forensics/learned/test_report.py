from collections.abc import Mapping
from dataclasses import asdict, replace
from datetime import UTC, datetime

import numpy as np
import pytest

from docfraud.data.benign_marks import Mark, MarkKind
from docfraud.data.models import BBox, Label, Technique
from docfraud.data.observation import Presence
from docfraud.evaluation.runs import CodeVersion
from docfraud.forensics.learned.inference import (
    CONTRAST,
    HALF_SQUARE,
    HIGHEST,
    QUARTER_SQUARE,
    SCORE_ORDER,
)
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
    size_check,
    size_class,
    technique_table,
    text_table,
    used_score,
    word_superiority,
    word_table,
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
from docfraud.forensics.learned.text_scale import WordScale
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
    word_value: float = 0.1,
) -> MeasuredReceipt:
    forged = bool(modified)
    pair: tuple[float, ...] = (word_value, word_value / 2)
    peaks = dict.fromkeys(WordScale, pair)
    return MeasuredReceipt(
        key=key,
        group=group,
        forged=forged,
        height=height,
        width=100,
        windows=windows,
        seconds=0.2,
        word_count=2,
        text_height=10.0,
        scores={
            HIGHEST: highest,
            CONTRAST: contrast,
            QUARTER_SQUARE: 0.9 * contrast,
            HALF_SQUARE: 0.8 * contrast,
        },
        lowest=lowest,
        peak=(15, 15),
        top_region=top or box(10, 10, 20, 10),
        cuts=(3.0, 1.5) if windows > 1 else None,
        modified=modified,
        sources=(),
        pixels=PixelScores(average_precision=0.4, chance=0.01, f1=0.3, iou=0.2) if forged else None,
        edits=edits,
        word_peaks=peaks if group is Group.HELD_OUT else None,
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
                    m.model_copy(update={"scores": dict.fromkeys(SCORE_ORDER, m.height / 1e4)})
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


def unsized_quarter(runs: Mapping[RunKey, RunResults]) -> dict[RunKey, RunResults]:
    """The same results with score 3 unrelated to the size of the receipt."""
    return {
        key: r.model_copy(
            update={
                "measured": tuple(
                    m.model_copy(
                        update={
                            "scores": {**m.scores, QUARTER_SQUARE: (int(m.key[1:]) * 17 % 41) / 100}
                        }
                    )
                    for m in r.measured
                )
            }
        )
        for key, r in runs.items()
    }


def test_the_order_ends_with_the_highest_value_when_no_candidate_passes() -> None:
    data = round_data()
    assert used_score(data.results, RESAMPLES) == (HIGHEST, True)
    assert used_score(sized(data.results), RESAMPLES) == (HIGHEST, False)
    assert used_score(unsized_quarter(sized(data.results)), RESAMPLES) == (QUARTER_SQUARE, True)


def test_the_score_table_shows_every_candidate_run_and_each_systems_mean() -> None:
    table = score_table(round_data())
    assert len(table.rows) == 4 * 12
    labels = [row[0] for row in table.rows if row[0]]
    assert labels == [
        f"1. {HIGHEST}",
        f"2. {CONTRAST}",
        f"3. {QUARTER_SQUARE} (added by ADR-0023)",
        f"4. {HALF_SQUARE} (added by ADR-0023)",
    ]
    runs = [row[1] for row in table.rows[:4]]
    assert runs == [
        "system 2, seed 0",
        "system 2, seed 1",
        "system 2, seed 2",
        "system 2, mean of 3 seeds",
    ]
    mean = table.rows[3]
    assert mean[2] == pytest.approx(table.rows[0][2])  # every seed has the same receipts here
    assert (mean[3], mean[4], mean[5]) == (None, None, None)  # a size check is not averaged


def test_the_size_check_can_leave_out_page_scans() -> None:
    data = round_data()
    first = data.results[RUNS[0]]
    scan = receipt("A00", height=7000, highest=0.99)
    changed = first.model_copy(update={"measured": (scan, *first.measured[1:])})
    every = size_check(changed, HIGHEST, RESAMPLES)
    without = size_check(changed, HIGHEST, RESAMPLES, page_scans=False)
    assert without == size_check(
        first.model_copy(update={"measured": first.measured[1:]}), HIGHEST, RESAMPLES
    )
    assert every != without


def test_word_superiority_counts_every_pair_of_words() -> None:
    rng = np.random.default_rng(4)
    first = [list(np.round(rng.random(n), 1)) for n in (3, 5, 2)]
    second = [list(np.round(rng.random(n) * 0.8, 1)) for n in (4, 1, 6, 2)]
    a, b = np.concatenate(first), np.concatenate(second)
    expected = float(((a[:, None] > b[None, :]) + 0.5 * (a[:, None] == b[None, :])).mean())
    estimate = word_superiority(first, second, resamples=RESAMPLES)
    assert estimate.value == pytest.approx(expected)
    assert estimate.low <= estimate.value <= estimate.high
    assert word_superiority([[1.0, 2.0]], [[1.0], [0.0]], resamples=10).value == 0.875
    with pytest.raises(ValueError, match="both groups"):
        word_superiority([], [[1.0]])


def scanned(data: RoundData, count: int = 5) -> RoundData:
    """The same round with its first authentic receipts as page scans whose words score high."""
    keys = {f"A{n:02d}" for n in range(count)}
    scan_words: tuple[float, ...] = (0.9, 0.8, 0.7)

    def changed(m: MeasuredReceipt) -> MeasuredReceipt:
        if m.key not in keys or m.group is not Group.HELD_OUT:
            return m
        return m.model_copy(
            update={
                "height": 7000,
                "word_count": 3,
                "text_height": 90.0,
                "word_peaks": dict.fromkeys(WordScale, scan_words),
            }
        )

    results = {
        key: r.model_copy(update={"measured": tuple(changed(m) for m in r.measured)})
        for key, r in data.results.items()
    }
    return replace(data, results=results)


def test_the_text_and_words_of_page_scans_are_compared_with_other_receipts() -> None:
    data = scanned(round_data())
    text = text_table(data)
    assert text.rows == [
        ["page scans", 5, 3.0, 90.0, pytest.approx(0.7)],
        ["other receipts", AUTHENTIC - 5, 2.0, 10.0, pytest.approx(0.042)],
    ]
    assert "kind: 0 of" in text.note
    words = word_table(data)
    assert words.columns[1:5] == [f"Blocks of {scale}" for scale in WordScale]
    assert words.rows[0][1:5] == ["1.00 (1.00 to 1.00)"] * 4  # every scan word scores higher
    assert words.rows[0][5] == f"{AUTHENTIC} of {AUTHENTIC}"  # (15, 15) lies below row 14
    taller = replace(data, words={key: [box(5, 5, 40, 20)] for key in data.words})
    assert word_table(taller).rows[0][5] == f"0 of {AUTHENTIC}"
    assert word_table(round_data()).rows[0][1] is None  # no page scan to compare


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
        "Text of page scans",
        "Words of page scans against other receipts",
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


def test_a_report_without_a_passing_score_keeps_the_highest_value() -> None:
    data = round_data()
    flat = sized(data.results)
    text = report(replace(data, results=flat))
    assert "By ADR-0023 (decision 3), the order ends there" in text
    assert f"keep the **{HIGHEST}**" in text
    assert "Document score on synthetic forgeries" in text
    assert "Its size check failed (ADR-0023, decision 3)." in text


def test_a_report_names_a_candidate_added_after_the_results() -> None:
    data = round_data()
    text = report(replace(data, results=unsized_quarter(sized(data.results))))
    assert f"the **{QUARTER_SQUARE}**. It was added by ADR-0023" in text
