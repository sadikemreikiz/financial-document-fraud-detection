"""The synthetic forgery engine of ADR-0021 against the real forgeries it draws from.

Writes reports/synthesis/engine.md: for every cross-validation round, the real modified regions
of its training folds and the synthetic forgeries the engine makes from a seeded sample of its
bases, side by side (techniques, entities, characters changed, regions per receipt), the
templates that could not be placed and why, and how often each variant was drawn; and the
synthetic development set of the round that holds out the design fold. Only numbers are
reported, no image; receipts of the clean folds are read by code only (ADR-0014, decision 1).

With ``--timing`` it measures instead how long a forgery takes and writes
reports/synthesis/engine-timing.md, which is not part of ``make reports`` (``make
bench-synthesis``).

Usage: ``uv run python experiments/synthetic_engine.py [--timing]``
"""

import sys
import time
from collections import Counter
from collections.abc import Mapping, Sequence
from functools import lru_cache
from pathlib import Path

import numpy as np

from docfraud.data.findit import load_findit2
from docfraud.data.models import Entity, Technique
from docfraud.data.normalize import load_grayscale
from docfraud.evaluation.design_fold import DESIGN_FOLD
from docfraud.evaluation.folds import CV_FOLDS, load_folds
from docfraud.markdown import Cell, Table, render_table
from docfraud.reading.ocr import CachedOcr, RapidOcrEngine
from docfraud.synthesis.engine import (
    ENGINE_VERSION,
    Engine,
    Forgery,
    Receipt,
    Stream,
    SyntheticRegion,
    directory_fonts,
)
from docfraud.synthesis.layout import words
from docfraud.synthesis.statistics import RoundStatistics, round_statistics

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "data" / "raw" / "findit2"
FOLDS = ROOT / "annotations" / "findit2" / "cv-folds.json"
OCR_CACHE = ROOT / "artifacts" / "ocr"
FONTS = ROOT / "artifacts" / "fonts"
OUTPUT = ROOT / "reports" / "synthesis" / "engine.md"
TIMING = ROOT / "reports" / "synthesis" / "engine-timing.md"
SAMPLE = 60
"""Bases per round, drawn with seed 0, from which one forgery each is made for this report."""
SAMPLE_SEED = 0
TECHNIQUES = (Technique.CPI, Technique.CPO, Technique.IMI, Technique.CUT, Technique.PIX)
ENTITIES = tuple(Entity)


def _share(counts: Mapping[str, float], key: str) -> str:
    total = sum(counts.values())
    return f"{100 * counts.get(key, 0) / total:.0f} %" if total else "—"


def real_techniques(statistics: RoundStatistics) -> dict[str, float]:
    """Real regions per technique; a region with several techniques counts a share to each."""
    counts: dict[str, float] = {}
    for template in statistics.templates:
        for technique in template.techniques:
            share = 1 / len(template.techniques)
            counts[technique.value] = counts.get(technique.value, 0.0) + share
    return counts


def changed_characters(region: SyntheticRegion) -> int:
    """Characters a synthetic region changed: replaced, or deleted; 1 for a free-hand edit."""
    if region.after is None:
        return 1
    if len(region.after) == len(region.before):
        return sum(a != b for a, b in zip(region.before, region.after, strict=True))
    return len(region.before) - len(region.after)


class Round:
    """The real regions of one round and the forgeries made from its sample of bases."""

    def __init__(
        self, label: str, statistics: RoundStatistics, forgeries: Sequence[Forgery]
    ) -> None:
        self.label = label
        self.statistics = statistics
        self.forgeries = forgeries
        self.regions = [r for f in forgeries for r in f.record.regions]


def make_rounds() -> list[Round]:
    """Every round's statistics and sample, and the development set."""
    documents = load_findit2(DATA)
    by_id = {d.doc_id: d for d in documents}
    folds = load_folds(FOLDS)
    ocr = CachedOcr(RapidOcrEngine(), OCR_CACHE)

    @lru_cache(maxsize=32)
    def receipt(doc_id: str) -> Receipt:
        page = load_grayscale(by_id[doc_id].image_path)
        return Receipt(doc_id, page, words(page, ocr.recognize(page)))

    fonts = directory_fonts(FONTS)
    rounds = []
    for held_out in range(CV_FOLDS):
        statistics = round_statistics(documents, folds, held_out, ocr)
        engine = Engine(statistics, documents, folds, receipt, fonts)
        pool = engine.pools[Stream.TRAINING]
        picks = np.random.default_rng([SAMPLE_SEED, held_out]).choice(
            len(pool), SAMPLE, replace=False
        )
        sample = [engine.forge(pool[int(i)], 0) for i in sorted(picks)]
        rounds.append(Round(f"round {held_out}", statistics, sample))
        if held_out == DESIGN_FOLD:
            development = [
                engine.forge(base, 0, Stream.DEVELOPMENT)
                for base in engine.pools[Stream.DEVELOPMENT]
            ]
            rounds.append(Round(f"round {held_out}, development set", statistics, development))
        print(f"round {held_out} done", flush=True)  # noqa: T201
    return rounds


def real_table(rounds: Sequence[Round]) -> Table:
    """The real regions each round draws from."""
    rows: list[list[Cell]] = []
    for r in rounds:
        if "development" in r.label:
            continue
        s = r.statistics
        rows.append(
            [
                r.label,
                len(s.regions_per_receipt),
                len(s.templates),
                ", ".join(f"{k} {v}" for k, v in sorted(s.skipped.items())) or "none",
                ", ".join(f"{k} {v}" for k, v in sorted(s.widths_measured_on.items())),
            ]
        )
    return Table(
        "Real regions of each round",
        [
            "Round",
            "Forged receipts",
            "Templates",
            "Regions without a template",
            "Width measured on",
        ],
        rows,
        "Templates: the modified regions of the round's training folds, the four folds other than "
        "the one it holds out. Width measured on: the OCR words a region overlaps, else its row, "
        "else the whole receipt.",
    )


def share_table(rounds: Sequence[Round], what: str) -> Table:
    """Techniques or entities: real regions against synthetic ones."""
    keys = [t.value for t in TECHNIQUES] if what == "technique" else [e.value for e in ENTITIES]
    rows: list[list[Cell]] = []
    for r in rounds:
        if "development" not in r.label:
            real = (
                real_techniques(r.statistics)
                if what == "technique"
                else Counter(t.entity.value for t in r.statistics.templates)
            )
            rows.append([f"{r.label}, real", *(_share(real, k) for k in keys)])
        made = Counter(
            (x.technique.value if what == "technique" else x.entity.value) for x in r.regions
        )
        rows.append([f"{r.label}, synthetic", *(_share(made, k) for k in keys)])
    return Table(
        f"Regions by {what}",
        ["Regions", *keys],
        rows,
        "Shares of modified regions. Real: the round's templates; a region labelled with several "
        "techniques counts a share to each. Synthetic: the regions placed in the round's sample "
        f"({SAMPLE} bases, one forgery each) or in the development set.",
    )


def _quantiles(values: Sequence[int]) -> list[Cell]:
    if not values:
        return [None] * 4
    return [float(np.percentile(values, q)) for q in (25, 50, 75, 90)]


def characters_table(rounds: Sequence[Round]) -> Table:
    """Characters changed per region, real against synthetic."""
    rows: list[list[Cell]] = []
    for r in rounds:
        if "development" not in r.label:
            rows.append(
                [f"{r.label}, real", *_quantiles([t.characters for t in r.statistics.templates])]
            )
        rows.append([f"{r.label}, synthetic", *_quantiles([x.span for x in r.regions])])
        rows.append(
            [
                f"{r.label}, synthetic, changed",
                *_quantiles([changed_characters(x) for x in r.regions]),
            ]
        )
    return Table(
        "Characters per region",
        ["Regions", "25th percentile", "Median", "75th percentile", "90th percentile"],
        rows,
        "Real: the width of a modified rectangle in character widths, separators included. "
        "Synthetic: the characters of its word that the edit spans, separators included, at most "
        "the whole word; changed: the letters and digits among them that it replaced or deleted "
        "(a free-hand edit counts 1, since its result is not known).",
    )


def placing_table(rounds: Sequence[Round]) -> Table:
    """How many regions were placed, and why templates were not."""
    reasons = sorted({k for r in rounds for f in r.forgeries for k in f.record.failures})
    rows: list[list[Cell]] = []
    for r in rounds:
        failures: Counter[str] = Counter()
        for f in r.forgeries:
            failures.update(f.record.failures)
        real = r.statistics.regions_per_receipt
        rows.append(
            [
                r.label,
                len(r.forgeries),
                sum(not f.record.regions for f in r.forgeries),
                len(r.regions),
                f"{np.mean([len(f.record.regions) for f in r.forgeries]):.2f}"
                f" ({np.mean(real):.2f})",
                *(failures[k] for k in reasons),
            ]
        )
    return Table(
        "Placing the regions",
        ["Forgeries", "Made", "Without a region", "Regions", "Mean regions (real)", *reasons],
        rows,
        "Each region draws up to 20 templates until one can be placed. The failure columns count "
        "templates that could not be placed, and regions given up after 20.",
    )


def variant_table(rounds: Sequence[Round]) -> Table:
    """How often each variant was drawn, pooled over all rounds."""
    counts: Counter[tuple[str, str]] = Counter()
    for r in rounds:
        for region in r.regions:
            for step, value in region.variants.items():
                bucket = value
                if step in {"scale and rotation", "position"} and value not in {"none", "aligned"}:
                    bucket = "slight" if step == "scale and rotation" else "shifted"
                counts[step, bucket] += 1
    rows: list[list[Cell]] = [[step, value, n] for (step, value), n in sorted(counts.items())]
    return Table(
        "Variants drawn",
        ["Step", "Variant", "Regions"],
        rows,
        "Every variant is drawn with equal probability (ADR-0021, decision 3); the counts pool the "
        "samples of all rounds and the development set. Covering with blank paper falls back to "
        "the flat paper tone where no blank patch lies nearby.",
    )


def timing() -> None:
    """Measure the time per forgery on the development round and write its report."""
    documents = load_findit2(DATA)
    by_id = {d.doc_id: d for d in documents}
    folds = load_folds(FOLDS)
    ocr = CachedOcr(RapidOcrEngine(), OCR_CACHE)
    loaded: dict[str, Receipt] = {}

    def receipt(doc_id: str) -> Receipt:
        if doc_id not in loaded:
            page = load_grayscale(by_id[doc_id].image_path)
            loaded[doc_id] = Receipt(doc_id, page, words(page, ocr.recognize(page)))
        return loaded[doc_id]

    statistics = round_statistics(documents, folds, DESIGN_FOLD, ocr)
    engine = Engine(statistics, documents, folds, receipt, directory_fonts(FONTS))
    pool = engine.pools[Stream.TRAINING][:SAMPLE]
    load, forge = [], []
    for base in pool:
        start = time.perf_counter()
        receipt(base)
        load.append(time.perf_counter() - start)
        start = time.perf_counter()
        engine.forge(base, 1)
        forge.append(time.perf_counter() - start)
    table = Table(
        "Time per forgery",
        ["Step", "Receipts", "Median seconds", "95th percentile", "Most"],
        [
            [name, len(v), float(np.median(v)), float(np.percentile(v, 95)), max(v)]
            for name, v in (
                ("load page, OCR (cached) and words", load),
                ("forge, receipt loaded", forge),
            )
        ],
        "Measured on the first training bases of the development round, on the machine that "
        "ran it.",
    )
    TIMING.parent.mkdir(parents=True, exist_ok=True)
    TIMING.write_text(
        "# Synthetic forgery engine: time per forgery\n\n"
        "> Generated by `experiments/synthetic_engine.py --timing` (`make bench-synthesis`). "
        "Not part of `make reports`: times differ between runs.\n\n" + render_table(table),
        encoding="utf-8",
    )
    print(f"written: {TIMING.relative_to(ROOT)}")  # noqa: T201


def main() -> None:
    """Make every round's sample and write the report."""
    if "--timing" in sys.argv:
        timing()
        return
    rounds = make_rounds()
    tables = [
        real_table(rounds),
        placing_table(rounds),
        share_table(rounds, "technique"),
        share_table(rounds, "entity"),
        characters_table(rounds),
        variant_table(rounds),
    ]
    header = (
        "# Synthetic forgery engine: real and synthetic regions\n\n"
        "> Generated by `experiments/synthetic_engine.py`. Do not edit by hand.\n\n"
        f"Engine version {ENGINE_VERSION}, as fixed by "
        "[ADR-0021](../../docs/decisions/0021-synthetic-forgery-engine.md); its development is "
        "recorded in [synthetic-engine-development.md]"
        "(../../docs/synthetic-engine-development.md). Numbers only: no image is shown, and "
        "receipts of the clean folds are read by code only.\n\n"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(header + "\n".join(render_table(t) for t in tables), encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
