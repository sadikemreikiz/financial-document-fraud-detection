"""Data behind the pre-registration of Phase 5 (ADR-0029).

Writes reports/fusion/preregistration-data.md. It regenerates every number that ADR-0029 cites
from data, except those of the copy-move feasibility check (experiments/copy_move_feasibility.py)
and those it quotes from earlier reports. Its inputs:

- labels and region annotations of the training and validation splits: group sizes of the
  breakdowns, copy-move counts;
- the frozen LLM reader's stored answers for the training receipts (no request is sent) and the
  cached OCR words: the accounting rules as one fusion input, and how often an annotated edit
  lies on a value the reader located;
- system 2's stored out-of-fold document scores of the training split, from the five rounds of
  Phase 4 (no network runs): the strength a fusion model can expect, the minimum sample size of
  Riley et al. (2020), and the operating points of a review policy;
- the file headers of the training receipts: the traits of the file-metadata check of track T3.

No validation receipt is scored and nothing of the test split is read.

Usage: ``uv run python experiments/fusion_preregistration.py`` (needs the dataset, the OCR cache,
the LLM answers under artifacts/llm and the Phase 4 rounds under artifacts/learned/rounds)
"""

import math
from collections import Counter, defaultdict
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image
from sklearn.linear_model import LogisticRegression

from docfraud.cli.common import CV_FOLDS_PATH, OCR_CACHE, TRACKS, cached_ocr
from docfraud.cli.llm import FROZEN, STORE
from docfraud.data.findit import load_findit2
from docfraud.data.models import BBox, Document, Entity, Label, RegionRole, Split, Technique
from docfraud.data.splits import select
from docfraud.detectors.rules import Status, check
from docfraud.detectors.scoring import any_fail, features, out_of_fold
from docfraud.evaluation.breakdowns import rules_blind
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS
from docfraud.evaluation.detection import average_precision, choose_threshold, roc_auc
from docfraud.evaluation.folds import load_folds
from docfraud.forensics.learned.inference import QUARTER_SQUARE
from docfraud.forensics.learned.results import RESULTS, Group, RunResults
from docfraud.forensics.learned.rounds import SEEDS, run_directory
from docfraud.forensics.learned.sampler import System
from docfraud.llm.frozen import FrozenReader
from docfraud.llm.record import FrozenVersion
from docfraud.llm.run import target
from docfraud.llm.store import ResponseStore
from docfraud.markdown import Cell, Table, render_table
from docfraud.reading.fields import FieldSources, Source
from docfraud.reading.pipeline import read_document

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "fusion" / "preregistration-data.md"
ROUNDS = ROOT / "artifacts" / "learned" / "rounds"
TECHNIQUE_ORDER = (*Technique,)
ENTITY_ORDER = (*Entity,)
INPUT_COUNTS = (5, 7, 8, 18)  # baseline, fusion, fusion of track T3, one input per rule outcome
REVIEW_SHARES = (0.02, 0.05, 0.10, 0.20)

type Labels = npt.NDArray[np.bool_]
type Scores = npt.NDArray[np.float64]


def forged(documents: Sequence[Document]) -> list[Document]:
    """The forged receipts among ``documents``."""
    return [d for d in documents if d.label is Label.FORGED]


def groups_table(train: Sequence[Document], val: Sequence[Document]) -> Table:
    """Forged receipts per technique, entity and rules-blind group: the breakdowns of RQ3."""
    parts = (forged(train), forged(val))

    def count(test: Callable[[Document], bool]) -> list[Cell]:
        return [sum(test(d) for d in part) for part in parts]

    by_technique = [
        Counter(t for d in part for t in {t for r in d.modified_regions for t in r.techniques})
        for part in parts
    ]
    by_entity = [
        Counter(e for d in part for e in {r.entity for r in d.modified_regions}) for part in parts
    ]
    rows: list[list[Cell]] = [["forged receipts", *(len(part) for part in parts)]]
    rows.append(["with region annotations", *count(lambda d: bool(d.modified_regions))])
    rows.extend(
        [f"technique {t.value.upper()}", *(c[t] for c in by_technique)] for t in TECHNIQUE_ORDER
    )
    rows.extend([f"entity {e.value}", *(c[e] for c in by_entity)] for e in ENTITY_ORDER)
    rows.append(["changing only what the accounting rules cannot check", *count(rules_blind)])
    return Table(
        "Groups of forged receipts",
        ("Group", "Training split", "Validation split"),
        rows,
        note="Labels and region annotations only. A receipt counts once in each technique and "
        "entity its modified regions use; 'changing only what the accounting rules cannot "
        "check' is `breakdowns.rules_blind` (date, merchant or other text, no amount and no "
        "product).",
    )


def system2_out_of_fold(train: Sequence[Document]) -> dict[int, tuple[Labels, Scores]]:
    """System 2's out-of-fold document scores of the training split, per seed, in ``train``'s order.

    Each receipt's score comes from the round that held out its fold, as fusion uses it
    (ADR-0028).
    """
    order = {d.doc_id: i for i, d in enumerate(train)}
    out: dict[int, tuple[Labels, Scores]] = {}
    for seed in SEEDS:
        scores = np.full(len(train), np.nan)
        labels = np.zeros(len(train), dtype=bool)
        for held_out in range(5):
            path = run_directory(ROUNDS, held_out, System.SYNTHETIC, seed) / RESULTS
            results = RunResults.model_validate_json(path.read_text(encoding="utf-8"))
            for receipt in results.measured:
                if receipt.group is Group.HELD_OUT:
                    scores[order[receipt.key]] = receipt.scores[QUARTER_SQUARE]
                    labels[order[receipt.key]] = receipt.forged
        if np.isnan(scores).any():
            msg = f"system 2, seed {seed}: training receipts without a held-out score"
            raise ValueError(msg)
        out[seed] = (labels, scores)
    return out


def cox_snell_r2(y: Labels, p: Scores) -> float:
    """Cox-Snell R squared of probabilities ``p`` against the null model of the same data."""
    n, phi = y.size, float(y.mean())
    null = n * (phi * math.log(phi) + (1 - phi) * math.log(1 - phi))
    p = np.clip(p, 1e-12, 1 - 1e-12)
    fitted = float(np.sum(np.where(y, np.log(p), np.log(1 - p))))
    return 1 - math.exp(2 * (null - fitted) / n)


def one_input_out_of_fold(y: Labels, s: Scores, fold_of: npt.NDArray[np.int_]) -> Scores:
    """Probabilities of a one-input logistic regression fitted on the other folds."""
    p = np.zeros(y.size)
    for fold in np.unique(fold_of):
        held = fold_of == fold
        model = LogisticRegression().fit(s[~held, None], y[~held])
        p[held] = model.predict_proba(s[held, None])[:, 1]
    return p


def system2_table(
    runs: Mapping[int, tuple[Labels, Scores]], fold_of: npt.NDArray[np.int_]
) -> tuple[Table, float]:
    """System 2 alone on the training split, and the lowest Cox-Snell R squared over seeds."""
    rows: list[list[Cell]] = []
    r2s: list[float] = []
    for seed, (y, s) in runs.items():
        r2 = cox_snell_r2(y, one_input_out_of_fold(y, s, fold_of))
        r2s.append(r2)
        rows.append([seed, int(y.size), int(y.sum()), roc_auc(y, s), average_precision(y, s), r2])
    table = Table(
        "System 2 alone, training split, out of fold",
        ("Seed", "Receipts", "Forged", "ROC-AUC", "Average precision", "Cox-Snell R²"),
        rows,
        note="Document score 'highest mean over a square of a quarter of the text height' "
        "(ADR-0023) from the round that held each receipt out. R² is that of a one-input "
        "logistic regression fitted in the stored folds, out of fold: the anticipated R² of "
        "a fusion model in the sample-size calculation below, a conservative choice because "
        "fusion adds sources.",
    )
    return table, min(r2s)


def riley_minimum(inputs: int, r2: float, n: int, events: int) -> int:
    """Minimum sample size of Riley et al. (2020, BMJ 368:m441) for a binary outcome.

    The largest of four criteria: expected shrinkage of at least 0.9, optimism in R² of at most
    0.05 of its maximum, the overall risk within 0.05, and a mean absolute prediction error of
    at most 0.05; rounded up, as the paper does.
    """
    phi = events / n
    max_r2 = 1 - math.exp(2 * n * (phi * math.log(phi) + (1 - phi) * math.log(1 - phi)) / n)
    shrinkage = 0.9
    by_shrinkage = inputs / ((shrinkage - 1) * math.log(1 - r2 / shrinkage))
    s2 = r2 / (r2 + 0.05 * max_r2)
    by_optimism = inputs / ((s2 - 1) * math.log(1 - r2 / s2))
    by_risk = (1.96 / 0.05) ** 2 * phi * (1 - phi)
    by_error = math.exp(
        (-0.508 + 0.259 * math.log(phi) + 0.504 * math.log(inputs) - math.log(0.05)) / 0.544
    )
    return math.ceil(max(by_shrinkage, by_optimism, by_risk, by_error))


def expected_shrinkage(inputs: int, r2: float, n: int) -> float:
    """The shrinkage S that solves Riley's first criterion for the sample at hand (bisection)."""
    low, high = r2 + 1e-9, 1 - 1e-12
    for _ in range(200):
        middle = (low + high) / 2
        if (middle - 1) * math.log(1 - r2 / middle) > inputs / n:
            low = middle  # the criterion falls as S grows: S must be larger
        else:
            high = middle
    return (low + high) / 2


def sample_size_table(r2: float, n: int, events: int) -> Table:
    """How many receipts a logistic regression with a given number of inputs needs."""
    rows: list[list[Cell]] = [
        [
            k,
            riley_minimum(k, r2, n, events),
            n,
            expected_shrinkage(k, r2, n),
        ]
        for k in INPUT_COUNTS
    ]
    return Table(
        f"Minimum sample size at R² {r2:.3f}",
        ("Inputs", "Receipts needed", "Training receipts", "Expected shrinkage"),
        rows,
        note=f"Riley et al. (2020) with {events} forged of {n} receipts and the lowest R² of the "
        "three seeds above. Expected shrinkage solves the first criterion for the training "
        "split; 1 means no overfitting. 5 inputs: the score-level baseline; 7: the fusion "
        "model; 8: the fusion model of track T3; 18: one input per accounting-rule outcome.",
    )


def rules_table(
    train: Sequence[Document], fold_of: npt.NDArray[np.int_], reader: FrozenReader
) -> Table:
    """Three ways to turn the accounting rules' outcomes into one fusion input, out of fold."""
    ocr = cached_ocr(ROOT / OCR_CACHE)
    y = np.array([d.label is Label.FORGED for d in train])
    rows: list[list[Cell]] = []
    for source, name in TRACKS.items():
        readings = [read_document(d, ocr, source) for d in train]
        llm = [
            reader.read(target(d.doc_id, source, r.rows)).fields
            for d, r in zip(train, readings, strict=True)
        ]
        for reader_name, fields in (("parser v2", [r.fields for r in readings]), ("LLM", llm)):
            outcomes = [check(f) for f in fields]
            failing = np.array(
                [sum(o.status is Status.FAIL for o in oc.values()) for oc in outcomes], dtype=float
            )
            for way, scores in (
                (
                    "Phase 3 scorer (logistic regression, 12 features)",
                    out_of_fold(features(outcomes), y, fold_of),
                ),
                ("any rule fails", any_fail(outcomes)),
                ("number of failing rules", failing),
            ):
                rows.append(
                    [name, reader_name, way, roc_auc(y, scores), average_precision(y, scores)]
                )
    return Table(
        "The accounting rules as one input, training split",
        ("Track", "Fields from", "Input", "ROC-AUC", "Average precision"),
        rows,
        note="Out of fold in the stored folds where the input has fitted parameters; the other "
        "two have none. Parser v2 reproduces Phase 3 (reports/rules/accounting-rules.md); the "
        "LLM rows use the frozen reader that Phase 5 reads with (ADR-0027). Chance: ROC-AUC "
        "0.5, average precision 0.167.",
    )


def located(fs: FieldSources) -> tuple[list[Source], list[Source]]:
    """All sources of read values, and those of amounts and item numbers."""
    money: list[Source] = [s for group in fs.amounts.values() for s in group]
    for item in fs.items:
        money += [s for s in (item.quantity, item.unit_price, item.amount) if s is not None]
    for row in fs.tax_summary:
        money += [s for s in (row.rate_percent, row.taxable_amount, row.tax) if s is not None]
    other: list[Source] = [*fs.merchant, *fs.dates]
    for item in fs.items:
        other += list(item.description)
    other += [s for s in (fs.item_count, fs.quantity_total) if s is not None]
    return money + other, money


def covered_share(boxes: Sequence[BBox], width: int, height: int) -> float:
    """Share of a page covered by the union of ``boxes``."""
    mask = np.zeros((height, width), dtype=bool)
    for b in boxes:
        mask[b.y : b.y2, b.x : b.x2] = True
    return float(mask.mean())


def linking_tables(train: Sequence[Document], reader: FrozenReader) -> list[Table]:
    """How often an annotated edit lies on a value the frozen reader located, per track."""
    ocr = cached_ocr(ROOT / OCR_CACHE)
    tables: list[Table] = []
    for source, name in TRACKS.items():
        values = with_box = 0
        hits: dict[str, Counter[str]] = defaultdict(Counter)
        shares: list[float] = []
        for d in train:
            reading = read_document(d, ocr, source)
            sources = reader.read(target(d.doc_id, source, reading.rows)).extraction.sources
            everything, money = located(sources)
            values += len(everything)
            with_box += sum(s.box is not None for s in everything)
            boxes = [s.box for s in everything if s.box is not None]
            money_boxes = [s.box for s in money if s.box is not None]
            if d.label is Label.AUTHENTIC:
                shares.append(covered_share(money_boxes, reading.ocr.width, reading.ocr.height))
                continue
            for r in d.modified_regions:
                group = hits[r.entity.value if r.entity else "not annotated"]
                group["regions"] += 1
                group["on a value"] += any(r.box.overlaps(b) for b in boxes)
                group["on an amount"] += any(r.box.overlaps(b) for b in money_boxes)
        rows: list[list[Cell]] = [
            [entity, c["regions"], c["on a value"], c["on an amount"]]
            for entity, c in sorted(hits.items())
        ]
        tables.append(
            Table(
                f"Annotated edits on located values, {name}, training split",
                (
                    "Entity of the modified region",
                    "Regions",
                    "On a read value",
                    "On an amount or item number",
                ),
                rows,
                note=f"{with_box} of {values} values the frozen reader read have a box (the "
                "segment or line they are printed in). A region counts if it shares a pixel "
                "with such a box. On the "
                f"{len(shares)} authentic receipts, the boxes of amounts and item numbers cover "
                f"a median {100 * float(np.median(shares)):.1f} % of the page (90th percentile "
                f"{100 * float(np.percentile(shares, 90)):.1f} %).",
            )
        )
    return tables


def copy_move_table(train: Sequence[Document]) -> Table:
    """Forged training receipts with copy-paste inside the receipt, and with annotated sources."""
    rows: list[list[Cell]] = []
    for part, members in (
        ("design fold", [d for d in forged(train) if d.doc_id in DESIGN_FOLD_IDS]),
        ("clean folds", [d for d in forged(train) if d.doc_id not in DESIGN_FOLD_IDS]),
        ("training split", forged(train)),
    ):
        cpi = [d for d in members if any(Technique.CPI in r.techniques for r in d.modified_regions)]
        rows.append(
            [
                part,
                len(members),
                len(cpi),
                sum(any(r.role is RegionRole.SOURCE for r in d.regions) for d in cpi),
            ]
        )
    return Table(
        "Copy-paste inside the receipt (CPI), training split",
        ("Part", "Forged receipts", "With CPI", "With CPI and an annotated source"),
        rows,
        note="Labels and region annotations only.",
    )


def policy_table(runs: Mapping[int, tuple[Labels, Scores]]) -> Table:
    """What the highest-scoring shares of the training split hold, ranked by system 2 alone."""
    rows: list[list[Cell]] = []
    for seed, (y, s) in runs.items():
        order = np.argsort(-s, kind="stable")
        positives = int(y.sum())
        for share in REVIEW_SHARES:
            k = round(share * y.size)
            if s[order[k - 1]] == s[order[k]]:
                msg = f"seed {seed}: tie at the cut-off of the highest {share:.0%}"
                raise ValueError(msg)
            top = int(y[order[:k]].sum())
            rows.append([seed, f"highest {share:.0%}", k, top, k - top, top / k, top / positives])
        flagged = s >= choose_threshold(y, s)
        top = int(y[flagged].sum())
        k = int(flagged.sum())
        rows.append(
            [seed, f"max-F1 point ({k / y.size:.0%})", k, top, k - top, top / k, top / positives]
        )
    return Table(
        "Operating points of system 2 alone, training split, out of fold",
        (
            "Seed",
            "Receipts sent to review",
            "Receipts",
            "Forged",
            "Authentic",
            "Precision",
            "Recall",
        ),
        rows,
        note="A stand-in for the fusion model, which is not built yet. The max-F1 point is "
        "chosen on these same scores, so its row is optimistic.",
    )


TRAITS: dict[str, Callable[[Image.Image], bool]] = {
    "alpha channel": lambda im: im.mode in {"LA", "RGBA", "PA"} or "transparency" in im.info,
    "ICC colour profile": lambda im: "icc_profile" in im.info,
    "EXIF": lambda im: "exif" in im.info or "Raw profile type exif" in im.info,
    "XMP": lambda im: any(
        k in im.info for k in ("xmp", "XML:com.adobe.xmp", "Raw profile type xmp")
    ),
    "creation date": lambda im: "date:create" in im.info,
}
"""Traits an editing program leaves in a PNG file, as the metadata check of track T3 reads them."""

OTHER_PROPERTIES: dict[str, Callable[[Image.Image], bool]] = {
    "no gamma chunk": lambda im: "gamma" not in im.info,
    "resolution (dpi) chunk": lambda im: "dpi" in im.info,
}
"""Properties also seen in the headers, not editing traits: kept out of the check."""


def metadata_tables(train: Sequence[Document]) -> list[Table]:
    """File-container traits of the training receipts by label."""
    labels = (Label.AUTHENTIC, Label.FORGED)
    found: dict[str, Counter[Label]] = defaultdict(Counter)
    totals: dict[Label, Counter[int]] = {label: Counter() for label in labels}
    formats: Counter[str] = Counter()
    for d in train:
        label = Label.FORGED if d.label is Label.FORGED else Label.AUTHENTIC
        with Image.open(d.image_path) as image:
            formats[str(image.format)] += 1
            traits = {name: test(image) for name, test in TRAITS.items()}
            others = {name: test(image) for name, test in OTHER_PROPERTIES.items()}
        for name, present in {**traits, **others}.items():
            found[name][label] += present
        found["any editing trait"][label] += any(traits.values())
        totals[label][sum(traits.values())] += 1
    size = Counter(Label.FORGED if d.label is Label.FORGED else Label.AUTHENTIC for d in train)
    order = [*TRAITS, "any editing trait", *OTHER_PROPERTIES]
    traits_table = Table(
        "File-container traits, training split",
        ("Trait", f"Authentic ({size[Label.AUTHENTIC]})", f"Forged ({size[Label.FORGED]})"),
        [[name, found[name][Label.AUTHENTIC], found[name][Label.FORGED]] for name in order],
        note="Read from the PNG headers with Pillow; no pixel is decoded. File formats: "
        + ", ".join(f"{k} {v}" for k, v in sorted(formats.items()))
        + ". The first five rows are the editing traits of the T3 check; the last two are "
        "shown for completeness and are not used.",
    )
    most = max(max(c) for c in totals.values())
    count_table = Table(
        "Number of editing traits per receipt, training split",
        ("Label", *(str(k) for k in range(most + 1))),
        [[label.value, *(totals[label][k] for k in range(most + 1))] for label in labels],
    )
    return [traits_table, count_table]


def main() -> None:
    """Compute every table from the training split and write the report."""
    documents = load_findit2(ROOT / "data" / "raw" / "findit2")
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    val = select(documents, Split.VAL)
    folds = load_folds(ROOT / CV_FOLDS_PATH)
    fold_of = np.array([folds.folds[d.doc_id] for d in train])
    version = FrozenVersion.model_validate_json((ROOT / FROZEN).read_text(encoding="utf-8"))
    reader = FrozenReader(
        version,
        (ROOT / version.instruction).read_text(encoding="utf-8"),
        ResponseStore(ROOT / STORE),
    )
    runs = system2_out_of_fold(train)
    strength, r2 = system2_table(runs, fold_of)
    n, events = len(train), len(forged(train))
    tables = [
        groups_table(train, val),
        strength,
        sample_size_table(r2, n, events),
        rules_table(train, fold_of, reader),
        *linking_tables(train, reader),
        copy_move_table(train),
        policy_table(runs),
        *metadata_tables(train),
    ]
    header = (
        "# Phase 5: data behind the pre-registration\n\n"
        "> Generated by `experiments/fusion_preregistration.py`. Do not edit by hand.\n\n"
        "The numbers that [ADR-0029](../../docs/decisions/0029-fusion-preregistration.md) "
        "cites from data, except the copy-move feasibility check "
        "([copy-move-feasibility.md](copy-move-feasibility.md)). Computed from labels and "
        "region annotations, the frozen LLM reader's stored answers, system 2's stored "
        "out-of-fold scores and the file headers of the training split, and from the labels "
        "of the validation split. No fusion model is fitted, no validation receipt is scored "
        "and nothing of the test split is read.\n\n"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(header + "\n".join(render_table(t) for t in tables), encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
