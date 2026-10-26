"""Development evidence for copy-move (ADR-0029, section 4; ADR-0030), on the design fold.

Writes reports/fusion/copy-move-development.md, the evidence behind the choices recorded in
docs/copy-move-development.md:

- every candidate tried on the real receipts of the design fold: which blocks count as
  informative and the block side, with the average precision of the document score, how often
  the pair lies on an annotated edit, the scores of authentic receipts and the size check of
  ADR-0015 (section 6) and ADR-0017;
- for the chosen detector, each receipt with copy-paste inside it: whether its pair hits, which
  region lies on the copy and which on its annotated source, and whether a break in the grey
  levels across a region's border tells the pasted region (ADR-0030);
- the chosen detector on synthetic copy-paste forgeries of design-fold bases, by the engine's
  variants (ADR-0021).

Only the design fold is read, which the viewing lock checks; numbers are optimistic. The time
per receipt is printed, not reported, because it depends on the machine.

Usage: ``uv run python experiments/copy_move_development.py``
"""

import time
from collections import Counter
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt

from docfraud.cli.common import CV_FOLDS_PATH, OCR_CACHE, cached_ocr
from docfraud.data.benign_marks import Mark, final_marks, load_rounds, marks_plan
from docfraud.data.findit import load_findit2
from docfraud.data.models import BBox, Document, Label, RegionRole, Technique
from docfraud.data.normalize import load_grayscale
from docfraud.data.observation import ASSISTANT
from docfraud.detectors.copy_move import (
    BLOCK_DIVISOR,
    INK_CONTRAST,
    MIN_BLOCK,
    UNSATURATED_SHARE,
    WHITE,
    CopyMovePair,
    duplicate_pair,
)
from docfraud.evaluation.design_fold import DESIGN_FOLD, DESIGN_FOLD_IDS, require_design_fold
from docfraud.evaluation.detection import average_precision, roc_auc
from docfraud.evaluation.folds import load_folds
from docfraud.evaluation.localisation import size_dependence
from docfraud.evaluation.mark_alarms import check_marks
from docfraud.forensics.learned.text_scale import text_height
from docfraud.markdown import Cell, Table, render_table
from docfraud.synthesis.engine import Stream, SyntheticRegion
from docfraud.synthesis.rounds import round_engine

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "fusion" / "copy-move-development.md"
FONTS = ROOT / "artifacts" / "fonts"
MARK_RECORDS = ROOT / "annotations" / "findit2" / "benign-marks"
SYNTHETIC_PER_BASE = 3  # forgeries 0 to 2 of each design-fold base, for more copies per variant

type Page = npt.NDArray[np.uint8]


@dataclass(frozen=True, slots=True)
class Candidate:
    """A way to find duplicated regions, as tried on the design fold."""

    name: str
    contrast: int
    unsaturated_share: float
    divisor: int


CANDIDATES = (
    Candidate("contrast only", INK_CONTRAST, 0.0, BLOCK_DIVISOR),
    Candidate("unsaturated only", 0, UNSATURATED_SHARE, BLOCK_DIVISOR),
    Candidate("both, text height / 3", INK_CONTRAST, UNSATURATED_SHARE, 3),
    Candidate("both, text height / 4 (chosen)", INK_CONTRAST, UNSATURATED_SHARE, BLOCK_DIVISOR),
    Candidate("both, text height / 5", INK_CONTRAST, UNSATURATED_SHARE, 5),
)
CHOSEN = CANDIDATES[3]


@dataclass(frozen=True, slots=True)
class Receipt:
    """A design-fold receipt with what copy-move is measured against."""

    document: Document
    page: Page
    height: float  # text height
    copies: tuple[BBox, ...]  # modified regions with copy-paste inside the receipt
    sources: tuple[BBox, ...]  # annotated copy sources
    edits: tuple[BBox, ...]  # every modified region
    words: tuple[BBox, ...]  # OCR word boxes

    @property
    def forged(self) -> bool:
        """Whether the receipt is forged."""
        return self.document.label is Label.FORGED


@dataclass(frozen=True, slots=True)
class Found:
    """A candidate's pair on one receipt and its document score."""

    pair: CopyMovePair | None
    score: float
    seconds: float


def find(receipt: Receipt, candidate: Candidate) -> Found:
    """Run one candidate on one receipt."""
    side = max(MIN_BLOCK, round(receipt.height / candidate.divisor))
    start = time.perf_counter()
    pair = duplicate_pair(
        receipt.page,
        side,
        contrast=candidate.contrast,
        unsaturated_share=candidate.unsaturated_share,
    )
    seconds = time.perf_counter() - start
    return Found(pair, pair.area / receipt.height**2 if pair else 0.0, seconds)


def hits(pair: CopyMovePair | None, boxes: Sequence[BBox]) -> bool:
    """Whether either region of a pair shares a pixel with any of ``boxes`` (ADR-0030)."""
    return pair is not None and any(r.overlaps(b) for r in (pair.first, pair.second) for b in boxes)


def candidates_table(receipts: Sequence[Receipt], found: dict[str, list[Found]]) -> Table:
    """Every candidate tried, on the real receipts of the design fold."""
    y = np.array([r.forged for r in receipts])
    cpi = np.array([bool(r.copies) for r in receipts])
    authentic = ~y
    pixels = [r.page.size for r in receipts]
    rows: list[list[Cell]] = []
    for candidate in CANDIDATES:
        results = found[candidate.name]
        s = np.array([f.score for f in results])
        keep = authentic | cpi
        with_pair = authentic & (s > 0)
        overall = size_dependence(
            s[authentic].tolist(), [p for p, a in zip(pixels, authentic, strict=True) if a]
        )
        evidence = size_dependence(
            s[with_pair].tolist(), [p for p, e in zip(pixels, with_pair, strict=True) if e]
        )
        found_on_copies = sum(
            hits(f.pair, r.edits + r.sources)
            for f, r in zip(results, receipts, strict=True)
            if r.copies
        )
        rows.append(
            [
                candidate.name,
                average_precision(y, s),
                average_precision(cpi[keep], s[keep]),
                roc_auc(cpi[keep], s[keep]),
                f"{found_on_copies} of {int(cpi.sum())}",
                float(s[authentic].max()),
                f"{int(with_pair.sum())} of {int(authentic.sum())}",
                f"{overall.value:.3f} ({overall.low:.3f} to {overall.high:.3f})",
                f"{evidence.value:.3f} ({evidence.low:.3f} to {evidence.high:.3f})",
            ]
        )
    return Table(
        "Candidates tried on the design fold",
        (
            "Candidate",
            "AP, all forged",
            "AP, copy-paste inside vs authentic",
            "ROC-AUC, copy-paste inside vs authentic",
            "Copy-paste receipts whose pair hits",
            "Highest authentic score",
            "Authentic receipts with a pair",
            "Size check, all authentic",
            "Size check, authentic with a pair",
        ),
        rows,
        note=f"{int(y.sum())} forged and {int(authentic.sum())} authentic receipts; chance "
        f"{y.mean():.3f} for all forged and {cpi[authentic | cpi].mean():.3f} for copy-paste "
        "inside against authentic. 'Contrast only' counts a block if its darkest and lightest "
        f"pixels differ by at least {INK_CONTRAST} grey levels; 'unsaturated only' if at least "
        f"{UNSATURATED_SHARE:.0%} of its pixels lie strictly between 0 and 255 and it is not "
        "flat; 'both' requires both. The block side is the text height divided by the number "
        f"given, at least {MIN_BLOCK} px. A pair hits if either region shares a pixel with a "
        "modified rectangle or an annotated copy source (ADR-0030). Size check: Spearman's rank "
        "correlation of score and pixels with its 95 % interval (ADR-0015, section 6); an "
        "interval that excludes zero fails.",
    )


def false_pairs_table(receipts: Sequence[Receipt], found: dict[str, list[Found]]) -> Table:
    """The highest-scoring authentic receipts of each candidate: what their pairs are made of."""
    rows: list[list[Cell]] = []
    for candidate in CANDIDATES:
        scored = [
            (f, r)
            for f, r in zip(found[candidate.name], receipts, strict=True)
            if not r.forged and f.pair is not None
        ]
        for f, r in sorted(scored, key=lambda x: (-x[0].score, x[1].document.doc_id))[:3]:
            pair = f.pair
            if pair is None:
                continue
            region = r.page[pair.first.y : pair.first.y2, pair.first.x : pair.first.x2]
            rows.append(
                [
                    candidate.name,
                    r.document.doc_id,
                    f.score,
                    f"{pair.second.y - pair.first.y}, {pair.second.x - pair.first.x}",
                    f"{r.page.shape[1]} x {r.page.shape[0]}",
                    float((region == WHITE).mean()),
                ]
            )
    return Table(
        "The three highest-scoring authentic receipts of each candidate",
        (
            "Candidate",
            "Receipt",
            "Score",
            "Offset (rows, columns)",
            "Page (width x height)",
            "Share of the first region at 255",
        ),
        rows,
        note="The offset is the shift from the first region of the pair to the second. A region "
        "almost entirely at 255 is paper clipped to white with a speck; offsets that are "
        "multiples of 8 px follow the grid of JPEG compression.",
    )


def seam(page: Page, box: BBox) -> float:
    """Mean absolute grey-level difference across the border of a box, inside against outside."""
    p = page.astype(np.int16)
    height, width = p.shape
    parts = []
    if box.y > 0:
        parts.append(np.abs(p[box.y, box.x : box.x2] - p[box.y - 1, box.x : box.x2]))
    if box.y2 < height:
        parts.append(np.abs(p[box.y2 - 1, box.x : box.x2] - p[box.y2, box.x : box.x2]))
    if box.x > 0:
        parts.append(np.abs(p[box.y : box.y2, box.x] - p[box.y : box.y2, box.x - 1]))
    if box.x2 < width:
        parts.append(np.abs(p[box.y : box.y2, box.x2 - 1] - p[box.y : box.y2, box.x2]))
    return float(np.concatenate(parts).mean())


def sides_tables(receipts: Sequence[Receipt], found: Sequence[Found]) -> list[Table]:
    """For the chosen detector: which region of each pair is the copy, and can pixels tell."""
    rows: list[list[Cell]] = []
    counts: Counter[str] = Counter()
    for receipt, result in zip(receipts, found, strict=True):
        if not receipt.copies:
            continue
        pair = result.pair
        hit = hits(pair, receipt.edits + receipt.sources)
        if pair is None or not hit:
            rows.append([receipt.document.doc_id, result.score, "no", None, None, None, None])
            continue
        regions = {"first": pair.first, "second": pair.second}
        on_copy = [k for k, b in regions.items() if any(b.overlaps(c) for c in receipt.copies)]
        on_source = [k for k, b in regions.items() if any(b.overlaps(c) for c in receipt.sources)]
        seams = {k: seam(receipt.page, b) for k, b in regions.items()}
        stronger = max(seams, key=lambda k: seams[k])
        if len(on_copy) == 1:
            counts["one region on a copy"] += 1
            counts["seam picks it"] += stronger == on_copy[0]
            counts["it is the second"] += on_copy[0] == "second"
        else:
            counts["both regions on copies"] += 1
        rows.append(
            [
                receipt.document.doc_id,
                result.score,
                "yes",
                ", ".join(on_copy) or "neither",
                ", ".join(on_source) or "neither",
                f"{seams['first']:.1f} / {seams['second']:.1f}",
                stronger,
            ]
        )
    per_receipt = Table(
        "Receipts with copy-paste inside, chosen detector",
        (
            "Receipt",
            "Score",
            "Pair hits",
            "Region on a copy",
            "Region on an annotated source",
            "Seam, first / second",
            "Stronger seam",
        ),
        rows,
        note="The first region is the upper one (on the same row, the left one). Seam: the mean "
        "absolute grey-level difference across the region's border; a pasted rectangle might "
        "break the paper there more than its source does.",
    )
    summary = Table(
        "Can pixels tell the pasted region?",
        ("Quantity", "Receipts"),
        [
            ["pairs that hit, both regions on annotated copies", counts["both regions on copies"]],
            ["pairs that hit, one region on an annotated copy", counts["one region on a copy"]],
            ["of these, the stronger seam is on the copy", counts["seam picks it"]],
            ["of these, the copy is the second region", counts["it is the second"]],
        ],
    )
    return [per_receipt, summary]


def pasted_exactly(region: SyntheticRegion) -> bool:
    """A copy pasted as a rectangle with its paper, neither scaled nor rotated."""
    return (
        region.variants["paste"] == "rectangle" and region.variants["scale and rotation"] == "none"
    )


def synthetic_tables(documents: Sequence[Document]) -> list[Table]:
    """The chosen detector on synthetic copy-paste of design-fold bases, by how it was pasted.

    A copy counts as found if a region of the pair shares a pixel with the pasted copy itself,
    not only with its source, which other copies on the same forgery may share.
    """
    ocr = cached_ocr(ROOT / OCR_CACHE)
    engine = round_engine(documents, load_folds(ROOT / CV_FOLDS_PATH), DESIGN_FOLD, ocr, FONTS)
    bases = engine.pools[Stream.DEVELOPMENT]
    require_design_fold(bases)
    forgeries: Counter[tuple[bool, bool]] = Counter()  # (has an exact paste, found)
    single: Counter[tuple[str, str, bool]] = (
        Counter()
    )  # forgeries with one copy: (paste, content, found)
    for base in bases:
        receipt = engine.receipts(base)
        height = text_height(
            [t.box for t in ocr.recognize(receipt.page).tokens], *receipt.page.shape
        )
        side = max(MIN_BLOCK, round(height / CHOSEN.divisor))
        for index in range(SYNTHETIC_PER_BASE):
            forgery = engine.forge(base, index, Stream.DEVELOPMENT)
            copies = [r for r in forgery.record.regions if r.technique is Technique.CPI]
            if not copies:
                continue
            pair = duplicate_pair(forgery.page, side)
            found = [hits(pair, [r.box]) for r in copies]
            forgeries[(any(map(pasted_exactly, copies)), False)] += 1
            forgeries[(any(map(pasted_exactly, copies)), True)] += any(found)
            if len(copies) == 1:
                v = copies[0].variants
                content = "as it was" if v["scale and rotation"] == "none" else "scaled or rotated"
                single[(v["paste"], content, False)] += 1
                single[(v["paste"], content, True)] += found[0]
    receipt_rows: list[list[Cell]] = [
        [label, f"{forgeries[(exact, True)]} of {forgeries[(exact, False)]}"]
        for exact, label in (
            (True, "at least one copy pasted as a rectangle, as it was"),
            (False, "only copies pasted as ink alone, or scaled or rotated"),
        )
    ]
    keys = sorted({k[:2] for k in single})
    single_rows: list[list[Cell]] = [
        [*k, f"{single[(*k, True)]} of {single[(*k, False)]}"] for k in keys
    ]
    note = (
        f"Forgeries 0 to {SYNTHETIC_PER_BASE - 1} of each of the {len(bases)} authentic "
        "design-fold receipts (ADR-0021, development stream) that hold copy-paste inside the "
        "receipt. A copy is found if a region of the pair shares a pixel with the pasted copy "
        "itself. The detector reports one pair per receipt, so of several copies on a forgery "
        "at most one is found."
    )
    return [
        Table(
            "Synthetic forgeries with copy-paste inside, chosen detector",
            ("Forgeries with", "Pair on one of their copies"),
            receipt_rows,
            note=note,
        ),
        Table(
            "Synthetic forgeries with a single copy, by how it was pasted",
            ("Paste", "Pasted content", "Copy found"),
            single_rows,
        ),
    ]


def marks_table(
    receipts: Sequence[Receipt], found: Sequence[Found], marks: dict[str, tuple[Mark, ...]]
) -> Table:
    """Whether the chosen detector's pair falls on a benign mark (benign marks guide)."""
    rows: list[list[Cell]] = []
    authentic = sorted(f.score for f, r in zip(found, receipts, strict=True) if not r.forged)
    listed: list[str] = []
    for label in (Label.AUTHENTIC, Label.FORGED):
        flagged = [
            (r, f)
            for r, f in zip(receipts, found, strict=True)
            if r.document.label is label and r.document.doc_id in marks
        ]
        on_mark = 0
        baselines: list[float] = []
        for r, f in flagged:
            pair = f.pair
            if pair is None:
                continue
            page = BBox(x=0, y=0, width=r.page.shape[1], height=r.page.shape[0])
            checks = [
                check_marks(
                    r.document.doc_id,
                    label,
                    region,
                    page,
                    r.words,
                    marks[r.document.doc_id],
                    evidence=True,
                    score=f.score,
                )
                for region in (pair.first, pair.second)
            ]
            hit = any(c.on_mark for c in checks)
            on_mark += hit
            baselines.append(1 - (1 - checks[0].baseline) * (1 - checks[1].baseline))
            if hit and label is Label.AUTHENTIC:
                rank = len(authentic) - authentic.index(f.score)
                listed.append(f"{r.document.doc_id} (score {f.score:.3f}, rank {rank})")
        rows.append(
            [
                label.value,
                len(flagged),
                len(baselines),
                on_mark,
                float(np.mean(baselines)) if baselines else None,
            ]
        )
    return Table(
        "The pair on a benign mark, chosen detector",
        ("Flagged receipts", "Count", "With a pair", "A region on a mark", "Mean random baseline"),
        rows,
        note="Flagged: the dataset marks a handwritten or digital annotation. On a mark: either "
        "region of the pair shares a pixel with a final box of the benign marks guide. Receipts "
        "without a pair have no region. Baseline: for each region, the share of 1,000 boxes of "
        "its size centred on OCR text that overlap a mark (ADR-0015), combined as the chance "
        "that either does (ADR-0030). Authentic receipts on a mark, with their rank among "
        f"{len(authentic)} authentic scores: " + (", ".join(listed) or "none") + ".",
    )


def main() -> None:
    """Run every candidate on the design fold and write the report."""
    documents = load_findit2(ROOT / "data" / "raw" / "findit2")
    design = sorted((d for d in documents if d.doc_id in DESIGN_FOLD_IDS), key=lambda d: d.doc_id)
    require_design_fold([d.doc_id for d in design])
    ocr = cached_ocr(ROOT / OCR_CACHE)
    receipts: list[Receipt] = []
    for d in design:
        page = load_grayscale(d.image_path)
        words = [t.box for t in ocr.recognize(page).tokens]
        receipts.append(
            Receipt(
                d,
                page,
                text_height(words, *page.shape),
                tuple(r.box for r in d.modified_regions if Technique.CPI in r.techniques),
                tuple(r.box for r in d.regions if r.role is RegionRole.SOURCE),
                tuple(r.box for r in d.modified_regions),
                tuple(words),
            )
        )
    found = {c.name: [find(r, c) for r in receipts] for c in CANDIDATES}
    final = final_marks(marks_plan(design), *load_rounds(MARK_RECORDS, ASSISTANT, "emre"))
    marks = {doc_id: tuple(receipt.marks) for doc_id, receipt in final.marks.items()}
    seconds = [f.seconds for f in found[CHOSEN.name]]
    print(f"chosen detector: median {np.median(seconds):.2f} s, at most {max(seconds):.1f} s")  # noqa: T201
    tables = [
        candidates_table(receipts, found),
        false_pairs_table(receipts, found),
        *sides_tables(receipts, found[CHOSEN.name]),
        marks_table(receipts, found[CHOSEN.name], marks),
        *synthetic_tables(documents),
    ]
    header = (
        "# Copy-move development on the design fold\n\n"
        "> Generated by `experiments/copy_move_development.py`. Do not edit by hand.\n\n"
        "The evidence behind [the development record](../../docs/copy-move-development.md) and "
        "[ADR-0030](../../docs/decisions/0030-copy-move-reports-a-pair.md). Only receipts of the "
        "design fold are read; its numbers are optimistic, because the candidates were tried "
        "on these receipts.\n\n"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(header + "\n".join(render_table(t) for t in tables), encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
