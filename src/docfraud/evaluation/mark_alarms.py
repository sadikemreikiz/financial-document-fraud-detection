"""Whether the top region of a map falls on a benign mark (benign marks guide, step 4.2).

In development, on the design fold only, a system's highest-scoring region on a flagged receipt
counts as on a benign mark if it shares at least one pixel with a final mark box
(``docs/benign-marks-guide.md``). This is how a system's false alarms are read during
development: an authentic receipt that scores high because of handwriting fails differently
from one that scores high for no visible reason. On the clean folds there are no boxes, and the
false-positive rate is reported by flag (evaluation protocol, section 7).

Nothing here views an image: the functions receive regions, boxes and scores.
"""

from collections.abc import Iterable, Sequence
from dataclasses import dataclass

import numpy as np

from docfraud.data.benign_marks import Mark
from docfraud.data.models import BBox, Label
from docfraud.evaluation.localisation import baseline_hit_rate, receipt_rng
from docfraud.markdown import Cell, Table


@dataclass(frozen=True, slots=True)
class MarkCheck:
    """The top region of a system's map on one flagged receipt, against its mark boxes."""

    doc_id: str
    label: Label
    evidence: bool  # whether the map is not constant; without evidence the region is the page
    kinds: tuple[str, ...]  # kinds of the marks the region overlaps, sorted
    baseline: float  # share of random boxes of the region's size on the text that overlap a mark
    score: float  # the system's document score

    @property
    def on_mark(self) -> bool:
        """Whether the region overlaps at least one mark box."""
        return bool(self.kinds)


def check_marks(  # noqa: PLR0913, PLR0917 - one argument per measured quantity
    doc_id: str,
    label: Label,
    region: BBox,
    page: BBox,
    word_boxes: Iterable[BBox],
    marks: Sequence[Mark],
    *,
    evidence: bool,
    score: float,
) -> MarkCheck:
    """Compare the top region of one flagged receipt with its marks and with chance.

    The baseline places random boxes as the localisation guard does (ADR-0015), from the
    receipt's own generator; a receipt without marks has a baseline of 0.
    """
    boxes = [mark.box for mark in marks]
    baseline = (
        baseline_hit_rate(region, page, word_boxes, boxes, receipt_rng(doc_id)) if boxes else 0.0
    )
    kinds = tuple(sorted({m.kind.value for m in marks if region.overlaps(m.box)}))
    return MarkCheck(doc_id, label, evidence, kinds, baseline, score)


def mark_tables(checks: Sequence[MarkCheck], authentic_scores: Sequence[float]) -> list[Table]:
    """The share of top regions on a mark, and the authentic receipts where it happens.

    Args:
        checks: One check per flagged receipt.
        authentic_scores: The document scores of every authentic receipt measured, against
            which an authentic receipt's score is ranked.
    """
    rows: list[list[Cell]] = []
    for label in (Label.AUTHENTIC, Label.FORGED):
        chosen = [c for c in checks if c.label is label]
        measured = [c for c in chosen if c.evidence]
        rows.append(
            [
                label.value,
                len(chosen),
                len(measured),
                sum(c.on_mark for c in measured),
                float(np.mean([c.baseline for c in measured])) if measured else None,
            ]
        )
    summary = Table(
        "Top region on a benign mark",
        [
            "Flagged receipts",
            "Count",
            "With evidence",
            "Top region on a mark",
            "Mean random baseline",
        ],
        rows,
        "Flagged: the dataset marks a handwritten or digital annotation. On a mark: the top "
        "region of ADR-0015 shares a pixel with a final box of the benign marks guide. Receipts "
        "without evidence have the whole page as top region and are left out. Baseline: the "
        "share of 1,000 boxes of the same size, centred on OCR text, that overlap a mark. "
        "Design fold: optimistic.",
    )
    scores = np.asarray(authentic_scores, dtype=np.float64)
    on_mark = sorted(
        (c for c in checks if c.label is Label.AUTHENTIC and c.evidence and c.on_mark),
        key=lambda c: (-c.score, c.doc_id),
    )
    listing = Table(
        "Authentic receipts whose top region is on a mark",
        ["Receipt", "Marks overlapped", "Document score", f"Rank among {len(scores)} authentic"],
        [
            [c.doc_id, ", ".join(c.kinds), c.score, 1 + int((scores > c.score).sum())]
            for c in on_mark
        ],
        "Rank 1 is the highest document score among the authentic receipts of the design fold, "
        "that is, the first false alarm at any threshold.",
    )
    return [summary, listing]
