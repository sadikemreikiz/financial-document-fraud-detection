"""RQ4 for audit reports: does the report's first region point at the forgery? (ADR-0029, 12).

The first region is the location of the evidence with the largest positive contribution
(:meth:`~docfraud.reporting.assessment.Assessment.first_region`). It hits if any of its boxes
shares a pixel with an annotated modified rectangle or copy source (ADR-0015, section 3); its
random baseline places one box of each size on the receipt's text, independently, and counts a
hit if any of them hits (ADR-0030). A report without a located piece of evidence pushing towards
forgery is a miss with a baseline of 0.
"""

from collections.abc import Sequence
from dataclasses import dataclass

from docfraud.data.models import BBox, Document
from docfraud.evaluation.localisation import annotated_rectangles, any_box_baseline, receipt_rng
from docfraud.reporting.assessment import Assessment


@dataclass(frozen=True, slots=True)
class FirstRegion:
    """The report's first region on one forged receipt, judged against the annotations."""

    located: bool
    hit: bool
    baseline: float


def first_region_hit(
    assessment: Assessment, document: Document, page: BBox, word_boxes: Sequence[BBox]
) -> FirstRegion:
    """Judge the first region of a receipt's report against its annotated edits."""
    boxes = assessment.first_region()
    if not boxes:
        return FirstRegion(located=False, hit=False, baseline=0.0)
    modified, sources = annotated_rectangles(document)
    targets = [*modified, *sources]
    hit = any(box.overlaps(t) for box in boxes for t in targets)
    rng = receipt_rng(document.doc_id)
    return FirstRegion(True, hit, any_box_baseline(boxes, page, word_boxes, targets, rng))
