"""The analysis of the benign-mark boxes, as the guide fixes it.

``docs/benign-marks-guide.md`` (section "Analysis, fixed now") prescribes what is reported, from
the final marks of every flagged receipt of the design fold: coverage, the owner's check per
round, the kinds of mark, whether they touch print, their size and position on the page, the
dataset's flags set against the marks found, and marks that overlap an annotated edit of a
forged receipt. Labels and flags are joined here; the check page never showed them.
"""

from collections import Counter
from collections.abc import Callable, Iterable, Mapping, Sequence

import numpy as np

from docfraud.data.benign_marks import (
    HAND_KINDS,
    CheckLog,
    FinalMarks,
    Mark,
    MarkKind,
    Verdict,
)
from docfraud.data.models import BBox, Document, Label
from docfraud.data.observation import Presence
from docfraud.markdown import Cell, Table, render_table

_THIRDS = ("top", "middle", "bottom")


def _flags(document: Document) -> str:
    if document.handwritten_annotation and document.digital_annotation:
        return "handwriting and digital note"
    return "handwriting" if document.handwritten_annotation else "digital note"


def _percent(share: float) -> str:
    return f"{100 * share:.2f} %"


def _third(mark: Mark, page: BBox) -> str:
    centre = mark.box.y + mark.box.height / 2
    return _THIRDS[min(2, int(3 * centre / page.height))]


def build_report(
    documents: Iterable[Document],
    plan: Sequence[str],
    final: FinalMarks,
    checks: Sequence[CheckLog],
    pages: Mapping[str, BBox],
) -> list[Table]:
    """Every table the guide prescribes, in the order of the report."""
    by_id = {d.doc_id: d for d in documents}
    marks = [(doc_id, mark) for doc_id in plan for mark in final.marks[doc_id].marks]
    kinds = [k for k in MarkKind if any(m.kind is k for _, m in marks)]
    return [
        _coverage(by_id, plan, final),
        _checks(plan, final, checks),
        _kinds(marks, kinds),
        _touching(marks, kinds),
        _sizes(marks, kinds, pages),
        _positions(marks, kinds, pages),
        _flags_against_marks(by_id, plan, final),
        _forged(by_id, plan, final),
    ]


def _coverage(by_id: Mapping[str, Document], plan: Sequence[str], final: FinalMarks) -> Table:
    groups: dict[tuple[str, str], list[str]] = {}
    for doc_id in plan:
        key = (_flags(by_id[doc_id]), by_id[doc_id].label or "unknown")
        groups.setdefault(key, []).append(doc_id)
    rows: list[list[Cell]] = []
    for (flags, label), receipts in sorted(groups.items()):
        found = [final.marks[d].marks for d in receipts]
        rows.append(
            [flags, str(label), len(receipts), sum(bool(m) for m in found), sum(map(len, found))]
        )
    all_found = [final.marks[d].marks for d in plan]
    rows.append(["all", "", len(plan), sum(bool(m) for m in all_found), sum(map(len, all_found))])
    return Table(
        "Receipts and boxes",
        ["Flags in the dataset", "Label", "Receipts", "With at least one mark", "Boxes"],
        rows,
        f"The flagged receipts of the design fold, boxed by the assistant and checked by the "
        f"owner in {final.rounds} round(s). The final marks of a receipt are those of the last "
        "round in which it was boxed.",
    )


def _checks(plan: Sequence[str], final: FinalMarks, checks: Sequence[CheckLog]) -> Table:
    rows: list[list[Cell]] = []
    for number, check in enumerate(checks, start=1):
        verdicts = Counter(c.verdict for c in check.receipts.values())
        times = sorted(c.checked_at for c in check.receipts.values())
        minutes = (times[-1] - times[0]).total_seconds() / 60 if times else 0.0
        rows.append(
            [
                number,
                len(check.receipts),
                *(verdicts[v] for v in Verdict),
                sum(final.accepted_in[d] == number for d in plan),
                f"{minutes:.1f}",
            ]
        )
    disputed = sorted(d for d in plan if final.accepted_in[d] is None)
    return Table(
        "The owner's check",
        [
            "Round",
            "Receipts checked",
            "Correct",
            "Something missing",
            "Wrong box",
            "Both",
            "Accepted",
            "Minutes from first to last answer",
        ],
        rows,
        f"Disputed after the last round: {', '.join(disputed) if disputed else 'none'}. The check "
        "is a review of the assistant's boxes, not an independent annotation; whether a mark "
        "touches print was not checked.",
    )


def _kinds(marks: Sequence[tuple[str, Mark]], kinds: Sequence[MarkKind]) -> Table:
    rows: list[list[Cell]] = [
        [k.value, sum(m.kind is k for _, m in marks), len({d for d, m in marks if m.kind is k})]
        for k in kinds
    ]
    rows.append(["all", len(marks), len({d for d, _ in marks})])
    return Table("Kinds of mark", ["Kind", "Marks", "Receipts with at least one"], rows)


def _touching(marks: Sequence[tuple[str, Mark]], kinds: Sequence[MarkKind]) -> Table:
    rows: list[list[Cell]] = []
    for kind in [*kinds, None]:
        chosen = [m for _, m in marks if kind is None or m.kind is kind]
        counts = Counter(m.touches_print for m in chosen)
        rows.append([kind.value if kind else "all", len(chosen), *(counts[p] for p in Presence)])
    return Table(
        "Marks touching print",
        ["Kind", "Marks", *(p.value for p in Presence)],
        rows,
        "Touching means that ink of the mark lies on or crosses printed characters.",
    )


def _sizes(
    marks: Sequence[tuple[str, Mark]], kinds: Sequence[MarkKind], pages: Mapping[str, BBox]
) -> Table:
    rows: list[list[Cell]] = []
    for kind in [*kinds, None]:
        chosen = [(d, m) for d, m in marks if kind is None or m.kind is kind]
        shares = np.array([m.box.area / pages[d].area for d, m in chosen])
        heights = np.array([m.box.height for _, m in chosen])
        rows.append(
            [
                kind.value if kind else "all",
                len(chosen),
                _percent(float(np.median(shares))),
                _percent(float(np.percentile(shares, 90))),
                _percent(float(shares.max())),
                int(np.median(heights)),
            ]
        )
    return Table(
        "Size of the boxes",
        [
            "Kind",
            "Marks",
            "Median share of the page",
            "90th percentile",
            "Maximum",
            "Median height in px",
        ],
        rows,
        "Share of the page: the area of the box divided by the area of the image. Many receipts "
        "were scanned as whole pages of up to 4961 x 7016 px, on which every mark covers a small "
        "share.",
    )


def _positions(
    marks: Sequence[tuple[str, Mark]], kinds: Sequence[MarkKind], pages: Mapping[str, BBox]
) -> Table:
    rows: list[list[Cell]] = []
    for kind in [*kinds, None]:
        chosen = [(d, m) for d, m in marks if kind is None or m.kind is kind]
        counts = Counter(_third(m, pages[d]) for d, m in chosen)
        rows.append([kind.value if kind else "all", len(chosen), *(counts[t] for t in _THIRDS)])
    return Table(
        "Position on the page",
        ["Kind", "Marks", *_THIRDS],
        rows,
        "The third of the image height that holds the centre of the box.",
    )


def _flags_against_marks(
    by_id: Mapping[str, Document], plan: Sequence[str], final: FinalMarks
) -> Table:
    def kinds_of(doc_id: str) -> set[MarkKind]:
        return {m.kind for m in final.marks[doc_id].marks}

    hand = [d for d in plan if by_id[d].handwritten_annotation]
    digital = [d for d in plan if by_id[d].digital_annotation]
    comparisons: list[tuple[str, list[str], Callable[[str], bool]]] = [
        (
            "flagged for handwriting, no mark of a hand kind",
            hand,
            lambda d: not kinds_of(d) & HAND_KINDS,
        ),
        (
            "flagged for a digital note, no digital note",
            digital,
            lambda d: MarkKind.DIGITAL_NOTE not in kinds_of(d),
        ),
        (
            "digital note found, not flagged for one",
            [d for d in plan if not by_id[d].digital_annotation],
            lambda d: MarkKind.DIGITAL_NOTE in kinds_of(d),
        ),
        (
            "mark of a hand kind found, flagged for a digital note only",
            [d for d in plan if not by_id[d].handwritten_annotation],
            lambda d: bool(kinds_of(d) & HAND_KINDS),
        ),
    ]
    rows: list[list[Cell]] = []
    for name, group, differs in comparisons:
        found = sorted(d for d in group if differs(d))
        rows.append([name, len(group), len(found), ", ".join(found) or None])
    return Table(
        "Flags against marks",
        ["Comparison", "Receipts compared", "Receipts that differ", "Receipts"],
        rows,
        "Hand kinds: handwriting, signature, stamp, highlighter and pen mark. Flags are compared, "
        "not changed.",
    )


def _forged(by_id: Mapping[str, Document], plan: Sequence[str], final: FinalMarks) -> Table:
    rows: list[list[Cell]] = []
    for doc_id in sorted(d for d in plan if by_id[d].label is Label.FORGED):
        regions = [r.box for r in by_id[doc_id].regions]
        found = final.marks[doc_id].marks
        overlapping = [m for m in found if any(m.box.overlaps(r) for r in regions)]
        rows.append([doc_id, len(found), len(regions), len(overlapping)])
    return Table(
        "Marks on forged receipts",
        ["Receipt", "Marks", "Annotated rectangles", "Marks overlapping a rectangle"],
        rows,
        "Annotated rectangles: modified regions and copy sources. Overlapping means sharing at "
        "least one pixel.",
    )


def render_report(tables: Sequence[Table]) -> str:
    """The Markdown report."""
    header = (
        "# Benign marks: design fold\n\n"
        "> Generated by `docfraud marks report`. Do not edit by hand.\n\n"
        "Where the handwriting, stamps and digital notes of the flagged design-fold receipts "
        "are (ADR-0014, step 4.2), analysed as the [benign marks guide]"
        "(../../docs/benign-marks-guide.md) fixed before these receipts were viewed for this "
        "step. The "
        "boxes are in `annotations/findit2/benign-marks/`.\n\n"
    )
    return header + "\n".join(render_table(t) for t in tables)
