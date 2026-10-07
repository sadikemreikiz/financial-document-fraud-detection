"""Data behind the pre-registration of pixel forensics (ADR-0014).

Writes reports/forensics/preregistration-data.md. It regenerates every number that ADR-0014
cites, from labels, region annotations and image headers only: no pixel is decoded and no model
is run, so the clean folds stay unseen.

- forged training receipts per fold by technique, editing program and modified entity, from
  which the design fold was chosen;
- the design fold and the clean folds: sizes, chance level, the receipts the accounting rules
  cannot see, and the authentic receipts each round may use as synthetic bases;
- receipt and region sizes behind processing at native resolution;
- training receipts a person worked with before ADR-0014, the exclusion list of its
  sensitivity check.

Usage: ``uv run python experiments/forensics_preregistration.py``
"""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from PIL import Image

from docfraud.data.findit import (
    DUPLICATE_GROUPS,
    KNOWN_CORRECTIONS,
    REVIEWED_DISTINCT,
    load_findit2,
)
from docfraud.data.gold import load_manifest
from docfraud.data.models import Document, Entity, Label, RegionRole, Split, Technique
from docfraud.data.splits import select
from docfraud.evaluation.folds import FoldAssignment, load_folds
from docfraud.markdown import Cell, Table, render_table

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "forensics" / "preregistration-data.md"
ANNOTATIONS = ROOT / "annotations" / "findit2"
FOLDS = ANNOTATIONS / "cv-folds.json"
DESIGN_FOLD = 2
ARITHMETIC_ENTITIES = {Entity.TOTAL_PAYMENT, Entity.PRODUCT}
NAMED_TECHNIQUES = (Technique.CPI, Technique.CPO, Technique.CUT, Technique.PIX, Technique.IMI)
TILE = 256
FITTED_SIDE = 1024


def techniques(document: Document) -> set[Technique]:
    """Techniques used in the modified regions of a receipt."""
    return {t for r in document.modified_regions for t in r.techniques}


def entities(document: Document) -> set[Entity]:
    """Entities changed by the modified regions of a receipt."""
    return {r.entity for r in document.modified_regions if r.entity is not None}


def rules_blind(document: Document) -> bool:
    """A forged receipt whose annotated changes avoid amounts and products."""
    changed = entities(document)
    return bool(changed) and not changed & ARITHMETIC_ENTITIES


def forged(documents: Iterable[Document]) -> list[Document]:
    """The forged receipts among ``documents``."""
    return [d for d in documents if d.label is Label.FORGED]


def folds_table(train: Sequence[Document], folds: FoldAssignment) -> Table:
    """Forged receipts per fold, the basis for choosing the design fold."""
    per_fold = [
        forged(d for d in train if folds.folds[d.doc_id] == f) for f in range(folds.n_folds)
    ]
    rows: list[list[Cell]] = [
        ["forged receipts", *(len(docs) for docs in per_fold)],
        [
            "without region annotations",
            *(sum(not d.modified_regions for d in docs) for docs in per_fold),
        ],
    ]
    rows.extend(
        [
            f"with {technique.value.upper()}",
            *(sum(technique in techniques(d) for d in docs) for docs in per_fold),
        ]
        for technique in NAMED_TECHNIQUES
    )
    rows.append(
        [
            "editing programs",
            *(len({d.editing_software for d in docs} - {None}) for docs in per_fold),
        ]
    )
    rows.append(
        [
            f"entity types (of {len(Entity)})",
            *(len(set().union(*(entities(d) for d in docs))) for docs in per_fold),
        ]
    )
    return Table(
        "Forged training receipts per cross-validation fold",
        ("", *(f"Fold {f}" for f in range(folds.n_folds))),
        rows,
        note=f"Folds from {FOLDS.relative_to(ROOT)} (seed {folds.seed}). A receipt counts once "
        "per technique and per entity, however many regions use it.",
    )


def roles_table(train: Sequence[Document], folds: FoldAssignment, n_val: tuple[int, int]) -> Table:
    """The design fold against the clean folds."""
    design = [d for d in train if folds.folds[d.doc_id] == DESIGN_FOLD]
    clean = [d for d in train if folds.folds[d.doc_id] != DESIGN_FOLD]
    groups = (design, clean, list(train))

    def column(docs: list[Document]) -> list[Cell]:
        fakes = forged(docs)
        authentic = [d for d in docs if d.label is Label.AUTHENTIC]
        return [
            len(docs),
            len(fakes),
            sum(bool(d.modified_regions) for d in fakes),
            sum(rules_blind(d) for d in fakes),
            *(sum(t in techniques(d) for d in fakes) for t in NAMED_TECHNIQUES),
            len(authentic),
            sum(d.handwritten_annotation for d in authentic),
            round(len(fakes) / len(docs), 3),
        ]

    names = (
        "receipts",
        "forged",
        "forged with region annotations",
        "forged, changing only date, merchant or other text",
        *(f"forged with {t.value.upper()}" for t in NAMED_TECHNIQUES),
        "authentic",
        "authentic with handwriting",
        "share forged (chance level of average precision)",
    )
    columns = [column(g) for g in groups]
    return Table(
        "Design fold and clean folds",
        ("", f"Design fold ({DESIGN_FOLD})", "Clean folds", "Training split"),
        [[name, *(c[i] for c in columns)] for i, name in enumerate(names)],
        note=f"The validation split, used only for thresholds and calibration, has {n_val[0]} "
        f"receipts, {n_val[1]} of them forged. The accounting rules see only changes of "
        "amounts and products.",
    )


def bases_table(train: Sequence[Document], folds: FoldAssignment) -> Table:
    """Authentic receipts each cross-validation round may use as synthetic bases."""
    rows: list[list[Cell]] = [
        [
            held,
            "development" if held == DESIGN_FOLD else "clean measurement",
            sum(d.label is Label.AUTHENTIC and folds.folds[d.doc_id] != held for d in train),
        ]
        for held in range(folds.n_folds)
    ]
    return Table(
        "Synthetic bases per cross-validation round",
        ("Held-out fold", "Round", "Authentic receipts in the training folds"),
        rows,
    )


def union_share(document: Document, width: int, height: int) -> float:
    """Share of the image covered by the union of the modified regions."""
    mask = np.zeros((height, width), dtype=bool)
    for r in document.modified_regions:
        mask[r.box.y : r.box.y2, r.box.x : r.box.x2] = True
    return float(mask.mean())


def sizes_table(train: Sequence[Document]) -> Table:
    """Receipt and region sizes, read from image headers and annotations."""
    sizes: dict[str, tuple[int, int]] = {}
    formats: set[str] = set()
    for d in train:
        with Image.open(d.image_path) as image:
            sizes[d.doc_id] = image.size
            formats.add(str(image.format))
    widths = np.array([w for w, _ in sizes.values()])
    heights = np.array([h for _, h in sizes.values()])
    rectangles = [(d.doc_id, r) for d in train for r in d.regions]
    outside = sum(r.box.x2 > sizes[i][0] or r.box.y2 > sizes[i][1] for i, r in rectangles)
    modified = [r for _, r in rectangles if r.role is RegionRole.MODIFIED]
    median_height = float(np.median([r.box.height for r in modified]))
    shares = [union_share(d, *sizes[d.doc_id]) for d in forged(train) if d.modified_regions]
    longest = int(max(widths.max(), heights.max()))
    rows: list[list[Cell]] = [
        ["receipts", len(train)],
        ["file formats", ", ".join(sorted(formats))],
        [
            "width in px: min / median / max",
            f"{widths.min()} / {np.median(widths):.0f} / {widths.max()}",
        ],
        [
            "height in px: min / median / max",
            f"{heights.min()} / {np.median(heights):.0f} / {heights.max()}",
        ],
        [
            f"receipts below {TILE} px in width or height",
            int(((widths < TILE) | (heights < TILE)).sum()),
        ],
        ["annotated rectangles (modified and copy sources)", len(rectangles)],
        ["annotated rectangles outside their image", outside],
        ["modified regions", len(modified)],
        ["median height of a modified region in px", f"{median_height:.0f}"],
        ["forged receipts with modified regions", len(shares)],
        [
            "median share of a forged receipt covered by its modified regions",
            f"{100 * np.median(shares):.2f} %",
        ],
        [
            f"median modified-region height after fitting the longest side ({longest} px) to "
            f"{FITTED_SIDE} px",
            f"{median_height * FITTED_SIDE / longest:.1f} px",
        ],
    ]
    return Table("Receipt and region sizes, training split", ("Quantity", "Value"), rows)


def contact_groups() -> dict[str, frozenset[str]]:
    """Training receipts a person worked with before ADR-0014, by occasion."""
    review = json.loads((ANNOTATIONS / "rule-failure-review.json").read_text(encoding="utf-8"))
    corrections = {c.doc_id for c in KNOWN_CORRECTIONS}
    corrections |= {m for g in DUPLICATE_GROUPS for m in g.members}
    corrections |= {m for p in REVIEWED_DISTINCT for m in p.members}
    return {
        "gold field set A (Phase 2)": load_manifest(
            ANNOTATIONS / "gold-fields" / "manifest.json"
        ).doc_ids,
        "gold field set B (Phase 2)": load_manifest(
            ANNOTATIONS / "gold-fields-b" / "manifest.json"
        ).doc_ids,
        "rule failure review (Phase 3)": frozenset(r["doc_id"] for r in review["receipts"]),
        "data corrections and copy resolution (Phase 1)": frozenset(corrections),
    }


def contacts_table(train: Sequence[Document], folds: FoldAssignment) -> tuple[Table, list[str]]:
    """Earlier contacts with training receipts, and the resulting exclusion list."""
    by_id = {d.doc_id: d for d in train}
    groups = contact_groups()
    groups["any of these (exclusion list)"] = frozenset().union(*groups.values())

    def row(name: str, ids: frozenset[str]) -> list[Cell]:
        docs = [by_id[i] for i in ids if i in by_id]
        clean = [d for d in docs if folds.folds[d.doc_id] != DESIGN_FOLD]
        return [name, len(docs), len(clean), len(forged(clean))]

    excluded = sorted(
        i
        for i in groups["any of these (exclusion list)"]
        if i in by_id and folds.folds[i] != DESIGN_FOLD
    )
    table = Table(
        "Training receipts a person worked with before ADR-0014",
        ("Occasion", "Training receipts", "In clean folds", "Forged, in clean folds"),
        [row(name, ids) for name, ids in groups.items()],
        note="Recorded contacts only. Parser and rule development also read transcripts and "
        "parser output of many training receipts without recording which (ADR-0014, "
        "Disclosures).",
    )
    return table, excluded


def main() -> None:
    """Count the training split by fold and write the report."""
    documents = load_findit2(ROOT / "data" / "raw" / "findit2")
    train = select(documents, Split.TRAIN)
    val = select(documents, Split.VAL)
    folds = load_folds(FOLDS)
    contacts, excluded = contacts_table(train, folds)
    tables = [
        folds_table(train, folds),
        roles_table(train, folds, (len(val), len(forged(val)))),
        bases_table(train, folds),
        sizes_table(train),
        contacts,
    ]
    header = (
        "# Pixel forensics: data behind the pre-registration\n\n"
        "> Generated by `experiments/forensics_preregistration.py`. Do not edit by hand.\n\n"
        "Every number cited by [ADR-0014](../../docs/decisions/0014-pixel-forensics-"
        "preregistration.md), computed from labels, region annotations and image headers of "
        "the training split. No pixel is decoded and no model is run.\n\n"
    )
    footer = (
        "\n### Exclusion list of the sensitivity check\n\n"
        f"The {len(excluded)} clean-fold receipts above: " + ", ".join(excluded) + ".\n"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    body = "\n".join(render_table(t) for t in tables)
    OUTPUT.write_text(header + body + footer, encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
