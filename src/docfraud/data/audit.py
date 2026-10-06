"""Data audit: composition, documented corrections, integrity and shortcut analysis.

The audit reads labels of every split, including test. This is the second pre-registered
exception of the evaluation protocol: it computes data statistics and never scores a model.

The shortcut analysis answers one question: can a label be predicted from traces that are not
evidence of forgery? It reports file-level traits before normalisation, and checks that they no
longer differ between classes after normalisation (ADR-0003).
"""

import statistics
from collections import Counter
from collections.abc import Callable, Iterable, Sequence
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from PIL import Image

from docfraud.data.findit import KNOWN_CORRECTIONS
from docfraud.data.models import Document, Entity, Label, RegionRole, Split, Technique
from docfraud.data.normalize import normalize_transcript
from docfraud.data.splits import check_disjoint
from docfraud.markdown import Cell, Table, render_table

_ALPHA_MODES = frozenset({"RGBA", "LA", "PA"})
EMPTY = "\N{EM DASH}"


@dataclass(frozen=True, slots=True)
class FileTraits:
    """Format-level properties of one document's files, before and after normalisation."""

    document: Document
    mode: str
    width: int
    height: int
    image_bytes: int
    effectively_gray: bool
    transcript_bytes: int
    crlf: bool
    trailing_whitespace: bool
    non_ascii: bool
    normalized_crlf: bool
    normalized_trailing_whitespace: bool
    normalized_transcript_bytes: int

    @property
    def has_alpha(self) -> bool:
        """Whether the stored image has an alpha channel."""
        return self.mode in _ALPHA_MODES

    @property
    def forged(self) -> bool:
        """Whether the document is labelled forged."""
        return self.document.label is Label.FORGED


def inspect_files(document: Document) -> FileTraits:
    """Measure the format-level traits of a document's image and transcript."""
    with Image.open(document.image_path) as image:
        mode, (width, height) = image.mode, image.size
        rgb = np.asarray(image.convert("RGB"))
    effectively_gray = bool(
        (rgb[..., 0] == rgb[..., 1]).all() and (rgb[..., 1] == rgb[..., 2]).all()
    )

    raw = document.transcript_path.read_bytes() if document.transcript_path else b""
    text = raw.decode("utf-8")
    normalized = normalize_transcript(text)
    return FileTraits(
        document=document,
        mode=mode,
        width=width,
        height=height,
        image_bytes=document.image_path.stat().st_size,
        effectively_gray=effectively_gray,
        transcript_bytes=len(raw),
        crlf="\r" in text,
        trailing_whitespace=_has_trailing_whitespace(text),
        non_ascii=not text.isascii(),
        normalized_crlf="\r" in normalized,
        normalized_trailing_whitespace=_has_trailing_whitespace(normalized),
        normalized_transcript_bytes=len(normalized.encode("utf-8")),
    )


def _has_trailing_whitespace(text: str) -> bool:
    return any(line != line.rstrip() for line in text.replace("\r\n", "\n").split("\n"))


def roc_auc(scores: Sequence[float], positives: Sequence[bool]) -> float:
    """Area under the ROC curve via the rank-sum statistic, with ties counted as one half."""
    values = np.asarray(scores, dtype=float)
    is_positive = np.asarray(positives, dtype=bool)
    n_pos, n_neg = int(is_positive.sum()), int((~is_positive).sum())
    if n_pos == 0 or n_neg == 0:
        msg = "ROC-AUC needs at least one positive and one negative"
        raise ValueError(msg)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(len(values), dtype=float)
    sorted_values = values[order]
    start = 0
    for end in range(1, len(values) + 1):
        if end == len(values) or sorted_values[end] != sorted_values[start]:
            ranks[order[start:end]] = (start + end + 1) / 2
            start = end
    return float((ranks[is_positive].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def build_audit(traits: Sequence[FileTraits]) -> list[Table]:
    """Compute every audit table from per-document traits."""
    documents = [t.document for t in traits]
    check_disjoint(documents)
    return [
        _composition(documents),
        _corrections(),
        _regions_by(
            documents,
            "Forgery techniques",
            "Technique",
            lambda r: sorted(r.techniques),
            "A region can combine several techniques, so counts may exceed the number of regions.",
        ),
        _regions_by(
            documents,
            "Modified entities",
            "Entity",
            lambda r: [r.entity],
            "Each modified region is annotated with exactly one entity type.",
        ),
        _region_geometry(traits),
        _software(documents),
        _integrity(traits),
        _shortcuts_binary(traits),
        _shortcuts_numeric(traits),
        _alpha_rule(traits),
    ]


def _share(flags: Iterable[bool]) -> float:
    values = list(flags)
    return sum(values) / len(values) if values else 0.0


def _composition(documents: Sequence[Document]) -> Table:
    rows: list[list[Cell]] = []
    for split in (*Split, None):
        docs = [d for d in documents if split is None or d.split is split]
        forged = sum(d.label is Label.FORGED for d in docs)
        rows.append(
            [
                split.value if split else "total",
                len(docs),
                forged,
                len(docs) - forged,
                f"{100 * forged / len(docs):.1f} %" if docs else EMPTY,
                sum(d.handwritten_annotation for d in docs),
                sum(d.digital_annotation for d in docs),
            ]
        )
    return Table(
        "Composition",
        ["Split", "Documents", "Forged", "Authentic", "Forged share", "Handwritten", "Digital"],
        rows,
        "Handwritten and digital annotations are benign modifications that occur on authentic "
        "and forged receipts alike.",
    )


def _corrections() -> Table:
    return Table(
        "Documented corrections",
        ["Document", "Issue", "Resolution"],
        [[c.doc_id, c.issue, c.resolution] for c in KNOWN_CORRECTIONS],
    )


def _regions_by(
    documents: Sequence[Document],
    title: str,
    column: str,
    keys: Callable[..., Sequence[Technique | Entity | None]],
    note: str,
) -> Table:
    counts: Counter[str] = Counter()
    docs_with: Counter[str] = Counter()
    for document in documents:
        seen: set[str] = set()
        for region in document.modified_regions:
            for key in keys(region):
                name = key.value if key is not None else "unspecified"
                counts[name] += 1
                seen.add(name)
        docs_with.update(seen)
    rows: list[list[Cell]] = [[name, n, docs_with[name]] for name, n in counts.most_common()]
    return Table(title, [column, "Modified regions", "Forged documents"], rows, note)


def _region_geometry(traits: Sequence[FileTraits]) -> Table:
    areas: list[float] = []
    relative: list[float] = []
    per_doc: list[int] = []
    sources = 0
    for t in traits:
        if not t.document.has_region_annotations:
            continue
        modified = t.document.modified_regions
        per_doc.append(len(modified))
        sources += sum(r.role is RegionRole.SOURCE for r in t.document.regions)
        for region in modified:
            areas.append(region.box.area)
            relative.append(100 * region.box.area / (t.width * t.height))
    rows: list[list[Cell]] = [
        ["Annotated forged documents", len(per_doc)],
        ["Modified regions", len(areas)],
        ["Copy-source regions", sources],
        ["Modified regions per document (median / max)", _median_max(per_doc)],
        ["Region area in px² (median / 90th percentile)", _median_p90(areas, ".0f")],
        ["Region area in % of the page (median / 90th percentile)", _median_p90(relative, ".3f")],
        [
            "Modified share of page pixels, all forged pages",
            f"{_modified_pixel_share(traits):.3f} %",
        ],
    ]
    return Table(
        "Forged regions",
        ["Statistic", "Value"],
        rows,
        "Areas come from annotated rectangles, which are larger than the edited pixels.",
    )


def _median_max(values: Sequence[int]) -> str:
    if not values:
        return EMPTY
    return f"{statistics.median(values):g} / {max(values)}"


def _median_p90(values: Sequence[float], spec: str) -> str:
    if not values:
        return EMPTY
    p90 = statistics.quantiles(values, n=10)[-1] if len(values) > 1 else values[0]
    return f"{statistics.median(values):{spec}} / {p90:{spec}}"


def _modified_pixel_share(traits: Sequence[FileTraits]) -> float:
    covered = total = 0
    for t in traits:
        if not t.document.has_region_annotations:
            continue
        mask = np.zeros((t.height, t.width), dtype=bool)
        for region in t.document.modified_regions:
            mask[region.box.y : region.box.y2, region.box.x : region.box.x2] = True
        covered += int(mask.sum())
        total += t.width * t.height
    return 100 * covered / total if total else 0.0


def _software(documents: Sequence[Document]) -> Table:
    counts = Counter(
        d.editing_software or "not recorded" for d in documents if d.label is Label.FORGED
    )
    return Table(
        "Editing software (forged documents)", ["Software", "Documents"], counts.most_common()
    )


def _integrity(traits: Sequence[FileTraits]) -> Table:
    out_of_bounds = sum(
        not r.box.fits_within(t.width, t.height) for t in traits for r in t.document.regions
    )
    unlocalisable = [
        t.document.doc_id for t in traits if t.forged and not t.document.has_region_annotations
    ]
    rows: list[list[Cell]] = [
        ["No document or source document in more than one split", "passed"],
        ["Images decoded", f"{len(traits)} / {len(traits)}"],
        ["Transcripts decoded as strict UTF-8", f"{len(traits)} / {len(traits)}"],
        ["Regions outside the image bounds", out_of_bounds],
        ["Forged documents without region annotations", ", ".join(unlocalisable) or "none"],
    ]
    return Table("Integrity checks", ["Check", "Result"], rows)


@dataclass(frozen=True, slots=True)
class _BinaryTrait:
    name: str
    treatment: str
    present: Callable[[FileTraits], bool]


_BINARY_TRAITS = (
    _BinaryTrait(
        "Image has an alpha channel",
        "removed: rendered onto white, then grayscale",
        lambda t: t.has_alpha,
    ),
    _BinaryTrait("Image stored in colour mode", "removed: grayscale", lambda t: t.mode != "L"),
    _BinaryTrait(
        "Image content is colour (channels differ)",
        "content, kept: property of the original scan",
        lambda t: not t.effectively_gray,
    ),
    _BinaryTrait("Transcript has CR line endings", "removed: LF only", lambda t: t.crlf),
    _BinaryTrait(
        "Transcript has trailing whitespace", "removed: stripped", lambda t: t.trailing_whitespace
    ),
    _BinaryTrait(
        "Transcript has non-ASCII characters",
        "content, kept: e.g. a wrong currency symbol",
        lambda t: t.non_ascii,
    ),
    _BinaryTrait(
        "After normalisation: CR line endings", "must be 0 %", lambda t: t.normalized_crlf
    ),
    _BinaryTrait(
        "After normalisation: trailing whitespace",
        "must be 0 %",
        lambda t: t.normalized_trailing_whitespace,
    ),
)


def _percent(traits: Sequence[FileTraits], trait: _BinaryTrait) -> str:
    return f"{100 * _share(trait.present(t) for t in traits):.1f} %"


def _shortcuts_binary(traits: Sequence[FileTraits]) -> Table:
    forged = [t for t in traits if t.forged]
    authentic = [t for t in traits if not t.forged]
    rows: list[list[Cell]] = [
        [trait.name, _percent(forged, trait), _percent(authentic, trait), trait.treatment]
        for trait in _BINARY_TRAITS
    ]
    return Table(
        "Shortcut audit: binary traits",
        ["Trait", "Forged", "Authentic", "Treatment"],
        rows,
        "A trait that differs between classes but is not evidence of forgery is a shortcut. "
        "Format traits are removed by normalisation; content traits are kept on purpose.",
    )


def _shortcuts_numeric(traits: Sequence[FileTraits]) -> Table:
    labels = [t.forged for t in traits]
    measures: list[tuple[str, Callable[[FileTraits], float]]] = [
        ("Image file size", lambda t: t.image_bytes),
        ("Image width", lambda t: t.width),
        ("Image height", lambda t: t.height),
        ("Image area", lambda t: t.width * t.height),
        ("Transcript bytes, raw", lambda t: t.transcript_bytes),
        ("Transcript bytes, normalised", lambda t: t.normalized_transcript_bytes),
    ]
    rows: list[list[Cell]] = [
        [name, roc_auc([f(t) for t in traits], labels)] for name, f in measures
    ]
    return Table(
        "Shortcut audit: numeric traits",
        ["Trait used as a fraud score", "ROC-AUC"],
        rows,
        "0.5 means no signal. The file size reflects re-saving with editing tools and is never "
        "visible to models, which receive decoded, normalised pixels.",
    )


def _alpha_rule(traits: Sequence[FileTraits]) -> Table:
    rows: list[list[Cell]] = []
    for split in Split:
        subset = [t for t in traits if t.document.split is split]
        tp = sum(t.has_alpha and t.forged for t in subset)
        fp = sum(t.has_alpha and not t.forged for t in subset)
        fn = sum(not t.has_alpha and t.forged for t in subset)
        precision = tp / (tp + fp) if tp + fp else 0.0
        recall = tp / (tp + fn) if tp + fn else 0.0
        f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
        rows.append([split.value, precision, recall, f1])
    return Table(
        "Shortcut rule: 'the image has an alpha channel'",
        ["Split", "Precision", "Recall", "F1"],
        rows,
        "This rule inspects only the file format. It is reported for transparency and as the "
        "reference for the practical track (T3); headline results never use it (ADR-0003).",
    )


def render_audit(tables: Sequence[Table], source: str) -> str:
    """Render the audit as a Markdown document."""
    header = (
        "# Data audit: Find-it-again\n\n"
        "> Generated by `docfraud data audit`. Do not edit by hand.\n\n"
        f"Source: {source}\n\n"
    )
    return header + "\n".join(render_table(t) for t in tables)


def audit_dataset(documents: Sequence[Document], output: Path, source: str) -> list[Table]:
    """Inspect all documents, write the Markdown audit to ``output`` and return its tables."""
    tables = build_audit([inspect_files(d) for d in documents])
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(render_audit(tables, source), encoding="utf-8")
    return tables
