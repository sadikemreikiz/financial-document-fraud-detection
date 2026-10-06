"""Adapter for the Find-it-again receipt forgery dataset (Tornés et al., ICDAR 2023).

Layout after extraction::

    findit2/
        train.txt, val.txt, test.txt     label files (CSV)
        train/, val/, test/              <id>.png images and <id>.txt transcripts

Each label row holds the image name, two benign-annotation flags, the forged flag and the
forgery annotation exported by the VGG Image Annotator as a Python literal.

The published files contain a few inconsistencies. Each is resolved by an explicit, documented
correction below. Anything else that looks inconsistent raises :class:`DataIntegrityError`
instead of being guessed, and a correction that is no longer needed raises as well, because it
means the upstream data changed.
"""

import ast
import csv
from collections import defaultdict
from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from docfraud.data.duplicates import (
    DistinctPair,
    DuplicateGroup,
    DuplicateKind,
    resolve_duplicates,
)
from docfraud.data.models import (
    BBox,
    Document,
    Entity,
    Label,
    Region,
    RegionRole,
    Split,
    Technique,
)
from docfraud.data.splits import check_disjoint

DATASET = "findit2"
SOURCE_DATASET = "sroie"
LABEL_FILES: Mapping[Split, str] = {
    Split.TRAIN: "train.txt",
    Split.VAL: "val.txt",
    Split.TEST: "test.txt",
}

_TECHNIQUES: Mapping[str, Technique] = {
    "CPI": Technique.CPI,
    "CPO": Technique.CPO,
    "IMI": Technique.IMI,
    "CUT": Technique.CUT,
    "PIX": Technique.PIX,
    "Autre": Technique.OTHER,  # French for "other"; participants used both spellings
    "Other": Technique.OTHER,
}
_NO_TECHNIQUE = "None"  # used on copy-source regions
_ENTITIES: Mapping[str, Entity] = {
    "Total/payment": Entity.TOTAL_PAYMENT,
    "Product": Entity.PRODUCT,
    "Metadata": Entity.METADATA,
    "Company": Entity.COMPANY,
    "Other": Entity.OTHER,
}
_ROLES: Mapping[str, RegionRole] = {"no": RegionRole.MODIFIED, "yes": RegionRole.SOURCE}


class DataIntegrityError(RuntimeError):
    """Raised when the dataset contains an inconsistency without a documented correction."""


@dataclass(frozen=True, slots=True)
class Correction:
    """A documented fix for a known defect in the published dataset."""

    doc_id: str
    issue: str
    resolution: str


DUPLICATE_RESOLUTIONS: Mapping[str, tuple[Split, Label]] = {
    "X51006619709": (Split.TRAIN, Label.FORGED),
}
UNANNOTATED_FORGERIES: frozenset[str] = frozenset({"X51008114260"})

KNOWN_CORRECTIONS: tuple[Correction, ...] = (
    Correction(
        doc_id="X51006619709",
        issue=(
            "Listed twice: in train.txt as authentic and in val.txt as forged. Both rows carry "
            "forgery region annotations, the paper's count of 163 forged receipts includes it "
            "as forged, its image is the only alpha-channel PNG among the 'authentic' receipts, "
            "and the file ships in train/."
        ),
        resolution="Kept in train, labelled forged, removed from validation.",
    ),
    Correction(
        doc_id="X51008114260",
        issue="Labelled forged but has no region annotations.",
        resolution="Used for classification; excluded from localisation metrics.",
    ),
)


_C, _T = DuplicateKind.COPY, DuplicateKind.TWIN

# Reviewed on 2026-10-06 (ADR-0009): every candidate proposed by find_candidates() was compared
# line by line; members share receipt number and date/time, differing only in recognition noise
# or, for twins, in the forged values.
DUPLICATE_GROUPS: tuple[DuplicateGroup, ...] = (
    DuplicateGroup(("X51005268262", "X51005301659"), _C, "identical pixels"),
    DuplicateGroup(("X51005301661", "X51005303661"), _C, "identical pixels"),
    DuplicateGroup(("X51005301666", "X51005268275"), _C, "identical pixels"),
    DuplicateGroup(("X51005453801", "X51009453801"), _C, "identical pixels"),
    DuplicateGroup(("X51005453804", "X51009453804"), _C, "identical pixels"),
    DuplicateGroup(("X51005684949", "X510056849111"), _C, "identical pixels"),
    DuplicateGroup(("X51006008091", "X51009008091"), _C, "identical pixels"),
    DuplicateGroup(("X51006328913", "X51006329388"), _C, "identical pixels"),
    DuplicateGroup(("X51006328937", "X51006329399"), _C, "identical pixels"),
    DuplicateGroup(("X51006328967", "X51006332575"), _C, "identical pixels"),
    DuplicateGroup(("X51007135037", "X51006401723"), _C, "identical pixels"),
    DuplicateGroup(("X51009453729", "X51005453729"), _C, "identical pixels"),
    DuplicateGroup(("X51006329395", "X51006328919"), _C, "identical transcripts, rescan"),
    DuplicateGroup(("X51007103643", "X51007103641"), _C, "identical transcripts, rescan"),
    DuplicateGroup(("X51006387740", "X51006401940"), _C, "same time 15:23:00, spacing differs"),
    DuplicateGroup(("X51009568881", "X51005568881"), _T, "quantity 5 (50PCS) forged to 2 (20PCS)"),
    DuplicateGroup(("X51006401853", "X51007135247"), _T, "amount 35.32 forged to 3535.32"),
    DuplicateGroup(("X51005447842", "X51009447842"), _T, "0.90 and 5.00 forged to 100.90, 105.00"),
    DuplicateGroup(("X51006008095", "X51009008095"), _T, "same receipt CS180319-0015, forged"),
    DuplicateGroup(("X51007339105", "X51007339115"), _T, "3.30 forged to 903.90, 5.00 to 1000.00"),
)

SENSITIVITY_EXCLUSIONS: frozenset[str] = frozenset(
    {
        "X51005268275",
        "X510056849111",
        "X51009008091",
        "X51006401723",
        "X51005453729",
        "X51006329388",
        "X51006328919",
        "X51005447842",
    }
)
"""Test receipts that had a copy or twin elsewhere in the published dataset. The final evaluation
also reports results without them, as a pre-registered sensitivity analysis (ADR-0009)."""

_SAME_TEMPLATE = "same shop template, different receipt number and date"
REVIEWED_DISTINCT: tuple[DistinctPair, ...] = (
    DistinctPair(("X51005442334", "X51007231341"), _SAME_TEMPLATE),
    DistinctPair(("X51005442384", "X51006414470"), _SAME_TEMPLATE),
    DistinctPair(("X51005724621", "X51005724625"), _SAME_TEMPLATE),
    DistinctPair(("X51005724624", "X51005724554"), _SAME_TEMPLATE),
    DistinctPair(("X51005742110", "X51005763964"), "consecutive receipt numbers, same items"),
    DistinctPair(("X51006414470", "X51006414633"), _SAME_TEMPLATE),
    DistinctPair(("X51006913060", "X51006913023"), _SAME_TEMPLATE),
    DistinctPair(("X51007231341", "X51006913023"), _SAME_TEMPLATE),
    DistinctPair(("X51007339647", "X51007339656"), _SAME_TEMPLATE),
    DistinctPair(("X51007339658", "X51007339638"), _SAME_TEMPLATE),
    DistinctPair(("X51007846388", "X51007846391"), _SAME_TEMPLATE),
)


@dataclass(frozen=True, slots=True)
class _Row:
    split: Split
    doc_id: str
    forged: bool
    digital: bool
    handwritten: bool
    annotation: str


def load_findit2(
    root: Path,
    *,
    duplicate_resolutions: Mapping[str, tuple[Split, Label]] = DUPLICATE_RESOLUTIONS,
    unannotated_forgeries: frozenset[str] = UNANNOTATED_FORGERIES,
    duplicate_groups: Sequence[DuplicateGroup] = DUPLICATE_GROUPS,
) -> tuple[Document, ...]:
    """Load the dataset into documents, applying the documented corrections.

    Args:
        root: Directory containing ``train.txt``, ``val.txt``, ``test.txt`` and the split
            folders.
        duplicate_resolutions: Split and label for documents listed in several label files.
        unannotated_forgeries: Forged documents known to lack region annotations.
        duplicate_groups: Reviewed copies of the same receipt, resolved with
            :func:`docfraud.data.duplicates.resolve_duplicates`.

    Returns:
        Documents ordered by split (train, val, test) and then by identifier.

    Raises:
        DataIntegrityError: On any inconsistency not covered by :data:`KNOWN_CORRECTIONS`, or if
            a known correction no longer applies.
        FileNotFoundError: If label files, images or transcripts are missing.
    """
    rows_by_id: defaultdict[str, list[_Row]] = defaultdict(list)
    for split, filename in LABEL_FILES.items():
        for row in _read_label_file(root / filename, split):
            rows_by_id[row.doc_id].append(row)

    context = _Context(root, unannotated_forgeries, triggered=set())
    documents = []
    for doc_id, rows in rows_by_id.items():
        split, label, row = _resolve_rows(doc_id, rows, duplicate_resolutions, context.triggered)
        documents.append(_build_document(context, split, label, row))

    stale = (set(duplicate_resolutions) | unannotated_forgeries) - context.triggered
    if stale:
        msg = f"known corrections no longer apply, upstream data changed: {sorted(stale)}"
        raise DataIntegrityError(msg)

    known = {d.doc_id for d in documents}
    unknown = sorted({m for g in duplicate_groups for m in g.members} - known)
    if unknown:
        msg = f"duplicate groups name unknown documents, upstream data changed: {unknown}"
        raise DataIntegrityError(msg)
    documents, _ = resolve_duplicates(documents, duplicate_groups)
    try:
        check_disjoint(documents)
    except ValueError as error:
        raise DataIntegrityError(str(error)) from error

    order = {split: index for index, split in enumerate(LABEL_FILES)}
    return tuple(sorted(documents, key=lambda d: (order[d.split or Split.TRAIN], d.doc_id)))


def _read_label_file(path: Path, split: Split) -> list[_Row]:
    expected = ["image", "digital annotation", "handwritten annotation", "forged"]
    with path.open(encoding="utf-8", newline="") as handle:
        reader = csv.DictReader(handle)
        if reader.fieldnames is None or reader.fieldnames[:4] != expected:
            msg = f"{path.name}: unexpected header {reader.fieldnames}"
            raise DataIntegrityError(msg)
        rows = []
        for raw in reader:
            image = raw["image"]
            if not image.endswith(".png"):
                msg = f"{path.name}: unexpected image name {image!r}"
                raise DataIntegrityError(msg)
            rows.append(
                _Row(
                    split=split,
                    doc_id=image.removesuffix(".png"),
                    forged=_flag(raw["forged"], path.name),
                    digital=_flag(raw["digital annotation"], path.name),
                    handwritten=_flag(raw["handwritten annotation"], path.name),
                    annotation=raw["forgery annotations"].strip(),
                )
            )
    return rows


def _flag(value: str, source: str) -> bool:
    if value not in {"0", "1"}:
        msg = f"{source}: expected a 0/1 flag, got {value!r}"
        raise DataIntegrityError(msg)
    return value == "1"


def _resolve_rows(
    doc_id: str,
    rows: list[_Row],
    resolutions: Mapping[str, tuple[Split, Label]],
    triggered: set[str],
) -> tuple[Split, Label, _Row]:
    if len(rows) == 1:
        row = rows[0]
        return row.split, Label.FORGED if row.forged else Label.AUTHENTIC, row
    if doc_id not in resolutions:
        msg = f"{doc_id} is listed in several splits: {[r.split.value for r in rows]}"
        raise DataIntegrityError(msg)
    split, label = resolutions[doc_id]
    triggered.add(doc_id)
    candidates = [r for r in rows if r.split is split]
    if len(candidates) != 1:
        msg = f"{doc_id}: resolution expects exactly one row in {split.value}"
        raise DataIntegrityError(msg)
    return split, label, candidates[0]


@dataclass(slots=True)
class _Context:
    root: Path
    unannotated_forgeries: frozenset[str]
    triggered: set[str]


def _build_document(context: _Context, split: Split, label: Label, row: _Row) -> Document:
    image_path = context.root / split.value / f"{row.doc_id}.png"
    transcript_path = image_path.with_suffix(".txt")
    for path in (image_path, transcript_path):
        if not path.is_file():
            raise FileNotFoundError(path)

    annotation = _parse_annotation(row) if row.annotation.startswith("{") else None
    if label is Label.AUTHENTIC and annotation is not None:
        msg = f"{row.doc_id} is authentic but carries forgery annotations"
        raise DataIntegrityError(msg)
    if label is Label.FORGED and annotation is None:
        if row.doc_id not in context.unannotated_forgeries:
            msg = f"{row.doc_id} is forged but has no region annotations"
            raise DataIntegrityError(msg)
        context.triggered.add(row.doc_id)

    regions: tuple[Region, ...] = ()
    software = None
    if annotation is not None:
        regions = tuple(_parse_region(row.doc_id, raw) for raw in annotation.get("regions", []))
        software = str(annotation.get("file_attributes", {}).get("Software used", "")).strip()

    return Document(
        dataset=DATASET,
        doc_id=row.doc_id,
        source_id=f"{SOURCE_DATASET}/{row.doc_id}",
        split=split,
        label=label,
        image_path=image_path,
        transcript_path=transcript_path,
        regions=regions,
        has_region_annotations=bool(regions),
        digital_annotation=row.digital,
        handwritten_annotation=row.handwritten,
        editing_software=software or None,
    )


def _parse_annotation(row: _Row) -> dict[str, Any]:
    try:
        value = ast.literal_eval(row.annotation)
    except (ValueError, SyntaxError) as error:
        msg = f"{row.doc_id}: forgery annotation is not a valid literal"
        raise DataIntegrityError(msg) from error
    if not isinstance(value, dict) or value.get("filename") != f"{row.doc_id}.png":
        msg = f"{row.doc_id}: forgery annotation does not describe this image"
        raise DataIntegrityError(msg)
    return value


def _parse_region(doc_id: str, raw: Mapping[str, Any]) -> Region:
    shape = raw["shape_attributes"]
    attributes = raw["region_attributes"]
    if shape.get("name") != "rect":
        msg = f"{doc_id}: unsupported region shape {shape.get('name')!r}"
        raise DataIntegrityError(msg)

    role_name = attributes.get("Original area")
    entity_name = attributes.get("Entity type")
    if role_name not in _ROLES or entity_name not in _ENTITIES:
        msg = f"{doc_id}: unknown region attributes {attributes}"
        raise DataIntegrityError(msg)

    marked = attributes.get("Modified area", {})
    names = {marked} if isinstance(marked, str) else {k for k, v in marked.items() if v}
    unknown = names - set(_TECHNIQUES) - {_NO_TECHNIQUE}
    if unknown:
        msg = f"{doc_id}: unknown forgery techniques {sorted(unknown)}"
        raise DataIntegrityError(msg)

    return Region(
        box=BBox(x=shape["x"], y=shape["y"], width=shape["width"], height=shape["height"]),
        role=_ROLES[role_name],
        techniques=frozenset(_TECHNIQUES[n] for n in names if n != _NO_TECHNIQUE),
        entity=_ENTITIES[entity_name],
    )
