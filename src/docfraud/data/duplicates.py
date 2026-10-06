"""Finding and resolving multiple copies of the same physical document.

Datasets built from scraped or pooled sources often contain the same receipt more than once
under different identifiers: identical files, rescans, and in forgery datasets the authentic
original next to its forged version ("twins"). Copies in different splits leak information from
training into evaluation, so they must be found and resolved before any split is used (ADR-0009).

Detection proposes candidate pairs from independent signals; a person reviews them; the reviewed
decisions are recorded with the dataset adapter; :func:`resolve_duplicates` applies them; and a
data test re-runs detection so that any new, undocumented candidate fails the build.
"""

import hashlib
import re
from collections import Counter
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from enum import StrEnum
from itertools import combinations

import numpy as np
import numpy.typing as npt
from PIL import Image

from docfraud.data.models import Document, Split
from docfraud.data.normalize import load_grayscale, read_transcript

SPLIT_PRIORITY: tuple[Split, ...] = (Split.TEST, Split.VAL, Split.TRAIN)
"""When copies sit in several splits, the copy in the earliest split listed here is kept, so the
evaluation splits stay as published and only training data shrinks."""


class DuplicateKind(StrEnum):
    """How the members of a duplicate group relate to each other."""

    COPY = "copy"  # the same receipt with the same label: identical files or rescans
    TWIN = "twin"  # an authentic original and its forged version


@dataclass(frozen=True, slots=True)
class DuplicateGroup:
    """Reviewed documents that are the same physical receipt."""

    members: tuple[str, ...]
    kind: DuplicateKind
    evidence: str


@dataclass(frozen=True, slots=True)
class DistinctPair:
    """A reviewed candidate pair that turned out to be two different transactions."""

    members: tuple[str, str]
    reason: str


@dataclass(frozen=True, slots=True)
class Dropped:
    """A document removed because another copy of the same receipt is kept."""

    document: Document
    kept: tuple[str, ...]
    reason: str


def resolve_duplicates(
    documents: Sequence[Document], groups: Iterable[DuplicateGroup]
) -> tuple[list[Document], list[Dropped]]:
    """Apply reviewed duplicate groups.

    Every member of a group gets the same ``source_id``, so split-leakage checks treat the group
    as one receipt. Members outside the highest-priority split present in the group are dropped.
    Within that split, extra copies with the same label are dropped and an authentic/forged
    twin pair is kept, because the two carry different labels and are both informative.

    Returns:
        The kept documents in input order, and the dropped ones with reasons.

    Raises:
        KeyError: If a group names a document that is not in ``documents``.
    """
    by_id = {d.doc_id: d for d in documents}
    replace: dict[str, Document] = {}
    dropped: list[Dropped] = []
    for group in groups:
        members = sorted((by_id[m] for m in group.members), key=lambda m: m.doc_id)
        source = min(m.source_id for m in members)
        top = min((m.split for m in members), key=_priority)
        survivors: list[Document] = []
        losers: list[tuple[Document, str]] = []
        for member in members:
            if member.split is not top:
                losers.append((member, f"another copy of this receipt is kept in {top}"))
            elif any(s.label is member.label for s in survivors):
                losers.append((member, "extra copy with the same label in the same split"))
            else:
                survivors.append(member)
        kept = tuple(s.doc_id for s in survivors)
        dropped += [Dropped(member, kept, reason) for member, reason in losers]
        for survivor in survivors:
            replace[survivor.doc_id] = survivor.model_copy(update={"source_id": source})
    gone = {d.document.doc_id for d in dropped}
    kept_documents = [replace.get(d.doc_id, d) for d in documents if d.doc_id not in gone]
    return kept_documents, dropped


def _priority(split: Split | None) -> int:
    return SPLIT_PRIORITY.index(split) if split in SPLIT_PRIORITY else len(SPLIT_PRIORITY)


@dataclass(frozen=True, slots=True)
class Candidate:
    """A pair of documents that might be the same receipt, with the signals that fired."""

    members: tuple[str, str]
    signals: tuple[str, ...]
    line_overlap: float


_TIME = re.compile(r"\b\d{1,2}:\d{2}(?::\d{2})?\b")
_DATE = re.compile(r"\b\d{1,2}[/.-]\d{1,2}[/.-]\d{2,4}\b")
_IDENTIFIER = re.compile(r"\b(?=[A-Z0-9/-]*\d)(?=[A-Z0-9/-]*[A-Z])[A-Z0-9/-]{5,}\b|\b\d{6,}\b")


def find_candidates(
    documents: Sequence[Document],
    *,
    min_line_overlap: float = 0.75,
    max_hash_distance: int = 4,
) -> list[Candidate]:
    """Propose pairs that may be copies of the same receipt.

    A pair is proposed when the normalised pixels are identical, or when most transcript lines
    are shared *and* at least one independent signal agrees: a near-identical perceptual hash,
    the same printed date and time, or a shared identifier (such as a receipt number) that no
    other document contains. Shared lines alone are not enough, because receipts from the same
    shop share their whole header and footer.
    """
    pages = [load_grayscale(d.image_path) for d in documents]
    exact = [hashlib.sha256(p.tobytes()).hexdigest() for p in pages]
    hashes = np.array([_dhash(p) for p in pages])
    texts = [read_transcript(d.transcript_path) if d.transcript_path else "" for d in documents]
    lines = [Counter(line for line in t.splitlines() if line.strip()) for t in texts]
    stamps = [(set(_TIME.findall(t)), set(_DATE.findall(t))) for t in texts]
    identifiers = [set(_IDENTIFIER.findall(t.upper())) for t in texts]
    frequency = Counter(x for found in identifiers for x in found)
    rare = [{x for x in found if frequency[x] == 2} for found in identifiers]  # noqa: PLR2004

    candidates = []
    for i, j in combinations(range(len(documents)), 2):
        pair = (documents[i].doc_id, documents[j].doc_id)
        overlap = _overlap(lines[i], lines[j])
        if exact[i] == exact[j]:
            candidates.append(Candidate(pair, ("identical pixels",), overlap))
            continue
        if overlap < min_line_overlap:
            continue
        distance = int((hashes[i] != hashes[j]).sum())
        same_stamp = bool(stamps[i][0] & stamps[j][0]) and bool(stamps[i][1] & stamps[j][1])
        signals = tuple(
            name
            for name, fired in (
                ("near-identical image", distance <= max_hash_distance),
                ("same date and time", same_stamp),
                ("shared rare identifier", bool(rare[i] & rare[j])),
            )
            if fired
        )
        if signals:
            candidates.append(Candidate(pair, signals, overlap))
    return candidates


def undocumented(
    candidates: Iterable[Candidate],
    groups: Iterable[DuplicateGroup],
    distinct: Iterable[DistinctPair],
) -> list[Candidate]:
    """Candidates that are neither inside a reviewed group nor a reviewed distinct pair."""
    grouped = [set(g.members) for g in groups]
    reviewed = {frozenset(p.members) for p in distinct}
    return [
        c
        for c in candidates
        if frozenset(c.members) not in reviewed and not any(set(c.members) <= g for g in grouped)
    ]


def group_index(groups: Iterable[DuplicateGroup]) -> Mapping[str, DuplicateGroup]:
    """Map each document identifier to its group."""
    return {member: group for group in groups for member in group.members}


def _dhash(page: npt.NDArray[np.uint8], size: int = 16) -> npt.NDArray[np.bool_]:
    resized = Image.fromarray(page).resize((size + 1, size), Image.Resampling.LANCZOS)
    small = np.asarray(resized, dtype=np.int16)
    return (small[:, 1:] > small[:, :-1]).flatten()


def _overlap(a: Counter[str], b: Counter[str]) -> float:
    smaller = min(sum(a.values()), sum(b.values()))
    return sum((a & b).values()) / smaller if smaller else 0.0
