"""Descriptive breakdowns of a detector's decisions (evaluation protocol, section 7).

Forged receipts are grouped by forgery technique, by the kind of information that was changed
and by the editing software; authentic receipts by whether they carry benign handwritten or
digital annotations. A receipt with several techniques or entities counts in each of its groups.
Groups are small, so every count is reported with its group size and nothing is tested.

A decision can be partial: at a review budget, receipts tied at the cut-off are reviewed with
equal probability, so each receipt carries the probability of being flagged and group counts are
expected values.
"""

from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from docfraud.data.models import Document, Label


def flagged_by_threshold(scores: Mapping[str, float], threshold: float) -> dict[str, float]:
    """1 for receipts at or above the threshold, else 0."""
    return {doc_id: float(score >= threshold) for doc_id, score in scores.items()}


def flagged_by_budget(scores: Mapping[str, float], budget: float) -> dict[str, float]:
    """Probability of each receipt being reviewed when only the top ``budget`` share is reviewed.

    Receipts tied at the cut-off share the remaining review capacity equally, as in
    ``docfraud.evaluation.detection.recall_at_budget``.
    """
    ids = list(scores)
    values: npt.NDArray[np.float64] = np.array([scores[i] for i in ids], dtype=float)
    capacity = budget * len(ids)
    reviewed = 0.0
    weights = np.zeros(len(ids))
    for value in np.unique(values)[::-1]:
        tied = values == value
        n = int(tied.sum())
        weights[tied] = min(1.0, max(0.0, (capacity - reviewed) / n))
        reviewed += n
    return dict(zip(ids, map(float, weights), strict=True))


def _forged_groups(document: Document) -> dict[str, set[str]]:
    regions = document.modified_regions
    return {
        "technique": {str(t) for r in regions for t in r.techniques} or {"not annotated"},
        "entity": {str(r.entity) for r in regions if r.entity is not None} or {"not annotated"},
        "software": {document.editing_software or "not recorded"},
    }


def _authentic_group(document: Document) -> str:
    if document.handwritten_annotation and document.digital_annotation:
        return "handwritten and digital"
    if document.handwritten_annotation:
        return "handwritten"
    if document.digital_annotation:
        return "digital"
    return "none"


def breakdown(documents: Sequence[Document], flagged: Mapping[str, float]) -> dict[str, Any]:
    """Expected numbers of forged receipts found and authentic receipts flagged, per group.

    Raises:
        ValueError: If a document has no label or no decision.
    """
    missing = [d.doc_id for d in documents if d.doc_id not in flagged]
    if missing or any(d.label is None for d in documents):
        msg = f"every document needs a label and a decision; missing decisions: {missing}"
        raise ValueError(msg)
    forged: dict[str, dict[str, dict[str, float]]] = defaultdict(
        lambda: defaultdict(lambda: {"receipts": 0.0, "found": 0.0})
    )
    authentic: dict[str, dict[str, float]] = defaultdict(lambda: {"receipts": 0.0, "flagged": 0.0})
    for d in documents:
        if d.label is Label.FORGED:
            for kind, names in _forged_groups(d).items():
                for name in names:
                    forged[kind][name]["receipts"] += 1
                    forged[kind][name]["found"] += flagged[d.doc_id]
        else:
            group = authentic[_authentic_group(d)]
            group["receipts"] += 1
            group["flagged"] += flagged[d.doc_id]
    return {
        "forged": {kind: dict(sorted(groups.items())) for kind, groups in forged.items()},
        "authentic": dict(sorted(authentic.items())),
    }
