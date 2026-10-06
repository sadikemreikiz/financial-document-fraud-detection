"""The gold set for field extraction: drawing it, and keeping it out of development.

Field extraction (the receipt parser now, an LLM later) is measured against manually verified
annotations. To keep that measurement honest, the gold receipts are drawn before the parser is
written and excluded from everything used to develop it. The procedure is documented in
``docs/annotation-guidelines.md``.
"""

import json
from collections.abc import Iterable, Sequence
from pathlib import Path

import numpy as np
from pydantic import BaseModel, ConfigDict

from docfraud.data.models import Document, Label, Split

GOLD_SEED = 20261006
GOLD_AUTHENTIC = 40
GOLD_FORGED = 10


class GoldManifest(BaseModel):
    """Which documents form the gold set and how they were drawn."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    dataset: str
    split: Split
    seed: int
    authentic: tuple[str, ...]
    forged: tuple[str, ...]

    @property
    def doc_ids(self) -> frozenset[str]:
        """All gold document identifiers."""
        return frozenset(self.authentic) | frozenset(self.forged)


def draw_gold_set(
    documents: Iterable[Document],
    *,
    n_authentic: int = GOLD_AUTHENTIC,
    n_forged: int = GOLD_FORGED,
    seed: int = GOLD_SEED,
) -> GoldManifest:
    """Draw a stratified random sample of training documents.

    Only the training split is eligible, so the gold set never touches validation or test data.
    The draw depends only on the seed and the sorted document identifiers.

    Raises:
        ValueError: If documents from several datasets are given or a stratum is too small.
    """
    train = sorted((d for d in documents if d.split is Split.TRAIN), key=lambda d: d.doc_id)
    datasets = {d.dataset for d in train}
    if len(datasets) != 1:
        msg = f"expected training documents of exactly one dataset, got {sorted(datasets)}"
        raise ValueError(msg)
    rng = np.random.default_rng(seed)
    picks: dict[Label, tuple[str, ...]] = {}
    for label, size in ((Label.AUTHENTIC, n_authentic), (Label.FORGED, n_forged)):
        pool = [d.doc_id for d in train if d.label is label]
        if size > len(pool):
            msg = f"cannot draw {size} {label.value} documents from {len(pool)}"
            raise ValueError(msg)
        chosen = rng.choice(len(pool), size=size, replace=False)
        picks[label] = tuple(sorted(pool[int(i)] for i in chosen))
    return GoldManifest(
        dataset=datasets.pop(),
        split=Split.TRAIN,
        seed=seed,
        authentic=picks[Label.AUTHENTIC],
        forged=picks[Label.FORGED],
    )


def save_manifest(manifest: GoldManifest, path: Path) -> None:
    """Write the manifest as indented JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(manifest.model_dump(mode="json"), indent=2) + "\n", encoding="utf-8")


def load_manifest(path: Path) -> GoldManifest:
    """Read a manifest written by :func:`save_manifest`."""
    return GoldManifest.model_validate_json(path.read_text(encoding="utf-8"))


def development_documents(
    documents: Sequence[Document], manifest: GoldManifest
) -> tuple[Document, ...]:
    """Training documents that may be used to develop extraction: training minus gold."""
    return tuple(
        d
        for d in documents
        if d.split is Split.TRAIN
        and d.dataset == manifest.dataset
        and d.doc_id not in manifest.doc_ids
    )
