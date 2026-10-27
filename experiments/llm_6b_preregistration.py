"""Data behind the pre-registration of Phase 6b (ADR-0031).

Writes reports/llm/preregistration-6b-data.md. It regenerates every number that ADR-0031 cites
from data. Its inputs:

- labels of the training split: the size of the development sets;
- the training images' sizes and OCR word boxes: how large an image is for the model once it
  fits the API's limits, in image tokens, and how high its letters still are;
- the LLM ledger: what the requests of Phase 6a and the validation reading cost;
- the training inputs table: which design-fold receipts the out-of-fold fusion model of track T2
  ranks among the highest 10 % of the training split, the agent's development cases.

No request is sent to a language model, no validation receipt is scored and nothing of the test
split is read.

Usage: ``uv run python experiments/llm_6b_preregistration.py`` (needs the dataset, the OCR
cache, artifacts/llm/ledger.jsonl and artifacts/fusion/training-inputs.json)
"""

import json
import math
import statistics
from collections import defaultdict
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from docfraud.cli.common import CV_FOLDS_PATH, GOLD_B_MANIFEST, GOLD_MANIFEST, OCR_CACHE, cached_ocr
from docfraud.cli.fusion import TRAINING_INPUTS
from docfraud.cli.llm import LEDGER
from docfraud.data.findit import load_findit2
from docfraud.data.gold import load_manifest
from docfraud.data.models import Document, Label, Split
from docfraud.data.normalize import load_grayscale
from docfraud.data.splits import select
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS
from docfraud.evaluation.folds import load_folds
from docfraud.forensics.learned.text_scale import text_height
from docfraud.fusion.development import VARIANTS, in_budget, score
from docfraud.fusion.training_inputs import load_table
from docfraud.markdown import Cell, Table, render_table

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "llm" / "preregistration-6b-data.md"
LONG_EDGE = 2576  # the API's largest long edge for the models of ADR-0031, in pixels
TOKEN_CAP = 4784  # the most image tokens one image costs at that size
PATCH = 28  # one image token per patch of 28 x 28 pixels
AUTHENTIC_CASES = 3  # authentic design-fold receipts added to the agent's development cases


def development_sets(train: Sequence[Document]) -> Table:
    """Receipts and forged receipts of each set a Phase 6b component could be developed on."""
    by_id = {d.doc_id: d for d in train}
    sets = {
        "design fold": DESIGN_FOLD_IDS,
        "gold set A": load_manifest(ROOT / GOLD_MANIFEST).doc_ids,
        "gold set B": load_manifest(ROOT / GOLD_B_MANIFEST).doc_ids,
        "training split": set(by_id),
    }
    rows: list[list[Cell]] = []
    for name, ids in sets.items():
        documents = [by_id[i] for i in ids if i in by_id]
        rows.append([name, len(documents), sum(d.label is Label.FORGED for d in documents)])
    return Table("Development sets", ("Set", "Receipts", "Forged"), rows)


def fitted_size(width: int, height: int) -> tuple[float, int]:
    """The scale at which an image fits the API's limits, and its image tokens there."""
    scale = min(1.0, LONG_EDGE / max(width, height))
    tokens = math.ceil(width * scale / PATCH) * math.ceil(height * scale / PATCH)
    if tokens > TOKEN_CAP:
        scale *= math.sqrt(TOKEN_CAP / tokens)
        tokens = TOKEN_CAP
    return scale, tokens


def image_table(train: Sequence[Document]) -> Table:
    """Image tokens and letter height of every training receipt once it fits the API's limits."""
    ocr = cached_ocr(ROOT / OCR_CACHE)
    tokens, letters, shrunk = [], [], 0
    for document in train:
        page = load_grayscale(document.image_path)
        height, width = page.shape
        scale, cost = fitted_size(width, height)
        words = [token.box for token in ocr.recognize(page).tokens]
        tokens.append(cost)
        letters.append(text_height(words, height, width) * scale)
        shrunk += scale < 1
    tenth = float(np.percentile(letters, 10))
    rows: list[list[Cell]] = [
        ["receipts", len(train)],
        ["receipts made smaller to fit", shrunk],
        ["image tokens, median", f"{statistics.median(tokens):.0f}"],
        ["image tokens, 90th percentile", f"{np.percentile(tokens, 90):.0f}"],
        ["image tokens, most", max(tokens)],
        ["letter height after fitting, median (px)", statistics.median(letters)],
        ["letter height after fitting, 10th percentile (px)", tenth],
        ["receipts with letters below 10 px after fitting", sum(h < 10 for h in letters)],  # noqa: PLR2004
    ]
    return Table(
        "Training images at the API's limits",
        ("Measure", "Value"),
        rows,
        note=f"An image fits when its long edge is at most {LONG_EDGE} px and it costs at most "
        f"{TOKEN_CAP} image tokens, one per {PATCH} x {PATCH} px patch; a larger one is made "
        "smaller to fit. The image is the normalised grayscale page every system sees; the "
        "letter height is the median height of its OCR word boxes, times the scale.",
    )


def cost_table() -> Table:
    """What the requests of Phase 6a and the validation reading cost, per model and service."""
    entries = [json.loads(line) for line in (ROOT / LEDGER).read_text().splitlines() if line]
    groups: dict[tuple[str, bool], list[dict[str, object]]] = defaultdict(list)
    for entry in entries:
        groups[(str(entry["model"]), bool(entry["batch"]))].append(entry)
    rows: list[list[Cell]] = []
    for (model, batch), group in sorted(groups.items(), key=lambda kv: (-len(kv[1]), kv[0])):
        usage = [g["usage"] for g in group]
        rows.append(
            [
                model,
                "batch" if batch else "direct",
                len(group),
                statistics.median(u["input_tokens"] for u in usage),  # type: ignore[index]
                statistics.median(u["output_tokens"] for u in usage),  # type: ignore[index]
                f"{statistics.median(float(str(g['cost_usd'])) for g in group):.4f}",
                f"{sum(float(str(g['cost_usd'])) for g in group):.3f}",
            ]
        )
    spent = sum(float(str(e["cost_usd"])) for e in entries)
    return Table(
        "Requests so far",
        (
            "Model",
            "Service",
            "Requests",
            "Input tokens, median",
            "Output tokens, median",
            "Cost per request, median (USD)",
            "Cost (USD)",
        ),
        rows,
        note=f"From the ledger: {len(entries)} requests, {spent:.2f} USD in all. Input tokens "
        "leave out the cached instruction, which is billed at a tenth. The batch service bills "
        "half of the direct price.",
    )


def agent_cases(train: Sequence[Document]) -> Table:
    """The design-fold receipts the out-of-fold fusion model of T2 would send to review."""
    ids = [d.doc_id for d in train]
    y = np.array([d.label is Label.FORGED for d in train])
    folds = load_folds(ROOT / CV_FOLDS_PATH)
    fold_of = np.array([folds.folds[i] for i in ids])
    rows_t2 = load_table(ROOT / TRAINING_INPUTS).rows("ocr", 0)
    if [r.receipt for r in rows_t2] != ids:
        msg = "the training inputs table is not the training split"
        raise ValueError(msg)
    p = score(rows_t2, VARIANTS[-1], y, fold_of).probabilities
    top = in_budget(ids, p)
    design = np.array([i in DESIGN_FOLD_IDS for i in ids])
    chosen = np.array(top) & design
    authentic = np.flatnonzero(design & ~y & ~np.array(top))
    highest = authentic[np.argsort(-p[authentic], kind="stable")[:AUTHENTIC_CASES]]
    rows: list[list[Cell]] = [
        ["training receipts among the highest 10 %", int(sum(top))],
        ["of them in the design fold", int(chosen.sum())],
        ["of them forged", int((chosen & y).sum())],
        ["highest-scoring authentic design-fold receipts added", len(highest)],
        ["their fusion probabilities, out of fold", ", ".join(f"{p[i]:.3f}" for i in highest)],
    ]
    return Table(
        "The agent's development cases",
        ("Measure", "Value"),
        rows,
        note="Out-of-fold probabilities of the fusion model of track T2, system 2 seed 0, as in "
        "the training report; the highest 10 % with ties counted as the training report counts "
        "them. The authentic receipts are added so that the agent also meets receipts that are "
        "not forged.",
    )


def main() -> None:
    """Compute every table and write the report."""
    documents = load_findit2(ROOT / "data" / "raw" / "findit2")
    train = sorted(select(documents, Split.TRAIN), key=lambda d: d.doc_id)
    tables = [development_sets(train), image_table(train), cost_table(), agent_cases(train)]
    header = (
        "# Phase 6b: data behind the pre-registration\n\n"
        "> Generated by `experiments/llm_6b_preregistration.py`. Do not edit by hand.\n\n"
        "The numbers that [ADR-0031](../../docs/decisions/0031-llm-detector-and-agent.md) cites "
        "from data. Computed from the labels, images and OCR word boxes of the training split, "
        "the LLM ledger and the training inputs table. No request is sent to a language model, "
        "no validation receipt is scored and nothing of the test split is read.\n\n"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(header + "\n".join(render_table(t) for t in tables), encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
