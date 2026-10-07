"""Development measurements behind the accounting rules (ADR-0012).

Writes reports/rules/rules-development.md from training receipts only. It regenerates every
number that ADR-0012 cites from the exploration before the rules were written, and the
measurements made while developing them:

- which forged training receipts change amounts or products, the only ones arithmetic can see;
- how often at least one check fails, Phase 2 checks against the rules, by label and text source;
- the same on the 80 gold receipts, with correct reading (gold annotations) and with parser v2;
- the manual diagnosis of 15 failing authentic receipts, recorded in
  annotations/findit2/rule-failure-review.json, after checking that the seed draws that sample;
- scorer variants compared in cross-validation, before the pre-registered one was kept.

Usage: ``uv run python experiments/rules_development.py``
"""

import json
from collections import Counter
from collections.abc import Callable
from pathlib import Path

import numpy as np
from sklearn.linear_model import LogisticRegression

from docfraud.data.findit import load_findit2
from docfraud.data.gold import GoldAnnotation, load_annotations, load_manifest
from docfraud.data.models import Document, Label, Split
from docfraud.data.splits import select
from docfraud.detectors.rules import Status, check
from docfraud.detectors.scoring import features
from docfraud.evaluation.detection import average_precision, recall_at_budget, roc_auc
from docfraud.evaluation.folds import load_folds
from docfraud.markdown import Cell, Table, render_table
from docfraud.reading.consistency import CHECKS
from docfraud.reading.fields import ReceiptFields
from docfraud.reading.ocr import CachedOcr, RapidOcrEngine
from docfraud.reading.pipeline import TextSource, read_document

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "reports" / "rules" / "rules-development.md"
REVIEW = ROOT / "annotations" / "findit2" / "rule-failure-review.json"
FOLDS = ROOT / "annotations" / "findit2" / "cv-folds.json"
SAMPLE_SEED = 20261007
SAMPLE_SIZE = 15
ARITHMETIC_ENTITIES = {"total_payment", "product"}

type Detector = Callable[[ReceiptFields], bool]


def phase2_fails(fields: ReceiptFields) -> bool:
    """At least one Phase 2 check fails."""
    return any(c(fields) is False for c in CHECKS.values())


def rules_fail(fields: ReceiptFields) -> bool:
    """At least one Phase 3 rule fails."""
    return any(o.status is Status.FAIL for o in check(fields).values())


def share(flags: list[bool]) -> str:
    """``n / total (percent)``."""
    return f"{sum(flags)} / {len(flags)} ({100 * sum(flags) / len(flags):.1f} %)"


def entity_table(forged: list[Document]) -> Table:
    """Which forged receipts change something that arithmetic can check."""
    kinds: Counter[str] = Counter()
    for d in forged:
        entities = {str(r.entity) for r in d.modified_regions}
        if not entities:
            kinds["no annotated region"] += 1
        elif entities & ARITHMETIC_ENTITIES:
            kinds["changes amounts or products"] += 1
        else:
            kinds["changes only date, merchant or other text"] += 1
    return Table(
        "Forged training receipts by what they change",
        ("Change", "Receipts"),
        [(kind, n) for kind, n in kinds.most_common()],
        note=f"{len(forged)} forged training receipts; modified regions by annotated entity.",
    )


def failing_table(
    docs: list[Document], fields: dict[TextSource, dict[str, ReceiptFields]]
) -> Table:
    """How often at least one check fails, Phase 2 checks against Phase 3 rules."""
    rows: list[list[Cell]] = []
    for name, detector in (("Phase 2 checks", phase2_fails), ("Phase 3 rules", rules_fail)):
        for source in TextSource:
            shares = [
                share([detector(fields[source][d.doc_id]) for d in docs if d.label is label])
                for label in (Label.AUTHENTIC, Label.FORGED)
            ]
            rows.append([f"{name}, {source.value}", *shares])
    return Table(
        "At least one check fails, training split (parser v2)",
        ("Checks, text source", "Authentic", "Forged"),
        rows,
    )


def gold_table(docs: dict[str, Document], fields: dict[str, ReceiptFields]) -> Table:
    """The same on the gold receipts, with correct reading and with the parser."""
    gold: dict[str, GoldAnnotation] = {}
    for manifest in sorted((ROOT / "annotations" / "findit2").glob("gold-fields*/manifest.json")):
        gold |= load_annotations(manifest.parent, load_manifest(manifest))
    rows: list[list[Cell]] = []
    for name, detector in (("Phase 2 checks", phase2_fails), ("Phase 3 rules", rules_fail)):
        readings: tuple[tuple[str, dict[str, ReceiptFields]], ...] = (
            ("gold annotations", {i: a.fields for i, a in gold.items()}),
            ("parser v2, transcript", fields),
        )
        for reading, read in readings:
            shares = [
                share([detector(read[i]) for i in sorted(gold) if docs[i].label is label])
                for label in (Label.AUTHENTIC, Label.FORGED)
            ]
            rows.append([f"{name}, {reading}", *shares])
    return Table(
        "At least one check fails, 80 gold receipts",
        ("Checks, reading", "Authentic", "Forged"),
        rows,
        note="The rules were developed on these receipts, so the rule rows are descriptive.",
    )


def review_table(docs: list[Document], fields: dict[str, ReceiptFields]) -> Table:
    """The manual diagnosis of a seeded sample of failing authentic receipts.

    Raises:
        ValueError: If the recorded review is not the sample the seed draws.
    """
    population = [d for d in docs if d.label is Label.AUTHENTIC and phase2_fails(fields[d.doc_id])]
    picks = np.random.default_rng(SAMPLE_SEED).choice(len(population), SAMPLE_SIZE, replace=False)
    drawn = sorted(population[int(i)].doc_id for i in picks)
    review = json.loads(REVIEW.read_text(encoding="utf-8"))
    recorded = sorted(r["doc_id"] for r in review["receipts"])
    if drawn != recorded:
        msg = f"the recorded review does not match the seeded sample: {drawn} != {recorded}"
        raise ValueError(msg)
    counts = Counter(" and ".join(r["causes"]) for r in review["receipts"])
    return Table(
        f"Why authentic receipts fail: {SAMPLE_SIZE} of {len(population)} failing receipts",
        ("Receipt", "Failing Phase 2 checks", "Cause", "Diagnosis"),
        [
            (
                r["doc_id"],
                ", ".join(n for n, c in CHECKS.items() if c(fields[r["doc_id"]]) is False),
                " and ".join(r["causes"]).replace("_", " "),
                r["note"],
            )
            for r in review["receipts"]
        ],
        note="Manual diagnosis recorded in annotations/findit2/rule-failure-review.json. "
        + "; ".join(f"{cause.replace('_', ' ')}: {n}" for cause, n in counts.most_common())
        + ".",
    )


def variants_table(
    docs: list[Document], fields: dict[TextSource, dict[str, ReceiptFields]]
) -> Table:
    """Scorer variants in the stored cross-validation, before the pre-registered one was kept."""
    ordered = sorted(docs, key=lambda d: d.doc_id)
    folds = load_folds(FOLDS)
    fold_of = np.array([folds.folds[d.doc_id] for d in ordered])
    y = np.array([d.label is Label.FORGED for d in ordered])

    def out_of_fold(x: np.ndarray) -> np.ndarray:
        scores = np.zeros(len(y))
        for fold in range(folds.n_folds):
            held = fold_of == fold
            model = LogisticRegression().fit(x[~held], y[~held])
            scores[held] = model.predict_proba(x[held])[:, 1]
        return scores

    rows: list[list[Cell]] = []
    for source in TextSource:
        outcomes = [check(fields[source][d.doc_id]) for d in ordered]
        x = features(outcomes)
        fails = x[:, 0::2]
        for name, scores in (
            ("logistic regression, fail and not applicable (kept)", out_of_fold(x)),
            ("logistic regression, fail only", out_of_fold(fails)),
            ("any rule fails", (fails.sum(axis=1) > 0).astype(float)),
            ("number of failing rules", fails.sum(axis=1)),
        ):
            rows.append(
                [
                    f"{source.value}: {name}",
                    f"{average_precision(y, scores):.3f}",
                    f"{roc_auc(y, scores):.3f}",
                    f"{recall_at_budget(y, scores, 0.1):.3f}",
                    f"{recall_at_budget(y, scores, 0.2):.3f}",
                ]
            )
    return Table(
        "Scorer variants, cross-validated on the training split",
        (
            "Scorer",
            "Average precision",
            "ROC-AUC",
            "Recall at 10 % review",
            "Recall at 20 % review",
        ),
        rows,
        note=f"Chance level: average precision {y.mean():.3f}, ROC-AUC 0.5. No variant is best on "
        "every metric and the differences are small for 94 forged receipts, so the "
        "pre-registered scorer was kept rather than choosing among them on the training split.",
    )


def main() -> None:
    """Read every training receipt with both text sources and write the report."""
    documents = load_findit2(ROOT / "data" / "raw" / "findit2")
    train = list(select(documents, Split.TRAIN))
    ocr = CachedOcr(RapidOcrEngine(), ROOT / "artifacts" / "ocr")
    fields = {
        source: {d.doc_id: read_document(d, ocr, source).fields for d in train}
        for source in TextSource
    }
    tables = [
        entity_table([d for d in train if d.label is Label.FORGED]),
        failing_table(train, fields),
        gold_table({d.doc_id: d for d in train}, fields[TextSource.TRANSCRIPT]),
        review_table(train, fields[TextSource.TRANSCRIPT]),
        variants_table(train, fields),
    ]
    header = (
        "# Accounting rules: development measurements\n\n"
        "> Generated by `experiments/rules_development.py`. Do not edit by hand.\n\n"
        "Training receipts only (ADR-0012). These numbers guided the rules; they are not an "
        "evaluation. The unbiased result is the validation measurement in "
        "[accounting-rules.md](accounting-rules.md).\n\n"
    )
    OUTPUT.parent.mkdir(parents=True, exist_ok=True)
    OUTPUT.write_text(header + "\n".join(render_table(t) for t in tables), encoding="utf-8")
    print(f"written: {OUTPUT.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
