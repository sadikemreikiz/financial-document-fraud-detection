"""The report of the frozen detector on the test split, rendered from its run records."""

from collections.abc import Mapping
from typing import Any

from docfraud.fusion.development_report import TRACK_NAMES, ci
from docfraud.fusion.final import CHANCE_RECALL, PUBLISHED_F1
from docfraud.markdown import Cell, Table, render_table


def _rows(parts: list[Mapping[str, Any]]) -> list[tuple[str, Mapping[str, Any]]]:
    return [(name, row) for part in parts for name, row in part["rows"].items()]


def render_report(
    claims: Mapping[str, Any], descriptive: Mapping[str, Any] | None, commit: str | None
) -> str:
    """The test-split report of the LLM detector: its claims, then the other models' rows."""
    parts = [claims] if descriptive is None else [claims, descriptive]
    p = claims["forged"] / claims["receipts"]
    metrics: list[list[Cell]] = [
        ["floor: every receipt flagged", f"{p:.3f}", "0.500", "0.100", "0.200", 2 * p / (1 + p)]
        + [None] * 5
    ]
    against: list[list[Cell]] = []
    for name, row in _rows(parts):
        m = row["table"]
        metrics.append(
            [
                name,
                ci(m["average_precision"]),
                ci(m["roc_auc"]),
                ci(m["recall_at_10pct"]),
                ci(m["recall_at_20pct"]),
                ci(m["f1"]),
                m["brier"],
                m["ece"],
                sum(row["failures"].values()),
                row["cost_usd"],
                m["threshold"],
            ]
        )
        rq6 = claims.get("rq6", {}).get("row") == name
        against.append(
            [
                f"{name} (RQ6)" if rq6 else name,
                TRACK_NAMES[row["track"]],
                ci(row["against_fusion"]),
                "yes" if row["against_fusion"]["above_zero"] else "no",
            ]
        )
    tables = [
        Table(
            "The detector on the test split",
            (
                "Row",
                "Average precision",
                "ROC-AUC",
                "Recall at 10 %",
                "Recall at 20 %",
                "F1 (95 % interval)",
                "Brier",
                "ECE",
                "Answer-form failures",
                "Cost, USD",
                "Frozen threshold",
            ),
            metrics,
            note=f"{claims['receipts']} test receipts, {claims['forged']} forged. Each row's Platt "
            "scaling and max-F1 threshold are those its validation run set, applied unchanged; a "
            "row without calibration metrics keeps its own score. A receipt whose answer failed "
            "the form ranks below every scored one. Intervals: stratified bootstrap, 10,000 "
            "resamples, seed 0.",
        ),
        Table(
            "Fusion against each row",
            (
                "Detector row",
                "Fusion of",
                "Fusion minus the row, average precision (95 % interval)",
                "Interval above zero",
            ),
            against,
            note="Paired bootstrap of the evaluation protocol. Only the row marked RQ6 is a "
            "pre-named test (ADR-0031, section 3); the others are descriptive.",
        ),
    ]
    header = (
        "# The LLM detector on the test split\n\n"
        "> Generated from the run records `llm-detector-test` and "
        "`llm-detector-test-descriptive` by `docfraud final reports`. Do not edit by hand.\n\n"
        "The frozen detector "
        "([ADR-0031](../../../docs/decisions/0031-llm-detector-and-agent.md), "
        "[ADR-0032](../../../docs/decisions/0032-stronger-model-on-the-design-fold.md)), run "
        f"once at commit `{(commit or 'unknown')[:7]}`: the frozen model's rows first, then the "
        "other models' ([ADR-0034](../../../docs/decisions/0034-phase-7-preregistration.md), "
        "decision 10).\n\n"
    )
    if "rq6" in claims:
        rq6 = claims["rq6"]
        header += (
            f"**RQ6, the ninth pre-named test.** Fusion of {TRACK_NAMES[rq6['track']]} minus the "
            f"row `{rq6['row']}`: {ci(rq6['difference'])} in average precision. By the rule of "
            f"ADR-0031 (section 3): {rq6['answer']}.\n\n"
        )
    if "paper" in claims:
        paper = claims["paper"]
        verdict = "both hold" if paper["met"] else "they do not both hold"
        header += (
            f"**The paper's result.** The row `{paper['row']}` reads the transcripts, as ChatGPT "
            f"did in the published comparison: F1 {ci(paper['f1'])} and recall at 10 % "
            f"{ci(paper['recall_at_10pct'])}. ADR-0013's two conditions, F1's lower bound above "
            f"{PUBLISHED_F1} and recall's above {CHANCE_RECALL:.2f}: {verdict}. Descriptive.\n\n"
        )
    return header + "\n".join(render_table(t) for t in tables)
