"""The report of the detector's check on the validation split, rendered from its run record."""

from collections.abc import Mapping
from typing import Any

from docfraud.fusion.development_report import TRACK_NAMES, ci
from docfraud.llm.detector_validation import RUN_NAME
from docfraud.markdown import Cell, Table, render_table

UNCALIBRATED = "†"
FITTED_NOTE = (
    "*F1 at the threshold that maximises it on these receipts, Brier score and ECE of "
    "probabilities whose Platt scaling was fitted on them: measured where they were fitted, "
    "unbiased only on the test split (ADR-0029, section 7 (c))."
)


def _name(row: str, result: Mapping[str, Any]) -> str:
    return row if result["fitted"]["platt"] is not None else f"{row} {UNCALIBRATED}"


def _metrics(results: Mapping[str, Any]) -> Table:
    p = results["forged"] / results["receipts"]
    rows: list[list[Cell]] = [
        ["floor: every receipt flagged", f"{p:.3f}", "0.500", "0.100", "0.200", 2 * p / (1 + p)]
        + [None] * 4
    ]
    for row, r in results["rows"].items():
        m = r["table"]
        failures = sum(r["failures"].values())
        rows.append(
            [
                _name(row, r),
                ci(m["average_precision"]),
                ci(m["roc_auc"]),
                ci(m["recall_at_10pct"]),
                ci(m["recall_at_20pct"]),
                m["f1"],
                m["brier"],
                m["ece"],
                failures,
                r["cost_usd"],
            ]
        )
    for track, m in results["fusion"].items():
        rows.append(
            [
                f"fusion of {TRACK_NAMES[track]}, Phase 5",
                ci(m["average_precision"]),
                ci(m["roc_auc"]),
                ci(m["recall_at_10pct"]),
                ci(m["recall_at_20pct"]),
                m["f1"],
                m["brier"],
                m["ece"],
                None,
                None,
            ]
        )
    return Table(
        "The detector on the validation split",
        (
            "Row",
            "Average precision",
            "ROC-AUC",
            "Recall at 10 %",
            "Recall at 20 %",
            "F1*",
            "Brier*",
            "ECE*",
            "Answer-form failures",
            "Cost, USD",
        ),
        rows,
        note=f"{results['receipts']} validation receipts, {results['forged']} forged. Each row "
        "is named by its input and its model; the frozen instruction is the same for all. A "
        "receipt whose answer failed the form ranks below every scored one. Platt scaling is "
        "fitted per row on these receipts, on the score. Fusion's rows are those of Phase 5's "
        "check, for reference. Intervals: stratified bootstrap, 10,000 resamples, seed 0. "
        f"{FITTED_NOTE} {UNCALIBRATED} Platt scaling would reverse the order: the row keeps the "
        "model's own score as a percentage and has no calibration metrics.",
    )


def _against_fusion(results: Mapping[str, Any]) -> Table:
    rq6 = results["rq6"]["row"]
    rows: list[list[Cell]] = [
        [
            f"{row} (RQ6)" if row == rq6 else row,
            TRACK_NAMES[r["track"]],
            ci(r["against_fusion"]),
            "yes" if r["against_fusion"]["above_zero"] else "no",
        ]
        for row, r in results["rows"].items()
    ]
    return Table(
        "Fusion against each row",
        (
            "Detector row",
            "Fusion of",
            "Fusion minus the row, average precision (95 % interval)",
            "Interval above zero",
        ),
        rows,
        note="Paired bootstrap of the evaluation protocol, stratified, 10,000 resamples, seed 0. "
        "Each row is set against fusion of the track with the same text source. Only the row "
        "marked RQ6 is a pre-named comparison, applied here as a check; the others are "
        "descriptive (ADR-0031, section 3; ADR-0032).",
    )


def _fitted(results: Mapping[str, Any]) -> Table:
    rows: list[list[Cell]] = []
    for row, r in results["rows"].items():
        platt = r["fitted"]["platt"]
        if platt is None:
            rows.append([row, None, None, f"no {UNCALIBRATED}", r["fitted"]["threshold"]])
        else:
            rows.append([row, platt["slope"], platt["offset"], "yes", r["fitted"]["threshold"]])
    return Table(
        "What Phase 7 applies to the test split",
        ("Row", "Platt slope", "Platt offset", "Calibrated", "Max-F1 threshold"),
        rows,
        note="Platt scaling maps a row's score s, from 0 to 100, to 1 / (1 + exp(-(slope s + "
        "offset))). The max-F1 threshold is a calibrated probability, or for a row marked "
        f"{UNCALIBRATED} the score divided by 100.",
    )


def _failures(results: Mapping[str, Any]) -> Table | None:
    rows: list[list[Cell]] = [
        [row, kind, count]
        for row, r in results["rows"].items()
        for kind, count in sorted(r["failures"].items())
    ]
    if not rows:
        return None
    return Table("Answer-form failures", ("Row", "Failure", "Receipts"), rows)


def render_report(results: Mapping[str, Any], commit: str | None) -> str:
    """The validation-split report of the LLM detector."""
    rq6 = results["rq6"]
    tables = [_metrics(results), _against_fusion(results), _fitted(results)]
    failures = _failures(results)
    if failures is not None:
        tables.append(failures)
    header = (
        "# The LLM detector on the validation split: the check\n\n"
        f"> Generated from the run record `{RUN_NAME}` by `docfraud detector "
        "validation-report`. Do not edit by hand.\n\n"
        "Every row of the frozen detector "
        "([ADR-0031](../../../docs/decisions/0031-llm-detector-and-agent.md), sections 3 and 4, "
        "and [ADR-0032](../../../docs/decisions/0032-stronger-model-on-the-design-fold.md)), run "
        "once at commit "
        f"`{(commit or 'unknown')[:7]}` on the {results['receipts']} validation receipts. "
        "Platt scaling and the max-F1 thresholds were set here. Average precision, ROC-AUC and "
        "recall at a budget are unbiased; F1, Brier score and ECE were measured on the receipts "
        "they were fitted on. The claims are made on the test split in Phase 7, by the same "
        "rules.\n\n"
        f"**RQ6, as a check.** Fusion of {TRACK_NAMES[rq6['track']]} minus the row "
        f"`{rq6['row']}`: {ci(rq6['difference'])} in average precision. By the rule of ADR-0031 "
        f"(section 3), the answer here would be: {rq6['answer']}.\n\n"
        "The comparison of the transcript row with the published ChatGPT result (ADR-0013's "
        "rule) needs F1 at a threshold set before the receipts are seen, so it is made on the "
        "test split only.\n\n"
    )
    return header + "\n".join(render_table(t) for t in tables)
