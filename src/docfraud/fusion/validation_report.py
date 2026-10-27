"""The report of Phase 5's check on the validation split, rendered from its run record."""

from collections.abc import Mapping
from typing import Any

from docfraud.fusion.development_report import (
    TRACK_NAMES,
    ablation,
    ci,
    comparisons,
    false_alarm_table,
    practical_table,
    rq3_tables,
    rq4_tables,
)
from docfraud.markdown import Cell, Table, render_table

UNCALIBRATED = "†"


def _ablation(name: str, track: Mapping[str, Any]) -> Table:
    marked = {
        row if track["fitted"][row]["calibrated"] else f"{row} {UNCALIBRATED}": m
        for row, m in track["table"].items()
    }
    return ablation(
        name,
        {**track, "table": marked},
        note=f"{track['receipts']} validation receipts, {track['forged']} forged; every row is "
        "fitted on the out-of-fold inputs of the whole training split and Platt-scaled on these "
        "receipts; system 2 seed 0. Intervals: stratified bootstrap, 10,000 resamples, seed 0. "
        "The floor row gives chance for each ranking metric and the F1 of flagging every "
        "receipt. *F1 at the threshold that maximises it on these receipts, Brier score and ECE "
        "of probabilities whose Platt scaling was fitted on them: measured where they were "
        f"fitted, unbiased only on the test split (ADR-0029, section 7 (c)). {UNCALIBRATED} Platt "
        "scaling would reverse the order: the row keeps its own probabilities and has no "
        "calibration metrics.",
    )


def _seeds(results: Mapping[str, Any]) -> Table:
    rows: list[list[Cell]] = []
    for track, name in TRACK_NAMES.items():
        values = list(results[track]["seeds"].values())
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        rows.append([name, *values, f"{mean:.3f} ± {sd:.3f}"])
    return Table(
        "Fusion by system 2 seed",
        ("Track", "AP, seed 0", "AP, seed 1", "AP, seed 2", "Mean ± sd"),
        rows,
        note="Each seed's fusion model is fitted on that seed's out-of-fold inputs of the "
        "training split and applied to the validation inputs of the same seed's model trained "
        "on the whole training split.",
    )


def _fitted(results: Mapping[str, Any]) -> Table:
    rows: list[list[Cell]] = []
    sources = [(TRACK_NAMES[t], results[t]["fitted"], results[t]["table"]) for t in TRACK_NAMES]
    sources.append(("T3 (OCR and file metadata)", results["T3"]["fitted"], results["T3"]["table"]))
    for name, fitted, table in sources:
        for row, f in fitted.items():
            platt = f["platt"]
            calibrated = "yes" if f["calibrated"] else f"no {UNCALIBRATED}"
            rows.append(
                [name, row, platt["slope"], platt["offset"], calibrated, table[row]["threshold"]]
            )
    return Table(
        "What Phase 7 applies to the test split",
        ("Track", "Row", "Platt slope", "Platt offset", "Calibrated", "Max-F1 threshold"),
        rows,
        note="Platt scaling maps a row's log-odds x to 1 / (1 + exp(-(slope x + offset))); its "
        "slope is also the row's calibration slope on these receipts, so a slope above 1 means "
        "the model fitted on the training split was too cautious here and below 1 too "
        "confident. The max-F1 threshold is a calibrated probability. The record keeps every "
        "model's coefficients, intercept and standardisation as well.",
    )


def _policy(policy: Mapping[str, Any]) -> Table:
    rows: list[list[Cell]] = []
    ranks = []
    for track, name in TRACK_NAMES.items():
        for budget, p in policy[track].items():
            rank = round(p["budget"] * p["receipts"])
            if track == next(iter(TRACK_NAMES)):
                ranks.append(f"the {_ordinal(rank)} highest of {p['receipts']} at {budget}")
            rows.append(
                [
                    name,
                    budget,
                    p["threshold"],
                    f"{p['reviewed']} of {p['receipts']}",
                    ci(p["review_share"]),
                    f"{p['forged_reviewed']} of {p['forged']}",
                    ci(p["recall"]),
                    ci(p["precision"]),
                ]
            )
    return Table(
        "The review policy",
        (
            "Track",
            "Budget",
            "Threshold",
            "Sent to review",
            "Share reviewed",
            "Forged reviewed",
            "Recall",
            "Precision of the reviewed",
        ),
        rows,
        note="A receipt is sent to review if its calibrated probability reaches the threshold, "
        "otherwise accepted. The threshold is a calibrated probability of these receipts, "
        f"{' and '.join(ranks)}, set without labels (ADR-0029, section 11), and Phase 7 applies "
        "it unchanged; the first budget is the policy, the second the alternative setting. "
        "Intervals: bootstrap over receipts with the threshold held fixed, 10,000 resamples, "
        f"seed 0.{_exact(policy)}",
    )


def _exact(policy: Mapping[str, Any]) -> str:
    """Exact intervals for shares of 0 or 1, whose bootstrap interval cannot vary."""
    found = []
    for track, name in TRACK_NAMES.items():
        for budget, p in policy[track].items():
            for share, x, n in (
                ("precision", p["forged_reviewed"], p["reviewed"]),
                ("recall", p["forged_reviewed"], p["forged"]),
            ):
                if n and x in {0, n}:
                    low, high = _clopper_pearson(x, n)
                    found.append(f"{name} at {budget}, {share} {x} of {n}: {low:.3f} to {high:.3f}")
    if not found:
        return ""
    return (
        " A share of 0 or 1 on these receipts is the same in every resample, so its bootstrap "
        "interval collapses to a point; the exact (Clopper-Pearson) 95 % intervals are: "
        + "; ".join(found)
        + "."
    )


def _clopper_pearson(x: int, n: int) -> tuple[float, float]:
    """The exact 95 % interval of a share of 0 or 1 out of ``n``, in closed form."""
    edge = 0.025 ** (1 / n)
    return (edge, 1.0) if x == n else (0.0, 1 - edge)


def _ordinal(n: int) -> str:
    suffix = "th" if 10 <= n % 100 <= 20 else {1: "st", 2: "nd", 3: "rd"}.get(n % 10, "th")  # noqa: PLR2004
    return f"{n}{suffix}"


def _grounding(grounding: Mapping[str, Any]) -> list[Table]:
    rows: list[list[Cell]] = []
    for track, name in TRACK_NAMES.items():
        g = grounding[track]
        rows.append([name, g["reports"], g["facts"], len(g["violations"]), g["rate"]])
    tables = [
        Table(
            "Grounding of the audit reports",
            ("Track", "Reports", "Facts", "Violations", "Grounding rate"),
            rows,
            note="Every validation receipt's audit report, from the fusion model and the review "
            "policy of its track; the independent checker reads every number and box back from "
            "the text and looks it up in the assessment (ADR-0029, section 12). Target: 100 %.",
        )
    ]
    violations: list[list[Cell]] = [
        [name, v] for track, name in TRACK_NAMES.items() for v in grounding[track]["violations"]
    ]
    if violations:
        tables.append(Table("Every grounding violation", ("Track", "Receipt: fact"), violations))
    return tables


def render_report(results: Mapping[str, Any], commit: str | None) -> str:
    """The validation-split report of Phase 5."""
    tables: list[Table] = []
    for track, name in TRACK_NAMES.items():
        tables += [
            _ablation(name, results[track]),
            comparisons(
                name,
                results[track],
                where="Paired bootstrap of the evaluation protocol. On the validation split the "
                "comparisons apply the rules of the claims as a check; the claims are made on the "
                "test split in Phase 7.",
            ),
        ]
    tables += [_seeds(results), _fitted(results), _policy(results["policy"])]
    tables.append(practical_table(results["T3"]["table"]))
    for track, name in TRACK_NAMES.items():
        tables += rq3_tables(
            name,
            results["rq3"][track],
            where="Per evidence: the mean contribution to the log-odds of the fusion model fitted "
            "on the whole training split, and in brackets on how many receipts it contributes "
            "most. Caught: forged receipts among the highest 10 % of the validation receipts "
            "(expected counts where scores tie). Descriptive; groups are small.",
        )
    tables.append(
        false_alarm_table(
            results["false_alarms"], where="the highest 10 % of the validation receipts"
        )
    )
    tables += rq4_tables(results["rq4"], review="Hits among the receipts sent to review")
    tables += _grounding(results["grounding"])
    header = (
        "# Phase 5 on the validation split: the check\n\n"
        "> Generated from the run record `fusion-validation-check` by `docfraud fusion "
        "validation-report`. Do not edit by hand.\n\n"
        "The check of the frozen system of "
        "[ADR-0029](../../docs/decisions/0029-fusion-preregistration.md) (section 1), made once "
        f"at commit `{(commit or 'unknown')[:7]}` on the {results['transcript']['receipts']} "
        "validation receipts. Every model was "
        "fitted on the training split; Platt scaling, the max-F1 thresholds and the review "
        "policy's thresholds were set here. Average precision, ROC-AUC and recall at a budget "
        "are unbiased; F1, Brier score and ECE were measured on the receipts they were fitted "
        "on. The claims are made on the test split in Phase 7, by the same rules.\n\n"
    )
    return header + "\n".join(render_table(t) for t in tables)
