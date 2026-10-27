"""The report of the comparison of the fusion inputs on both splits, from its run record."""

from collections.abc import Mapping
from typing import Any

from docfraud.fusion.development_report import TRACK_NAMES
from docfraud.markdown import Cell, Table, render_table


def _pair(row: Mapping[str, Any], key: str) -> str:
    return f"{row['training'][key]:.3f} → {row['validation'][key]:.3f}"


def render_report(results: Mapping[str, Any], commit: str | None) -> str:
    """The input-shift report of Phase 5 (ADR-0029, section 7 (d))."""
    rows: list[list[Cell]] = [
        [
            c["input"],
            TRACK_NAMES[c["track"]] if c["track"] else "both",
            c["seed"] if c["seed"] is not None else "all",
            _pair(c, "mean"),
            _pair(c, "zeros"),
            _pair(c, "median"),
            _pair(c, "q90"),
            c["distance"],
            c["critical"],
            f"{c['p_value']:.4f}",
        ]
        for c in results["comparisons"]
    ]
    receipts = results["receipts"]
    n = len(results["comparisons"])
    table = Table(
        "Every fusion input, training → validation",
        (
            "Input",
            "Text source",
            "System 2 seed",
            "Mean",
            "Share of zeros",
            "Median",
            "90th percentile",
            "KS distance D",
            "D for p < 0.05",
            "p",
        ),
        rows,
        note=f"{receipts['training']} training receipts, out of fold; {receipts['validation']} "
        "validation receipts, from the models trained on the whole training split. D is the "
        "largest gap between the two cumulative distributions; for an input of 0 or 1 it is the "
        "difference between the shares of 1. p: permutation test, "
        f"{results['permutations']:,} shuffles of the sides, seed 0; 'D for p < 0.05' is the "
        "95th percentile of the shuffled distances. An input is compared once where it does not "
        "depend on the text source ('both') or on system 2's seed ('all'). "
        f"{results['below_0_05']} of {n} comparisons have p < 0.05; about {n / 20:.0f} would by "
        "chance, and no correction is made.",
    )
    header = (
        "# Phase 5: the fusion inputs on both splits\n\n"
        "> Generated from the run record `fusion-input-shift` by `docfraud fusion "
        "input-shift-report`. Do not edit by hand.\n\n"
        "The check of [ADR-0029](../../docs/decisions/0029-fusion-preregistration.md), section "
        f"7 (d), at commit `{(commit or 'unknown')[:7]}`, made before the validation check and "
        "without labels: the training inputs come from the cross-validation models, the "
        "validation inputs from the models trained on the whole training split. Every "
        "difference is reported and none changes what follows. Two differences are expected by "
        "construction: the rules' score of a training receipt comes from the scorer of its fold, "
        "one of five, and that of a validation receipt from one scorer; system 2's training "
        "scores come from the round models, its validation scores from the models of the whole "
        "training split. A difference may also come from the receipts themselves, which no "
        "comparison without labels can tell apart. The method was chosen on random splits of "
        "the training inputs ([input-shift-method.md](input-shift-method.md)).\n\n"
    )
    return header + render_table(table)
