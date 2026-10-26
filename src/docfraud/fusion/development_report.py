"""The report of Phase 5 on the training split, rendered from its run record (ADR-0029)."""

from collections.abc import Mapping
from typing import Any

from docfraud.markdown import Cell, Table, render_table

TRACK_NAMES = {"transcript": "T1 (transcripts)", "ocr": "T2 (OCR)"}


def _ci(e: Mapping[str, float]) -> str:
    return f"{e['value']:.3f} ({e['low']:.3f} to {e['high']:.3f})"


def _ablation(name: str, track: Mapping[str, Any]) -> Table:
    p = track["forged"] / track["receipts"]
    rows: list[list[Cell]] = [
        [
            "floor: every receipt flagged",
            f"{p:.3f}",
            "0.500",
            "0.100",
            "0.200",
            2 * p / (1 + p),
            None,
            None,
        ]
    ]
    for row, m in track["table"].items():
        rows.append(
            [
                row,
                _ci(m["average_precision"]),
                _ci(m["roc_auc"]),
                _ci(m["recall_at_10pct"]),
                _ci(m["recall_at_20pct"]),
                m["f1"],
                m["brier"],
                m["ece"],
            ]
        )
    return Table(
        f"Ablation table, {name}",
        (
            "Row",
            "Average precision",
            "ROC-AUC",
            "Recall at 10 %",
            "Recall at 20 %",
            "F1*",
            "Brier*",
            "ECE*",
        ),
        rows,
        note=f"{track['receipts']} training receipts, {track['forged']} forged, out of fold; "
        "system 2 seed 0. Intervals: stratified bootstrap, 10,000 resamples, seed 0. The floor "
        "row gives chance for each ranking metric and the F1 of flagging every receipt. *F1 at "
        "the threshold "
        "that maximises it on the same scores, Brier score and ECE of the models' own "
        "probabilities before Platt scaling: optimistic.",
    )


def _comparisons(name: str, track: Mapping[str, Any]) -> Table:
    rows: list[list[Cell]] = [
        [other, _ci(d), "yes" if d["above_zero"] else "no"]
        for other, d in track["comparisons"].items()
    ]
    verdict = "would hold" if track["fusion_beats_every_single_source"] else "would not hold"
    return Table(
        f"Comparisons of section 8, {name}",
        (
            "Fusion against",
            "Difference in average precision (95 % interval)",
            "Interval above zero",
        ),
        rows,
        note="Paired bootstrap of the evaluation protocol. On this split the comparisons are a "
        f"reference; the claims are made on the test split in Phase 7. RQ2's rule {verdict} here: "
        "it needs every single-source interval above zero (intersection-union test). The "
        "comparison with the score-level baseline is the secondary test of linking.",
    )


def _seeds_and_more(results: Mapping[str, Any]) -> list[Table]:
    seed_rows: list[list[Cell]] = []
    for track, name in TRACK_NAMES.items():
        values = list(results[track]["seeds"].values())
        mean = sum(values) / len(values)
        sd = (sum((v - mean) ** 2 for v in values) / (len(values) - 1)) ** 0.5
        seed_rows.append(
            [name, *values, f"{mean:.3f} ± {sd:.3f}", results[track]["calibration_slope"]]
        )
    subset_rows: list[list[Cell]] = []
    for track, name in TRACK_NAMES.items():
        r = results[track]
        for subset, values in r["sensitivity"].items():
            subset_rows += [
                [name, f"{subset} ({r['subsets'][subset]})", row, ap] for row, ap in values.items()
            ]
    return [
        Table(
            "Fusion by system 2 seed, and the overfitting check",
            ("Track", "AP, seed 0", "AP, seed 1", "AP, seed 2", "Mean ± sd", "Calibration slope"),
            seed_rows,
            note="The calibration slope of the fusion model's out-of-fold probabilities (section 7 "
            "(a)): about 0.94 expected; below 0.9 would be overfitting.",
        ),
        Table(
            "Average precision without the receipts shaped in development",
            ("Track", "Receipts kept", "Row", "Average precision"),
            subset_rows,
            note="The same out-of-fold scores, measured without the design fold (pixel systems and "
            "copy-move were developed on it) and without gold set A (the reader's instruction was "
            "developed on it), section 1.",
        ),
    ]


def _practical(t3: Mapping[str, Any]) -> Table:
    rows: list[list[Cell]] = [
        [row, _ci(m["average_precision"]), _ci(m["roc_auc"]), _ci(m["recall_at_10pct"])]
        for row, m in t3.items()
    ]
    return Table(
        "Track T3: the file-metadata check, reported apart",
        ("Row", "Average precision", "ROC-AUC", "Recall at 10 %"),
        rows,
        note="Track T3 is T2 with the file-metadata input (ADR-0029, section 13). Re-saving a file "
        "removes the traits (ADR-0003); T3 enters no claim.",
    )


def _rq3(name: str, groups: Mapping[str, Any]) -> list[Table]:
    tables = []
    for kind, members in groups.items():
        if not members:
            continue
        first = next(iter(members.values()))
        evidence = list(first["mean_contribution"])
        catchers = list(first["caught"])
        rows: list[list[Cell]] = []
        for group, g in members.items():
            n = g["receipts"]
            rows.append(
                [
                    group,
                    n,
                    *(f"{g['mean_contribution'][e]:+.2f} ({g['largest'][e]})" for e in evidence),
                    *(f"{g['caught'][c]:.1f} of {n}" for c in catchers),
                ]
            )
        tables.append(
            Table(
                f"RQ3 by {kind}, {name}",
                ("Group", "Forged", *evidence, *(f"caught by {c}" for c in catchers)),
                rows,
                note="Per evidence: the mean contribution to the log-odds of the fusion model of "
                "each receipt's fold, and in brackets on how many receipts it contributes most. "
                "Caught: forged receipts among the highest 10 % of the training receipts "
                "(expected counts where scores tie). Descriptive; groups are small.",
            )
        )
    return tables


def _rq4(rq4: Mapping[str, Any]) -> list[Table]:
    rows: list[list[Cell]] = []
    for track, name in TRACK_NAMES.items():
        g = rq4[track]
        verdict = "points at forgeries" if g["detects"] else "probable shortcut"
        rows.append(
            [
                name,
                g["receipts"],
                g["located"],
                _ci(g["hit_rate"]),
                g["baseline"],
                verdict,
                f"{g['in_review_hits']} of {g['in_review']}",
            ]
        )
    tables = [
        Table(
            "RQ4: the report's first region",
            (
                "Track",
                "Forged with regions",
                "With a first region",
                "Hit rate (95 % interval)",
                "Random baseline",
                "Verdict",
                "Hits among the highest 10 %",
            ),
            rows,
            note="The first region is the location of the evidence with the largest positive "
            "contribution (ADR-0029, section 12); any of its boxes on a modified rectangle or copy "
            "source is a hit; the baseline places one random box per region (ADR-0030). A report "
            "without a first region is a miss with a baseline of 0.",
        )
    ]
    for kind in ("technique", "entity"):
        technique_rows: list[list[Cell]] = []
        for track, name in TRACK_NAMES.items():
            technique_rows += [
                [name, group, f"{g['hits']} of {g['receipts']}"]
                for group, g in rq4[track]["by_" + kind].items()
            ]
        tables.append(
            Table(
                f"RQ4 by {kind}", ("Track", kind.capitalize(), "First region hits"), technique_rows
            )
        )
    return tables


def render_report(results: Mapping[str, Any], commit: str | None) -> str:
    """The training-split report of Phase 5."""
    tables: list[Table] = []
    for track, name in TRACK_NAMES.items():
        tables += [_ablation(name, results[track]), _comparisons(name, results[track])]
    tables += _seeds_and_more(results)
    tables.append(_practical(results["T3"]))
    for track, name in TRACK_NAMES.items():
        tables += _rq3(name, results["rq3"][track])
    tables += _rq4(results["rq4"])
    header = (
        "# Phase 5 on the training split\n\n"
        "> Generated from the run record `fusion-training-oof` by `docfraud fusion "
        "training-report`. Do not edit by hand.\n\n"
        "The development reference of "
        "[ADR-0029](../../docs/decisions/0029-fusion-preregistration.md) "
        f"(section 1), at commit `{(commit or 'unknown')[:7]}`: every number is out of fold on "
        "the 562 training receipts and optimistic. The validation split, scored once when the "
        "system is frozen, is the check; the test split, in Phase 7, carries the claims.\n\n"
    )
    return header + "\n".join(render_table(t) for t in tables)
