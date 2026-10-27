"""The report of Phase 5 on the training split, rendered from its run record (ADR-0029)."""

from collections.abc import Mapping
from typing import Any

from docfraud.markdown import Cell, Table, render_table

TRACK_NAMES = {"transcript": "T1 (transcripts)", "ocr": "T2 (OCR)"}


def ci(e: Mapping[str, float]) -> str:
    """An estimate with its interval, as reports write it."""
    return f"{e['value']:.3f} ({e['low']:.3f} to {e['high']:.3f})"


def ablation(name: str, track: Mapping[str, Any], *, note: str | None = None) -> Table:
    """The ablation table of a track, with the floors; ``note`` replaces the training split's."""
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
                ci(m["average_precision"]),
                ci(m["roc_auc"]),
                ci(m["recall_at_10pct"]),
                ci(m["recall_at_20pct"]),
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
        note=note
        or f"{track['receipts']} training receipts, {track['forged']} forged, out of fold; "
        "system 2 seed 0. Intervals: stratified bootstrap, 10,000 resamples, seed 0. The floor "
        "row gives chance for each ranking metric and the F1 of flagging every receipt. *F1 at "
        "the threshold "
        "that maximises it on the same scores, Brier score and ECE of the models' own "
        "probabilities before Platt scaling: optimistic.",
    )


def comparisons(name: str, track: Mapping[str, Any], *, where: str | None = None) -> Table:
    """The comparisons of section 8; ``where`` replaces the training split's first sentences."""
    rows: list[list[Cell]] = [
        [other, ci(d), "yes" if d["above_zero"] else "no"]
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
        note=(
            where
            or "Paired bootstrap of the evaluation protocol. On this split the comparisons are a "
            "reference; the claims are made on the test split in Phase 7."
        )
        + f" RQ2's rule {verdict} here: "
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
            [
                name,
                *values,
                f"{mean:.3f} ± {sd:.3f}",
                results[track]["calibration_slope"],
                results[track]["calibration_slope_unpenalised"],
            ]
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
            (
                "Track",
                "AP, seed 0",
                "AP, seed 1",
                "AP, seed 2",
                "Mean ± sd",
                "Calibration slope",
                "Slope without the penalty*",
            ),
            seed_rows,
            note="The calibration slope of the fusion model's out-of-fold probabilities (section 7 "
            "(a)): about 0.94 expected; below 0.9 would be overfitting, and above 1 the "
            "probabilities are more moderate than the outcomes, which Platt scaling on the "
            "validation split rescales. *The same inputs fitted out of fold without "
            "scikit-learn's default L2 penalty, the maximum-likelihood fit that the expected 0.94 "
            "assumes: a diagnostic outside the pre-registration, showing what the penalty does.",
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


def practical_table(t3: Mapping[str, Any]) -> Table:
    """Track T3 next to the fusion model of T2."""
    rows: list[list[Cell]] = [
        [row, ci(m["average_precision"]), ci(m["roc_auc"]), ci(m["recall_at_10pct"])]
        for row, m in t3.items()
    ]
    return Table(
        "Track T3: the file-metadata check, reported apart",
        ("Row", "Average precision", "ROC-AUC", "Recall at 10 %"),
        rows,
        note="Track T3 is T2 with the file-metadata input (ADR-0029, section 13). Re-saving a file "
        "removes the traits (ADR-0003); T3 enters no claim.",
    )


def rq3_tables(name: str, groups: Mapping[str, Any], *, where: str | None = None) -> list[Table]:
    """The tables of RQ3; ``where`` replaces the training split's description of the scores."""
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
                note=where
                or "Per evidence: the mean contribution to the log-odds of the fusion model of "
                "each receipt's fold, and in brackets on how many receipts it contributes most. "
                "Caught: forged receipts among the highest 10 % of the training receipts "
                "(expected counts where scores tie). Descriptive; groups are small.",
            )
        )
    return tables


def rq4_tables(
    rq4: Mapping[str, Any], *, review: str = "Hits among the highest 10 %"
) -> list[Table]:
    """The tables of RQ4; ``review`` names the column of the receipts sent to review."""
    rows: list[list[Cell]] = []
    for track, name in TRACK_NAMES.items():
        g = rq4[track]
        verdict = "points at forgeries" if g["detects"] else "probable shortcut"
        rows.append(
            [
                name,
                g["receipts"],
                g["located"],
                ci(g["hit_rate"]),
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
                review,
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


def false_alarm_table(alarms: Mapping[str, Any], *, where: str) -> Table:
    """Authentic receipts each catcher flags, by benign mark (ADR-0029, section 10)."""
    rows: list[list[Cell]] = []
    for track, name in TRACK_NAMES.items():
        catchers = alarms[track]
        groups = next(iter(catchers.values()))
        for group, g in groups.items():
            n = g["receipts"]
            rows.append(
                [
                    name,
                    group,
                    n,
                    *(f"{c[group]['flagged']:.1f} of {n:.0f}" for c in catchers.values()),
                ]
            )
    first = next(iter(alarms.values()))
    return Table(
        "False alarms on authentic receipts, by benign mark",
        ("Track", "Marks", "Authentic", *(f"flagged by {c}" for c in first)),
        rows,
        note="Authentic receipts by whether the dataset flags handwritten or digital marks on "
        f"them (evaluation protocol, section 7), and how many each row flags among {where} "
        "(expected counts where scores tie). Descriptive; groups are small.",
    )


def render_report(results: Mapping[str, Any], commit: str | None) -> str:
    """The training-split report of Phase 5."""
    tables: list[Table] = []
    for track, name in TRACK_NAMES.items():
        tables += [ablation(name, results[track]), comparisons(name, results[track])]
    tables += _seeds_and_more(results)
    tables.append(practical_table(results["T3"]))
    for track, name in TRACK_NAMES.items():
        tables += rq3_tables(name, results["rq3"][track])
    tables.append(
        false_alarm_table(
            results["false_alarms"], where="the highest 10 % of the training receipts"
        )
    )
    tables += rq4_tables(results["rq4"])
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
