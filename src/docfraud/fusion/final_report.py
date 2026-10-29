"""The reports of Phase 7's fusion runs on the test split, rendered from their run records.

- :func:`render_test_report`: the frozen system once on the test split (``fusion test-check``):
  ADR-0013's success criterion, the eight pre-named tests of fusion, every table of the check,
  and the same again without the eight receipts that had copies;
- :func:`render_track_r`: the test images saved again as JPEG (``fusion resaved-check``);
- :func:`render_errors`: every error of the review policy and its cause (``fusion
  error-analysis``).
"""

from collections.abc import Mapping, Sequence
from typing import Any

from docfraud.fusion.development_report import (
    TRACK_NAMES,
    ci,
    false_alarm_table,
    practical_table,
    rq3_tables,
    rq4_tables,
)
from docfraud.fusion.final import CHANCE_RECALL, PUBLISHED_F1
from docfraud.markdown import Cell, Table, render_table

TEST_NOTE = (
    "Every row is the model fitted on the whole training split, with the Platt scaling and the "
    "max-F1 threshold that Phase 5's check set on the validation split, applied unchanged: F1 "
    "at that threshold, the Brier score and the expected calibration error are unbiased here. "
    "Intervals: stratified bootstrap over receipts, 10,000 resamples, seed 0."
)


def _f1(found: Mapping[str, Any]) -> Cell:
    f1 = found["f1"]
    return ci(f1) if isinstance(f1, Mapping) else f1


def ablation_table(name: str, track: Mapping[str, Any], *, where: str) -> Table:
    """A track's ablation table at the frozen thresholds, with the floors."""
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
                _f1(m),
                m["brier"],
                m["ece"],
                m["threshold"],
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
            "F1 (95 % interval)",
            "Brier",
            "ECE",
            "Frozen threshold",
        ),
        rows,
        note=f"{track['receipts']} {where} receipts, {track['forged']} forged. {TEST_NOTE} The "
        "floor row gives chance for each ranking metric and the F1 of flagging every receipt. A "
        "row without calibration metrics keeps its own probabilities, as on the validation split.",
    )


def success_table(results: Mapping[str, Any], *, where: str) -> Table:
    """ADR-0013's success criterion, per track, for the fused system."""
    rows: list[list[Cell]] = []
    for track, name in TRACK_NAMES.items():
        s = results[track]["success"]
        p = results[track]["forged"] / results[track]["receipts"]
        rows.append(
            [
                name,
                ci(s["f1"]),
                2 * p / (1 + p),
                "yes" if s["f1_above_published"] else "no",
                ci(s["recall_at_10pct"]),
                "yes" if s["recall_above_chance"] else "no",
                "**met**" if s["met"] else "**not met**",
            ]
        )
    return Table(
        f"ADR-0013's success criterion, {where}",
        (
            "Track",
            "F1 (95 % interval)",
            "F1 of flagging every receipt",
            f"F1's lower bound above {PUBLISHED_F1}",
            "Recall at 10 % (95 % interval)",
            f"Recall's lower bound above {CHANCE_RECALL:.2f}",
            "Criterion",
        ),
        rows,
        note="An improvement over the best published result (F1 28.39, ChatGPT, on this split) "
        "is claimed for a track only if both lower bounds exceed their references "
        "([ADR-0013](../../docs/decisions/0013-success-criterion-and-floors.md)). T1 reads the "
        "transcripts, as the published result did; T2 reads our OCR of the image alone.",
    )


def claims_table(name: str, track: Mapping[str, Any]) -> Table:
    """The pre-named tests of fusion on one track, and RQ2's answer by its rule."""
    rows: list[list[Cell]] = [
        [other, ci(d), "yes" if d["above_zero"] else "no"]
        for other, d in track["comparisons"].items()
    ]
    rq2 = (
        "fusion beats every single source"
        if track["fusion_beats_every_single_source"]
        else "inconclusive: not every single-source interval lies above zero"
    )
    return Table(
        f"The pre-named tests, {name}",
        ("Fusion against", "Difference in average precision (95 % interval)", "Claimed"),
        rows,
        note="Paired bootstrap of average precision; a difference is claimed only if its interval "
        "lies above zero ([ADR-0029](../../docs/decisions/0029-fusion-preregistration.md), "
        f"section 8). RQ2, by the intersection-union rule: {rq2}. The comparison with the "
        "score-level baseline is the secondary test of linking. Eight tests on the two tracks, "
        "nine with RQ6, and no correction for multiplicity.",
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
        note="Each seed's fusion model is fitted on that seed's out-of-fold inputs of the training "
        "split and applied to the test inputs of the same seed's model trained on the whole "
        "training split. The headline is seed 0, chosen in advance.",
    )


def policy_table(policy: Mapping[str, Any]) -> Table:
    """The review policy at its frozen thresholds."""
    rows: list[list[Cell]] = []
    for track, name in TRACK_NAMES.items():
        for budget, p in policy[track].items():
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
            "Frozen threshold",
            "Sent to review",
            "Share reviewed",
            "Forged reviewed",
            "Recall",
            "Precision of the reviewed",
        ),
        rows,
        note="A receipt is sent to review if its calibrated probability reaches the threshold "
        "Phase 5's check set on the validation split at each budget, applied unchanged, so the "
        "share sent to review here need not equal the budget; the first budget is the policy. "
        "Intervals: bootstrap over receipts with the threshold held fixed, 10,000 resamples, "
        "seed 0.",
    )


def _grounding(grounding: Mapping[str, Any]) -> list[Table]:
    rows: list[list[Cell]] = [
        [name, g["reports"], g["facts"], len(g["violations"]), g["rate"]]
        for track, name in TRACK_NAMES.items()
        if (g := grounding[track])
    ]
    tables = [
        Table(
            "Grounding of the audit reports",
            ("Track", "Reports", "Facts", "Violations", "Grounding rate"),
            rows,
            note="Every test receipt's audit report; the independent checker reads every number "
            "and box back from the text and looks it up in the assessment (ADR-0029, section "
            "12). Target: 100 %.",
        )
    ]
    violations: list[list[Cell]] = [
        [name, v] for track, name in TRACK_NAMES.items() for v in grounding[track]["violations"]
    ]
    if violations:
        tables.append(Table("Every grounding violation", ("Track", "Receipt: fact"), violations))
    return tables


def _tables(results: Mapping[str, Any], *, where: str) -> list[Table]:
    """Every table of one measurement of the test split."""
    tables = [success_table(results, where=where)]
    tables += [claims_table(name, results[track]) for track, name in TRACK_NAMES.items()]
    for track, name in TRACK_NAMES.items():
        tables.append(ablation_table(name, results[track], where=where))
    tables += [_seeds(results), policy_table(results["policy"])]
    tables.append(practical_table(results["T3"]["table"]))
    for track, name in TRACK_NAMES.items():
        tables += rq3_tables(
            name,
            results["rq3"][track],
            where="Per evidence: the mean contribution to the log-odds of the fusion model, and in "
            "brackets on how many receipts it contributes most. Caught: forged receipts among "
            f"the highest 10 % of the {where} receipts (expected counts where scores tie). "
            "Descriptive; groups are small.",
        )
    tables.append(false_alarm_table(results["false_alarms"], where=f"the highest 10 % of {where}"))
    tables += rq4_tables(results["rq4"], review="Hits among the receipts sent to review")
    tables += _grounding(results["grounding"])
    return tables


def render_test_report(results: Mapping[str, Any], commit: str | None) -> str:
    """The test-split report of the fused system."""
    tables = _tables(results, where="test")
    sensitivity = _tables(results["sensitivity"], where="test without copies")
    header = (
        "# The frozen system on the test split\n\n"
        "> Generated from the run record `fusion-test-check` by `docfraud final reports`. Do not "
        "edit by hand.\n\n"
        "The system frozen in Phase 5 and checked on the validation split, applied once, at "
        f"commit `{(commit or 'unknown')[:7]}`, to the {results['transcript']['receipts']} test "
        "receipts, as [ADR-0034](../../docs/decisions/0034-phase-7-preregistration.md) fixes. "
        "Nothing was fitted or chosen on them. The success criterion and the nine pre-named "
        "tests are the claims; everything else is descriptive.\n\n"
    )
    second = (
        "\n## Without the eight receipts that had copies\n\n"
        "Every result again on the test receipts without the eight that had a copy or twin "
        "elsewhere in the published dataset "
        "([ADR-0009](../../docs/decisions/0009-duplicate-receipts.md)); every forged receipt "
        "remains. Similar results show that the copies did not matter.\n\n"
    )
    return (
        header
        + "\n".join(render_table(t) for t in tables)
        + second
        + "\n".join(render_table(t, heading_level=4) for t in sensitivity)
    )


def render_track_r(results: Mapping[str, Any], commit: str | None, fix: str | None = None) -> str:
    """The report of track R: the frozen T2 model on the test images saved again as JPEG.

    ``fix`` is the tag of the fix its inputs were made after, if any, which the report discloses
    (ADR-0034, decision 2).
    """
    rows: list[list[Cell]] = []
    for name, found in [("original images", results["original"])] + [
        (f"saved as JPEG at quality {q}", results[q]["table"])
        for q in sorted((k for k in results if k != "original"), key=int, reverse=True)
    ]:
        rows.append(
            [
                name,
                ci(found["average_precision"]),
                ci(found["roc_auc"]),
                ci(found["recall_at_10pct"]),
                ci(found["recall_at_20pct"]),
                _f1(found),
            ]
        )
    difference: list[list[Cell]] = [
        [
            f"quality {q}",
            ci(results[q]["against_original"]),
            "yes" if results[q]["success"]["met"] else "no",
        ]
        for q in sorted((k for k in results if k != "original"), key=int, reverse=True)
    ]
    tables = [
        Table(
            "Fusion of T2 on the test images, as distributed and saved again",
            (
                "Images",
                "Average precision",
                "ROC-AUC",
                "Recall at 10 %",
                "Recall at 20 %",
                "F1 (95 % interval)",
            ),
            rows,
            note=f"{TEST_NOTE} Each image is rendered on white paper where transparent, converted "
            "to RGB and saved as JPEG with Pillow's default settings and no metadata, then read "
            "from the start of the T2 pipeline: OCR, the frozen field reader on the new text, "
            "system 2 at seed 0, system 1, copy-move and the links.",
        ),
        Table(
            "What saving the file again costs",
            (
                "Images",
                "Original minus saved, average precision (95 % interval)",
                "ADR-0013's two conditions would hold",
            ),
            difference,
            note="Paired bootstrap over receipts. Descriptive: track R enters no claim; it shows "
            "whether the headline rests on traits a forger removes by saving the file again "
            "([ADR-0003](../../docs/decisions/0003-content-only-headline.md)).",
        ),
    ]
    header = (
        "# Track R: the test images saved again\n\n"
        "> Generated from the run record `fusion-resaved-check` by `docfraud final reports`. Do "
        "not edit by hand.\n\n"
        f"Measured once at commit `{(commit or 'unknown')[:7]}` "
        "([ADR-0034](../../docs/decisions/0034-phase-7-preregistration.md), decision 4).\n\n"
    )
    if fix:
        header += (
            f"**A fix after the unlock.** The inputs were made after the one fix of Phase 7, at "
            f"the tag `{fix}`. Their first run stopped on the transcript of a saved image, a "
            "request the frozen reader had never read; the fix reads only the saved images' OCR "
            "text, which is all that track R uses. No model, setting, threshold or instruction "
            "changed (ADR-0034, decision 2; the [Phase 7 page](../README.md)).\n\n"
        )
    return header + "\n".join(render_table(t) for t in tables)


def _counts(counts: Mapping[str, int]) -> str:
    return ", ".join(f"{k} {v}" for k, v in sorted(counts.items(), key=lambda kv: -kv[1])) or "none"


def _largest(contributions: Sequence[Mapping[str, Any]]) -> str:
    return "; ".join(f"{c['group']} {c['contribution']:+.2f}" for c in contributions)


def render_errors(results: Mapping[str, Any], commit: str | None) -> str:
    """The report of the error analysis: every error of T2's review policy and its cause."""
    misses, alarms = results["misses"], results["false_alarms"]
    summary = Table(
        "The errors of the review policy",
        ("Errors", "Count", "By cause", "By edited entity", "By technique"),
        [
            [
                "forged receipts accepted",
                misses["count"],
                _counts(misses["by_cause"]),
                _counts(misses["by_entity"]),
                _counts(misses["by_technique"]),
            ],
            [
                "authentic receipts sent to review",
                alarms["count"],
                _counts(alarms["by_cause"]),
                None,
                None,
            ],
        ],
        note=f"T2 at the {results['policy']['budget'] * 100:.0f} % budget, threshold "
        f"{results['policy']['threshold']:.3f} from Phase 5's check, on {results['receipts']} test "
        f"receipts ({results['forged']} forged, {results['reviewed']} sent to review). Causes "
        "follow the first rule that applies "
        "([ADR-0034](../../docs/decisions/0034-phase-7-preregistration.md), decision 8). Of the "
        f"false alarms, {alarms['with_benign_marks']} carry handwritten or digital marks.",
    )
    rows: list[list[Cell]] = []
    for e in results["errors"]:
        if e["kind"] == "miss":
            about = (
                f"{', '.join(e['entities']) or 'no entity'}; {', '.join(e['techniques']) or '-'}; "
                f"{e['software'] or 'software not recorded'}"
            )
        else:
            marks = [
                kind
                for kind, flag in (
                    ("handwritten", e["handwritten_annotation"]),
                    ("digital", e["digital_annotation"]),
                )
                if flag
            ]
            about = f"marks: {', '.join(marks) or 'none'}"
        rows.append(
            [
                e["receipt"],
                e["kind"],
                e["cause"],
                e["probability"],
                about,
                _largest(e["largest_contributions"]),
            ]
        )
    every = Table(
        "Every error",
        ("Receipt", "Error", "Cause", "Probability", "What the data say", "Largest contributions"),
        rows,
        note="For a miss: the edited entities, the techniques and the editing software, from the "
        "annotations. Contributions are to the log-odds of forgery. One sentence per error, "
        "drafted from these and the audit report and checked by the owner, is on the Phase 7 "
        "page.",
    )
    header = (
        "# The errors of the system on the test split\n\n"
        "> Generated from the run record `fusion-test-errors` by `docfraud final reports`. Do not "
        "edit by hand.\n\n"
        f"Found once at commit `{(commit or 'unknown')[:7]}`.\n\n"
    )
    return header + "\n".join(render_table(t) for t in (summary, every))
