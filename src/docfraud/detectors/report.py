"""The generated accounting-rules report, rendered from run records (ADR-0012)."""

from typing import Any

from docfraud.detectors.rules import RULES
from docfraud.detectors.scoring import FEATURES
from docfraud.evaluation.runs import RunRecord
from docfraud.markdown import Cell, Table, render_table

TRACKS = (("transcript", "T1"), ("ocr", "T2"))
METRICS = (
    ("precision", "Precision"),
    ("recall", "Recall"),
    ("f1", "F1"),
    ("average_precision", "Average precision"),
    ("roc_auc", "ROC-AUC"),
    ("recall_at_10pct_review", "Recall at 10 % review"),
    ("recall_at_20pct_review", "Recall at 20 % review"),
)
DIFFERENCES = (
    "f1",
    "average_precision",
    "roc_auc",
    "recall_at_10pct_review",
    "recall_at_20pct_review",
)
NAMES = dict(METRICS)


def _pct(e: dict[str, float]) -> str:
    return f"{e['value'] * 100:.1f} [{e['low'] * 100:.1f}, {e['high'] * 100:.1f}]"


def _raw(e: dict[str, float] | None) -> str:
    if e is None:
        return "\N{EM DASH}"
    return f"{e['value']:.3f} [{e['low']:.3f}, {e['high']:.3f}]"


def _counts(r: dict[str, Any]) -> str:
    c = r["counts"]
    return f"{c['tp']} / {c['fp']} / {c['fn']}"


def _share(c: dict[str, int]) -> str:
    return f"{c['fail']} / {c['applicable']}"


def _run(record: RunRecord) -> str:
    code = record.code.commit[:7] if record.code.commit else "unknown"
    return (
        f"`{record.run_id}` (commit {code}{', uncommitted changes' if record.code.dirty else ''})"
    )


def _detection(results: dict[str, Any], bow: dict[str, Any], title: str, note: str) -> Table:
    """Detection metrics of every system side by side.

    "Flag everything" is the floor: at this class balance it reaches an F1 near 0.29 without any
    information, so F1 is read against it, and the ranking metrics carry most of the evidence.
    """
    systems = [
        *((f"Rules {t}", results[k]["learned"]) for k, t in TRACKS),
        *((f"Any rule fails {t}", results[k]["any_fail"]) for k, t in TRACKS),
        ("BoW + LR", bow),
        ("Flag everything", results["transcript"]["flag_all"]),
    ]
    rows: list[list[Cell]] = [
        [label, *(_pct(r["metrics"][key]) for _, r in systems)] for key, label in METRICS
    ]
    rows.append(
        ["Forged found / authentic flagged / forged missed", *(_counts(r) for _, r in systems)]
    )
    for key, label in (
        ("brier", "Brier score"),
        ("expected_calibration_error", "Calibration error"),
    ):
        rows.append([label, *(_raw(r["metrics"].get(key)) for _, r in systems)])
    return Table(title, ("Metric", *(name for name, _ in systems)), rows, note=note)


def _versus(results: dict[str, Any], title: str) -> Table:
    rows: list[list[Cell]] = []
    for key in DIFFERENCES:
        cells: list[Cell] = [NAMES[key]]
        for track, _ in TRACKS:
            d = results[track]["versus_bow"][key]
            star = " *" if d["excludes_zero"] else ""
            cells.append(
                f"{d['value'] * 100:+.1f} [{d['low'] * 100:+.1f}, {d['high'] * 100:+.1f}]{star}"
            )
        rows.append(cells)
    return Table(
        title,
        ("Metric", "Rules T1 - BoW + LR", "Rules T2 - BoW + LR"),
        rows,
        note="Percentage points; paired stratified bootstrap over the same receipts. * marks an "
        "interval that excludes zero.",
    )


def _per_rule(results: dict[str, Any], title: str) -> Table:
    rows: list[list[Cell]] = [
        [
            rule,
            *(
                _share(results[track]["rules"][rule][label])
                for track, _ in TRACKS
                for label in ("authentic", "forged")
            ),
        ]
        for rule in RULES
    ]
    return Table(
        title,
        ("Rule", "T1 authentic", "T1 forged", "T2 authentic", "T2 forged"),
        rows,
        note="Receipts on which the rule failed / receipts on which it applied.",
    )


def _breakdown_tables(scope: dict[str, Any], label: str) -> list[Table]:
    """Forged receipts found and authentic receipts flagged, per group (protocol, section 7)."""
    systems = [(f"Any rule fails {t}", scope[k]["any_fail"]) for k, t in TRACKS] + [
        (f"Top 10 % review {t}", scope[k]["learned_at_budget"]) for k, t in TRACKS
    ]
    tables = []
    first = scope["transcript"]["any_fail"]
    for kind, title in (
        ("technique", "forgery technique"),
        ("entity", "changed information"),
        ("software", "editing software"),
    ):
        rows: list[list[Cell]] = [
            [
                group,
                int(counts["receipts"]),
                *(f"{s[1]['forged'][kind][group]['found']:.1f}" for s in systems),
            ]
            for group, counts in first["forged"][kind].items()
        ]
        tables.append(
            Table(
                f"{label}: forged receipts found, by {title}",
                ("Group", "Forged receipts", *(name for name, _ in systems)),
                rows,
                note="A receipt counts in every group it belongs to. Top 10 % review: the learned "
                "scorer's 10 % most suspicious receipts; receipts tied at the cut-off count in "
                "part, so counts can be fractional. Groups are small: descriptive only.",
            )
        )
    tables.append(
        Table(
            f"{label}: authentic receipts flagged, by benign annotation",
            ("Annotation", "Authentic receipts", *(name for name, _ in systems)),
            [
                [
                    group,
                    int(counts["receipts"]),
                    *(f"{s[1]['authentic'][group]['flagged']:.1f}" for s in systems),
                ]
                for group, counts in first["authentic"].items()
            ],
            note="Handwritten and digital annotations (case numbers, notes, stamps) are benign "
            "and must not raise alarms.",
        )
    )
    return tables


def render_rules_report(
    cv: RunRecord,
    bow_cv: RunRecord,
    validation: RunRecord | None,
    breakdowns: RunRecord | None = None,
) -> str:
    """Correct-reading level, cross-validated development result and the validation result."""
    r = cv.results
    gold = r["gold_fields"]
    parts = [
        "# Accounting rules\n\n"
        "Generated by `docfraud rules report` from run records; do not edit by hand.\n\n"
        "The rules ([`docfraud.detectors.rules`](../../src/docfraud/detectors/rules.py)) check "
        "that a receipt's printed amounts add up. A logistic regression over their outcomes "
        "(fail or not applicable, per rule) gives the score. Development and measurement follow "
        "[ADR-0012](../../docs/decisions/0012-accounting-rules-evaluation.md): the rules were "
        "written by looking at training receipts, so results on the training split are "
        "optimistic, and the unbiased result is the single measurement on the validation "
        "split. Fields are read by parser v2 from the dataset transcripts (T1) or our OCR "
        "(T2).\n\n",
        render_table(
            Table(
                "With correct reading: gold fields of 80 training receipts",
                ("Rule", "Authentic", "Forged"),
                [
                    *(
                        [
                            rule,
                            _share(gold["rules"][rule]["authentic"]),
                            _share(gold["rules"][rule]["forged"]),
                        ]
                        for rule in RULES
                    ),
                    [
                        "Any rule fails",
                        f"{gold['any_fail']['counts']['fp']} / "
                        f"{gold['any_fail']['n_documents'] - gold['any_fail']['n_forged']}",
                        f"{gold['any_fail']['counts']['tp']} / {gold['any_fail']['n_forged']}",
                    ],
                ],
                note="Receipts on which the rule failed / receipts on which it applied. The "
                "fields are the hand annotations, so every failure is a property of the receipt, "
                "not a reading error. Descriptive only: the rules were developed on these "
                "receipts.",
            )
        ),
        "\n",
        render_table(
            _detection(
                r,
                bow_cv.results,
                "Training split, cross-validated (development, optimistic)",
                f"Out-of-fold scores of the {r['transcript']['learned']['n_documents']} training "
                f"receipts ({r['transcript']['learned']['n_forged']} forged), 5 stored folds; "
                "percentages with 95 % stratified bootstrap intervals. 'Flag everything' marks "
                "every receipt as forged: the floor for F1 and the chance level for ranking. "
                "Operating thresholds "
                f"(F1-optimal out of fold): T1 {r['transcript']['threshold']:.3f}, T2 "
                f"{r['ocr']['threshold']:.3f}. Runs {_run(cv)} and {_run(bow_cv)}.",
            )
        ),
        "\n",
        render_table(_versus(r, "Rules compared with BoW + LR, cross-validated")),
        "\n",
        render_table(_per_rule(r, "Rule outcomes on the training split")),
        "\n",
        render_table(
            Table(
                "Learned weights (fit on the whole training split)",
                ("Outcome", "T1", "T2"),
                [
                    [name, *(f"{r[track]['weights'][name]:+.2f}" for track, _ in TRACKS)]
                    for name in FEATURES
                ],
                note="Change in the log-odds of forgery when the outcome occurs, relative to the "
                "rule passing.",
            )
        ),
    ]
    if validation is None:
        parts.append(
            "\n### Validation split\n\nNot measured yet. It is measured once, after the rules "
            "and the scoring are frozen at a commit (ADR-0012).\n"
        )
    else:
        v = validation.results
        parts += [
            "\n",
            render_table(
                _detection(
                    v,
                    v["bow"],
                    "Validation split, measured once (unbiased)",
                    f"{v['bow']['n_documents']} validation receipts ({v['bow']['n_forged']} "
                    "forged). Scorers and BoW + LR fitted on the whole training split; rule "
                    "thresholds from the training cross-validation, BoW + LR at 0.5. Run "
                    f"{_run(validation)}.",
                )
            ),
            "\n",
            render_table(_versus(v, "Rules compared with BoW + LR on the validation split")),
            "\n",
            render_table(_per_rule(v, "Rule outcomes on the validation split")),
        ]
    if breakdowns is not None:
        b = breakdowns.results
        parts.append(f"\n## Breakdowns\n\nFrom run {_run(breakdowns)}.\n\n")
        for scope, label in (
            ("validation", "Validation"),
            ("training", "Training, cross-validated"),
        ):
            if scope in b:
                parts += [render_table(t) + "\n" for t in _breakdown_tables(b[scope], label)]
    return "".join(parts)
