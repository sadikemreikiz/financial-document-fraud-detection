"""Generated reports of the LLM reader: comparison, rounds, gold set B and false alarms."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from docfraud.evaluation.bootstrap import Estimate
from docfraud.evaluation.extraction import (
    RULE_INPUTS_EXACT,
    DocumentScore,
    estimate,
    render_extraction_report,
)
from docfraud.llm.adjudication import Kind
from docfraud.llm.decision import ModelStep, ReaderDecision
from docfraud.llm.false_alarms import TrackResult
from docfraud.llm.gold_b import AdjudicationCounts, Latency
from docfraud.markdown import Cell, Table, render_table


@dataclass(frozen=True, slots=True)
class ReaderRun:
    """One reader on one gold set: scores and rejections per text source, and what it cost.

    ``rejections`` and ``cost_usd`` are ``None`` for a reader that names no labels and costs
    nothing, the parser.
    """

    name: str
    scores: Mapping[str, Sequence[DocumentScore]]
    rejections: Mapping[str, int] | None = None
    cost_usd: Decimal | None = None


def _exact(scores: Sequence[DocumentScore]) -> str:
    exact = sum(s.rule_inputs_exact for s in scores)
    return f"{exact} of {len(scores)}"


def _interval(scores: Sequence[DocumentScore]) -> str:
    found = estimate(RULE_INPUTS_EXACT, scores)
    if found is None:
        return "\N{EM DASH}"
    return f"{found.value:.3f} [{found.low:.3f}, {found.high:.3f}]"


def _cost(run: ReaderRun) -> Cell:
    return None if run.cost_usd is None else f"{run.cost_usd:.4f}"


def render_model_comparison(
    header: str, runs: Sequence[ReaderRun], steps: Sequence[ModelStep], chosen: str
) -> str:
    """The report of the model comparison on gold set A (ADR-0025, decision 2)."""
    tracks = list(runs[0].scores)
    rows: list[list[Cell]] = []
    for run in runs:
        rejections = run.rejections or {}
        rows.append(
            [
                run.name,
                *(_exact(run.scores[track]) for track in tracks),
                *(
                    rejections.get(track) if run.rejections is not None else None
                    for track in tracks
                ),
                _cost(run),
            ]
        )
    readers = Table(
        "Receipts with all rule inputs exact",
        (
            "Reader",
            *(f"Exact: {track}" for track in tracks),
            *(f"Rejections: {track}" for track in tracks),
            "Cost (USD)",
        ),
        rows,
        note="Rule inputs: amounts, items and tax summary rows (ADR-0025, decision 5). Gold set "
        "A is the development set, so every number here is optimistic; parser v2 is shown for "
        "context and was developed on other receipts.",
    )
    ladder = Table(
        "Steps up the ladder",
        ("Smaller", "Larger", "Difference [95 % interval]", "Larger clearly better"),
        [
            (
                step.smaller,
                step.larger,
                f"{step.difference.value:+.3f} [{step.difference.low:+.3f}, "
                f"{step.difference.high:+.3f}]",
                step.larger_is_clearly_better,
            )
            for step in steps
        ],
        note="Difference in the share of (receipt, text source) pairs with all rule inputs "
        "exact, larger minus smaller; paired bootstrap over receipts (10,000 resamples). A "
        "larger model is taken only if the interval lies above zero.",
    )
    return (
        header
        + render_table(readers)
        + "\n"
        + render_table(ladder)
        + f"\n**Chosen model:** `{chosen}`.\n"
    )


def _alarm_table(false_alarms: Mapping[str, Mapping[str, int]], authentic: int) -> Table:
    """False alarms of the accounting rules per set of fields and text source."""
    tracks = list(next(iter(false_alarms.values())))
    return Table(
        "Accounting rules on authentic receipts",
        ("Fields", *tracks),
        [(name, *(by_track[t] for t in tracks)) for name, by_track in false_alarms.items()],
        note=f"Authentic receipts, of {authentic}, on which at least one accounting rule fails: "
        "false alarms caused by the fields alone (ADR-0025, decision 6).",
    )


def render_round(
    header: str,
    run: ReaderRun,
    forged: frozenset[str],
    false_alarms: Mapping[str, Mapping[str, int]],
    authentic: int,
) -> str:
    """The report of one development round on gold set A, both text sources.

    ``false_alarms`` holds, per set of fields, the count per text source.
    """
    tracks = list(run.scores)
    rejections = run.rejections or {}
    primary = Table(
        "Primary metric, rejections and cost",
        ("Text source", "Exact", "Share [95 % interval]", "Rejections"),
        [
            (track, _exact(run.scores[track]), _interval(run.scores[track]), rejections.get(track))
            for track in tracks
        ],
        note=f"Receipts whose rule inputs are all exact (ADR-0025, decision 5). Cost of the "
        f"round: {_cost(run) or 0} USD.",
    )
    tables = render_table(primary) + "\n" + render_table(_alarm_table(false_alarms, authentic))
    return render_extraction_report(run.scores, forged, header + tables + "\n")


KIND_NAMES = {
    Kind.AMOUNTS: "Amounts of one role",
    Kind.ITEMS: "Item list",
    Kind.TAX_ROWS: "Tax summary rows",
}


def render_gold_b(  # noqa: PLR0913 - one argument per part of the report
    header: str,
    *,
    decisions: Sequence[ReaderDecision],
    scores: Mapping[str, Mapping[str, Sequence[DocumentScore]]],
    counts: Sequence[AdjudicationCounts],
    readings: Sequence[str],
    false_alarms: Mapping[str, Mapping[str, int]],
    rejections: Mapping[str, int],
    latencies: Mapping[str, Latency | None],
    forged: frozenset[str],
    authentic: int,
) -> str:
    """The report of gold set B (ADR-0025, decisions 5 and 6).

    ``scores`` holds, per text source, the scores of parser v2 and then of the LLM reader;
    ``false_alarms`` holds, per set of fields, the count per text source.
    """
    decision = Table(
        "Decision",
        (
            "Text source",
            *(f"Exact: {reader}" for reader in next(iter(scores.values()))),
            "Difference [95 % interval]",
            "Exact for the LLM only",
            "Exact for parser v2 only",
            "McNemar p",
            "LLM replaces parser v2",
        ),
        [
            (
                d.source,
                *(_exact(s) for s in scores[d.source].values()),
                f"{d.difference.value:+.3f} [{d.difference.low:+.3f}, {d.difference.high:+.3f}]",
                d.llm_only,
                d.parser_only,
                f"{d.p_value:.4f}",
                d.passes,
            )
            for d in decisions
        ],
        note="Receipts whose rule inputs (amounts, items and tax summary rows) are all exact "
        "against the adjudicated gold set. Difference: LLM minus parser v2 in that share, paired "
        "bootstrap over receipts (10,000 resamples). The LLM reader replaces parser v2 for a "
        "text source only if the interval lies above zero and the exact McNemar test on the "
        "receipts exact for one reader only gives p < 0.05, two-sided (ADR-0025, decision 5).",
    )
    adjudication = Table(
        "Adjudication",
        (
            "Question",
            "Asked",
            "First answer equalled a reading",
            "Gold changed",
            *(f"Agreed: {name}" for name in readings),
        ),
        [
            (
                KIND_NAMES[c.kind],
                c.questions,
                c.first_matched,
                c.gold_changed,
                *(c.agreed.get(name, 0) for name in readings),
            )
            for c in counts
        ],
        note="One question per receipt and amount role in dispute, and one per receipt whose "
        "item list or tax summary is in dispute (ADR-0026). Agreed: questions whose final answer "
        "equals that reading. Values on which the gold set and all four readings agree were not "
        "asked, so an error they share stays in the gold set.",
    )
    tracks = list(scores)
    alarms = _alarm_table(false_alarms, authentic)
    rejected = Table(
        "LLM reader: rejections and time to answer",
        ("Text source", "Rejections", "Timed answers", "Median (s)", "95th percentile (s)"),
        [
            (
                track,
                rejections[track],
                *(
                    (None, None, None)
                    if (timed := latencies[track]) is None
                    else (timed.timed, f"{timed.median:.2f}", f"{timed.p95:.2f}")
                ),
            )
            for track in tracks
        ],
        note="Rejections: references in the answers that do not hold what their field needs; "
        "the field is left empty. Time: from sending a request on its own to its answer, the "
        "client's retries included (ADR-0025, decision 6).",
    )
    sections = [
        render_extraction_report(scores[track], forged, f"## {track}\n\n") for track in tracks
    ]
    return (
        header
        + "\n".join(render_table(t) for t in (decision, adjudication, alarms, rejected))
        + "\n"
        + "\n".join(sections)
    )


def _difference(estimate: Estimate) -> str:
    return f"{estimate.value:+.3f} [{estimate.low:+.3f}, {estimate.high:+.3f}]"


def render_false_alarms(
    header: str, results: Mapping[str, TrackResult], *, authentic: int, forged: int
) -> str:
    """The report of ADR-0027: per text source, whether the LLM reader replaces parser v2."""
    readers = list(next(iter(results.values())).alarms)
    decision = Table(
        "Decision",
        (
            "Text source",
            *(f"False alarms: {reader}" for reader in readers),
            "False-alarm difference [95 % interval]",
            "Parser v2 only",
            "LLM only",
            "McNemar p",
            *(f"Caught: {reader}" for reader in readers),
            "Catch difference [95 % interval]",
            "Catch rate falls by less",
            "LLM replaces parser v2",
        ),
        [
            (
                track,
                *(f"{a.false_alarms} of {authentic}" for a in r.alarms.values()),
                _difference(r.decision.false_alarms),
                r.decision.parser_only,
                r.decision.llm_only,
                f"{r.decision.p_value:.4f}",
                *(f"{a.catches} of {forged}" for a in r.alarms.values()),
                _difference(r.decision.catches),
                r.decision.catches_hold,
                r.decision.passes,
            )
            for track, r in results.items()
        ],
        note="False alarms: authentic receipts on which at least one accounting rule fails; "
        "caught: forged receipts on which at least one fails. Differences: LLM minus parser v2 in "
        "the share, paired bootstrap over receipts (10,000 resamples). Parser v2 only, LLM only: "
        "authentic receipts alarmed with that reader's fields only. The LLM reader replaces "
        "parser v2 for a text source only if the false-alarm interval lies below zero, the exact "
        "McNemar test gives p < 0.05, two-sided, in its favour, and its catch rate falls by less "
        "than its false-alarm rate (ADR-0027).",
    )
    separation = Table(
        "Catch rate minus false-alarm rate",
        ("Text source", *readers, "Difference [95 % interval]"),
        [
            (
                track,
                *(
                    f"{a.catches / forged - a.false_alarms / authentic:+.3f}"
                    for a in r.alarms.values()
                ),
                _difference(r.separation),
            )
            for track, r in results.items()
        ],
        note="How far apart the rules hold forged and authentic receipts with each reader's "
        "fields. Difference: LLM minus parser v2, receipts resampled within each class (10,000 "
        "resamples). Reported, not decisive.",
    )
    by_rule = [
        Table(
            f"Rules that fail, {track}",
            (
                "Rule",
                *(f"Authentic: {reader}" for reader in readers),
                *(f"Forged: {reader}" for reader in readers),
            ),
            [
                (
                    name,
                    *(r.rules[reader][0][name] for reader in readers),
                    *(r.rules[reader][1][name] for reader in readers),
                )
                for name in r.rules[readers[0]][0]
            ],
            note="Receipts on which each rule fails; a receipt can fail several. Reported, not "
            "decisive.",
        )
        for track, r in results.items()
    ]
    rejections = Table(
        "LLM reader: receipts with rejected references",
        ("Text source", "Receipts"),
        [(track, r.rejected) for track, r in results.items()],
        note="Receipts on which at least one reference of the answer did not hold what its field "
        "needs; such a field is left empty.",
    )
    return header + "\n".join(render_table(t) for t in (decision, separation, *by_rule, rejections))
