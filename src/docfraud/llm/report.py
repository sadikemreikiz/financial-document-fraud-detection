"""Generated reports of the LLM reader: the model comparison and the development rounds."""

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from decimal import Decimal

from docfraud.evaluation.extraction import (
    RULE_INPUTS_EXACT,
    DocumentScore,
    estimate,
    render_extraction_report,
)
from docfraud.llm.decision import ModelStep
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


def render_round(header: str, run: ReaderRun, forged: frozenset[str]) -> str:
    """The report of one development round on gold set A, both text sources."""
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
    return render_extraction_report(run.scores, forged, header + render_table(primary) + "\n")
