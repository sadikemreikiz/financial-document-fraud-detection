"""The report of the agent's run on the validation split, rendered from its run record."""

from collections.abc import Mapping
from typing import Any

from docfraud.fusion.development_report import ci
from docfraud.llm.agent_validation import RUN_NAME
from docfraud.markdown import Cell, Table, render_table


def _yes(flag: bool | None) -> str | None:
    return None if flag is None else ("yes" if flag else "no")


def _summary(results: Mapping[str, Any], split: str) -> Table:
    g, cost, seconds = results["grounding"], results["cost_usd"], results["seconds"]
    rows: list[list[Cell]] = [
        ["Case files in the fixed form", f"{g['case_files']} of {results['receipts']}"],
        ["Grounding rate", f"{g['rate']:.4f} ({len(g['violations'])} of {g['facts']} facts)"],
        [
            "Cost per case, median (95th percentile), USD",
            f"{cost['median']:.4f} ({cost['p95']:.4f})",
        ],
        ["Cost of all cases, USD", f"{cost['total']:.4f}"],
    ]
    if seconds["median"] is not None:
        rows.append(
            [
                "Time per case, median (95th percentile), s",
                f"{seconds['median']:.1f} ({seconds['p95']:.1f}), {seconds['timed']} timed",
            ]
        )
    return Table(
        f"The agent on the {split} split",
        ("Measure", "Result"),
        rows,
        note="A fact is a number or a box of a rendered case file; it is grounded if a tool "
        "output the agent received for that receipt holds it (ADR-0031, section 6). The cost and "
        "time of a case are those of all its steps; a case is timed if every step was.",
    )


def _pointing(results: Mapping[str, Any]) -> Table | None:
    found = [
        (name, results[key])
        for name, key in (
            ("the agent's first observation", "first_location"),
            ("the audit report's first region", "report_first_region"),
        )
        if results[key] is not None
    ]
    if not found:
        return None
    rows: list[list[Cell]] = [
        [
            name,
            r["receipts"],
            r["located"],
            r["hits"],
            ci(r["hit_rate"]),
            r["baseline"],
            _yes(r["detects"]),
        ]
        for name, r in found
    ]
    return Table(
        "Where it points, on the forged receipts with region annotations",
        (
            "Location",
            "Receipts",
            "Located",
            "Hits",
            "Hit rate (95 % interval)",
            "Random baseline",
            "Lower bound above the baseline",
        ),
        rows,
        note="A location hits if it shares a pixel with an annotated modified rectangle or copy "
        "source; one without a location is a miss with a baseline of 0. The baseline places a "
        "box of the same size at random on the receipt's text (ADR-0015, ADR-0030). The audit "
        "report's first region may hold two boxes, a copy-move pair, and hits if either does; "
        "a case file's observation holds one box. Descriptive (ADR-0031, section 6). Interval: "
        "bootstrap over receipts, 10,000 resamples, seed 0.",
    )


def _violations(results: Mapping[str, Any]) -> Table | None:
    found = results["grounding"]["violations"]
    if not found:
        return None
    rows: list[list[Cell]] = [list(v.split(": ", 1)) for v in found]
    return Table("Grounding violations", ("Receipt", "Fact not found in the tool outputs"), rows)


def _failures(results: Mapping[str, Any]) -> Table | None:
    rows: list[list[Cell]] = [[k, n] for k, n in sorted(results["failures"].items())]
    if not rows:
        return None
    return Table("Cases without a case file in the fixed form", ("Failure", "Cases"), rows)


def _cases(results: Mapping[str, Any], files: str) -> Table:
    rows: list[list[Cell]] = []
    for c in results["cases"]:
        pointed, report = c["first_location"], c["report_first_region"]
        rows.append(
            [
                c["receipt"],
                "forged" if c["forged"] else "authentic",
                c["failure"] or "yes",
                c["facts"],
                None if c["violations"] is None else len(c["violations"]),
                None if pointed is None else _yes(pointed["hit"]),
                None if report is None else _yes(report["hit"]),
                c["tool_calls"],
                c["tokens"],
                f"{c['cost_usd']:.4f}",
                None if c["seconds"] is None else f"{c['seconds']:.1f}",
            ]
        )
    return Table(
        "Every case",
        (
            "Receipt",
            "Label",
            "Fixed form",
            "Facts",
            "Violations",
            "First observation hits",
            "Report's first region hits",
            "Tool calls",
            "Tokens",
            "Cost, USD",
            "Time, s",
        ),
        rows,
        note=f"The case files are in [`{files}`]({files}), one per receipt.",
    )


def render_report(  # noqa: PLR0913 - the record, and where and how it was made
    results: Mapping[str, Any],
    config: Mapping[str, Any],
    commit: str | None,
    *,
    split: str = "validation",
    run_name: str = RUN_NAME,
    command: str = "docfraud agent validation-report",
) -> str:
    """The report of the investigation agent on the validation split, or on the test split.

    The validation run records the policy in its configuration, the test run in its results,
    where the policy of Phase 5's check is applied unchanged.
    """
    frozen = config["frozen"]
    policy = config.get("policy") or results["policy"]
    unchanged = "" if split == "validation" else ", applied unchanged"
    tables = [
        t
        for t in (
            _summary(results, split),
            _pointing(results),
            _violations(results),
            _failures(results),
            _cases(results, f"{split}/"),
        )
        if t is not None
    ]
    header = (
        f"# The investigation agent on the {split} split\n\n"
        f"> Generated from the run record `{run_name}` by `{command}`. "
        "Do not edit by hand.\n\n"
        "The frozen agent "
        "([ADR-0031](../../../docs/decisions/0031-llm-detector-and-agent.md), sections 5 and 6, "
        "and [ADR-0033](../../../docs/decisions/0033-agent-development-rule.md)): version "
        f"{frozen['version']} of its instruction, `{frozen['settings']['model']}` at "
        f"{frozen['settings']['effort']} effort, at most 8 tool calls and "
        f"{frozen['token_budget']:,} tokens per case, frozen at `{frozen['commit'][:7]}`. It ran "
        f"once, at commit `{(commit or 'unknown')[:7]}`, on the {results['receipts']} {split} "
        "receipts that the review policy of track T2 sends to review (calibrated probability at "
        f"least {policy['threshold']:.3f}, the {policy['budget'] * 100:.0f} % budget of Phase 5's "
        f"check{unchanged}); {results['forged']} of them are forged. Its case files state no "
        "verdict: the reviewer decides.\n\n"
    )
    return header + "\n".join(render_table(t) for t in tables)
