"""The report of where the pixel systems and copy-move point on the test split."""

from collections.abc import Mapping
from typing import Any

from docfraud.fusion.development_report import ci
from docfraud.markdown import Cell, Table, render_table

NAMES = {"2": "system 2, synthetic forgeries", "1": "system 1, JPEG lattice"}


def _maybe(e: Mapping[str, float] | None) -> Cell:
    return None if e is None else ci(e)


def _guard(found: Mapping[str, Any] | None) -> list[Cell]:
    if found is None:
        return [0, None, None, None]
    verdict = "points at forgeries" if found["detects"] else "probable shortcut"
    return [found["receipts"], ci(found["hit_rate"]), found["baseline"], verdict]


def render_report(results: Mapping[str, Any], commit: str | None) -> str:
    """The test-split localisation report of systems 2 and 1 and of copy-move."""
    pixel: list[list[Cell]] = []
    guard: list[list[Cell]] = []
    for key, name in NAMES.items():
        r = results[key]
        pixel.append(
            [
                name,
                r["threshold"],
                r["forged_with_regions"],
                _maybe(r["pixel_average_precision"]),
                r["chance"],
                _maybe(r["f1"]),
                _maybe(r["iou"]),
            ]
        )
        guard.append([name, "edits and copy sources", *_guard(r["guard"])])
        guard.append([name, "copy sources", *_guard(r["sources"])])
    copy = results["copy_move"]
    guard.append(["copy-move's pair", "edits and copy sources", *_guard(copy["guard"]["all"])])
    sources: list[list[Cell]] = [
        [
            what,
            found["receipts"],
            ci(found["rate"]),
            found["baseline"],
            "yes" if found["above_baseline"] else "no",
        ]
        for what, found in (
            ("either region of the pair on a copy source", copy["copy_sources"]["either_region"]),
            ("the pair on an edit and its source", copy["copy_sources"]["pair"]),
        )
    ]
    tables = [
        Table(
            "Pixel metrics",
            (
                "System",
                "Frozen pixel threshold",
                "Forged with regions",
                "Pixel AP (95 % interval)",
                "Chance",
                "F1 (95 % interval)",
                "IoU (95 % interval)",
            ),
            pixel,
            note="Per forged test receipt with region annotations, then averaged with equal "
            "weight; pixels of a copy source outside every modified rectangle are ignored "
            "(ADR-0015, section 5). F1 and IoU at the pixel threshold chosen on the forged "
            "validation receipts before the test split was unlocked. Intervals by the bootstrap "
            "over receipts.",
        ),
        Table(
            "Where the top region falls",
            (
                "Map",
                "Targets",
                "Receipts",
                "Hit rate (95 % interval)",
                "Random baseline",
                "Verdict",
            ),
            guard,
            note="The top region of each map, or copy-move's pair, hits if it shares a pixel "
            "with a target; the baseline places boxes of the same size at random on the "
            "receipt's text (ADR-0015, ADR-0030). It points at forgeries if the lower bound of "
            "its hit rate exceeds the baseline (ADR-0014, decision 4).",
        ),
        Table(
            "Copy-move and the sources of copies",
            ("What hits", "Receipts", "Rate (95 % interval)", "Random baseline", "Above it"),
            sources,
            note="Copy-move source localisation (evaluation protocol, section 5), on the forged "
            "test receipts with an annotated copy source, as Phase 5 measured the clean folds.",
        ),
    ]
    header = (
        "# Where the pixel systems and copy-move point on the test split\n\n"
        "> Generated from the run record `forensics-test-localisation` by `docfraud final "
        "reports`. Do not edit by hand.\n\n"
        f"Measured once at commit `{(commit or 'unknown')[:7]}`, with the maps fusion reads "
        "([ADR-0034](../../docs/decisions/0034-phase-7-preregistration.md), decisions 1 and 5). "
        "Descriptive.\n\n"
    )
    return header + "\n".join(render_table(t) for t in tables)
