"""Figures of the clean measurement (ADR-0020) and of Phase 5's validation check (ADR-0029).

Every figure is written as an SVG for a light and a dark page. To reports/forensics/figures: the
document-level average precision of every Phase 4 system with its 95 % interval against chance,
and how often each system's top region hits an edit against a random box of the same size; the
numbers are computed by docfraud.forensics.clean_report from what the measurement kept, as in
its report. To reports/fusion/figures: the average precision of the fusion model, the
score-level baseline and each single source on the validation split, for both text sources,
read from the validation check's run record. The reports' tables are the figures' table view.

matplotlib is not a project dependency; ``make figures`` supplies a pinned version. The SVGs carry
no date and a fixed hash salt, so that the same numbers give the same files.

Usage: ``make figures``
"""

from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib as mpl

mpl.use("svg")
import matplotlib.pyplot as plt

from docfraud.evaluation.bootstrap import Estimate
from docfraud.evaluation.runs import JsonRunStore
from docfraud.forensics.clean import PROBE
from docfraud.forensics.clean_report import (
    MAP_SYSTEMS,
    CleanData,
    Sources,
    guard,
    load,
    localised,
    pooled,
    precision_interval,
    probe_arrays,
)

ROOT = Path(__file__).resolve().parents[1]
FIGURES = ROOT / "reports" / "forensics" / "figures"
FUSION_FIGURES = ROOT / "reports" / "fusion" / "figures"
VALIDATION_RUN = "fusion-validation-check"

LABELS = {
    1: "System 1 · JPEG lattice",
    2: "System 2 · synthetic forgeries",
    3: "System 3 · real forgeries",
    4: "System 4 · both",
    5: "Probe · global trace (diagnostic)",
    6: "System 6 · TruFor",
}


@dataclass(frozen=True, slots=True)
class Theme:
    """The chart colours of one page theme (the reference palette of the dataviz method)."""

    name: str
    surface: str
    primary: str
    secondary: str
    muted: str
    grid: str
    axis: str
    series: tuple[str, str]  # categorical slots 1 and 2


LIGHT = Theme(
    "light",
    "#fcfcfb",
    "#0b0b0b",
    "#52514e",
    "#898781",
    "#e1e0d9",
    "#c3c2b7",
    ("#2a78d6", "#eb6834"),
)
DARK = Theme(
    "dark", "#1a1a19", "#f0efec", "#c3c2b7", "#898781", "#2c2c2a", "#383835", ("#3987e5", "#d95926")
)


def _style(theme: Theme) -> dict[str, Any]:
    return {
        "svg.hashsalt": "docfraud",
        "svg.fonttype": "path",
        "font.family": ["Helvetica Neue", "Helvetica", "Arial", "DejaVu Sans"],
        "font.size": 9,
        "figure.facecolor": theme.surface,
        "axes.facecolor": theme.surface,
        "axes.edgecolor": theme.axis,
        "axes.labelcolor": theme.secondary,
        "xtick.color": theme.muted,
        "ytick.color": theme.primary,
        "text.color": theme.primary,
    }


ROW = 0.42  # inches per row of marks


def _axes(theme: Theme, rows: int, top: float, bottom: float) -> tuple[Any, Any]:
    """A figure with ``top`` and ``bottom`` inches around its rows, for titles and notes."""
    height = ROW * rows + top + bottom
    figure, axes = plt.subplots(figsize=(7.2, height))
    figure.subplots_adjust(left=0.34, right=0.97, top=1 - top / height, bottom=bottom / height)
    for side in ("top", "right", "left"):
        axes.spines[side].set_visible(False)
    axes.spines["bottom"].set_color(theme.axis)
    axes.spines["bottom"].set_linewidth(1)
    axes.tick_params(axis="y", length=0, pad=8)
    axes.tick_params(axis="x", length=0, labelcolor=theme.muted)
    axes.grid(axis="x", color=theme.grid, linewidth=1, linestyle="-")
    axes.set_axisbelow(True)
    axes.set_xlim(0, 1)
    axes.set_xticks([0, 0.2, 0.4, 0.6, 0.8, 1.0])
    axes.set_ylim(rows - 0.4, -0.6)
    return figure, axes


def _interval(  # noqa: PLR0913 - one mark and where it goes
    axes: Any,
    row: float,
    estimate: Estimate,
    colour: str,
    theme: Theme,
    *,
    label: bool = True,
    marker: str = "o",
) -> None:
    axes.hlines(row, estimate.low, estimate.high, color=colour, linewidth=2, capstyle="round")
    axes.plot(
        estimate.value,
        row,
        marker,
        markersize=8 if marker == "o" else 7,
        color=colour,
        markeredgecolor=theme.surface,
        markeredgewidth=2,
        zorder=3,
    )
    if label:
        axes.text(
            estimate.high + 0.015, row, f"{estimate.value:.2f}", va="center", color=theme.secondary
        )


def _title(figure: Any, title: str, subtitle: str, theme: Theme) -> None:
    height = figure.get_figheight()
    figure.text(0.012, 1 - 0.12 / height, title, fontsize=11, fontweight="bold", va="top")
    figure.text(0.012, 1 - 0.42 / height, subtitle, va="top", color=theme.secondary)


def _note(figure: Any, note: str, theme: Theme) -> None:
    figure.text(0.012, 0.1 / figure.get_figheight(), note, fontsize=8, color=theme.muted)


def _save(figure: Any, name: str, theme: Theme, directory: Path = FIGURES) -> Path:
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{name}-{theme.name}.svg"
    figure.savefig(path, format="svg", metadata={"Date": None})
    plt.close(figure)
    lines = path.read_text(encoding="utf-8").splitlines()  # matplotlib leaves trailing spaces
    path.write_text("\n".join(line.rstrip() for line in lines) + "\n", encoding="utf-8")
    return path


def precision_figure(data: CleanData, order: Sequence[int], theme: Theme) -> Path:
    """Average precision of every system with its interval, against chance."""
    estimates = {
        s: precision_interval(pooled(data, s).labels, pooled(data, s).scores, data.resamples)
        for s in MAP_SYSTEMS
    }
    labels, scores, _ = probe_arrays(data)
    estimates[PROBE] = precision_interval(labels, scores, data.resamples)
    chance = float(labels.mean())
    rows = [*order, PROBE]
    with plt.rc_context(_style(theme)):
        figure, axes = _axes(theme, len(rows), top=0.95, bottom=0.55)
        axes.axvline(chance, color=theme.muted, linewidth=1)
        axes.text(chance + 0.01, -0.55, f"chance {chance:.2f}", color=theme.muted, va="bottom")
        for row, system in enumerate(rows):
            colour = theme.muted if system == PROBE else theme.series[0]
            _interval(axes, row, estimates[system], colour, theme)
        axes.set_yticks(range(len(rows)), [LABELS[s] for s in rows])
        axes.set_xlabel("Average precision")
        _title(
            figure,
            "Document-level average precision on the clean folds",
            f"{labels.size} receipts, {int(labels.sum())} forged · dots with 95 % intervals · "
            "systems 2 to 4: seed 0",
            theme,
        )
        return _save(figure, "clean-precision", theme)


def localisation_figure(data: CleanData, order: Sequence[int], theme: Theme) -> Path:
    """How often the top region hits an edit, against a random box of the same size."""
    guards = {s: guard(localised(data, pooled(data, s).receipts), data.resamples) for s in order}
    forged = next(g.receipts for g in guards.values() if g is not None)
    with plt.rc_context(_style(theme)):
        figure, axes = _axes(theme, len(order), top=1.15, bottom=0.8)
        for row, system in enumerate(order):
            found = guards[system]
            if found is None:
                continue
            _interval(axes, row, found.hit_rate, theme.series[0], theme)
            axes.plot(
                found.baseline,
                row,
                "D",
                markersize=7,
                color=theme.series[1],
                markeredgecolor=theme.surface,
                markeredgewidth=2,
                zorder=3,
            )
        axes.set_yticks(range(len(order)), [LABELS[s] for s in order])
        axes.set_xlabel("Share of forged receipts")
        handles = [
            axes.plot([], [], "o", markersize=8, color=theme.series[0])[0],
            axes.plot([], [], "D", markersize=7, color=theme.series[1])[0],
        ]
        axes.legend(
            handles,
            ["top region hits the edit (95 % interval)", "random box of the same size"],
            loc="lower left",
            bbox_to_anchor=(0, 1.02),
            ncol=2,
            frameon=False,
            borderaxespad=0,
            handletextpad=0.4,
            columnspacing=1.6,
            labelcolor=theme.secondary,
        )
        _title(
            figure,
            "Does the most suspicious region find the edit?",
            f"{forged} forged receipts with region annotations · systems 2 to 4: seed 0",
            theme,
        )
        _note(
            figure,
            "System 1 makes the whole page its top region where it has no evidence, which hits "
            "with a baseline of 1 (ADR-0015).",
            theme,
        )
        return _save(figure, "clean-localisation", theme)


FUSION_ROWS = {
    "fusion": "Fusion",
    "score-level baseline": "Fusion without the links",
    "copy-move": "Copy-move alone",
    "pixel evidence": "Pixel evidence alone",
    "accounting rules": "Accounting rules alone",
}
"""The rows of the validation figure: the fusion model, its baseline and RQ2's comparators."""

TRACK_MARKS = (("transcript", "T1 · transcripts", "o"), ("ocr", "T2 · OCR", "D"))


def validation_figure(results: Any, theme: Theme) -> Path:
    """Average precision of fusion, its baseline and each single source, on both text sources."""
    receipts, forged = results["transcript"]["receipts"], results["transcript"]["forged"]
    chance = forged / receipts
    with plt.rc_context(_style(theme)):
        figure, axes = _axes(theme, len(FUSION_ROWS), top=1.15, bottom=0.8)
        figure.subplots_adjust(right=0.8)  # room for the column of values
        axes.axvline(chance, color=theme.muted, linewidth=1)
        values_at = axes.get_yaxis_transform()  # x across the axes' width, y in rows
        for row, name in enumerate(FUSION_ROWS):
            values = []
            for k, (track, label, marker) in enumerate(TRACK_MARKS):
                e = results[track]["table"][name]["average_precision"]
                estimate = Estimate(e["value"], e["low"], e["high"])
                offset = -0.14 if k == 0 else 0.14
                _interval(
                    axes, row + offset, estimate, theme.series[k], theme, label=False, marker=marker
                )
                values.append(f"{label.split(' ')[0]} {estimate.value:.2f}")
            axes.text(
                1.03,
                row,
                " · ".join(values),
                transform=values_at,
                va="center",
                color=theme.secondary,
            )
        axes.set_yticks(range(len(FUSION_ROWS)), list(FUSION_ROWS.values()))
        axes.set_xlabel("Average precision")
        handles = [
            axes.plot([], [], marker, markersize=8, color=theme.series[k])[0]
            for k, (_, _, marker) in enumerate(TRACK_MARKS)
        ]
        axes.legend(
            handles,
            [label for _, label, _ in TRACK_MARKS],
            loc="lower left",
            bbox_to_anchor=(0, 1.02),
            ncol=2,
            frameon=False,
            borderaxespad=0,
            handletextpad=0.4,
            columnspacing=1.6,
            labelcolor=theme.secondary,
        )
        _title(
            figure,
            "Average precision on the validation split, checked once",
            f"{receipts} receipts, {forged} forged · dots with 95 % intervals · system 2 seed 0",
            theme,
        )
        _note(
            figure,
            f"Grey line: chance, the share of forged receipts ({chance:.2f}). The claims are made "
            "on the test split in Phase 7.",
            theme,
        )
        return _save(figure, "validation-precision", theme, FUSION_FIGURES)


def main() -> None:
    """Write every figure in both themes."""
    data = load(Sources.of(ROOT))
    precision = {
        s: precision_interval(pooled(data, s).labels, pooled(data, s).scores, data.resamples).value
        for s in MAP_SYSTEMS
    }
    order = sorted(MAP_SYSTEMS, key=lambda s: -precision[s])  # the highest first
    validation = JsonRunStore(ROOT / "reports" / "runs").records(VALIDATION_RUN)[-1].results
    for theme in (LIGHT, DARK):
        for path in (
            precision_figure(data, order, theme),
            localisation_figure(data, order, theme),
            validation_figure(validation, theme),
        ):
            print(f"written: {path.relative_to(ROOT)}")  # noqa: T201


if __name__ == "__main__":
    main()
