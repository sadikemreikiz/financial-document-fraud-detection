"""Minimal Markdown table rendering for generated reports.

Reports are generated from data so that no number in the documentation is copied by hand
(evaluation protocol, section 8).
"""

from collections.abc import Sequence
from dataclasses import dataclass

type Cell = str | int | float | None


@dataclass(frozen=True, slots=True)
class Table:
    """A titled table with an optional explanatory note."""

    title: str
    columns: Sequence[str]
    rows: Sequence[Sequence[Cell]]
    note: str = ""

    def __post_init__(self) -> None:
        for row in self.rows:
            if len(row) != len(self.columns):
                msg = f"{self.title}: row {row!r} does not match columns {self.columns!r}"
                raise ValueError(msg)


def format_cell(value: Cell) -> str:
    """Format a cell: floats with three decimals, ``None`` as an em dash."""
    if value is None:
        return "\N{EM DASH}"
    if isinstance(value, bool):
        return "yes" if value else "no"
    if isinstance(value, float):
        return f"{value:.3f}"
    return str(value).replace("|", "\\|")


def render_table(table: Table, heading_level: int = 3) -> str:
    """Render a table with its title as a heading and its note below it."""
    lines = [f"{'#' * heading_level} {table.title}", ""]
    lines.append("| " + " | ".join(table.columns) + " |")
    lines.append("|" + "|".join("---" for _ in table.columns) + "|")
    lines.extend("| " + " | ".join(format_cell(c) for c in row) + " |" for row in table.rows)
    if table.note:
        lines.extend(["", table.note])
    return "\n".join(lines) + "\n"
