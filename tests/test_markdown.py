import pytest

from docfraud.markdown import Cell, Table, format_cell, render_table


@pytest.mark.parametrize(
    ("value", "expected"),
    [
        (None, "\N{EM DASH}"),
        (True, "yes"),
        (False, "no"),
        (0.12345, "0.123"),
        (7, "7"),
        ("a|b", "a\\|b"),
    ],
)
def test_format_cell(value: Cell, expected: str) -> None:
    assert format_cell(value) == expected


def test_render_table_with_note() -> None:
    table = Table("Counts", ["Split", "N"], [["train", 3], ["val", 0.5]], note="A note.")
    assert render_table(table, heading_level=2) == (
        "## Counts\n\n| Split | N |\n|---|---|\n| train | 3 |\n| val | 0.500 |\n\nA note.\n"
    )


def test_row_width_must_match_columns() -> None:
    with pytest.raises(ValueError, match="does not match"):
        Table("Bad", ["A", "B"], [["only one"]])
