import pytest

from docfraud.fusion.rehearsal import differences
from docfraud.fusion.training_inputs import InputsTable
from tests.fusion.test_validation import tables

NONE = {"rows_missing": 0, "rows_added": 0, "input_names": 0, "inputs": {}, "findings": 0}


def test_a_table_made_again_the_same_way_shows_no_difference() -> None:
    validation = tables()[2]
    assert differences(validation, tables()[2]) == NONE | {"values": 0}


def test_every_kind_of_difference_is_counted() -> None:
    validation = tables()[2]
    first, second, *rest = validation.receipts
    moved = first.model_copy(
        update={"inputs": first.inputs | {"rules": first.inputs["rules"] + 0.5}}
    )
    renamed = second.model_copy(update={"inputs": {"other": 1.0}, "findings": ()})
    rebuilt = InputsTable(receipts=(moved, renamed, *rest[:-1]))
    found = differences(validation, rebuilt)
    assert (found["rows_missing"], found["rows_added"]) == (1, 0)
    assert found["inputs"] == {"rules": pytest.approx(0.5)}
    assert found["input_names"] == 1
    assert found["findings"] == (1 if second.findings else 0)
    assert rebuilt != validation
