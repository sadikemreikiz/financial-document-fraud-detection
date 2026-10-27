import ast
import importlib.util
import inspect
import typing
from pathlib import Path

import numpy as np
import pytest

import docfraud
from docfraud.data.models import BBox
from docfraud.detectors.scoring import features
from docfraud.forensics.learned.inference import QUARTER_SQUARE, LearnedOutput
from docfraud.forensics.learned.results import Group, Item, measure_receipt
from docfraud.fusion.evidence import SystemTwo, receipt_evidence
from docfraud.fusion.validation_inputs import (
    INPUT_CODE,
    rules_scorers,
    system_two,
    validation_inputs,
)
from tests.fusion.test_training_inputs import FAIL, PAGE, PASS, SOURCES

ROOT = Path(docfraud.__file__).parents[2]

ENTRY_POINTS = (
    "docfraud.data.findit",
    "docfraud.data.normalize",
    "docfraud.detectors.metadata",
    "docfraud.detectors.scoring",
    "docfraud.fusion.evidence",
    "docfraud.llm.frozen",
    "docfraud.llm.run",
    "docfraud.reading.pipeline",
)
"""What the commands call to turn a receipt into inputs, apart from system 2's network."""

SYSTEM_2_CODE = (
    "src/docfraud/forensics/learned/inference.py",
    "src/docfraud/forensics/learned/network.py",
    "src/docfraud/forensics/trufor.py",
)
"""The network's code, which imports training code of its own and is listed by file."""


def test_system_2_is_measured_as_the_rounds_measured_their_held_out_folds() -> None:
    rng = np.random.default_rng(0)
    logits = rng.normal(-3, 2, size=(60, 80)).astype(np.float32)
    logits[20:30, 40:50] += 6
    output = LearnedOutput(logits, (BBox(x=0, y=0, width=80, height=60),), 0.1)
    words = [BBox(x=5, y=5 + 12 * k, width=40, height=10) for k in range(4)]
    page = np.zeros((60, 80), dtype=np.uint8)
    stored = measure_receipt(
        Item("R", Group.HELD_OUT, page, False, words=tuple(words)), output, 0.5
    )
    found = system_two(output, words)
    assert found == SystemTwo(stored.scores[QUARTER_SQUARE], stored.top_region)
    assert found.region.overlaps(BBox(x=40, y=20, width=10, height=10))


def test_validation_inputs_cannot_take_a_label_an_annotation_or_a_document() -> None:
    assert list(inspect.signature(validation_inputs).parameters) == [
        "receipts",
        "readings",
        "page",
        "system_2",
        "scorers",
    ]
    hints = str(typing.get_type_hints(validation_inputs))
    for forbidden in ("Document", "Label", "Region", "bool"):
        assert forbidden not in hints


def scorers() -> dict[str, typing.Any]:
    forged = [i % 4 == 0 for i in range(20)]
    outcomes = [FAIL if f else PASS for f in forged]
    return rules_scorers({"transcript": outcomes, "ocr": outcomes}, forged)


def test_validation_rows_are_the_training_rows_with_the_rules_scored_by_the_whole_split() -> None:
    fitted = scorers()
    receipts = ["V0", "V1", "V2"]
    readings = {"transcript": [(FAIL, SOURCES), (PASS, SOURCES), (PASS, SOURCES)]}
    system_2 = {
        seed: {r: SystemTwo(0.2 + seed, BBox(x=105, y=62, width=8, height=8)) for r in receipts}
        for seed in (0, 1)
    }
    table = validation_inputs(
        receipts, readings, page=[PAGE] * 3, system_2=system_2, scorers=fitted
    )
    assert len(table.receipts) == 3 * 2
    rows = table.rows("transcript", 1)
    assert [r.receipt for r in rows] == receipts
    expected = fitted["transcript"].scores(features([FAIL, PASS, PASS]))
    assert [r.inputs["rules"] for r in rows] == pytest.approx(expected)
    assert expected[0] > expected[1]  # the failing rule raises the score
    inputs, findings = receipt_evidence(float(expected[0]), system_2[1]["V0"], PAGE, FAIL, SOURCES)
    assert (rows[0].inputs, rows[0].findings) == (inputs, findings)


def test_validation_inputs_refuse_missing_outputs_scorers_and_lengths() -> None:
    fitted = scorers()
    box = BBox(x=5, y=5, width=4, height=4)
    readings = {"ocr": [(PASS, SOURCES)] * 2}
    with pytest.raises(ValueError, match="no output for 1 receipts"):
        validation_inputs(
            ["V0", "V1"],
            readings,
            page=[PAGE] * 2,
            system_2={0: {"V0": SystemTwo(0.1, box)}},
            scorers=fitted,
        )
    with pytest.raises(ValueError, match="no rules scorer for ocr"):
        validation_inputs(["V0", "V1"], readings, page=[PAGE] * 2, system_2={}, scorers={})
    with pytest.raises(ValueError, match="one page and reading"):
        validation_inputs(["V0"], readings, page=[PAGE], system_2={}, scorers=fitted)
    with pytest.raises(ValueError, match="one rule outcome per training receipt"):
        rules_scorers({"ocr": [PASS]}, [True, False])


def _module_path(module: str) -> Path | None:
    spec = importlib.util.find_spec(module)
    if spec is None or spec.origin is None:
        return None
    return Path(spec.origin).relative_to(ROOT)


def _is_package(module: str) -> bool:
    spec = importlib.util.find_spec(module)
    return spec is not None and spec.submodule_search_locations is not None


def _imports(module: str) -> set[str]:
    """The project's modules that ``module`` imports, including modules imported by name."""
    path = _module_path(module)
    assert path is not None
    found = set()
    for node in ast.walk(ast.parse((ROOT / path).read_text(encoding="utf-8"))):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("docfraud"):
            found.add(node.module)
            if _is_package(node.module):
                named = (f"{node.module}.{a.name}" for a in node.names)
                found |= {m for m in named if _module_path(m) is not None}
    return found


def test_the_input_code_holds_everything_that_turns_a_receipt_into_inputs() -> None:
    seen: set[str] = set()
    todo = list(ENTRY_POINTS)
    while todo:
        module = todo.pop()
        if module not in seen:
            seen.add(module)
            todo += _imports(module)
    files = {_module_path(m) for m in seen}
    uncovered = sorted(
        p.as_posix()
        for p in files
        if p is not None
        and p.name != "__init__.py"
        and not any(p.as_posix().startswith(prefix) for prefix in INPUT_CODE)
    )
    assert len(files) > 30  # the walk reached the reader, the rules, the detectors and linking
    assert uncovered == []
    assert set(SYSTEM_2_CODE) <= set(INPUT_CODE)
    assert all((ROOT / prefix).exists() for prefix in INPUT_CODE)
