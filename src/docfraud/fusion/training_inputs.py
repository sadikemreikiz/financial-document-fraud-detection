"""The fusion inputs of the training split, out of fold (ADR-0028; ADR-0029, section 14, step 3).

Every input with fitted parameters is the output of the cross-validation round that held the
receipt out: system 2's stored score and top region, per seed, and the accounting rules' score
from the Phase 3 scorer fitted on the other folds. The other findings have no fitted parameters
and are computed directly. The labels enter only that scorer's fit on the other folds.

The table keeps every receipt's inputs and findings per text source and system 2 seed; the run
record keeps only aggregates.
"""

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np
from pydantic import BaseModel, ConfigDict

from docfraud.detectors.finding import COPY_MOVE, Finding
from docfraud.detectors.rules import Outcome
from docfraud.detectors.scoring import features, out_of_fold
from docfraud.forensics.learned.inference import QUARTER_SQUARE
from docfraud.forensics.learned.results import Group, read_results
from docfraud.forensics.learned.rounds import run_directory
from docfraud.forensics.learned.sampler import System
from docfraud.fusion.evidence import SystemTwo, receipt_evidence
from docfraud.fusion.inputs import PRACTICAL
from docfraud.reading.fields import FieldSources

RUN_NAME = "fusion-training-inputs"
ROUNDS = range(5)


class ReceiptInputs(BaseModel):
    """One receipt's fusion inputs and findings, for one text source and system 2 seed."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    receipt: str
    track: str
    seed: int
    inputs: dict[str, float]  # those of track T3; tracks T1 and T2 take all but the last
    findings: tuple[Finding, ...]


class InputsTable(BaseModel):
    """The inputs of every training receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    receipts: tuple[ReceiptInputs, ...]

    def rows(self, track: str, seed: int) -> list[ReceiptInputs]:
        """The receipts of one track and seed, in the table's order."""
        return [r for r in self.receipts if r.track == track and r.seed == seed]


def save_table(table: InputsTable, path: Path) -> None:
    """Write a table as JSON."""
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(table.model_dump_json(), encoding="utf-8")


def load_table(path: Path) -> InputsTable:
    """Read a table written by :func:`save_table`."""
    return InputsTable.model_validate_json(path.read_text(encoding="utf-8"))


def stored_system_2(rounds: Path, seed: int) -> dict[str, SystemTwo]:
    """System 2's output on every training receipt from the round that held it out.

    Raises:
        ValueError: If a receipt is held out by two rounds.
    """
    found: dict[str, SystemTwo] = {}
    for held_out in ROUNDS:
        results = read_results(run_directory(rounds, held_out, System.SYNTHETIC, seed))
        for receipt in results.measured:
            if receipt.group is not Group.HELD_OUT:
                continue
            if receipt.key in found:
                msg = f"{receipt.key} is held out by two rounds"
                raise ValueError(msg)
            found[receipt.key] = SystemTwo(receipt.scores[QUARTER_SQUARE], receipt.top_region)
    return found


def training_inputs(  # noqa: PLR0913 - the receipts, their folds, readings and system outputs
    receipts: Sequence[str],
    forged: Sequence[bool],
    fold_of: Sequence[int],
    readings: Mapping[str, Sequence[tuple[Mapping[str, Outcome], FieldSources]]],
    *,
    page: Sequence[tuple[Finding, ...]],
    system_2: Mapping[int, Mapping[str, SystemTwo]],
) -> InputsTable:
    """Every receipt's inputs per text source and seed, with the rules' score out of fold.

    ``readings`` holds, per text source, every receipt's rule outcomes and field sources;
    ``page`` every receipt's page findings; ``system_2`` per seed every receipt's output.

    Raises:
        ValueError: If a receipt lacks a system 2 output or the sequences differ in length.
    """
    n = len(receipts)
    if not len(forged) == len(fold_of) == len(page) == n or any(
        len(r) != n for r in readings.values()
    ):
        msg = "one label, fold, page and reading per receipt"
        raise ValueError(msg)
    y, folds = np.asarray(forged, dtype=bool), np.asarray(fold_of)
    rows = []
    for track, reading in readings.items():
        scores = out_of_fold(features([outcomes for outcomes, _ in reading]), y, folds)
        for seed, outputs in system_2.items():
            missing = [r for r in receipts if r not in outputs]
            if missing:
                msg = f"system 2, seed {seed}: no output for {len(missing)} receipts"
                raise ValueError(msg)
            for i, receipt in enumerate(receipts):
                outcomes, sources = reading[i]
                inputs, findings = receipt_evidence(
                    float(scores[i]), outputs[receipt], page[i], outcomes, sources
                )
                rows.append(
                    ReceiptInputs(
                        receipt=receipt, track=track, seed=seed, inputs=inputs, findings=findings
                    )
                )
    return InputsTable(receipts=tuple(rows))


def summary(table: InputsTable) -> dict[str, Any]:
    """Aggregates of the table for the run record: per track and seed, mean inputs and shares."""
    out: dict[str, Any] = {}
    for track in sorted({r.track for r in table.receipts}):
        for seed in sorted({r.seed for r in table.receipts}):
            rows = table.rows(track, seed)
            matrix = np.array([[r.inputs[name] for name in PRACTICAL] for r in rows])
            out[f"{track}, seed {seed}"] = {
                "receipts": len(rows),
                "means": dict(zip(PRACTICAL, map(float, matrix.mean(axis=0)), strict=True)),
                "copy_move_pairs": sum(
                    any(f.detector == COPY_MOVE for f in r.findings) for r in rows
                ),
            }
    return out
