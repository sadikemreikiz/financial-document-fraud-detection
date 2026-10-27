"""The fusion inputs of the validation split (ADR-0029, sections 2, 7 (d) and (e), 14, step 5).

Every learned component that scores a validation receipt was trained on the whole training
split: system 2 by the round that holds out no fold, with seeds 0 to 2, and the accounting
rules' scorer of Phase 3, fitted here on the frozen reader's fields of every training receipt.
The findings without fitted parameters (system 1, copy-move, the file's editing traits) and the
links come from the same functions as on the training split (:mod:`docfraud.fusion.evidence`),
and system 2's output is measured as the rounds measured their held-out folds
(:func:`system_two`), so that both sides of the fusion model's inputs are made the same way.

:data:`INPUT_CODE` is the code that turns a receipt into inputs on both splits; the validation
inputs are computed only if none of it changed since the training inputs were. System 2's
stored outputs on the training split are older than that: the rounds measured them at commit
``2a8998f``, and since then the code of :func:`system_two` has changed only by
:func:`~docfraud.evaluation.localisation.any_box_baseline`, a function it does not call.

No label of a validation receipt can reach an input: :func:`validation_inputs` takes receipt
identifiers, readings, page findings, system 2's outputs and the fitted scorers, nothing that
carries a label or an annotation. Training labels enter only the scorers' fit.
"""

from collections.abc import Mapping, Sequence

import numpy as np

from docfraud.data.models import BBox
from docfraud.detectors.finding import Finding
from docfraud.detectors.rules import Outcome
from docfraud.detectors.scoring import Scorer, features
from docfraud.evaluation.localisation import top_region
from docfraud.forensics.learned.inference import QUARTER_SQUARE
from docfraud.forensics.learned.results import MapOutput
from docfraud.forensics.learned.text_scale import text_height
from docfraud.fusion.evidence import SystemTwo, receipt_evidence
from docfraud.fusion.training_inputs import InputsTable, ReceiptInputs
from docfraud.linking.values import located_values
from docfraud.reading.fields import FieldSources

RUN_NAME = "fusion-validation-inputs"

INPUT_CODE = (
    "src/docfraud/data",
    "src/docfraud/detectors",
    "src/docfraud/evaluation/bootstrap.py",
    "src/docfraud/evaluation/detection.py",
    "src/docfraud/evaluation/localisation.py",
    "src/docfraud/forensics/lattice_map.py",
    "src/docfraud/forensics/quantisation.py",
    "src/docfraud/forensics/trufor.py",
    "src/docfraud/forensics/learned/inference.py",
    "src/docfraud/forensics/learned/network.py",
    "src/docfraud/forensics/learned/text_scale.py",
    "src/docfraud/fusion/evidence.py",
    "src/docfraud/fusion/inputs.py",
    "src/docfraud/fusion/training_inputs.py",
    "src/docfraud/linking",
    "src/docfraud/llm",
    "src/docfraud/reading",
)
"""The code that turns a receipt into fusion inputs, on the training and the validation split."""

type Reading = tuple[Mapping[str, Outcome], FieldSources]


def system_two(output: MapOutput, words: Sequence[BBox]) -> SystemTwo:
    """System 2's output on one receipt, measured as the rounds measured their held-out folds.

    The document score of the fusion model, the highest mean over a square of a quarter of the
    receipt's text height (ADR-0023), and the top region of ADR-0015, both on the map in double
    precision, as :func:`~docfraud.forensics.learned.results.measure_receipt` measures them.
    """
    values = output.values
    height, width = values.shape
    score = output.scores(text_height(words, height, width), values)[QUARTER_SQUARE]
    return SystemTwo(score, top_region(values))


def rules_scorers(
    outcomes: Mapping[str, Sequence[Mapping[str, Outcome]]], forged: Sequence[bool]
) -> dict[str, Scorer]:
    """The rules' scorer of each text source, fitted on every training receipt (section 6).

    ``outcomes`` holds, per text source, the rule outcomes of every training receipt.

    Raises:
        ValueError: If a text source does not have one outcome per label.
    """
    y = np.asarray(forged, dtype=bool)
    if any(len(o) != y.size for o in outcomes.values()):
        msg = "one rule outcome per training receipt and text source"
        raise ValueError(msg)
    return {track: Scorer.fit(features(o), y) for track, o in outcomes.items()}


def validation_inputs(
    receipts: Sequence[str],
    readings: Mapping[str, Sequence[Reading]],
    *,
    page: Sequence[tuple[Finding, ...]],
    system_2: Mapping[int, Mapping[str, SystemTwo]],
    scorers: Mapping[str, Scorer],
) -> InputsTable:
    """Every receipt's inputs per text source and seed, with the rules scored by ``scorers``.

    ``readings`` holds, per text source, every receipt's rule outcomes and field sources;
    ``page`` every receipt's page findings; ``system_2`` per seed every receipt's output. The
    rows are made as :func:`~docfraud.fusion.training_inputs.training_inputs` makes them, with
    the rules' score of the scorer fitted on the whole training split instead of out of fold.

    Raises:
        ValueError: If a receipt lacks a system 2 output, a text source lacks a scorer or the
            sequences differ in length.
    """
    n = len(receipts)
    if len(page) != n or any(len(r) != n for r in readings.values()):
        msg = "one page and reading per receipt"
        raise ValueError(msg)
    if missing := sorted(set(readings) - set(scorers)):
        msg = f"no rules scorer for {', '.join(missing)}"
        raise ValueError(msg)
    rows = []
    for track, reading in readings.items():
        scores = scorers[track].scores(features([outcomes for outcomes, _ in reading]))
        for seed, outputs in system_2.items():
            if absent := [r for r in receipts if r not in outputs]:
                msg = f"system 2, seed {seed}: no output for {len(absent)} receipts"
                raise ValueError(msg)
            for i, receipt in enumerate(receipts):
                outcomes, sources = reading[i]
                inputs, findings = receipt_evidence(
                    float(scores[i]), outputs[receipt], page[i], outcomes, sources
                )
                rows.append(
                    ReceiptInputs(
                        receipt=receipt,
                        track=track,
                        seed=seed,
                        inputs=inputs,
                        findings=findings,
                        values=located_values(sources),
                    )
                )
    return InputsTable(receipts=tuple(rows))
