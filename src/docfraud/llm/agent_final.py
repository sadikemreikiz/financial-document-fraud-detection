"""The frozen agent on the test split, once (ADR-0031, sections 5, 6 and 10; ADR-0034, 7).

The agent runs on the test receipts that the review policy of T2 sends to review: the fusion
model of Phase 5's check, with its recorded Platt scaling, and the policy's recorded threshold,
applied unchanged; the threshold is not set again on the test receipts, so the queue need not
hold 10 % of them. Each case is measured as on the validation split
(:mod:`docfraud.llm.agent_validation`).

Before it, a rehearsal applies the same code to the validation receipts from the stored steps,
without sending a request, and must give back the validation run's record: the queue, every case
file and every measure.
"""

from collections.abc import Mapping, Sequence
from typing import Any

import numpy as np
import numpy.typing as npt

from docfraud.decision import BUDGETS, ReviewPolicy
from docfraud.fusion.development_report import TRACK_NAMES
from docfraud.fusion.final import frozen_row
from docfraud.fusion.training_inputs import ReceiptInputs
from docfraud.fusion.validation import FUSION_ROW
from docfraud.llm.agent_cases import TRACK
from docfraud.llm.agent_rounds import AGENT_CODE
from docfraud.llm.agent_validation import Queue
from docfraud.reporting.assessment import assess

RUN_NAME = "llm-agent-test"
REHEARSAL_RUN = "llm-agent-rehearsal"

TEST_CODE = (
    *AGENT_CODE,
    "src/docfraud/fusion",
    "src/docfraud/llm/agent_final.py",
    "src/docfraud/llm/agent_validation.py",
    "src/docfraud/llm/run.py",
    "src/docfraud/llm/store.py",
)
"""What makes the agent's cases, requests and measures; unchanged from the rehearsal to the test."""

type Labels = npt.NDArray[np.bool_]


def frozen_queue(
    training: Sequence[ReceiptInputs],
    training_labels: Labels,
    rows: Sequence[ReceiptInputs],
    check: Mapping[str, Any],
) -> Queue:
    """Track T2's review queue among ``rows``, with what Phase 5's check recorded applied.

    Raises:
        ValueError: If the fusion model fitted again is not the recorded one.
    """
    fused = frozen_row(
        training, training_labels, rows, FUSION_ROW, check[TRACK]["fitted"]["fusion"]
    )
    stated = check["policy"][TRACK][f"{BUDGETS[0] * 100:.0f} %"]
    policy = ReviewPolicy(stated["budget"], stated["threshold"])
    reviewed = policy.reviewed(fused.probabilities)
    assessments = {
        row.receipt: assess(
            row.receipt,
            TRACK_NAMES[TRACK],
            {name: row.inputs[name] for name in fused.model.names},
            row.findings,
            model=fused.model,
            platt=fused.platt,
            policy=policy,
        )
        for row, sent in zip(rows, reviewed, strict=True)
        if sent
    }
    return Queue(policy, assessments)
