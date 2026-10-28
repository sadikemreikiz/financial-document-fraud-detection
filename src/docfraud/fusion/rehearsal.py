"""Phase 7's rehearsal of the inputs: today's code must make the validation inputs again.

The test inputs are made by the code that made the validation inputs (ADR-0029, section 2).
Phase 6b changed files of that code (:data:`~docfraud.fusion.validation_inputs.INPUT_CODE`)
without meaning to change any input, so a check that no file changed would fail for no reason
and prove nothing either way. Instead, the rehearsal makes the validation inputs again with
today's code and compares them with the recorded table; the test inputs are made only by code
that reproduces it exactly (ADR-0034, decision 13).
"""

from typing import Any

from docfraud.fusion.training_inputs import InputsTable

REHEARSAL_RUN = "fusion-inputs-rehearsal"
TEST_INPUTS_RUN = "fusion-test-inputs"
RESAVED_INPUTS_RUN = "fusion-resaved-inputs"


def differences(recorded: InputsTable, rebuilt: InputsTable) -> dict[str, Any]:
    """Where a rebuilt table differs from the recorded one.

    Rows are matched by receipt, track and seed. For the rows in both: the largest absolute
    difference of each input that differs, and how many rows differ in their input names, their
    findings or the values the links relate regions to.
    """
    old = {(r.receipt, r.track, r.seed): r for r in recorded.receipts}
    new = {(r.receipt, r.track, r.seed): r for r in rebuilt.receipts}
    shared = sorted(old.keys() & new.keys())
    largest: dict[str, float] = {}
    for key in shared:
        a, b = old[key].inputs, new[key].inputs
        for name in a.keys() & b.keys():
            gap = abs(a[name] - b[name])
            if gap > 0:
                largest[name] = max(largest.get(name, 0.0), gap)
    return {
        "rows_missing": len(old.keys() - new.keys()),
        "rows_added": len(new.keys() - old.keys()),
        "input_names": sum(old[k].inputs.keys() != new[k].inputs.keys() for k in shared),
        "inputs": dict(sorted(largest.items())),
        "findings": sum(old[k].findings != new[k].findings for k in shared),
        "values": sum(old[k].values != new[k].values for k in shared),
    }
