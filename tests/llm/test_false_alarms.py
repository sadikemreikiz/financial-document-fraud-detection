import pytest

from docfraud.detectors.rules import RULES
from docfraud.llm.false_alarms import (
    Alarms,
    failing,
    judge,
    rule_counts,
    separation_difference,
)
from docfraud.llm.gold_b import LLM, PARSER, ReaderOutput
from docfraud.llm.reader import Rejection
from docfraud.reading.fields import Extraction, FieldSources, ReceiptFields
from docfraud.reading.pipeline import TextSource

T1, T2 = TextSource.TRANSCRIPT, TextSource.OCR


def fields(**amounts: str) -> ReceiptFields:
    return ReceiptFields.model_validate({"amounts": {r: [v] for r, v in amounts.items()}})


RIGHT = fields(total_due="30.00", tendered="50.00", change="20.00")
WRONG = fields(total_due="3.00", tendered="50.00", change="20.00")  # paid - due != change


def output(
    doc_id: str, reader: str, source: TextSource, read: ReceiptFields, *, rejected: bool = False
) -> ReaderOutput:
    return ReaderOutput(
        doc_id=doc_id,
        reader=reader,
        source=source,
        extraction=Extraction(fields=read, sources=FieldSources()),
        rejections=(Rejection("amounts", "n9", "no such label"),) if rejected else (),
    )


def test_a_receipt_raises_an_alarm_when_a_rule_fails() -> None:
    assert failing(WRONG) == ("change_given",)
    assert failing(RIGHT) == ()
    assert failing(ReceiptFields()) == ()  # an empty reading raises none


def test_rule_counts_name_every_rule_in_order() -> None:
    counts = rule_counts([("change_given",), (), ("change_given", "tax_rates")])
    assert list(counts) == list(RULES)
    assert (counts["change_given"], counts["tax_rates"], counts["items_reconcile"]) == (2, 1, 0)


def test_the_separation_weighs_catches_against_false_alarms() -> None:
    forged = [True, True, False, False]
    parser = [True, True, True, False]  # catches 2 of 2, false alarms 1 of 2: 0.5
    fewer_both = [True, False, False, False]  # catches 1 of 2, false alarms 0 of 2: 0.5
    assert separation_difference(forged, parser, fewer_both).value == pytest.approx(0.0)
    fewer_alarms = [True, True, False, False]  # catches 2 of 2, false alarms 0 of 2: 1.0
    estimate = separation_difference(forged, parser, fewer_alarms)
    assert estimate.value == pytest.approx(0.5)
    assert estimate.low <= estimate.value <= estimate.high
    with pytest.raises(ValueError, match="authentic and forged"):
        separation_difference([True, True], [True, True], [True, True])


def test_judging_a_text_source_counts_alarms_per_class_and_reader() -> None:
    forged = {"A1": False, "A2": False, "F1": True}
    readings = [
        output("A1", PARSER, T1, WRONG),
        output("A1", LLM, T1, RIGHT),
        output("A2", PARSER, T1, RIGHT),
        output("A2", LLM, T1, RIGHT),
        output("F1", PARSER, T1, WRONG),
        output("F1", LLM, T1, WRONG, rejected=True),
        output("A1", LLM, T2, WRONG),  # another text source: not part of this judgement
    ]
    result = judge("T1 transcript", T1, forged, readings)
    assert result.alarms == {PARSER: Alarms(1, 1), LLM: Alarms(0, 1)}
    assert (result.decision.parser_only, result.decision.llm_only) == (1, 0)
    assert not result.decision.passes  # one receipt is far from enough
    assert result.rules[PARSER][0]["change_given"] == 1
    assert result.rules[LLM] == (rule_counts([]), rule_counts([("change_given",)]))
    assert result.separation.value == pytest.approx(0.5)  # same catches, half the false alarms
    assert result.rejected == 1
    with pytest.raises(ValueError, match="one reading per reader"):
        judge("T2 OCR", T2, forged, readings)
