from decimal import Decimal

from docfraud.evaluation.bootstrap import Estimate
from docfraud.evaluation.extraction import score_document
from docfraud.llm.adjudication import GOLD as GOLD_READING
from docfraud.llm.adjudication import Kind
from docfraud.llm.decision import ModelStep, decide_reader
from docfraud.llm.false_alarms import judge
from docfraud.llm.gold_b import LLM, PARSER, AdjudicationCounts, Latency, ReaderOutput
from docfraud.llm.report import (
    ReaderRun,
    render_false_alarms,
    render_gold_b,
    render_model_comparison,
    render_round,
)
from docfraud.reading.fields import AmountRole, Extraction, FieldSources, ReceiptFields
from docfraud.reading.pipeline import TextSource

D = Decimal

GOLD = ReceiptFields.model_validate({"amounts": {"total_due": ["5.00"]}})
WRONG = ReceiptFields.model_validate({"amounts": {"total_due": ["5.01"]}})


def run(name: str, right: bool, cost: Decimal | None = Decimal("0.0123")) -> ReaderRun:
    fields = GOLD if right else WRONG
    scores = {
        "T1 transcript": [score_document("A", GOLD, fields), score_document("B", GOLD, GOLD)],
        "T2 OCR": [score_document("A", GOLD, WRONG), score_document("B", GOLD, GOLD)],
    }
    rejections = None if cost is None else {"T1 transcript": 0, "T2 OCR": 3}
    return ReaderRun(name, scores, rejections, cost)


def test_the_model_comparison_lists_readers_steps_and_the_choice() -> None:
    step = ModelStep("small", "large", Estimate(0.1, -0.05, 0.25))
    text = render_model_comparison(
        "# Title\n\n",
        [run("small", False), run("large", True), run("parser v2", True, None)],
        [step],
        "small",
    )
    assert "| small | 1 of 2 | 1 of 2 | 0 | 3 | 0.0123 |" in text
    assert "| parser v2 | 2 of 2 | 1 of 2 | — | — | — |" in text
    assert "| small | large | +0.100 [-0.050, +0.250] | no |" in text
    assert text.endswith("**Chosen model:** `small`.\n")


def test_a_round_report_leads_with_the_primary_metric() -> None:
    alarms = {
        "gold set A": {"T1 transcript": 0, "T2 OCR": 0},
        "LLM": {"T1 transcript": 1, "T2 OCR": 2},
    }
    text = render_round("# Round 1\n\n", run("claude-haiku-5-5", True), frozenset({"A"}), alarms, 1)
    assert text.startswith("# Round 1\n\n### Primary metric, rejections and cost")
    assert "### Accounting rules on authentic receipts" in text
    assert "| LLM | 1 | 2 |" in text
    assert "Authentic receipts, of 1, on which" in text
    assert "| T2 OCR | 1 of 2 | 0.500 [" in text
    assert "Cost of the round: 0.0123 USD." in text
    assert "### Headline metrics" in text


def test_the_gold_b_report_leads_with_the_decision_per_text_source() -> None:
    scores = {
        track: {
            "parser v2": [score_document("A", GOLD, WRONG), score_document("B", GOLD, GOLD)],
            "LLM": [score_document("A", GOLD, GOLD), score_document("B", GOLD, GOLD)],
        }
        for track in ("T1 transcript", "T2 OCR")
    }
    decisions = [decide_reader(track, [False, True], [True, True]) for track in scores]
    counts = [
        AdjudicationCounts(Kind.AMOUNTS, 2, 1, 1, {GOLD_READING: 1, "LLM transcript": 2}),
        AdjudicationCounts(Kind.ITEMS, 0, 0, 0, {}),
    ]
    text = render_gold_b(
        "# Gold set B\n\n",
        decisions=decisions,
        scores=scores,
        counts=counts,
        readings=[GOLD_READING, "LLM transcript"],
        false_alarms={"parser v2": {"T1 transcript": 1, "T2 OCR": 0}},
        rejections={"T1 transcript": 0, "T2 OCR": 2},
        latencies={"T1 transcript": Latency(2, 1.5, 2.25), "T2 OCR": None},
        forged=frozenset({"B"}),
        authentic=1,
    )
    assert text.startswith("# Gold set B\n\n### Decision")
    assert "| T1 transcript | 1 of 2 | 2 of 2 | +0.500 [" in text
    assert "| 1 | 0 | 1.0000 | no |" in text  # one receipt better is far from enough
    assert "| Amounts of one role | 2 | 1 | 1 | 1 | 2 |" in text
    assert "| Item list | 0 | 0 | 0 | 0 | 0 |" in text
    assert "| parser v2 | 1 | 0 |" in text
    assert "| T1 transcript | 0 | 2 | 1.50 | 2.25 |" in text
    assert "| T2 OCR | 2 | — | — | — |" in text
    assert "## T2 OCR\n\n### Headline metrics" in text
    assert "### Difference LLM - parser v2" in text


def test_the_false_alarm_report_leads_with_the_decision_per_text_source() -> None:
    right = ReceiptFields.model_validate(
        {"amounts": {"total_due": ["30.00"], "tendered": ["50.00"], "change": ["20.00"]}}
    )
    wrong = right.model_copy(update={"amounts": {**right.amounts, AmountRole.TOTAL_DUE: (D(3),)}})
    forged = {"A1": False, "A2": False, "F1": True}

    def reading(doc_id: str, reader: str, read: ReceiptFields) -> ReaderOutput:
        return ReaderOutput(
            doc_id=doc_id,
            reader=reader,
            source=TextSource.TRANSCRIPT,
            extraction=Extraction(fields=read, sources=FieldSources()),
        )

    readings = [
        reading("A1", PARSER, wrong),
        reading("A1", LLM, right),
        reading("A2", PARSER, right),
        reading("A2", LLM, right),
        reading("F1", PARSER, wrong),
        reading("F1", LLM, wrong),
    ]
    result = judge("T1 transcript", TextSource.TRANSCRIPT, forged, readings)
    text = render_false_alarms(
        "# False alarms\n\n", {"T1 transcript": result}, authentic=2, forged=1
    )
    assert text.startswith("# False alarms\n\n### Decision")
    assert "| T1 transcript | 1 of 2 | 0 of 2 | -0.500 [" in text
    assert "| 1 | 0 | 1.0000 | 1 of 1 | 1 of 1 | +0.000 [+0.000, +0.000] | yes | no |" in text
    assert "| T1 transcript | +0.500 | +1.000 | +0.500 [" in text
    assert "### Rules that fail, T1 transcript" in text
    assert "| change_given | 1 | 0 | 1 | 1 |" in text
    assert "| T1 transcript | 0 |" in text  # no rejected references
