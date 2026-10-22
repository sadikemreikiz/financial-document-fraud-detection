from decimal import Decimal

from docfraud.evaluation.bootstrap import Estimate
from docfraud.evaluation.extraction import score_document
from docfraud.llm.decision import ModelStep
from docfraud.llm.report import ReaderRun, render_model_comparison, render_round
from docfraud.reading.fields import ReceiptFields

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
    text = render_round("# Round 1\n\n", run("claude-haiku-5-5", True), frozenset({"A"}))
    assert text.startswith("# Round 1\n\n### Primary metric, rejections and cost")
    assert "| T2 OCR | 1 of 2 | 0.500 [" in text
    assert "Cost of the round: 0.0123 USD." in text
    assert "### Headline metrics" in text
