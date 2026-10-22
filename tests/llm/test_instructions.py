import re
from pathlib import Path

from docfraud.llm.answer import Answer
from docfraud.llm.labels import CLOSE, OPEN, label_rows
from docfraud.llm.reader import read_answer
from docfraud.reading.layout import Row, Segment

INSTRUCTIONS = Path(__file__).parents[2] / "src" / "docfraud" / "llm" / "instructions"


def test_the_first_instruction_shows_rows_and_an_answer_as_the_reader_has_them() -> None:
    text = (INSTRUCTIONS / "extraction-v1.md").read_text(encoding="utf-8")
    shown, answer = re.findall(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
    printed = [
        re.sub(rf"{OPEN}[nd]\d+{CLOSE}", "", line.split(": ", 1)[1]) for line in shown.splitlines()
    ]
    rows = [Row((Segment(line),)) for line in printed]
    labelled = label_rows(rows)
    assert labelled.text == shown.rstrip("\n")  # the example is what the labeller writes
    reading = read_answer(Answer.model_validate_json(answer), labelled, rows)
    assert reading.rejections == ()
    assert reading.fields.model_dump(mode="json")["amounts"] == {
        "subtotal": ["18.96"],
        "rounding": ["-0.01"],
        "total_due": ["18.95"],
        "tendered": ["20.00"],
        "change": ["1.05"],
    }
