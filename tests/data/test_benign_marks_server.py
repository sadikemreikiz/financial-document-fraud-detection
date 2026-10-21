import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from docfraud.data.benign_marks import (
    MarkKind,
    ReceiptMarks,
    Verdict,
    load_check_log,
    marks_plan,
)
from docfraud.data.benign_marks_server import MarkCheckApp
from docfraud.data.benign_marks_text import KIND_TEXT, PAGE_TEXT, VERDICT_TEXT
from docfraud.data.local_page import Response
from docfraud.data.normalize import load_grayscale
from docfraud.data.observation_text import PageLanguage
from tests.data.test_benign_marks import (
    CLEAN,
    FORGED,
    HAND,
    PLAIN,
    mark,
    mark_log,
    marks_documents,
)

JSON = "application/json"
LABEL_WORDS = ("forged", "authentic", "label", "handwritten_annotation", "digital_annotation")


def round_one(root: Path) -> tuple[Any, ...]:
    documents = marks_documents(root)
    receipts = {doc_id: ReceiptMarks() for doc_id in marks_plan(documents)}
    receipts[HAND] = ReceiptMarks(marks=(mark(), mark(kind="stamp", touches_print="yes")))
    receipts[FORGED] = ReceiptMarks(note="no mark found")
    return documents, mark_log(receipts)


@pytest.fixture
def app(tmp_path: Path) -> MarkCheckApp:
    documents, marks = round_one(tmp_path / "d")
    return MarkCheckApp(documents, marks, tmp_path / "work" / "emre-1.json", "emre", commit="abc")


def body(response: Response) -> Any:
    assert response.content_type == JSON
    return json.loads(response.body)


def post(app: MarkCheckApp, target: str, payload: Any) -> Response:
    return app.handle("POST", target, json.dumps(payload).encode(), JSON)


def answer(verdict: str = "correct", **changes: object) -> dict[str, object]:
    return {"verdict": verdict, "points": [], "note": "", "display": None} | changes


def test_the_page_shows_the_round_without_labels(app: MarkCheckApp) -> None:
    page = app.handle("GET", "/")
    assert page.status == 200
    assert b"Benign mark check" in page.body
    state = body(app.handle("GET", "/api/state"))
    assert (state["checker"], state["round"], state["answered"]) == ("emre", 1, [])
    assert len(state["receipts"]) == 3
    assert [v for v, _ in state["verdicts"]] == list(Verdict)
    responses = [app.handle("GET", "/api/state")]
    responses += [app.handle("GET", f"/api/receipt/{d}") for d in state["receipts"]]
    for response in responses:
        text = response.body.decode().lower()
        assert not [word for word in LABEL_WORDS if word in text]


def test_a_receipt_comes_with_its_numbered_marks(app: MarkCheckApp) -> None:
    data = body(app.handle("GET", f"/api/receipt/{HAND}"))
    assert (data["width"], data["height"], data["note"], data["answer"]) == (40, 60, "", None)
    assert [m["kind"] for m in data["marks"]] == ["handwriting", "stamp"]
    assert data["marks"][0]["box"] == {"x": 2, "y": 3, "width": 10, "height": 5}
    assert body(app.handle("GET", f"/api/receipt/{FORGED}"))["note"] == "no mark found"


def test_images_are_whole_normalised_pages_of_the_round_only(
    app: MarkCheckApp, tmp_path: Path
) -> None:
    response = app.handle("GET", f"/image/{HAND}.png")
    assert (response.status, response.content_type) == (200, "image/png")
    pixels = np.asarray(Image.open(io.BytesIO(response.body)))
    assert (pixels == load_grayscale(tmp_path / "d" / "train" / f"{HAND}.png")).all()
    assert app.handle("GET", f"/image/{CLEAN}.png").status == 403
    assert app.handle("GET", f"/image/{PLAIN}.png").status == 404  # design fold, not flagged
    assert app.handle("GET", f"/api/receipt/{PLAIN}").status == 404


def test_checks_are_saved_at_once_and_resumed(app: MarkCheckApp, tmp_path: Path) -> None:
    assert body(post(app, f"/api/receipt/{HAND}", answer())) == {"saved": HAND}
    problem = answer("missing", points=[{"x": 39, "y": 59}], note="a stamp at the bottom")
    assert post(app, f"/api/receipt/{FORGED}", problem).status == 200
    log_path = tmp_path / "work" / "emre-1.json"
    saved = load_check_log(log_path, "emre", 1).receipts
    assert (saved[HAND].verdict, saved[HAND].commit, saved[HAND].language) == (
        "correct",
        "abc",
        "en",
    )
    documents, marks = round_one(tmp_path / "d")
    resumed = MarkCheckApp(documents, marks, log_path, "emre", commit="def")
    assert sorted(body(resumed.handle("GET", "/api/state"))["answered"]) == sorted([HAND, FORGED])
    again = body(resumed.handle("GET", f"/api/receipt/{FORGED}"))["answer"]
    assert (again["verdict"], again["points"]) == ("missing", [{"x": 39, "y": 59}])


def test_invalid_checks_are_refused_and_change_nothing(app: MarkCheckApp) -> None:
    target = f"/api/receipt/{HAND}"
    assert post(app, target, answer("purple")).status == 400
    assert post(app, target, answer(points=[{"x": 1, "y": 1}])).status == 400
    outside = post(app, target, answer("wrong_box", points=[{"x": 40, "y": 1}]))
    assert (outside.status, body(outside)["error"]) == (400, "a point lies outside the page")
    assert post(app, f"/api/receipt/{PLAIN}", answer()).status == 404
    assert post(app, target, []).status == 400
    assert app.handle("POST", target, b"{", JSON).status == 400
    assert app.handle("POST", target, b"{}", "text/plain").status == 415
    assert app.handle("PUT", "/api/state").status == 405
    assert app.handle("GET", "/api/unknown").status == 404
    assert app.log.receipts == {}


def test_marks_outside_the_plan_are_refused(tmp_path: Path) -> None:
    documents, marks = round_one(tmp_path / "d")
    strays = marks.model_copy(update={"receipts": marks.receipts | {PLAIN: ReceiptMarks()}})
    with pytest.raises(ValueError, match="outside the plan"):
        MarkCheckApp(documents, strays, tmp_path / "emre-1.json", "emre", commit=None)


def test_the_turkish_page_records_english_values(tmp_path: Path) -> None:
    documents, marks = round_one(tmp_path / "d")
    turkish = MarkCheckApp(
        documents,
        marks,
        tmp_path / "emre-1.json",
        "emre",
        commit="abc",
        language=PageLanguage.TURKISH,
    )
    state = body(turkish.handle("GET", "/api/state"))
    assert state["ui"]["save"] == "Kaydet ve sonraki"
    assert state["verdicts"][0] == ["correct", "Do\N{LATIN SMALL LETTER G WITH BREVE}ru"]
    marks_shown = body(turkish.handle("GET", f"/api/receipt/{HAND}"))["marks"]
    assert marks_shown[1]["kind"] == "damga"
    assert post(turkish, f"/api/receipt/{HAND}", answer("wrong_box")).status == 200
    assert (turkish.log.receipts[HAND].verdict, turkish.log.receipts[HAND].language) == (
        "wrong_box",
        "tr",
    )


def test_every_word_of_the_page_has_a_translation() -> None:
    english, turkish = (PAGE_TEXT[language] for language in PageLanguage)
    assert set(english) == set(turkish)
    for language in PageLanguage:
        assert set(VERDICT_TEXT[language]) == set(Verdict)
        assert set(KIND_TEXT[language]) == set(MarkKind)
