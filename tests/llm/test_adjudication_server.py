import json
from decimal import Decimal
from pathlib import Path

import numpy as np
import pytest

from docfraud.data.observation import ASSISTANT
from docfraud.data.observation_text import PageLanguage
from docfraud.llm.adjudication import GOLD, load_log, questions_for
from docfraud.llm.adjudication_server import AdjudicationApp
from docfraud.reading.fields import AmountRole, ReceiptFields

GOLD_FIELDS = ReceiptFields.model_validate({"amounts": {"total_due": ["30.00"]}})
MISREAD = ReceiptFields.model_validate({"amounts": {"total_due": ["3.00"]}})
QUESTIONS = questions_for("R1", {GOLD: GOLD_FIELDS, "llm T1": MISREAD})
KEY = "R1:amounts:total_due"


def app(tmp_path: Path, language: PageLanguage = PageLanguage.TURKISH) -> AdjudicationApp:
    return AdjudicationApp(
        QUESTIONS,
        lambda _: np.full((4, 3), 255, dtype=np.uint8),
        tmp_path / "log.json",
        reviewer="owner",
        commit="abc",
        language=language,
    )


def get(page: AdjudicationApp, target: str) -> dict[str, object]:
    response = page.handle("GET", target)
    assert response.status == 200, response.body
    loaded: dict[str, object] = json.loads(response.body)
    return loaded


def post(page: AdjudicationApp, target: str, payload: object) -> tuple[int, dict[str, object]]:
    response = page.handle("POST", target, json.dumps(payload).encode(), "application/json")
    return response.status, json.loads(response.body)


def test_the_page_and_its_state(tmp_path: Path) -> None:
    page = app(tmp_path)
    assert page.handle("GET", "/").content_type.startswith("text/html")
    state = get(page, "/api/state")
    assert (state["questions"], state["done"], state["language"]) == ([KEY], [], "tr")
    question = get(page, f"/api/question/{KEY}")
    assert question["candidates"] is None  # nothing is shown before the first answer
    assert question["role"] == {
        "value": "total_due",
        "label": "Ödenecek toplam",
        "help": question["role"]["help"],  # type: ignore[index]
    }
    image = page.handle("GET", "/image/R1.png")
    assert (image.status, image.content_type) == (200, "image/png")
    assert page.handle("GET", "/image/R2.png").status == 404
    assert page.handle("GET", "/api/question/unknown").status == 404


def test_a_first_answer_equal_to_a_reading_is_final(tmp_path: Path) -> None:
    page = app(tmp_path)
    status, question = post(page, f"/api/question/{KEY}/first", {"amounts": ["30.00"]})
    assert status == 200
    assert question["answer"]["matched"] is True  # type: ignore[index]
    assert question["candidates"] is None
    assert get(page, "/api/state")["done"] == [KEY]
    assert post(page, f"/api/question/{KEY}/first", {"amounts": ["3.00"]})[0] == 409
    assert post(page, f"/api/question/{KEY}/final", {"amounts": ["3.00"]})[0] == 409
    saved = load_log(tmp_path / "log.json", "owner").answers[KEY]
    assert saved.final is not None
    assert saved.final.amounts == (Decimal("30.00"),)
    assert saved.commit == "abc"


def test_a_different_first_answer_reveals_the_readings_without_sources(tmp_path: Path) -> None:
    page = app(tmp_path, PageLanguage.ENGLISH)
    status, question = post(page, f"/api/question/{KEY}/first", {"amounts": ["33.00"]})
    assert status == 200
    candidates = question["candidates"]
    assert isinstance(candidates, list)
    assert sorted(c["amounts"][0] for c in candidates) == ["3.00", "30.00"]
    assert all("sources" not in json.dumps(c) for c in candidates)
    assert get(page, "/api/state")["done"] == []
    status, question = post(page, f"/api/question/{KEY}/final", {"amounts": ["30.00"]})
    assert status == 200
    assert question["answer"]["final"] == {"amounts": ["30.00"], "items": [], "tax_rows": []}  # type: ignore[index]
    assert (
        get(page, f"/api/question/{KEY}")["candidates"] is not None
    )  # still shown, for the record


def test_requests_that_are_refused(tmp_path: Path) -> None:
    page = app(tmp_path)
    assert post(page, f"/api/question/{KEY}/final", {"amounts": ["30.00"]})[0] == 409
    assert page.handle("POST", f"/api/question/{KEY}/first", b"{}", "text/plain").status == 415
    assert post(page, f"/api/question/{KEY}/first", ["not", "an", "object"])[0] == 400
    assert post(page, f"/api/question/{KEY}/first", {"amounts": ["thirty"]})[0] == 400
    assert post(page, "/api/unknown", {})[0] == 404
    assert page.handle("PUT", "/").status == 405
    assert AmountRole.TOTAL_DUE.value in KEY


def test_only_the_owner_adjudicates(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only the owner"):
        AdjudicationApp(
            QUESTIONS,
            lambda _: np.zeros((1, 1), np.uint8),
            tmp_path / "l.json",
            reviewer=ASSISTANT,
            commit=None,
        )
