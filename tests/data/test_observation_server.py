import http.client
import io
import json
import threading
from collections.abc import Iterator
from pathlib import Path
from typing import Any

import numpy as np
import pytest
from PIL import Image

from docfraud.data.normalize import load_grayscale
from docfraud.data.observation import ASSISTANT, load_log
from docfraud.data.observation_server import ObservationApp, Response, make_server
from tests.data.test_observation import CLEAN, FOLD, observation_documents, region_answers

JSON = "application/json"
LABEL_WORDS = ("cpi", "techniques", "entity", "total_payment", "metadata", "gimp", "software")


@pytest.fixture
def app(tmp_path: Path) -> ObservationApp:
    return ObservationApp(
        observation_documents(tmp_path / "d"), tmp_path / "logs" / "emre.json", "emre", "abc1234"
    )


def body(response: Response) -> Any:
    assert response.content_type == JSON
    return json.loads(response.body)


def post(app: ObservationApp, target: str, payload: dict[str, Any]) -> Response:
    return app.handle("POST", target, json.dumps(payload, default=str).encode(), JSON)


def page_answers(**changes: object) -> dict[str, object]:
    answers = region_answers(**changes)
    del answers["observed_at"], answers["commit"]  # the server adds them
    return answers | {
        "note": "",
        "display": {
            "device_pixel_ratio": 2,
            "screen_width": 1280,
            "screen_height": 832,
            "window_width": 1280,
            "window_height": 760,
        },
    }


def test_the_page_and_the_plan_reveal_no_labels(app: ObservationApp) -> None:
    page = app.handle("GET", "/")
    assert page.status == 200
    assert b"Forgery observation" in page.body
    state = app.handle("GET", "/api/state")
    data = body(state)
    assert len(data["regions"]) == 3
    assert sorted(data["receipts"]) == sorted(FOLD[:2])
    assert data["form"][0]["name"] == "visibility"
    responses = [state] + [app.handle("GET", f"/api/region/{key}") for key in data["regions"]]
    responses += [app.handle("GET", f"/api/receipt/{doc_id}") for doc_id in data["receipts"]]
    for response in responses:
        text = response.body.decode().lower()
        assert not [word for word in LABEL_WORDS if word in text]


def test_a_region_comes_as_a_strip_with_its_rectangle(app: ObservationApp) -> None:
    data = body(app.handle("GET", f"/api/region/{FOLD[0]}-0"))
    assert (data["top"], data["bottom"], data["width"], data["height"]) == (0, 60, 40, 60)
    assert data["box"] == {"x": 5, "y": 10, "width": 10, "height": 8}
    assert data["answer"] is None
    receipt = body(app.handle("GET", f"/api/receipt/{FOLD[0]}"))
    assert [b["y"] for b in receipt["boxes"]] == [10, 40]


def test_images_are_the_normalised_rows_requested(app: ObservationApp, tmp_path: Path) -> None:
    response = app.handle("GET", f"/image/{FOLD[0]}.png?top=10&bottom=30")
    assert (response.status, response.content_type) == (200, "image/png")
    pixels = np.asarray(Image.open(io.BytesIO(response.body)))
    expected = load_grayscale(tmp_path / "d" / "train" / f"{FOLD[0]}.png")[10:30]
    assert (pixels == expected).all()
    assert app.handle("GET", f"/image/{FOLD[0]}.png?top=30&bottom=10").status == 400
    assert app.handle("GET", f"/image/{FOLD[0]}.png?top=x").status == 400


def test_receipts_outside_the_design_fold_are_never_shown(app: ObservationApp) -> None:
    assert app.handle("GET", f"/image/{CLEAN}.png").status == 403
    assert app.handle("GET", f"/image/{FOLD[2]}.png").status == 404  # design fold, not planned
    assert app.handle("GET", f"/api/receipt/{CLEAN}").status == 404
    assert app.handle("GET", "/api/region/unknown").status == 404


def sample_keys(app: ObservationApp) -> tuple[str, str]:
    state = body(app.handle("GET", "/api/state"))
    full = state["full_form"][0]
    short = next(key for key in state["regions"] if key not in state["full_form"])
    return full, short


def test_answers_are_saved_at_once_and_resumed(app: ObservationApp, tmp_path: Path) -> None:
    full, short = sample_keys(app)
    assert body(post(app, f"/api/region/{full}", page_answers())) == {"saved": full}
    log_path = tmp_path / "logs" / "emre.json"
    saved = load_log(log_path, "emre").regions[full]
    assert (saved.commit, saved.display and saved.display.device_pixel_ratio) == ("abc1234", 2)
    visibility = {"visibility": "not_found", "note": "", "display": None}
    assert post(app, f"/api/region/{short}", visibility).status == 200
    resumed = ObservationApp(observation_documents(tmp_path / "d"), log_path, "emre", "def5678")
    assert sorted(body(resumed.handle("GET", "/api/state"))["answered"]["regions"]) == sorted(
        [full, short]
    )
    assert body(resumed.handle("GET", f"/api/region/{full}"))["answer"]["tone"] == "darker"
    assert body(resumed.handle("GET", f"/api/region/{short}"))["answer"]["tone"] is None


def test_the_full_form_is_asked_on_the_sample_only(app: ObservationApp) -> None:
    full, short = sample_keys(app)
    response = post(app, f"/api/region/{short}", page_answers())
    assert (response.status, body(response)["error"]) == (
        400,
        f"region {short} asks visibility only",
    )
    response = post(app, f"/api/region/{full}", {"visibility": "at_100"})
    assert (response.status, body(response)["error"]) == (400, f"region {full} asks the full form")
    assert app.log.regions == {}


def test_invalid_answers_are_refused_and_change_nothing(app: ObservationApp) -> None:
    key = sample_keys(app)[0]
    assert post(app, f"/api/region/{key}", page_answers(visibility=None)).status == 400
    assert post(app, f"/api/region/{key}", page_answers(tone="purple")).status == 400
    receipt = {"unannotated_edit": "yes", "global_trace": "no"}
    assert post(app, f"/api/receipt/{FOLD[0]}", receipt).status == 400
    assert post(app, "/api/region/unknown", page_answers()).status == 404
    assert app.handle("POST", f"/api/region/{key}", b"[]", JSON).status == 400
    assert app.handle("POST", f"/api/region/{key}", b"{", JSON).status == 400
    plain = app.handle("POST", f"/api/region/{key}", b"{}", "text/plain")
    assert plain.status == 415  # a cross-site form cannot send JSON without a preflight
    assert app.handle("PUT", "/api/state").status == 405
    assert app.log.regions == {}


def test_receipt_answers_are_saved(app: ObservationApp) -> None:
    answers = {
        "unannotated_edit": "yes",
        "unannotated_edit_where": "the date at the top",
        "global_trace": "unclear",
    }
    assert post(app, f"/api/receipt/{FOLD[1]}", answers).status == 200
    assert app.log.receipts[FOLD[1]].unannotated_edit_where == "the date at the top"


def test_the_assistant_does_not_use_the_page(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="exported crops"):
        ObservationApp(observation_documents(tmp_path / "d"), tmp_path / "c.json", ASSISTANT, None)


@pytest.fixture
def server(app: ObservationApp) -> Iterator[int]:
    running = make_server(app, 0)
    thread = threading.Thread(target=running.serve_forever, daemon=True)
    thread.start()
    yield running.server_port
    running.shutdown()
    running.server_close()


def request(port: int, method: str, target: str, **headers: str) -> tuple[int, bytes]:
    connection = http.client.HTTPConnection("127.0.0.1", port, timeout=5)
    connection.request(method, target, body=b"{}" if method == "POST" else None, headers=headers)
    response = connection.getresponse()
    result = response.status, response.read()
    connection.close()
    return result


def test_the_server_answers_local_requests_only(server: int) -> None:
    status, payload = request(server, "GET", "/api/state")
    assert status == 200
    assert json.loads(payload)["observer"] == "emre"
    assert request(server, "GET", "/api/state", Host="attacker.example")[0] == 403
    assert request(server, "POST", f"/api/region/{FOLD[0]}-0", **{"Content-Type": JSON})[0] == 400
