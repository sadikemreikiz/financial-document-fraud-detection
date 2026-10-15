import json
from collections import Counter
from pathlib import Path

import numpy as np
import pytest

from docfraud.data.models import BBox, Entity, Technique
from docfraud.data.observation import Visibility
from docfraud.evaluation.design_fold import DESIGN_FOLD, DESIGN_FOLD_IDS
from docfraud.markdown import render_table
from docfraud.synthesis.engine import ENGINE_VERSION, ForgeryRecord, Stream, SyntheticRegion
from docfraud.synthesis.review import (
    FOUND_LIMIT,
    REVIEW_ITEMS,
    ReviewItem,
    ReviewLog,
    load_log,
    log_path,
    outcome,
    review_sample,
    review_tables,
    save_log,
)
from docfraud.synthesis.review_server import ReviewApp

BASES = sorted(DESIGN_FOLD_IDS)[:30]
DISPLAY = {
    "device_pixel_ratio": 2.0,
    "screen_width": 1470,
    "screen_height": 956,
    "window_width": 1400,
    "window_height": 900,
}


def record(base: str, regions: int = 2, stream: Stream = Stream.DEVELOPMENT) -> ForgeryRecord:
    techniques = (Technique.CPI, Technique.IMI, Technique.CUT)
    return ForgeryRecord(
        engine_version=ENGINE_VERSION,
        base=base,
        held_out=DESIGN_FOLD,
        stream=stream,
        index=0,
        regions=tuple(
            SyntheticRegion(
                technique=techniques[n % 3],
                entity=Entity.TOTAL_PAYMENT,
                box=BBox(x=10 + 30 * n, y=300, width=12, height=20),
                span=1,
                before="12.50",
                after="18.50",
                variants={},
            )
            for n in range(regions)
        ),
        failures={},
    )


RECORDS = [record(base) for base in BASES]


def test_the_sample_is_fixed_by_its_round_and_draws_from_every_forgery() -> None:
    first = review_sample(RECORDS, 1)
    assert len(first) == REVIEW_ITEMS
    assert review_sample(list(reversed(RECORDS)), 1) == first  # order of the records is irrelevant
    assert review_sample(RECORDS, 2) != first
    assert len({item.key for item in first}) == REVIEW_ITEMS


def test_the_sample_takes_only_the_development_set() -> None:
    with pytest.raises(ValueError, match="development set"):
        review_sample([record(BASES[0], stream=Stream.TRAINING)], 1)
    with pytest.raises(ValueError, match="needs"):
        review_sample(RECORDS[:3], 1)


def _answer(visibility: Visibility, note: str = "") -> dict[str, object]:
    return {
        "visibility": visibility.value,
        "plausible": "yes",
        "note": note,
        "display": DISPLAY,
        "answered_at": "2026-10-15T10:00:00Z",
        "commit": "abc1234",
        "language": "tr",
    }


def _log(found: int, items: tuple[ReviewItem, ...]) -> ReviewLog:
    answers = {
        item.key: _answer(Visibility.AT_100 if n < found else Visibility.NOT_FOUND, "x" * (n == 0))
        for n, item in enumerate(items)
    }
    return ReviewLog.model_validate({"reviewer": "emre", "round": 1, "answers": answers})


def test_five_edits_found_at_100_percent_exceed_the_criterion() -> None:
    items = review_sample(RECORDS, 1)
    below = outcome(_log(FOUND_LIMIT - 1, items), items, RECORDS)
    at = outcome(_log(FOUND_LIMIT, items), items, RECORDS)
    assert (below.found, below.exceeded) == (4, False)
    assert (at.found, at.exceeded) == (5, True)
    assert sum(sum(c.values()) for c in at.by_technique.values()) == REVIEW_ITEMS
    with pytest.raises(ValueError, match="not finished"):
        outcome(ReviewLog(reviewer="emre", round=1), items, RECORDS)


def test_the_report_compares_with_the_real_edits_and_lists_the_notes() -> None:
    items = review_sample(RECORDS, 1)
    log = _log(2, items)
    real = Counter({Visibility.AT_100: 2, Visibility.NOT_FOUND: 38, Visibility.UNCLEAR: 5})
    visible, by_technique, plausible, timing, notes = review_tables(
        outcome(log, items, RECORDS), real, items, log, RECORDS, real_seconds=[7.0, 9.0]
    )
    assert timing.rows[0][:3] == ["real, step 4.2, visibility only", 2, 8.0]
    assert visible.rows[0][1:3] == [45, 2]
    assert visible.rows[1][1:3] == [REVIEW_ITEMS, 2]
    assert len(notes.rows) == 1
    assert "Notes" in render_table(notes)
    assert by_technique.rows
    assert plausible.rows[0] == ["yes", REVIEW_ITEMS]


def test_logs_are_saved_and_belong_to_one_reviewer_and_round(tmp_path: Path) -> None:
    path = log_path(tmp_path, "emre", 1)
    items = review_sample(RECORDS, 1)
    save_log(_log(1, items), path)
    assert load_log(path, "emre", 1) == _log(1, items)
    with pytest.raises(ValueError, match="round 1 of 'emre'"):
        load_log(path, "emre", 2)
    assert load_log(tmp_path / "none.json", "emre", 1).answers == {}
    with pytest.raises(ValueError, match="version"):
        ReviewLog(version=0, reviewer="emre", round=1)


def _app(tmp_path: Path, items: tuple[ReviewItem, ...] | None = None) -> ReviewApp:
    page = np.full((900, 300), 230, dtype=np.uint8)
    return ReviewApp(
        items or review_sample(RECORDS, 1),
        lambda _base: page,
        tmp_path / "emre-1.json",
        reviewer="emre",
        round_number=1,
        commit="abc1234",
    )


def test_the_page_shows_a_strip_and_records_answers(tmp_path: Path) -> None:
    app = _app(tmp_path)
    page = app.handle("GET", "/")
    assert page.status == 200
    assert b"Realism Review" in page.body
    state = json.loads(app.handle("GET", "/api/state").body)
    assert len(state["items"]) == REVIEW_ITEMS
    assert state["ui"]["title"] == "Gerçekçilik incelemesi"
    assert "technique" not in json.dumps(state)
    key = state["items"][0]
    item = json.loads(app.handle("GET", f"/api/item/{key}").body)
    assert (item["top"], item["bottom"]) == (100, 520)  # 200 px above and below the edit
    image = app.handle("GET", f"/image/{key}.png?top=100&bottom=520")
    assert image.content_type == "image/png"
    assert app.handle("GET", f"/image/{key}.png?top=600&bottom=520").status == 400
    payload = {"visibility": "not_found", "plausible": "unclear", "display": DISPLAY}
    saved = app.handle("POST", f"/api/item/{key}", json.dumps(payload).encode(), "application/json")
    assert saved.status == 200
    assert app.log.answers[key].visibility is Visibility.NOT_FOUND
    assert key in json.loads(_app(tmp_path).handle("GET", "/api/state").body)["answered"]


def test_the_page_refuses_bad_requests(tmp_path: Path) -> None:
    app = _app(tmp_path)
    key = json.loads(app.handle("GET", "/api/state").body)["items"][0]
    assert app.handle("POST", f"/api/item/{key}", b"{}", "text/plain").status == 415
    assert app.handle("POST", f"/api/item/{key}", b"[]", "application/json").status == 400
    assert app.handle("POST", f"/api/item/{key}", b"{}", "application/json").status == 400
    assert app.handle("POST", "/api/item/unknown", b"{}", "application/json").status == 404
    assert app.handle("GET", "/api/item/unknown").status == 404
    assert app.handle("PUT", "/").status == 405


def test_only_design_fold_receipts_are_shown(tmp_path: Path) -> None:
    outside = ReviewItem("X99-0", "X99", 0, BBox(x=0, y=0, width=5, height=5))
    app = _app(tmp_path, (outside,))
    assert app.handle("GET", "/image/X99-0.png").status == 403


def test_the_assistant_does_not_review(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="only the owner"):
        ReviewApp(
            (),
            lambda _base: np.zeros((1, 1), dtype=np.uint8),
            tmp_path / "x.json",
            reviewer="claude",
            round_number=1,
            commit=None,
        )
