import base64
import io
import json
from pathlib import Path

import numpy as np
import pytest
from PIL import Image as PilImage

from docfraud.llm.claude import params
from docfraud.llm.cost import Ledger
from docfraud.llm.detector import (
    IMAGE_END,
    IMAGE_START,
    LONG_EDGE,
    MAX_PIXELS,
    SONNET,
    DetectorRow,
    Judgement,
    Reading,
    fitted_size,
    instruction_for,
    judgement_schema,
    page_image,
    read_judgement,
    receipt_text,
    request,
)
from docfraud.llm.detector_rounds import (
    MARGIN,
    UNSCORED,
    DevelopmentRound,
    kept_version,
    load_rounds,
    row_result,
    save_round,
    scores,
)
from docfraud.llm.provider import Image, Request, Response, Settings, Thinking, Usage
from docfraud.llm.run import Runner, worst_case
from docfraud.llm.store import ResponseStore
from docfraud.reading.layout import Row, Segment
from tests.fixtures.llm import FakeDetector

INSTRUCTION = f"Judge the receipt.\n\n{IMAGE_START}\nLook at the image.\n{IMAGE_END}\n\nAnswer.\n"
ROWS = [Row((Segment("TOTAL"), Segment("12.00"))), Row((Segment("CASH 20.00"),))]
WRITTEN = [Path(f"src/docfraud/llm/instructions/detector-v{v}.md") for v in (1, 2)]


def page(height: int = 40, width: int = 30) -> np.ndarray:
    return np.random.default_rng(0).integers(0, 256, size=(height, width), dtype=np.uint8)


def test_each_row_gets_the_instruction_with_or_without_its_part_about_the_image() -> None:
    main = instruction_for(INSTRUCTION, DetectorRow.IMAGE_AND_OCR)
    text = instruction_for(INSTRUCTION, DetectorRow.TRANSCRIPT)
    assert "Look at the image." in main
    assert IMAGE_START not in main
    assert IMAGE_END not in main
    assert "image" not in text
    assert text == "Judge the receipt.\n\n\n\nAnswer.\n"
    with pytest.raises(ValueError, match="once"):
        instruction_for("no markers", DetectorRow.TRANSCRIPT)


@pytest.mark.parametrize("path", WRITTEN, ids=lambda p: p.stem)
def test_each_version_marks_its_part_about_the_image_once(path: Path) -> None:
    written = path.read_text(encoding="utf-8")
    transcript_copy = instruction_for(written, DetectorRow.TRANSCRIPT)
    assert "image" not in transcript_copy.lower()
    assert "suspicion" in transcript_copy
    assert len(instruction_for(written, DetectorRow.IMAGE_AND_OCR)) > len(transcript_copy)


def test_a_page_is_made_smaller_only_to_fit_the_limits() -> None:
    assert fitted_size(800, 1200) == (800, 1200)
    width, height = fitted_size(2000, 7000)
    assert max(width, height) <= LONG_EDGE
    assert width * height <= MAX_PIXELS
    assert abs(width / height - 2000 / 7000) < 0.01
    wide, tall = fitted_size(2500, 2500)
    assert wide * tall <= MAX_PIXELS < 2500 * 2500


def test_the_same_page_always_gives_the_same_image() -> None:
    first, second = page_image(page()), page_image(page())
    assert first == second
    decoded = PilImage.open(io.BytesIO(base64.standard_b64decode(first.data)))
    assert (decoded.mode, decoded.size) == ("L", (30, 40))
    assert np.array_equal(np.asarray(decoded), page())
    big = page_image(np.zeros((6000, 3000), dtype=np.uint8))
    assert (big.width, big.height) == fitted_size(3000, 6000)
    with pytest.raises(ValueError, match="grayscale"):
        page_image(np.zeros((4, 4, 3), dtype=np.uint8))


def test_the_main_row_sends_the_image_before_the_text_and_the_other_the_text_only() -> None:
    image = page_image(page())
    main = request(DetectorRow.IMAGE_AND_OCR, SONNET, INSTRUCTION, ROWS, image)
    text = request(DetectorRow.TRANSCRIPT, SONNET, INSTRUCTION, ROWS)
    assert main.user == text.user == receipt_text(ROWS) == "TOTAL 12.00\nCASH 20.00"
    assert main.images == (image,)
    assert main.answer_schema == judgement_schema() == Judgement.model_json_schema()
    sent = next(iter(params(main)["messages"]))["content"]
    assert [block["type"] for block in sent] == ["image", "text"]  # type: ignore[index]
    assert next(iter(params(text)["messages"]))["content"] == text.user
    with pytest.raises(ValueError, match="with an image"):
        request(DetectorRow.IMAGE_AND_OCR, SONNET, INSTRUCTION, ROWS)
    with pytest.raises(ValueError, match="without an image"):
        request(DetectorRow.TRANSCRIPT, SONNET, INSTRUCTION, ROWS, image)


def test_an_image_changes_the_key_but_a_request_without_images_keeps_its_old_key() -> None:
    settings = Settings(
        model="claude-haiku-5-5", max_tokens=4000, thinking=Thinking.DISABLED, effort="medium"
    )
    plain = Request(
        settings=settings,
        system="Read the receipt.",
        user="r0: CASH 52.00",
        answer_schema={"type": "object", "properties": {"a": {"type": "string"}}},
    )
    # the key this request had before requests could carry images
    assert plain.key == "1f7ec0396f623d931a1751a83d4a2f290b5476200628b82b7229a635065dfaa3"
    one = plain.model_copy(update={"images": (page_image(page()),)})
    other = plain.model_copy(update={"images": (page_image(page(41)),)})
    assert len({plain.key, one.key, other.key}) == 3


def test_an_image_counts_in_the_worst_case_cost() -> None:
    image = Image(data="", width=1500, height=1500)
    assert image.tokens == 3000
    text = request(DetectorRow.TRANSCRIPT, SONNET, INSTRUCTION, ROWS)
    with_image = text.model_copy(update={"images": (image,)})
    assert worst_case([with_image]) > worst_case([text])


def response(answer: str | None, **change: object) -> Response:
    fields: dict[str, object] = {
        "key": "k",
        "model": "claude-sonnet-5-5",
        "answer": answer,
        "stop_reason": "end_turn" if answer is not None else "refusal",
        "usage": Usage(),
        "batch": True,
    }
    return Response.model_validate(fields | change)


def test_a_judgement_is_read_or_its_failure_named() -> None:
    good = read_judgement(response(json.dumps({"suspicion": 73, "reasons": ["the total"]})))
    assert (good.suspicion, good.reasons, good.failure) == (73, ("the total",), None)
    assert read_judgement(None).failure == "not answered"
    assert read_judgement(response(None)).failure == "refusal"
    assert read_judgement(response(None, error="overloaded_error")).failure == "overloaded_error"
    assert read_judgement(response("not json")).failure == "answer form: JSONDecodeError"
    assert read_judgement(response('{"suspicion": "high"}')).failure == (
        "answer form: ValidationError"
    )
    assert read_judgement(response('{"suspicion": 140, "reasons": []}')).failure == (
        "score 140 outside 0 to 100"
    )


def test_the_runner_answers_any_requests_but_one_model_at_a_time(tmp_path: Path) -> None:
    runner = Runner(
        FakeDetector(), ResponseStore(tmp_path / "store"), Ledger(tmp_path / "ledger.jsonl")
    )
    built = [
        request(DetectorRow.TRANSCRIPT, SONNET, INSTRUCTION, [Row((Segment(text),))])
        for text in ("PRINTED", "EDITED", "PRINTED")
    ]
    answered = runner.answer(built)
    assert [read_judgement(r).suspicion for r in answered] == [10, 90, 10]
    haiku = built[0].model_copy(update={"settings": SONNET.model_copy(update={"model": "x"})})
    with pytest.raises(ValueError, match="one model per run"):
        runner.answer([built[0], haiku])


def test_unscored_receipts_rank_below_every_score() -> None:
    readings = [Reading(90), Reading(None, failure="refusal"), Reading(0)]
    assert list(scores(readings)) == [90.0, UNSCORED, 0.0]
    result = row_result([True, True, False], readings, "0.0100")
    assert (result.receipts, result.forged, result.failures) == (3, 2, 1)
    assert 0 < result.average_precision < 1
    with pytest.raises(ValueError, match="one reading"):
        row_result([True], readings, "0")


def development(version: int, ap: float, failures: int = 0) -> DevelopmentRound:
    result = row_result([True, False], [Reading(80), Reading(20)], "0.01")
    main = result.model_copy(update={"average_precision": ap, "failures": failures})
    return DevelopmentRound(
        version=version,
        settings=SONNET,
        instruction=f"detector-v{version}.md",
        instruction_sha256="0" * 64,
        commit="a" * 40,
        rows={DetectorRow.IMAGE_AND_OCR.value: main, DetectorRow.TRANSCRIPT.value: result},
    )


def test_a_revision_is_kept_only_if_clearly_better_without_more_failures() -> None:
    first = development(1, 0.40, failures=2)
    assert kept_version({1: first}) == 1
    assert kept_version({1: first, 2: development(2, 0.40 + MARGIN, failures=2)}) == 2
    assert kept_version({1: first, 2: development(2, 0.40 + MARGIN - 0.01)}) == 1
    assert kept_version({1: first, 2: development(2, 0.60, failures=3)}) == 1
    with pytest.raises(ValueError, match="version 1"):
        kept_version({2: development(2, 0.6)})
    with pytest.raises(ValueError, match="versions"):
        kept_version({1: first, 3: development(3, 0.9)})


def test_rounds_are_saved_and_loaded_by_version(tmp_path: Path) -> None:
    for version in (1, 2):
        save_round(development(version, 0.5), tmp_path / f"round-v{version}.json")
    loaded = load_rounds(tmp_path)
    assert set(loaded) == {1, 2}
    assert loaded[2] == development(2, 0.5)
