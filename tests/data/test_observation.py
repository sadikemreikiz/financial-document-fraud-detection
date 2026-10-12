import typing
from datetime import UTC, datetime
from pathlib import Path

import numpy as np
import pytest
from PIL import Image
from pydantic import BaseModel, ValidationError

from docfraud.data.models import BBox, Document
from docfraud.data.normalize import load_grayscale
from docfraud.data.observation import (
    ASSISTANT,
    FORM,
    ObservationLog,
    Presence,
    ReceiptObservation,
    RegionObservation,
    crop_rows,
    export_crops,
    load_log,
    observation_plan,
    observed_pixels,
    save_log,
    seal,
    seal_matches,
)
from docfraud.evaluation.design_fold import DESIGN_FOLD_IDS, NotInDesignFoldError
from tests.fixtures.findit import Entry, annotation, load_toy, region, write_dataset

# Identifiers of real receipts, so that the viewing lock applies; the images are toy images.
FOLD = ("X51005230617", "X51005230621", "X51005268472")
CLEAN = "X00016469622"
NOW = datetime(2026, 10, 12, tzinfo=UTC)


def observation_documents(root: Path) -> tuple[Document, ...]:
    entries = [
        Entry(
            FOLD[0],
            forged="1",
            annotation=annotation(
                FOLD[0],
                [
                    region(x=5, y=10, width=10, height=8),
                    region(x=20, y=30, width=5, height=5, modified={"None": True}, original="yes"),
                    region(x=1, y=40, width=6, height=5, entity="Metadata"),
                ],
            ),
        ),
        Entry(
            FOLD[1],
            forged="1",
            annotation=annotation(FOLD[1], [region(x=2, y=20, width=8, height=6)]),
        ),
        Entry(FOLD[2]),
        Entry(CLEAN, forged="1", annotation=annotation(CLEAN, [region()])),
    ]
    return load_toy(write_dataset(root, entries))


def region_answers(**changes: object) -> dict[str, object]:
    answers: dict[str, object] = {
        "visibility": "at_100",
        "change": "single_characters",
        "source": "other_line",
        "background": "matches",
        "edge": "not_visible",
        "alignment": "aligned",
        "font": "same",
        "tone": "darker",
        "sharpness": "same",
        "scaled_or_rotated": "no",
        "box_fit": "tight",
        "observed_at": NOW,
        "commit": "abc1234",
    }
    return answers | changes


def receipt_answers(**changes: object) -> dict[str, object]:
    answers: dict[str, object] = {
        "unannotated_edit": "no",
        "global_trace": "no",
        "observed_at": NOW,
        "commit": "abc1234",
    }
    return answers | changes


def test_toy_identifiers_sit_where_the_tests_assume() -> None:
    assert set(FOLD) <= DESIGN_FOLD_IDS
    assert CLEAN not in DESIGN_FOLD_IDS


def test_the_form_asks_every_recorded_answer_once() -> None:
    assert sorted(f.number for f in FORM) == list(range(1, 14))
    assert FORM[0].name == "visibility"  # asked before the rectangle is shown
    records: dict[str, type[BaseModel]] = {
        "region": RegionObservation,
        "receipt": ReceiptObservation,
    }
    for field in FORM:
        annotation_type = records[field.scope].model_fields[field.name].annotation
        enum = (typing.get_args(annotation_type) or (annotation_type,))[0]  # drops "| None"
        assert set(field.choices) == {member.value for member in enum}, field.name
        if field.detail:
            assert field.detail in records[field.scope].model_fields
    assert {f.name for f in FORM if not f.assistant} == {
        "visibility",
        "unannotated_edit",
        "global_trace",
    }


def test_plan_covers_the_design_fold_forgeries_in_a_seeded_order(tmp_path: Path) -> None:
    documents = observation_documents(tmp_path / "d")
    plan = observation_plan(documents)
    assert sorted(item.key for item in plan.regions) == [
        f"{FOLD[0]}-0",
        f"{FOLD[0]}-1",
        f"{FOLD[1]}-0",
    ]
    assert sorted(plan.receipts) == sorted(FOLD[:2])
    assert observation_plan(reversed(documents)) == plan
    second = {item.key: item.box for item in plan.regions}[f"{FOLD[0]}-1"]
    assert second == BBox(x=1, y=40, width=6, height=5)  # the copy source is not a region


@pytest.mark.parametrize(
    ("box", "expected"),
    [
        (BBox(x=0, y=500, width=10, height=20), (300, 720)),
        (BBox(x=0, y=50, width=10, height=20), (0, 270)),
        (BBox(x=0, y=900, width=10, height=90), (700, 1000)),
    ],
)
def test_crop_rows_stay_inside_the_image(box: BBox, expected: tuple[int, int]) -> None:
    assert crop_rows(box, height=1000) == expected


def test_observed_pixels_refuse_receipts_outside_the_design_fold(tmp_path: Path) -> None:
    documents = {d.doc_id: d for d in observation_documents(tmp_path / "d")}
    with pytest.raises(NotInDesignFoldError):
        observed_pixels(documents[CLEAN])
    assert observed_pixels(documents[FOLD[1]]).shape == (60, 40)


def test_exported_crops_mark_the_region_without_changing_it(tmp_path: Path) -> None:
    root = tmp_path / "d"
    box = region(x=1650, y=300, width=40, height=20)
    entries = [Entry(FOLD[0], forged="1", annotation=annotation(FOLD[0], [box]))]
    write_dataset(root, entries)
    rng = np.random.default_rng(1)
    noise = rng.integers(0, 256, size=(700, 1700), dtype=np.uint8)
    Image.fromarray(noise).save(root / "train" / f"{FOLD[0]}.png")
    documents = load_toy(root)

    files = export_crops(documents, observation_plan(documents), tmp_path / "crops")

    names = [f.name for f in files]
    assert names == [
        f"01-{FOLD[0]}-0-close-up-0.png",
        f"01-{FOLD[0]}-0-strip-0.png",
        f"01-{FOLD[0]}-0-strip-1.png",
    ]
    page = load_grayscale(root / "train" / f"{FOLD[0]}.png")
    close_up = np.asarray(Image.open(files[0]))
    assert close_up.shape == (420, 250, 3)  # rows 100-520, columns 1450-1700
    inside = close_up[200:220, 200:240]
    assert (inside == page[300:320, 1650:1690, None]).all()
    assert tuple(close_up[198, 210]) == (220, 0, 0)  # outline just above the region
    strip = [np.asarray(Image.open(f)) for f in files[1:]]
    assert [s.shape for s in strip] == [(420, 1500, 3), (420, 200, 3)]


def test_logs_are_saved_and_resumed(tmp_path: Path) -> None:
    path = tmp_path / "logs" / "emre.json"
    assert load_log(path, "emre") == ObservationLog(observer="emre")
    log = ObservationLog(
        observer="emre",
        regions={f"{FOLD[0]}-0": RegionObservation.model_validate(region_answers())},
        receipts={FOLD[0]: ReceiptObservation.model_validate(receipt_answers())},
    )
    save_log(log, path)
    assert load_log(path, "emre") == log
    assert not path.with_name("emre.json.partial").exists()
    with pytest.raises(ValueError, match="log of 'emre'"):
        load_log(path, "someone")


def test_the_owner_answers_visibility_and_the_assistant_does_not() -> None:
    region = RegionObservation.model_validate(region_answers(visibility=None))
    with pytest.raises(ValidationError, match="answer on visibility"):
        ObservationLog(observer="emre", regions={"k": region})
    assert ObservationLog(observer=ASSISTANT, regions={"k": region}).regions
    with pytest.raises(ValidationError, match="assistant answers neither"):
        ObservationLog(
            observer=ASSISTANT, regions={"k": RegionObservation.model_validate(region_answers())}
        )


def test_a_yes_on_the_whole_receipt_needs_a_description() -> None:
    with pytest.raises(ValidationError, match="needs a description"):
        ReceiptObservation.model_validate(receipt_answers(global_trace="yes"))
    described = receipt_answers(global_trace="yes", global_trace_what="lighter everywhere")
    assert ReceiptObservation.model_validate(described).global_trace is Presence.YES


def test_logs_of_another_form_version_are_refused() -> None:
    with pytest.raises(ValidationError, match="form version 0"):
        ObservationLog(observer="emre", form_version=0)


def test_a_seal_detects_any_later_change(tmp_path: Path) -> None:
    log_path, seal_path = tmp_path / "claude.json", tmp_path / "records" / "claude.sha256"
    save_log(ObservationLog(observer=ASSISTANT), log_path)
    digest = seal(log_path, seal_path)
    assert seal_path.read_text(encoding="utf-8") == f"{digest}  claude.json\n"
    assert seal_matches(log_path, seal_path)
    log_path.write_text(log_path.read_text(encoding="utf-8") + " ", encoding="utf-8")
    assert not seal_matches(log_path, seal_path)
