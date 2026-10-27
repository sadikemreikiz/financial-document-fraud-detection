"""The standalone LLM detector of Phase 6b (ADR-0031, sections 1 to 4).

A language model judges one receipt and answers with a suspicion score from 0 to 100 and short
reasons. It is a baseline outside the system: its score never enters fusion, the decision policy
or the audit report.

Two rows are measured:

- **image and OCR text** (track T2): the receipt's normalised grayscale page, as every system
  sees it, encoded as PNG and made smaller here, if needed, to fit the provider's limits, so
  that the request is the same each time it is built; with it, the OCR text rows;
- **transcript** (track T1): the dataset's transcript alone, the input of the published ChatGPT
  baseline.

One instruction serves both rows. The transcript row's copy leaves out the part between
:data:`IMAGE_START` and :data:`IMAGE_END`, which is about the image; the main row keeps it
without the markers. The model answers in a fixed form (structured output); an answer outside
it, or a score outside 0 to 100, is a failure of the answer form, counted in development.
"""

import base64
import io
import json
from collections.abc import Sequence
from dataclasses import dataclass
from enum import StrEnum

import numpy as np
import numpy.typing as npt
from PIL import Image as PilImage
from pydantic import BaseModel, ConfigDict, Field, ValidationError

from docfraud.llm.provider import (
    IMAGE_PIXELS_PER_TOKEN,
    Image,
    Request,
    Response,
    Settings,
    Thinking,
)
from docfraud.reading.layout import Row

LONG_EDGE = 2576
"""The longest image edge the models of ADR-0031 accept without making the image smaller."""

IMAGE_TOKENS = 4784
"""The most input tokens an image costs at that size."""

MAX_PIXELS = IMAGE_TOKENS * IMAGE_PIXELS_PER_TOKEN
"""The largest image area sent, so that the provider never makes an image smaller itself."""

SONNET = Settings(
    model="claude-sonnet-5-5", max_tokens=8000, thinking=Thinking.ADAPTIVE, effort="medium"
)
"""Both rows of the detector (ADR-0031, section 4)."""

HAIKU = Settings(
    model="claude-haiku-5-5", max_tokens=8000, thinking=Thinking.ADAPTIVE, effort="medium"
)
"""The descriptive row of the cheapest current model, on the main row's input."""

IMAGE_START = "<!-- image -->"
IMAGE_END = "<!-- /image -->"


class DetectorRow(StrEnum):
    """What the detector sees."""

    IMAGE_AND_OCR = "image and OCR text"
    TRANSCRIPT = "transcript"


class Judgement(BaseModel):
    """The detector's answer for one receipt."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    suspicion: int = Field(description="How likely the receipt was forged, from 0 to 100")
    reasons: list[str] = Field(description="Up to five short reasons for the score")


def judgement_schema() -> dict[str, object]:
    """The JSON schema of :class:`Judgement`, in the form the structured-output API accepts."""
    return Judgement.model_json_schema()


def instruction_for(text: str, row: DetectorRow) -> str:
    """The instruction one row receives: with or without its part about the image.

    Raises:
        ValueError: If the part about the image is not marked exactly once.
    """
    if text.count(IMAGE_START) != 1 or text.count(IMAGE_END) != 1:
        msg = f"the instruction must mark its part about the image once with {IMAGE_START}"
        raise ValueError(msg)
    before, rest = text.split(IMAGE_START)
    image_part, after = rest.split(IMAGE_END)
    kept = image_part if row is DetectorRow.IMAGE_AND_OCR else ""
    return (before + kept + after).strip() + "\n"


def fitted_size(width: int, height: int) -> tuple[int, int]:
    """The size an image is sent at: within the longest edge and the largest area."""
    scale = min(1.0, LONG_EDGE / max(width, height))
    scale = min(scale, (MAX_PIXELS / (width * height)) ** 0.5)
    return max(1, int(width * scale)), max(1, int(height * scale))


def page_image(page: npt.NDArray[np.uint8]) -> Image:
    """A normalised grayscale page as the image the main row sends.

    The page is made smaller with a Lanczos filter if it exceeds the provider's limits, and
    encoded as PNG without metadata, so the same page always gives the same request.

    Raises:
        ValueError: If the page is not a two-dimensional uint8 grayscale image.
    """
    if page.ndim != 2 or page.dtype != np.uint8:  # noqa: PLR2004 - a page has two axes
        msg = "a page is a two-dimensional uint8 grayscale image"
        raise ValueError(msg)
    height, width = page.shape
    size = fitted_size(width, height)
    picture = PilImage.fromarray(page, mode="L")
    if size != (width, height):
        picture = picture.resize(size, PilImage.Resampling.LANCZOS)
    buffer = io.BytesIO()
    picture.save(buffer, format="PNG", optimize=False)
    data = base64.standard_b64encode(buffer.getvalue()).decode("ascii")
    return Image(data=data, width=size[0], height=size[1])


def receipt_text(rows: Sequence[Row]) -> str:
    """The text the detector reads: one line per row, as printed."""
    return "\n".join(row.text for row in rows)


def request(
    row: DetectorRow,
    settings: Settings,
    instruction: str,
    rows: Sequence[Row],
    page: Image | None = None,
) -> Request:
    """The request of one receipt for one row.

    Raises:
        ValueError: If the main row lacks the page or the transcript row is given one.
    """
    if (row is DetectorRow.IMAGE_AND_OCR) != (page is not None):
        msg = f"the {row.value} row is sent {'with' if page is None else 'without'} an image"
        raise ValueError(msg)
    return Request(
        settings=settings,
        system=instruction_for(instruction, row),
        user=receipt_text(rows),
        answer_schema=judgement_schema(),
        images=(page,) if page is not None else (),
    )


@dataclass(frozen=True, slots=True)
class Reading:
    """The detector's reading of one receipt: its score, or why there is none."""

    suspicion: int | None
    reasons: tuple[str, ...] = ()
    failure: str | None = None


def read_judgement(response: Response | None) -> Reading:
    """The score and reasons of a stored response, or the failure of its answer form."""
    if response is None:
        return Reading(None, failure="not answered")
    if response.answer is None:
        return Reading(None, failure=response.error or response.stop_reason or "no answer")
    try:
        judgement = Judgement.model_validate(json.loads(response.answer))
    except (json.JSONDecodeError, ValidationError) as error:
        return Reading(None, failure=f"answer form: {type(error).__name__}")
    if not 0 <= judgement.suspicion <= 100:  # noqa: PLR2004 - the scale of the answer
        return Reading(None, failure=f"score {judgement.suspicion} outside 0 to 100")
    return Reading(judgement.suspicion, tuple(judgement.reasons))
