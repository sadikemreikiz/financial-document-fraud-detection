"""Optical character recognition behind an engine-independent interface (ADR-0008).

Engines turn a normalised grayscale page into :class:`Token` objects that keep their position on
the page, because later layers link pixel evidence to the values it touches.
"""

import hashlib
from collections.abc import Iterable, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING, Any, Protocol, cast

import numpy as np
import numpy.typing as npt
from pydantic import BaseModel, ConfigDict, Field

from docfraud.data.models import BBox

if TYPE_CHECKING:
    from rapidocr import RapidOCR


class Token(BaseModel):
    """A recognised word with its axis-aligned position and confidence."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    text: str = Field(min_length=1)
    box: BBox
    confidence: float = Field(ge=0.0, le=1.0)
    line: int = Field(ge=0, description="Index of the text line in reading order")


class OcrResult(BaseModel):
    """All tokens recognised on one page."""

    model_config = ConfigDict(frozen=True, extra="forbid")

    engine: str
    width: int = Field(gt=0)
    height: int = Field(gt=0)
    tokens: tuple[Token, ...]

    @property
    def lines(self) -> list[str]:
        """Text lines in reading order, words joined by single spaces."""
        words: dict[int, list[str]] = {}
        for token in self.tokens:
            words.setdefault(token.line, []).append(token.text)
        return [" ".join(words[index]) for index in sorted(words)]

    @property
    def text(self) -> str:
        """The page as plain text, one line per text line."""
        return "".join(f"{line}\n" for line in self.lines)


class OcrEngine(Protocol):
    """Anything that can read a normalised grayscale page."""

    @property
    def identifier(self) -> str:
        """Engine, version and settings; changes whenever outputs may change."""
        ...

    def recognize(self, image: npt.NDArray[np.uint8]) -> OcrResult:
        """Recognise all text on a ``(height, width)`` grayscale page."""
        ...


@dataclass(frozen=True, slots=True)
class RapidOcrSettings:
    """Settings of the RapidOCR engine that influence its output.

    Attributes:
        model_type: PP-OCR model size, ``"small"`` or ``"medium"``.
        max_side_len: Pages whose longer side exceeds this are downscaled before detection.
    """

    model_type: str = "small"
    max_side_len: int = 2000


class RapidOcrEngine:
    """PP-OCRv6 through RapidOCR and ONNX Runtime.

    The model files ship with the package pin and are verified by RapidOCR against SHA-256
    digests. The engine is created lazily so that importing this module stays cheap.
    """

    def __init__(self, settings: RapidOcrSettings | None = None) -> None:
        """Configure the engine; models load on first use."""
        self.settings = settings or RapidOcrSettings()

    @property
    def identifier(self) -> str:
        """For example ``rapidocr-3.9.2-PP-OCRv6-small-2000``."""
        from importlib.metadata import version  # noqa: PLC0415 - keep import cost on use

        s = self.settings
        return f"rapidocr-{version('rapidocr')}-PP-OCRv6-{s.model_type}-{s.max_side_len}"

    @cached_property
    def _engine(self) -> RapidOCR:
        from rapidocr import ModelType, OCRVersion, RapidOCR  # noqa: PLC0415

        model_type = ModelType(self.settings.model_type)
        return RapidOCR(
            params={
                "Global.log_level": "warning",
                "Global.max_side_len": self.settings.max_side_len,
                "Det.ocr_version": OCRVersion.PPOCRV6,
                "Det.model_type": model_type,
                "Rec.ocr_version": OCRVersion.PPOCRV6,
                "Rec.model_type": model_type,
            }
        )

    def recognize(self, image: npt.NDArray[np.uint8]) -> OcrResult:
        """Recognise a grayscale page; RapidOCR expects three channels."""
        from rapidocr.utils.output import RapidOCROutput  # noqa: PLC0415

        output = self._engine(np.stack([image] * 3, axis=-1), return_word_box=True)
        if not isinstance(output, RapidOCROutput):
            msg = f"expected a full OCR result, got {type(output).__name__}"
            raise TypeError(msg)
        # RapidOCR annotates word_results as one flat tuple of words, but at runtime it holds one
        # tuple of words per text line; a test against the real engine guards this shape.
        lines = cast("Iterable[Sequence[tuple[str, float, Any]]]", output.word_results)
        height, width = image.shape
        return tokens_from_word_results(
            self.identifier, width, height, lines if output.txts else ()
        )


def tokens_from_word_results(
    engine: str,
    width: int,
    height: int,
    word_results: Iterable[Sequence[tuple[str, float, Any]]],
) -> OcrResult:
    """Convert RapidOCR word results (per line: word, score, quadrilateral) into tokens.

    Quadrilaterals become the smallest enclosing axis-aligned box, clipped to the page. Words
    that are blank or have no box are dropped.
    """
    tokens = []
    for line_index, words in enumerate(word_results):
        for text, score, quad in words:
            stripped = text.strip()
            if not stripped or quad is None:
                continue
            box = _enclosing_box(quad, width, height)
            if box is not None:
                tokens.append(
                    Token(
                        text=stripped,
                        box=box,
                        confidence=float(np.clip(score, 0.0, 1.0)),
                        line=line_index,
                    )
                )
    return OcrResult(engine=engine, width=width, height=height, tokens=tuple(tokens))


def _enclosing_box(quad: Any, width: int, height: int) -> BBox | None:
    points = np.asarray(quad, dtype=float).reshape(-1, 2)
    x0, y0 = np.floor(points.min(axis=0)).astype(int)
    x1, y1 = np.ceil(points.max(axis=0)).astype(int)
    x0, y0 = max(int(x0), 0), max(int(y0), 0)
    x1, y1 = min(int(x1), width), min(int(y1), height)
    if x1 <= x0 or y1 <= y0:
        return None
    return BBox(x=x0, y=y0, width=x1 - x0, height=y1 - y0)


class CachedOcr:
    """Wraps an engine with an on-disk cache keyed by engine identifier and page content.

    Keying by the pixels rather than by file path means a re-encoded or moved file is still
    found, and a changed page never returns a stale result.
    """

    def __init__(self, engine: OcrEngine, directory: Path) -> None:
        """Store results for ``engine`` under ``directory``."""
        self.engine = engine
        self.directory = directory / engine.identifier

    @property
    def identifier(self) -> str:
        """The wrapped engine's identifier; caching never changes outputs."""
        return self.engine.identifier

    def recognize(self, image: npt.NDArray[np.uint8]) -> OcrResult:
        """Return the cached result, running the engine only on a cache miss."""
        digest = hashlib.sha256(np.ascontiguousarray(image).tobytes())
        digest.update(str(image.shape).encode())
        path = self.directory / f"{digest.hexdigest()}.json"
        if path.exists():
            return OcrResult.model_validate_json(path.read_text(encoding="utf-8"))
        result = self.engine.recognize(image)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp")
        temporary.write_text(result.model_dump_json(), encoding="utf-8")
        temporary.replace(path)
        return result
