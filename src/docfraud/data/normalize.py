"""Content normalisation that removes file-format traces without touching content.

Find-it-again shows that editing tools leave traces unrelated to the forgery itself: alpha
channels in re-saved PNGs and Windows line endings or trailing whitespace in edited transcripts
(ADR-0003). Every model input passes through these functions, so such traces cannot become a
signal. Content is preserved: characters such as a wrong currency symbol are evidence and stay.
"""

import unicodedata
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

PAPER_WHITE = 255


def normalize_transcript(text: str) -> str:
    """Normalise line endings, trailing whitespace and Unicode composition.

    Args:
        text: Transcript as decoded from the file.

    Returns:
        The transcript with LF line endings, no trailing whitespace on any line, no blank lines
        at the end, exactly one final newline and Unicode in NFC form. Empty input stays empty.
    """
    text = unicodedata.normalize("NFC", text)
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""


def read_transcript(path: Path) -> str:
    """Read a UTF-8 transcript strictly and normalise it.

    Raises:
        UnicodeDecodeError: If the file is not valid UTF-8. Silent replacement would hide data
            problems, so decoding errors are surfaced.
    """
    return normalize_transcript(path.read_text(encoding="utf-8", errors="strict"))


def to_grayscale(image: Image.Image) -> npt.NDArray[np.uint8]:
    """Render an image as a person would see it on paper and convert it to 8-bit grayscale.

    Transparent pixels are composited onto white paper instead of being dropped: dropping the
    alpha channel exposes whatever colour the editing tool stored underneath (often black), which
    would create an artificial trace. Grayscale conversion uses the ITU-R 601-2 luma transform.

    Args:
        image: A decoded image in any Pillow mode.

    Returns:
        A ``(height, width)`` array of dtype ``uint8``.
    """
    if image.mode in {"RGBA", "LA", "PA"} or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        paper = Image.new("RGBA", rgba.size, (PAPER_WHITE, PAPER_WHITE, PAPER_WHITE, 255))
        image = Image.alpha_composite(paper, rgba)
    return np.asarray(image.convert("L"), dtype=np.uint8)


def load_grayscale(path: Path) -> npt.NDArray[np.uint8]:
    """Load an image file and normalise it with :func:`to_grayscale`."""
    with Image.open(path) as image:
        return to_grayscale(image)
