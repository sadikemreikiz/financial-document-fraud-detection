"""Content normalisation that removes file-format traces without touching content.

Find-it-again shows that editing tools leave traces unrelated to the forgery itself: alpha
channels in re-saved PNGs and Windows line endings or trailing whitespace in edited transcripts
(ADR-0003). One forged transcript also kept the SROIE format, with box coordinates before every
line. Every model input passes through these functions, so such traces cannot become a signal.
Content is preserved: characters such as a wrong currency symbol are evidence and stay.
"""

import re
import unicodedata
from pathlib import Path

import numpy as np
import numpy.typing as npt
from PIL import Image

PAPER_WHITE = 255

# SROIE stores every text line after the eight corner coordinates of its box
# ("x1,y1,x2,y2,x3,y3,x4,y4,text"); Find-it-again transcripts are plain lines.
BOX_PREFIX = re.compile(r"^\d+(?:,\d+){7},")


def normalize_transcript(text: str) -> str:
    """Normalise line endings, trailing whitespace, box coordinates and Unicode composition.

    Args:
        text: Transcript as decoded from the file.

    Returns:
        The transcript with LF line endings, no trailing whitespace on any line, no blank lines
        at the end, exactly one final newline, no SROIE box coordinates and Unicode in NFC
        form. Empty input stays empty.

    Raises:
        ValueError: If only some lines carry box coordinates. A transcript is either in the
            SROIE format or plain; a mix is an unknown defect, and stripping the matching lines
            could remove content from a plain transcript.
    """
    text = unicodedata.normalize("NFC", text)
    lines = [line.rstrip() for line in text.replace("\r\n", "\n").replace("\r", "\n").split("\n")]
    lines = _strip_box_coordinates(lines)
    while lines and not lines[-1]:
        lines.pop()
    return "\n".join(lines) + "\n" if lines else ""


def _strip_box_coordinates(lines: list[str]) -> list[str]:
    filled = [line for line in lines if line]
    boxed = sum(BOX_PREFIX.match(line) is not None for line in filled)
    if not boxed:
        return lines
    if boxed < len(filled):
        msg = f"{boxed} of {len(filled)} lines carry SROIE box coordinates; expected all or none"
        raise ValueError(msg)
    return [BOX_PREFIX.sub("", line) for line in lines]


def read_transcript(path: Path) -> str:
    """Read a UTF-8 transcript strictly and normalise it.

    Raises:
        UnicodeDecodeError: If the file is not valid UTF-8. Silent replacement would hide data
            problems, so decoding errors are surfaced.
        ValueError: If only some lines carry box coordinates (see :func:`normalize_transcript`).
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
