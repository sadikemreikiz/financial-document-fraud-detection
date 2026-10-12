from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from docfraud.data.normalize import (
    load_grayscale,
    normalize_transcript,
    read_transcript,
    to_grayscale,
)


@pytest.mark.parametrize(
    ("raw", "expected"),
    [
        ("A\r\nB\r\n", "A\nB\n"),
        ("A\rB", "A\nB\n"),
        ("A  \nB\t\n", "A\nB\n"),
        ("A\n\n\n", "A\n"),
        ("A", "A\n"),
        ("", ""),
        ("\n \n", ""),
        ("café", "café\n"),
        ("TOTAL 5.00 €", "TOTAL 5.00 €\n"),
        ("1,2,3,4,5,6,7,8,NO.2, JALAN\r\n\n1,2,3,4,5,6,7,8,RM 1,50\n", "NO.2, JALAN\n\nRM 1,50\n"),
        ("1,2,3,4,5,6,7,TOTAL\n12,50,RM\n", "1,2,3,4,5,6,7,TOTAL\n12,50,RM\n"),
    ],
    ids=[
        "crlf",
        "cr",
        "trailing",
        "blank-tail",
        "no-newline",
        "empty",
        "only-space",
        "nfc",
        "euro-kept",
        "box-coordinates",
        "numbers-kept",
    ],
)
def test_normalize_transcript(raw: str, expected: str) -> None:
    assert normalize_transcript(raw) == expected


def test_normalize_transcript_keeps_interior_blank_lines() -> None:
    assert normalize_transcript("A\n\nB") == "A\n\nB\n"


def test_normalize_transcript_refuses_partial_box_coordinates() -> None:
    with pytest.raises(ValueError, match="1 of 2 lines carry SROIE box coordinates"):
        normalize_transcript("1,2,3,4,5,6,7,8,TOTAL\nRM 1.50\n")


def test_read_transcript_is_strict(tmp_path: Path) -> None:
    path = tmp_path / "t.txt"
    path.write_bytes(b"TOTAL \xff")
    with pytest.raises(UnicodeDecodeError):
        read_transcript(path)
    path.write_bytes(b"TOTAL 1\r\n")
    assert read_transcript(path) == "TOTAL 1\n"


def test_transparent_pixels_become_white_not_black() -> None:
    rgba = np.zeros((3, 4, 4), dtype=np.uint8)  # black, fully transparent
    rgba[1:, :, 3] = 255  # rows 1-2 opaque black
    gray = to_grayscale(Image.fromarray(rgba, mode="RGBA"))
    assert gray.dtype == np.uint8
    assert gray.shape == (3, 4)
    assert (gray[0] == 255).all()
    assert (gray[1:] == 0).all()


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA", "LA"])
def test_gray_content_is_identical_across_storage_modes(mode: str) -> None:
    values = np.array([[0, 64], [128, 255]], dtype=np.uint8)
    reference = Image.fromarray(values, mode="L")
    stored = reference.convert(mode)
    assert np.array_equal(to_grayscale(stored), values)


def test_palette_with_transparency_is_composited() -> None:
    image = Image.new("P", (2, 1))
    image.putpalette([0, 0, 0] * 256)
    image.info["transparency"] = 0
    assert (to_grayscale(image) == 255).all()


def test_load_grayscale_reads_files(tmp_path: Path) -> None:
    path = tmp_path / "i.png"
    Image.new("RGBA", (5, 2), (10, 10, 10, 255)).save(path)
    assert load_grayscale(path).shape == (2, 5)
