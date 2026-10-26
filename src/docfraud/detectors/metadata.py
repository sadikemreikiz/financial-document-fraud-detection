"""The file-metadata check of track T3: traits an editing program leaves in the file (ADR-0029, 13).

It reads the PNG header only, never the pixels, and has no training. A file counts as edited if
it carries any of five traits: an alpha channel, an ICC colour profile, EXIF data, XMP data or a
creation date. The list was fixed from the headers of the training split, where 88 of the 94
forged receipts and 1 of the 468 authentic ones carry at least one trait
(reports/fusion/preregistration-data.md). Re-saving a file removes the traits,
so the check is reported only in track T3, apart from the content-only headline (ADR-0003).
"""

from collections.abc import Callable
from pathlib import Path

from PIL import Image

TRAITS: dict[str, Callable[[Image.Image], bool]] = {
    "alpha channel": lambda im: im.mode in {"LA", "RGBA", "PA"} or "transparency" in im.info,
    "ICC colour profile": lambda im: "icc_profile" in im.info,
    "EXIF": lambda im: "exif" in im.info or "Raw profile type exif" in im.info,
    "XMP": lambda im: any(
        k in im.info for k in ("xmp", "XML:com.adobe.xmp", "Raw profile type xmp")
    ),
    "creation date": lambda im: "date:create" in im.info,
}
"""The editing traits, by name, as Pillow exposes them for a PNG file."""


def image_traits(image: Image.Image) -> tuple[str, ...]:
    """The editing traits an opened image carries, in the order of :data:`TRAITS`."""
    return tuple(name for name, test in TRAITS.items() if test(image))


def editing_traits(path: Path) -> tuple[str, ...]:
    """The editing traits of an image file; its pixels are not decoded."""
    with Image.open(path) as image:
        return image_traits(image)
