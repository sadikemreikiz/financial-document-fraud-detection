from pathlib import Path
from typing import Any

from PIL import Image
from PIL.PngImagePlugin import PngInfo

from docfraud.detectors.metadata import TRAITS, editing_traits


def save(path: Path, mode: str = "L", **options: Any) -> Path:
    colour = 255 if mode == "L" else (255,) * len(mode)
    Image.new(mode, (8, 6), colour).save(path, format="PNG", **options)
    return path


def test_a_scanned_receipt_carries_no_editing_trait(tmp_path: Path) -> None:
    assert editing_traits(save(tmp_path / "plain.png")) == ()
    assert editing_traits(save(tmp_path / "colour.png", "RGB")) == ()


def test_an_icc_profile_and_exif_data_are_traits(tmp_path: Path) -> None:
    assert editing_traits(save(tmp_path / "icc.png", icc_profile=b"profile")) == (
        "ICC colour profile",
    )
    exif = Image.Exif()
    exif[0x0131] = "editor"  # the software tag
    assert editing_traits(save(tmp_path / "exif.png", exif=exif)) == ("EXIF",)


def test_an_alpha_channel_xmp_and_a_creation_date_are_traits(tmp_path: Path) -> None:
    info = PngInfo()
    info.add_itxt("XML:com.adobe.xmp", "<x:xmpmeta/>")
    info.add_text("date:create", "2016-01-01T00:00:00")
    path = save(tmp_path / "edited.png", "RGBA", pnginfo=info)
    assert editing_traits(path) == ("alpha channel", "XMP", "creation date")
    assert list(TRAITS) == ["alpha channel", "ICC colour profile", "EXIF", "XMP", "creation date"]
