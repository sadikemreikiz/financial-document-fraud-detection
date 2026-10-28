from pathlib import Path

import numpy as np
from PIL import Image, PngImagePlugin

from docfraud.data.models import Document, Split
from docfraud.data.resave import digest, resave, resaved


def test_a_transparent_image_is_saved_on_white_paper_without_its_metadata(tmp_path: Path) -> None:
    image = Image.new("RGBA", (40, 30), (0, 0, 0, 0))
    image.paste((0, 0, 0, 255), (10, 10, 20, 20))
    source = tmp_path / "R1.png"
    info = PngImagePlugin.PngInfo()
    info.add_text("Software", "an editor")
    image.save(source, pnginfo=info)
    saved = resave(source, 95, tmp_path / "q95")
    assert saved == tmp_path / "q95" / "R1.jpg"
    with Image.open(saved) as jpeg:
        assert (jpeg.format, jpeg.mode) == ("JPEG", "RGB")
        assert not {"exif", "icc_profile", "comment", "Software"} & set(jpeg.info)
        pixels = np.asarray(jpeg.convert("L"))
    assert pixels[2, 2] > 245  # the transparent paper is white, not the black stored under it
    assert pixels[15, 15] < 10


def test_saving_again_gives_the_same_file_and_a_lower_quality_a_smaller_one(
    tmp_path: Path,
) -> None:
    rng = np.random.default_rng(0)
    source = tmp_path / "R1.png"
    Image.fromarray(rng.integers(0, 256, (60, 80), dtype=np.uint8)).save(source)
    first = resave(source, 95, tmp_path / "a").read_bytes()
    assert resave(source, 95, tmp_path / "b").read_bytes() == first
    assert len(resave(source, 75, tmp_path / "c").read_bytes()) < len(first)


def test_saved_receipts_keep_their_identity_and_the_digest_names_their_files(
    tmp_path: Path,
) -> None:
    source = tmp_path / "R1.png"
    Image.new("L", (40, 30), 200).save(source)
    receipt = Document(
        dataset="toy",
        doc_id="R1",
        source_id="R1",
        split=Split.TEST,
        image_path=source,
        transcript_path=tmp_path / "R1.txt",
    )
    (high,) = resaved([receipt], 95, tmp_path / "q95")
    (low,) = resaved([receipt], 75, tmp_path / "q75")
    assert (high.doc_id, high.split, high.transcript_path) == (
        "R1",
        Split.TEST,
        receipt.transcript_path,
    )
    assert high.image_path == tmp_path / "q95" / "R1.jpg"
    assert digest([high]) == digest(resaved([receipt], 95, tmp_path / "again"))
    assert digest([high]) != digest([low])
