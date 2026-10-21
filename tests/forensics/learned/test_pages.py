import pickle
from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from docfraud.data.models import Document, Label, Split
from docfraud.data.normalize import load_grayscale
from docfraud.forensics.learned.pages import PageCache


def document(
    tmp_path: Path, doc_id: str, mode: str = "RGB", split: Split | None = Split.TRAIN
) -> Document:
    """A document whose image is random noise in the given mode."""
    rng = np.random.default_rng(len(doc_id))
    channels = {"L": (), "RGB": (3,), "RGBA": (4,)}[mode]
    pixels = rng.integers(0, 256, size=(23, 17, *channels), dtype=np.uint8)
    path = tmp_path / "images" / f"{doc_id}.png"
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(pixels, mode=mode).save(path)
    return Document(
        dataset="test",
        doc_id=doc_id,
        source_id=doc_id,
        split=split,
        label=Label.AUTHENTIC,
        image_path=path,
    )


@pytest.mark.parametrize("mode", ["L", "RGB", "RGBA"])
def test_a_stored_page_is_the_page_every_system_decodes(tmp_path: Path, mode: str) -> None:
    cache = PageCache(tmp_path / "pages")
    receipt = document(tmp_path, f"receipt-{mode}", mode)
    assert cache.build([receipt]) == 1
    page = cache.page(receipt.doc_id)
    assert isinstance(page, np.memmap)
    assert page.dtype == np.uint8
    assert np.array_equal(page, load_grayscale(receipt.image_path))


def test_a_stored_page_cannot_be_written(tmp_path: Path) -> None:
    cache = PageCache(tmp_path / "pages")
    receipt = document(tmp_path, "receipt")
    cache.build([receipt])
    page = cache.page(receipt.doc_id)
    with pytest.raises(ValueError, match="read-only"):
        page[0, 0] = 0


def test_pages_are_decoded_once(tmp_path: Path) -> None:
    cache = PageCache(tmp_path / "pages")
    first, second = document(tmp_path, "a"), document(tmp_path, "b")
    assert cache.build([first]) == 1
    progress: list[tuple[int, int]] = []
    assert cache.build([first, second], on_progress=lambda *step: progress.append(step)) == 1
    assert progress == [(1, 2), (2, 2)]
    assert sorted(p.name for p in cache.directory.iterdir()) == ["a.npy", "b.npy"]


@pytest.mark.parametrize("split", [Split.VAL, Split.TEST, None])
def test_only_training_pages_are_stored(tmp_path: Path, split: Split | None) -> None:
    cache = PageCache(tmp_path / "pages")
    receipts = [document(tmp_path, "train"), document(tmp_path, "other", split=split)]
    with pytest.raises(ValueError, match="training split"):
        cache.build(receipts)
    assert not cache.directory.exists()


def test_a_page_not_stored_is_an_error(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        PageCache(tmp_path).page("missing")


def test_a_file_that_is_not_a_page_is_refused(tmp_path: Path) -> None:
    cache = PageCache(tmp_path)
    cache.directory.mkdir(parents=True)
    np.save(cache.path("colour"), np.zeros((4, 4, 3), dtype=np.uint8))
    with pytest.raises(ValueError, match="two-dimensional uint8"):
        cache.page("colour")


@pytest.mark.parametrize("doc_id", ["", "../escape", "a/b", ".hidden"])
def test_identifiers_must_be_plain_file_names(tmp_path: Path, doc_id: str) -> None:
    with pytest.raises(ValueError, match="identifier"):
        PageCache(tmp_path).path(doc_id)


def test_the_cache_travels_to_other_processes_without_pages(tmp_path: Path) -> None:
    cache = PageCache(tmp_path / "pages")
    receipt = document(tmp_path, "receipt")
    cache.build([receipt])
    copy = pickle.loads(pickle.dumps(cache))  # noqa: S301 - our own object
    assert len(pickle.dumps(cache)) < 1000
    assert np.array_equal(copy.page(receipt.doc_id), cache.page(receipt.doc_id))
