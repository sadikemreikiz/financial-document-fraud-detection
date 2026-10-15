"""The training pages, decoded once and read through memory maps (ADR-0022, decision 4).

Training draws tiles from receipts drawn at random at every step, in several processes. Decoding
a page scan again for each tile would be slow, and each process holding its own copy of the
3.4 GiB of training pages would not fit the reference machine's memory. Instead every page is
decoded once, by :func:`~docfraud.data.normalize.load_grayscale` as every system decodes it, into
an uncompressed ``.npy`` file; :meth:`PageCache.page` maps the file into memory, so that the
operating system reads only the rows a tile needs and the processes share them.

Only pages of the training split are stored, so that no page of the test split is ever copied.
"""

from collections.abc import Callable, Iterable
from pathlib import Path

import numpy as np
import numpy.typing as npt

from docfraud.data.models import Document, Split
from docfraud.data.normalize import load_grayscale

FORMAT = 1
"""Version of the stored pages, part of the directory name."""

_PAGE_DIMENSIONS = 2

type Progress = Callable[[int, int], None]


class PageCache:
    """Normalised grayscale pages of the training split in uncompressed files.

    The cache holds no open file, so that it can be handed to other processes as it is.
    """

    def __init__(self, directory: Path) -> None:
        """Store pages under ``directory``."""
        self.directory = directory / f"format{FORMAT}"

    def path(self, doc_id: str) -> Path:
        """Where the page of a document is stored.

        Raises:
            ValueError: If the identifier is not a plain file name.
        """
        if not doc_id or Path(doc_id).name != doc_id or doc_id.startswith("."):
            msg = f"not a document identifier: {doc_id!r}"
            raise ValueError(msg)
        return self.directory / f"{doc_id}.npy"

    def build(self, documents: Iterable[Document], on_progress: Progress | None = None) -> int:
        """Decode and store every page that is not stored yet.

        A page is written to a temporary file that is renamed when complete, so that an
        interrupted build never leaves a page that looks complete.

        Args:
            documents: Documents of the training split.
            on_progress: Called with ``(documents_done, documents_total)`` after each document.

        Returns:
            How many pages were decoded.

        Raises:
            ValueError: If a document is not of the training split.
        """
        documents = tuple(documents)
        outside = [d.doc_id for d in documents if d.split is not Split.TRAIN]
        if outside:
            msg = f"only pages of the training split are stored, not {', '.join(outside[:3])}"
            raise ValueError(msg)
        self.directory.mkdir(parents=True, exist_ok=True)
        decoded = 0
        for done, document in enumerate(documents, start=1):
            path = self.path(document.doc_id)
            if not path.exists():
                temporary = path.with_suffix(".tmp.npy")
                np.save(temporary, load_grayscale(document.image_path), allow_pickle=False)
                temporary.replace(path)
                decoded += 1
            if on_progress is not None:
                on_progress(done, len(documents))
        return decoded

    def page(self, doc_id: str) -> npt.NDArray[np.uint8]:
        """The stored page of a document, mapped read-only into memory.

        Raises:
            FileNotFoundError: If the page has not been stored.
            ValueError: If the file is not a grayscale page.
        """
        page: npt.NDArray[np.uint8] = np.load(self.path(doc_id), mmap_mode="r", allow_pickle=False)
        if page.dtype != np.uint8 or page.ndim != _PAGE_DIMENSIONS:
            msg = f"{doc_id}: a stored page is two-dimensional uint8, not {page.dtype} {page.shape}"
            raise ValueError(msg)
        return page
