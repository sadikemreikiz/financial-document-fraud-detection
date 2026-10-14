"""System 6 of ADR-0019: TruFor, a pretrained forensic model, at the receipt's native resolution.

TruFor (Guillaro et al., CVPR 2023) runs unchanged, with its published weights. Neither its code
nor its weights are part of this repository, whose MIT licence cannot cover them: ``docfraud
forensics download-trufor`` fetches both, pinned to a commit and to checksums
(:data:`TRUFOR_CODE`, :data:`TRUFOR_WEIGHTS`), and :class:`TruForModel` loads them from there.
Every report with TruFor's results credits its authors (:data:`CITATION`).

A receipt of at most ``TILE * TILE`` pixels is analysed in one pass. A larger one is analysed in
tiles at its native resolution (:func:`tiles`) that start at multiples of the JPEG block size, so
that every tile sees the receipt's block grid where the whole receipt has it; each pixel of the
maps comes from the tile in which it lies farthest from a tile edge inside the page
(:func:`stitch`). Nothing is resampled. TruFor's integrity score pools statistics over every
pixel of its maps (:func:`pooled_statistics`); for a tiled receipt they are pooled over the
stitched maps, so that the score describes the whole receipt as one pass would.

Only :class:`TruForModel` needs PyTorch, which it imports when it is built; tiling, stitching
and caching run without it. TruFor's other libraries are not project dependencies (ADR-0019).
The choices are recorded in ``docs/system-6-development.md``.
"""

import hashlib
import importlib
import math
import sys
import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Protocol

import numpy as np
import numpy.typing as npt

from docfraud.data.download import PinnedFile
from docfraud.data.models import BBox

CITATION = (
    "F. Guillaro, D. Cozzolino, A. Sud, N. Dufour and L. Verdoliva, TruFor: Leveraging All-Round "
    "Clues for Trustworthy Image Forgery Detection and Localization, CVPR 2023, pp. 20606-20615"
)

TRUFOR_COMMIT = "ae54475df6f41a491d7615100feb19263dec13f7"
"""The commit of TruFor's repository whose code is run."""

TRUFOR_CODE = PinnedFile(
    name="trufor-code",
    url=f"https://github.com/grip-unina/TruFor/archive/{TRUFOR_COMMIT}.zip",
    filename=f"TruFor-{TRUFOR_COMMIT}.zip",
    size_bytes=124_143_535,
    sha256="0ddef05c821d594bad690a42323abb59b4cd8f64708d7a16c24cd7247d70ce8f",
    homepage="https://github.com/grip-unina/TruFor",
    citation=CITATION,
)

TRUFOR_WEIGHTS = PinnedFile(
    name="trufor-weights",
    url="https://www.grip.unina.it/download/prog/TruFor/TruFor_weights.zip",
    filename="TruFor_weights.zip",
    size_bytes=260_878_690,
    sha256="953f1f7eda0dd2c5ece322ae9c185ba1079c1265aa5fdf319ef5a20604d206d8",
    homepage="https://grip-unina.github.io/TruFor/",
    citation=CITATION,
)

SOURCE_DIRECTORY = Path(f"TruFor-{TRUFOR_COMMIT}") / "test_docker" / "src"
"""Where TruFor's inference code lies, relative to the directory the archives unpack into."""

WEIGHTS_FILE = Path("weights") / "trufor.pth.tar"
"""Where the weights lie, relative to the directory the archives unpack into."""

TILE = 1448
"""Side of a tile in pixels, a multiple of ``GRID``. A tile has 2.1 MP, a page size that took
7.3 GB of the reference machine's 16 GB in the smoke test of ADR-0019, where 4.2 MP ran out of
memory."""

OVERLAP = 128
"""Fewest pixels that neighbouring tiles share, a multiple of ``GRID``."""

GRID = 8
"""The JPEG block size: every tile starts at a multiple of it."""

_PAGE_DIMENSIONS = 2


@dataclass(frozen=True, slots=True)
class WindowOutput:
    """TruFor's outputs on one window, each map at the window's resolution."""

    localisation: npt.NDArray[np.float32]  # probability that a pixel was edited
    confidence: npt.NDArray[np.float32]  # logit of the reliability of the localisation
    difference: npt.NDArray[np.float32]  # logit of the localisation: edited minus pristine
    logit: float  # TruFor's detection logit on the window


class WindowModel(Protocol):
    """Anything that analyses one window of a page as TruFor does."""

    @property
    def identifier(self) -> str:
        """Model, version and settings; changes whenever outputs may change."""
        ...

    def predict(self, window: npt.NDArray[np.uint8]) -> WindowOutput:
        """The maps of a ``(height, width)`` window and its detection logit."""
        ...

    def detect(self, statistics: npt.NDArray[np.float64]) -> float:
        """The detection logit of TruFor's last layers for the statistics of a page."""
        ...


def axis_windows(length: int) -> tuple[tuple[int, int], ...]:
    """Start and size of the windows along one axis of ``length`` pixels.

    An axis no longer than ``TILE`` is one window. Otherwise windows of ``TILE`` pixels start
    every ``TILE - OVERLAP`` pixels, and the last one starts at the last multiple of ``GRID``
    that lets a window reach the end; it is up to ``GRID - 1`` pixels longer than ``TILE``.
    """
    if length < 1:
        msg = "an axis has at least one pixel"
        raise ValueError(msg)
    if length <= TILE:
        return ((0, length),)
    stride = TILE - OVERLAP
    last = (length - TILE) // GRID * GRID
    starts = [i * stride for i in range(math.ceil((length - TILE) / stride)) if i * stride < last]
    return (*((start, TILE) for start in starts), (last, length - last))


def tiles(height: int, width: int) -> tuple[BBox, ...]:
    """The windows a page is analysed in, row by row: the whole page if it is small enough.

    A page of at most ``TILE * TILE`` pixels is one window, whatever its shape; a larger one is
    tiled along each axis by :func:`axis_windows`.
    """
    if height * width <= TILE * TILE:
        return (BBox(x=0, y=0, width=width, height=height),)
    return tuple(
        BBox(x=x, y=y, width=w, height=h)
        for y, h in axis_windows(height)
        for x, w in axis_windows(width)
    )


def owners(windows: Sequence[tuple[int, int]], length: int) -> npt.NDArray[np.intp]:
    """For each position along an axis, the window it lies farthest inside of.

    The distance counts only window edges inside the page: beyond the page there is nothing a
    window could miss. Ties go to the earlier window.
    """
    positions = np.arange(length)
    owner = np.full(length, -1, dtype=np.intp)
    best = np.full(length, -1, dtype=np.intp)
    for index, (start, size) in enumerate(windows):
        end = start + size
        span = positions[start:end]
        before = span - start if start > 0 else np.full(span.shape, length)
        after = end - 1 - span if end < length else np.full(span.shape, length)
        distance = np.minimum(before, after)
        better = distance > best[start:end]
        owner[start:end][better] = index
        best[start:end][better] = distance[better]
    if (owner < 0).any():
        msg = "the windows do not cover the axis"
        raise ValueError(msg)
    return owner


def stitch(
    shape: tuple[int, int], windows: Sequence[BBox], maps: Sequence[npt.NDArray[np.float32]]
) -> npt.NDArray[np.float32]:
    """Join the maps of the windows of :func:`tiles` into one map of the page.

    Along each axis separately, every pixel takes the window it lies farthest inside of
    (:func:`owners`); together this is a tile in which it lies farthest from an edge.

    Raises:
        ValueError: If the windows are not a grid that covers the page, or a map does not have
            the size of its window.
    """
    height, width = shape
    rows = sorted({(w.y, w.height) for w in windows})
    columns = sorted({(w.x, w.width) for w in windows})
    if len(windows) != len(rows) * len(columns) or len(maps) != len(windows):
        msg = "one map for each window of a grid"
        raise ValueError(msg)
    row_owner, column_owner = owners(rows, height), owners(columns, width)
    stitched = np.empty(shape, dtype=np.float32)
    for window, values in zip(windows, maps, strict=True):
        if values.shape != (window.height, window.width):
            msg = f"a map of {values.shape} for a window of {(window.height, window.width)}"
            raise ValueError(msg)
        owned_rows = row_owner[window.y : window.y2] == rows.index((window.y, window.height))
        owned_columns = column_owner[window.x : window.x2] == columns.index(
            (window.x, window.width)
        )
        stitched[window.y : window.y2, window.x : window.x2][np.ix_(owned_rows, owned_columns)] = (
            values[np.ix_(owned_rows, owned_columns)]
        )
    return stitched


def sigmoid(logit: float) -> float:
    """The logistic function, in double precision so that scores near 1 stay distinct."""
    return float(1 / (1 + np.exp(-np.float64(logit))))


def _logsumexp(values: npt.NDArray[np.float64]) -> float:
    top = float(values.max())
    return top + float(np.log(np.exp(values - top).sum()))


def _weighted_statistics(
    values: npt.NDArray[np.float64], log_weights: npt.NDArray[np.float64]
) -> list[float]:
    """TruFor's ``weighted_statistics_pooling``: soft minimum, soft maximum, mean, mean square."""
    log_weights = log_weights - _logsumexp(log_weights)
    weights = np.exp(log_weights)
    return [
        -_logsumexp(log_weights - values),
        _logsumexp(log_weights + values),
        float((weights * values).sum()),
        float((weights * values * values).sum()),
    ]


def pooled_statistics(
    confidence: npt.NDArray[np.float32], difference: npt.NDArray[np.float32]
) -> npt.NDArray[np.float64]:
    """The eight statistics from which TruFor's detection layers compute its integrity score.

    As TruFor's ``confpool`` detection (``models/cmx/builder_np_conf.py``) computes them over
    every pixel of a page: the statistics of the confidence logits with equal weights, then those
    of the localisation logits weighted by the confidence. Computed in double precision.
    """
    if confidence.shape != difference.shape:
        msg = "the confidence and localisation maps differ in size"
        raise ValueError(msg)
    reliability = confidence.astype(np.float64).ravel()
    localisation = difference.astype(np.float64).ravel()
    log_confidence = -np.logaddexp(0.0, -reliability)  # log of the sigmoid
    return np.array(
        _weighted_statistics(reliability, np.zeros_like(reliability))
        + _weighted_statistics(localisation, log_confidence)
    )


@dataclass(frozen=True, slots=True)
class TruForOutput:
    """TruFor's outputs on one page."""

    localisation: npt.NDArray[np.float32]  # (height, width): probability that a pixel was edited
    windows: tuple[BBox, ...]
    logits: tuple[float, ...]  # TruFor's detection logit on each window
    pooled_logit: float  # the detection logit for the statistics of the whole page's maps
    seconds: float  # time the analysis took on the machine that ran it

    def receipt_logit(self) -> float:
        """TruFor's detection logit for the whole receipt.

        TruFor's own logit when the receipt was one window; otherwise the logit of its last
        layers for the statistics pooled over the stitched maps.
        """
        return self.logits[0] if len(self.windows) == 1 else self.pooled_logit

    def integrity_score(self) -> float:
        """TruFor's integrity score of the whole receipt: the sigmoid of :meth:`receipt_logit`."""
        return sigmoid(self.receipt_logit())

    def highest_tile_score(self) -> float:
        """The integrity score of the highest-scoring window, the first candidate tried."""
        return sigmoid(max(self.logits))


def analyse(page: npt.NDArray[np.uint8], model: WindowModel) -> TruForOutput:
    """Run the model on every window of the page and join the maps.

    Raises:
        ValueError: If the page is not a two-dimensional grayscale image.
    """
    if page.ndim != _PAGE_DIMENSIONS or page.dtype != np.uint8:
        msg = "a page is a two-dimensional uint8 grayscale image"
        raise ValueError(msg)
    start = time.perf_counter()
    windows = tiles(*page.shape)
    outputs = [model.predict(page[w.y : w.y2, w.x : w.x2]) for w in windows]
    height, width = page.shape

    def joined(name: str) -> npt.NDArray[np.float32]:
        maps = [getattr(output, name) for output in outputs]
        whole = maps[0] if len(windows) == 1 else stitch((height, width), windows, maps)
        if whole.shape != page.shape:
            msg = f"a map of {whole.shape} for a page of {page.shape}"
            raise ValueError(msg)
        return whole

    localisation = joined("localisation")
    pooled = model.detect(pooled_statistics(joined("confidence"), joined("difference")))
    return TruForOutput(
        localisation,
        windows,
        tuple(output.logit for output in outputs),
        pooled,
        time.perf_counter() - start,
    )


class CachedTruFor:
    """Wraps a model with an on-disk cache keyed by model, tiling and page content.

    A page scan takes minutes; the cache lets a report be regenerated without running the model
    again, and never returns the output of another page, model or tiling.
    """

    FORMAT = 2
    """Version of the stored outputs; the first stored no pooled logit."""

    def __init__(self, model: WindowModel, directory: Path) -> None:
        """Store outputs of ``model`` under ``directory``."""
        self.model = model
        self.directory = (
            directory / f"{model.identifier}-tile{TILE}-overlap{OVERLAP}-format{self.FORMAT}"
        )

    def analyse(self, page: npt.NDArray[np.uint8]) -> TruForOutput:
        """Return the cached output, running the model only on a cache miss."""
        digest = hashlib.sha256(np.ascontiguousarray(page).tobytes())
        digest.update(str(page.shape).encode())
        path = self.directory / f"{digest.hexdigest()}.npz"
        if path.exists():
            with np.load(path) as stored:
                return TruForOutput(
                    stored["localisation"],
                    tuple(BBox(x=x, y=y, width=w, height=h) for x, y, w, h in stored["windows"]),
                    tuple(float(v) for v in stored["logits"]),
                    float(stored["pooled_logit"]),
                    float(stored["seconds"]),
                )
        output = analyse(page, self.model)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_suffix(".tmp.npz")
        np.savez_compressed(
            temporary,
            localisation=output.localisation,
            windows=np.array([[w.x, w.y, w.width, w.height] for w in output.windows]),
            logits=np.array(output.logits, dtype=np.float64),
            pooled_logit=np.float64(output.pooled_logit),
            seconds=np.float64(output.seconds),
        )
        temporary.replace(path)
        return output


class TruForModel:  # pragma: no cover - needs TruFor's code and weights, which CI does not fetch
    """TruFor's network with its published weights, on the Apple GPU when there is one.

    Args:
        directory: Where ``docfraud forensics download-trufor`` unpacked the archives.
        device: A PyTorch device; by default the Apple GPU if available, else the CPU.
    """

    def __init__(self, directory: Path, device: str | None = None) -> None:
        """Load TruFor's code and weights; fails if either is missing."""
        import torch  # noqa: PLC0415 - slow to import, needed only once a model is built

        self._torch = torch
        source = (directory / SOURCE_DIRECTORY).resolve()
        if not source.is_dir():
            msg = f"no TruFor code under {directory}: run docfraud forensics download-trufor"
            raise FileNotFoundError(msg)
        config, network = _import_trufor(source)
        self.device = torch.device(
            device or ("mps" if torch.backends.mps.is_available() else "cpu")
        )
        model = network(cfg=config)
        model.load_state_dict(_load_weights(torch, directory / WEIGHTS_FILE)["state_dict"])
        self._model = model.to(self.device).eval()

    @property
    def identifier(self) -> str:
        """For example ``trufor-ae54475-953f1f7e-torch2.14.1-mps``."""
        return (
            f"trufor-{TRUFOR_COMMIT[:7]}-{TRUFOR_WEIGHTS.sha256[:8]}"
            f"-torch{self._torch.__version__}-{self.device.type}"
        )

    def predict(self, window: npt.NDArray[np.uint8]) -> WindowOutput:
        """TruFor's maps and detection logit, from the window on three channels divided by 256.

        The input is built as TruFor's own loader builds it (``test_docker/src/data_core.py``),
        and the map and logit are read as its test script reads them (``trufor_test.py``); the
        logits behind them are those its detection pools (``models/cmx/builder_np_conf.py``).
        """
        torch = self._torch
        rgb = np.repeat(window[np.newaxis], 3, axis=0).astype(np.float32) / 256
        with torch.no_grad():
            prediction, confidence, detection, _ = self._model(
                torch.from_numpy(rgb)[None].to(self.device)
            )
            heat = torch.softmax(prediction[0], dim=0)[1].cpu().numpy()
            difference = (prediction[0, 1] - prediction[0, 0]).cpu().numpy()
            reliability = confidence[0, 0].cpu().numpy()
            logit = float(detection.item())
        if self.device.type == "mps":
            torch.mps.empty_cache()
        return WindowOutput(
            np.asarray(heat, dtype=np.float32),
            np.asarray(reliability, dtype=np.float32),
            np.asarray(difference, dtype=np.float32),
            logit,
        )

    def detect(self, statistics: npt.NDArray[np.float64]) -> float:
        """TruFor's detection layers (``self.detection`` of its network) on eight statistics."""
        torch = self._torch
        features = torch.from_numpy(statistics.astype(np.float32))[None].to(self.device)
        with torch.no_grad():
            return float(self._model.detection(features).item())


def _import_trufor(source: Path) -> tuple[Any, Any]:  # pragma: no cover - needs TruFor's code
    """TruFor's configuration, merged with its inference settings, and its network class.

    TruFor's code imports its own top-level modules ``config`` and ``models``, so its directory
    goes on the module path; a module of the same name from elsewhere is refused.
    """
    for name in ("config", "models"):
        loaded = sys.modules.get(name)
        if loaded is not None and not Path(str(loaded.__file__)).resolve().is_relative_to(source):
            msg = f"a module {name!r} from {loaded.__file__} hides TruFor's own"
            raise ImportError(msg)
    if str(source) not in sys.path:
        sys.path.append(str(source))
    config_module = importlib.import_module("config")
    builder = importlib.import_module("models.cmx.builder_np_conf")
    for module in (config_module, builder):
        if not Path(str(module.__file__)).resolve().is_relative_to(source):
            msg = f"{module.__name__} was imported from {module.__file__}, not from {source}"
            raise ImportError(msg)
    config = config_module._C.clone()  # TruFor exposes its defaults under this name
    config.defrost()
    config.merge_from_file(str(source / "trufor.yaml"))
    config.freeze()
    return config, builder.myEncoderDecoder


def _load_weights(torch: Any, path: Path) -> Any:  # pragma: no cover - needs TruFor's weights
    """The checkpoint, loaded so that unpickling cannot execute code (ADR-0019, decision 4).

    Next to the weights the checkpoint stores numpy scalars, such as its best loss value, which
    are the only objects allowed besides tensors.
    """
    scalar = importlib.import_module("numpy._core.multiarray").scalar  # its numpy 2 location
    allowed = [
        (scalar, "numpy.core.multiarray.scalar"),
        np.dtype,
        type(np.dtype(np.float64)),
        type(np.dtype(np.int64)),
    ]
    with torch.serialization.safe_globals(allowed):
        return torch.load(path, map_location="cpu", weights_only=True)
