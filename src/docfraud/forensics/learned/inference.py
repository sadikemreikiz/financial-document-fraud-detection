"""A trained network on whole receipts: maps and document scores (ADR-0022, decision 1).

A receipt is analysed in system 6's windows (:func:`docfraud.forensics.trufor.tiles`): in one
pass if it has at most 1448 x 1448 pixels, otherwise in windows of 1448 pixels that overlap by at
least 128 and start at multiples of 8, joined so that each pixel comes from the window in which
it lies farthest from an inner edge (:func:`docfraud.forensics.trufor.stitch`). Nothing is
resampled. The map is the probability that a pixel was edited, the sigmoid of the network's
logit.

The document score is chosen by a fixed order: first the highest value of the map; if its size
check fails, the highest value minus the receipt's own 99th percentile (:class:`LearnedOutput`).
Both are computed from the logits in double precision, so that probabilities near 1 stay
distinct up to logits of about 36, where single precision already rounds them to 1 at about 17.
"""

import hashlib
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import numpy.typing as npt
import torch
from torch import nn

from docfraud.data.models import BBox
from docfraud.forensics import trufor
from docfraud.forensics.learned.network import UNet
from docfraud.forensics.learned.training import default_device

_PAGE_DIMENSIONS = 2

HIGHEST = "highest value of the map"
CONTRAST = "highest value of the map minus the receipt's 99th percentile"
SCORE_ORDER = (HIGHEST, CONTRAST)
"""The document scores in the order of ADR-0022: the first whose size check passes is used."""


def probability(logits: npt.NDArray[np.floating]) -> npt.NDArray[np.float64]:
    """The probability of each logit, in double precision."""
    values = logits.astype(np.float64)
    return np.exp(-np.logaddexp(0.0, -values))


@dataclass(frozen=True, slots=True)
class LearnedOutput:
    """The network's outputs on one receipt."""

    logits: npt.NDArray[np.float32]  # (height, width)
    windows: tuple[BBox, ...]
    seconds: float  # time the analysis took on the machine that ran it

    @property
    def localisation(self) -> npt.NDArray[np.float32]:
        """The map: the probability that each pixel was edited."""
        return probability(self.logits).astype(np.float32)

    @property
    def probabilities(self) -> npt.NDArray[np.float64]:
        """The map in double precision, as it is measured, so that values near 1 stay distinct."""
        return probability(self.logits)

    def highest(self) -> float:
        """The first document score: the highest probability of the map."""
        return float(probability(np.array([self.logits.max()]))[0])

    def contrast(self) -> float:
        """The second document score: the highest probability minus the 99th percentile."""
        probabilities = self.probabilities
        return float(probabilities.max() - np.percentile(probabilities, 99))

    def scores(self) -> dict[str, float]:
        """Both document scores, by name, in the order of ``SCORE_ORDER``."""
        return {HIGHEST: self.highest(), CONTRAST: self.contrast()}


def load_model(path: Path) -> UNet:
    """A network with the weights a finished run saved, loaded tensor by tensor.

    Raises:
        RuntimeError: If a tensor is missing, unexpected or of another shape.
    """
    network = UNet()
    network.load_state_dict(torch.load(path, map_location="cpu", weights_only=True), strict=True)
    return network


class LearnedModel:
    """A trained network that analyses whole pages, on the Apple GPU when there is one.

    Args:
        network: The trained network.
        name: What the weights are, for example the run and the digest of its weights file.
        device: A PyTorch device; by default the Apple GPU if available, else the processor.
    """

    def __init__(self, network: nn.Module, name: str, device: torch.device | None = None) -> None:
        """Move the network to the device in evaluation mode."""
        self.device = device or default_device()
        self.network = network.to(self.device).eval()
        self.name = name

    @classmethod
    def from_file(cls, path: Path, device: torch.device | None = None) -> LearnedModel:
        """The network saved at ``path``, named after the first digits of the file's SHA-256."""
        digest = hashlib.sha256(path.read_bytes()).hexdigest()[:12]
        return cls(load_model(path), f"learned-{digest}", device)

    @property
    def identifier(self) -> str:
        """Weights, PyTorch version and device; changes whenever outputs may change."""
        return f"{self.name}-torch{torch.__version__}-{self.device.type}"

    def window_logits(self, window: npt.NDArray[np.uint8]) -> npt.NDArray[np.float32]:
        """The network's logit for every pixel of a ``(height, width)`` window."""
        with torch.no_grad():
            writable = np.array(window)  # a copy, since pages may be read-only
            pages = torch.from_numpy(writable)[None, None].to(self.device)
            logits = self.network(pages)[0, 0].float().cpu().numpy()
        return np.asarray(logits, dtype=np.float32)

    def analyse(self, page: npt.NDArray[np.uint8]) -> LearnedOutput:
        """Run the network on every window of the page and join the logits.

        Raises:
            ValueError: If the page is not a two-dimensional grayscale image.
        """
        if page.ndim != _PAGE_DIMENSIONS or page.dtype != np.uint8:
            msg = "a page is a two-dimensional uint8 grayscale image"
            raise ValueError(msg)
        start = time.perf_counter()
        height, width = page.shape
        windows = trufor.tiles(height, width)
        logits = [self.window_logits(page[w.y : w.y2, w.x : w.x2]) for w in windows]
        joined = logits[0] if len(windows) == 1 else trufor.stitch((height, width), windows, logits)
        if self.device.type == "mps":  # pragma: no cover - CI has no Apple GPU
            torch.mps.empty_cache()
        return LearnedOutput(joined, windows, time.perf_counter() - start)
