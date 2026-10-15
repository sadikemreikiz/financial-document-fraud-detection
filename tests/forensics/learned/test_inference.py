import math
from pathlib import Path

import numpy as np
import pytest
import torch
from torch import Tensor, nn

from docfraud.forensics import trufor
from docfraud.forensics.learned.inference import (
    CONTRAST,
    HIGHEST,
    SCORE_ORDER,
    LearnedModel,
    LearnedOutput,
    load_model,
)
from docfraud.forensics.learned.network import UNet

CPU = torch.device("cpu")


class Pointwise(nn.Module):
    """A logit from each pixel alone, so that windows joined give the whole page's map."""

    def forward(self, pages: Tensor) -> Tensor:
        return (pages.float() - 128) / 16


class Neighbourly(nn.Module):
    """A logit from the pixels within ``radius`` of a pixel, padded with zeros at the edges."""

    def __init__(self, radius: int) -> None:
        super().__init__()
        self.pool = nn.AvgPool2d(2 * radius + 1, stride=1, padding=radius, count_include_pad=True)

    def forward(self, pages: Tensor) -> Tensor:
        logits: Tensor = self.pool(pages.float() / 64)
        return logits


def page(height: int, width: int, seed: int = 0) -> np.ndarray:
    return np.random.default_rng(seed).integers(0, 256, size=(height, width), dtype=np.uint8)


@pytest.fixture
def small_tiles(monkeypatch: pytest.MonkeyPatch) -> None:
    """Windows of 48 pixels overlapping by 16."""
    monkeypatch.setattr(trufor, "TILE", 48)
    monkeypatch.setattr(trufor, "OVERLAP", 16)


def sigmoid(values: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-values.astype(np.float64)))


def test_a_read_only_page_is_analysed_as_it_is() -> None:
    receipt = page(30, 20)
    receipt.setflags(write=False)  # as pages decoded from a file or mapped from the cache are
    output = LearnedModel(Pointwise(), "pointwise", CPU).analyse(receipt)
    np.testing.assert_array_equal(output.logits, (receipt.astype(np.float32) - 128) / 16)


def test_a_small_page_is_one_window() -> None:
    model = LearnedModel(Pointwise(), "pointwise", CPU)
    receipt = page(30, 20)
    output = model.analyse(receipt)
    assert output.windows == trufor.tiles(30, 20)
    assert len(output.windows) == 1
    assert output.logits.dtype == np.float32
    assert np.allclose(output.localisation, sigmoid((receipt.astype(np.float64) - 128) / 16))


@pytest.mark.usefixtures("small_tiles")
def test_windows_of_a_large_page_are_joined_into_one_map() -> None:
    receipt = page(130, 100)
    output = LearnedModel(Pointwise(), "pointwise", CPU).analyse(receipt)
    assert len(output.windows) > 1
    assert output.logits.shape == receipt.shape
    assert np.array_equal(output.logits, ((receipt.astype(np.float32) - 128) / 16))


@pytest.mark.usefixtures("small_tiles")
def test_each_pixel_comes_from_the_window_it_lies_farthest_inside() -> None:
    receipt = page(130, 100, seed=1)
    model = LearnedModel(Neighbourly(radius=10), "neighbourly", CPU)
    output = model.analyse(receipt)
    maps = [model.window_logits(receipt[w.y : w.y2, w.x : w.x2]) for w in output.windows]
    assert np.array_equal(output.logits, trufor.stitch(receipt.shape, output.windows, maps))
    whole = model.window_logits(receipt)
    assert not np.array_equal(output.logits, whole)  # the windows' edges reach some pixels
    # rows 50-59 and columns 45-59 lie more than 10 px inside the window that owns them
    assert np.array_equal(output.logits[50:60, 45:60], whole[50:60, 45:60])


@pytest.mark.usefixtures("small_tiles")
def test_a_view_narrower_than_half_the_overlap_hides_every_cut() -> None:
    receipt = page(130, 100, seed=2)
    model = LearnedModel(Neighbourly(radius=8), "neighbourly", CPU)
    assert np.array_equal(model.analyse(receipt).logits, model.window_logits(receipt))


@pytest.mark.parametrize("bad", [np.zeros((4, 4, 3), dtype=np.uint8), np.zeros((4, 4))])
def test_only_grey_pages_are_analysed(bad: np.ndarray) -> None:
    with pytest.raises(ValueError, match="grayscale"):
        LearnedModel(Pointwise(), "pointwise", CPU).analyse(bad)


def test_document_scores_come_from_the_logits() -> None:
    logits = np.zeros((40, 25), dtype=np.float32)  # the 99th percentile falls among the zeros
    logits[0, 0] = 30.0  # a probability that rounds to 1 in single precision
    output = LearnedOutput(logits, (), 0.0)
    assert output.localisation.max() == np.float32(1.0)
    assert output.highest() == pytest.approx(1 - math.exp(-30), abs=1e-16)
    assert output.highest() < 1.0
    assert output.contrast() == pytest.approx(output.highest() - 0.5)
    assert list(output.scores()) == list(SCORE_ORDER) == [HIGHEST, CONTRAST]


def test_saved_weights_load_into_the_network(tmp_path: Path) -> None:
    torch.manual_seed(3)
    network = UNet()
    path = tmp_path / "model.pt"
    torch.save(network.state_dict(), path)
    loaded = load_model(path)
    for name, tensor in network.state_dict().items():
        assert torch.equal(tensor, loaded.state_dict()[name]), name
    model = LearnedModel.from_file(path, CPU)
    assert model.name.startswith("learned-")
    assert model.identifier == f"{model.name}-torch{torch.__version__}-cpu"
    assert not model.network.training


def test_weights_of_another_network_are_refused(tmp_path: Path) -> None:
    path = tmp_path / "model.pt"
    torch.save({"conv.weight": torch.zeros(1)}, path)
    with pytest.raises(RuntimeError, match="Missing key"):
        load_model(path)
