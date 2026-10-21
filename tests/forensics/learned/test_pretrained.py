import hashlib
import pickle
from pathlib import Path

import pytest
import torch

from docfraud.data.download import ChecksumMismatchError, PinnedFile
from docfraud.forensics.learned.pretrained import IMAGENET_WEIGHTS, read_weights


def pinned(path: Path) -> PinnedFile:
    """A pin of the file as it is now."""
    return PinnedFile(
        name="test-weights",
        url=f"https://example.org/{path.name}",
        filename=path.name,
        size_bytes=path.stat().st_size,
        sha256=hashlib.sha256(path.read_bytes()).hexdigest(),
        homepage="https://example.org",
        citation="none",
    )


def saved(tmp_path: Path, content: object) -> Path:
    path = tmp_path / "weights.pth"
    torch.save(content, path)
    return path


def test_the_pin_names_torchvisions_file() -> None:
    assert IMAGENET_WEIGHTS.url.endswith(f"/{IMAGENET_WEIGHTS.filename}")
    assert IMAGENET_WEIGHTS.filename == f"resnet18-{IMAGENET_WEIGHTS.sha256[:8]}.pth"


def test_a_pinned_file_gives_its_tensors(tmp_path: Path) -> None:
    tensors = {"conv1.weight": torch.arange(6.0).view(2, 3), "fc.bias": torch.zeros(4)}
    path = saved(tmp_path, tensors)
    state = read_weights(path, pinned(path))
    assert state.keys() == tensors.keys()
    for name, tensor in tensors.items():
        assert torch.equal(state[name], tensor)
        assert state[name].device.type == "cpu"


def test_a_changed_file_is_refused(tmp_path: Path) -> None:
    path = saved(tmp_path, {"weight": torch.zeros(3)})
    pin = pinned(path)
    content = bytearray(path.read_bytes())
    content[len(content) // 2] ^= 0xFF
    path.write_bytes(bytes(content))
    with pytest.raises(ChecksumMismatchError):
        read_weights(path, pin)


class Payload:
    """An object that is not a tensor or a plain container."""


def test_a_file_with_other_objects_is_refused(tmp_path: Path) -> None:
    path = saved(tmp_path, {"weight": torch.zeros(3), "extra": Payload()})
    with pytest.raises(pickle.UnpicklingError):
        read_weights(path, pinned(path))
