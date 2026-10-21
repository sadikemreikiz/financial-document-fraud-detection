"""The ImageNet weights the encoder starts from (ADR-0022, decision 1).

The file is torchvision's ResNet-18 ``IMAGENET1K_V1``, the weights torchvision downloads for
``resnet18(weights="IMAGENET1K_V1")``. It is not stored in this repository: ``docfraud forensics
download-resnet`` fetches it, pinned by URL, size and SHA-256 (:data:`IMAGENET_WEIGHTS`), and
:func:`read_weights` checks it again before every use. ImageNet's terms allow non-commercial
research and education, and torchvision notes that pretrained weights may carry the terms of
their data, so the models trained from them are for non-commercial research.

This module imports PyTorch only to read the file, so that the command line can download it
without loading PyTorch.
"""

from pathlib import Path
from typing import TYPE_CHECKING

from docfraud.data.download import PinnedFile, verify

if TYPE_CHECKING:
    from torch import Tensor

IMAGENET_WEIGHTS = PinnedFile(
    name="resnet18-imagenet",
    url="https://download.pytorch.org/models/resnet18-f37072fd.pth",
    filename="resnet18-f37072fd.pth",
    size_bytes=46_830_571,
    sha256="f37072fd47e89c5e827621c5baffa7500819f7896bbacec160b1a16c560e07ec",
    homepage="https://github.com/pytorch/vision/tree/main/references/classification#resnet",
    citation=(
        "K. He, X. Zhang, S. Ren and J. Sun, Deep Residual Learning for Image Recognition, "
        "CVPR 2016, pp. 770-778; weights trained on ImageNet (O. Russakovsky et al., IJCV 2015) "
        "by torchvision"
    ),
)
"""torchvision's ResNet-18 weights; the first eight digits of the checksum name the file."""


def read_weights(path: Path, source: PinnedFile = IMAGENET_WEIGHTS) -> dict[str, Tensor]:
    """The tensors of a pinned weight file, by name, after checking its size and checksum.

    PyTorch's weights-only loader reads it, which accepts tensors and plain containers only, so
    that reading a weight file can never execute code.

    Raises:
        docfraud.data.download.ChecksumMismatchError: If the file is not the pinned one.
        pickle.UnpicklingError: If the file holds anything but tensors.
    """
    import torch  # noqa: PLC0415 - slow to import, needed only to read the file

    verify(source, path)
    state: dict[str, Tensor] = torch.load(path, map_location="cpu", weights_only=True)
    return state
