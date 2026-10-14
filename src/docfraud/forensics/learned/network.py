"""The network of systems 2 to 4: a U-Net with a ResNet-18 encoder (ADR-0022, decision 1).

The encoder has the layout and the parameter names of torchvision's ResNet-18, without its
classifier, so that torchvision's ImageNet weights load into it tensor by tensor
(:func:`load_encoder`); torchvision itself is not a dependency. The decoder returns to the input's
resolution: at each scale it upsamples bilinearly, joins the encoder's features of that scale and
applies a 3 x 3 convolution, batch normalisation and ReLU. A last stage joins channels computed
from the input at full resolution, which keep traces of one or two pixels that the encoder's first
stride would lose, before a 1 x 1 convolution gives one logit per pixel.

The network takes grayscale pages as they are stored, ``uint8`` from 0 to 255, and scales and
normalises them itself, so that training and inference cannot prepare them differently.
"""

import math
from collections.abc import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional

IMAGENET_MEAN = (0.485, 0.456, 0.406)
"""Per-channel means of the ImageNet images the encoder's weights were trained on."""

IMAGENET_STD = (0.229, 0.224, 0.225)
"""Per-channel standard deviations of the same images."""

ENCODER_CHANNELS = (64, 64, 128, 256, 512)
"""Channels of the encoder's features at 1/2, 1/4, 1/8, 1/16 and 1/32 of the resolution."""

DECODER_CHANNELS = (256, 128, 64, 64)
"""Channels of the decoder at 1/16, 1/8, 1/4 and 1/2 of the resolution."""

FULL_RESOLUTION_CHANNELS = 16
"""Channels computed from the input at full resolution."""

LAST_CHANNELS = 32
"""Channels of the last stage, at full resolution, before the logit."""

CLASSIFIER = frozenset({"fc.weight", "fc.bias"})
"""Tensors of torchvision's ResNet-18 that classify ImageNet images; the encoder has none."""

_PAGE_DIMENSIONS = 4


def _conv_block(inputs: int, outputs: int) -> nn.Sequential:
    """A 3 x 3 convolution, batch normalisation and ReLU."""
    return nn.Sequential(
        nn.Conv2d(inputs, outputs, 3, padding=1, bias=False),
        nn.BatchNorm2d(outputs),
        nn.ReLU(inplace=True),
    )


class BasicBlock(nn.Module):
    """A residual block of ResNet-18, with torchvision's names for its parts."""

    def __init__(self, inputs: int, outputs: int, stride: int) -> None:
        """From ``inputs`` to ``outputs`` channels, dividing the resolution by ``stride``."""
        super().__init__()
        self.conv1 = nn.Conv2d(inputs, outputs, 3, stride, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(outputs)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(outputs, outputs, 3, 1, 1, bias=False)
        self.bn2 = nn.BatchNorm2d(outputs)
        self.downsample: nn.Sequential | None = None
        if stride != 1 or inputs != outputs:
            self.downsample = nn.Sequential(
                nn.Conv2d(inputs, outputs, 1, stride, bias=False), nn.BatchNorm2d(outputs)
            )

    def forward(self, x: Tensor) -> Tensor:
        """The block's output: two convolutions added to the input, or to its projection."""
        identity = x if self.downsample is None else self.downsample(x)
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        result: Tensor = self.relu(out + identity)
        return result


def _layer(inputs: int, outputs: int, stride: int) -> nn.Sequential:
    """Two residual blocks, the first changing channels and resolution."""
    return nn.Sequential(BasicBlock(inputs, outputs, stride), BasicBlock(outputs, outputs, 1))


class ResNet18Encoder(nn.Module):
    """ResNet-18 without its classifier, returning its features at every scale."""

    def __init__(self) -> None:
        """An encoder with PyTorch's default initial weights, to be replaced by ImageNet's."""
        super().__init__()
        self.conv1 = nn.Conv2d(3, 64, 7, 2, 3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(3, 2, 1)
        self.layer1 = _layer(64, 64, 1)
        self.layer2 = _layer(64, 128, 2)
        self.layer3 = _layer(128, 256, 2)
        self.layer4 = _layer(256, 512, 2)

    def forward(self, x: Tensor) -> tuple[Tensor, Tensor, Tensor, Tensor, Tensor]:
        """Features at 1/2, 1/4, 1/8, 1/16 and 1/32 of the resolution of ``x``."""
        half: Tensor = self.relu(self.bn1(self.conv1(x)))
        quarter: Tensor = self.layer1(self.maxpool(half))
        eighth: Tensor = self.layer2(quarter)
        sixteenth: Tensor = self.layer3(eighth)
        thirty_second: Tensor = self.layer4(sixteenth)
        return half, quarter, eighth, sixteenth, thirty_second


def imagenet_input(grey: Tensor) -> Tensor:
    """Grey values from 0 to 1, copied into three channels and normalised as ImageNet's images.

    Each colour weight of the encoder's first layer then acts on the grey value, and no weight
    needs to change.
    """
    mean = torch.tensor(IMAGENET_MEAN, dtype=grey.dtype, device=grey.device).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD, dtype=grey.dtype, device=grey.device).view(1, 3, 1, 1)
    return (grey.expand(-1, 3, -1, -1) - mean) / std


def _join(coarse: Tensor, skip: Tensor) -> Tensor:
    """``coarse`` upsampled bilinearly to the size of ``skip``, followed by ``skip``'s channels."""
    upsampled = functional.interpolate(
        coarse, size=skip.shape[-2:], mode="bilinear", align_corners=False
    )
    return torch.cat([upsampled, skip], dim=1)


class UNet(nn.Module):
    """The network of systems 2 to 4: one logit per pixel that the pixel was edited."""

    def __init__(self) -> None:
        """A network with PyTorch's default initial weights; see :meth:`initialise`."""
        super().__init__()
        self.encoder = ResNet18Encoder()
        skips = ENCODER_CHANNELS[-2::-1]  # 1/16, 1/8, 1/4, 1/2
        coarser = (ENCODER_CHANNELS[-1], *DECODER_CHANNELS[:-1])
        self.decoder = nn.ModuleList(
            _conv_block(below + skip, outputs)
            for below, skip, outputs in zip(coarser, skips, DECODER_CHANNELS, strict=True)
        )
        self.full_resolution = _conv_block(1, FULL_RESOLUTION_CHANNELS)
        self.last = _conv_block(DECODER_CHANNELS[-1] + FULL_RESOLUTION_CHANNELS, LAST_CHANNELS)
        self.head = nn.Conv2d(LAST_CHANNELS, 1, 1)

    def forward(self, pages: Tensor) -> Tensor:
        """Logits of shape ``(n, 1, height, width)`` for ``uint8`` pages of the same shape.

        Raises:
            ValueError: If the pages are not a batch of one-channel ``uint8`` images.
        """
        if pages.dtype != torch.uint8 or pages.ndim != _PAGE_DIMENSIONS or pages.shape[1] != 1:
            msg = f"pages are uint8 of shape (n, 1, height, width), not {pages.dtype} {pages.shape}"
            raise ValueError(msg)
        grey = pages.float() / 255
        *skips, features = self.encoder(imagenet_input(grey))
        for block, skip in zip(self.decoder, reversed(skips), strict=True):
            features = block(_join(features, skip))
        features = self.last(_join(features, self.full_resolution(grey)))
        logits: Tensor = self.head(features)
        return logits

    def initialise(self, generator: torch.Generator, prior: float) -> None:
        """Draw every weight outside the encoder from ``generator`` and set the output bias.

        The weights are drawn as PyTorch draws them by default, but from the run's own generator,
        so that a seed fixes them whatever else has used PyTorch's global one. The output bias is
        the logit of ``prior``, the share of forged pixels expected in the training tiles, so that
        the network starts at that share instead of spending its first steps pushing every output
        down to it (Lin et al., 2017).

        Raises:
            ValueError: If ``prior`` is not strictly between 0 and 1.
        """
        if not 0 < prior < 1:
            msg = f"the share of forged pixels lies strictly between 0 and 1, not {prior}"
            raise ValueError(msg)
        with torch.no_grad():
            for part in (self.decoder, self.full_resolution, self.last, self.head):
                for module in part.modules():
                    if isinstance(module, nn.Conv2d):
                        _draw_convolution(module, generator)
                    elif isinstance(module, nn.BatchNorm2d):
                        module.reset_parameters()
            bias = self.head.bias
            if bias is None:  # pragma: no cover - the head is built with a bias
                msg = "the output layer has no bias"
                raise TypeError(msg)
            bias.fill_(math.log(prior / (1 - prior)))


def _draw_convolution(convolution: nn.Conv2d, generator: torch.Generator) -> None:
    """PyTorch's default initialisation of a convolution, drawn from ``generator``."""
    nn.init.kaiming_uniform_(convolution.weight, a=math.sqrt(5), generator=generator)
    if convolution.bias is not None:
        bound = 1 / math.sqrt(convolution.weight[0].numel())
        nn.init.uniform_(convolution.bias, -bound, bound, generator=generator)


def load_encoder(network: UNet, state: Mapping[str, Tensor]) -> None:
    """Load the weights of torchvision's ResNet-18 into the encoder, every tensor by name and shape.

    The classifier's tensors (:data:`CLASSIFIER`) are the only ones left out. Batch counters that
    checkpoints older than PyTorch 0.4.1 lack are added by PyTorch with the value 0; they only
    matter to batch normalisation without a momentum, which ResNet-18 does not use.

    Raises:
        RuntimeError: If a tensor is missing, unexpected or of another shape.
    """
    network.encoder.load_state_dict(
        {name: tensor for name, tensor in state.items() if name not in CLASSIFIER}, strict=True
    )
