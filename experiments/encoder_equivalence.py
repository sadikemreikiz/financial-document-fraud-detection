"""One-time check that the encoder of systems 2 to 4 is torchvision's ResNet-18 (ADR-0022).

The network is written in this project (``docfraud.forensics.learned.network``), so that
torchvision is not a dependency. This check loads the pinned ImageNet weights into torchvision's
own ResNet-18 and into the project's encoder and compares them:

1. the file: torchvision downloads the pinned URL for ``IMAGENET1K_V1``, its preprocessing uses
   the means and standard deviations the network normalises with, and both models count the
   parameters torchvision states;
2. the features after every stage of the encoder, in evaluation mode on the processor, for a
   batch of random ImageNet-normalised images and for a random grey page prepared as the network
   prepares every page, and the class scores of torchvision's own forward against the encoder
   followed by the classifier that the encoder leaves out;
3. the running statistics of batch normalisation after one forward pass in training mode.

It prints a table for the development record (``docs/learned-models-development.md``) and stops
unless the working tree is a clean commit, which the record names. torchvision is supplied by
``uv run --with`` for this check only.

Usage: ``make encoder-check`` (after ``docfraud forensics download-resnet``)
"""

from importlib.metadata import version
from pathlib import Path
from typing import Any

import torch
from torch import Tensor, nn
from torch.nn import functional
from torchvision.models import ResNet18_Weights, resnet18

from docfraud.evaluation.runs import code_version
from docfraud.forensics.learned.network import (
    CLASSIFIER,
    IMAGENET_MEAN,
    IMAGENET_STD,
    UNet,
    imagenet_input,
    load_encoder,
)
from docfraud.forensics.learned.pretrained import IMAGENET_WEIGHTS, read_weights
from docfraud.markdown import Cell, Table, render_table

ROOT = Path(__file__).resolve().parents[1]
WEIGHTS = ROOT / "artifacts" / "learned" / "pretrained" / IMAGENET_WEIGHTS.filename
SEED = 20261014
STAGES = ("1/2", "1/4", "1/8", "1/16", "1/32")


def torchvision_stages(model: Any, x: Tensor) -> tuple[Tensor, ...]:  # torchvision's ResNet
    """The features of torchvision's ResNet-18 after each stage, as its own forward runs them."""
    half = model.relu(model.bn1(model.conv1(x)))
    quarter = model.layer1(model.maxpool(half))
    eighth = model.layer2(quarter)
    sixteenth = model.layer3(eighth)
    return half, quarter, eighth, sixteenth, model.layer4(sixteenth)


def largest_difference(first: Tensor, second: Tensor) -> float:
    """The largest absolute difference between two tensors of the same shape."""
    if first.shape != second.shape:
        msg = f"shapes differ: {tuple(first.shape)} and {tuple(second.shape)}"
        raise ValueError(msg)
    return float((first - second).abs().max())


def shown(difference: float) -> str:
    """A difference as the record shows it: 0 when the values are identical."""
    return "0" if difference == 0 else f"{difference:.1e}"


def compare_features(reference: Any, encoder: nn.Module, x: Tensor) -> list[float]:
    """The largest difference at each stage between torchvision's model and the encoder."""
    with torch.no_grad():
        expected = torchvision_stages(reference, x)
        actual = encoder(x)
    return [largest_difference(e, a) for e, a in zip(expected, actual, strict=True)]


def main() -> None:
    """Run the comparisons and print the table for the record."""
    code = code_version(ROOT)
    if code.dirty is not False:
        msg = "commit all changes first: the record names the commit this check ran at"
        raise SystemExit(msg)
    weights = ResNet18_Weights.IMAGENET1K_V1
    preprocessing = weights.transforms()
    state = read_weights(WEIGHTS)

    reference = resnet18(weights=None)
    reference.load_state_dict(state, strict=True)
    network = UNet()
    load_encoder(network, state)
    encoder = network.encoder
    reference_parameters = sum(p.numel() for p in reference.parameters())
    classifier_parameters = sum(state[name].numel() for name in CLASSIFIER)
    encoder_parameters = sum(p.numel() for p in encoder.parameters())

    rows: list[list[Cell]] = [
        [
            "torchvision's URL for IMAGENET1K_V1 is the pinned one",
            weights.url == IMAGENET_WEIGHTS.url,
        ],
        [
            "torchvision's means and standard deviations are the network's",
            tuple(preprocessing.mean) == IMAGENET_MEAN and tuple(preprocessing.std) == IMAGENET_STD,
        ],
        ["Parameters stated by torchvision", weights.meta["num_params"]],
        ["Parameters of torchvision's model with the file loaded", reference_parameters],
        [
            "Parameters of the encoder plus the classifier left out",
            encoder_parameters + classifier_parameters,
        ],
    ]

    generator = torch.Generator().manual_seed(SEED)
    mean = torch.tensor(IMAGENET_MEAN).view(1, 3, 1, 1)
    std = torch.tensor(IMAGENET_STD).view(1, 3, 1, 1)
    images = (torch.rand(4, 3, 224, 224, generator=generator) - mean) / std
    page = torch.randint(0, 256, (1, 1, 435, 768), dtype=torch.uint8, generator=generator)
    grey_page = imagenet_input(page.float() / 255)
    reference.eval()
    encoder.eval()
    for name, x in (
        ("4 random images of 224 x 224 px", images),
        ("a grey page of 435 x 768 px", grey_page),
    ):
        for stage, difference in zip(STAGES, compare_features(reference, encoder, x), strict=True):
            rows.append([f"Largest difference at {stage}, {name}", shown(difference)])
    with torch.no_grad():
        pooled = functional.adaptive_avg_pool2d(encoder(images)[-1], 1).flatten(1)
        scores = functional.linear(pooled, state["fc.weight"], state["fc.bias"])
        rows.append(
            [
                "Largest difference of the 1,000 class scores, torchvision's own forward against "
                "the encoder followed by the classifier, 4 random images",
                shown(largest_difference(reference(images), scores)),
            ]
        )

    reference.train()
    encoder.train()
    with torch.no_grad():
        reference(images)
        encoder(images)
    statistics = [
        largest_difference(tensor, encoder.state_dict()[name])
        for name, tensor in reference.state_dict().items()
        if name.endswith(("running_mean", "running_var"))
    ]
    rows.append(
        ["Largest difference of running statistics after one training pass", shown(max(statistics))]
    )

    table = Table(
        title="The encoder against torchvision's ResNet-18",
        columns=["Check", "Result"],
        rows=rows,
        note=(
            f"Commit {code.commit}; PyTorch {torch.__version__}, torchvision "
            f"{version('torchvision')}; processor; weights {IMAGENET_WEIGHTS.filename} "
            f"(SHA-256 {IMAGENET_WEIGHTS.sha256}); random inputs with seed {SEED}."
        ),
    )
    print(render_table(table))  # noqa: T201 - the table is pasted into the record


if __name__ == "__main__":
    main()
