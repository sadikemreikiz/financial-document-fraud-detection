import math

import pytest
import torch

from docfraud.forensics.learned.network import (
    CLASSIFIER,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ResNet18Encoder,
    UNet,
    imagenet_input,
    load_encoder,
)

RESNET18_PARAMETERS = 11_689_512
"""Parameters of torchvision's ResNet-18, its classifier included."""

CLASSIFIER_PARAMETERS = 512 * 1000 + 1000


def pages(*shape: int, seed: int = 0) -> torch.Tensor:
    generator = torch.Generator().manual_seed(seed)
    return torch.randint(0, 256, shape, dtype=torch.uint8, generator=generator)


def classifier_state(seed: int = 0) -> dict[str, torch.Tensor]:
    """Encoder weights drawn at random, with a classifier as torchvision's file has one."""
    torch.manual_seed(seed)
    state = dict(ResNet18Encoder().state_dict())
    state["fc.weight"] = torch.zeros(1000, 512)
    state["fc.bias"] = torch.zeros(1000)
    return state


def count(module: torch.nn.Module) -> int:
    return sum(parameter.numel() for parameter in module.parameters())


def test_encoder_has_resnet18_parameters_without_the_classifier() -> None:
    assert count(ResNet18Encoder()) == RESNET18_PARAMETERS - CLASSIFIER_PARAMETERS


def test_encoder_uses_torchvision_names_and_shapes() -> None:
    state = ResNet18Encoder().state_dict()
    assert state["conv1.weight"].shape == (64, 3, 7, 7)
    assert state["layer2.0.downsample.0.weight"].shape == (128, 64, 1, 1)
    assert state["layer2.0.downsample.1.running_mean"].shape == (128,)
    assert state["layer4.1.bn2.running_var"].shape == (512,)
    assert "layer1.0.downsample.0.weight" not in state  # same channels, same resolution
    assert sum(name.endswith("num_batches_tracked") for name in state) == 20


def test_network_has_the_size_of_the_timed_model() -> None:
    assert round(count(UNet()) / 1e6, 1) == 13.6


@pytest.mark.parametrize(("height", "width"), [(64, 64), (37, 53), (1, 1)])
def test_maps_have_the_size_of_the_page(height: int, width: int) -> None:
    network = UNet().eval()
    with torch.no_grad():
        logits = network(pages(2, 1, height, width))
    assert logits.shape == (2, 1, height, width)
    assert logits.dtype == torch.float32
    assert torch.isfinite(logits).all()


@pytest.mark.parametrize(
    "batch",
    [
        pages(1, 1, 32, 32).float(),  # not uint8
        pages(1, 3, 32, 32),  # three channels
        pages(1, 32, 32),  # no channel axis
    ],
)
def test_only_uint8_grey_pages_are_accepted(batch: torch.Tensor) -> None:
    with pytest.raises(ValueError, match="uint8"):
        UNet()(batch)


def test_grey_values_are_normalised_as_imagenet_images() -> None:
    grey = torch.full((1, 1, 2, 2), 0.5)
    normalised = imagenet_input(grey)
    assert normalised.shape == (1, 3, 2, 2)
    for channel, (mean, std) in enumerate(zip(IMAGENET_MEAN, IMAGENET_STD, strict=True)):
        assert torch.allclose(normalised[0, channel], torch.tensor((0.5 - mean) / std))


def test_the_same_page_gives_the_same_map() -> None:
    network = UNet().eval()
    page = pages(1, 1, 48, 40)
    with torch.no_grad():
        assert torch.equal(network(page), network(page))


def test_every_parameter_takes_part_in_the_map() -> None:
    network = UNet().train()
    network(pages(2, 1, 64, 64)).sum().backward()
    unused = [name for name, p in network.named_parameters() if p.grad is None or not p.grad.any()]
    assert unused == []


def initialised(seed: int, prior: float = 0.1) -> UNet:
    torch.manual_seed(1234)  # the global generator, which initialise must not depend on
    network = UNet()
    network.initialise(torch.Generator().manual_seed(seed), prior)
    return network


def decoder_state(network: UNet) -> dict[str, torch.Tensor]:
    return {k: v for k, v in network.state_dict().items() if not k.startswith("encoder.")}


def test_a_seed_fixes_the_initial_decoder() -> None:
    first, second = initialised(seed=7), initialised(seed=7)
    torch.manual_seed(99)
    third = UNet()
    third.initialise(torch.Generator().manual_seed(7), 0.1)
    for name, tensor in decoder_state(first).items():
        assert torch.equal(tensor, decoder_state(second)[name]), name
        assert torch.equal(tensor, decoder_state(third)[name]), name


def test_another_seed_draws_another_decoder() -> None:
    first, second = initialised(seed=7), initialised(seed=8)
    assert not torch.equal(first.head.weight, second.head.weight)
    name = "decoder.0.0.weight"
    assert not torch.equal(first.state_dict()[name], second.state_dict()[name])


def test_initialise_leaves_the_encoder_alone() -> None:
    torch.manual_seed(5)
    network = UNet()
    before = {k: v.clone() for k, v in network.encoder.state_dict().items()}
    network.initialise(torch.Generator().manual_seed(0), 0.1)
    for name, tensor in network.encoder.state_dict().items():
        assert torch.equal(tensor, before[name]), name


def test_initial_weights_follow_pytorch_default_bounds() -> None:
    head = initialised(seed=3).head
    bound = 1 / math.sqrt(head.weight[0].numel())  # kaiming_uniform_ with a = sqrt(5)
    assert head.weight.abs().max() <= bound


def test_output_bias_is_the_logit_of_the_prior() -> None:
    network = initialised(seed=0, prior=0.02)
    assert network.head.bias is not None
    assert network.head.bias.item() == pytest.approx(math.log(0.02 / 0.98))
    with torch.no_grad():
        network.eval()
        assert torch.sigmoid(network(pages(1, 1, 32, 32))).mean() < 0.5


@pytest.mark.parametrize("prior", [0.0, 1.0, -0.1, 1.5])
def test_prior_must_lie_strictly_between_0_and_1(prior: float) -> None:
    with pytest.raises(ValueError, match="strictly between"):
        UNet().initialise(torch.Generator(), prior)


def test_encoder_loads_torchvision_weights_and_skips_the_classifier() -> None:
    state = classifier_state(seed=1)
    network = UNet()
    load_encoder(network, state)
    for name, tensor in network.encoder.state_dict().items():
        assert torch.equal(tensor, state[name]), name
    assert not any(name in network.state_dict() for name in CLASSIFIER)


def test_weights_without_batch_counters_still_load() -> None:
    state = {k: v for k, v in classifier_state().items() if "num_batches_tracked" not in k}
    load_encoder(UNet(), state)


def test_a_missing_tensor_is_refused() -> None:
    state = classifier_state()
    del state["layer3.1.conv2.weight"]
    with pytest.raises(RuntimeError, match=r"layer3\.1\.conv2\.weight"):
        load_encoder(UNet(), state)


def test_an_unexpected_tensor_is_refused() -> None:
    state = classifier_state()
    state["layer5.0.conv1.weight"] = torch.zeros(1)
    with pytest.raises(RuntimeError, match=r"layer5\.0\.conv1\.weight"):
        load_encoder(UNet(), state)


def test_a_tensor_of_another_shape_is_refused() -> None:
    state = classifier_state()
    state["conv1.weight"] = torch.zeros(64, 1, 7, 7)
    with pytest.raises(RuntimeError, match="size mismatch"):
        load_encoder(UNet(), state)
