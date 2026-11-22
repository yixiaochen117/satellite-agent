"""Backbones with Sentinel-2 pretrained weights from TorchGeo."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn

from .bands import S2_ALL_BANDS

ARCHS = ("resnet18", "resnet50")


def _weights_enum(arch: str):
    from torchgeo.models import ResNet18_Weights, ResNet50_Weights

    return {"resnet18": ResNet18_Weights, "resnet50": ResNet50_Weights}[arch]


def list_s2_weights(arch: str = "resnet18") -> list[str]:
    """Names of the 13-band Sentinel-2 weights available for ``arch``."""
    return [w.name for w in _weights_enum(arch)
            if w.name.startswith("SENTINEL2_ALL") and w.meta.get("in_chans") == 13]


def build_model(num_classes: int, bands: Sequence[str] = S2_ALL_BANDS,
                arch: str = "resnet18", weights: str | None = "SENTINEL2_ALL_MOCO",
                drop_rate: float = 0.0) -> nn.Module:
    """Create a timm ResNet with TorchGeo Sentinel-2 weights and a fresh head.

    If ``bands`` is a subset of the 13 pretraining bands (e.g. L2A data without B10,
    or RGB only), the pretrained first conv is sliced to just those input channels.
    """
    import timm
    from torchgeo.models import resnet18, resnet50

    if arch not in ARCHS:
        raise ValueError(f"arch must be one of {ARCHS}")
    bands = tuple(bands)
    fn = {"resnet18": resnet18, "resnet50": resnet50}[arch]

    if weights:
        w = getattr(_weights_enum(arch), weights)
        if w.meta.get("in_chans") != 13:
            raise ValueError(f"{weights} is not a 13-band Sentinel-2 weight; "
                             f"use one of {list_s2_weights(arch)}")
        model = fn(weights=w, drop_rate=drop_rate)
        if bands != S2_ALL_BANDS:
            idx = torch.tensor([S2_ALL_BANDS.index(b) for b in bands])
            old = model.conv1
            new = nn.Conv2d(len(bands), old.out_channels, old.kernel_size, old.stride,
                            old.padding, bias=old.bias is not None)
            with torch.no_grad():
                # Rescale so the summed activation keeps roughly the same magnitude.
                new.weight.copy_(old.weight[:, idx] * (13 / len(bands)))
                if old.bias is not None:
                    new.bias.copy_(old.bias)
            model.conv1 = new
    else:
        model = timm.create_model(arch, in_chans=len(bands), drop_rate=drop_rate)

    model.reset_classifier(num_classes)
    return model


def head_parameters(model: nn.Module) -> list[nn.Parameter]:
    return list(model.get_classifier().parameters())


def backbone_parameters(model: nn.Module) -> list[nn.Parameter]:
    head = {id(p) for p in head_parameters(model)}
    return [p for p in model.parameters() if id(p) not in head]
