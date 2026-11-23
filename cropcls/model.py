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
                drop_rate: float = 0.0, timesteps: int = 1) -> nn.Module:
    """Create a timm ResNet with TorchGeo Sentinel-2 weights and a fresh head.

    If ``bands`` is a subset of the 13 pretraining bands (e.g. L2A data without B10,
    or RGB only), the pretrained first conv is sliced to just those input channels.
    With ``timesteps > 1`` the ResNet is shared across dates and a small temporal
    Transformer (see :class:`TemporalResNet`) sits on top.
    """
    if timesteps > 1:
        backbone = build_model(0, bands, arch, weights, 0.0)
        return TemporalResNet(backbone, num_classes, timesteps, len(bands), drop_rate)

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


class TemporalHead(nn.Module):
    """Date embeddings + 2-layer Transformer encoder + CLS token -> logits."""

    def __init__(self, dim: int, num_classes: int, timesteps: int, drop_rate: float,
                 d_model: int = 256, layers: int = 2, heads: int = 4) -> None:
        super().__init__()
        self.proj = nn.Linear(dim, d_model)
        self.pos = nn.Parameter(torch.zeros(1, timesteps + 1, d_model))
        self.cls = nn.Parameter(torch.zeros(1, 1, d_model))
        nn.init.trunc_normal_(self.pos, std=0.02)
        nn.init.trunc_normal_(self.cls, std=0.02)
        layer = nn.TransformerEncoderLayer(d_model, heads, 2 * d_model, dropout=0.1,
                                           batch_first=True, norm_first=True)
        self.encoder = nn.TransformerEncoder(layer, layers, enable_nested_tensor=False)
        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(drop_rate)
        self.fc = nn.Linear(d_model, num_classes)

    def forward(self, feats: torch.Tensor) -> torch.Tensor:  # (B, T, dim)
        x = self.proj(feats)
        x = torch.cat([self.cls.expand(x.shape[0], -1, -1), x], 1) + self.pos
        x = self.encoder(x)
        return self.fc(self.drop(self.norm(x[:, 0])))


class TemporalResNet(nn.Module):
    """Shared per-date ResNet encoder + temporal Transformer.

    Input is ``(B, T*C, H, W)`` (dates stacked time-major in the channel axis), which
    keeps augmentation identical across dates of a sample.
    """

    def __init__(self, backbone: nn.Module, num_classes: int, timesteps: int,
                 in_chans: int, drop_rate: float = 0.2) -> None:
        super().__init__()
        self.backbone = backbone
        self.timesteps = timesteps
        self.in_chans = in_chans
        self.head = TemporalHead(backbone.num_features, num_classes, timesteps, drop_rate)

    def get_classifier(self) -> nn.Module:
        return self.head

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        b, _, h, w = x.shape
        x = x.reshape(b * self.timesteps, self.in_chans, h, w)
        x = x.contiguous(memory_format=torch.channels_last)
        feats = self.backbone(x).reshape(b, self.timesteps, -1)
        return self.head(feats)
