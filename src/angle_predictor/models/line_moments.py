from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def weighted_line_offset(
    weights: torch.Tensor, radius_squared: torch.Tensor, positive_offsets: torch.Tensor
) -> torch.Tensor:
    support = (weights.float() * radius_squared).sum(-2).squeeze(1)
    half = positive_offsets.numel()
    positive = support[:, half + 1 :]
    negative = support[:, :half].flip(-1)
    sine = ((positive - negative) * (2 * positive_offsets).sin()).sum(-1)
    cosine = ((positive + negative) * (2 * positive_offsets).cos()).sum(-1) + support[:, half]
    return 0.5 * torch.atan2(sine, cosine.clamp_min(1e-12))


class WeightedLineMoment(nn.Module):
    def __init__(
        self, channels: int, crop_size: tuple[int, int], radial_stride: int, window_deg: float
    ) -> None:
        super().__init__()
        if not 0 < window_deg < 45:
            raise ValueError("Line moments require a window between zero and 45 degrees")
        rows = math.ceil(crop_size[0] / radial_stride)
        radius = (
            2 * torch.arange(rows, dtype=torch.float32) * radial_stride / (crop_size[0] - 1) - 1
        )
        self.register_buffer("radius_squared", radius.square().view(1, 1, -1, 1))
        offsets = torch.linspace(0, math.radians(window_deg), crop_size[1] // 2 + 1)[1:]
        self.register_buffer("positive_offsets", offsets)
        self.projection = nn.Conv2d(channels, 1, 1)
        nn.init.zeros_(self.projection.weight)
        assert self.projection.bias is not None
        nn.init.zeros_(self.projection.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        with torch.amp.autocast(device_type=features.device.type, enabled=False):
            weights = F.softplus(self.projection(features.float()))
            radius_squared, offsets = self.radius_squared, self.positive_offsets
            assert isinstance(radius_squared, torch.Tensor) and isinstance(offsets, torch.Tensor)
            return weighted_line_offset(weights, radius_squared, offsets)
