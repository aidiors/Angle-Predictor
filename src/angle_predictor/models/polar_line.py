from __future__ import annotations

import math

import torch
from torch import nn
from torch.nn import functional as F


def signed_polar_to_circle(image: torch.Tensor) -> torch.Tensor:
    """Reindex [B,C,2R,W] into [B,C,R,2W], with positive radius and 2π angles.

    No resampling: the negative-radius half supplies angles in [π,2π).
    Even radial size is required so both halves use the same radius samples.
    """
    if image.ndim != 4 or image.shape[-2] < 2 or image.shape[-2] % 2:
        raise ValueError("Expected [B,C,H,W] signed-polar input with even H >= 2")
    negative, positive = image.chunk(2, dim=-2)
    return torch.cat((positive, negative.flip(-2)), dim=-1)


class CircularRadialConv(nn.Module):
    """Circular angle padding; zero radial padding; stride only along radius."""

    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        *,
        stride: int = 1,
        groups: int = 1,
        radial_kernel: int = 3,
    ) -> None:
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            (radial_kernel, 3),
            stride=(stride, 1),
            padding=((radial_kernel - 1) // 2, 0),
            groups=groups,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(image, (1, 1, 0, 0), mode="circular"))


class RadialBlock(nn.Module):
    def __init__(self, channels: int) -> None:
        super().__init__()
        self.depthwise = CircularRadialConv(channels, channels, groups=channels)
        # Normalize channels at each location, without coupling angle positions.
        self.norm = nn.LayerNorm(channels)
        self.expand = nn.Conv2d(channels, 4 * channels, 1)
        self.contract = nn.Conv2d(4 * channels, channels, 1)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        hidden = self.depthwise(image)
        hidden = self.norm(hidden.permute(0, 2, 3, 1)).permute(0, 3, 1, 2)
        return image + self.contract(F.gelu(self.expand(hidden)))


class CircularLineHead(nn.Module):
    """Predict circular column scores and a bounded offset within each bin.

    Absolute angle enters only at decoding, keeping scores shift equivariant.
    The output order is [sin(2θ), cos(2θ)], as in the existing loss/inference.
    """

    def __init__(self, channels: int, hidden: int, normalize_output: bool = True) -> None:
        super().__init__()
        self.normalize_output = normalize_output
        self.radius_query = nn.Conv2d(channels, 1, 1)
        self.angle_conv = nn.Sequential(
            nn.Conv1d(3 * channels, hidden, 5, padding=2, padding_mode="circular"),
            nn.GELU(),
            nn.Conv1d(hidden, hidden, 5, padding=2, padding_mode="circular"),
            nn.GELU(),
        )
        self.score = nn.Conv1d(hidden, 1, 1)
        self.offset = nn.Conv1d(hidden, 1, 1)
        nn.init.zeros_(self.offset.weight)
        assert self.offset.bias is not None
        nn.init.zeros_(self.offset.bias)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        # Work in FP32 for attention, offsets and the circular moment.
        weights = self.radius_query(features).softmax(dim=-2)
        attended = (weights * features).sum(dim=-2)
        hidden = self.angle_conv(
            torch.cat((features.mean(dim=-2), features.amax(dim=-2), attended), dim=1)
        )
        logits = self.score(hidden).squeeze(1)
        width = logits.shape[-1]
        theta = torch.arange(width, device=hidden.device, dtype=torch.float32) * (math.pi / width)
        theta = theta + self.offset(hidden).squeeze(1).tanh() * (math.pi / (2 * width))
        probability = logits.softmax(dim=-1)
        output = torch.stack(
            (
                (probability * (2 * theta).sin()).sum(dim=-1),
                (probability * (2 * theta).cos()).sum(dim=-1),
            ),
            dim=-1,
        )
        return F.normalize(output, dim=-1, eps=1e-6) if self.normalize_output else output


class PolarLineModel(nn.Module):
    """Full-resolution angle estimator with learned radial feature aggregation."""

    def __init__(
        self,
        channels: tuple[int, ...] = (16, 32, 64),
        blocks_per_stage: int = 2,
        head_hidden: int = 128,
        normalize_output: bool = True,
    ) -> None:
        super().__init__()
        if not channels or any(channel <= 0 for channel in channels) or blocks_per_stage <= 0:
            raise ValueError("channels and blocks_per_stage must be positive")
        stages: list[nn.Module] = []
        previous = 3
        for index, channel in enumerate(channels):
            stages.extend(
                (
                    CircularRadialConv(
                        previous,
                        channel,
                        stride=4 if index == 0 else 2,
                        radial_kernel=7 if index == 0 else 3,
                    ),
                    nn.GELU(),
                    *(RadialBlock(channel) for _ in range(blocks_per_stage)),
                )
            )
            previous = channel
        self.backbone = nn.Sequential(*stages)
        self.head = CircularLineHead(channels[-1], head_hidden, normalize_output)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone(signed_polar_to_circle(image)).float()
        # Both half-rays represent the same undirected line. Average after
        # shared local feature extraction, so π rotations leave scores unchanged.
        first, opposite = features.chunk(2, dim=-1)
        features = (first + opposite) * 0.5
        if image.device.type in {"cuda", "cpu"}:
            with torch.amp.autocast(device_type=image.device.type, enabled=False):
                return self.head(features)
        return self.head(features)
