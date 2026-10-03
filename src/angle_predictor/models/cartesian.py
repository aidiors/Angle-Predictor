from __future__ import annotations

import math
from typing import Any

import timm
import torch
from torch import nn
from torch.nn import functional as F

from angle_predictor.models.polar_refinement import RadialAntialias, rotate_double_angle


class CartesianConvNeXt(nn.Module):
    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        head: str = "stock",
        head_hidden: int = 256,
        spatial_bins: int = 4,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.backbone: Any = timm.create_model(backbone_name, pretrained=pretrained, num_classes=2)
        self.stock_head = head == "stock"
        if head == "spatial":
            channels = self.backbone.num_features
            self.backbone.reset_classifier(0)
            self.backbone.head = nn.Identity()
            self.spatial_head = nn.Sequential(
                nn.AdaptiveAvgPool2d((spatial_bins, spatial_bins)),
                nn.Flatten(),
                nn.LayerNorm(channels * spatial_bins**2),
                nn.Linear(channels * spatial_bins**2, head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, 2),
            )
        elif head != "stock":
            raise ValueError("Unknown Cartesian head")

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.backbone.forward_features(image)
        with torch.amp.autocast(device_type=image.device.type, enabled=False):
            vector = (
                self.backbone.forward_head(features.float())
                if self.stock_head
                else self.spatial_head(features.float())
            )
            return F.normalize(vector, dim=-1, eps=1e-6)


def crop_cartesian_strip(
    image: torch.Tensor,
    angle: torch.Tensor,
    crop_size: tuple[int, int],
    strip_width_px: float,
) -> torch.Tensor:
    height, width = image.shape[-2:]
    along = torch.linspace(-1, 1, crop_size[0], device=image.device)
    across = torch.linspace(-1, 1, crop_size[1], device=image.device)
    along_grid, across_grid = torch.meshgrid(along, across, indexing="ij")
    distance = along_grid * ((min(height, width) - 1) / 2)
    transverse = across_grid * (strip_width_px / 2)
    cosine = angle.float().cos()[:, None, None]
    sine = angle.float().sin()[:, None, None]
    x = (distance * cosine - transverse * sine) * (2 / (width - 1))
    y = (distance * sine + transverse * cosine) * (2 / (height - 1))
    grid = torch.stack((x, y), dim=-1)
    with torch.amp.autocast(device_type=image.device.type, enabled=False):
        return F.grid_sample(
            image.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
        )


class CartesianRefinement(nn.Module):
    def __init__(
        self,
        coarse: nn.Module,
        crop_size: tuple[int, int] = (384, 65),
        strip_width_px: float = 32.0,
        window_deg: float = 4.0,
        fine_channels: tuple[int, ...] = (16, 32, 128),
        refinement_hidden: int = 128,
        train_coarse: bool = False,
    ) -> None:
        super().__init__()
        self.coarse = coarse.requires_grad_(train_coarse)
        self.train_coarse = train_coarse
        self.crop_size = crop_size
        self.strip_width_px = strip_width_px
        self.window_deg = window_deg
        along = torch.linspace(-1, 1, crop_size[0])
        across = torch.linspace(-1, 1, crop_size[1])
        coordinates = torch.stack(torch.meshgrid(along, across, indexing="ij"), dim=0)
        self.register_buffer("coordinates", coordinates.unsqueeze(0))
        blocks: list[nn.Module] = []
        channels = 5
        for out_channels in fine_channels:
            blocks.extend(
                [
                    RadialAntialias(),
                    nn.Conv2d(channels, out_channels, (5, 23), (2, 1), (2, 11)),
                    nn.GroupNorm(math.gcd(4, out_channels), out_channels),
                    nn.SiLU(),
                ]
            )
            channels = out_channels
        self.fine = nn.Sequential(*blocks)
        output_layer = nn.Linear(refinement_hidden, 1)
        self.correction = nn.Sequential(
            nn.Linear(2 * channels * crop_size[1] + 2, refinement_hidden),
            nn.GELU(),
            output_layer,
        )
        nn.init.zeros_(output_layer.weight)
        nn.init.zeros_(output_layer.bias)
        self.train()

    def train(self, mode: bool = True) -> CartesianRefinement:
        super().train(mode)
        if not self.train_coarse:
            self.coarse.eval()
        return self

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_coarse):
            coarse = self.coarse(image).float()
            angle = (0.5 * torch.atan2(coarse[:, 0], coarse[:, 1])).remainder(math.pi)
        crop = crop_cartesian_strip(image, angle, self.crop_size, self.strip_width_px)
        coordinates = self.get_buffer("coordinates").expand(len(image), -1, -1, -1)
        features = self.fine(torch.cat((crop, coordinates), dim=1)).float()
        with torch.amp.autocast(device_type=image.device.type, enabled=False):
            profile = torch.cat((features.mean(-2), features.amax(-2)), dim=1).flatten(1)
            offset = self.correction(torch.cat((profile, coarse), dim=1)).squeeze(-1)
            offset = offset.tanh() * math.radians(self.window_deg)
            return rotate_double_angle(coarse, offset)
