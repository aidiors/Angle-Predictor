from __future__ import annotations

import math
from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F

from angle_predictor.models.line_moments import WeightedLineMoment


def crop_signed_polar(
    image: torch.Tensor,
    angle: torch.Tensor,
    *,
    crop_size: tuple[int, int],
    window_deg: float,
) -> torch.Tensor:
    """Sample around predicted angles, reflecting radius on each angular wrap."""
    height, width = crop_size
    offsets = torch.linspace(
        -math.radians(window_deg),
        math.radians(window_deg),
        width,
        device=image.device,
        dtype=torch.float32,
    )
    # Original column j represents j*pi/W, including column zero at theta=0.
    columns = image.shape[-1] + (angle[:, None].float() + offsets) * (image.shape[-1] / math.pi)
    extended = torch.cat((image.flip(-2), image, image.flip(-2)), dim=-1)
    x = (2 * columns / (extended.shape[-1] - 1) - 1)[:, None].expand(-1, height, -1)
    radius = torch.linspace(-1, 1, height, device=image.device, dtype=torch.float32)
    y = radius[None, :, None].expand(len(image), -1, width)
    grid = torch.stack((x, y), dim=-1)
    # Keep subpixel sampling in FP32 even when convolutional features use BF16.
    with torch.amp.autocast(device_type=image.device.type, enabled=False):
        return F.grid_sample(
            extended.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
        )


def rotate_double_angle(vector: torch.Tensor, offset: torch.Tensor) -> torch.Tensor:
    """Rotate [sin(2 theta), cos(2 theta)] by an offset expressed in radians."""
    sine, cosine = torch.sin(2 * offset), torch.cos(2 * offset)
    return torch.stack(
        (vector[:, 0] * cosine + vector[:, 1] * sine, vector[:, 1] * cosine - vector[:, 0] * sine),
        dim=-1,
    )


def crop_cartesian_polar(
    image: torch.Tensor,
    angle: torch.Tensor,
    *,
    crop_size: tuple[int, int],
    window_deg: float,
) -> torch.Tensor:
    height, width = image.shape[-2:]
    cx, cy = (width - 1) / 2, (height - 1) / 2
    radius = torch.linspace(
        -min(cx, cy), min(cx, cy), crop_size[0], device=image.device, dtype=torch.float32
    )
    offsets = torch.linspace(
        -math.radians(window_deg),
        math.radians(window_deg),
        crop_size[1],
        device=image.device,
        dtype=torch.float32,
    )
    theta = angle[:, None, None].float() + offsets[None, None, :]
    x = cx + radius[None, :, None] * theta.cos()
    y = cy + radius[None, :, None] * theta.sin()
    grid = torch.stack((2 * x / (width - 1) - 1, 2 * y / (height - 1) - 1), dim=-1)
    with torch.amp.autocast(device_type=image.device.type, enabled=False):
        return F.grid_sample(
            image.float(), grid, mode="bilinear", padding_mode="border", align_corners=True
        )


def radial_attention_profile(features: torch.Tensor, query: torch.Tensor) -> torch.Tensor:
    values = features.float()
    scores = torch.einsum("bcrw,c->brw", values, query.float()) / math.sqrt(values.shape[1])
    weights = scores.softmax(dim=-2).unsqueeze(1)
    return torch.cat(((values * weights).sum(-2), values.amax(-2)), dim=1)


def angular_heatmap_offset(scores: torch.Tensor, positive_offsets: torch.Tensor) -> torch.Tensor:
    """Decode symmetric odd-width crop scores to a continuous offset in radians."""
    probabilities = scores.float().softmax(-1)
    half = positive_offsets.numel()
    # Pair equal coordinates before summing so uniform scores yield exactly zero.
    imbalance = probabilities[..., half + 1 :] - probabilities[..., :half].flip(-1)
    return (imbalance * positive_offsets).sum(-1)


def radial_feature_profile(features: torch.Tensor, bins: int = 1) -> torch.Tensor:
    """Keep mean/max support within each radial bin, retaining all angular columns."""
    values = features.float()
    if bins == 1:
        # Preserve the original reduction operations for legacy checkpoints.
        return torch.cat((values.mean(-2), values.amax(-2)), dim=1)
    size = (bins, values.shape[-1])
    means = F.adaptive_avg_pool2d(values, size)
    maxima = F.adaptive_max_pool2d(values, size)
    return torch.cat((means, maxima), dim=1).flatten(1, 2)


def radial_top4_profile(features: torch.Tensor) -> torch.Tensor:
    values = features.float()
    if values.shape[-2] < 4:
        raise ValueError("Top-four pooling requires at least four radial rows")
    strongest = values.topk(4, dim=-2, sorted=False).values.mean(-2)
    return torch.cat((values.mean(-2), strongest), dim=1)


class RadialAntialias(nn.Module):
    """Apply a fixed unit-sum radial filter without mixing angular columns."""

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        padded = F.pad(features, (0, 0, 1, 1), mode="replicate")
        return (padded[..., :-2, :] + 2 * padded[..., 1:-1, :] + padded[..., 2:, :]) * 0.25


class PolarRefinementModel(nn.Module):
    """Coarse predictor and local fine branch, optionally trained jointly."""

    def __init__(
        self,
        coarse: nn.Module,
        crop_size: tuple[int, int] = (192, 33),
        window_deg: float = 3.0,
        fine_channels: tuple[int, ...] = (16, 32, 32),
        refinement_hidden: int = 128,
        train_coarse: bool = False,
        refinement_head: Literal["mlp", "local_mlp", "heatmap"] = "mlp",
        detach_crop_angle: bool = False,
        fine_angular_kernel: int = 5,
        radial_pool_bins: int = 1,
        fine_radial_antialias: bool = False,
        fine_radial_kernel: int = 5,
        fine_radial_strides: tuple[int, ...] | None = None,
        radial_pool_mode: Literal["mean_max", "attention_max", "mean_top4"] = "mean_max",
        fine_crop_source: Literal["polar", "cartesian"] = "polar",
        fine_detail_skip: bool = False,
        fine_radial_reflection: bool = False,
        fine_reflection_readout: Literal["profile", "offset"] = "profile",
        fine_geometric_residual: bool = False,
        fine_angular_antisymmetry: bool = False,
    ) -> None:
        super().__init__()
        if fine_angular_kernel <= 0 or fine_angular_kernel % 2 != 1:
            raise ValueError("fine_angular_kernel must be positive and odd")
        if fine_radial_kernel <= 0 or fine_radial_kernel % 2 != 1:
            raise ValueError("fine_radial_kernel must be positive and odd")
        strides = (2,) * len(fine_channels) if fine_radial_strides is None else fine_radial_strides
        if len(strides) != len(fine_channels) or any(stride not in (1, 2) for stride in strides):
            raise ValueError("fine_radial_strides must match fine_channels with values1 or2")
        radial_rows = math.ceil(crop_size[0] / math.prod(strides))
        if radial_pool_bins <= 0 or radial_pool_bins > radial_rows:
            raise ValueError("radial_pool_bins must fit the fine feature radial rows")
        self.radial_pool_bins = radial_pool_bins
        if radial_pool_mode not in ("mean_max", "attention_max", "mean_top4"):
            raise ValueError("Unknown radial pool mode")
        if radial_pool_mode == "attention_max" and radial_pool_bins != 1:
            raise ValueError("attention_max requires one radial pool bin")
        if radial_pool_mode == "mean_top4" and (radial_pool_bins != 1 or radial_rows < 4):
            raise ValueError("mean_top4 requires one radial bin and at least four feature rows")
        self.radial_pool_mode = radial_pool_mode
        if fine_crop_source not in ("polar", "cartesian"):
            raise ValueError("Unknown fine crop source")
        self.fine_crop_source = fine_crop_source
        if fine_radial_reflection and (
            radial_pool_bins != 1 or radial_pool_mode != "mean_max" or fine_detail_skip
        ):
            raise ValueError(
                "radial reflection requires global mean/max pooling without detail skip"
            )
        self.fine_radial_reflection = fine_radial_reflection
        if fine_reflection_readout not in ("profile", "offset"):
            raise ValueError("Unknown reflection readout")
        if fine_reflection_readout == "offset" and not fine_radial_reflection:
            raise ValueError("offset reflection readout requires radial reflection")
        self.fine_reflection_readout = fine_reflection_readout
        if fine_angular_antisymmetry and (
            refinement_head != "mlp"
            or radial_pool_bins != 1
            or radial_pool_mode != "mean_max"
            or fine_detail_skip
            or (fine_radial_reflection and fine_reflection_readout != "offset")
            or fine_geometric_residual
        ):
            raise ValueError("Angular antisymmetry requires plain global mean/max MLP")
        self.fine_angular_antisymmetry = fine_angular_antisymmetry
        if fine_geometric_residual and (
            refinement_head != "mlp"
            or radial_pool_bins != 1
            or radial_pool_mode != "mean_max"
            or fine_detail_skip
            or fine_radial_reflection
            or not 0 < window_deg < 45
        ):
            raise ValueError("Geometric residual requires plain global mean/max MLP and window <45")
        if fine_detail_skip and (
            len(fine_channels) < 2 or radial_pool_bins != 1 or radial_pool_mode != "mean_max"
        ):
            raise ValueError(
                "fine detail skip requires multiple layers and global mean/max pooling"
            )
        self.detail_layer = 3 if fine_radial_antialias else 2
        self.register_parameter(
            "radius_query",
            nn.Parameter(torch.zeros(fine_channels[-1]))
            if radial_pool_mode == "attention_max"
            else None,
        )
        self.train_coarse = train_coarse
        self.coarse = coarse.requires_grad_(train_coarse)
        if not train_coarse:
            self.coarse.eval()
        self.crop_size = crop_size
        self.window_deg = window_deg
        self.refinement_head = refinement_head
        self.detach_crop_angle = detach_crop_angle
        blocks: list[nn.Module] = []
        channels = 5  # RGB and signed/absolute radial coordinates.
        for out_channels, radial_stride in zip(fine_channels, strides, strict=True):
            if fine_radial_antialias:
                blocks.append(RadialAntialias())
            blocks.extend(
                [
                    nn.Conv2d(
                        channels,
                        out_channels,
                        (fine_radial_kernel, fine_angular_kernel),
                        stride=(radial_stride, 1),
                        padding=(fine_radial_kernel // 2, fine_angular_kernel // 2),
                    ),
                    nn.GroupNorm(math.gcd(4, out_channels), out_channels),
                    nn.SiLU(),
                ]
            )
            channels = out_channels
        self.fine = nn.Sequential(*blocks)
        if refinement_head in ("mlp", "local_mlp"):
            output_layer = nn.Linear(refinement_hidden, 1)
            conditioning_channels = 2 if refinement_head == "mlp" else 0
            self.correction = nn.Sequential(
                nn.Linear(
                    2 * channels * radial_pool_bins * crop_size[1] + conditioning_channels,
                    refinement_hidden,
                ),
                nn.GELU(),
                output_layer,
            )
        elif refinement_head == "heatmap":
            output_layer = nn.Conv1d(refinement_hidden, 1, 1)
            self.correction = nn.Sequential(
                nn.Conv1d(2 * channels * radial_pool_bins, refinement_hidden, 5, padding=2),
                nn.GELU(),
                output_layer,
            )
            offsets = torch.linspace(0, math.radians(window_deg), crop_size[1] // 2 + 1)[1:]
            self.register_buffer("positive_offsets", offsets)
        else:
            raise ValueError(f"Unknown refinement head: {refinement_head}")
        nn.init.zeros_(output_layer.weight)
        assert output_layer.bias is not None
        nn.init.zeros_(output_layer.bias)
        radius = torch.linspace(-1, 1, crop_size[0]).view(1, 1, -1, 1)
        self.register_buffer("radius", radius)
        self.detail_projection: nn.Conv1d | None = None
        if fine_detail_skip:
            with torch.random.fork_rng(devices=[]):
                self.detail_projection = nn.Conv1d(2 * fine_channels[0], 2 * channels, 1)
            nn.init.zeros_(self.detail_projection.weight)
            assert self.detail_projection.bias is not None
            nn.init.zeros_(self.detail_projection.bias)
        self.geometric_readout: WeightedLineMoment | None = None
        if fine_geometric_residual:
            with torch.random.fork_rng(devices=[]):
                self.geometric_readout = WeightedLineMoment(
                    channels, crop_size, math.prod(strides), window_deg
                )

    def train(self, mode: bool = True) -> PolarRefinementModel:
        super().train(mode)
        if not self.train_coarse:
            self.coarse.eval()
        return self

    def forward(self, image: torch.Tensor, cartesian: torch.Tensor | None = None) -> torch.Tensor:
        with torch.set_grad_enabled(torch.is_grad_enabled() and self.train_coarse):
            coarse = self.coarse(image).float()
            angle = (0.5 * torch.atan2(coarse[:, 0], coarse[:, 1])).remainder(math.pi)
        crop_angle = angle.detach() if self.detach_crop_angle else angle
        if self.fine_crop_source == "cartesian":
            if cartesian is None:
                raise ValueError("Direct fine sampling requires a Cartesian image")
            crop = crop_cartesian_polar(
                cartesian, crop_angle, crop_size=self.crop_size, window_deg=self.window_deg
            )
        else:
            crop = crop_signed_polar(
                image, crop_angle, crop_size=self.crop_size, window_deg=self.window_deg
            )
        offset = self.refine_crop(crop, coarse)
        return rotate_double_angle(coarse, offset)

    def refine_crop(self, crop: torch.Tensor, coarse: torch.Tensor) -> torch.Tensor:
        if self.fine_angular_antisymmetry:
            crop = torch.cat((crop, crop.flip(-1)), dim=0)
        if self.fine_radial_reflection:
            crop = torch.cat((crop, crop.flip(-2)), dim=0)
        radius_buffer = self.radius
        assert isinstance(radius_buffer, torch.Tensor)
        radius = radius_buffer.expand(len(crop), -1, -1, self.crop_size[1])
        features = torch.cat((crop, radius, radius.abs()), dim=1)
        detail = None
        if self.detail_projection is None:
            features = self.fine(features)
        else:
            for index, layer in enumerate(self.fine):
                features = layer(features)
                if index == self.detail_layer:
                    detail = features
        with torch.amp.autocast(device_type=crop.device.type, enabled=False):
            query = self.radius_query
            if isinstance(query, torch.Tensor):
                profile = radial_attention_profile(features, query)
            elif self.radial_pool_mode == "mean_top4":
                profile = radial_top4_profile(features)
            else:
                profile = radial_feature_profile(features, self.radial_pool_bins)
            if self.detail_projection is not None:
                assert detail is not None
                profile = profile + self.detail_projection(radial_feature_profile(detail))
            late_reflection = self.fine_reflection_readout == "offset"
            if self.fine_radial_reflection and not late_reflection:
                original, mirrored = profile.chunk(2, dim=0)
                profile = (original + mirrored) * 0.5
            if self.refinement_head == "heatmap":
                scores = self.correction(profile).squeeze(1)
                positive_offsets = self.positive_offsets
                assert isinstance(positive_offsets, torch.Tensor)
                offset = angular_heatmap_offset(scores, positive_offsets)
            else:
                hidden = profile.flatten(1)
                if self.refinement_head == "mlp":
                    views = len(hidden) // len(coarse)
                    conditioning = coarse.repeat(views, 1) if views > 1 else coarse
                    hidden = torch.cat((hidden, conditioning), dim=1)
                score = self.correction(hidden).squeeze(-1)
                if self.geometric_readout is not None:
                    geometry = self.geometric_readout(features) / math.radians(self.window_deg)
                    score = score + torch.atanh(geometry.clamp(-1 + 1e-6, 1 - 1e-6))
                offset = torch.tanh(score) * math.radians(self.window_deg)
            if late_reflection:
                original_offset, mirrored_offset = offset.chunk(2, dim=0)
                offset = (original_offset + mirrored_offset) * 0.5
            if self.fine_angular_antisymmetry:
                original_offset, mirrored_offset = offset.chunk(2, dim=0)
                offset = (original_offset - mirrored_offset) * 0.5
            return offset
