from __future__ import annotations

import math
from collections.abc import Callable
from functools import partial
from typing import Literal

import torch
from torch.nn import functional as F

AngleLossKind = Literal["mse", "smooth_l1", "vector_charbonnier", "angular_mae", "angular_huber"]
AngleLossFn = Callable[[torch.Tensor, torch.Tensor], torch.Tensor]
ANGLE_LOSS_KINDS: tuple[AngleLossKind, ...] = (
    "mse",
    "smooth_l1",
    "vector_charbonnier",
    "angular_mae",
    "angular_huber",
)


def _float_vectors(pred: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    if pred.ndim != 2 or pred.shape[-1] != 2 or pred.shape != target.shape:
        raise ValueError("Expected matching pred and target tensors [B,2]")
    if pred.shape[0] == 0:
        raise ValueError("Angle loss requires a nonempty batch")
    return pred.float(), target.float()


def mse_angle_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Mean squared error on raw double-angle vectors."""
    p, t = _float_vectors(pred, target)
    return F.mse_loss(p, t)


def smooth_l1_angle_loss(
    pred: torch.Tensor, target: torch.Tensor, *, beta: float = 0.1
) -> torch.Tensor:
    """Component-wise SmoothL1 on raw double-angle vectors."""
    p, t = _float_vectors(pred, target)
    return F.smooth_l1_loss(p, t, beta=beta)


def vector_charbonnier_angle_loss(
    pred: torch.Tensor, target: torch.Tensor, *, epsilon: float = 0.1
) -> torch.Tensor:
    """Smooth robust loss on the length of the two-component error vector."""
    p, t = _float_vectors(pred, target)
    squared_error = (p - t).square().sum(dim=-1)
    return (torch.sqrt(squared_error + epsilon * epsilon) - epsilon).mean()


def angular_mae_loss(pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    """Geodesic MAE; zero predictions are a degenerate optimum."""
    p, t = _float_vectors(pred, target)
    p = F.normalize(p, p=2, dim=-1, eps=1e-6)
    t = F.normalize(t, p=2, dim=-1, eps=1e-6)
    sin_delta = p[:, 0] * t[:, 1] - p[:, 1] * t[:, 0]
    cos_delta = (p * t).sum(dim=1).clamp(-1.0, 1.0)
    delta_2theta = torch.atan2(sin_delta, cos_delta).abs()
    return (0.5 * delta_2theta).mean()


def angular_huber_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    *,
    beta_rad: float = math.radians(0.1),
) -> torch.Tensor:
    """Apply SmoothL1 to the shortest signed error between undirected lines."""
    p, t = _float_vectors(pred, target)
    if not math.isfinite(beta_rad) or beta_rad <= 0:
        raise ValueError("beta_rad must be finite and positive")

    t = F.normalize(t, p=2, dim=-1, eps=1e-6)
    degenerate = p.norm(dim=-1) < 1e-6
    safe_p = torch.where(degenerate.unsqueeze(-1), t, p)
    p_hat = F.normalize(safe_p, p=2, dim=-1, eps=1e-6)
    sin_delta = p_hat[:, 0] * t[:, 1] - p_hat[:, 1] * t[:, 0]
    cos_delta = (p_hat * t).sum(dim=-1).clamp(-1.0, 1.0)
    delta = 0.5 * torch.atan2(sin_delta, cos_delta)
    angular_loss = F.smooth_l1_loss(delta, torch.zeros_like(delta), beta=beta_rad, reduction="none")
    degenerate_loss = (1.0 - (p * t).sum(dim=-1)).clamp_min(0.0)
    return torch.where(degenerate, degenerate_loss, angular_loss).mean()


def build_angle_loss(
    kind: AngleLossKind = "vector_charbonnier",
    *,
    smooth_l1_beta: float = 0.1,
    charbonnier_epsilon: float = 0.1,
    angular_huber_beta_deg: float = 0.1,
) -> AngleLossFn:
    """Select the loss once before training."""
    if kind not in ANGLE_LOSS_KINDS:
        raise ValueError(f"Unknown angle loss {kind!r}; choose from {ANGLE_LOSS_KINDS}")
    if not math.isfinite(smooth_l1_beta) or smooth_l1_beta <= 0:
        raise ValueError("smooth_l1_beta must be finite and positive")
    if not math.isfinite(charbonnier_epsilon) or charbonnier_epsilon <= 0:
        raise ValueError("charbonnier_epsilon must be finite and positive")
    if not math.isfinite(angular_huber_beta_deg) or angular_huber_beta_deg <= 0:
        raise ValueError("angular_huber_beta_deg must be finite and positive")

    if kind == "mse":
        return mse_angle_loss
    if kind == "smooth_l1":
        return partial(smooth_l1_angle_loss, beta=smooth_l1_beta)
    if kind == "vector_charbonnier":
        return partial(vector_charbonnier_angle_loss, epsilon=charbonnier_epsilon)
    if kind == "angular_huber":
        return partial(angular_huber_loss, beta_rad=math.radians(angular_huber_beta_deg))
    return angular_mae_loss
