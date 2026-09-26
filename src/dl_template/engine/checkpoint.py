"""Training checkpoint persistence."""

from __future__ import annotations

from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from torch import nn, optim


def save_checkpoint(
    model: nn.Module,
    path: Path,
    *,
    optimizer: optim.Optimizer | None = None,
    epoch: int | None = None,
    metadata: Mapping[str, Any] | None = None,
) -> None:
    """Save resumable state without coupling the checkpoint to MLflow."""
    path.parent.mkdir(parents=True, exist_ok=True)
    payload: dict[str, Any] = {
        "model_state_dict": model.state_dict(),
        "metadata": dict(metadata or {}),
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    torch.save(payload, path)
