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
    scheduler: optim.lr_scheduler.LRScheduler | None = None,
    epoch: int | None = None,
    step: int | None = None,
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
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if epoch is not None:
        payload["epoch"] = epoch
    if step is not None:
        payload["step"] = step
    torch.save(payload, path)
