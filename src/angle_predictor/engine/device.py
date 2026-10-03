from __future__ import annotations

from typing import Literal

import torch

DeviceName = Literal["auto", "cpu", "cuda"]


def resolve_device(requested: DeviceName) -> torch.device:
    """Resolve an explicit or automatic compute device."""
    if requested == "auto":
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if requested == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but is not available")
    return torch.device(requested)
