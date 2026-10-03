from __future__ import annotations

import math
from pathlib import Path
from typing import Any

import numpy as np
import torch
from PIL import Image

from angle_predictor.config.angle_experiment import AngleExperimentParams
from angle_predictor.data.angle_transforms import AngleBatchPreprocessor
from angle_predictor.models.line_angle import AnglePredictor, build_angle_network


def load_angle_predictor(
    checkpoint: str | Path, device: torch.device
) -> tuple[AnglePredictor, tuple[int, int], str]:
    """Restore the complete inference path from a training checkpoint."""
    payload: dict[str, Any] = torch.load(checkpoint, map_location="cpu", weights_only=True)
    metadata = payload.get("metadata", {})
    if metadata.get("format_version") != 1:
        raise ValueError("Unsupported angle checkpoint format")
    params = AngleExperimentParams.model_validate(metadata["params"])
    input_size = tuple(metadata["input_size"])
    if len(input_size) != 2 or any(not isinstance(dimension, int) for dimension in input_size):
        raise ValueError("Invalid input size in angle checkpoint")
    model = AnglePredictor(
        AngleBatchPreprocessor(input_size=input_size, **params.preprocessing.model_dump()),
        build_angle_network(params.model, load_pretrained=False),
        inference_precision=params.training.precision,
    )
    model.load_state_dict(payload["model_state_dict"], strict=True)
    model.to(device).eval()
    return model, input_size, params.training.precision


def load_rgb_image(path: str | Path, input_size: tuple[int, int]) -> torch.Tensor:
    """Load one RGB image without resizing the angle geometry."""
    with Image.open(path) as source:
        image = source.convert("RGB")
        if image.size != (input_size[1], input_size[0]):
            raise ValueError(
                f"{path}: expected {input_size[1]}x{input_size[0]} pixels, "
                f"got {image.width}x{image.height}"
            )
        pixels = np.asarray(image, dtype=np.uint8).copy()
    return torch.from_numpy(pixels).permute(2, 0, 1)


def predict_angle_batch(
    model: AnglePredictor, images: torch.Tensor, precision: str
) -> tuple[torch.Tensor, torch.Tensor]:
    """Return angles in degrees and raw double-angle vectors."""
    device = next(model.parameters()).device
    use_bf16 = precision == "bf16" and device.type == "cuda"
    with (
        torch.inference_mode(),
        torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16),
    ):
        vectors = model(images.to(device, non_blocking=device.type == "cuda"))
    vectors = vectors.float().cpu()
    if (vectors.norm(dim=-1) < 1e-6).any():
        raise ValueError("Model produced an undefined angle vector")
    angles = (0.5 * torch.atan2(vectors[:, 0], vectors[:, 1])).remainder(math.pi)
    return angles * (180 / math.pi), vectors
