from __future__ import annotations

import json
from pathlib import Path
from typing import Literal

import numpy as np
import torch
from torch.utils.data import Dataset

Split = Literal["train", "val", "test"]
LABEL_FORMAT = ["sin(2*line_angle_rad)", "cos(2*line_angle_rad)"]


def make_split_indices(
    num_samples: int,
    *,
    seed: int = 42,
    train_fraction: float = 0.8,
    val_fraction: float = 0.1,
) -> dict[str, np.ndarray]:
    """Make disjoint, reproducible sample splits from a dataset index permutation."""
    if num_samples <= 0 or seed < 0:
        raise ValueError("num_samples must be positive and seed must be non-negative")
    if not 0 < train_fraction < 1 or not 0 < val_fraction < 1:
        raise ValueError("train_fraction and val_fraction must be in (0, 1)")
    if train_fraction + val_fraction >= 1:
        raise ValueError("train_fraction + val_fraction must be less than 1")

    permutation = np.random.default_rng(seed).permutation(num_samples)
    train_end = int(num_samples * train_fraction)
    val_end = train_end + int(num_samples * val_fraction)
    return {
        "train": permutation[:train_end],
        "val": permutation[train_end:val_end],
        "test": permutation[val_end:],
    }


class AngleMemmapDataset(Dataset[tuple[torch.Tensor, torch.Tensor]]):
    """Yield ``(RGB uint8 CHW, [sin(2θ), cos(2θ)] float32)`` samples.

    Each worker opens read-only memmaps lazily. Only one image and label are
    copied per item, so a 29 GiB dataset does not enter process memory.
    """

    def __init__(
        self,
        root: str | Path,
        split: Split,
        *,
        seed: int = 42,
        train_fraction: float = 0.8,
        val_fraction: float = 0.1,
    ) -> None:
        self.root = Path(root)
        if split not in {"train", "val", "test"}:
            raise ValueError(f"Unknown split: {split!r}")

        with (self.root / "meta.json").open(encoding="utf-8") as stream:
            meta = json.load(stream)
        count = meta.get("num_samples")
        size = meta.get("img_size")
        if not isinstance(count, int) or count <= 0:
            raise ValueError("meta.json has an invalid num_samples")
        if not isinstance(size, int) or size <= 1:
            raise ValueError("meta.json has an invalid img_size")
        if meta.get("images_shape") != [count, size, size, 3]:
            raise ValueError("Unsupported images_shape in meta.json")
        if meta.get("labels_shape") != [count, 2]:
            raise ValueError("Unsupported labels_shape in meta.json")
        if meta.get("images_dtype") != "uint8" or meta.get("labels_dtype") != "float32":
            raise ValueError("Unsupported dataset dtypes")
        if meta.get("images_channels") != "RGB" or meta.get("labels_format") != LABEL_FORMAT:
            raise ValueError("Unsupported image channels or angle label convention")
        convention = meta.get("angle_convention", {})
        if (
            convention.get("angle_range") != "[0, pi)"
            or convention.get("coordinate_system") != "image coordinates: x right, y down"
            or convention.get("orientation_period")
            != "pi; theta and theta + pi are the same undirected line"
        ):
            raise ValueError("Unsupported angle convention")

        self.images_shape = (count, size, size, 3)
        self.labels_shape = (count, 2)
        self.image_size = (size, size)
        self.images_path = self.root / "images.dat"
        self.labels_path = self.root / "labels.dat"
        if self.images_path.stat().st_size != count * size * size * 3:
            raise ValueError("images.dat size does not match meta.json")
        if self.labels_path.stat().st_size != count * 2 * np.dtype(np.float32).itemsize:
            raise ValueError("labels.dat size does not match meta.json")

        self.indices = make_split_indices(
            count, seed=seed, train_fraction=train_fraction, val_fraction=val_fraction
        )[split]
        self._images: np.memmap | None = None
        self._labels: np.memmap | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def __getstate__(self) -> dict[str, object]:
        """Do not serialize open mappings into spawned DataLoader workers."""
        state = self.__dict__.copy()
        state["_images"] = None
        state["_labels"] = None
        return state

    def close(self) -> None:
        """Release this process's file mappings (useful on Windows)."""
        self._images = None
        self._labels = None

    def __getitem__(self, index: int) -> tuple[torch.Tensor, torch.Tensor]:
        if not 0 <= index < len(self):
            raise IndexError(index)
        if self._images is None:
            self._images = np.memmap(
                self.images_path, dtype=np.uint8, mode="r", shape=self.images_shape
            )
            self._labels = np.memmap(
                self.labels_path, dtype=np.float32, mode="r", shape=self.labels_shape
            )
        assert self._labels is not None
        sample_index = int(self.indices[index])
        image = torch.from_numpy(np.array(self._images[sample_index], copy=True)).permute(2, 0, 1)
        label = torch.from_numpy(np.array(self._labels[sample_index], copy=True))
        return image, label
