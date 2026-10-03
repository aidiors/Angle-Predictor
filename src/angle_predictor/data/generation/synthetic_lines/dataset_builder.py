from __future__ import annotations

import json
import logging
import os
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import numpy as np
from tqdm import tqdm

from .environment_renderer import (
    DEFAULT_ENVIRONMENTS_ROOT,
    available_environment_names,
)
from .image_generator import (
    AIMING_LINE_RENDERING,
    BACKGROUND_ASSET_CROP_SIZE,
    BACKGROUND_ASSET_EXTENSIONS,
    BACKGROUND_ASSETS_ROOT,
    BACKGROUND_SOURCE_PROBABILITIES,
    DEFAULT_CENTER_ASSETS_ROOT,
    PROCEDURAL_BACKGROUND_TYPE_PROBABILITIES,
    SYNTHETIC_AUGMENTATION_SETTINGS,
    encode_line_angle_degrees,
    generate_synthetic_image,
)

logger = logging.getLogger(__name__)

DATASET_LABEL_FORMAT = ["sin(2*line_angle_rad)", "cos(2*line_angle_rad)"]
DATASET_ANGLE_CONVENTION = {
    "angle_name": "line_angle_rad",
    "angle_units": "radians",
    "angle_range": "[0, pi)",
    "coordinate_system": "image coordinates: x right, y down",
    "zero_angle": "0 rad = horizontal line",
    "right_angle": "pi/2 rad = vertical line",
    "orientation_period": "pi; theta and theta + pi are the same undirected line",
    "label_format": DATASET_LABEL_FORMAT,
}

_worker_images_memmap: np.memmap | None = None
_worker_labels_memmap: np.memmap | None = None


def _initialize_worker_memmaps(
    images_path: str,
    labels_path: str,
    images_shape: tuple[int, int, int, int],
    labels_shape: tuple[int, int],
) -> None:
    """Open the shared output files once in each worker process."""
    global _worker_images_memmap, _worker_labels_memmap
    _worker_images_memmap = np.memmap(images_path, dtype=np.uint8, mode="r+", shape=images_shape)
    _worker_labels_memmap = np.memmap(labels_path, dtype=np.float32, mode="r+", shape=labels_shape)


def _generate_single_sample(
    args: tuple[int, int, int, str],
) -> tuple[int, np.ndarray, tuple[float, float]]:
    """Generate one sample with a seed derived only from dataset seed and index."""
    sample_index, base_seed, img_size, environments_folder = args
    sample_seed = np.random.SeedSequence([base_seed, sample_index])
    sample_rng = np.random.Generator(np.random.PCG64(sample_seed))
    image, line_angle_degrees = generate_synthetic_image(
        img_size=img_size,
        rng=sample_rng,
        environments_root=environments_folder,
    )
    return sample_index, image, encode_line_angle_degrees(line_angle_degrees)


def _generate_and_store_sample(args: tuple[int, int, int, str]) -> None:
    """Write a sample into its own slot without sending image bytes to the parent."""
    if _worker_images_memmap is None or _worker_labels_memmap is None:
        raise RuntimeError("Worker output memmaps have not been initialized")
    sample_index, image, label = _generate_single_sample(args)
    _worker_images_memmap[sample_index] = image
    _worker_labels_memmap[sample_index] = label


def generate_synthetic_dataset_memmap(
    output_folder: str | os.PathLike[str] | None = None,
    num_samples: int = 50_000,
    img_size: int = 256,
    environments_folder: str | os.PathLike[str] | None = None,
    seed: int = 42,
    batch_size: int = 500,
    max_workers: int | None = None,
) -> None:
    """Generate deterministic RGB images and angle labels to ``images.dat``."""
    if num_samples <= 0:
        raise ValueError("num_samples must be positive")
    if img_size <= 0:
        raise ValueError("img_size must be positive")
    if batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    environment_root = Path(environments_folder or DEFAULT_ENVIRONMENTS_ROOT).resolve()
    environment_names = available_environment_names(environment_root)
    if max_workers is None:
        max_workers = min(8, max(1, (os.cpu_count() or 1) - 1))
    if max_workers <= 0:
        raise ValueError("max_workers must be positive")

    if output_folder is None:
        output_folder = "synthetic_memmap_line_angle"
    output_path = Path(output_folder)
    output_path.mkdir(parents=True, exist_ok=True)
    if any(output_path.iterdir()):
        logger.warning(
            "Directory already contains files; dataset outputs will be overwritten: %s", output_path
        )

    images_path = output_path / "images.dat"
    labels_path = output_path / "labels.dat"
    meta_path = output_path / "meta.json"
    logger.info(
        "Generating %d %dx%d samples from %d layered environments with %d workers",
        num_samples,
        img_size,
        img_size,
        len(environment_names),
        max_workers,
    )

    images_memmap = np.memmap(
        images_path,
        dtype=np.uint8,
        mode="w+",
        shape=(num_samples, img_size, img_size, 3),
    )
    labels_memmap = np.memmap(
        labels_path,
        dtype=np.float32,
        mode="w+",
        shape=(num_samples, 2),
    )

    def store_result(
        result: tuple[int, np.ndarray, tuple[float, float]],
        image_map: np.memmap,
        label_map: np.memmap,
    ) -> None:
        index, image, label = result
        image_map[index] = image
        label_map[index] = label

    progress = tqdm(total=num_samples, desc="Generating samples")
    try:
        if max_workers == 1:
            for sample_index in range(num_samples):
                result = _generate_single_sample(
                    (sample_index, seed, img_size, str(environment_root))
                )
                store_result(result, images_memmap, labels_memmap)
                progress.update(1)
                if (sample_index + 1) % batch_size == 0:
                    images_memmap.flush()
                    labels_memmap.flush()
        else:
            with ProcessPoolExecutor(
                max_workers=max_workers,
                initializer=_initialize_worker_memmaps,
                initargs=(
                    str(images_path.resolve()),
                    str(labels_path.resolve()),
                    (num_samples, img_size, img_size, 3),
                    (num_samples, 2),
                ),
            ) as executor:
                for first_index in range(0, num_samples, batch_size):
                    batch_end = min(first_index + batch_size, num_samples)
                    futures = [
                        executor.submit(
                            _generate_and_store_sample,
                            (sample_index, seed, img_size, str(environment_root)),
                        )
                        for sample_index in range(first_index, batch_end)
                    ]
                    for future in as_completed(futures):
                        future.result()
                        progress.update(1)
                    images_memmap.flush()
                    labels_memmap.flush()
    finally:
        progress.close()
        images_memmap.flush()
        labels_memmap.flush()
        del images_memmap
        del labels_memmap

    meta: dict[str, Any] = {
        "dataset_version": "v12_background_sources_center_sprites",
        "num_samples": num_samples,
        "img_size": img_size,
        "seed": seed,
        "random_number_generation": {
            "bit_generator": "PCG64",
            "sample_seed_derivation": "SeedSequence([dataset_seed, sample_index])",
            "worker_and_batch_independent": True,
        },
        "environments": list(environment_names),
        "environment_assets": str(environment_root),
        "images_shape": [num_samples, img_size, img_size, 3],
        "images_dtype": "uint8",
        "images_channels": "RGB",
        "labels_shape": [num_samples, 2],
        "labels_dtype": "float32",
        "labels_format": DATASET_LABEL_FORMAT,
        "angle_convention": DATASET_ANGLE_CONVENTION,
        "label_decoding": "line_angle_rad = 0.5 * atan2(label[0], label[1]); if negative, add pi",
        "background_rendering": {
            "source_probabilities": BACKGROUND_SOURCE_PROBABILITIES,
            "source": (
                "layered environment assets, procedural color fields, or a cropped image asset"
            ),
            "camera_focus": "uniform sample inside a random scrolling scenery tile",
            "uncovered_pixels": "environment background colour unless full coverage is selected",
            "full_coverage_probability": SYNTHETIC_AUGMENTATION_SETTINGS[
                "background_full_coverage_probability"
            ],
            "procedural_type_probabilities": PROCEDURAL_BACKGROUND_TYPE_PROBABILITIES,
            "background_image_assets_root": str(BACKGROUND_ASSETS_ROOT),
            "background_image_asset_extensions": sorted(BACKGROUND_ASSET_EXTENSIONS),
            "image_asset_crop_size_pixels": BACKGROUND_ASSET_CROP_SIZE,
            "image_asset_crop_selection": (
                "uniform image choice and uniform top-left position for a full 256x256 crop; "
                "crop is resized to the requested output size"
            ),
            "center_sprite_assets_root": str(DEFAULT_CENTER_ASSETS_ROOT),
            "full_coverage_fill": (
                "random scenery-layer texture tile composited beneath rendered layers "
                "where those layers leave pixels uncovered"
            ),
            "features": [
                "parallax and zoom-factor transform",
                "RGBA textures and layer tint",
                "texture scroll",
                "alpha test and source-alpha blend",
                "weather layers sampled as a scene variant",
            ],
        },
        "aiming_line_rendering": AIMING_LINE_RENDERING,
        "visual_augmentation": SYNTHETIC_AUGMENTATION_SETTINGS,
        "generation_script": "dataset_builder.py",
    }
    meta_path.write_text(json.dumps(meta, indent=2, ensure_ascii=False), encoding="utf-8")
    logger.info("Dataset saved to %s", output_path.resolve())
    logger.info("  images.dat: %.2f GiB", images_path.stat().st_size / 1024**3)
    logger.info("  labels.dat: %.1f KiB", labels_path.stat().st_size / 1024)
