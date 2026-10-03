import logging
import os
from pathlib import Path

import cv2
import numpy as np

from .image_generator import (
    ANGLE_CONVENTION,
    decode_line_angle_degrees,
    encode_line_angle_degrees,
    generate_synthetic_image,
)

logger = logging.getLogger(__name__)


def test_synthetic_images(
    num_samples: int = 10,
    output_dir: str = "test_output",
    img_size: int = 256,
    seed: int = 12345,
) -> list[tuple[str, float, float, float]]:
    """Generate ``num_samples`` synthetic images and save them to ``output_dir``.

    Returns a list of ``(path, line_angle_degrees, sin2angle, cos2angle)`` tuples.
    """
    os.makedirs(output_dir, exist_ok=True)

    base_rng = np.random.default_rng(seed)

    logger.info("Angle convention: %s", ANGLE_CONVENTION["angle_name"])
    logger.info("  angle range = [0, 180) degrees; 0 = right, positive clockwise")
    logger.info("  labels = [sin(2*line_angle_rad), cos(2*line_angle_rad)]")

    results: list[tuple[str, float, float, float]] = []
    for i in range(num_samples):
        sample_rng = np.random.default_rng(base_rng.integers(0, 1_000_000))
        img, line_angle_degrees = generate_synthetic_image(
            img_size=img_size,
            rng=sample_rng,
        )
        sin2angle, cos2angle = encode_line_angle_degrees(line_angle_degrees)
        decoded_angle_degrees = decode_line_angle_degrees(sin2angle, cos2angle)
        decoded_error = abs(decoded_angle_degrees - line_angle_degrees)
        decoded_error = min(decoded_error, 180.0 - decoded_error)
        if decoded_error > 1e-6:
            raise RuntimeError(
                "Angle round-trip encoding error: "
                f"angle={line_angle_degrees}, decoded={decoded_angle_degrees}"
            )

        angle_deg = line_angle_degrees
        filename = (
            f"sample_{i:02d}_"
            f"line_angle_deg={angle_deg:06.2f}_"
            f"sin2a={sin2angle:+.3f}_"
            f"cos2a={cos2angle:+.3f}.png"
        )
        filepath = os.path.join(output_dir, filename)
        cv2.imwrite(filepath, cv2.cvtColor(img, cv2.COLOR_RGB2BGR))
        results.append((filepath, line_angle_degrees, sin2angle, cos2angle))
        logger.info("%d/%d: %s", i + 1, num_samples, filename)

    logger.info("Done. Images saved to: %s", Path(output_dir).resolve())
    return results


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
    test_synthetic_images(num_samples=12, output_dir="test_output", img_size=256, seed=42)
