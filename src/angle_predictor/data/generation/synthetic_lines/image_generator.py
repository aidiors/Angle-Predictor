from __future__ import annotations

import math
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from .environment_renderer import (
    DEFAULT_ENVIRONMENTS_ROOT,
    BackgroundEnvironment,
    available_environment_names,
    load_environment,
    render_environment_background,
    render_environment_texture_backdrop,
)

ANGLE_CONVENTION = {
    "angle_name": "line_angle_degrees",
    "angle_units": "degrees",
    "angle_range": "[0, 180)",
    "coordinate_system": "image coordinates: x right, y down",
    "zero_angle": "0 degrees = horizontal toward the right",
    "positive_direction": "clockwise in image coordinates",
    "right_angle": "90 degrees = vertical downward",
    "orientation_period": "180 degrees; theta and theta + 180 degrees are the same undirected line",
    "label_angle_conversion": "line_angle_rad = radians(line_angle_degrees)",
    "label_format": ["sin(2*line_angle_rad)", "cos(2*line_angle_rad)"],
}

AIMING_LINE_WIDTH = 1.25
# The renderer supplies alpha 1.0 to glColor4f; line smoothing controls pixel coverage.
AIMING_LINE_ALPHA = 1.0
AIMING_LINE_RGBA = (1.0, 1.0, 1.0, AIMING_LINE_ALPHA)
BACKGROUND_ASSETS_ROOT = Path(__file__).resolve().parent / "assets" / "backgrounds"
DEFAULT_CENTER_ASSETS_ROOT = Path(__file__).resolve().parent / "assets" / "center"
BACKGROUND_ASSET_EXTENSIONS = {".png", ".jpg", ".jpeg"}
BACKGROUND_ASSET_CROP_SIZE = 256
BACKGROUND_SOURCE_PROBABILITIES = {
    "environment_asset": 0.60,
    "procedural": 0.20,
    "image_asset": 0.20,
}
PROCEDURAL_BACKGROUND_TYPES = (
    "solid",
    "horizontal_gradient",
    "vertical_gradient",
    "diagonal_gradient",
    "radial_gradient",
)
PROCEDURAL_BACKGROUND_TYPE_PROBABILITIES = {
    background_type: 1.0 / len(PROCEDURAL_BACKGROUND_TYPES)
    for background_type in PROCEDURAL_BACKGROUND_TYPES
}
BACKGROUND_FULL_COVERAGE_PROBABILITY = 0.85
BACKGROUND_TINT_PROBABILITY = 0.70
TINT_COLOR_PROBABILITIES = {"red": 0.45, "white": 0.45, "random": 0.10}
TINT_COVERAGE_RANGE = (0.60, 1.0)
TINT_ALPHA_RANGE = (0.0, 0.5)
DISTRACTOR_LINE_COUNT_PROBABILITIES = (0.30, 0.30, 0.22, 0.13, 0.05)
DISTRACTOR_FULL_FRAME_LINE_PROBABILITY = 0.40
ARC_COUNT_PROBABILITIES = (0.45, 0.32, 0.18, 0.05)
ARC_RELATION_PROBABILITIES = {
    "endpoint": 0.30,
    "orthogonal_endpoint": 0.50,
    "near_line": 0.20,
}
ARC_RADIUS_SCALE_RANGE = (0.30, 1.35)
DISTRACTOR_PROBABILITY = 0.60
DISTRACTOR_COLOR_PROBABILITIES = {"white": 0.20, "random": 0.80}
CENTER_OCCLUDER_UNDER_LINE_PROBABILITY = 0.05
CENTER_OCCLUDER_LAYER_PROBABILITIES = {
    "above_primary_line": 1.0 - CENTER_OCCLUDER_UNDER_LINE_PROBABILITY,
    "below_primary_line": CENTER_OCCLUDER_UNDER_LINE_PROBABILITY,
}
CENTER_LINE_VISIBLE_MARGIN_PIXELS = 2.0
CENTER_SPRITE_PROBABILITY = 0.70
CENTER_SPRITE_COLOR_SHIFT_PROBABILITY = 0.50
MAX_SHAPE_DISTRACTORS = 4
SYNTHETIC_AUGMENTATION_SETTINGS = {
    "background_source_probabilities": BACKGROUND_SOURCE_PROBABILITIES,
    "procedural_background_type_probabilities": PROCEDURAL_BACKGROUND_TYPE_PROBABILITIES,
    "background_image_asset_extensions": sorted(BACKGROUND_ASSET_EXTENSIONS),
    "background_image_asset_crop_size_pixels": BACKGROUND_ASSET_CROP_SIZE,
    "background_image_assets_root": str(BACKGROUND_ASSETS_ROOT),
    "background_full_coverage_probability": BACKGROUND_FULL_COVERAGE_PROBABILITY,
    "background_tint_probability": BACKGROUND_TINT_PROBABILITY,
    "background_tint_color_probabilities": TINT_COLOR_PROBABILITIES,
    "background_tint_coverage_range": TINT_COVERAGE_RANGE,
    "background_tint_alpha_range": TINT_ALPHA_RANGE,
    "distractor_probability": DISTRACTOR_PROBABILITY,
    "distractor_color_probabilities": DISTRACTOR_COLOR_PROBABILITIES,
    "distractor_line_count_probabilities_zero_to_four": DISTRACTOR_LINE_COUNT_PROBABILITIES,
    "full_frame_line_probability_per_distractor_line": DISTRACTOR_FULL_FRAME_LINE_PROBABILITY,
    "full_frame_line_probability_baseline_assumption": 0.20,
    "full_frame_line_probability_multiplier": 2.0,
    "arc_count_probabilities_zero_to_three": ARC_COUNT_PROBABILITIES,
    "arc_relation_probabilities": ARC_RELATION_PROBABILITIES,
    "arc_radius_scale_of_short_side": ARC_RADIUS_SCALE_RANGE,
    "shape_distractor_count_range": [0, MAX_SHAPE_DISTRACTORS],
    "center_occluder_given_distractors": True,
    "center_occluder_color_probabilities": DISTRACTOR_COLOR_PROBABILITIES,
    "center_occluder_layer_probabilities": CENTER_OCCLUDER_LAYER_PROBABILITIES,
    "center_line_visible_margin_pixels": CENTER_LINE_VISIBLE_MARGIN_PIXELS,
    "center_sprite_probability": CENTER_SPRITE_PROBABILITY,
    "center_distractor_type_probabilities": {
        "dds_sprite": CENTER_SPRITE_PROBABILITY,
        "procedural_shape": 1.0 - CENTER_SPRITE_PROBABILITY,
    },
    "center_sprite_color_shift_probability": CENTER_SPRITE_COLOR_SHIFT_PROBABILITY,
    "center_sprite_assets_root": str(DEFAULT_CENTER_ASSETS_ROOT),
}
AIMING_LINE_RENDERING = {
    "primitive": "finite GL_LINES-style segments",
    "center_line_count": 1,
    "center_line_extent": "randomized finite length; always crosses the image center",
    "minimum_maximum_extents": False,
    "distractor_probability": DISTRACTOR_PROBABILITY,
    "optional_detached_lines": True,
    "detached_line_count_probabilities_zero_to_four": DISTRACTOR_LINE_COUNT_PROBABILITIES,
    "full_frame_line_probability_per_distractor_line": DISTRACTOR_FULL_FRAME_LINE_PROBABILITY,
    "detached_line_anchor": "at a nonzero perpendicular distance from the image center",
    "optional_arcs": True,
    "arc_count_probabilities_zero_to_three": ARC_COUNT_PROBABILITIES,
    "arc_relation_probabilities": ARC_RELATION_PROBABILITIES,
    "arc_radius_scale_of_short_side": ARC_RADIUS_SCALE_RANGE,
    "arc_placement": [
        "near the center line",
        "attached to a center-line endpoint",
        "orthogonal to the center line at an endpoint",
    ],
    "distractor_color_probabilities": DISTRACTOR_COLOR_PROBABILITIES,
    "center_occluder_layer_probabilities": CENTER_OCCLUDER_LAYER_PROBABILITIES,
    "center_line_protection": (
        "center object leaves at least 2 px of line visible on both sides; "
        "object is above the line 95% of the time and below it 5% of the time"
    ),
    "center_sprite_probability": CENTER_SPRITE_PROBABILITY,
    "center_sprite_color_shift_probability": CENTER_SPRITE_COLOR_SHIFT_PROBABILITY,
    "label_line": "central",
    "width_pixels": AIMING_LINE_WIDTH,
    "smooth": True,
    "smooth_hint": "GL_NICEST",
    "blend": "SRC_ALPHA, ONE_MINUS_SRC_ALPHA",
    "color_rgba": AIMING_LINE_RGBA,
    "alpha_source": "per-segment vertex alpha",
    "effective_pixel_alpha": "vertex alpha multiplied by line-smoothing coverage",
    "rasterizer": "software pixel-area coverage; driver-level GL coverage can differ",
}


def angle_to_line_vector(line_angle_rad: float) -> tuple[float, float]:
    """Return the unit vector along an undirected line in image coordinates."""
    return math.cos(line_angle_rad), math.sin(line_angle_rad)


def encode_line_angle(line_angle_rad: float) -> tuple[float, float]:
    """Encode an undirected line as ``(sin(2θ), cos(2θ))``."""
    return math.sin(2.0 * line_angle_rad), math.cos(2.0 * line_angle_rad)


def decode_line_angle(sin_2angle: float, cos_2angle: float) -> float:
    """Decode ``(sin(2θ), cos(2θ))`` into radians in ``[0, pi)``."""
    line_angle_rad = 0.5 * math.atan2(sin_2angle, cos_2angle)
    if line_angle_rad < 0.0:
        line_angle_rad += math.pi
    return line_angle_rad


def encode_line_angle_degrees(line_angle_degrees: float) -> tuple[float, float]:
    """Encode a canonical undirected line angle in degrees from ``[0, 180)``."""
    if not math.isfinite(line_angle_degrees) or not 0.0 <= line_angle_degrees < 180.0:
        raise ValueError("line_angle_degrees must be finite and in [0, 180)")
    return encode_line_angle(math.radians(line_angle_degrees))


def decode_line_angle_degrees(sin_2angle: float, cos_2angle: float) -> float:
    """Decode the label to the canonical degree range ``[0, 180)``."""
    return math.degrees(decode_line_angle(sin_2angle, cos_2angle))


def sample_background_fill_mode(rng: np.random.Generator) -> bool:
    """Return whether uncovered environment pixels should be filled this sample."""
    return bool(rng.random() < BACKGROUND_FULL_COVERAGE_PROBABILITY)


def sample_background_source(rng: np.random.Generator) -> str:
    """Choose the source family for a sample's background."""
    choice = float(rng.random())
    environment_limit = BACKGROUND_SOURCE_PROBABILITIES["environment_asset"]
    procedural_limit = environment_limit + BACKGROUND_SOURCE_PROBABILITIES["procedural"]
    if choice < environment_limit:
        return "environment_asset"
    if choice < procedural_limit:
        return "procedural"
    return "image_asset"


@lru_cache(maxsize=8)
def available_background_asset_paths(
    assets_root: str | Path = BACKGROUND_ASSETS_ROOT,
) -> tuple[Path, ...]:
    """Return supported image assets large enough to contain a 256 px square."""
    root = Path(assets_root)
    if not root.is_dir():
        return ()
    paths: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() not in BACKGROUND_ASSET_EXTENSIONS:
            continue
        try:
            with Image.open(path) as image:
                if (
                    image.width >= BACKGROUND_ASSET_CROP_SIZE
                    and image.height >= BACKGROUND_ASSET_CROP_SIZE
                ):
                    paths.append(path)
        except OSError:
            continue
    return tuple(paths)


@lru_cache(maxsize=64)
def _load_background_asset_rgb(asset_path: str) -> np.ndarray:
    """Keep the bundled background images decoded within each worker process."""
    with Image.open(asset_path) as source:
        rgba = source.convert("RGBA")
        opaque = Image.new("RGBA", rgba.size, (0, 0, 0, 255))
        opaque.alpha_composite(rgba)
        pixels = np.asarray(opaque.convert("RGB"), dtype=np.uint8)
    pixels.setflags(write=False)
    return pixels


def render_background_asset_crop(
    image_width: int,
    image_height: int,
    rng: np.random.Generator,
    *,
    assets_root: str | Path = BACKGROUND_ASSETS_ROOT,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Choose a random full 256x256 crop and scale it to the output viewport."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("background output dimensions must be positive")
    asset_paths = available_background_asset_paths(assets_root)
    if not asset_paths:
        raise FileNotFoundError(
            f"No PNG, JPG, or JPEG background at least 256x256 found in {Path(assets_root)}"
        )
    asset_path = asset_paths[int(rng.integers(0, len(asset_paths)))]
    source = _load_background_asset_rgb(str(asset_path))
    max_x = source.shape[1] - BACKGROUND_ASSET_CROP_SIZE
    max_y = source.shape[0] - BACKGROUND_ASSET_CROP_SIZE
    x0 = int(rng.integers(0, max_x + 1))
    y0 = int(rng.integers(0, max_y + 1))
    x1 = x0 + BACKGROUND_ASSET_CROP_SIZE
    y1 = y0 + BACKGROUND_ASSET_CROP_SIZE
    crop = source[y0:y1, x0:x1]
    if (image_width, image_height) == (BACKGROUND_ASSET_CROP_SIZE, BACKGROUND_ASSET_CROP_SIZE):
        rendered = crop.copy()
    else:
        rendered = np.asarray(
            Image.fromarray(crop).resize(
                (image_width, image_height),
                resample=Image.Resampling.LANCZOS,
            ),
            dtype=np.uint8,
        ).copy()
    return rendered, {
        "asset_path": str(asset_path),
        "source_size_pixels": [int(source.shape[1]), int(source.shape[0])],
        "crop_bbox_xyxy": (x0, y0, x1, y1),
    }


def render_procedural_background(
    image_width: int,
    image_height: int,
    rng: np.random.Generator,
    *,
    background_type: str | None = None,
) -> tuple[np.ndarray, str]:
    """Render a random solid color or one of four full-frame color gradients."""
    if image_width <= 0 or image_height <= 0:
        raise ValueError("background output dimensions must be positive")
    if background_type is None:
        background_type = str(
            rng.choice(
                PROCEDURAL_BACKGROUND_TYPES,
                p=tuple(PROCEDURAL_BACKGROUND_TYPE_PROBABILITIES.values()),
            )
        )
    if background_type not in PROCEDURAL_BACKGROUND_TYPE_PROBABILITIES:
        raise ValueError(f"unsupported procedural background type: {background_type}")

    first_color = rng.integers(0, 256, size=3).astype(np.float32)
    if background_type == "solid":
        return np.broadcast_to(
            first_color.astype(np.uint8), (image_height, image_width, 3)
        ).copy(), background_type

    second_color = rng.integers(0, 256, size=3).astype(np.float32)
    x_progress = np.linspace(0.0, 1.0, image_width, dtype=np.float32)[None, :]
    y_progress = np.linspace(0.0, 1.0, image_height, dtype=np.float32)[:, None]
    if background_type == "horizontal_gradient":
        progress = np.broadcast_to(x_progress, (image_height, image_width))
    elif background_type == "vertical_gradient":
        progress = np.broadcast_to(y_progress, (image_height, image_width))
    elif background_type == "diagonal_gradient":
        progress = (x_progress + y_progress) / 2.0
    else:
        center_x = float(rng.uniform(0.0, max(image_width - 1, 1)))
        center_y = float(rng.uniform(0.0, max(image_height - 1, 1)))
        pixel_y, pixel_x = np.ogrid[:image_height, :image_width]
        distance = np.hypot(pixel_x - center_x, pixel_y - center_y)
        farthest_corner = max(
            math.hypot(center_x, center_y),
            math.hypot(image_width - 1 - center_x, center_y),
            math.hypot(center_x, image_height - 1 - center_y),
            math.hypot(image_width - 1 - center_x, image_height - 1 - center_y),
            1.0,
        )
        progress = np.clip(distance / farthest_corner, 0.0, 1.0)
    image = first_color + (second_color - first_color) * progress[..., None]
    return np.rint(np.clip(image, 0, 255)).astype(np.uint8), background_type


@lru_cache(maxsize=8)
def available_center_sprite_paths(
    assets_root: str | Path = DEFAULT_CENTER_ASSETS_ROOT,
) -> tuple[Path, ...]:
    """Return readable DDS sprites available for center overlays."""
    root = Path(assets_root)
    if not root.is_dir():
        return ()
    paths: list[Path] = []
    for path in sorted(root.iterdir()):
        if not path.is_file() or path.suffix.lower() != ".dds":
            continue
        try:
            with Image.open(path) as image:
                if image.width > 0 and image.height > 0:
                    paths.append(path)
        except OSError:
            continue
    return tuple(paths)


@lru_cache(maxsize=32)
def _load_center_sprite_rgba(asset_path: str) -> np.ndarray:
    with Image.open(asset_path) as source:
        pixels = np.asarray(source.convert("RGBA"), dtype=np.uint8)
    pixels.setflags(write=False)
    return pixels


def shift_center_sprite_hue(sprite_rgba: np.ndarray, hue_shift: float) -> np.ndarray:
    """Rotate visible sprite hues while keeping alpha and dark outlines intact."""
    if sprite_rgba.ndim != 3 or sprite_rgba.shape[2] != 4 or sprite_rgba.dtype != np.uint8:
        raise ValueError("sprite_rgba must be a uint8 RGBA array")
    if not math.isfinite(hue_shift) or not 0.0 <= hue_shift < 1.0:
        raise ValueError("hue_shift must be finite and in [0, 1)")

    hsv = np.asarray(Image.fromarray(sprite_rgba, mode="RGBA").convert("HSV")).copy()
    alpha = sprite_rgba[:, :, 3]
    visible = (alpha > 0) & (hsv[:, :, 2] >= 48)
    hue_offset = round(hue_shift * 255.0)
    hsv[:, :, 0][visible] = (hsv[:, :, 0][visible].astype(np.uint16) + hue_offset) % 256
    hsv[:, :, 1][visible] = np.maximum(hsv[:, :, 1][visible], 160)
    shifted_rgb = np.asarray(Image.fromarray(hsv, mode="HSV").convert("RGB"))
    shifted = np.dstack((shifted_rgb, alpha))
    return shifted.astype(np.uint8, copy=False)


def fill_uncovered_background(
    image: np.ndarray,
    coverage_mask: np.ndarray,
    fallback_background: np.ndarray,
) -> np.ndarray:
    """Fill uncovered pixels from a full-frame texture backdrop."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 RGB array")
    if coverage_mask.shape != image.shape[:2] or coverage_mask.dtype != np.bool_:
        raise ValueError("coverage_mask must be a boolean array matching the image size")
    if fallback_background.shape != image.shape or fallback_background.dtype != np.uint8:
        raise ValueError("fallback_background must be a uint8 RGB array matching the image size")
    filled = image.copy()
    filled[~coverage_mask] = fallback_background[~coverage_mask]
    return filled


def native_fire_guide_angles_rad(
    yaw_rad: float,
    fire_angle_offset_rad: float,
    min_fire_angle_rad: float,
    max_fire_angle_rad: float,
    *,
    side_one: bool,
    ignore_slope: bool = False,
) -> tuple[float, float, float]:
    """Return central and boundary angles for a guide.

    The returned angles use the world-coordinate convention. Dataset geometry is
    centered independently, then the side directions are offset from the
    central line by the same angular differences.
    """
    values = (yaw_rad, fire_angle_offset_rad, min_fire_angle_rad, max_fire_angle_rad)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("Aiming angles must be finite")

    center = yaw_rad + fire_angle_offset_rad + 3.0 * math.pi / 2.0
    q = math.pi if ignore_slope else yaw_rad + math.pi / 2.0
    if side_one:
        max_direction = q + math.pi + max_fire_angle_rad
        min_direction = q + math.pi + min_fire_angle_rad
    else:
        max_direction = q + 2.0 * math.pi - max_fire_angle_rad
        min_direction = q + 2.0 * math.pi - min_fire_angle_rad
    return center, max_direction, min_direction


def _validate_stroke(
    image: np.ndarray,
    line_width: float,
    colour_rgba: tuple[float, float, float, float],
) -> None:
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 RGB array")
    if not math.isfinite(line_width) or line_width <= 0:
        raise ValueError("line width must be finite and positive")
    if len(colour_rgba) != 4 or not all(math.isfinite(value) for value in colour_rgba):
        raise ValueError("colour_rgba must contain four finite values")


def _projection_cdf(value: np.ndarray, normal_x: float, normal_y: float) -> np.ndarray:
    """Pixel-area CDF for a unit square projected on a 2D normal."""
    a = abs(normal_x)
    b = abs(normal_y)
    small_component = np.minimum(a, b) < 1e-8
    linear_cdf = np.clip((value + (a + b) / 2.0) / np.maximum(a + b, 1e-8), 0.0, 1.0)
    half_span = (a + b) / 2.0
    first = np.maximum(value + half_span, 0.0)
    second = np.maximum(value + (a - b) / 2.0, 0.0)
    third = np.maximum(value + (b - a) / 2.0, 0.0)
    fourth = np.maximum(value - half_span, 0.0)
    cdf = (first**2 - second**2 - third**2 + fourth**2) / np.maximum(2.0 * a * b, 1e-8)
    return np.where(small_component, linear_cdf, np.clip(cdf, 0.0, 1.0))


def _line_pixel_coverage(
    distance: np.ndarray,
    line_width: float,
    *,
    normal_x: float,
    normal_y: float,
) -> np.ndarray:
    """Compute area coverage of a pixel by an antialiased line strip."""
    half_width = line_width / 2.0
    return np.clip(
        _projection_cdf(half_width - distance, normal_x, normal_y)
        - _projection_cdf(-half_width - distance, normal_x, normal_y),
        0.0,
        1.0,
    )


def _blend_stroke_coverage(
    image: np.ndarray,
    coverage: np.ndarray,
    colour_rgba: tuple[float, float, float, float],
) -> None:
    coverage *= colour_rgba[3]
    if not np.any(coverage):
        return
    source_rgb = np.asarray(colour_rgba[:3], dtype=np.float32) * 255.0
    destination = image.astype(np.float32)
    alpha = coverage[..., None]
    blended = source_rgb * alpha + destination * (1.0 - alpha)
    active = coverage > 0.0
    image[active] = np.rint(np.clip(blended[active], 0.0, 255.0)).astype(np.uint8)


def draw_gl_smooth_segment(
    image: np.ndarray,
    start_xy: tuple[float, float],
    end_xy: tuple[float, float],
    *,
    line_width: float = AIMING_LINE_WIDTH,
    colour_rgba: tuple[float, float, float, float] = AIMING_LINE_RGBA,
) -> None:
    """Blend one finite antialiased line segment into an RGB image."""
    _validate_stroke(image, line_width, colour_rgba)
    if not all(math.isfinite(value) for value in (*start_xy, *end_xy)):
        raise ValueError("segment endpoints must be finite")

    delta_x = end_xy[0] - start_xy[0]
    delta_y = end_xy[1] - start_xy[1]
    segment_length = math.hypot(delta_x, delta_y)
    if segment_length <= 0.0:
        raise ValueError("segment endpoints must be different")
    direction_x = delta_x / segment_length
    direction_y = delta_y / segment_length
    midpoint_x = (start_xy[0] + end_xy[0]) / 2.0
    midpoint_y = (start_xy[1] + end_xy[1]) / 2.0

    height, width = image.shape[:2]
    pixel_y, pixel_x = np.ogrid[:height, :width]
    delta_x = pixel_x.astype(np.float32) + 0.5 - midpoint_x
    delta_y = pixel_y.astype(np.float32) + 0.5 - midpoint_y
    perpendicular_distance = np.abs(-direction_y * delta_x + direction_x * delta_y)
    along_distance = direction_x * delta_x + direction_y * delta_y
    side_coverage = _line_pixel_coverage(
        perpendicular_distance,
        line_width,
        normal_x=abs(direction_y),
        normal_y=abs(direction_x),
    )
    end_coverage = _line_pixel_coverage(
        along_distance,
        segment_length,
        normal_x=abs(direction_x),
        normal_y=abs(direction_y),
    )
    _blend_stroke_coverage(image, side_coverage * end_coverage, colour_rgba)


def draw_gl_smooth_line(
    image: np.ndarray,
    line_angle_rad: float,
    *,
    negative_length: float | None = None,
    positive_length: float | None = None,
    line_width: float = AIMING_LINE_WIDTH,
    colour_rgba: tuple[float, float, float, float] = AIMING_LINE_RGBA,
) -> None:
    """Draw a centered stroke, optionally with finite length on each side.

    With finite lengths, the resulting segment crosses the exact image center.
    Leaving both lengths unset retains an across-viewport line for callers that
    need that geometry.
    """
    if not math.isfinite(line_angle_rad):
        raise ValueError("line angle must be finite")
    if (negative_length is None) != (positive_length is None):
        raise ValueError("negative_length and positive_length must be set together")
    if negative_length is None:
        half_length = 2.0 * math.hypot(*image.shape[:2])
        negative_length = positive_length = half_length
    else:
        assert positive_length is not None
        if (
            not math.isfinite(negative_length)
            or not math.isfinite(positive_length)
            or negative_length <= 0.0
            or positive_length <= 0.0
        ):
            raise ValueError("center-line lengths must be finite and positive")

    direction_x, direction_y = angle_to_line_vector(line_angle_rad)
    center_x = image.shape[1] / 2.0
    center_y = image.shape[0] / 2.0
    start_xy = (
        center_x - direction_x * negative_length,
        center_y - direction_y * negative_length,
    )
    end_xy = (
        center_x + direction_x * positive_length,
        center_y + direction_y * positive_length,
    )
    draw_gl_smooth_segment(
        image,
        start_xy,
        end_xy,
        line_width=line_width,
        colour_rgba=colour_rgba,
    )


def _disk_pixel_coverage(
    delta_x: np.ndarray,
    delta_y: np.ndarray,
    radius: float,
) -> np.ndarray:
    radial_distance = np.hypot(delta_x, delta_y)
    normal_x = np.abs(delta_x) / np.maximum(radial_distance, 1e-8)
    normal_y = np.abs(delta_y) / np.maximum(radial_distance, 1e-8)
    return _projection_cdf(radius - radial_distance, normal_x, normal_y)


def draw_gl_smooth_arc(
    image: np.ndarray,
    *,
    center_xy: tuple[float, float],
    radius: float,
    start_angle_rad: float,
    sweep_angle_rad: float,
    line_width: float = AIMING_LINE_WIDTH,
    colour_rgba: tuple[float, float, float, float] = AIMING_LINE_RGBA,
) -> None:
    """Blend an open circular stroke with smooth coverage and rounded ends."""
    _validate_stroke(image, line_width, colour_rgba)
    values = (*center_xy, radius, start_angle_rad, sweep_angle_rad)
    if not all(math.isfinite(value) for value in values):
        raise ValueError("arc geometry must be finite")
    if radius <= 0.0 or not 0.0 < sweep_angle_rad <= 2.0 * math.pi:
        raise ValueError("arc radius and sweep must be positive, with sweep at most 2*pi")

    height, width = image.shape[:2]
    pixel_y, pixel_x = np.ogrid[:height, :width]
    delta_x = pixel_x.astype(np.float32) + 0.5 - center_xy[0]
    delta_y = pixel_y.astype(np.float32) + 0.5 - center_xy[1]
    radial_distance = np.hypot(delta_x, delta_y)
    normal_x = np.abs(delta_x) / np.maximum(radial_distance, 1e-8)
    normal_y = np.abs(delta_y) / np.maximum(radial_distance, 1e-8)
    ring_coverage = _line_pixel_coverage(
        np.abs(radial_distance - radius),
        line_width,
        normal_x=normal_x,
        normal_y=normal_y,
    )
    relative_angle = np.mod(np.arctan2(delta_y, delta_x) - start_angle_rad, 2.0 * math.pi)
    arc_body = ring_coverage * (relative_angle <= sweep_angle_rad)

    end_angle_rad = start_angle_rad + sweep_angle_rad
    start_xy = (
        center_xy[0] + radius * math.cos(start_angle_rad),
        center_xy[1] + radius * math.sin(start_angle_rad),
    )
    end_xy = (
        center_xy[0] + radius * math.cos(end_angle_rad),
        center_xy[1] + radius * math.sin(end_angle_rad),
    )
    cap_radius = line_width / 2.0
    start_cap = _disk_pixel_coverage(
        pixel_x.astype(np.float32) + 0.5 - start_xy[0],
        pixel_y.astype(np.float32) + 0.5 - start_xy[1],
        cap_radius,
    )
    end_cap = _disk_pixel_coverage(
        pixel_x.astype(np.float32) + 0.5 - end_xy[0],
        pixel_y.astype(np.float32) + 0.5 - end_xy[1],
        cap_radius,
    )
    _blend_stroke_coverage(image, np.maximum(arc_body, np.maximum(start_cap, end_cap)), colour_rgba)


def _sample_background_camera_focus(
    environment: BackgroundEnvironment,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Choose a camera center that lands inside a random authored scenery tile."""
    layers = [
        layer
        for layer in environment.layers
        if not layer.foreground
        and not layer.weather
        and layer.grid_size_x is not None
        and layer.grid_size_y is not None
        and layer.scroll_rate_x != 0.0
        and layer.scroll_rate_y != 0.0
        and layer.tiles
    ]
    if not layers:
        # Keep non-bundled environments usable when they do not define a
        # parallax tile with enough information to derive its map bounds.
        return float(rng.uniform(-500.0, 500.0)), float(rng.uniform(-350.0, 350.0))

    layer = layers[int(rng.integers(0, len(layers)))]
    tile = layer.tiles[int(rng.integers(0, len(layer.tiles)))]
    assert layer.grid_size_x is not None
    assert layer.grid_size_y is not None

    tile_left = (tile.grid_x + layer.grid_tile_offset_x) * layer.grid_size_x
    tile_top = (tile.grid_y + layer.grid_tile_offset_y) * layer.grid_size_y
    layer_x = float(rng.uniform(tile_left, tile_left + layer.grid_size_x))
    layer_y = float(rng.uniform(tile_top, tile_top + layer.grid_size_y))
    return layer_x / layer.scroll_rate_x, layer_y / layer.scroll_rate_y


def _random_rgb_color(rng: np.random.Generator) -> tuple[int, int, int]:
    while True:
        channels = rng.integers(0, 256, size=3)
        color = int(channels[0]), int(channels[1]), int(channels[2])
        if color not in ((255, 255, 255), (255, 0, 0)):
            return color


def _sample_color_category(
    rng: np.random.Generator,
    *,
    white_probability: float,
) -> tuple[str, tuple[int, int, int]]:
    if rng.random() < white_probability:
        return "white", (255, 255, 255)
    return "random", _random_rgb_color(rng)


def _sample_tint_color(rng: np.random.Generator) -> tuple[str, tuple[int, int, int]]:
    choice = float(rng.random())
    red_limit = TINT_COLOR_PROBABILITIES["red"]
    white_limit = red_limit + TINT_COLOR_PROBABILITIES["white"]
    if choice < red_limit:
        return "red", (255, 0, 0)
    if choice < white_limit:
        return "white", (255, 255, 255)
    return "random", _random_rgb_color(rng)


def _sample_tint_spec(
    image_width: int,
    image_height: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    target_coverage = float(rng.uniform(*TINT_COVERAGE_RANGE))
    width_fraction = float(rng.uniform(target_coverage, 1.0))
    rect_width = min(image_width, math.ceil(width_fraction * image_width))
    rect_height = min(
        image_height,
        math.ceil(target_coverage * image_width * image_height / rect_width),
    )
    x0 = int(rng.integers(0, image_width - rect_width + 1))
    y0 = int(rng.integers(0, image_height - rect_height + 1))
    color_category, color_rgb = _sample_tint_color(rng)
    return {
        "bbox_xyxy": (x0, y0, x0 + rect_width, y0 + rect_height),
        "coverage_fraction": rect_width * rect_height / (image_width * image_height),
        "color_category": color_category,
        "color_rgb": color_rgb,
        "alpha": float(rng.uniform(*TINT_ALPHA_RANGE)),
    }


def _sample_shape_spec(
    *,
    shape: str,
    bbox_xyxy: tuple[int, int, int, int],
    color_category: str,
    color_rgb: tuple[int, int, int],
) -> dict[str, Any]:
    return {
        "shape": shape,
        "bbox_xyxy": bbox_xyxy,
        "color_category": color_category,
        "color_rgb": color_rgb,
    }


def _sample_free_shape(
    image_width: int,
    image_height: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    shape = "circle" if rng.random() < 0.5 else "square"
    side = max(5, round(rng.uniform(0.06, 0.22) * min(image_width, image_height)))
    side = min(side, image_width, image_height)
    x0 = int(rng.integers(0, image_width - side + 1))
    y0 = int(rng.integers(0, image_height - side + 1))
    color_category, color_rgb = _sample_color_category(
        rng,
        white_probability=DISTRACTOR_COLOR_PROBABILITIES["white"],
    )
    return _sample_shape_spec(
        shape=shape,
        bbox_xyxy=(x0, y0, x0 + side, y0 + side),
        color_category=color_category,
        color_rgb=color_rgb,
    )


def _sample_center_sprite(
    image_width: int,
    image_height: int,
    rng: np.random.Generator,
    primary_line_start_xy: tuple[float, float],
    primary_line_end_xy: tuple[float, float],
    sprite_paths: tuple[Path, ...],
) -> dict[str, Any] | None:
    asset_path = sprite_paths[int(rng.integers(0, len(sprite_paths)))]
    sprite_rgba = _load_center_sprite_rgba(str(asset_path))
    source_height, source_width = sprite_rgba.shape[:2]
    line_delta = (
        primary_line_end_xy[0] - primary_line_start_xy[0],
        primary_line_end_xy[1] - primary_line_start_xy[1],
    )
    line_length = math.hypot(*line_delta)
    direction_x, direction_y = line_delta[0] / line_length, line_delta[1] / line_length
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    negative_extent = (center_x - primary_line_start_xy[0]) * direction_x + (
        center_y - primary_line_start_xy[1]
    ) * direction_y
    positive_extent = (primary_line_end_xy[0] - center_x) * direction_x + (
        primary_line_end_xy[1] - center_y
    ) * direction_y
    frame_extent = min(
        extent / abs(component)
        for extent, component in zip((center_x, center_y), (direction_x, direction_y), strict=True)
        if abs(component) > 1e-12
    )
    max_half_span = (
        min(negative_extent, positive_extent, frame_extent) - CENTER_LINE_VISIBLE_MARGIN_PIXELS
    )
    if max_half_span <= 0.0:
        return None

    projected_half_span = 0.5 * (abs(direction_x) * source_width + abs(direction_y) * source_height)
    scale = min(1.0, max_half_span / projected_half_span)
    box_width = max(1, math.floor(source_width * scale))
    box_height = max(1, math.floor(source_height * scale))

    def fitted_half_span() -> float:
        x0 = math.floor(center_x - box_width / 2.0)
        y0 = math.floor(center_y - box_height / 2.0)
        offset = abs(
            (x0 + box_width / 2.0 - center_x) * direction_x
            + (y0 + box_height / 2.0 - center_y) * direction_y
        )
        return 0.5 * (abs(direction_x) * box_width + abs(direction_y) * box_height) + offset

    while fitted_half_span() > max_half_span:
        if abs(direction_x) * box_width >= abs(direction_y) * box_height and box_width > 1:
            box_width -= 1
        elif box_height > 1:
            box_height -= 1
        elif box_width > 1:
            box_width -= 1
        else:
            return None

    x0 = math.floor(center_x - box_width / 2.0)
    y0 = math.floor(center_y - box_height / 2.0)
    hue_shift = (
        float(rng.random()) if rng.random() < CENTER_SPRITE_COLOR_SHIFT_PROBABILITY else None
    )
    return {
        "kind": "sprite",
        "asset_path": str(asset_path),
        "bbox_xyxy": (x0, y0, x0 + box_width, y0 + box_height),
        "color_category": "asset",
        "color_shift_hue": hue_shift,
    }


def _sample_center_occluder(
    image_width: int,
    image_height: int,
    rng: np.random.Generator,
    primary_line_start_xy: tuple[float, float],
    primary_line_end_xy: tuple[float, float],
    center_assets_root: str | Path = DEFAULT_CENTER_ASSETS_ROOT,
) -> dict[str, Any] | None:
    sprite_paths = available_center_sprite_paths(center_assets_root)
    if sprite_paths and rng.random() < CENTER_SPRITE_PROBABILITY:
        sprite_spec = _sample_center_sprite(
            image_width,
            image_height,
            rng,
            primary_line_start_xy,
            primary_line_end_xy,
            sprite_paths,
        )
        if sprite_spec is not None:
            return sprite_spec

    shape = str(rng.choice(("square", "circle", "rectangle")))
    short_side = min(image_width, image_height)
    base_size = rng.uniform(0.08, 0.24) * short_side
    if shape == "rectangle":
        aspect = float(rng.uniform(0.55, 1.8))
        box_width = max(4, round(base_size * math.sqrt(aspect)))
        box_height = max(4, round(base_size / math.sqrt(aspect)))
    else:
        box_width = box_height = max(4, round(base_size))
    box_width = min(box_width, image_width)
    box_height = min(box_height, image_height)
    line_delta = (
        primary_line_end_xy[0] - primary_line_start_xy[0],
        primary_line_end_xy[1] - primary_line_start_xy[1],
    )
    line_length = math.hypot(*line_delta)
    line_direction = (line_delta[0] / line_length, line_delta[1] / line_length)
    center_x = image_width / 2.0
    center_y = image_height / 2.0
    center_extent = (center_x, center_y)
    negative_extent = (center_x - primary_line_start_xy[0]) * line_direction[0] + (
        center_y - primary_line_start_xy[1]
    ) * line_direction[1]
    positive_extent = (primary_line_end_xy[0] - center_x) * line_direction[0] + (
        primary_line_end_xy[1] - center_y
    ) * line_direction[1]
    frame_extent = min(
        extent / abs(component)
        for extent, component in zip(center_extent, line_direction, strict=True)
        if abs(component) > 1e-12
    )
    max_shape_half_span = (
        min(negative_extent, positive_extent, frame_extent) - CENTER_LINE_VISIBLE_MARGIN_PIXELS
    )
    if max_shape_half_span <= 0.0:
        return None

    def projected_half_span() -> float:
        bbox_x0 = math.floor(center_x - box_width / 2.0)
        bbox_y0 = math.floor(center_y - box_height / 2.0)
        bbox_center_x = bbox_x0 + box_width / 2.0
        bbox_center_y = bbox_y0 + box_height / 2.0
        center_offset = abs(
            (bbox_center_x - center_x) * line_direction[0]
            + (bbox_center_y - center_y) * line_direction[1]
        )
        if shape == "circle":
            shape_half_span = min(box_width, box_height) / 2.0
        else:
            shape_half_span = 0.5 * (
                abs(line_direction[0]) * box_width + abs(line_direction[1]) * box_height
            )
        return shape_half_span + center_offset

    while projected_half_span() > max_shape_half_span:
        if shape in {"circle", "square"}:
            if box_width <= 1 or box_height <= 1:
                return None
            box_width -= 1
            box_height -= 1
        elif (
            abs(line_direction[0]) * box_width >= abs(line_direction[1]) * box_height
            and box_width > 1
        ) or box_height <= 1:
            if box_width <= 1:
                return None
            box_width -= 1
        else:
            box_height -= 1

    x0 = math.floor(center_x - box_width / 2.0)
    y0 = math.floor(center_y - box_height / 2.0)
    color_category, color_rgb = _sample_color_category(
        rng,
        white_probability=DISTRACTOR_COLOR_PROBABILITIES["white"],
    )
    shape_spec = _sample_shape_spec(
        shape=shape,
        bbox_xyxy=(x0, y0, x0 + box_width, y0 + box_height),
        color_category=color_category,
        color_rgb=color_rgb,
    )
    shape_spec["kind"] = "shape"
    return shape_spec


def _clip_infinite_line_to_frame(
    point_xy: tuple[float, float],
    direction_xy: tuple[float, float],
    image_width: int,
    image_height: int,
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Return both frame-edge intersections of a line through an in-frame point."""
    lower = -math.inf
    upper = math.inf
    for coordinate, direction, maximum in (
        (point_xy[0], direction_xy[0], float(image_width)),
        (point_xy[1], direction_xy[1], float(image_height)),
    ):
        if abs(direction) < 1e-12:
            if not 0.0 <= coordinate <= maximum:
                raise ValueError("infinite line does not intersect the image frame")
            continue
        first = -coordinate / direction
        second = (maximum - coordinate) / direction
        lower = max(lower, min(first, second))
        upper = min(upper, max(first, second))

    if not math.isfinite(lower) or not math.isfinite(upper) or lower >= upper:
        raise ValueError("infinite line does not cross the image frame")
    start_xy = (
        point_xy[0] + lower * direction_xy[0],
        point_xy[1] + lower * direction_xy[1],
    )
    end_xy = (
        point_xy[0] + upper * direction_xy[0],
        point_xy[1] + upper * direction_xy[1],
    )
    return start_xy, end_xy


def sample_synthetic_overlay_plan(
    *,
    image_shape: tuple[int, int],
    primary_line_start_xy: tuple[float, float],
    primary_line_end_xy: tuple[float, float],
    rng: np.random.Generator,
    include_extra_lines: bool = True,
    include_arcs: bool = True,
    include_tint: bool = True,
    include_shapes: bool = True,
    include_center_occluder: bool = True,
    center_assets_root: str | Path = DEFAULT_CENTER_ASSETS_ROOT,
) -> dict[str, Any]:
    """Sample all nondeterministic foreground effects without drawing them."""
    image_height, image_width = image_shape
    if image_width <= 0 or image_height <= 0:
        raise ValueError("image dimensions must be positive")
    if not all(math.isfinite(value) for value in (*primary_line_start_xy, *primary_line_end_xy)):
        raise ValueError("primary-line endpoints must be finite")
    line_dx = primary_line_end_xy[0] - primary_line_start_xy[0]
    line_dy = primary_line_end_xy[1] - primary_line_start_xy[1]
    line_length = math.hypot(line_dx, line_dy)
    if line_length <= 0.0:
        raise ValueError("primary-line endpoints must be different")
    direction = (line_dx / line_length, line_dy / line_length)
    normal = (-direction[1], direction[0])
    short_side = min(image_width, image_height)

    tint = (
        _sample_tint_spec(image_width, image_height, rng)
        if include_tint and rng.random() < BACKGROUND_TINT_PROBABILITY
        else None
    )
    distractors_enabled = bool(rng.random() < DISTRACTOR_PROBABILITY)

    detached_lines: list[dict[str, Any]] = []
    if include_extra_lines and distractors_enabled:
        line_count = int(rng.choice(5, p=DISTRACTOR_LINE_COUNT_PROBABILITIES))
        center_x = image_width / 2.0
        center_y = image_height / 2.0
        diagonal = math.hypot(image_width, image_height)
        for _ in range(line_count):
            angle = float(rng.uniform(0.0, math.pi))
            line_direction = (math.cos(angle), math.sin(angle))
            line_normal = (-line_direction[1], line_direction[0])
            offset = float(rng.uniform(0.10, 0.22) * short_side)
            side = -1.0 if rng.random() < 0.5 else 1.0
            full_frame = rng.random() < DISTRACTOR_FULL_FRAME_LINE_PROBABILITY
            if full_frame:
                midpoint = (
                    center_x + side * offset * line_normal[0],
                    center_y + side * offset * line_normal[1],
                )
                start_xy, end_xy = _clip_infinite_line_to_frame(
                    midpoint, line_direction, image_width, image_height
                )
            else:
                along_offset = float(rng.uniform(-0.12, 0.12) * diagonal)
                segment_length = float(rng.uniform(0.15, 0.38) * diagonal)
                midpoint = (
                    center_x + along_offset * line_direction[0] + side * offset * line_normal[0],
                    center_y + along_offset * line_direction[1] + side * offset * line_normal[1],
                )
                start_xy = (
                    midpoint[0] - 0.5 * segment_length * line_direction[0],
                    midpoint[1] - 0.5 * segment_length * line_direction[1],
                )
                end_xy = (
                    midpoint[0] + 0.5 * segment_length * line_direction[0],
                    midpoint[1] + 0.5 * segment_length * line_direction[1],
                )
            color_category, color_rgb = _sample_color_category(
                rng,
                white_probability=DISTRACTOR_COLOR_PROBABILITIES["white"],
            )
            detached_lines.append(
                {
                    "start_xy": start_xy,
                    "end_xy": end_xy,
                    "angle_rad": angle,
                    "full_frame": full_frame,
                    "color_category": color_category,
                    "color_rgb": color_rgb,
                }
            )

    arcs: list[dict[str, Any]] = []
    arc_count = (
        int(rng.choice(4, p=ARC_COUNT_PROBABILITIES)) if include_arcs and distractors_enabled else 0
    )
    attached_endpoints: set[str] = set()
    for _ in range(arc_count):
        relation_choice = float(rng.random())
        endpoint_limit = ARC_RELATION_PROBABILITIES["endpoint"]
        orthogonal_limit = endpoint_limit + ARC_RELATION_PROBABILITIES["orthogonal_endpoint"]
        if relation_choice < endpoint_limit:
            relation = "endpoint"
        elif relation_choice < orthogonal_limit:
            relation = "orthogonal_endpoint"
        else:
            relation = "near_line"

        if relation in {"endpoint", "orthogonal_endpoint"}:
            available_endpoints = [
                name for name in ("start", "end") if name not in attached_endpoints
            ]
            if not available_endpoints:
                available_endpoints = ["start", "end"]
            endpoint_name = available_endpoints[int(rng.integers(0, len(available_endpoints)))]
            endpoint_xy = primary_line_start_xy if endpoint_name == "start" else primary_line_end_xy
            radius = float(rng.uniform(*ARC_RADIUS_SCALE_RANGE) * short_side)
            if relation == "orthogonal_endpoint":
                direction_sign = -1.0 if rng.random() < 0.5 else 1.0
                radial_direction = (
                    direction_sign * direction[0],
                    direction_sign * direction[1],
                )
                terminal_angle = math.atan2(radial_direction[1], radial_direction[0])
            else:
                terminal_angle = float(rng.uniform(0.0, 2.0 * math.pi))
            sweep_angle = float(rng.uniform(math.radians(90.0), math.radians(210.0)))
            arc_center = (
                endpoint_xy[0] - radius * math.cos(terminal_angle),
                endpoint_xy[1] - radius * math.sin(terminal_angle),
            )
            start_angle = terminal_angle - sweep_angle
            attached_endpoints.add(endpoint_name)
        else:
            line_fraction = float(rng.uniform(0.15, 0.85))
            line_point = (
                primary_line_start_xy[0] + line_fraction * line_dx,
                primary_line_start_xy[1] + line_fraction * line_dy,
            )
            radius = float(rng.uniform(*ARC_RADIUS_SCALE_RANGE) * short_side)
            gap = float(rng.uniform(2.0, 12.0))
            side = -1.0 if rng.random() < 0.5 else 1.0
            arc_center = (
                line_point[0] + side * (radius + gap) * normal[0],
                line_point[1] + side * (radius + gap) * normal[1],
            )
            closest_angle = math.atan2(line_point[1] - arc_center[1], line_point[0] - arc_center[0])
            sweep_angle = float(rng.uniform(math.radians(90.0), math.radians(210.0)))
            start_angle = closest_angle - sweep_angle / 2.0
            endpoint_name = None
            endpoint_xy = None

        color_category, color_rgb = _sample_color_category(
            rng,
            white_probability=DISTRACTOR_COLOR_PROBABILITIES["white"],
        )
        arcs.append(
            {
                "relation": relation,
                "endpoint_name": endpoint_name,
                "endpoint_xy": endpoint_xy,
                "center_xy": arc_center,
                "radius": radius,
                "start_angle_rad": start_angle,
                "sweep_angle_rad": sweep_angle,
                "color_category": color_category,
                "color_rgb": color_rgb,
            }
        )

    shape_count = (
        int(rng.integers(0, MAX_SHAPE_DISTRACTORS + 1))
        if include_shapes and distractors_enabled
        else 0
    )
    shape_distractors = [
        _sample_free_shape(image_width, image_height, rng) for _ in range(shape_count)
    ]
    center_occluder = None
    center_occluder_under_line = False
    if include_center_occluder and distractors_enabled:
        center_occluder = _sample_center_occluder(
            image_width,
            image_height,
            rng,
            primary_line_start_xy,
            primary_line_end_xy,
            center_assets_root,
        )
        if center_occluder is not None:
            center_occluder_under_line = rng.random() < CENTER_OCCLUDER_UNDER_LINE_PROBABILITY

    return {
        "distractors_enabled": distractors_enabled,
        "background_tint": tint,
        "detached_lines": detached_lines,
        "arcs": arcs,
        "shape_distractors": shape_distractors,
        "center_occluder": center_occluder,
        "center_occluder_under_line": center_occluder_under_line,
    }


def apply_background_tint(image: np.ndarray, tint_spec: dict[str, Any]) -> None:
    """Alpha-blend a sampled rectangular tint into an RGB background in place."""
    if image.ndim != 3 or image.shape[2] != 3 or image.dtype != np.uint8:
        raise ValueError("image must be a uint8 RGB array")
    x0, y0, x1, y1 = (int(value) for value in tint_spec["bbox_xyxy"])
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError("tint bbox must lie inside the image")
    alpha = float(tint_spec["alpha"])
    color = np.asarray(tint_spec["color_rgb"], dtype=np.float32)
    if not 0.0 <= alpha <= 1.0 or color.shape != (3,) or np.any(color < 0) or np.any(color > 255):
        raise ValueError("tint alpha and RGB color must be in range")
    region = image[y0:y1, x0:x1].astype(np.float32)
    image[y0:y1, x0:x1] = np.rint(region * (1.0 - alpha) + color * alpha).astype(np.uint8)


def _draw_filled_shape(image: np.ndarray, shape_spec: dict[str, Any]) -> None:
    x0, y0, x1, y1 = (float(value) for value in shape_spec["bbox_xyxy"])
    if not (0.0 <= x0 < x1 <= image.shape[1] and 0.0 <= y0 < y1 <= image.shape[0]):
        raise ValueError("shape bbox must lie inside the image")
    color = np.asarray(shape_spec["color_rgb"], dtype=np.uint8)
    if color.shape != (3,):
        raise ValueError("shape color must contain three RGB channels")
    pixel_y, pixel_x = np.ogrid[: image.shape[0], : image.shape[1]]
    pixel_x = pixel_x.astype(np.float32) + 0.5
    pixel_y = pixel_y.astype(np.float32) + 0.5
    if shape_spec["shape"] == "circle":
        center_x = (x0 + x1) / 2.0
        center_y = (y0 + y1) / 2.0
        radius_x = (x1 - x0) / 2.0
        radius_y = (y1 - y0) / 2.0
        mask = ((pixel_x - center_x) / radius_x) ** 2 + (
            (pixel_y - center_y) / radius_y
        ) ** 2 <= 1.0
    elif shape_spec["shape"] in {"square", "rectangle"}:
        mask = (pixel_x >= x0) & (pixel_x < x1) & (pixel_y >= y0) & (pixel_y < y1)
    else:
        raise ValueError(f"unsupported distractor shape: {shape_spec['shape']}")
    image[mask] = color


def _draw_center_occluder(image: np.ndarray, occluder_spec: dict[str, Any]) -> None:
    if occluder_spec.get("kind", "shape") != "sprite":
        _draw_filled_shape(image, occluder_spec)
        return

    x0, y0, x1, y1 = (int(value) for value in occluder_spec["bbox_xyxy"])
    if not (0 <= x0 < x1 <= image.shape[1] and 0 <= y0 < y1 <= image.shape[0]):
        raise ValueError("center sprite bbox must lie inside the image")
    sprite = _load_center_sprite_rgba(str(occluder_spec["asset_path"]))
    hue_shift = occluder_spec.get("color_shift_hue")
    if hue_shift is not None:
        sprite = shift_center_sprite_hue(sprite, float(hue_shift))
    target_size = (x1 - x0, y1 - y0)
    if sprite.shape[1::-1] != target_size:
        sprite_image = Image.fromarray(sprite, mode="RGBA").resize(
            target_size,
            resample=Image.Resampling.LANCZOS,
        )
        sprite = np.asarray(sprite_image, dtype=np.uint8)

    destination = image[y0:y1, x0:x1].astype(np.float32)
    source_rgb = sprite[:, :, :3].astype(np.float32)
    alpha = sprite[:, :, 3:4].astype(np.float32) / 255.0
    image[y0:y1, x0:x1] = np.rint(
        np.clip(source_rgb * alpha + destination * (1.0 - alpha), 0.0, 255.0)
    ).astype(np.uint8)


def draw_synthetic_distractors(
    image: np.ndarray,
    plan: dict[str, Any],
    *,
    draw_center_occluder: bool = True,
) -> None:
    """Draw distractor strokes and shapes; draw a center occluder last."""
    for line in plan.get("detached_lines", []):
        color_rgb = line.get("color_rgb", (255, 255, 255))
        color_rgba = (*(channel / 255.0 for channel in color_rgb), AIMING_LINE_ALPHA)
        draw_gl_smooth_segment(
            image,
            line["start_xy"],
            line["end_xy"],
            colour_rgba=color_rgba,
        )
    for arc in plan.get("arcs", []):
        color_rgb = arc.get("color_rgb", (255, 255, 255))
        color_rgba = (*(channel / 255.0 for channel in color_rgb), AIMING_LINE_ALPHA)
        draw_gl_smooth_arc(
            image,
            center_xy=arc["center_xy"],
            radius=arc["radius"],
            start_angle_rad=arc["start_angle_rad"],
            sweep_angle_rad=arc["sweep_angle_rad"],
            colour_rgba=color_rgba,
        )
    for shape in plan.get("shape_distractors", []):
        _draw_filled_shape(image, shape)
    center_occluder = plan.get("center_occluder")
    if draw_center_occluder and center_occluder is not None:
        _draw_center_occluder(image, center_occluder)


def draw_synthetic_aiming_overlay(
    image: np.ndarray,
    line_angle_rad: float,
    *,
    rng: np.random.Generator,
    include_extra_lines: bool = True,
    include_arcs: bool = True,
    include_tint: bool = True,
    include_shapes: bool = True,
    include_center_occluder: bool = True,
    center_assets_root: str | Path = DEFAULT_CENTER_ASSETS_ROOT,
) -> dict[str, Any]:
    """Draw the primary center line and a sampled set of visual distractors."""
    _validate_stroke(image, AIMING_LINE_WIDTH, AIMING_LINE_RGBA)
    if not math.isfinite(line_angle_rad):
        raise ValueError("line angle must be finite")
    if image.shape[0] <= 0 or image.shape[1] <= 0:
        raise ValueError("image dimensions must be positive")

    height, width = image.shape[:2]
    center_x = width / 2.0
    center_y = height / 2.0
    diagonal = math.hypot(width, height)
    direction_x, direction_y = angle_to_line_vector(line_angle_rad)
    total_length = float(rng.uniform(0.22, 1.30) * diagonal)
    negative_length = total_length * float(rng.uniform(0.38, 0.62))
    positive_length = total_length - negative_length
    center_start = (
        center_x - direction_x * negative_length,
        center_y - direction_y * negative_length,
    )
    center_end = (
        center_x + direction_x * positive_length,
        center_y + direction_y * positive_length,
    )
    plan = sample_synthetic_overlay_plan(
        image_shape=(height, width),
        primary_line_start_xy=center_start,
        primary_line_end_xy=center_end,
        rng=rng,
        include_extra_lines=include_extra_lines,
        include_arcs=include_arcs,
        include_tint=include_tint,
        include_shapes=include_shapes,
        include_center_occluder=include_center_occluder,
        center_assets_root=center_assets_root,
    )
    if plan["background_tint"] is not None:
        apply_background_tint(image, plan["background_tint"])
    center_occluder = plan["center_occluder"]
    under_line = plan["center_occluder_under_line"]
    draw_synthetic_distractors(image, plan, draw_center_occluder=False)
    if center_occluder is not None and under_line:
        _draw_center_occluder(image, center_occluder)
    draw_gl_smooth_segment(image, center_start, center_end)
    if center_occluder is not None and not under_line:
        _draw_center_occluder(image, center_occluder)
    return {"central_line": {"start_xy": center_start, "end_xy": center_end}, **plan}


def generate_synthetic_image(
    img_size: int = 256,
    rng: np.random.Generator | None = None,
    *,
    environment: BackgroundEnvironment | str | Path | None = None,
    environments_root: str | Path = DEFAULT_ENVIRONMENTS_ROOT,
    background_assets_root: str | Path = BACKGROUND_ASSETS_ROOT,
    center_assets_root: str | Path = DEFAULT_CENTER_ASSETS_ROOT,
    camera_focus: tuple[float, float] | None = None,
    camera_zoom: float | None = None,
    time_seconds: float | None = None,
    include_weather: bool | None = None,
    full_background_coverage: bool | None = None,
    include_extra_lines: bool = True,
    include_arcs: bool = True,
) -> tuple[np.ndarray, float]:
    """Create a sampled full-frame background and its aiming/shape overlays.

    Backgrounds use a layered environment, a generated color field, or a random
    256x256 crop from a user-provided image. Explicit ``environment`` values
    continue to force the layered environment renderer. The returned angle
    belongs to the primary center line in degrees in [0, 180).

    """
    if img_size <= 0:
        raise ValueError("img_size must be positive")
    if rng is None:
        rng = np.random.default_rng()
    background_source = (
        "environment_asset" if environment is not None else sample_background_source(rng)
    )
    if background_source == "procedural":
        background, _ = render_procedural_background(img_size, img_size, rng)
    elif background_source == "image_asset":
        background, _ = render_background_asset_crop(
            img_size,
            img_size,
            rng,
            assets_root=background_assets_root,
        )
    else:
        if full_background_coverage is None:
            full_background_coverage = sample_background_fill_mode(rng)

        if isinstance(environment, BackgroundEnvironment):
            selected_environment = environment
        elif environment is not None:
            selected_environment = load_environment(environment)
        else:
            environment_names = available_environment_names(environments_root)
            environment_name = environment_names[int(rng.integers(0, len(environment_names)))]
            selected_environment = load_environment(Path(environments_root) / environment_name)

        if camera_focus is None:
            camera_focus = _sample_background_camera_focus(selected_environment, rng)
        if camera_zoom is None:
            camera_zoom = float(rng.uniform(0.55, 1.45))
        if time_seconds is None:
            time_seconds = float(rng.uniform(0.0, 3600.0))
        if include_weather is None:
            include_weather = bool(rng.random() < 0.25)

        rendered_background = render_environment_background(
            selected_environment,
            img_size,
            camera_focus=camera_focus,
            camera_zoom=camera_zoom,
            time_seconds=time_seconds,
            include_weather=include_weather,
            return_coverage=full_background_coverage,
        )
        if full_background_coverage:
            assert isinstance(rendered_background, tuple)
            background, coverage_mask = rendered_background
            backdrop = render_environment_texture_backdrop(selected_environment, img_size, rng)
            background = fill_uncovered_background(background, coverage_mask, backdrop)
        else:
            assert isinstance(rendered_background, np.ndarray)
            background = rendered_background

    line_angle_degrees = float(rng.random() * 180.0)
    line_angle_rad = math.radians(line_angle_degrees)
    draw_synthetic_aiming_overlay(
        background,
        line_angle_rad,
        rng=rng,
        include_extra_lines=include_extra_lines,
        include_arcs=include_arcs,
        center_assets_root=center_assets_root,
    )

    return background, line_angle_degrees
