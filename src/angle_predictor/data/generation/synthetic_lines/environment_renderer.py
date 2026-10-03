from __future__ import annotations

import ast
import math
import re
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image

DEFAULT_ENVIRONMENTS_ROOT = Path(__file__).resolve().parent / "assets" / "environments"
DEFAULT_BACKGROUND_SIZE_SCALE = 1.05
_IMAGE_EXTENSIONS = (".dds", ".png", ".tga")


@dataclass(frozen=True)
class BackgroundTile:
    grid_x: int
    grid_y: int
    texture_path: Path
    clamp_t: bool = False


@dataclass(frozen=True)
class BackgroundLayer:
    name: str
    zoom_factor: float
    grid_size_x: float | None
    grid_size_y: float | None
    grid_tile_offset_x: float
    grid_tile_offset_y: float
    scroll_rate_x: float
    scroll_rate_y: float
    texture_scroll_rate_x: float
    texture_scroll_rate_y: float
    foreground: bool
    weather: bool
    align_top: bool
    colour: tuple[float, float, float, float]
    tiles: tuple[BackgroundTile, ...]


@dataclass(frozen=True)
class BackgroundEnvironment:
    name: str
    root: Path
    background_colour: tuple[float, float, float, float]
    layers: tuple[BackgroundLayer, ...]


class _LuaTableParser:
    """Parse the literal tables used by layered background.lua files."""

    _token_pattern = re.compile(
        r"""("(?:\\.|[^"\\])*"|'(?:\\.|[^'\\])*'|\.\.|[{}=,;]|"""
        r"""-?(?:\d+\.\d*|\.\d+|\d+)(?:[eE][+-]?\d+)?|[A-Za-z_][A-Za-z_0-9]*|\S)"""
    )
    _long_comment_pattern = re.compile(r"--\[(=*)\[.*?\]\1\]", re.S)

    def __init__(self, source: str) -> None:
        source = self._long_comment_pattern.sub("", source)
        source = re.sub(r"--[^\r\n]*", "", source)
        self.tokens = [token for token in self._token_pattern.findall(source) if token]
        self.index = 0

    def parse_assignments(self) -> dict[str, Any]:
        result: dict[str, Any] = {}
        while self.index < len(self.tokens):
            if (
                self.index + 1 < len(self.tokens)
                and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", self.tokens[self.index])
                and self.tokens[self.index + 1] == "="
            ):
                key = self.tokens[self.index]
                self.index += 2
                result[key] = self._parse_expression()
            else:
                self.index += 1
        return result

    def _parse_expression(self) -> Any:
        value = self._parse_atom()
        while self._peek() == "..":
            self.index += 1
            rhs = self._parse_atom()
            value = f"{value or ''}{rhs or ''}"
        return value

    def _parse_atom(self) -> Any:
        token = self._peek()
        if token is None:
            raise ValueError("Unexpected end of Lua config")
        if token == "{":
            return self._parse_table()

        self.index += 1
        if token[0:1] in ('"', "'"):
            return ast.literal_eval(token)
        if token == "true":
            return True
        if token == "false":
            return False
        if token == "nil":
            return None
        if token == "bgpath":
            return ""
        try:
            number = float(token)
        except ValueError:
            return token
        return int(number) if number.is_integer() else number

    def _parse_table(self) -> list[Any] | dict[str, Any]:
        self.index += 1  # opening brace
        sequence: list[Any] = []
        mapping: dict[str, Any] = {}

        while self._peek() not in (None, "}"):
            if (
                self.index + 1 < len(self.tokens)
                and re.fullmatch(r"[A-Za-z_][A-Za-z_0-9]*", self.tokens[self.index])
                and self.tokens[self.index + 1] == "="
            ):
                key = self.tokens[self.index]
                self.index += 2
                mapping[key] = self._parse_expression()
            else:
                sequence.append(self._parse_expression())

            if self._peek() in (",", ";"):
                self.index += 1
            elif self._peek() != "}":
                raise ValueError(f"Expected ',' or '}}' in Lua table, got {self._peek()!r}")

        if self._peek() != "}":
            raise ValueError("Unterminated Lua table")
        self.index += 1
        if mapping and sequence:
            mapping.update({str(index + 1): value for index, value in enumerate(sequence)})
            return mapping
        return mapping if mapping else sequence

    def _peek(self) -> str | None:
        if self.index >= len(self.tokens):
            return None
        return self.tokens[self.index]


def _as_float(value: Any, default: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return default
    return float(value)


def _as_bool(value: Any, default: bool = False) -> bool:
    return value if isinstance(value, bool) else default


def _normalise_colour(
    value: Any, default: tuple[float, float, float, float]
) -> tuple[float, float, float, float]:
    if not isinstance(value, list) or len(value) < 3:
        return default
    components = list(value[:4])
    if len(components) == 3:
        components.append(255)
    normalised = []
    for component in components:
        channel = _as_float(component, 255.0)
        if channel > 1.0:
            channel /= 255.0
        normalised.append(float(np.clip(channel, 0.0, 1.0)))
    return normalised[0], normalised[1], normalised[2], normalised[3]


def _resolve_texture(background_dir: Path, texture_name: str) -> Path:
    texture_name = texture_name.replace("\\", "/")
    requested = (background_dir / texture_name).resolve()
    if requested.is_file():
        return requested

    # The installed Lua files include a couple of .png names whose files are
    # packaged as .dds and resolved by filename stem
    matches = [
        (background_dir / f"{Path(texture_name).stem}{extension}").resolve()
        for extension in _IMAGE_EXTENSIONS
        if (background_dir / f"{Path(texture_name).stem}{extension}").is_file()
    ]
    if matches:
        return matches[0]
    raise FileNotFoundError(f"Background texture not found: {requested}")


@lru_cache(maxsize=32)
def _load_environment_cached(config_path: str) -> BackgroundEnvironment:
    config = Path(config_path)
    background_dir = config.parent
    root = background_dir.parent
    parsed = _LuaTableParser(config.read_text(encoding="utf-8-sig")).parse_assignments()

    background_colour = _normalise_colour(parsed.get("BackgroundColour"), (0.0, 0.0, 0.0, 1.0))
    raw_layers = parsed.get("Layers", [])
    if not isinstance(raw_layers, list):
        raise ValueError(f"Layers must be a Lua array in {config}")

    layers: list[BackgroundLayer] = []
    for layer_index, raw_layer in enumerate(raw_layers):
        if not isinstance(raw_layer, dict):
            continue
        raw_tiles = raw_layer.get("Tiles", [])
        if not isinstance(raw_tiles, list):
            raise ValueError(f"Layer {layer_index} Tiles must be a Lua array in {config}")
        tiles: list[BackgroundTile] = []
        for raw_tile in raw_tiles:
            if not isinstance(raw_tile, dict):
                continue
            texture_name = raw_tile.get("TextureFileName")
            if not isinstance(texture_name, str) or not texture_name:
                continue
            tiles.append(
                BackgroundTile(
                    grid_x=int(_as_float(raw_tile.get("GridX"), 0.0)),
                    grid_y=int(_as_float(raw_tile.get("GridY"), 0.0)),
                    texture_path=_resolve_texture(background_dir, texture_name),
                    clamp_t=_as_bool(raw_tile.get("ClampT")),
                )
            )

        if not tiles:
            continue
        colour = _normalise_colour(raw_layer.get("Colour"), (1.0, 1.0, 1.0, 1.0))
        name = raw_layer.get("Name")
        layers.append(
            BackgroundLayer(
                name=name if isinstance(name, str) else f"layer_{layer_index}",
                zoom_factor=_as_float(raw_layer.get("ZoomFactor"), 1.0),
                grid_size_x=(
                    _as_float(raw_layer.get("GridSizeX"), 0.0)
                    if raw_layer.get("GridSizeX") is not None
                    else None
                ),
                grid_size_y=(
                    _as_float(raw_layer.get("GridSizeY"), 0.0)
                    if raw_layer.get("GridSizeY") is not None
                    else None
                ),
                grid_tile_offset_x=_as_float(raw_layer.get("GridTileOffsetX"), 0.0),
                grid_tile_offset_y=_as_float(raw_layer.get("GridTileOffsetY"), 0.0),
                scroll_rate_x=_as_float(raw_layer.get("ScrollRateX"), 0.0),
                scroll_rate_y=_as_float(raw_layer.get("ScrollRateY"), 0.0),
                texture_scroll_rate_x=_as_float(raw_layer.get("TextureScrollRateX"), 0.0),
                texture_scroll_rate_y=_as_float(raw_layer.get("TextureScrollRateY"), 0.0),
                foreground=_as_bool(raw_layer.get("Foreground")),
                weather=_as_bool(raw_layer.get("Weather")),
                align_top=_as_bool(raw_layer.get("AlignTop")),
                colour=colour,
                tiles=tuple(tiles),
            )
        )

    if not layers:
        raise ValueError(f"No background layers with textures found in {config}")
    return BackgroundEnvironment(root.name, root, background_colour, tuple(layers))


def load_environment(environment: str | Path) -> BackgroundEnvironment:
    """Load one environment by name, environment directory, or Lua file path."""
    path = Path(environment)
    if not path.exists():
        path = DEFAULT_ENVIRONMENTS_ROOT / str(environment)
    if path.is_dir() and path.name == "background":
        config = path / "background.lua"
    elif path.is_dir() and (path / "background" / "background.lua").is_file():
        config = path / "background" / "background.lua"
    elif path.is_dir() and (path / "background.lua").is_file():
        config = path / "background.lua"
    else:
        config = path
    if not config.is_file():
        raise FileNotFoundError(f"Background config not found: {config}")
    return _load_environment_cached(str(config.resolve()))


def available_environment_names(
    environments_root: str | Path = DEFAULT_ENVIRONMENTS_ROOT,
) -> tuple[str, ...]:
    root = Path(environments_root)
    if not root.is_dir():
        raise FileNotFoundError(f"Environment assets directory not found: {root}")
    names = tuple(
        sorted(
            path.name
            for path in root.iterdir()
            if path.is_dir() and (path / "background" / "background.lua").is_file()
        )
    )
    if not names:
        raise FileNotFoundError(f"No environment background.lua files found in {root}")
    return names


@lru_cache(maxsize=128)
def _load_texture(texture_path: str) -> np.ndarray:
    with Image.open(texture_path) as source:
        rgba = np.asarray(source.convert("RGBA"), dtype=np.uint8).copy()
    rgba.setflags(write=False)
    return rgba


@lru_cache(maxsize=32)
def _load_texture_float32(texture_path: str) -> np.ndarray:
    # Remapping uint8 would round each sampled channel before alpha blending.
    texture = _load_texture(texture_path).astype(np.float32)
    texture.setflags(write=False)
    return texture


def render_environment_texture_backdrop(
    environment: BackgroundEnvironment | str | Path,
    img_size: int | tuple[int, int],
    rng: np.random.Generator,
) -> np.ndarray:
    """Resize a random scenery-layer tile for uncovered full-frame regions."""
    if isinstance(img_size, int):
        width = height = img_size
    else:
        width, height = img_size
    if width <= 0 or height <= 0:
        raise ValueError("img_size dimensions must be positive")
    if not isinstance(environment, BackgroundEnvironment):
        environment = load_environment(environment)

    scenery_layers = [
        layer
        for layer in environment.layers
        if not layer.foreground and not layer.weather and layer.tiles
    ]
    if not scenery_layers:
        scenery_layers = [layer for layer in environment.layers if layer.tiles]
    if not scenery_layers:
        base = np.asarray(environment.background_colour[:3], dtype=np.float32) * 255.0
        return np.broadcast_to(base, (height, width, 3)).round().astype(np.uint8).copy()

    layer = scenery_layers[int(rng.integers(0, len(scenery_layers)))]
    tile = layer.tiles[int(rng.integers(0, len(layer.tiles)))]
    texture = _load_texture(str(tile.texture_path))
    resized = np.asarray(
        Image.fromarray(texture, mode="RGBA").resize(
            (width, height), resample=Image.Resampling.LANCZOS
        ),
        dtype=np.uint8,
    )
    tint = np.asarray(layer.colour, dtype=np.float32)
    source_rgb = resized[..., :3].astype(np.float32) * tint[:3]
    source_alpha = resized[..., 3:4].astype(np.float32) / 255.0 * tint[3]
    base_rgb = np.asarray(environment.background_colour[:3], dtype=np.float32) * 255.0
    backdrop = source_rgb * source_alpha + base_rgb * (1.0 - source_alpha)
    return np.rint(np.clip(backdrop, 0.0, 255.0)).astype(np.uint8)


def _sample_texture(texture: np.ndarray, u: np.ndarray, v: np.ndarray, clamp_t: bool) -> np.ndarray:
    height, width = texture.shape[:2]
    map_x = np.broadcast_to((u * width - 0.5).astype(np.float32)[None, :], (v.size, u.size)).copy()
    map_y = np.broadcast_to((v * height - 0.5).astype(np.float32)[:, None], (v.size, u.size)).copy()
    if clamp_t:
        # remap has one border mode, so clamp only Y before wrapping X.
        np.clip(map_y, 0, height - 1, out=map_y)
    return cv2.remap(
        texture.astype(np.float32, copy=False),
        map_x,
        map_y,
        interpolation=cv2.INTER_LINEAR,
        borderMode=cv2.BORDER_WRAP,
    )


def _composite_tile(
    canvas: np.ndarray,
    layer: BackgroundLayer,
    tile: BackgroundTile,
    *,
    camera_focus: tuple[float, float],
    camera_zoom: float,
    time_seconds: float,
    layer_scale: float,
    grid_size_x: float,
    grid_size_y: float,
    grid_offset_x: float,
    grid_offset_y: float,
    coverage_mask: np.ndarray | None = None,
) -> None:
    height, width = canvas.shape[:2]
    scale = layer_scale / camera_zoom
    tile_world_x = tile.grid_x * grid_size_x + grid_offset_x * grid_size_x
    tile_world_y = tile.grid_y * grid_size_y + grid_offset_y * grid_size_y

    x0 = scale * (tile_world_x - camera_focus[0] * layer.scroll_rate_x) + width / 2.0
    x1 = scale * (tile_world_x + grid_size_x - camera_focus[0] * layer.scroll_rate_x) + width / 2.0
    y0 = scale * (tile_world_y - camera_focus[1] * layer.scroll_rate_y) + height / 2.0
    y1 = scale * (tile_world_y + grid_size_y - camera_focus[1] * layer.scroll_rate_y) + height / 2.0

    left = max(0, math.floor(min(x0, x1)))
    right = min(width, math.ceil(max(x0, x1)))
    top = max(0, math.floor(min(y0, y1)))
    bottom = min(height, math.ceil(max(y0, y1)))
    if left >= right or top >= bottom:
        return

    pixel_x = np.arange(left, right, dtype=np.float32) + 0.5
    pixel_y = np.arange(top, bottom, dtype=np.float32) + 0.5
    u = (pixel_x - x0) / (x1 - x0)
    v = (pixel_y - y0) / (y1 - y0)
    u_phase = _fractional_scroll(time_seconds, layer.texture_scroll_rate_x, grid_size_x)
    v_phase = _fractional_scroll(time_seconds, layer.texture_scroll_rate_y, grid_size_y)
    texture = _load_texture_float32(str(tile.texture_path))
    sampled = _sample_texture(texture, u + u_phase, v + v_phase, tile.clamp_t)

    tint = np.asarray(layer.colour, dtype=np.float32)
    source_alpha = sampled[..., 3] / 255.0 * tint[3]
    visible = source_alpha >= 0.02  # Native GL alpha-test threshold.
    if not np.any(visible):
        return
    source_rgb = sampled[..., :3] / 255.0 * tint[:3]
    destination = canvas[top:bottom, left:right]
    blended = source_rgb * source_alpha[..., None] + destination * (1.0 - source_alpha[..., None])
    destination[visible] = blended[visible]
    if coverage_mask is not None:
        coverage_mask[top:bottom, left:right] |= visible


def _fractional_scroll(time_seconds: float, rate: float, grid_size: float) -> float:
    if rate == 0.0 or grid_size == 0.0:
        return 0.0
    offset = time_seconds * rate / grid_size
    # C++ float-to-int truncates toward zero; Python's modulo/floor differs for
    # negative scroll, so use the native fractional-part convention.
    return offset - math.trunc(offset)


def render_environment_background(
    environment: BackgroundEnvironment | str | Path,
    img_size: int | tuple[int, int],
    *,
    camera_focus: tuple[float, float] = (0.0, 0.0),
    camera_zoom: float = 1.0,
    time_seconds: float = 0.0,
    include_weather: bool = False,
    return_coverage: bool = False,
) -> np.ndarray | tuple[np.ndarray, np.ndarray]:
    """Render one RGB background at a camera position and elapsed scene time.

    ``img_size`` is an integer for a square viewport, or ``(width, height)``.
    Layers are composed in original Lua order, with the background pass before
    its foreground pass. Weather layers are skipped unless explicitly enabled.
    """
    if isinstance(img_size, int):
        width = height = img_size
    else:
        width, height = img_size
    if width <= 0 or height <= 0:
        raise ValueError("img_size dimensions must be positive")
    if not math.isfinite(camera_zoom) or camera_zoom <= 0:
        raise ValueError("camera_zoom must be a finite positive number")
    if not math.isfinite(time_seconds):
        raise ValueError("time_seconds must be finite")

    if not isinstance(environment, BackgroundEnvironment):
        environment = load_environment(environment)

    canvas = np.empty((height, width, 3), dtype=np.float32)
    canvas[:] = np.asarray(environment.background_colour[:3], dtype=np.float32)
    coverage_mask = np.zeros((height, width), dtype=bool) if return_coverage else None

    for foreground_pass in (False, True):
        for layer in environment.layers:
            if layer.foreground != foreground_pass or (layer.weather and not include_weather):
                continue
            layer_scale = 1.0 + (1.0 - layer.zoom_factor) * (camera_zoom - 1.0)
            if layer_scale <= 0:
                continue
            default_grid_scale = camera_zoom / layer_scale * DEFAULT_BACKGROUND_SIZE_SCALE
            grid_size_x = layer.grid_size_x or width * default_grid_scale
            grid_size_y = layer.grid_size_y or height * default_grid_scale
            if grid_size_x <= 0 or grid_size_y <= 0:
                continue
            grid_offset_x = layer.grid_tile_offset_x
            grid_offset_y = layer.grid_tile_offset_y
            if layer.align_top:
                # Align the first row's upper edge to the viewport top while
                # retaining the layer's parallax response to camera motion.
                grid_offset_y = camera_focus[
                    1
                ] * layer.scroll_rate_y / grid_size_y - height * camera_zoom / (
                    2.0 * layer_scale * grid_size_y
                )

            for tile in layer.tiles:
                _composite_tile(
                    canvas,
                    layer,
                    tile,
                    camera_focus=camera_focus,
                    camera_zoom=camera_zoom,
                    time_seconds=time_seconds,
                    layer_scale=layer_scale,
                    grid_size_x=grid_size_x,
                    grid_size_y=grid_size_y,
                    grid_offset_x=grid_offset_x,
                    grid_offset_y=grid_offset_y,
                    coverage_mask=coverage_mask,
                )

    rendered = np.rint(np.clip(canvas, 0.0, 1.0) * 255.0).astype(np.uint8)
    if coverage_mask is not None:
        return rendered, coverage_mask
    return rendered
