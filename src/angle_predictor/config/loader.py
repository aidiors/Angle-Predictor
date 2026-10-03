from __future__ import annotations

import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv

from angle_predictor.config.schema import ExperimentConfig


def load_config(path: Path, overrides: list[str] | None = None) -> ExperimentConfig:
    """Load YAML and apply ``--set dotted.key=value`` overrides."""
    load_dotenv(override=False)
    raw_config = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw_config, dict):
        raise TypeError(f"Expected a YAML mapping in {path}")

    _apply_local_storage_overrides(raw_config)

    for override in overrides or []:
        _apply_override(raw_config, override)

    return ExperimentConfig.model_validate(raw_config)


def _apply_local_storage_overrides(config: dict[str, Any]) -> None:
    output_dir = os.environ.get("ANGLE_OUTPUT_DIR")
    params = config.get("params")
    if not output_dir or not isinstance(params, dict):
        return
    training = params.get("training")
    if isinstance(training, dict):
        training["output_dir"] = output_dir


def _apply_override(config: dict[str, Any], override: str) -> None:
    key, separator, raw_value = override.partition("=")
    segments = key.split(".")
    if not separator or not key or any(not segment for segment in segments):
        raise ValueError(f"Invalid override {override!r}; expected dotted.key=value")

    destination = config
    for segment in segments[:-1]:
        value = destination.get(segment)
        if not isinstance(value, dict):
            raise TypeError(f"Cannot apply {override!r}: {segment!r} is not a mapping")
        destination = value

    destination[segments[-1]] = yaml.safe_load(raw_value)
