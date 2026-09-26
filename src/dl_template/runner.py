"""Resolve and run the configured experiment entrypoint."""

from __future__ import annotations

import importlib
from collections.abc import Callable
from pathlib import Path

from dl_template.config.schema import ExperimentConfig
from dl_template.tracking.mlflow import active_run


def run_experiment(config: ExperimentConfig, config_path: Path) -> str:
    """Run one configured Python entrypoint inside an MLflow run."""
    with active_run(config, config_path) as run:
        entrypoint = _load_entrypoint(config.entrypoint)
        entrypoint(config)
    return run.info.run_id


def _load_entrypoint(entrypoint: str) -> Callable[[ExperimentConfig], None]:
    module_name, separator, function_name = entrypoint.partition(":")
    if not separator or not module_name or not function_name:
        raise ValueError(f"Invalid experiment entrypoint {entrypoint!r}; expected module:function")

    module = importlib.import_module(module_name)
    function = getattr(module, function_name, None)
    if not callable(function):
        raise TypeError(f"Experiment entrypoint {entrypoint!r} is not callable")
    return function
