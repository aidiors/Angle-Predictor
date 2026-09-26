"""Shared MLflow tracking utilities."""

from __future__ import annotations

import hashlib
import json
import os
import platform
import subprocess
import tempfile
from collections.abc import Iterator
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import torch
import yaml
from dotenv import load_dotenv
from mlflow.data.numpy_dataset import from_numpy

from dl_template.config.schema import DatasetConfig, ExperimentConfig, TrackingConfig

DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"


def resolve_tracking_uri(config: TrackingConfig) -> str:
    """Resolve the explicit URI, environment variable, or local default."""
    return config.uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


@contextmanager
def active_run(config: ExperimentConfig, config_path: Path) -> Iterator[mlflow.ActiveRun]:
    """Start an MLflow run and record reproducibility metadata."""
    load_dotenv(override=False)
    tracking_uri = resolve_tracking_uri(config.tracking)
    mlflow.set_tracking_uri(tracking_uri)
    mlflow.set_experiment(config.tracking.experiment_name)

    with mlflow.start_run(
        run_name=config.name,
        log_system_metrics=config.tracking.system_metrics,
    ) as run:
        effective = config.model_dump(mode="json")
        mlflow.set_tags(_run_tags(config, config_path, effective))
        mlflow.log_params(_flatten_params(effective))
        mlflow.log_text(
            yaml.safe_dump(effective, sort_keys=False, allow_unicode=True),
            "config/effective.yaml",
        )
        yield run


def log_tensor_dataset_input(
    config: DatasetConfig,
    *,
    name: str,
    context: str,
    features: torch.Tensor,
    targets: torch.Tensor,
) -> None:
    """Log tensor dataset lineage without coupling data storage to MLflow."""
    dataset = from_numpy(
        features.detach().cpu().numpy(),
        source=config.source,
        name=name,
        targets=targets.detach().cpu().numpy(),
    )
    mlflow.log_input(dataset, context=context, tags={"version": config.version})
    if context == "source":
        mlflow.set_tag("dataset.digest", dataset.digest)


def log_tensor_dataset_snapshot(features: torch.Tensor, targets: torch.Tensor) -> None:
    """Log a small dataset snapshot; avoid this for real large datasets."""
    with tempfile.TemporaryDirectory() as temp_dir:
        snapshot_path = Path(temp_dir) / "source.pt"
        torch.save({"features": features.cpu(), "targets": targets.cpu()}, snapshot_path)
        mlflow.log_artifact(str(snapshot_path), artifact_path="datasets")


def log_params(params: dict[str, str]) -> None:
    mlflow.log_params(params)


def log_metrics(metrics: dict[str, float], *, step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_checkpoint(path: Path) -> None:
    mlflow.log_artifact(str(path), artifact_path="checkpoints")


def log_pytorch_model(model: torch.nn.Module, input_example: Any) -> None:
    """Log a PyTorch model with an inference signature and example input."""
    input_tensor = torch.from_numpy(input_example)
    model_device = next(model.parameters(), input_tensor).device
    model_input = input_tensor.to(model_device)
    with torch.inference_mode():
        output_example = model(model_input).detach().cpu().numpy()
    signature = mlflow.models.infer_signature(input_example, output_example)
    mlflow.pytorch.log_model(
        model,
        name="model",
        signature=signature,
        input_example=model_input,
    )


def _run_tags(
    config: ExperimentConfig,
    config_path: Path,
    effective_config: dict[str, Any],
) -> dict[str, str]:
    tags = {
        "project": "dl-ml-template",
        "dataset.name": config.dataset.name,
        "dataset.version": config.dataset.version,
        "config.path": config_path.as_posix(),
        "config.sha256": _config_digest(effective_config),
        "runtime.python": platform.python_version(),
        "runtime.torch": torch.__version__,
        "runtime.cuda": torch.version.cuda or "none",
    }
    commit = _git_value("rev-parse", "HEAD")
    if commit:
        tags["code.git_commit"] = commit
    tags["code.git_dirty"] = str(bool(_git_value("status", "--porcelain"))).lower()
    return tags


def _git_value(*arguments: str) -> str:
    try:
        result = subprocess.run(
            ["git", *arguments],
            check=True,
            capture_output=True,
            cwd=Path.cwd(),
            text=True,
        )
    except FileNotFoundError, subprocess.CalledProcessError:
        return ""
    return result.stdout.strip()


def _config_digest(values: dict[str, Any]) -> str:
    canonical = json.dumps(values, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _flatten_params(values: dict[str, Any], prefix: str = "") -> dict[str, str]:
    flattened: dict[str, str] = {}
    for key, value in values.items():
        full_key = f"{prefix}.{key}" if prefix else key
        if isinstance(value, dict):
            flattened.update(_flatten_params(value, full_key))
        else:
            rendered = json.dumps(value, ensure_ascii=False) if value is not None else "null"
            # The full effective config is stored as an artifact; params stay searchable and short.
            if len(rendered) <= 500:
                flattened[full_key] = rendered
    return flattened
