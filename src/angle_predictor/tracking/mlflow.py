from __future__ import annotations

import hashlib
import json
import os
import platform
import shutil
import subprocess
import tempfile
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from pathlib import Path
from typing import Any

import mlflow
import mlflow.pytorch
import numpy as np
import torch
import yaml
from dotenv import load_dotenv
from mlflow.data.numpy_dataset import from_numpy
from mlflow.tracking import MlflowClient

from angle_predictor.config.schema import DatasetConfig, ExperimentConfig, TrackingConfig

DEFAULT_TRACKING_URI = "http://127.0.0.1:5001"


def resolve_tracking_uri(config: TrackingConfig) -> str:
    """Resolve the explicit URI, environment variable, or local default."""
    return config.uri or os.environ.get("MLFLOW_TRACKING_URI", DEFAULT_TRACKING_URI)


@contextmanager
def active_run(config: ExperimentConfig, config_path: Path) -> Iterator[mlflow.ActiveRun]:
    """Start an MLflow run and record reproducibility metadata."""
    load_dotenv(override=False)
    os.environ["MLFLOW_SUPPRESS_PRINTING_URL_TO_STDOUT"] = "true"
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


def log_params(params: Mapping[str, Any]) -> None:
    mlflow.log_params(_flatten_params(dict(params)))


def log_metrics(metrics: dict[str, float], *, step: int | None = None) -> None:
    mlflow.log_metrics(metrics, step=step)


def log_checkpoint(path: Path) -> None:
    mlflow.log_artifact(str(path), artifact_path="checkpoints")


def log_pytorch_model(
    model: torch.nn.Module,
    input_example: np.ndarray,
    *,
    registered_model_name: str | None = None,
    model_version_tags: Mapping[str, Any] | None = None,
) -> str | None:
    """Log a self-contained PyTorch model and optionally register a version."""
    input_tensor = torch.from_numpy(input_example)
    model_device = next(model.parameters(), input_tensor).device
    model_input = input_tensor.to(model_device)
    training_modes = [(module, module.training) for module in model.modules()]
    model.eval()
    try:
        with torch.inference_mode():
            output_example = model(model_input).detach().cpu().numpy()
        signature = mlflow.models.infer_signature(input_example, output_example)
        with tempfile.TemporaryDirectory(prefix="angle-predictor-model-code-") as temp_dir:
            code_path = _write_model_code_bundle(Path(temp_dir))
            model_info = mlflow.pytorch.log_model(
                model,
                name="model",
                signature=signature,
                input_example=input_example,
                code_paths=[str(code_path)],
                registered_model_name=registered_model_name,
                serialization_format="pickle",
            )
    finally:
        for module, was_training in training_modes:
            module.training = was_training

    if registered_model_name is None:
        return None
    version = getattr(model_info, "registered_model_version", None)
    if version is None:
        raise RuntimeError(
            f"MLflow did not return a registered version for {registered_model_name!r}"
        )
    version = str(version)
    client = MlflowClient()
    for key, value in (model_version_tags or {}).items():
        client.set_model_version_tag(registered_model_name, version, key, str(value))
    client.set_registered_model_alias(registered_model_name, "candidate", version)
    mlflow.set_tags(
        {
            "model.registry_name": registered_model_name,
            "model.version": version,
            "model.alias": "candidate",
        }
    )
    return version


def _write_model_code_bundle(destination: Path) -> Path:
    package_root = Path(__file__).resolve().parents[1]
    package = destination / "angle_predictor"
    # Include transitive imports of the model, such as config, losses and optimizers.
    source_files = sorted(
        path.relative_to(package_root)
        for path in package_root.rglob("*.py")
        if "generation" not in path.relative_to(package_root).parts
    )
    for source_file in source_files:
        target = package / source_file
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(package_root / source_file, target)
    return package


def _run_tags(
    config: ExperimentConfig,
    config_path: Path,
    effective_config: dict[str, Any],
) -> dict[str, str]:
    tags = {
        "project": "angle-predictor",
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
            rendered = value if isinstance(value, str) else json.dumps(value, ensure_ascii=False)
            if len(rendered) <= 500:
                flattened[full_key] = rendered
    return flattened
