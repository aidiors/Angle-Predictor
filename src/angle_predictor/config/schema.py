from __future__ import annotations

from typing import Any

from pydantic import BaseModel, ConfigDict, Field, field_validator


class TrackingConfig(BaseModel):
    """MLflow settings shared by every run."""

    model_config = ConfigDict(extra="forbid")

    uri: str | None = None
    experiment_name: str = "angle-predictor"
    system_metrics: bool = True
    registered_model_name: str | None = "angle-predictor"

    @field_validator("registered_model_name")
    @classmethod
    def validate_registered_model_name(cls, value: str | None) -> str | None:
        if value is not None and not value.strip():
            raise ValueError("registered_model_name must not be empty")
        return value


class DatasetConfig(BaseModel):
    """Dataset identity and lineage metadata, not the dataset contents."""

    model_config = ConfigDict(extra="forbid")

    name: str
    version: str
    source: str
    log_snapshot: bool = False


class ExperimentConfig(BaseModel):
    """Common run settings passed to a task-specific experiment entrypoint."""

    model_config = ConfigDict(extra="forbid")

    name: str
    entrypoint: str
    seed: int = 42
    tracking: TrackingConfig = Field(default_factory=TrackingConfig)
    dataset: DatasetConfig
    params: dict[str, Any] = Field(default_factory=dict)
