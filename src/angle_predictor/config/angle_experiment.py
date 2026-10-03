from __future__ import annotations

from math import prod
from pathlib import Path
from typing import Annotated, Literal

from pydantic import BaseModel, ConfigDict, Field, model_validator

from angle_predictor.engine.optimizers import AdamWConfig, MuonAdamWConfig, UltralyticsMuSGDConfig
from angle_predictor.engine.optimizers.factory import OptimizerConfig
from angle_predictor.losses.angle import AngleLossKind


class AngleDataConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    root: Path
    batch_size: int = Field(default=32, gt=0)
    num_workers: int = Field(default=4, ge=0)
    split_seed: int = Field(default=42, ge=0)
    train_fraction: float = Field(default=0.8, gt=0, lt=1)
    val_fraction: float = Field(default=0.1, gt=0, lt=1)

    @model_validator(mode="after")
    def validate_splits(self) -> AngleDataConfig:
        if self.train_fraction + self.val_fraction >= 1:
            raise ValueError("train_fraction + val_fraction must be below 1")
        return self


class AngleModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["convnext"] = "convnext"
    backbone_name: str = "convnext_tiny"
    pretrained: bool = True
    out_index: int = Field(default=3, ge=0)
    final_stage_blocks: int | None = Field(default=None, gt=0)
    penultimate_stage_blocks: int | None = Field(default=None, gt=0, le=9)
    head_hidden: int = Field(default=256, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    residual_scale: float = Field(default=0.25, ge=0)
    normalize_output: bool = True

    @model_validator(mode="after")
    def validate_stage_pruning(self) -> AngleModelConfig:
        if self.penultimate_stage_blocks is not None and (
            self.backbone_name != "convnext_tiny" or self.out_index not in {2, 3}
        ):
            raise ValueError(
                "penultimate_stage_blocks requires convnext_tiny with out_index=2 or 3"
            )
        if self.final_stage_blocks is not None and (
            self.backbone_name != "convnext_tiny" or self.out_index != 3
        ):
            raise ValueError("final_stage_blocks requires convnext_tiny with out_index=3")
        if self.final_stage_blocks is not None and self.final_stage_blocks > 3:
            raise ValueError("convnext_tiny final stage has only three blocks")
        return self


class PolarLineModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["polar_line"] = "polar_line"
    channels: tuple[int, ...] = (16, 32, 64)
    blocks_per_stage: int = Field(default=2, gt=0)
    head_hidden: int = Field(default=128, gt=0)
    normalize_output: bool = True

    @model_validator(mode="after")
    def validate_channels(self) -> PolarLineModelConfig:
        if not self.channels or any(channel <= 0 for channel in self.channels):
            raise ValueError("channels must be a nonempty sequence of positive integers")
        return self


class AnglePreprocessingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    projection: Literal["signed_polar", "cartesian"] = "signed_polar"
    output_size: tuple[int, int] = (384, 360)
    mean: tuple[float, float, float] = (0.5, 0.5, 0.5)
    std: tuple[float, float, float] = (0.5, 0.5, 0.5)
    horizontal_flip_probability: float = Field(default=0.5, ge=0, le=1)
    vertical_flip_probability: float = Field(default=0.5, ge=0, le=1)
    jitter_probability: float = Field(default=0.7, ge=0, le=1)
    gaussian_probability: float = Field(default=0.3, ge=0, le=1)
    speckle_probability: float = Field(default=0.15, ge=0, le=1)


class AngleLossConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: AngleLossKind = "vector_charbonnier"
    smooth_l1_beta: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    charbonnier_epsilon: float = Field(default=0.1, gt=0, allow_inf_nan=False)
    angular_huber_beta_deg: float = Field(default=0.1, gt=0, allow_inf_nan=False)


class PolarRefinementModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["polar_refinement"] = "polar_refinement"
    coarse_model: AngleModelConfig
    initialization: Literal["checkpoint", "imagenet"] = "checkpoint"
    coarse_checkpoint: Path | None = None
    coarse_run_id: str | None = Field(default=None, min_length=32, max_length=32)
    coarse_split_seed: int | None = Field(default=None, ge=0)
    crop_size: tuple[int, int] = (192, 33)
    window_deg: float = Field(default=3.0, gt=0, lt=90, allow_inf_nan=False)
    fine_channels: tuple[int, ...] = (16, 32, 32)
    fine_angular_kernel: int = Field(default=5, gt=0)
    fine_radial_kernel: int = Field(default=5, gt=0)
    fine_radial_strides: tuple[Literal[1, 2], ...] | None = None
    radial_pool_bins: int = Field(default=1, gt=0)
    fine_radial_antialias: bool = False
    refinement_hidden: int = Field(default=128, gt=0)
    refinement_head: Literal["mlp", "local_mlp", "heatmap"] = "mlp"
    detach_crop_angle: bool = False
    train_coarse: bool = False

    @model_validator(mode="after")
    def validate_refinement(self) -> PolarRefinementModelConfig:
        lineage = (self.coarse_checkpoint, self.coarse_run_id, self.coarse_split_seed)
        if self.initialization == "checkpoint" and any(value is None for value in lineage):
            raise ValueError("checkpoint refinement requires complete coarse lineage")
        if self.initialization == "imagenet":
            if any(value is not None for value in lineage):
                raise ValueError("ImageNet refinement must not carry task checkpoint lineage")
            if not self.train_coarse or not self.coarse_model.pretrained:
                raise ValueError(
                    "ImageNet refinement requires a pretrained, trainable coarse model"
                )
        if self.crop_size[0] < 8 or self.crop_size[1] < 3 or self.crop_size[1] % 2 != 1:
            raise ValueError("refinement crop requires radius >=8 and odd angular width >=3")
        if not self.fine_channels or any(c <= 0 for c in self.fine_channels):
            raise ValueError("fine_channels must be nonempty and positive")
        if self.fine_angular_kernel % 2 != 1:
            raise ValueError("fine_angular_kernel must be odd to preserve angular width")
        if self.fine_radial_kernel % 2 != 1:
            raise ValueError("fine_radial_kernel must be odd to preserve radial stride geometry")
        if self.fine_radial_strides is not None and len(self.fine_radial_strides) != len(
            self.fine_channels
        ):
            raise ValueError("fine_radial_strides must match fine_channels length")
        radial_stride = prod(self.fine_radial_strides or (2,) * len(self.fine_channels))
        radial_rows = (self.crop_size[0] + radial_stride - 1) // radial_stride
        if self.radial_pool_bins > radial_rows:
            raise ValueError("radial_pool_bins must not exceed the fine feature radial rows")
        if not self.coarse_model.normalize_output:
            raise ValueError("refinement requires a normalized coarse double-angle vector")
        return self


class SharedPolarModelConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["shared_polar"] = "shared_polar"
    backbone_name: Literal["convnext_tiny"] = "convnext_tiny"
    pretrained: bool = True
    fusion_channels: int = Field(default=64, gt=0)
    head_hidden: int = Field(default=256, gt=0)
    dropout: float = Field(default=0.1, ge=0, lt=1)
    residual_scale: float = Field(default=0.25, ge=0)
    angular_dilations: tuple[int, int, int] = (1, 1, 1)

    @model_validator(mode="after")
    def validate_angular_dilations(self) -> SharedPolarModelConfig:
        if any(d <= 0 for d in self.angular_dilations):
            raise ValueError("angular dilations must be positive")
        return self


class CartesianConvNeXtConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["cartesian_convnext"] = "cartesian_convnext"
    backbone_name: Literal["convnext_tiny"] = "convnext_tiny"
    pretrained: bool = True
    head: Literal["stock", "spatial"] = "stock"
    head_hidden: int = Field(default=256, gt=0)
    spatial_bins: int = Field(default=4, gt=0, le=8)
    dropout: float = Field(default=0.1, ge=0, lt=1)


class CartesianRefinementConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["cartesian_refinement"] = "cartesian_refinement"
    coarse_model: CartesianConvNeXtConfig
    coarse_checkpoint: Path
    coarse_run_id: str = Field(min_length=32, max_length=32)
    coarse_split_seed: int = Field(ge=0)
    crop_size: tuple[int, int] = (384, 65)
    strip_width_px: float = Field(default=32.0, gt=0, le=256, allow_inf_nan=False)
    window_deg: float = Field(default=4.0, gt=0, lt=90, allow_inf_nan=False)
    fine_channels: tuple[int, ...] = (16, 32, 128)
    refinement_hidden: int = Field(default=128, gt=0)
    train_coarse: bool = False

    @model_validator(mode="after")
    def validate_geometry(self) -> CartesianRefinementConfig:
        if min(self.crop_size) < 8 or not self.fine_channels:
            raise ValueError("Cartesian refinement requires a nonempty encoder and crop >=8")
        if any(c <= 0 for c in self.fine_channels):
            raise ValueError("fine_channels must be positive")
        return self


class AdamWOptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["adamw"] = "adamw"
    lr: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    betas: tuple[float, float] = (0.9, 0.999)


class MuonAdamWOptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["muon_adamw"] = "muon_adamw"
    muon_lr: float = Field(default=0.02, gt=0, allow_inf_nan=False)
    muon_momentum: float = Field(default=0.95, ge=0, lt=1)
    muon_weight_decay: float = Field(default=0.1, ge=0, allow_inf_nan=False)
    adamw_lr: float = Field(default=3e-4, gt=0, allow_inf_nan=False)
    adamw_weight_decay: float = Field(default=0.01, ge=0, allow_inf_nan=False)
    adamw_betas: tuple[float, float] = (0.9, 0.999)


class MuSGDOptimizerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["musgd"] = "musgd"
    lr: float = Field(default=0.01, gt=0, allow_inf_nan=False)
    weight_decay: float = Field(default=0.0005, ge=0, allow_inf_nan=False)
    momentum: float = Field(default=0.937, ge=0, lt=1)
    nesterov: bool = True
    muon: float = Field(default=0.2, ge=0, allow_inf_nan=False)
    sgd: float = Field(default=1.0, ge=0, allow_inf_nan=False)
    boosted_parameter_names: frozenset[str] = Field(default_factory=frozenset)

    @model_validator(mode="after")
    def validate_factors(self) -> MuSGDOptimizerConfig:
        if self.muon == 0 and self.sgd == 0:
            raise ValueError("At least one of muon and sgd must be positive")
        return self


OptimizerChoice = Annotated[
    AdamWOptimizerConfig | MuonAdamWOptimizerConfig | MuSGDOptimizerConfig,
    Field(discriminator="kind"),
]


class CosineSchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["cosine_annealing"] = "cosine_annealing"
    eta_min: float = Field(default=1e-8, ge=0, allow_inf_nan=False)


class TwoPhaseCosineSchedulerConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    kind: Literal["cosine_two_phase"] = "cosine_two_phase"
    fast_decay_steps: int = Field(default=5000, gt=0)
    tail_start_factor: float = Field(default=0.01, gt=0, lt=1, allow_inf_nan=False)
    eta_min: float = Field(default=1e-8, ge=0, allow_inf_nan=False)


class AngleTrainingConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")

    max_steps: int = Field(default=5200, gt=0)
    warmup_steps: int = Field(default=200, ge=0)
    clip_grad_norm: float = Field(default=1.0, gt=0, allow_inf_nan=False)
    precision: Literal["bf16", "fp32"] = "bf16"
    log_every_steps: int = Field(default=50, gt=0)
    validate_every_steps: int = Field(default=1300, gt=0)
    validate_at_steps: tuple[int, ...] = ()
    output_dir: Path = Path("outputs/experiments")
    keep_local_checkpoints: bool = False
    initialization_checkpoint: Path | None = None
    initialization_run_id: str | None = Field(default=None, min_length=32, max_length=32)
    coarse_lr_scale: float = Field(default=1.0, gt=0, le=1, allow_inf_nan=False)

    @model_validator(mode="after")
    def validate_steps(self) -> AngleTrainingConfig:
        if self.warmup_steps >= self.max_steps:
            raise ValueError("warmup_steps must be smaller than max_steps")
        if any(step <= 0 or step > self.max_steps for step in self.validate_at_steps):
            raise ValueError("validate_at_steps must be within the training budget")
        if len(set(self.validate_at_steps)) != len(self.validate_at_steps):
            raise ValueError("validate_at_steps must not contain duplicates")
        if (self.initialization_checkpoint is None) != (self.initialization_run_id is None):
            raise ValueError("initialization requires both checkpoint and source run ID")
        return self

    def validation_steps(self) -> tuple[int, ...]:
        return tuple(
            sorted(
                set(range(self.validate_every_steps, self.max_steps + 1, self.validate_every_steps))
                | set(self.validate_at_steps)
                | {self.max_steps}
            )
        )


class AngleExperimentParams(BaseModel):
    model_config = ConfigDict(extra="forbid")

    data: AngleDataConfig
    # A plain union also accepts historical ConvNeXt configs without `kind`.
    model: (
        AngleModelConfig
        | PolarLineModelConfig
        | PolarRefinementModelConfig
        | SharedPolarModelConfig
        | CartesianConvNeXtConfig
        | CartesianRefinementConfig
    ) = Field(default_factory=AngleModelConfig)
    preprocessing: AnglePreprocessingConfig = Field(default_factory=AnglePreprocessingConfig)
    loss: AngleLossConfig = Field(default_factory=AngleLossConfig)
    optimizer: OptimizerChoice = Field(default_factory=AdamWOptimizerConfig)
    scheduler: CosineSchedulerConfig | TwoPhaseCosineSchedulerConfig = Field(
        default_factory=CosineSchedulerConfig
    )
    training: AngleTrainingConfig = Field(default_factory=AngleTrainingConfig)

    @model_validator(mode="after")
    def validate_model_input(self) -> AngleExperimentParams:
        cartesian = isinstance(self.model, CartesianConvNeXtConfig | CartesianRefinementConfig)
        if cartesian != (self.preprocessing.projection == "cartesian"):
            raise ValueError("Cartesian networks require Cartesian preprocessing")
        if cartesian and self.preprocessing.output_size != (256, 256):
            raise ValueError("Cartesian comparisons preserve the native 256x256 image")
        if isinstance(self.scheduler, TwoPhaseCosineSchedulerConfig) and not (
            self.training.warmup_steps < self.scheduler.fast_decay_steps < self.training.max_steps
        ):
            raise ValueError("Two-phase cosine requires warmup < fast decay < max_steps")
        if isinstance(self.model, PolarLineModelConfig) and self.preprocessing.output_size[0] % 2:
            raise ValueError("polar_line requires an even signed-polar radial size")
        if (
            isinstance(self.model, PolarRefinementModelConfig | CartesianRefinementConfig)
            and getattr(self.model, "initialization", "checkpoint") == "checkpoint"
            and self.model.coarse_split_seed != self.data.split_seed
        ):
            raise ValueError("refinement and coarse predictor must use the same split seed")
        return self

    def to_optimizer_config(self) -> OptimizerConfig:
        settings = self.optimizer.model_dump(exclude={"kind"})
        if isinstance(self.optimizer, AdamWOptimizerConfig):
            return AdamWConfig(**settings)
        if isinstance(self.optimizer, MuonAdamWOptimizerConfig):
            return MuonAdamWConfig(**settings)
        return UltralyticsMuSGDConfig(**settings)
