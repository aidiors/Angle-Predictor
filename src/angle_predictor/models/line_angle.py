from __future__ import annotations

import hashlib
import json
import math
from typing import Any, Literal

import timm
import torch
from torch import nn
from torch.nn import functional as F

from angle_predictor.config.angle_experiment import (
    AngleExperimentParams,
    AngleModelConfig,
    CartesianConvNeXtConfig,
    CartesianRefinementConfig,
    PolarLineModelConfig,
    PolarRefinementModelConfig,
    SharedPolarModelConfig,
)
from angle_predictor.data.angle_transforms import AngleBatchPreprocessor
from angle_predictor.models.cartesian import CartesianConvNeXt, CartesianRefinement
from angle_predictor.models.feedback import ConvNeXtFeedback
from angle_predictor.models.polar_line import PolarLineModel
from angle_predictor.models.polar_refinement import PolarRefinementModel


class PolarAngleRegressionHead(nn.Module):
    """Reduce a polar feature map to a double-angle vector."""

    def __init__(
        self,
        in_ch: int,
        hidden: int = 256,
        dropout: float = 0.1,
        residual_scale: float = 0.25,
        normalize_output: bool = True,
        angular_dilations: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        if len(angular_dilations) != 3 or any(d <= 0 for d in angular_dilations):
            raise ValueError("angular dilations require three positive values")
        first, second, third = angular_dilations
        self.residual_scale = residual_scale
        self.normalize_output = normalize_output
        self.radius_query = nn.Parameter(torch.zeros(in_ch))
        self.angle_conv = nn.Sequential(
            nn.Conv1d(
                3 * in_ch + 2,
                hidden,
                5,
                padding=2 * first,
                dilation=first,
                padding_mode="circular",
            ),
            nn.GELU(),
            nn.Conv1d(
                hidden,
                hidden,
                5,
                padding=2 * second,
                dilation=second,
                padding_mode="circular",
            ),
            nn.GELU(),
            nn.Conv1d(
                hidden,
                hidden,
                3,
                padding=third,
                dilation=third,
                padding_mode="circular",
            ),
            nn.GELU(),
        )
        self.score = nn.Conv1d(hidden, 1, 1)
        self.delta_head = nn.Sequential(
            nn.LayerNorm(hidden + 2),
            nn.Linear(hidden + 2, hidden),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 2),
        )

    @staticmethod
    def _angle_pos(
        batch_size: int, width: int, device: torch.device, dtype: torch.dtype
    ) -> torch.Tensor:
        theta = torch.arange(width, device=device, dtype=dtype) * (math.pi / width)
        position = torch.stack((torch.sin(2 * theta), torch.cos(2 * theta)), dim=0)
        return position.unsqueeze(0).expand(batch_size, -1, -1)

    def forward(self, features: torch.Tensor) -> torch.Tensor:
        batch, channels, _, width = features.shape
        mean_radius = features.mean(dim=2)
        max_radius = features.amax(dim=2)
        radius_scores = torch.einsum("bchw,c->bhw", features, self.radius_query)
        radius_attention = torch.softmax(radius_scores / math.sqrt(channels), dim=1).unsqueeze(1)
        attended_radius = (features * radius_attention).sum(dim=2)
        position = self._angle_pos(batch, width, features.device, features.dtype)
        hidden = self.angle_conv(
            torch.cat((mean_radius, max_radius, attended_radius, position), dim=1)
        )
        angle_attention = torch.softmax(self.score(hidden).squeeze(1).float(), dim=-1)
        angle_attention = angle_attention.to(hidden.dtype)
        pooled_features = (hidden * angle_attention.unsqueeze(1)).sum(dim=-1)
        pooled_position = (position * angle_attention.unsqueeze(1)).sum(dim=-1)
        delta = self.delta_head(torch.cat((pooled_features, pooled_position), dim=-1).float())
        output = pooled_position.float() + self.residual_scale * delta
        return F.normalize(output, p=2, dim=1, eps=1e-6) if self.normalize_output else output


class LineAngleModel(nn.Module):
    """ConvNeXt feature backbone with the original polar angle head."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        out_index: int = 3,
        head_hidden: int = 256,
        pretrained: bool = True,
        dropout: float = 0.1,
        residual_scale: float = 0.25,
        normalize_output: bool = True,
        final_stage_blocks: int | None = None,
        penultimate_stage_blocks: int | None = None,
        feedback_mode: Literal["none", "fixed", "gated", "routed"] = "none",
    ) -> None:
        super().__init__()
        self.backbone: Any = timm.create_model(
            backbone_name,
            pretrained=pretrained,
            features_only=True,
            out_indices=(out_index,),
            in_chans=3,
        )
        if final_stage_blocks is not None:
            if (
                backbone_name != "convnext_tiny"
                or out_index != 3
                or not 1 <= final_stage_blocks <= 3
            ):
                raise ValueError(
                    "final_stage_blocks requires convnext_tiny stage 3 and 1..3 blocks"
                )
            # Load original pretrained weights before dropping trailing blocks.
            stage = self.backbone.get_submodule("stages_3")
            stage.blocks = nn.Sequential(*list(stage.blocks.children())[:final_stage_blocks])
        if penultimate_stage_blocks is not None:
            if (
                backbone_name != "convnext_tiny"
                or out_index not in {2, 3}
                or not 1 <= penultimate_stage_blocks <= 9
            ):
                raise ValueError(
                    "penultimate_stage_blocks requires convnext_tiny stage 2 and 1..9 blocks"
                )
            stage = self.backbone.get_submodule("stages_2")
            stage.blocks = nn.Sequential(*list(stage.blocks.children())[:penultimate_stage_blocks])
        channels = self.backbone.feature_info.channels()[-1]
        self.head = PolarAngleRegressionHead(
            channels,
            hidden=head_hidden,
            dropout=dropout,
            residual_scale=residual_scale,
            normalize_output=normalize_output,
        )
        self.feedback = None
        if feedback_mode != "none":
            if backbone_name != "convnext_tiny" or out_index != 3:
                raise ValueError("Feedback requires ConvNeXt Tiny stage 4 output")
            with torch.random.fork_rng(devices=[]):
                self.feedback = ConvNeXtFeedback(feedback_mode)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = (
            self.backbone(image)[-1]
            if self.feedback is None
            else self.feedback(self.backbone, image)
        ).float()
        if image.device.type in {"cuda", "cpu"}:
            with torch.amp.autocast(device_type=image.device.type, enabled=False):
                return self.head(features)
        return self.head(features)


class SharedPolarModel(nn.Module):
    """Fuse stride-4 detail and stride-32 context from one trainable backbone."""

    def __init__(
        self,
        backbone_name: str = "convnext_tiny",
        pretrained: bool = True,
        fusion_channels: int = 64,
        head_hidden: int = 256,
        dropout: float = 0.1,
        residual_scale: float = 0.25,
        angular_dilations: tuple[int, int, int] = (1, 1, 1),
    ) -> None:
        super().__init__()
        self.backbone: Any = timm.create_model(
            backbone_name, pretrained=pretrained, features_only=True, out_indices=(0, 3), in_chans=3
        )
        early_channels, deep_channels = self.backbone.feature_info.channels()
        self.detail = nn.Conv2d(early_channels, fusion_channels, 1)
        self.context = nn.Conv2d(deep_channels, fusion_channels, 1)
        self.head = PolarAngleRegressionHead(
            2 * fusion_channels,
            head_hidden,
            dropout,
            residual_scale,
            normalize_output=True,
            angular_dilations=angular_dilations,
        )

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        early, deep = self.backbone(image)
        detail = self.detail(early)
        context = self.context(deep)
        # Deep context is global; the early feature map retains angular detail.
        context = F.interpolate(
            context, size=detail.shape[-2:], mode="bilinear", align_corners=False
        )
        with torch.amp.autocast(device_type=image.device.type, enabled=False):
            return self.head(torch.cat((detail.float(), context.float()), dim=1))


class AnglePredictor(nn.Module):
    """Apply the same signed-polar input transform during training and inference."""

    def __init__(
        self,
        preprocessor: AngleBatchPreprocessor,
        network: nn.Module,
        inference_precision: Literal["bf16", "fp32"] = "fp32",
    ) -> None:
        super().__init__()
        self.preprocessor = preprocessor
        self.network = network
        self.inference_precision = inference_precision

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        if (
            isinstance(self.network, PolarRefinementModel)
            and self.network.fine_crop_source == "cartesian"
        ):
            polar, cartesian, _ = self.preprocessor.forward_with_cartesian(images)
            return self.network(polar, cartesian)
        polar_images, _ = self.preprocessor(images, augment=False)
        return self.network(polar_images)

    def forward_augmented(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        if (
            isinstance(self.network, PolarRefinementModel)
            and self.network.fine_crop_source == "cartesian"
        ):
            polar, cartesian, adjusted_targets = self.preprocessor.forward_with_cartesian(
                images, targets, augment=True
            )
            assert adjusted_targets is not None
            return self.network(polar, cartesian), adjusted_targets
        polar_images, adjusted_targets = self.preprocessor(images, targets, augment=True)
        assert adjusted_targets is not None
        return self.network(polar_images), adjusted_targets


def build_angle_network(
    config: AngleModelConfig
    | PolarLineModelConfig
    | PolarRefinementModelConfig
    | SharedPolarModelConfig
    | CartesianConvNeXtConfig
    | CartesianRefinementConfig,
    *,
    load_pretrained: bool = True,
    training_params: AngleExperimentParams | None = None,
) -> nn.Module:
    if isinstance(config, CartesianRefinementConfig):
        coarse = build_angle_network(config.coarse_model, load_pretrained=False)
        if load_pretrained:
            if training_params is None:
                raise ValueError("Cartesian refinement requires experiment context")
            payload = torch.load(config.coarse_checkpoint, map_location="cpu", weights_only=True)
            _validate_refinement_source(payload, config, training_params)
            coarse.load_state_dict(
                {
                    key.removeprefix("network."): value
                    for key, value in payload["model_state_dict"].items()
                    if key.startswith("network.")
                },
                strict=True,
            )
        options = config.model_dump(
            exclude={
                "kind",
                "coarse_model",
                "coarse_checkpoint",
                "coarse_run_id",
                "coarse_split_seed",
            }
        )
        return CartesianRefinement(coarse, **options)
    if isinstance(config, PolarRefinementModelConfig):
        payload = None
        if load_pretrained and config.initialization == "checkpoint":
            if training_params is None:
                raise ValueError("refinement initialization requires the experiment data context")
            assert config.coarse_checkpoint is not None
            payload = torch.load(config.coarse_checkpoint, map_location="cpu", weights_only=True)
            _validate_refinement_source(payload, config, training_params)
        coarse = build_angle_network(
            config.coarse_model,
            load_pretrained=load_pretrained and config.initialization == "imagenet",
        )
        if payload is not None:
            weights = {
                key.removeprefix("network."): value
                for key, value in payload["model_state_dict"].items()
                if key.startswith("network.")
            }
            coarse.load_state_dict(weights, strict=True)
        return PolarRefinementModel(
            coarse,
            crop_size=config.crop_size,
            window_deg=config.window_deg,
            fine_channels=config.fine_channels,
            fine_angular_kernel=config.fine_angular_kernel,
            fine_radial_kernel=config.fine_radial_kernel,
            fine_radial_strides=config.fine_radial_strides,
            radial_pool_bins=config.radial_pool_bins,
            radial_pool_mode=config.radial_pool_mode,
            fine_crop_source=config.fine_crop_source,
            fine_detail_skip=config.fine_detail_skip,
            fine_radial_reflection=config.fine_radial_reflection,
            fine_reflection_readout=config.fine_reflection_readout,
            fine_geometric_residual=config.fine_geometric_residual,
            fine_angular_antisymmetry=config.fine_angular_antisymmetry,
            fine_radial_antialias=config.fine_radial_antialias,
            refinement_hidden=config.refinement_hidden,
            refinement_head=config.refinement_head,
            detach_crop_angle=config.detach_crop_angle,
            train_coarse=config.train_coarse,
        )
    options = config.model_dump(exclude={"kind"})
    if isinstance(config, PolarLineModelConfig):
        return PolarLineModel(**options)
    if not load_pretrained:
        options["pretrained"] = False
    if isinstance(config, SharedPolarModelConfig):
        return SharedPolarModel(**options)
    if isinstance(config, CartesianConvNeXtConfig):
        return CartesianConvNeXt(**options)
    return LineAngleModel(**options)


def _validate_refinement_source(
    payload: dict[str, Any],
    config: PolarRefinementModelConfig | CartesianRefinementConfig,
    params: AngleExperimentParams,
) -> None:
    """Reject coarse weights from different splits, pixels, geometry or model definitions."""
    metadata = payload["metadata"]
    source = AngleExperimentParams.model_validate(metadata["params"])
    if metadata.get("format_version") != 1 or metadata.get("run_id") != config.coarse_run_id:
        raise ValueError("coarse checkpoint format/run ID differs from refinement lineage")
    if source.model != config.coarse_model:
        raise ValueError("coarse checkpoint model differs from its configured architecture")
    if source.data.split_seed != params.data.split_seed:
        raise ValueError("coarse checkpoint was trained on a different split seed")
    if (
        source.data.root.resolve() != params.data.root.resolve()
        or source.data.train_fraction != params.data.train_fraction
        or source.data.val_fraction != params.data.val_fraction
    ):
        raise ValueError("coarse checkpoint uses different data/split fractions")
    if source.preprocessing != params.preprocessing:
        raise ValueError("coarse checkpoint preprocessing differs from refinement preprocessing")
    meta_path = params.data.root / "meta.json"
    meta_bytes = meta_path.read_bytes()
    data_metadata = json.loads(meta_bytes)
    if metadata.get("dataset_meta_sha256") != hashlib.sha256(meta_bytes).hexdigest():
        raise ValueError("coarse checkpoint dataset fingerprint differs from current pixels")
    if tuple(metadata["input_size"]) != tuple(data_metadata["images_shape"][1:3]):
        raise ValueError("coarse checkpoint input geometry differs from the dataset")


def validate_initialization_source(payload: dict[str, Any], params: AngleExperimentParams) -> None:
    """Validate a full-model warm start without carrying source optimizer state."""
    metadata = payload["metadata"]
    source = AngleExperimentParams.model_validate(metadata["params"])
    if (
        metadata.get("format_version") != 1
        or metadata.get("run_id") != params.training.initialization_run_id
    ):
        raise ValueError("initialization checkpoint format/run ID differs from source lineage")
    expected_model = params.model.model_dump(exclude={"train_coarse"})
    source_model = source.model.model_dump(exclude={"train_coarse"})
    if params.training.initialization_mode == "feedback_base":
        if (
            not isinstance(params.model, AngleModelConfig)
            or not isinstance(source.model, AngleModelConfig)
            or params.model.feedback_mode == "none"
            or source.model.feedback_mode != "none"
        ):
            raise ValueError("Feedback initialization requires an original coarse source")
        expected_model["feedback_mode"] = "none"
    if source_model != expected_model:
        raise ValueError("initialization checkpoint architecture differs from target")
    if source.data.split_seed != params.data.split_seed:
        raise ValueError("initialization checkpoint was trained on a different split seed")
    if (
        source.data.root.resolve() != params.data.root.resolve()
        or source.data.train_fraction != params.data.train_fraction
        or source.data.val_fraction != params.data.val_fraction
        or source.preprocessing != params.preprocessing
    ):
        raise ValueError("initialization checkpoint data/preprocessing differs from target")
    meta_bytes = (params.data.root / "meta.json").read_bytes()
    if metadata.get("dataset_meta_sha256") != hashlib.sha256(meta_bytes).hexdigest():
        raise ValueError("initialization checkpoint dataset fingerprint differs")
    if tuple(metadata["input_size"]) != tuple(json.loads(meta_bytes)["images_shape"][1:3]):
        raise ValueError("initialization checkpoint input geometry differs")


def load_initialization_weights(
    model: AnglePredictor, weights: dict, params: AngleExperimentParams
) -> None:
    if params.training.initialization_mode == "full":
        model.load_state_dict(weights, strict=True)
        return
    missing = {key for key in model.state_dict() if key.startswith("network.feedback.")}
    if not missing or set(model.state_dict()) - missing != set(weights):
        raise ValueError(
            "Feedback transfer must preserve every original backbone/head/buffer tensor"
        )
    merged = {**model.state_dict(), **weights}
    model.load_state_dict(merged, strict=True)
