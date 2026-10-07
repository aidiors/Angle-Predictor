from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections.abc import Iterator
from contextlib import suppress
from typing import Protocol

import mlflow
import numpy as np
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

from angle_predictor.config.angle_experiment import (
    AngleExperimentParams,
    PolarRefinementModelConfig,
    TwoPhaseCosineSchedulerConfig,
)
from angle_predictor.config.schema import ExperimentConfig
from angle_predictor.data.angle_dataset import AngleMemmapDataset
from angle_predictor.data.angle_transforms import AngleBatchPreprocessor
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.engine.device import resolve_device
from angle_predictor.engine.optimizers import build_optimizer
from angle_predictor.engine.optimizers.factory import ChainedOptimizer
from angle_predictor.engine.seed import seed_everything
from angle_predictor.losses.angle import AngleLossFn, build_angle_loss
from angle_predictor.models.line_angle import (
    AnglePredictor,
    build_angle_network,
    load_initialization_weights,
    validate_initialization_source,
)
from angle_predictor.tracking.mlflow import (
    log_checkpoint,
    log_metrics,
    log_params,
    log_pytorch_model,
)


def _scheduler(
    optimizer: optim.Optimizer, params: AngleExperimentParams
) -> optim.lr_scheduler.LRScheduler:
    training = params.training
    if isinstance(params.scheduler, TwoPhaseCosineSchedulerConfig):
        settings = params.scheduler

        def multiplier(base_lr: float):
            tail_lr = base_lr * settings.tail_start_factor
            if tail_lr <= settings.eta_min:
                raise ValueError("eta_min must be below every group's two-phase tail LR")

            def value(step: int) -> float:
                step = min(max(step, 0), training.max_steps)
                if training.warmup_steps and step <= training.warmup_steps:
                    progress = step / training.warmup_steps
                    return (1 - progress) / training.warmup_steps + progress
                if step <= settings.fast_decay_steps:
                    progress = (step - training.warmup_steps) / (
                        settings.fast_decay_steps - training.warmup_steps
                    )
                    lr = tail_lr + 0.5 * (base_lr - tail_lr) * (1 + math.cos(math.pi * progress))
                else:
                    progress = (step - settings.fast_decay_steps) / (
                        training.max_steps - settings.fast_decay_steps
                    )
                    lr = settings.eta_min + 0.5 * (tail_lr - settings.eta_min) * (
                        1 + math.cos(math.pi * progress)
                    )
                return lr / base_lr

            return value

        return optim.lr_scheduler.LambdaLR(
            optimizer, lr_lambda=[multiplier(group["lr"]) for group in optimizer.param_groups]
        )
    cosine = optim.lr_scheduler.CosineAnnealingLR(
        optimizer,
        T_max=training.max_steps - training.warmup_steps,
        eta_min=params.scheduler.eta_min,
    )
    if training.warmup_steps == 0:
        return cosine
    warmup = optim.lr_scheduler.LinearLR(
        optimizer,
        start_factor=1 / training.warmup_steps,
        end_factor=1.0,
        total_iters=training.warmup_steps,
    )
    return optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup, cosine], milestones=[training.warmup_steps]
    )


def _angle_errors_deg(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    prediction = prediction.float()
    target = target.float()
    predicted_norm = prediction.norm(dim=-1)
    prediction = nn.functional.normalize(prediction, dim=-1, eps=1e-6)
    target = nn.functional.normalize(target, dim=-1, eps=1e-6)
    sine = prediction[:, 0] * target[:, 1] - prediction[:, 1] * target[:, 0]
    cosine = (prediction * target).sum(dim=-1)
    error = 0.5 * torch.atan2(sine, cosine).abs() * (180.0 / math.pi)
    return torch.where(predicted_norm < 1e-6, 90.0, error)


def _train_angle_mae_deg(angle_error_sum_deg: torch.Tensor, sample_count: int) -> float:
    if sample_count <= 0:
        raise ValueError("sample_count must be positive")
    return float(angle_error_sum_deg.item()) / sample_count


class _AngleUpdateModel(Protocol):
    def parameters(self, recurse: bool = True) -> Iterator[nn.Parameter]: ...

    def forward_augmented(
        self, images: torch.Tensor, targets: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]: ...


def _train_angle_update(
    model: _AngleUpdateModel,
    optimizer: optim.Optimizer,
    criterion: AngleLossFn,
    microbatches: list[tuple[torch.Tensor, torch.Tensor]],
    device: torch.device,
    use_bf16: bool,
    clip_grad_norm: float,
    step: int,
) -> tuple[float, torch.Tensor, int, torch.Tensor]:
    sample_count = sum(len(images) for images, _ in microbatches)
    if sample_count <= 0:
        raise ValueError("An optimizer update requires at least one sample")
    optimizer.zero_grad(set_to_none=True)
    loss_sum = 0.0
    angle_sum = torch.zeros((), device=device)
    for images, targets in microbatches:
        images = images.to(device, non_blocking=device.type == "cuda")
        targets = targets.to(device, non_blocking=device.type == "cuda")
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
            prediction, adjusted_targets = model.forward_augmented(images, targets)
            loss = criterion(prediction, adjusted_targets)
        if not torch.isfinite(loss):
            raise FloatingPointError(f"Nonfinite training loss at step {step}")
        (loss * (len(images) / sample_count)).backward()
        loss_sum += loss.item() * len(images)
        angle_sum += _angle_errors_deg(prediction.detach(), adjusted_targets).sum()
    grad_norm = nn.utils.clip_grad_norm_(
        model.parameters(), clip_grad_norm, error_if_nonfinite=True
    )
    optimizer.step()
    return loss_sum, angle_sum, sample_count, grad_norm


def _validate(
    model: AnglePredictor,
    loader: DataLoader,
    criterion: AngleLossFn,
    device: torch.device,
    use_bf16: bool,
) -> dict[str, float]:
    model.eval()
    total_loss = 0.0
    total_error = 0.0
    total_count = 0
    degenerate_count = 0
    with torch.inference_mode():
        for images, targets in loader:
            images = images.to(device, non_blocking=device.type == "cuda")
            targets = targets.to(device, non_blocking=device.type == "cuda")
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                prediction = model(images)
                loss = criterion(prediction, targets)
            count = len(images)
            total_loss += loss.item() * count
            total_error += _angle_errors_deg(prediction, targets).sum().item()
            degenerate_count += (prediction.float().norm(dim=-1) < 1e-6).sum().item()
            total_count += count
    model.train()
    return {
        "val/loss": total_loss / total_count,
        "val/angle_mae_deg": total_error / total_count,
        "val/degenerate_fraction": degenerate_count / total_count,
    }


def _build_model(params: AngleExperimentParams, input_size: tuple[int, int]) -> AnglePredictor:
    preprocessor = AngleBatchPreprocessor(
        input_size=input_size,
        **params.preprocessing.model_dump(),
    )
    initialization = params.training.initialization_checkpoint
    network = build_angle_network(
        params.model, training_params=params, load_pretrained=initialization is None
    )
    model = AnglePredictor(
        preprocessor,
        network,
        inference_precision=params.training.precision,
    )
    if initialization is not None:
        payload = torch.load(initialization, map_location="cpu", weights_only=True)
        validate_initialization_source(payload, params)
        load_initialization_weights(model, payload["model_state_dict"], params)
    return model


def _build_angle_optimizer(network: nn.Module, params: AngleExperimentParams) -> optim.Optimizer:
    optimizer = build_optimizer(network, params.to_optimizer_config())
    scale = params.training.coarse_lr_scale
    if scale != 1:
        if (
            not isinstance(params.model, PolarRefinementModelConfig)
            or not params.model.train_coarse
        ):
            raise ValueError("coarse_lr_scale requires joint refinement")
        coarse_ids = {id(p) for p in network.get_submodule("coarse").parameters()}
        parts = optimizer.optimizers if isinstance(optimizer, ChainedOptimizer) else (optimizer,)
        for part in parts:
            groups = list(part.param_groups)
            part.param_groups.clear()
            for group in groups:
                for is_coarse in (False, True):
                    members = [p for p in group["params"] if (id(p) in coarse_ids) == is_coarse]
                    if members:
                        part.add_param_group(
                            {
                                **group,
                                "params": members,
                                "lr": group["lr"] * (scale if is_coarse else 1),
                            }
                        )
        if isinstance(optimizer, ChainedOptimizer):
            optimizer.param_groups = [group for part in parts for group in part.param_groups]
    return optimizer


def run(config: ExperimentConfig) -> None:
    """Train the configured angle baseline inside the runner's MLflow run."""
    params = AngleExperimentParams.model_validate(config.params)
    seed_everything(config.seed)
    device = resolve_device("auto")
    use_bf16 = params.training.precision == "bf16"
    if use_bf16 and (device.type != "cuda" or not torch.cuda.is_bf16_supported()):
        raise RuntimeError("BF16 training requires a CUDA device with BF16 support")

    root = params.data.root.resolve()
    meta_path = root / "meta.json"
    metadata = json.loads(meta_path.read_text(encoding="utf-8"))
    if metadata.get("dataset_version") != config.dataset.version:
        raise ValueError("Dataset version differs from meta.json")
    meta_sha256 = hashlib.sha256(meta_path.read_bytes()).hexdigest()
    data_options = {
        "seed": params.data.split_seed,
        "train_fraction": params.data.train_fraction,
        "val_fraction": params.data.val_fraction,
    }
    train_data = AngleMemmapDataset(root, "train", **data_options)
    val_data = AngleMemmapDataset(root, "val", **data_options)
    if not len(train_data) or not len(val_data):
        raise ValueError("Train and validation splits must be nonempty")
    train_loader = DataLoader(
        train_data,
        batch_size=params.data.batch_size,
        shuffle=True,
        generator=torch.Generator().manual_seed(config.seed),
        num_workers=params.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=params.data.num_workers > 0,
    )
    val_loader = DataLoader(
        val_data,
        batch_size=params.data.batch_size,
        shuffle=False,
        num_workers=params.data.num_workers,
        pin_memory=device.type == "cuda",
        persistent_workers=params.data.num_workers > 0,
    )

    model = _build_model(params, train_data.image_size).to(device)
    optimizer = _build_angle_optimizer(model.network, params)
    if params.scheduler.eta_min >= min(group["lr"] for group in optimizer.param_groups):
        raise ValueError("Cosine eta_min must be below every optimizer learning rate")
    scheduler = _scheduler(optimizer, params)
    criterion = build_angle_loss(**params.loss.model_dump())
    effective = params.model_dump(mode="json")
    log_params(
        {
            "resolved": effective,
            "runtime": {
                "device": str(device),
                "input_size": train_data.image_size,
                "train_samples": len(train_data),
                "val_samples": len(val_data),
                "model_parameters": sum(parameter.numel() for parameter in model.parameters()),
                "effective_batch_size": (
                    params.data.batch_size * params.training.gradient_accumulation_steps
                ),
            },
        }
    )
    mlflow.log_artifact(str(meta_path), artifact_path="dataset")
    mlflow.set_tag("dataset.meta_sha256", meta_sha256)
    active = mlflow.active_run()
    if active is None:
        raise RuntimeError("The angle experiment requires an active MLflow run")
    run_id = active.info.run_id
    output_dir = params.training.output_dir / config.name / run_id
    output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_metadata = {
        "format_version": 1,
        "experiment_name": config.name,
        "run_id": run_id,
        "dataset_name": config.dataset.name,
        "dataset_version": config.dataset.version,
        "dataset_meta_sha256": meta_sha256,
        "input_size": train_data.image_size,
        "params": effective,
    }

    model.train()
    train_iterator = iter(train_loader)
    accumulated_loss = 0.0
    accumulated_angle_error_deg = torch.zeros((), device=device)
    accumulated_count = 0
    best_error = math.inf
    best_step = 0
    best_path = output_dir / "best.pt"
    last_path = output_dir / "last.pt"
    validation_steps = set(params.training.validation_steps())
    for step in range(1, params.training.max_steps + 1):
        microbatches = []
        for _ in range(params.training.gradient_accumulation_steps):
            try:
                microbatches.append(next(train_iterator))
            except StopIteration:
                train_iterator = iter(train_loader)
                microbatches.append(next(train_iterator))
        loss_sum, angle_sum, count, grad_norm = _train_angle_update(
            model,
            optimizer,
            criterion,
            microbatches,
            device,
            use_bf16,
            params.training.clip_grad_norm,
            step,
        )
        scheduler.step()
        accumulated_loss += loss_sum
        accumulated_angle_error_deg += angle_sum
        accumulated_count += count

        if step % params.training.log_every_steps == 0 or step == params.training.max_steps:
            rates = [group["lr"] for group in optimizer.param_groups]
            mean_loss = accumulated_loss / accumulated_count
            angle_mae_deg = _train_angle_mae_deg(accumulated_angle_error_deg, accumulated_count)
            log_metrics(
                {
                    "train/loss": mean_loss,
                    "train/angle_mae_deg": angle_mae_deg,
                    "train/grad_norm": float(grad_norm),
                    "train/lr_min": min(rates),
                    "train/lr_max": max(rates),
                },
                step=step,
            )
            print(
                f"step {step}/{params.training.max_steps} "
                f"train_loss={mean_loss:.5f} "
                f"train_angle_mae_deg={angle_mae_deg:.3f} lr={max(rates):.3g}",
                flush=True,
            )
            accumulated_loss = 0.0
            accumulated_angle_error_deg.zero_()
            accumulated_count = 0

        if step in validation_steps:
            metrics = _validate(model, val_loader, criterion, device, use_bf16)
            log_metrics(metrics, step=step)
            print(
                f"step {step}/{params.training.max_steps} "
                f"val_angle_mae_deg={metrics['val/angle_mae_deg']:.3f} "
                f"val_loss={metrics['val/loss']:.5f}",
                flush=True,
            )
            if metrics["val/angle_mae_deg"] < best_error:
                best_error = metrics["val/angle_mae_deg"]
                best_step = step
                save_checkpoint(
                    model,
                    best_path,
                    step=step,
                    metadata={**checkpoint_metadata, "val_angle_mae_deg": best_error},
                )
                mlflow.set_tag("best_step", str(step))
            save_checkpoint(
                model,
                last_path,
                optimizer=optimizer,
                scheduler=scheduler,
                step=step,
                metadata={**checkpoint_metadata, "val_angle_mae_deg": metrics["val/angle_mae_deg"]},
            )
            model.train()

    log_checkpoint(best_path)
    log_checkpoint(last_path)
    mlflow.set_tag("best_val_angle_mae_deg", str(best_error))
    best_payload = torch.load(best_path, map_location=device, weights_only=True)
    model.load_state_dict(best_payload["model_state_dict"], strict=True)
    model.eval()
    sample_image, _ = val_data[0]
    input_example = np.expand_dims(sample_image.numpy(), axis=0).copy()
    log_pytorch_model(
        model,
        input_example,
        registered_model_name=config.tracking.registered_model_name,
        model_version_tags={
            "validation.angle_mae_deg": f"{best_error:.8f}",
            "checkpoint.step": str(best_step),
            "dataset.name": config.dataset.name,
            "dataset.version": config.dataset.version,
            "loss.kind": params.loss.kind,
            "optimizer.kind": params.optimizer.kind,
        },
    )
    train_data.close()
    val_data.close()
    if not params.training.keep_local_checkpoints:
        shutil.rmtree(output_dir, ignore_errors=True)
        with suppress(OSError):
            output_dir.parent.rmdir()
