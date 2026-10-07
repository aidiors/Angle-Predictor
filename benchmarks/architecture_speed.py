from __future__ import annotations

import argparse
import json
import math
import statistics
import time
from pathlib import Path
from typing import Any

import torch

from angle_predictor.config.angle_experiment import AngleExperimentParams
from angle_predictor.config.loader import load_config
from angle_predictor.data.angle_transforms import AngleBatchPreprocessor
from angle_predictor.experiments.angle import _build_angle_optimizer
from angle_predictor.losses.angle import build_angle_loss
from angle_predictor.models.line_angle import AnglePredictor, build_angle_network
from angle_predictor.models.polar_refinement import PolarRefinementModel


def operation_counts(config_path: Path) -> dict:
    params = AngleExperimentParams.model_validate(load_config(config_path).params)
    with torch.device("meta"):
        network = build_angle_network(params.model, load_pretrained=False).eval()
    counts = []

    def count(module, _inputs, output):
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d)):
            width = module.in_channels // module.groups * math.prod(module.kernel_size)
        else:
            width = module.in_features
        counts.append(output.numel() * width)

    handles = [
        module.register_forward_hook(count)
        for module in network.modules()
        if isinstance(module, (torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear))
    ]
    shape = (1, 3, *params.preprocessing.output_size)
    try:
        with torch.inference_mode():
            if isinstance(network, PolarRefinementModel):
                if network.refinement_head != "mlp" or network.radial_pool_bins != 1:
                    raise ValueError("Operation counting supports the primary mean/max MLP model")
                network.coarse(torch.empty(shape, device="meta"))
                coarse = sum(counts)
                counts.clear()
                views = (2 if network.fine_angular_antisymmetry else 1) * (
                    2 if network.fine_radial_reflection else 1
                )
                features = network.fine(torch.empty((views, 5, *network.crop_size), device="meta"))
                profile = 2 * features.shape[1] * network.crop_size[1] + 2
                network.correction(torch.empty((views, profile), device="meta"))
                fine = sum(counts)
            else:
                network(torch.empty(shape, device="meta"))
                coarse, fine = sum(counts), 0
    finally:
        for handle in handles:
            handle.remove()
    return {
        "coarse_macs": coarse,
        "fine_macs": fine,
        "total_macs": coarse + fine,
        "gflops_two_per_mac": 2 * (coarse + fine) / 1e9,
        "scope": "Convolutions and linear layers, excluding sampling, reductions and activations",
    }


def measure(
    config_path: Path, iterations: int, warmup: int = 5, *, training_step: bool = False
) -> dict:
    params = AngleExperimentParams.model_validate(load_config(config_path).params)
    device = torch.device("cuda")
    model = AnglePredictor(
        AngleBatchPreprocessor(**params.preprocessing.model_dump()),
        build_angle_network(params.model, load_pretrained=False),
    ).to(device)
    result = {
        "parameters": sum(p.numel() for p in model.parameters()),
        "weights": "Random initialization, no checkpoint or dataset required",
    }
    for batch_size in (1, 32):
        images = torch.randint(0, 256, (batch_size, 3, 256, 256), dtype=torch.uint8, device=device)
        model.eval()
        times = []
        with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
            for step in range(warmup + iterations):
                torch.cuda.synchronize()
                start = time.perf_counter()
                model(images)
                torch.cuda.synchronize()
                if step >= warmup:
                    times.append((time.perf_counter() - start) * 1000)
        result[f"inference_batch{batch_size}_median_ms"] = statistics.median(times)
    if not training_step:
        return result
    optimizer = _build_angle_optimizer(model.network, params)
    criterion = build_angle_loss(**params.loss.model_dump())
    targets = torch.randn(32, 2, device=device)
    targets = torch.nn.functional.normalize(targets, dim=-1)
    model.train()
    times = []
    torch.cuda.reset_peak_memory_stats()
    for step in range(warmup + iterations):
        torch.cuda.synchronize()
        start = time.perf_counter()
        optimizer.zero_grad(set_to_none=True)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            prediction, adjusted = model.forward_augmented(images, targets)
            loss = criterion(prediction, adjusted)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), params.training.clip_grad_norm)
        optimizer.step()
        torch.cuda.synchronize()
        if step >= warmup:
            times.append((time.perf_counter() - start) * 1000)
    result["training_batch32_median_ms"] = statistics.median(times)
    result["training_peak_allocated_mib"] = torch.cuda.max_memory_allocated() / 1024**2
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Measure predictor latency and memory")
    parser.add_argument("--iterations", type=int, default=20)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--operations-only", action="store_true")
    parser.add_argument("--training-step", action="store_true")
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/experiments/convnext_tiny_musgd_vector_charbonnier_tuned.yaml"),
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=Path("configs/final/refinement.yaml"),
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    if args.iterations < 1:
        parser.error("iterations must be positive")
    output: dict[str, Any] = {"torch": torch.__version__}
    if not args.operations_only:
        output.update(gpu=torch.cuda.get_device_name(0), precision="bf16")
    for name, config_path in (
        ("baseline", args.baseline_config),
        ("candidate", args.candidate_config),
    ):
        output[name] = operation_counts(config_path)
        if not args.operations_only:
            output[name].update(
                measure(config_path, args.iterations, training_step=args.training_step)
            )
            torch.cuda.empty_cache()
        print(name, json.dumps(output[name]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
