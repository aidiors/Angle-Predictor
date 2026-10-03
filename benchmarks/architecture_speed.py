from __future__ import annotations

import argparse
import json
import statistics
import time
from pathlib import Path

import torch

from angle_predictor.config.angle_experiment import AngleExperimentParams
from angle_predictor.config.loader import load_config
from angle_predictor.data.angle_transforms import AngleBatchPreprocessor
from angle_predictor.experiments.angle import _build_angle_optimizer
from angle_predictor.losses.angle import build_angle_loss
from angle_predictor.models.line_angle import AnglePredictor, build_angle_network


def measure(config_path: Path, iterations: int, warmup: int = 5) -> dict:
    params = AngleExperimentParams.model_validate(load_config(config_path).params)
    device = torch.device("cuda")
    model = AnglePredictor(
        AngleBatchPreprocessor(**params.preprocessing.model_dump()),
        build_angle_network(params.model, load_pretrained=False),
    ).to(device)
    result = {"parameters": sum(p.numel() for p in model.parameters())}
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
    parser.add_argument(
        "--baseline-config",
        type=Path,
        default=Path("configs/experiments/convnext_tiny_musgd_vector_charbonnier_tuned.yaml"),
    )
    parser.add_argument(
        "--candidate-config",
        type=Path,
        default=Path("configs/experiments/polar_line_musgd_vector_charbonnier.yaml"),
    )
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.manual_seed(42)
    torch.backends.cudnn.benchmark = True
    output = {"gpu": torch.cuda.get_device_name(0), "torch": torch.__version__, "precision": "bf16"}
    for name, config_path in (
        ("baseline", args.baseline_config),
        ("candidate", args.candidate_config),
    ):
        output[name] = measure(config_path, args.iterations)
        torch.cuda.empty_cache()
        print(name, json.dumps(output[name]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(output, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
