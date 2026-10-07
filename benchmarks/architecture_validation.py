from __future__ import annotations

import argparse
import csv
import json
import os
import tempfile
from pathlib import Path

import numpy as np
import torch
from mlflow import MlflowClient
from torch.utils.data import DataLoader

from angle_predictor.config.angle_experiment import AngleExperimentParams
from angle_predictor.data.angle_dataset import AngleMemmapDataset
from angle_predictor.experiments.angle import _angle_errors_deg
from angle_predictor.inference.angle import load_angle_predictor


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare models on matching validation images")
    parser.add_argument("--baseline-run", required=True)
    parser.add_argument("--candidate-run", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--cache-dir", type=Path)
    parser.add_argument("--export-predictions", action="store_true")
    parser.add_argument(
        "--tracking-uri", default=os.environ.get("MLFLOW_TRACKING_URI", "http://127.0.0.1:5001")
    )
    parser.add_argument("--device", choices=("cpu", "cuda"), default="cuda")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--num-workers", type=int, default=4)
    args = parser.parse_args()
    if args.batch_size < 1 or args.num_workers < 0:
        parser.error("batch size must be positive and workers nonnegative")
    torch.set_num_threads(4)
    with tempfile.TemporaryDirectory(prefix="angle-validation-") as directory:
        compare(args, args.cache_dir or Path(directory))


def compare(args, cache_root: Path) -> None:
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri=args.tracking_uri)
    device = torch.device(args.device)
    expected_indices = None
    result = {
        "baseline_run": args.baseline_run,
        "candidate_run": args.candidate_run,
        "device": str(device),
        "split": "validation",
        "test_used": False,
    }
    all_errors = {}
    all_angles = {}
    histories = []
    expected_data = None
    expected_preprocessing = None
    for name, run_id in (("baseline", args.baseline_run), ("candidate", args.candidate_run)):
        run = client.get_run(run_id)
        if run.info.status != "FINISHED":
            raise ValueError(f"{name} run is not finished")
        end_time = run.info.end_time
        if end_time is None:
            raise ValueError(f"{name} run has no completion timestamp")
        cache = (cache_root / run_id).resolve()
        cache.mkdir(parents=True, exist_ok=True)
        path = client.download_artifacts(run_id, "checkpoints/best.pt", str(cache))
        payload = torch.load(path, weights_only=True, map_location="cpu")
        params = AngleExperimentParams.model_validate(payload["metadata"]["params"])
        data = params.data.model_dump(mode="json", exclude={"num_workers", "batch_size"})
        preprocessing = params.preprocessing.model_dump(mode="json")
        if expected_data is not None and (
            data != expected_data or preprocessing != expected_preprocessing
        ):
            raise ValueError("Baseline and candidate use different data or preprocessing")
        expected_data, expected_preprocessing = data, preprocessing
        dataset = AngleMemmapDataset(
            params.data.root,
            "val",
            seed=params.data.split_seed,
            train_fraction=params.data.train_fraction,
            val_fraction=params.data.val_fraction,
        )
        if expected_indices is not None and not np.array_equal(expected_indices, dataset.indices):
            raise ValueError("Validation indices differ")
        expected_indices = dataset.indices.copy()
        loader = DataLoader(
            dataset,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            pin_memory=device.type == "cuda",
        )
        model, _, precision = load_angle_predictor(path, device)
        errors = []
        angles = []
        with torch.inference_mode():
            for images, targets in loader:
                with torch.autocast(
                    device.type,
                    dtype=torch.bfloat16,
                    enabled=precision == "bf16" and device.type == "cuda",
                ):
                    prediction = model(images.to(device, non_blocking=True))
                errors.append(_angle_errors_deg(prediction, targets.to(device)).cpu().numpy())
                if args.export_predictions:
                    angles.append(
                        (0.5 * torch.atan2(prediction[:, 0], prediction[:, 1]))
                        .remainder(torch.pi)
                        .mul(180 / torch.pi)
                        .cpu()
                        .numpy()
                    )
        errors = np.concatenate(errors)
        all_errors[name] = errors
        if args.export_predictions:
            all_angles[name] = np.concatenate(angles)
        summary = {
            "samples": len(errors),
            "mae_deg": float(errors.mean(dtype=np.float64)),
            "median_deg": float(np.median(errors)),
            "p95_deg": float(np.quantile(errors, 0.95)),
            "p99_deg": float(np.quantile(errors, 0.99)),
            "max_deg": float(errors.max()),
            "fraction_above_0.1_deg": float((errors > 0.1).mean()),
            "logged_best_mae_deg": float(run.data.tags.get("best_val_angle_mae_deg", "nan")),
            "wall_time_minutes": (end_time - run.info.start_time) / 60000,
        }
        result[name] = summary
        print(name, json.dumps(summary), flush=True)
        for metric in ("val/angle_mae_deg", "train/angle_mae_deg"):
            histories.extend(
                {"model": name, "metric": metric, "step": item.step, "value": item.value}
                for item in client.get_metric_history(run_id, metric)
            )
        dataset.close()
        del model, payload
        if device.type == "cuda":
            torch.cuda.empty_cache()
    result["candidate_minus_baseline_mae_deg"] = (
        result["candidate"]["mae_deg"] - result["baseline"]["mae_deg"]
    )
    result["accuracy_preserved_on_this_split"] = (
        result["candidate"]["mae_deg"] <= result["baseline"]["mae_deg"]
    )
    (args.output_dir / "validation.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    with (args.output_dir / "metric-history.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=("model", "metric", "step", "value"))
        writer.writeheader()
        writer.writerows(histories)
    with (args.output_dir / "sample-errors.csv").open("w", newline="", encoding="utf-8") as stream:
        writer = csv.writer(stream)
        writer.writerow(("dataset_index", "baseline_error_deg", "candidate_error_deg"))
        writer.writerows(
            zip(dataset.indices, all_errors["baseline"], all_errors["candidate"], strict=True)
        )
    if args.export_predictions:
        with (args.output_dir / "sample-predictions.csv").open(
            "w", newline="", encoding="utf-8"
        ) as stream:
            writer = csv.writer(stream)
            writer.writerow(("dataset_index", "baseline_angle_deg", "candidate_angle_deg"))
            writer.writerows(
                zip(dataset.indices, all_angles["baseline"], all_angles["candidate"], strict=True)
            )


if __name__ == "__main__":
    main()
