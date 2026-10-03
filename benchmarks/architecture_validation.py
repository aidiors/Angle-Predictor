from __future__ import annotations

import argparse
import csv
import json
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
    parser.add_argument("--cache-dir", type=Path, default=Path("outputs/architecture-evaluation"))
    parser.add_argument("--export-predictions", action="store_true")
    args = parser.parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    client = MlflowClient(tracking_uri="http://127.0.0.1:5001")
    result = {"baseline_run": args.baseline_run, "candidate_run": args.candidate_run}
    all_errors = {}
    all_angles = {}
    histories = []
    expected_data = None
    expected_preprocessing = None
    for name, run_id in (("baseline", args.baseline_run), ("candidate", args.candidate_run)):
        run = client.get_run(run_id)
        if run.info.status != "FINISHED":
            raise ValueError(f"{name} run is not finished")
        cache = (args.cache_dir / run_id).resolve()
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
        loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
        model, _, precision = load_angle_predictor(path, torch.device("cuda"))
        errors = []
        angles = []
        with torch.inference_mode():
            for images, targets in loader:
                with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                    prediction = model(images.to("cuda", non_blocking=True))
                errors.append(_angle_errors_deg(prediction, targets.to("cuda")).cpu().numpy())
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
            "logged_best_mae_deg": float(run.data.tags["best_val_angle_mae_deg"]),
            "wall_time_minutes": (run.info.end_time - run.info.start_time) / 60000,
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
