from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import os
import shutil
import statistics
import subprocess
import sys
import tempfile
import time
from contextlib import suppress
from copy import deepcopy
from io import StringIO
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from mlflow import MlflowClient
from torch.utils.data import DataLoader

from angle_predictor.config.angle_experiment import AngleExperimentParams, CartesianRefinementConfig
from angle_predictor.config.loader import load_config
from angle_predictor.data.angle_dataset import AngleMemmapDataset
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.engine.seed import seed_everything
from angle_predictor.experiments.angle import (
    _angle_errors_deg,
    _build_angle_optimizer,
    _build_model,
)
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.losses.angle import build_angle_loss
from angle_predictor.models.cartesian import CartesianRefinement


def write_json(path: Path, value: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2), encoding="utf-8")
    temporary.replace(path)


def sha256(path: Path) -> str:
    checksum = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            checksum.update(chunk)
    return checksum.hexdigest()


def archive_result(client: MlflowClient, output: Path, result: dict) -> dict:
    run_id, key = result["run_id"], result["key"]
    if client.get_run(run_id).info.status != "FINISHED":
        raise RuntimeError(f"Cannot archive unfinished run {run_id}")
    evaluation = output / key
    client.log_artifacts(run_id, str(evaluation), "comparison/evaluation")
    files = [(path, f"comparison/evaluation/{path.name}") for path in evaluation.iterdir()]
    for path in (output / "configs" / f"{key}.yaml", output / "logs" / f"{key}.log"):
        if path.is_file():
            client.log_artifact(run_id, str(path), "comparison/execution")
            files.append((path, f"comparison/execution/{path.name}"))
    if "checkpoint" in result:
        checkpoint = Path(result["checkpoint"])
        for name in ("best.pt", "last.pt"):
            path = checkpoint.parent / name
            if not path.is_file():
                raise FileNotFoundError(path)
            files.append((path, f"checkpoints/{name}"))
    verified = []
    with tempfile.TemporaryDirectory(prefix="angle-mlflow-verify-") as directory:
        for path, artifact in files:
            if path.is_file():
                restored = Path(client.download_artifacts(run_id, artifact, directory))
                if sha256(path) != sha256(restored):
                    raise RuntimeError(f"MLflow artifact differs from staging: {artifact}")
                verified.append(artifact)
    client.log_dict(run_id, result, "comparison/result.json")
    for name, value in result["metrics"].items():
        client.log_metric(run_id, f"comparison/{name}", value, step=result["total_steps"])
    return {"key": key, "run_id": run_id, "verified_artifacts": verified}


def finalize_comparison(client: MlflowClient, output: Path, report: Path) -> None:
    state = json.loads((output / "state.json").read_text())
    if state["status"] != "COMPLETE" or state.get("child_pid"):
        raise RuntimeError("Staging must be retained until all GPU work has completed")
    assert_idle(client)
    output = output.resolve()
    project_outputs = Path(__file__).resolve().parents[1] / "outputs"
    if not output.is_relative_to(project_outputs) or output == project_outputs:
        raise ValueError("Automatic cleanup requires a staging subdirectory inside project outputs")
    manifest = [archive_result(client, output, result) for result in state["results"]]
    campaign_run = state["results"][0]["run_id"]
    for path in (output / "state.json", output / "preflight.json"):
        client.log_artifact(campaign_run, str(path), "comparison/campaign")
    for path in output.glob("features_seed*.npz"):
        client.log_artifact(campaign_run, str(path), "comparison/campaign")
    client.log_dict(
        campaign_run, {"runs": manifest}, "comparison/campaign/artifact-verification.json"
    )
    rows = [
        {
            "model": result["stage"],
            "seed": result["seed"],
            "total_steps": result["total_steps"],
            **{key: value for key, value in result["metrics"].items() if key != "seed"},
        }
        for result in state["results"]
    ]
    report.mkdir(parents=True, exist_ok=True)
    stream = StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    exported = stream.getvalue()
    results_path = report / "results.csv"
    if results_path.exists() and results_path.read_text(encoding="utf-8") != exported:
        raise RuntimeError("Existing completed report differs, choose a new report directory")
    results_path.write_text(exported, encoding="utf-8")
    shutil.rmtree(output)
    with suppress(OSError):
        project_outputs.rmdir()
    print("ARCHIVED_AND_CLEANED", output, flush=True)


def assert_idle(client: MlflowClient) -> None:
    experiments = client.search_experiments(max_results=1000)
    runs = client.search_runs(
        [x.experiment_id for x in experiments], "attributes.status = 'RUNNING'", max_results=1000
    )
    if runs:
        raise RuntimeError(f"Existing RUNNING runs: {[x.info.run_id for x in runs]}")


def config_params(path: Path, output: Path) -> AngleExperimentParams:
    return AngleExperimentParams.model_validate(
        load_config(path, [f"params.training.output_dir={output.as_posix()}"]).params
    )


def preflight(config_dir: Path, output: Path) -> None:
    client = MlflowClient(tracking_uri="http://127.0.0.1:5001")
    assert_idle(client)
    checks = []
    for seed in (42, 43):
        for stage in ("plain", "spatial", "refinement"):
            assert_idle(client)
            seed_everything(seed)
            params = config_params(config_dir / f"{stage}_seed{seed}.yaml", output)
            with tempfile.TemporaryDirectory(prefix="angle-cartesian-smoke-") as directory:
                if stage == "refinement":
                    source = config_params(config_dir / f"spatial_seed{seed}.yaml", output)
                    coarse_model = _build_model(source, (256, 256))
                    source_path = Path(directory) / "coarse.pt"
                    save_checkpoint(
                        coarse_model,
                        source_path,
                        step=0,
                        metadata={
                            "format_version": 1,
                            "run_id": "0" * 32,
                            "params": source.model_dump(mode="json"),
                            "input_size": [256, 256],
                            "dataset_meta_sha256": hashlib.sha256(
                                (source.data.root / "meta.json").read_bytes()
                            ).hexdigest(),
                        },
                    )
                    assert isinstance(params.model, CartesianRefinementConfig)
                    params.model.coarse_checkpoint = source_path
                    del coarse_model
                model = _build_model(params, (256, 256)).cuda()
                if stage == "refinement":
                    assert isinstance(model.network, CartesianRefinement)
                    original = {
                        key: value.cpu().clone()
                        for key, value in model.network.coarse.state_dict().items()
                    }
                optimizer = _build_angle_optimizer(model.network, params)
                criterion = build_angle_loss(**params.loss.model_dump())
                images = torch.randint(0, 256, (32, 3, 256, 256), dtype=torch.uint8, device="cuda")
                targets = torch.nn.functional.normalize(torch.randn(32, 2, device="cuda"), dim=-1)
                model.train()
                torch.cuda.reset_peak_memory_stats()
                for _ in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        vectors, labels = model.forward_augmented(images, targets)
                        loss = criterion(vectors, labels)
                    assert torch.isfinite(vectors).all() and torch.isfinite(loss)
                    loss.backward()
                    gradients = [p.grad for p in model.parameters() if p.grad is not None]
                    assert gradients and all(torch.isfinite(g).all() for g in gradients)
                    assert any(g.abs().sum() > 0 for g in gradients)
                    optimizer.step()
                if stage == "refinement":
                    assert isinstance(model.network, CartesianRefinement)
                    assert not model.network.coarse.training
                    assert all(p.grad is None for p in model.network.coarse.parameters())
                    assert all(
                        torch.equal(value.cpu(), original[key])
                        for key, value in model.network.coarse.state_dict().items()
                    )
                    assert any(
                        p.grad is not None and p.grad.abs().sum() > 0
                        for p in model.network.fine.parameters()
                    )
                model.eval()
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    expected = model(images[:1]).clone()
                checkpoint = Path(directory) / "portable.pt"
                metadata = {
                    "format_version": 1,
                    "input_size": [256, 256],
                    "params": params.model_dump(mode="json"),
                }
                metadata["params"]["data"]["root"] = "missing-data"
                if stage == "refinement":
                    metadata["params"]["model"]["coarse_checkpoint"] = "missing-parent.pt"
                save_checkpoint(model, checkpoint, step=0, metadata=metadata)
                restored, _, _ = load_angle_predictor(checkpoint, torch.device("cuda"))
                with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
                    assert torch.equal(expected, restored(images[:1]))
                checks.append(
                    {
                        "seed": seed,
                        "stage": stage,
                        "parameters": sum(p.numel() for p in model.parameters()),
                        "peak_mib": torch.cuda.max_memory_allocated() / 2**20,
                        "finite_gradients": True,
                        "portable_bitwise": True,
                    }
                )
                print("PREFLIGHT", json.dumps(checks[-1]), flush=True)
                del restored, model, optimizer, images, targets, vectors, loss, gradients
                gc.collect()
                torch.cuda.empty_cache()
    write_json(output / "preflight.json", checks)


def evaluate(checkpoint: Path, seed: int, root: Path, output: Path) -> dict:
    model, _, precision = load_angle_predictor(checkpoint, torch.device("cuda"))
    payload = torch.load(checkpoint, map_location="cpu", weights_only=True)
    params = AngleExperimentParams.model_validate(payload["metadata"]["params"])
    assert params.data.split_seed == seed
    dataset = AngleMemmapDataset(root, "val", seed=seed, train_fraction=0.8, val_fraction=0.1)
    assert len(dataset) == 15000
    loader = DataLoader(dataset, batch_size=32, num_workers=4, pin_memory=True)
    errors, vectors = [], []
    with torch.inference_mode():
        for images, targets in loader:
            with torch.autocast("cuda", dtype=torch.bfloat16, enabled=precision == "bf16"):
                prediction = model(images.cuda(non_blocking=True))
            errors.append(_angle_errors_deg(prediction, targets.cuda()).cpu().numpy())
            vectors.append(prediction.float().cpu().numpy())
    values = np.concatenate(errors)
    predictions = np.concatenate(vectors)
    summary = {
        "seed": seed,
        "samples": len(values),
        "mae_deg": float(values.mean(dtype=np.float64)),
        "median_deg": float(np.median(values)),
        "p95_deg": float(np.quantile(values, 0.95)),
        "p99_deg": float(np.quantile(values, 0.99)),
        "max_deg": float(values.max()),
        "above_0_1": int((values > 0.1).sum()),
        "parameters": sum(p.numel() for p in model.parameters()),
        "best_step": payload["step"],
    }
    output.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        output / "validation.npz", indices=dataset.indices, errors=values, vectors=predictions
    )
    images = images[:1].cuda()
    timings = []
    with torch.inference_mode(), torch.autocast("cuda", dtype=torch.bfloat16):
        for i in range(45):
            torch.cuda.synchronize()
            start = time.perf_counter()
            model(images)
            torch.cuda.synchronize()
            if i >= 5:
                timings.append((time.perf_counter() - start) * 1000)
    summary["inference_batch1_median_ms"] = statistics.median(timings)
    write_json(output / "metrics.json", summary)
    dataset.close()
    del model, payload, loader, dataset
    gc.collect()
    torch.cuda.empty_cache()
    return summary


def audit_run(client: MlflowClient, run_id: str, steps: int) -> dict:
    run = client.get_run(run_id)
    if run.info.status != "FINISHED":
        raise RuntimeError(f"Run {run_id} is {run.info.status}")
    history = client.get_metric_history(run_id, "train/loss")
    if not history or max(x.step for x in history) != steps:
        raise RuntimeError("Incomplete training budget")
    expected = set(range(1300, steps + 1, 1300)) | {steps}
    actual = {x.step for x in client.get_metric_history(run_id, "val/angle_mae_deg")}
    if actual != expected:
        raise RuntimeError("Validation milestones differ")
    return {"status": run.info.status, "last_train_step": steps, "validation_steps": sorted(actual)}


def run_comparison(config_dir: Path, output: Path, references: list[str]) -> None:
    client = MlflowClient(tracking_uri="http://127.0.0.1:5001")
    state_path = output / "state.json"
    state: dict[str, Any] = (
        json.loads(state_path.read_text()) if state_path.exists() else {"results": []}
    )
    state["runner_pid"] = os.getpid()
    ready = json.loads((output / "preflight.json").read_text())
    assert len(ready) == 6
    for seed, reference in zip((42, 43), references, strict=True):
        own = {}
        for stage in ("plain", "spatial", "refinement", "continuation"):
            key = f"{stage}_seed{seed}"
            existing = next((x for x in state["results"] if x["key"] == key), None)
            if existing:
                audit_run(client, existing["run_id"], existing["new_steps"])
                own[stage] = existing
                continue
            assert_idle(client)
            path = config_dir / f"{stage}_seed{seed}.yaml"
            options = deepcopy(yaml.safe_load(path.read_text()))
            training = options["params"]["training"]
            training["output_dir"] = str((output / "checkpoints").resolve())
            if stage in ("refinement", "continuation"):
                coarse = own["spatial"]
                options["params"]["model"].update(
                    coarse_checkpoint=coarse["checkpoint"], coarse_run_id=coarse["run_id"]
                )
            if stage == "continuation":
                fine = own["refinement"]
                training.update(
                    initialization_checkpoint=fine["checkpoint"],
                    initialization_run_id=fine["run_id"],
                )
            effective = output / "configs" / f"{key}.yaml"
            effective.parent.mkdir(parents=True, exist_ok=True)
            if not effective.exists():
                effective.write_text(yaml.safe_dump(options, sort_keys=False), encoding="utf-8")
            else:
                assert yaml.safe_load(effective.read_text()) == options
            config = load_config(
                effective, [f"params.training.output_dir={training['output_dir']}"]
            )
            experiment = client.get_experiment_by_name(config.tracking.experiment_name)
            matching = (
                client.search_runs(
                    [experiment.experiment_id],
                    f"tags.mlflow.runName = '{config.name}'",
                    max_results=10,
                )
                if experiment
                else []
            )
            state.update(
                status="TRAINING", current=key, run_id=matching[0].info.run_id if matching else None
            )
            write_json(state_path, state)
            if not matching:
                log = output / "logs" / f"{key}.log"
                log.parent.mkdir(parents=True, exist_ok=True)
                with log.open("w", encoding="utf-8") as stream:
                    child = subprocess.Popen(
                        [
                            sys.executable,
                            "-m",
                            "angle_predictor.cli.main",
                            "run",
                            "--config",
                            str(effective),
                            "--set",
                            f"params.training.output_dir={training['output_dir']}",
                        ],
                        stdout=stream,
                        stderr=subprocess.STDOUT,
                    )
                    state["child_pid"] = child.pid
                    write_json(state_path, state)
                    returncode = child.wait()
                    state["child_pid"] = None
                    write_json(state_path, state)
                    if returncode:
                        raise RuntimeError(f"Training child failed with code {returncode}: {log}")
                experiment = client.get_experiment_by_name(config.tracking.experiment_name)
                if experiment is None:
                    raise RuntimeError("Training experiment was not created")
                matching = client.search_runs(
                    [experiment.experiment_id],
                    f"tags.mlflow.runName = '{config.name}'",
                    max_results=10,
                )
            if len(matching) != 1:
                raise RuntimeError("Ambiguous run identity")
            run_id = matching[0].info.run_id
            audit = audit_run(client, run_id, training["max_steps"])
            assert_idle(client)
            checkpoint = Path(training["output_dir"]) / config.name / run_id / "best.pt"
            if not checkpoint.is_file():
                destination = checkpoint.parent
                checkpoint = Path(
                    client.download_artifacts(run_id, "checkpoints/best.pt", str(destination))
                )
                client.download_artifacts(run_id, "checkpoints/last.pt", str(destination))
            state.update(status="VALIDATION_AND_PROFILING", current=key, run_id=run_id)
            write_json(state_path, state)
            metrics = evaluate(checkpoint, seed, Path(config.params["data"]["root"]), output / key)
            total = (
                15600
                if stage in ("plain", "continuation")
                else 5200
                if stage == "spatial"
                else 10400
            )
            lineage = [] if stage in ("plain", "spatial") else [own["spatial"]["run_id"]]
            if stage == "continuation":
                lineage.append(own["refinement"]["run_id"])
            if stage in ("refinement", "continuation"):
                source = torch.load(
                    own["spatial"]["checkpoint"], map_location="cpu", weights_only=True
                )
                final = torch.load(checkpoint, map_location="cpu", weights_only=True)
                assert all(
                    torch.equal(
                        value,
                        final["model_state_dict"]["network.coarse." + key.removeprefix("network.")],
                    )
                    for key, value in source["model_state_dict"].items()
                    if key.startswith("network.")
                )
                del source, final
            result = {
                "key": key,
                "seed": seed,
                "stage": stage,
                "run_id": run_id,
                "new_steps": training["max_steps"],
                "total_steps": total,
                "source_run_ids": lineage,
                "checkpoint": str(checkpoint),
                "audit": audit,
                "metrics": metrics,
            }
            assert total <= 15600
            state["results"].append(result)
            own[stage] = result
            write_json(state_path, state)
            print("COMPLETED", key, json.dumps(metrics), flush=True)
        reference_key = f"polar_seed{seed}"
        if not next((x for x in state["results"] if x["key"] == reference_key), None):
            assert_idle(client)
            audit_run(client, reference, 5200)
            destination = output / "reference-checkpoints" / reference
            destination.mkdir(parents=True, exist_ok=True)
            checkpoint = Path(
                client.download_artifacts(reference, "checkpoints/best.pt", str(destination))
            )
            metrics = evaluate(
                checkpoint, seed, Path(config.params["data"]["root"]), output / reference_key
            )
            state["results"].append(
                {
                    "key": reference_key,
                    "seed": seed,
                    "stage": "polar",
                    "run_id": reference,
                    "total_steps": 15600,
                    "metrics": metrics,
                }
            )
            write_json(state_path, state)
            print("REFERENCE", seed, json.dumps(metrics), flush=True)
    state.update(status="COMPLETE", current=None, run_id=None, child_pid=None)
    write_json(state_path, state)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and compare Cartesian line-angle controls")
    parser.add_argument("mode", choices=("preflight", "run", "finalize"))
    parser.add_argument(
        "--config-dir", type=Path, default=Path("configs/experiments/cartesian_comparison")
    )
    parser.add_argument("--output-dir", type=Path, default=Path("outputs/cartesian-comparison"))
    parser.add_argument("--reference-runs", nargs=2)
    parser.add_argument(
        "--report-dir", type=Path, default=Path("docs/reports/cartesian-comparison/data")
    )
    parser.add_argument("--keep-staging", action="store_true")
    args = parser.parse_args()
    torch.set_num_threads(4)
    torch.backends.cudnn.benchmark = True
    os.environ["MLFLOW_DISABLE_AGENT_HINT"] = "1"
    args.output_dir.mkdir(parents=True, exist_ok=True)
    lock = args.output_dir / "runner.lock"
    with lock.open("x") as stream:
        stream.write(str(os.getpid()))
    try:
        if args.mode == "preflight":
            preflight(args.config_dir, args.output_dir)
        elif args.mode == "run":
            if not args.reference_runs:
                parser.error("run requires two reference run IDs")
            run_comparison(args.config_dir, args.output_dir, args.reference_runs)
            if not args.keep_staging:
                finalize_comparison(
                    MlflowClient(tracking_uri="http://127.0.0.1:5001"),
                    args.output_dir,
                    args.report_dir,
                )
        else:
            finalize_comparison(
                MlflowClient(tracking_uri="http://127.0.0.1:5001"),
                args.output_dir,
                args.report_dir,
            )
    finally:
        lock.unlink(missing_ok=True)


if __name__ == "__main__":
    main()
