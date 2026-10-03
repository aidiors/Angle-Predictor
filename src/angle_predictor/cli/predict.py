from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import torch

from angle_predictor.tracking.mlflow import DEFAULT_TRACKING_URI


def main() -> None:
    parser = argparse.ArgumentParser(prog="angle-predict")
    model_source = parser.add_mutually_exclusive_group(required=True)
    model_source.add_argument("--checkpoint", type=Path)
    model_source.add_argument(
        "--model-uri",
        help="MLflow model URI, such as models:/angle-predictor@candidate",
    )
    parser.add_argument("--tracking-uri", help="MLflow tracking server URL for registry models")
    parser.add_argument(
        "--input", type=Path, required=True, help="RGB image or directory of images"
    )
    parser.add_argument("--output", type=Path, help="optional JSON output file")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", choices=("auto", "cpu", "cuda"), default="auto")
    args = parser.parse_args()
    if args.batch_size < 1:
        parser.error("--batch-size must be positive")
    if args.device == "cuda" and not torch.cuda.is_available():
        parser.error("CUDA is unavailable")
    device = torch.device(
        "cuda"
        if args.device == "cuda" or args.device == "auto" and torch.cuda.is_available()
        else "cpu"
    )
    if args.checkpoint is not None:
        from angle_predictor.inference.angle import load_angle_predictor

        model, input_size, precision = load_angle_predictor(args.checkpoint, device)
    else:
        import mlflow
        import mlflow.pytorch

        tracking_uri = (
            args.tracking_uri or os.environ.get("MLFLOW_TRACKING_URI") or DEFAULT_TRACKING_URI
        )
        mlflow.set_tracking_uri(tracking_uri)
        if args.model_uri is None:
            parser.error("--model-uri is required when --checkpoint is not set")
        model = mlflow.pytorch.load_model(args.model_uri)
        try:
            input_size = tuple(model.preprocessor.polar.input_size)
            precision = model.inference_precision
        except AttributeError as error:
            raise ValueError("The registry model is missing angle inference metadata") from error
        model.to(device).eval()

    from angle_predictor.inference.angle import load_rgb_image, predict_angle_batch

    if args.input.is_dir():
        paths = sorted(
            path
            for path in args.input.iterdir()
            if path.suffix.lower() in {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        )
    elif args.input.is_file():
        paths = [args.input]
    else:
        parser.error(f"Input path does not exist: {args.input}")
    if not paths:
        parser.error("No supported images found")
    results = []
    for start in range(0, len(paths), args.batch_size):
        chunk = paths[start : start + args.batch_size]
        images = torch.stack([load_rgb_image(path, input_size) for path in chunk])
        angles, vectors = predict_angle_batch(model, images, precision)
        for path, angle, vector in zip(chunk, angles, vectors, strict=True):
            results.append(
                {
                    "image": str(path),
                    "angle_deg": angle.item(),
                    "sin_2theta": vector[0].item(),
                    "cos_2theta": vector[1].item(),
                }
            )
    rendered = json.dumps(results, ensure_ascii=False, indent=2)
    if args.output is not None:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered + "\n", encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
