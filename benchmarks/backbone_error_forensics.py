from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


def bilinear(image: np.ndarray, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    x = np.clip(x, 0, image.shape[1] - 1.000001)
    y = np.clip(y, 0, image.shape[0] - 1.000001)
    ix, iy = x.astype(np.int32), y.astype(np.int32)
    ax, ay = (x - ix)[..., None], (y - iy)[..., None]
    return (
        image[iy, ix] * (1 - ax) * (1 - ay)
        + image[iy, ix + 1] * ax * (1 - ay)
        + image[iy + 1, ix] * (1 - ax) * ay
        + image[iy + 1, ix + 1] * ax * ay
    )


def pixel_proxies(image: np.ndarray, angle: float) -> dict:
    theta = math.radians(angle)
    radius = np.concatenate((np.arange(-125, -19), np.arange(20, 126)))
    offsets = np.array([0, -3, 3, -5, 5])[:, None]
    x = 127.5 + radius * math.cos(theta) - offsets * math.sin(theta)
    y = 127.5 + radius * math.sin(theta) + offsets * math.cos(theta)
    samples = bilinear(image, x, y).mean(axis=-1)
    contrast = samples[0] - np.median(samples[1:], axis=0)
    support = contrast > 15
    return {
        "pixel_image_mean_brightness": float(image.mean()),
        "pixel_image_std_brightness": float(image.std()),
        "pixel_positive_target_contrast": float(np.maximum(contrast, 0).mean()),
        "pixel_signed_target_contrast": float(contrast.mean()),
        "pixel_target_support_fraction": float(support.mean()),
        "pixel_support_fraction_negative_radius": float(support[radius < 0].mean()),
        "pixel_support_fraction_positive_radius": float(support[radius > 0].mean()),
        "pixel_target_contrast_p90": float(np.quantile(contrast, 0.9)),
    }


def summarize(mask: np.ndarray, baseline: np.ndarray, candidate: np.ndarray) -> dict:
    b, c = baseline[mask], candidate[mask]
    if not len(b):
        return {"samples": 0}
    return {
        "samples": len(b),
        "baseline_mae_deg": float(b.mean()),
        "candidate_mae_deg": float(c.mean()),
        "candidate_minus_baseline_deg": float((c - b).mean()),
        "candidate_better_fraction": float((c < b).mean()),
        "baseline_above_0.1_deg": int((b > 0.1).sum()),
        "candidate_above_0.1_deg": int((c > 0.1).sum()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Analyze paired line-angle prediction errors")
    parser.add_argument("--result-dir", type=Path, required=True)
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/synthetic_lines_150k"))
    args = parser.parse_args()
    root = args.result_dir
    with (root / "sample-errors.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    indices = np.array([int(r["dataset_index"]) for r in rows])
    b = np.array([float(r["baseline_error_deg"]) for r in rows])
    c = np.array([float(r["candidate_error_deg"]) for r in rows])
    meta = json.loads((args.dataset / "meta.json").read_text())
    labels = np.memmap(
        args.dataset / "labels.dat", dtype="float32", mode="r", shape=tuple(meta["labels_shape"])
    )
    target = labels[indices].astype(np.float64)
    angles = np.rad2deg(0.5 * np.arctan2(target[:, 0], target[:, 1])) % 180
    seam = np.minimum(angles, 180 - angles)
    vertical = np.abs(angles - 90)
    groups = {
        "all": np.ones(len(b), dtype=bool),
        "seam_within_2.5_deg": seam < 2.5,
        "vertical_within_2.5_deg": vertical < 2.5,
        "away_from_both_axes": (seam >= 2.5) & (vertical >= 2.5),
    }
    features_path = root.parent / "baseline_forensics/features.csv"
    cache_key = hashlib.sha256(
        (args.dataset / "meta.json").read_bytes() + indices.astype("<i8").tobytes()
    ).hexdigest()[:16]
    pixel_cache = root.parent / "pixel_features" / f"{cache_key}.csv"
    if pixel_cache.exists():
        features_path = pixel_cache
    pixel_features = {}
    if features_path.exists():
        with features_path.open(encoding="utf-8") as stream:
            pixel_features = {int(r["dataset_index"]): r for r in csv.DictReader(stream)}
    if not all(int(i) in pixel_features for i in indices):
        images = np.memmap(
            args.dataset / "images.dat",
            mode="r",
            dtype=meta["images_dtype"],
            shape=tuple(meta["images_shape"]),
        )
        features = []
        for index, angle in zip(indices, angles, strict=True):
            cached = pixel_features.get(int(index))
            proxy = (
                cached if cached is not None else pixel_proxies(images[int(index)], float(angle))
            )
            features.append(
                {
                    "dataset_index": int(index),
                    "pixel_positive_target_contrast": float(
                        proxy["pixel_positive_target_contrast"]
                    ),
                    "pixel_target_support_fraction": float(proxy["pixel_target_support_fraction"]),
                }
            )
        pixel_cache.parent.mkdir(parents=True, exist_ok=True)
        with pixel_cache.open("w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(features[0]))
            writer.writeheader()
            writer.writerows(features)
        pixel_features = {r["dataset_index"]: r for r in features}
    if pixel_features and all(int(i) in pixel_features for i in indices):
        contrast = np.array(
            [float(pixel_features[int(i)]["pixel_positive_target_contrast"]) for i in indices]
        )
        support = np.array(
            [float(pixel_features[int(i)]["pixel_target_support_fraction"]) for i in indices]
        )
        contrast_edges = np.quantile(contrast, [0.2, 0.4, 0.6, 0.8])
        contrast_bin = np.digitize(contrast, contrast_edges)
        for quintile in range(5):
            groups[f"pixel_target_contrast_quintile_Q{quintile + 1}"] = contrast_bin == quintile
        groups["pixel_target_support_below_0.25"] = support < 0.25
        groups["pixel_target_support_at_least_0.75"] = support >= 0.75
    for start in range(0, 180, 10):
        groups[f"angle_{start:03d}_{start + 10:03d}"] = (angles >= start) & (angles < start + 10)
    result = {
        "groups": {name: summarize(mask, b, c) for name, mask in groups.items()},
        "interpretation": (
            "Paired validation diagnostics; angle bins are descriptive, not a held-out test. "
            "No scene-level length/occlusion metadata; no causal conclusion from pixel proxies."
        ),
    }
    predictions_path = root / "sample-predictions.csv"
    if predictions_path.exists():
        with predictions_path.open(encoding="utf-8") as stream:
            predictions = list(csv.DictReader(stream))
        if [int(r["dataset_index"]) for r in predictions] != indices.tolist():
            raise ValueError("Prediction and error rows do not match")
        signed = {
            name: (np.array([float(r[f"{name}_angle_deg"]) for r in predictions]) - angles + 90)
            % 180
            - 90
            for name in ("baseline", "candidate")
        }
        result["signed_error_groups"] = {
            group: {name: float(errors[mask].mean()) for name, errors in signed.items()}
            for group, mask in groups.items()
            if mask.any()
        }
        result["signed_error_precision_note"] = (
            "Decoded float32 angle differences can differ by microdegrees from the "
            "cross/dot MAE metric. These values diagnose bias, not the acceptance criterion."
        )
    difference = c - b
    rng = np.random.default_rng(42)
    draws = [difference[rng.integers(0, len(b), len(b))].mean() for _ in range(1000)]
    result["paired_bootstrap_mae_difference_95_percent_deg"] = np.quantile(
        draws, [0.025, 0.975]
    ).tolist()
    selections = {
        "worst_candidate": np.argsort(c)[-16:][::-1],
        "worst_regressions": np.argsort(difference)[-16:][::-1],
        "largest_improvements": np.argsort(difference)[:16],
    }
    result["examples"] = {
        name: [
            {
                "dataset_index": int(indices[i]),
                "true_angle_deg": float(angles[i]),
                "baseline_error_deg": float(b[i]),
                "candidate_error_deg": float(c[i]),
            }
            for i in selection
        ]
        for name, selection in selections.items()
    }
    (root / "forensics.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    images = np.memmap(
        args.dataset / "images.dat", dtype="uint8", mode="r", shape=tuple(meta["images_shape"])
    )
    size = meta["img_size"]
    try:
        font = ImageFont.truetype("C:/Windows/Fonts/consola.ttf", 15)
    except OSError:
        font = ImageFont.load_default()
    for name, selection in selections.items():
        sheet = Image.new("RGB", (4 * size, 4 * (size + 65)), "#181c24")
        draw = ImageDraw.Draw(sheet)
        for n, i in enumerate(selection):
            x, y = (n % 4) * size, (n // 4) * (size + 65)
            sheet.paste(Image.fromarray(np.array(images[indices[i]])), (x, y))
            draw.text(
                (x + 5, y + size + 3),
                f"idx={indices[i]} GT={angles[i]:.4f}\nbase={b[i]:.4f} new={c[i]:.4f}",
                fill="white",
                font=font,
            )
        sheet.save(root / f"{name}.png")
    lines = [
        "# Парная форензика ошибок",
        "",
        "Статистика по тем же validation изображениям, что у соответствующего baseline. "
        "Это описательный анализ; длина линии и перекрытие не размечены.",
        "",
        "| Группа | N | Baseline MAE ° | Candidate MAE ° | Разница ° |",
        "| --- | ---: | ---: | ---: | ---: |",
    ]
    for name, stats in result["groups"].items():
        if stats["samples"]:
            lines.append(
                f"| {name} | {stats['samples']} "
                f"| {stats['baseline_mae_deg']:.6f} "
                f"| {stats['candidate_mae_deg']:.6f} "
                f"| {stats['candidate_minus_baseline_deg']:+.6f} |"
            )
    low, high = result["paired_bootstrap_mae_difference_95_percent_deg"]
    lines += [
        "",
        f"Парный bootstrap 95% CI разницы MAE: [{low:+.6f}, {high:+.6f}]°. "
        "Не учитывает отбор вариантов по validation и вариативность обучения; "
        "не заменяет replay на другом seed.",
        "",
        "[Все числа](forensics.json). "
        "На изображениях указаны ground truth угол и абсолютные ошибки.",
        "",
    ]
    for name in selections:
        lines += [f"## {name}", "", f"![{name}]({name}.png)", ""]
    (root / "forensics.md").write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps(result["groups"]["all"]), flush=True)


if __name__ == "__main__":
    main()
