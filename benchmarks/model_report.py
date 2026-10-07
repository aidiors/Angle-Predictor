from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

PROJECT = Path(__file__).resolve().parents[1]
COLORS = ["#8b9bb4", "#d49a51", "#7564b5", "#178578"]


def save(figure, directory: Path, name: str) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    figure.savefig(directory / f"{name}.png", dpi=180, bbox_inches="tight")
    svg = directory / f"{name}.svg"
    figure.savefig(svg, bbox_inches="tight")
    svg.write_text(
        "\n".join(line.rstrip() for line in svg.read_text(encoding="utf-8").splitlines()) + "\n",
        encoding="utf-8",
    )
    plt.close(figure)


def quality(rows: list[dict], directory: Path) -> None:
    models = ["rgb_plain", "rgb_refinement", "previous_polar", "four_view"]
    names = [
        "Stock RGB\nConvNeXt Tiny",
        "Spatial RGB\nrefinement",
        "Previous polar\nsingle view",
        "Primary polar\nfour views",
    ]
    figure, axes = plt.subplots(1, 2, figsize=(12, 4.4))
    for axis, metric, title in zip(
        axes, ["mae_deg", "p99_deg"], ["Mean absolute error", "99th-percentile error"], strict=True
    ):
        for index, model in enumerate(models):
            values = [
                float(
                    next(r for r in rows if r["model"] == model and r["seed"] == str(seed))[metric]
                )
                for seed in [42, 43]
            ]
            positions = np.array([0, 1]) + (index - 1.5) * 0.19
            bars = axis.bar(positions, values, width=0.18, color=COLORS[index], label=names[index])
            axis.bar_label(bars, labels=[f"{v:.4f}" for v in values], padding=3, fontsize=8)
        axis.set(
            yscale="log",
            xticks=[0, 1],
            xticklabels=["Seed 42", "Seed 43"],
            title=title,
            ylabel="Angular error (degrees, log scale)",
        )
        axis.set_ylim(
            bottom=0.0009 if metric == "mae_deg" else 0.009,
            top=0.085 if metric == "mae_deg" else 0.7,
        )
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, frameon=False, fontsize=9)
    figure.subplots_adjust(top=0.78, bottom=0.2, wspace=0.3)
    figure.text(
        0.5,
        0.02,
        (
            "15 600 task updates per model · 15 000 matched validation images per seed\n"
            "RGB / previous-polar exports are from different BF16 sessions · Test split unused"
        ),
        ha="center",
        fontsize=9,
        color="#536175",
    )
    save(figure, directory, "quality")


def symmetry(rows: list[dict], directory: Path) -> None:
    figure, axes = plt.subplots(1, 3, figsize=(13, 4.2))
    for axis, metric, title in zip(
        axes, ["mae_deg", "p99_deg", "max_deg"], ["MAE", "P99", "Maximum error"], strict=True
    ):
        for index, model in enumerate(["angular", "four_view"]):
            values = [
                float(
                    next(r for r in rows if r["model"] == model and r["seed"] == str(seed))[metric]
                )
                for seed in [42, 43, 44]
            ]
            bars = axis.bar(
                np.arange(3) + (index - 0.5) * 0.35,
                values,
                width=0.33,
                color=[COLORS[2], COLORS[3]][index],
                label=["Angular antisymmetry", "Four-view projection"][index],
            )
            axis.bar_label(
                bars,
                labels=[f"{v:.5f}" if metric != "max_deg" else f"{v:.3f}" for v in values],
                padding=3,
                fontsize=8,
            )
        axis.set(
            xticks=range(3),
            xticklabels=["Seed 42", "Seed 43", "Seed 44"],
            title=title,
            ylabel="Degrees",
        )
        axis.set_ylim(0, max(p.get_height() for p in axis.patches) * 1.25)
        axis.grid(axis="y", alpha=0.2)
        axis.set_axisbelow(True)
    handles, labels = axes[0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=2, frameon=False)
    figure.subplots_adjust(top=0.8, bottom=0.18, wspace=0.35)
    figure.text(
        0.5,
        0.02,
        (
            "Paired validation · 15 600 updates per model · Four vs two fine views\n"
            "Errors above 0.1°: 6 → 5 / 4 → 3 / 6 → 7"
        ),
        ha="center",
        fontsize=9,
        color="#536175",
    )
    save(figure, directory, "symmetry")


def architecture(directory: Path) -> None:
    figure, axis = plt.subplots(figsize=(13, 6.2))
    axis.set(xlim=(0, 13), ylim=(0, 6.2))
    axis.axis("off")

    def box(x, y, width, height, text, color="#e9eef5", size=10):
        axis.add_patch(
            FancyBboxPatch(
                (x, y),
                width,
                height,
                boxstyle="round,pad=0.07",
                facecolor=color,
                edgecolor="#c1ccd9",
            )
        )
        axis.text(
            x + width / 2,
            y + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=size,
            color="#1f3048",
        )

    def arrow(start, end):
        axis.add_patch(
            FancyArrowPatch(
                start, end, arrowstyle="-|>", mutation_scale=13, linewidth=1.5, color="#536175"
            )
        )

    box(0.2, 4.6, 1.6, 1.0, "RGB input\n256 × 256")
    box(2.2, 4.6, 2.0, 1.0, "Signed polar\n384 × 360")
    box(4.6, 4.6, 3.0, 1.0, "ConvNeXt Tiny + angular head\nCoarse direction, once")
    box(8.2, 4.6, 2.4, 1.0, "Local polar crop C\n384 × 65, ±4°")
    arrow((1.8, 5.1), (2.2, 5.1))
    arrow((4.2, 5.1), (4.6, 5.1))
    arrow((7.6, 5.1), (8.2, 5.1))
    box(0.3, 2.6, 2.2, 1.1, "Four crop views\nC, AC, RC, ARC", "#e8e3f4")
    box(
        3.1,
        2.3,
        4.2,
        1.7,
        (
            "Shared fine CNN\n"
            "RGB + fixed signed / absolute radius\n"
            "16 → 32 → 128, GroupNorm + SiLU\n"
            "Antialias + 5 × 23, strides 2/2/2"
        ),
        "#e2f1ed",
    )
    box(
        7.9,
        2.3,
        4.2,
        1.7,
        (
            "Shared per-view readout\n"
            "Radial mean + max, all 65 angles\n"
            "MLP 16 642 → 128 → 1\n"
            "b(view) = 4° × tanh(score)"
        ),
        "#e2f1ed",
    )
    arrow((9.4, 4.6), (9.4, 4.2))
    arrow((9.4, 4.2), (1.4, 4.2))
    arrow((1.4, 4.2), (1.4, 3.7))
    arrow((2.5, 3.15), (3.1, 3.15))
    arrow((7.3, 3.15), (7.9, 3.15))
    box(
        3.1,
        0.3,
        9.0,
        1.2,
        (
            "δ = [b(C) − b(AC) + b(RC) − b(ARC)] / 4\n"
            "Rotate coarse [sin(2θ), cos(2θ)] by 2δ\n"
            "Angular reversal changes sign · Radial reversal preserves correction"
        ),
        "#d5ebe5",
        11,
    )
    arrow((10, 2.3), (10, 1.5))
    axis.text(
        0.3,
        1.2,
        "A: reverse angular columns\nR: reverse radial rows\n34 033 768 parameters",
        fontsize=10,
        va="center",
        color="#536175",
    )
    axis.set_title(
        "Primary architecture: four-view signed polar refinement", loc="left", fontsize=16, pad=12
    )
    save(figure, directory, "architecture")


def examples(report: Path, dataset: Path, directory: Path) -> None:
    meta = json.loads((dataset / "meta.json").read_text(encoding="utf-8"))
    images = np.memmap(
        dataset / "images.dat",
        dtype=meta["images_dtype"],
        mode="r",
        shape=tuple(meta["images_shape"]),
    )
    evidence = json.loads((report / "data/visual-examples.json").read_text(encoding="utf-8"))
    for item in evidence["examples"]:
        image = images[item["dataset_index"]]
        radius = np.linspace(-127.5, 127.5, 384)[:, None]
        theta = np.linspace(0, np.pi, 360, endpoint=False)[None, :]
        x = np.clip(127.5 + radius * np.cos(theta), 0, 254.999999)
        y = np.clip(127.5 + radius * np.sin(theta), 0, 254.999999)
        ix, iy = x.astype(int), y.astype(int)
        ax, ay = (x - ix)[..., None], (y - iy)[..., None]
        polar = (
            image[iy, ix] * (1 - ax) * (1 - ay)
            + image[iy, ix + 1] * ax * (1 - ay)
            + image[iy + 1, ix] * (1 - ax) * ay
            + image[iy + 1, ix + 1] * ax * ay
        )
        figure, axes = plt.subplots(1, 4, figsize=(12, 3.5))
        axes[0].imshow(image)
        axes[0].set_title("Input RGB")
        axes[1].imshow(np.clip(polar / 255, 0, 1), aspect="auto")
        axes[1].set_title("Signed polar representation")
        for axis, prefix, name in [
            (axes[2], "plain", "Stock RGB ConvNeXt"),
            (axes[3], "four_view", "Primary four-view model"),
        ]:
            axis.imshow(image)
            for angle, color, style in [
                (item["target_deg"], "#40e285", "--"),
                (item[f"{prefix}_angle_deg"], "#ff5858", "-"),
            ]:
                rad = np.deg2rad(angle)
                axis.plot(
                    127.5 + np.array([-180, 180]) * np.cos(rad),
                    127.5 + np.array([-180, 180]) * np.sin(rad),
                    color=color,
                    linestyle=style,
                    linewidth=1.3,
                )
            axis.set(
                xlim=(0, 255),
                ylim=(255, 0),
                title=f"{name}\nError {item[f'{prefix}_error_deg']:.6f}°",
            )
        for axis in axes:
            axis.set_xticks([])
            axis.set_yticks([])
        figure.suptitle(
            f"Validation image {item['dataset_index']} · Target {item['target_deg']:.4f}°",
            fontsize=12,
        )
        figure.text(
            0.5,
            0.02,
            "Green dashed: target · Red: prediction · Selected illustrative examples, seed 42",
            ha="center",
            fontsize=9,
        )
        figure.subplots_adjust(top=0.77, bottom=0.13, wspace=0.12)
        save(figure, directory, item["filename"])


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Regenerate primary-model figures from saved evidence without inference"
    )
    parser.add_argument(
        "--report-dir", type=Path, default=PROJECT / "docs/reports/four-view-refinement"
    )
    parser.add_argument(
        "--dataset",
        type=Path,
        help="Optional memmap dataset for the three prediction illustrations",
    )
    args = parser.parse_args()
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.facecolor": "white",
            "axes.titleweight": "normal",
        }
    )
    with (args.report_dir / "data/results.csv").open(encoding="utf-8") as stream:
        rows = list(csv.DictReader(stream))
    directory = args.report_dir / "figures"
    quality(rows, directory)
    symmetry(rows, directory)
    architecture(directory)
    if args.dataset is not None:
        examples(args.report_dir, args.dataset, directory)


if __name__ == "__main__":
    main()
