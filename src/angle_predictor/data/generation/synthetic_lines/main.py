from __future__ import annotations

import argparse
import logging
from pathlib import Path

from .dataset_builder import generate_synthetic_dataset_memmap
from .environment_renderer import DEFAULT_ENVIRONMENTS_ROOT

PROJECT_ROOT = Path(__file__).resolve().parents[5]


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate synthetic line-angle data")
    parser.add_argument(
        "--output",
        type=Path,
        default=PROJECT_ROOT / "data" / "datasets" / "synthetic_memmap",
        help="Output directory for images.dat, labels.dat, and meta.json",
    )
    parser.add_argument("--num-samples", type=int, default=100_000)
    parser.add_argument("--img-size", type=int, default=256)
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Dataset seed; reuse it with the same inputs to reproduce every sample (default: 42)",
    )
    parser.add_argument("--batch-size", type=int, default=500)
    parser.add_argument("--workers", type=int, default=None)
    parser.add_argument(
        "--environments",
        type=Path,
        default=DEFAULT_ENVIRONMENTS_ROOT,
        help="Directory containing environment/*/background/background.lua",
    )
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main() -> None:
    args = _parse_args()
    logging.basicConfig(
        level=logging.INFO if args.verbose else logging.WARNING,
        format="%(levelname)s: %(message)s",
    )
    generate_synthetic_dataset_memmap(
        output_folder=args.output,
        num_samples=args.num_samples,
        img_size=args.img_size,
        environments_folder=args.environments,
        seed=args.seed,
        batch_size=args.batch_size,
        max_workers=args.workers,
    )


if __name__ == "__main__":
    main()
