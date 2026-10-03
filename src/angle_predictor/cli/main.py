from __future__ import annotations

import argparse
from pathlib import Path

from pydantic import ValidationError

from angle_predictor.config.loader import load_config
from angle_predictor.runner import run_experiment
from angle_predictor.tracking.mlflow import resolve_tracking_uri


def main() -> None:
    parser = argparse.ArgumentParser(prog="angle-train")
    commands = parser.add_subparsers(dest="command", required=True)
    run_parser = commands.add_parser("run", help="run one experiment configuration")
    run_parser.add_argument("--config", type=Path, required=True, help="path to a YAML config")
    run_parser.add_argument(
        "--set",
        dest="overrides",
        action="append",
        default=[],
        metavar="KEY=VALUE",
        help="override a dotted config key; may be specified more than once",
    )
    args = parser.parse_args()

    config_path = args.config.resolve()
    try:
        config = load_config(config_path, args.overrides)
        run_id = run_experiment(config, config_path)
    except (OSError, TypeError, ValueError, ValidationError) as error:
        parser.error(str(error))

    print(f"Run {run_id} logged to {resolve_tracking_uri(config.tracking)}")


if __name__ == "__main__":
    main()
