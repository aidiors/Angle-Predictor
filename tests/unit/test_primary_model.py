import json
from pathlib import Path
from unittest import TestCase

from angle_predictor.config.angle_experiment import AngleExperimentParams
from angle_predictor.config.loader import load_config
from benchmarks.architecture_speed import operation_counts


class PrimaryModelTests(TestCase):
    def test_reproduction_configs_match_selected_architecture_and_frozen_training(self):
        selection = json.loads(Path("configs/final/selection.json").read_text(encoding="utf-8"))
        fresh = AngleExperimentParams.model_validate(
            load_config(Path("configs/final/refinement.yaml")).params
        )
        continuation = AngleExperimentParams.model_validate(
            load_config(Path("configs/final/continuation.yaml")).params
        )
        self.assertEqual(fresh.model, continuation.model)
        options = fresh.model.model_dump(mode="json")
        for name in (
            "crop_size",
            "fine_channels",
            "fine_angular_antisymmetry",
            "fine_radial_reflection",
            "fine_reflection_readout",
            "train_coarse",
        ):
            self.assertEqual(options[name], selection["model"][name])
        self.assertEqual(fresh.training.max_steps, 5200)
        self.assertEqual(continuation.training.max_steps, 5200)
        self.assertEqual(selection["default_seed"], 42)
        self.assertEqual({row["seed"] for row in selection["checkpoints"]}, {42, 43, 44})

    def test_shape_counter_separates_one_coarse_pass_from_four_shared_fine_views(self):
        counts = operation_counts(Path("configs/final/refinement.yaml"))
        self.assertEqual(counts["coarse_macs"], 12132314368)
        self.assertEqual(counts["fine_macs"], 4 * 1954002304)
        self.assertEqual(counts["total_macs"], 19948323584)
        self.assertAlmostEqual(counts["gflops_two_per_mac"], 39.896647168)
