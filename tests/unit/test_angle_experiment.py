from pathlib import Path
from unittest import TestCase
from unittest.mock import patch

import torch

from angle_predictor.config.angle_experiment import AngleExperimentParams, AngleLossConfig
from angle_predictor.config.loader import load_config
from angle_predictor.engine.optimizers import build_optimizer
from angle_predictor.experiments.angle import _scheduler, _train_angle_mae_deg


class AngleExperimentTests(TestCase):
    def test_train_angle_mae_is_reported_in_degrees(self) -> None:
        value = _train_angle_mae_deg(torch.tensor(45.0), sample_count=3)
        self.assertAlmostEqual(value, 15.0)

    def test_selected_baseline_config_and_optimizer(self) -> None:
        config = load_config(
            Path("configs/experiments/convnext_tiny_musgd_vector_charbonnier.yaml")
        )
        params = AngleExperimentParams.model_validate(config.params)
        self.assertEqual(params.data.batch_size, 32)
        self.assertEqual(params.model.backbone_name, "convnext_tiny")
        self.assertEqual(params.training.precision, "bf16")
        self.assertEqual(params.training.max_steps, 5200)
        self.assertEqual(params.training.warmup_steps, 200)
        self.assertEqual(params.training.clip_grad_norm, 1.0)
        self.assertFalse(params.training.keep_local_checkpoints)
        self.assertEqual(params.scheduler.eta_min, 1e-8)
        self.assertEqual(params.loss.kind, "vector_charbonnier")
        self.assertEqual(params.loss.charbonnier_epsilon, 0.1)
        self.assertEqual(AngleLossConfig().kind, "vector_charbonnier")

        model = torch.nn.Linear(3, 2)
        optimizer = build_optimizer(model, params.to_optimizer_config())
        self.assertEqual(optimizer.muon, 0.7)
        self.assertEqual(optimizer.sgd, 0.3)
        for group in optimizer.param_groups:
            self.assertEqual(group["lr"], 0.0003)
            self.assertEqual(group["momentum"], 0.95)
            self.assertTrue(group["nesterov"])
        self.assertEqual(
            next(group["weight_decay"] for group in optimizer.param_groups if group["use_muon"]),
            0.025,
        )

        scheduler = _scheduler(optimizer, params)
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0003 / 200)
        for _ in range(200):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 0.0003)
        for _ in range(5000):
            optimizer.step()
            scheduler.step()
        self.assertAlmostEqual(optimizer.param_groups[0]["lr"], 1e-8)

    def test_output_directory_can_be_overridden_without_moving_dataset(self) -> None:
        output_dir = "D:/Data/Angle-Predictor/outputs/experiments"
        config_path = Path("configs/experiments/convnext_tiny_musgd_vector_charbonnier.yaml")
        with patch.dict("os.environ", {"ANGLE_OUTPUT_DIR": output_dir}):
            config = load_config(config_path)
        params = AngleExperimentParams.model_validate(config.params)
        self.assertEqual(params.training.output_dir, Path(output_dir))
        self.assertEqual(params.data.root, Path("data/datasets/synthetic_lines_150k"))

    def test_angular_huber_experiment_config(self) -> None:
        config = load_config(Path("configs/experiments/convnext_tiny_musgd_angular_huber.yaml"))
        params = AngleExperimentParams.model_validate(config.params)
        self.assertEqual(params.loss.kind, "angular_huber")
        self.assertEqual(params.loss.angular_huber_beta_deg, 0.1)
        self.assertEqual(params.training.max_steps, 5200)

    def test_optimizer_choice_can_change(self) -> None:
        config = load_config(
            Path("configs/experiments/convnext_tiny_musgd_vector_charbonnier.yaml")
        )
        for kind, settings in (
            ("adamw", {"lr": 0.0003, "weight_decay": 0.025}),
            (
                "muon_adamw",
                {"muon_lr": 0.02, "adamw_lr": 0.0003, "adamw_weight_decay": 0.025},
            ),
        ):
            with self.subTest(kind=kind):
                config.params["optimizer"] = {"kind": kind, **settings}
                params = AngleExperimentParams.model_validate(config.params)
                optimizer = build_optimizer(torch.nn.Linear(3, 2), params.to_optimizer_config())
                self.assertTrue(optimizer.param_groups)
