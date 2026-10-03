from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import numpy as np
import torch
from torch import nn

from angle_predictor.tracking.mlflow import (
    _flatten_params,
    _write_model_code_bundle,
    log_params,
    log_pytorch_model,
)


class TrackingTests(TestCase):
    def test_flattened_parameters_are_searchable_strings(self) -> None:
        flattened = _flatten_params(
            {
                "optimizer": {"kind": "adamw", "lr": 0.0003},
                "loss": {"kind": "mse"},
                "enabled": True,
            }
        )
        self.assertEqual(flattened["optimizer.kind"], "adamw")
        self.assertEqual(flattened["optimizer.lr"], "0.0003")
        self.assertEqual(flattened["loss.kind"], "mse")
        self.assertEqual(flattened["enabled"], "true")

    def test_model_logging_uses_numpy_example_and_restores_training_mode(self) -> None:
        model = nn.Sequential(nn.BatchNorm1d(3), nn.Linear(3, 2))
        model.train()
        model[0].eval()
        before_mean = model[0].running_mean.detach().clone()
        example = np.ones((2, 3), dtype=np.float32)
        modes: list[bool] = []

        def capture(logged_model: nn.Module, **kwargs: object) -> None:
            modes.append(logged_model.training)
            self.assertIs(kwargs["input_example"], example)

        with patch("angle_predictor.tracking.mlflow.mlflow.pytorch.log_model", side_effect=capture):
            log_pytorch_model(model, example)

        self.assertEqual(modes, [False])
        self.assertTrue(model.training)
        self.assertFalse(model[0].training)
        torch.testing.assert_close(model[0].running_mean, before_mean)

    def test_log_params_flattens_resolved_choices(self) -> None:
        with patch("angle_predictor.tracking.mlflow.mlflow.log_params") as logged:
            log_params({"resolved": {"loss": "mse", "optimizer": {"kind": "adamw"}}})
        logged.assert_called_once_with({"resolved.loss": "mse", "resolved.optimizer.kind": "adamw"})

    def test_model_logging_registers_version_tags_and_candidate_alias(self) -> None:
        model = nn.Linear(3, 2)
        example = np.ones((2, 3), dtype=np.float32)
        with (
            patch(
                "angle_predictor.tracking.mlflow.mlflow.pytorch.log_model",
                return_value=SimpleNamespace(registered_model_version=4),
            ) as logged_model,
            patch("angle_predictor.tracking.mlflow.MlflowClient") as client_factory,
            patch("angle_predictor.tracking.mlflow.mlflow.set_tags") as run_tags,
        ):
            version = log_pytorch_model(
                model,
                example,
                registered_model_name="angle-predictor",
                model_version_tags={"validation.angle_mae_deg": "0.25"},
            )

        self.assertEqual(version, "4")
        self.assertEqual(logged_model.call_args.kwargs["serialization_format"], "pickle")
        self.assertTrue(logged_model.call_args.kwargs["code_paths"])
        client = client_factory.return_value
        client.set_model_version_tag.assert_called_once_with(
            "angle-predictor", "4", "validation.angle_mae_deg", "0.25"
        )
        client.set_registered_model_alias.assert_called_once_with(
            "angle-predictor", "candidate", "4"
        )
        run_tags.assert_called_once_with(
            {
                "model.registry_name": "angle-predictor",
                "model.version": "4",
                "model.alias": "candidate",
            }
        )

    def test_model_code_bundle_contains_runtime_modules_only(self) -> None:
        with TemporaryDirectory() as temp_dir:
            package = _write_model_code_bundle(Path(temp_dir))

            self.assertTrue((package / "models" / "line_angle.py").is_file())
            self.assertTrue((package / "data" / "angle_transforms.py").is_file())
            self.assertTrue((package / "models" / "polar_line.py").is_file())
            self.assertTrue((package / "config" / "angle_experiment.py").is_file())
            self.assertTrue((package / "engine" / "optimizers" / "factory.py").is_file())
            self.assertFalse((package / "data" / "generation").exists())
