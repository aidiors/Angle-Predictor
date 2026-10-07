from copy import deepcopy
from pathlib import Path
from unittest import TestCase

import torch
from torch import nn

from angle_predictor.config.angle_experiment import AngleExperimentParams
from angle_predictor.config.loader import load_config
from angle_predictor.engine.optimizers import build_optimizer
from angle_predictor.experiments.angle import _train_angle_update
from angle_predictor.losses.angle import build_angle_loss


class IndependentAngleModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear = nn.Linear(3, 2)

    def forward_augmented(self, images, targets):
        return nn.functional.normalize(self.linear(images), dim=-1), targets


class GradientAccumulationTests(TestCase):
    def params(self):
        return AngleExperimentParams.model_validate(
            load_config(
                Path("configs/experiments/convnext_tiny_musgd_vector_charbonnier_tuned.yaml")
            ).params
        )

    def test_musgd_update_matches_full_batch_including_partial_microbatches(self):
        torch.manual_seed(42)
        full = IndependentAngleModel()
        accumulated = deepcopy(full)
        params = self.params()
        full_optimizer = build_optimizer(full, params.to_optimizer_config())
        accumulated_optimizer = build_optimizer(accumulated, params.to_optimizer_config())
        criterion = build_angle_loss(**params.loss.model_dump())
        for _ in range(3):
            images = torch.randn(32, 3)
            targets = nn.functional.normalize(torch.randn(32, 2), dim=-1)
            full_result = _train_angle_update(
                full,
                full_optimizer,
                criterion,
                [(images, targets)],
                torch.device("cpu"),
                False,
                1.0,
                1,
            )
            micro_result = _train_angle_update(
                accumulated,
                accumulated_optimizer,
                criterion,
                [(images[:13], targets[:13]), (images[13:], targets[13:])],
                torch.device("cpu"),
                False,
                1.0,
                1,
            )
            for left, right in zip(full.parameters(), accumulated.parameters(), strict=True):
                torch.testing.assert_close(left, right, rtol=1e-5, atol=1e-7)
            self.assertEqual(full_result[2], 32)
            self.assertEqual(micro_result[2], 32)
            self.assertAlmostEqual(full_result[0], micro_result[0], places=5)
            torch.testing.assert_close(full_result[1], micro_result[1], rtol=1e-5, atol=1e-5)

    def test_nonfinite_later_microbatch_does_not_update_weights(self):
        model = IndependentAngleModel()
        before = deepcopy(model.state_dict())
        params = self.params()
        optimizer = build_optimizer(model, params.to_optimizer_config())
        criterion = build_angle_loss(**params.loss.model_dump())
        targets = nn.functional.normalize(torch.randn(4, 2), dim=-1)
        with self.assertRaisesRegex(FloatingPointError, "step 7"):
            _train_angle_update(
                model,
                optimizer,
                criterion,
                [(torch.randn(4, 3), targets), (torch.full((4, 3), float("nan")), targets)],
                torch.device("cpu"),
                False,
                1.0,
                7,
            )
        for key, value in model.state_dict().items():
            torch.testing.assert_close(value, before[key], rtol=0, atol=0)

    def test_accumulation_config_defaults_to_one_and_rejects_zero(self):
        params = self.params()
        self.assertEqual(params.training.gradient_accumulation_steps, 1)
        options = params.model_dump(mode="json")
        options["training"]["gradient_accumulation_steps"] = 0
        with self.assertRaises(ValueError):
            AngleExperimentParams.model_validate(options)
