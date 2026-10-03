import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from pydantic import ValidationError

from angle_predictor.config.angle_experiment import AngleExperimentParams, PolarLineModelConfig
from angle_predictor.config.loader import load_config
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.engine.optimizers import build_optimizer
from angle_predictor.experiments.angle import _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.losses.angle import vector_charbonnier_angle_loss
from angle_predictor.models.polar_line import PolarLineModel, signed_polar_to_circle


class PolarLineModelTests(TestCase):
    def setUp(self) -> None:
        torch.manual_seed(7)
        self.model = PolarLineModel(channels=(4, 8), blocks_per_stage=1, head_hidden=8).eval()
        # Verify equivariance with learned, nonzero local offsets as well.
        torch.nn.init.normal_(self.model.head.offset.weight, std=0.1)

    def test_reindex_preserves_samples_and_matches_signed_seam(self) -> None:
        image = torch.arange(24).reshape(1, 1, 4, 6)
        circle = signed_polar_to_circle(image)
        self.assertEqual(circle.shape, (1, 1, 2, 12))
        self.assertTrue(torch.equal(circle[..., :6], image[..., 2:, :]))
        self.assertTrue(torch.equal(circle[..., 6:], image[..., :2, :].flip(-2)))
        self.assertTrue(torch.equal(circle.sort().values.flatten().sort().values, image.flatten()))
        with self.assertRaises(ValueError):
            signed_polar_to_circle(torch.zeros(1, 3, 5, 6))

    def test_signed_angular_shift_rotates_output_across_seam(self) -> None:
        image = torch.randn(2, 3, 16, 36)
        original = self.model(image)
        for shift in (1, 7, 35):
            shifted_image = torch.cat((image[..., -shift:].flip(-2), image[..., :-shift]), dim=-1)
            shifted = self.model(shifted_image)
            angle = 2 * math.pi * shift / image.shape[-1]
            expected = torch.stack(
                (
                    original[:, 0] * math.cos(angle) + original[:, 1] * math.sin(angle),
                    original[:, 1] * math.cos(angle) - original[:, 0] * math.sin(angle),
                ),
                dim=-1,
            )
            torch.testing.assert_close(shifted, expected, atol=2e-4, rtol=2e-4)

    def test_half_turn_and_full_angular_resolution(self) -> None:
        image = torch.randn(2, 3, 16, 36)
        torch.testing.assert_close(self.model(image), self.model(image.flip(-2)))
        features = self.model.backbone(signed_polar_to_circle(image))
        self.assertEqual(features.shape[-2:], (1, 72))

    def test_gradients_reach_backbone_scores_and_offsets(self) -> None:
        prediction = self.model(torch.randn(2, 3, 16, 36))
        loss = vector_charbonnier_angle_loss(prediction, torch.tensor([[0.6, 0.8], [-0.8, 0.6]]))
        loss.backward()
        for name, parameter in self.model.named_parameters():
            with self.subTest(parameter=name):
                self.assertIsNotNone(parameter.grad)
                self.assertTrue(torch.isfinite(parameter.grad).all())
        self.assertGreater(self.model.head.offset.weight.grad.abs().sum().item(), 0)

    def test_config_optimizer_and_checkpoint_roundtrip(self) -> None:
        config = load_config(Path("configs/experiments/polar_line_musgd_vector_charbonnier.yaml"))
        params = AngleExperimentParams.model_validate(config.params)
        self.assertIsInstance(params.model, PolarLineModelConfig)
        params.model = PolarLineModelConfig(channels=(4, 8), blocks_per_stage=1, head_hidden=8)
        params.preprocessing.output_size = (16, 36)
        model = _build_model(params, (16, 16)).eval()
        images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
        expected = model(images)
        optimizer = build_optimizer(model.network, params.to_optimizer_config())
        optimizer.zero_grad()
        vector_charbonnier_angle_loss(expected, torch.tensor([[0.6, 0.8], [-0.8, 0.6]])).backward()
        optimizer.step()
        expected = model(images).detach()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(
                model,
                path,
                metadata={
                    "format_version": 1,
                    "input_size": [16, 16],
                    "params": params.model_dump(mode="json"),
                },
            )
            restored, size, _ = load_angle_predictor(path, torch.device("cpu"))
            self.assertEqual(size, (16, 16))
            torch.testing.assert_close(restored(images), expected)

    def test_config_rejects_odd_radius_and_invalid_channels(self) -> None:
        with self.assertRaises(ValidationError):
            PolarLineModelConfig(channels=())
        with self.assertRaises(ValidationError):
            AngleExperimentParams.model_validate(
                {
                    "data": {"root": "data"},
                    "model": {"kind": "polar_line"},
                    "preprocessing": {"output_size": [15, 36]},
                }
            )
