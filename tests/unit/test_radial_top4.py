from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch
from pydantic import ValidationError

from angle_predictor.config.angle_experiment import PolarRefinementModelConfig
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.experiments.angle import _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.models.polar_refinement import (
    PolarRefinementModel,
    radial_feature_profile,
    radial_top4_profile,
)
from tests.unit import test_polar_refinement as fixtures


class RadialTop4Tests(TestCase):
    def test_mean_unchanged_and_top_four_average_replaces_single_peak(self):
        features = torch.tensor([1, 2, 3, 4, 5, 100], dtype=torch.float32).view(1, 1, 6, 1)
        original = radial_feature_profile(features)
        actual = radial_top4_profile(features)
        torch.testing.assert_close(actual[:, :1], original[:, :1], rtol=0, atol=0)
        self.assertEqual(actual[0, 1, 0].item(), 28)
        self.assertEqual(original[0, 1, 0].item(), 100)
        reduced = features.clone()
        reduced[0, 0, -1, 0] -= 1
        self.assertEqual((actual - radial_top4_profile(reduced))[0, 1, 0].item(), 0.25)

    def test_fp32_output_and_gradients_to_multiple_support_rows(self):
        values = torch.arange(12, dtype=torch.bfloat16).view(1, 1, 6, 2).requires_grad_()
        profile = radial_top4_profile(values)
        self.assertEqual(profile.dtype, torch.float32)
        profile[:, 1:].sum().backward()
        self.assertTrue(torch.isfinite(values.grad).all())
        self.assertEqual((values.grad != 0).sum().item(), 8)
        torch.testing.assert_close(values.grad[:, :, 2:], torch.full_like(values[:, :, 2:], 0.25))

    def test_legacy_weights_rng_and_zero_correction_are_preserved(self):
        options = dict(crop_size=(24, 9), fine_channels=(4, 8), refinement_hidden=8)
        torch.manual_seed(65)
        baseline = PolarRefinementModel(fixtures.ToyCoarse(), **options).eval()
        rng = torch.random.get_rng_state()
        torch.manual_seed(65)
        candidate = PolarRefinementModel(
            fixtures.ToyCoarse(), **options, radial_pool_mode="mean_top4"
        ).eval()
        self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
        self.assertEqual(set(baseline.state_dict()), set(candidate.state_dict()))
        for name, value in baseline.state_dict().items():
            torch.testing.assert_close(candidate.state_dict()[name], value, rtol=0, atol=0)
        image = torch.randn(2, 3, 32, 36)
        torch.testing.assert_close(candidate(image), baseline(image), rtol=0, atol=0)
        torch.nn.init.normal_(candidate.correction[-1].weight, std=0.001)
        candidate.train()
        output = candidate(image)
        self.assertTrue(torch.isfinite(output).all())
        output[:, 0].sum().backward()
        self.assertTrue(all(p.grad is None for p in candidate.coarse.parameters()))
        self.assertFalse(candidate.coarse.training)
        gradients = [p.grad for p in candidate.fine.parameters()]
        self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
        self.assertTrue(any(g.abs().sum() > 0 for g in gradients))

    def test_factory_and_portable_inference_use_top_four_without_ancestors(self):
        with TemporaryDirectory() as directory:
            params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
            params.model = PolarRefinementModelConfig.model_validate(
                {**params.model.model_dump(), "radial_pool_mode": "mean_top4"}
            )
            with patch(
                "angle_predictor.models.line_angle.LineAngleModel",
                side_effect=lambda **_: fixtures.ToyCoarse(),
            ):
                model = _build_model(params, (16, 16)).eval()
                torch.nn.init.normal_(model.network.correction[-1].weight, std=0.01)
                images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
                expected = model(images)
                metadata = {
                    "format_version": 1,
                    "input_size": [16, 16],
                    "params": params.model_dump(mode="json"),
                }
                metadata["params"]["data"]["root"] = "missing-data"
                metadata["params"]["model"]["coarse_checkpoint"] = "missing-coarse.pt"
                path = Path(directory) / "portable.pt"
                save_checkpoint(model, path, metadata=metadata)
                restored, _, _ = load_angle_predictor(path, torch.device("cpu"))
                self.assertEqual(restored.network.radial_pool_mode, "mean_top4")
                torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)

    def test_invalid_geometry_and_skip_combination_are_rejected(self):
        for options in (
            dict(radial_pool_bins=2),
            dict(crop_size=(8, 9), fine_channels=(4, 8)),
            dict(fine_detail_skip=True),
        ):
            with self.assertRaises(ValidationError):
                PolarRefinementModelConfig(radial_pool_mode="mean_top4", **options)
            with self.assertRaises(ValueError):
                PolarRefinementModel(fixtures.ToyCoarse(), radial_pool_mode="mean_top4", **options)
        with self.assertRaises(ValueError):
            radial_top4_profile(torch.ones(1, 2, 3, 5))
