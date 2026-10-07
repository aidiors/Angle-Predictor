from copy import deepcopy
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
from angle_predictor.models.polar_refinement import PolarRefinementModel
from tests.unit import test_polar_refinement as fixtures

ToyCoarse = fixtures.ToyCoarse


class FineDetailSkipTests(TestCase):
    def test_zero_projection_preserves_weights_rng_and_nonzero_correction(self):
        for antialias in (False, True):
            options = {
                "crop_size": (24, 9),
                "fine_channels": (4, 8),
                "refinement_hidden": 8,
                "fine_radial_antialias": antialias,
            }
            torch.manual_seed(57)
            baseline = PolarRefinementModel(ToyCoarse(), **options).eval()
            rng = torch.random.get_rng_state()
            torch.manual_seed(57)
            skip = PolarRefinementModel(ToyCoarse(), **options, fine_detail_skip=True).eval()
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            for name, weight in baseline.state_dict().items():
                self.assertTrue(torch.equal(weight, skip.state_dict()[name]))
            self.assertEqual(
                set(skip.state_dict()) - set(baseline.state_dict()),
                {"detail_projection.weight", "detail_projection.bias"},
            )
            with torch.no_grad():
                baseline.correction[-1].weight.normal_()
                skip.correction[-1].load_state_dict(baseline.correction[-1].state_dict())
                image = torch.randn(3, 3, 32, 36)
                torch.testing.assert_close(baseline(image), skip(image), rtol=0, atol=0)

    def test_projection_and_early_layer_learn_with_frozen_coarse(self):
        model = PolarRefinementModel(
            ToyCoarse(), crop_size=(24, 9), fine_channels=(4, 8), fine_detail_skip=True
        ).train()
        source = deepcopy(model.coarse.state_dict())
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
        image = torch.randn(3, 3, 32, 36)
        target = torch.nn.functional.normalize(torch.randn(3, 2), dim=-1)
        for _ in range(4):
            optimizer.zero_grad()
            output = model(image)
            (output - target).square().mean().backward()
            optimizer.step()
        self.assertTrue(torch.isfinite(output).all())
        projection = model.detail_projection
        self.assertIsNotNone(projection)
        for param in projection.parameters():
            self.assertIsNotNone(param.grad)
            self.assertTrue(torch.isfinite(param.grad).all())
            self.assertGreater(float(param.grad.abs().sum()), 0)
        self.assertGreater(float(projection.weight.detach().abs().sum()), 0)
        self.assertGreater(float(model.fine[0].weight.grad.abs().sum()), 0)
        self.assertFalse(model.coarse.training)
        self.assertTrue(all(p.grad is None for p in model.coarse.parameters()))
        for name, value in model.coarse.state_dict().items():
            self.assertTrue(torch.equal(value, source[name]))

    def test_factory_portability_and_strict_projection_loading(self):
        with TemporaryDirectory() as directory:
            params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
            params.model = PolarRefinementModelConfig.model_validate(
                {**params.model.model_dump(), "fine_detail_skip": True}
            )
            with patch(
                "angle_predictor.models.line_angle.LineAngleModel",
                side_effect=lambda **_: ToyCoarse(),
            ):
                model = _build_model(params, (16, 16)).eval()
                images = torch.randint(0, 256, (3, 3, 16, 16), dtype=torch.uint8)
                expected = model(images)
                metadata = {
                    "format_version": 1,
                    "run_id": "2" * 32,
                    "input_size": [16, 16],
                    "params": params.model_dump(mode="json"),
                }
                metadata["params"]["data"]["root"] = "missing-data"
                metadata["params"]["model"]["coarse_checkpoint"] = "missing-coarse.pt"
                path = Path(directory) / "portable.pt"
                save_checkpoint(model, path, metadata=metadata)
                restored, _, _ = load_angle_predictor(path, torch.device("cpu"))
                torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
                weights = dict(restored.state_dict())
                weights.pop("network.detail_projection.weight")
                with self.assertRaisesRegex(RuntimeError, "Missing key"):
                    restored.load_state_dict(weights, strict=True)

    def test_unsupported_pooling_and_single_layer_are_rejected(self):
        for extra in (
            {"radial_pool_bins": 2},
            {"radial_pool_mode": "attention_max"},
            {"fine_channels": (8,)},
        ):
            with TemporaryDirectory() as directory:
                params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
                with self.assertRaises(ValidationError):
                    PolarRefinementModelConfig.model_validate(
                        {**params.model.model_dump(), "fine_detail_skip": True, **extra}
                    )
        with self.assertRaises(ValueError):
            PolarRefinementModel(ToyCoarse(), fine_channels=(8,), fine_detail_skip=True)
