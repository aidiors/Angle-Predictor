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


class RadialReflectionTests(TestCase):
    def options(self):
        return dict(crop_size=(24, 9), fine_channels=(4, 8), fine_radial_antialias=True)

    def test_default_preserves_weights_rng_and_legacy_predictions(self):
        torch.manual_seed(42)
        default = PolarRefinementModel(fixtures.ToyCoarse(), **self.options()).eval()
        rng = torch.random.get_rng_state()
        for enabled in (False, True):
            torch.manual_seed(42)
            model = PolarRefinementModel(
                fixtures.ToyCoarse(), **self.options(), fine_radial_reflection=enabled
            ).eval()
            self.assertTrue(torch.equal(rng, torch.random.get_rng_state()))
            self.assertEqual(list(default.state_dict()), list(model.state_dict()))
            for key, value in default.state_dict().items():
                self.assertTrue(torch.equal(value, model.state_dict()[key]))
            images = torch.randn(2, 3, 32, 36)
            torch.testing.assert_close(model(images), default(images), rtol=0, atol=0)
            if not enabled:
                with torch.no_grad():
                    default.correction[-1].weight.normal_(std=0.01)
                    model.load_state_dict(default.state_dict(), strict=True)
                torch.testing.assert_close(model(images), default(images), rtol=0, atol=0)
                torch.nn.init.zeros_(default.correction[-1].weight)

    def test_conditional_crop_reversal_is_exactly_invariant(self):
        model = PolarRefinementModel(
            fixtures.ToyCoarse(), **self.options(), fine_radial_reflection=True
        ).eval()
        torch.nn.init.normal_(model.correction[-1].weight, std=0.01)
        crop = torch.randn(2, 3, 24, 9)
        image = torch.randn(2, 3, 32, 36)
        with patch("angle_predictor.models.polar_refinement.crop_signed_polar", return_value=crop):
            normal = model(image)
        with patch(
            "angle_predictor.models.polar_refinement.crop_signed_polar",
            return_value=crop.flip(-2),
        ):
            reflected = model(image)
        torch.testing.assert_close(normal, reflected, rtol=0, atol=0)

    def test_views_reverse_only_rgb_and_keep_coordinate_grid(self):
        model = PolarRefinementModel(
            fixtures.ToyCoarse(), **self.options(), fine_radial_reflection=True
        ).eval()
        captured = []
        hook = model.fine.register_forward_pre_hook(lambda module, args: captured.append(args[0]))
        try:
            model(torch.randn(2, 3, 32, 36))
        finally:
            hook.remove()
        values = captured[0]
        self.assertEqual(values.shape, (4, 5, 24, 9))
        torch.testing.assert_close(values[2:, :3], values[:2, :3].flip(-2), rtol=0, atol=0)
        torch.testing.assert_close(values[2:, 3:], values[:2, 3:], rtol=0, atol=0)

    def test_both_views_receive_gradients_while_coarse_stays_frozen(self):
        model = PolarRefinementModel(
            fixtures.ToyCoarse(), **self.options(), fine_radial_reflection=True
        ).train()
        torch.nn.init.normal_(model.correction[-1].weight, std=0.01)
        captured = []

        def retain(module, args, output):
            output.retain_grad()
            captured.append(output)

        hook = model.fine.register_forward_hook(retain)
        try:
            output = model(torch.randn(2, 3, 32, 36))
            self.assertTrue(torch.isfinite(output).all())
            output[:, 0].sum().backward()
        finally:
            hook.remove()
        grad = captured[0].grad
        self.assertIsNotNone(grad)
        for view in grad.chunk(2, dim=0):
            self.assertTrue(torch.isfinite(view).all())
            self.assertGreater(view.abs().sum().item(), 0)
        self.assertFalse(model.coarse.training)
        self.assertTrue(all(p.grad is None for p in model.coarse.parameters()))
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in model.fine.parameters()
            )
        )

    def test_factory_checkpoint_restores_flag_without_ancestors(self):
        with TemporaryDirectory() as directory:
            params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
            params.model = PolarRefinementModelConfig.model_validate(
                {**params.model.model_dump(), "fine_radial_reflection": True}
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
                self.assertTrue(restored.network.fine_radial_reflection)
                torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)

    def test_unsupported_pooling_and_skip_combinations_are_rejected(self):
        for extra in (
            {"radial_pool_bins": 2},
            {"radial_pool_mode": "attention_max"},
            {"radial_pool_mode": "mean_top4"},
            {"fine_detail_skip": True},
        ):
            with self.assertRaisesRegex(ValueError, "radial reflection"):
                PolarRefinementModel(fixtures.ToyCoarse(), fine_radial_reflection=True, **extra)
            with TemporaryDirectory() as directory:
                params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
                with self.assertRaisesRegex(ValidationError, "radial reflection"):
                    PolarRefinementModelConfig.model_validate(
                        {**params.model.model_dump(), "fine_radial_reflection": True, **extra}
                    )
