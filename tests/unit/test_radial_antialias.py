from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch

from angle_predictor.config.angle_experiment import (
    AngleExperimentParams,
    AngleModelConfig,
    PolarRefinementModelConfig,
)
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.experiments.angle import _build_angle_optimizer, _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.losses.angle import build_angle_loss
from angle_predictor.models.polar_refinement import PolarRefinementModel, RadialAntialias
from tests.unit import test_polar_refinement as fixtures


class RadialAntialiasTests(TestCase):
    def test_filter_preserves_constants_and_angular_columns_with_gradients(self):
        module = RadialAntialias()
        self.assertFalse(list(module.parameters()))
        for dtype in (torch.float32, torch.bfloat16):
            constants = torch.arange(5, dtype=dtype).view(1, 1, 1, 5).expand(2, 3, 7, 5)
            torch.testing.assert_close(module(constants), constants, rtol=0, atol=0)
            impulse = torch.zeros(1, 1, 7, 5, dtype=dtype)
            impulse[..., 3, 2] = 1
            impulse.requires_grad_()
            result = module(impulse)
            expected = torch.zeros_like(impulse)
            expected[..., 2:5, 2] = torch.tensor([0.25, 0.5, 0.25], dtype=dtype)
            torch.testing.assert_close(result, expected, rtol=0, atol=0)
            self.assertEqual(result.dtype, dtype)
            result.sum().backward()
            self.assertTrue(torch.isfinite(impulse.grad).all())
            self.assertTrue((impulse.grad > 0).all())

    def test_filter_removes_alternating_radial_signal_interior(self):
        signal = torch.tensor([1.0, -1.0, 1.0, -1.0, 1.0, -1.0, 1.0]).view(1, 1, 7, 1)
        expected = torch.zeros_like(signal)
        expected[..., 0, :] = 0.5
        expected[..., -1, :] = 0.5
        torch.testing.assert_close(RadialAntialias()(signal), expected, rtol=0, atol=0)

    def test_legacy_default_preserves_state_and_fine_features_bitwise(self):
        params = PolarRefinementModelConfig(
            coarse_model=AngleModelConfig(),
            initialization="imagenet",
            train_coarse=True,
        )
        self.assertFalse(params.fine_radial_antialias)
        torch.manual_seed(42)
        implicit = PolarRefinementModel(fixtures.ToyCoarse()).eval()
        torch.manual_seed(42)
        explicit = PolarRefinementModel(fixtures.ToyCoarse(), fine_radial_antialias=False).eval()
        self.assertEqual(implicit.state_dict().keys(), explicit.state_dict().keys())
        for key, value in implicit.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
        features = torch.randn(2, 5, 192, 33)
        torch.testing.assert_close(implicit.fine(features), explicit.fine(features), rtol=0, atol=0)
        self.assertFalse(any(isinstance(layer, RadialAntialias) for layer in implicit.fine))

    def test_antialias_keeps_odd_and_even_geometry_and_parameter_count(self):
        for height in (16, 17):
            model = PolarRefinementModel(
                fixtures.ToyCoarse(),
                crop_size=(height, 5),
                fine_channels=(4, 8),
                fine_radial_antialias=True,
            )
            control = PolarRefinementModel(
                fixtures.ToyCoarse(),
                crop_size=(height, 5),
                fine_channels=(4, 8),
            )
            self.assertEqual(
                sum(p.numel() for p in model.parameters()),
                sum(p.numel() for p in control.parameters()),
            )
            features = model.fine(torch.randn(2, 5, height, 5))
            self.assertEqual(tuple(features.shape), (2, 8, (height + 3) // 4, 5))

    @patch(
        "angle_predictor.models.line_angle.LineAngleModel",
        side_effect=lambda **_: fixtures.ToyCoarse(),
    )
    def test_actual_factory_frozen_training_and_portable_checkpoint(self, _mock):
        torch.manual_seed(42)
        with TemporaryDirectory() as directory:
            original, source, ancestor = fixtures.PolarRefinementTests().make_source(directory)
            settings = original.model_dump(mode="json")
            settings["model"].update(
                crop_size=[384, 33],
                fine_channels=[16, 32, 64],
                fine_angular_kernel=9,
                radial_pool_bins=1,
                refinement_hidden=128,
                window_deg=4.0,
                fine_radial_antialias=True,
            )
            params = AngleExperimentParams.model_validate(settings)
            model = _build_model(params, (16, 16)).eval()
            self.assertEqual(
                sum(isinstance(layer, RadialAntialias) for layer in model.network.fine), 3
            )
            self.assertEqual(model.network.correction[0].in_features, 4226)
            images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
            torch.testing.assert_close(model(images), source(images), rtol=0, atol=0)
            before = {
                key: value.clone() for key, value in model.network.coarse.state_dict().items()
            }
            optimizer = _build_angle_optimizer(model.network, params)
            optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            self.assertEqual(optimizer_ids, {id(p) for p in model.parameters() if p.requires_grad})
            self.assertFalse(optimizer_ids & {id(p) for p in model.network.coarse.parameters()})
            criterion = build_angle_loss(**params.loss.model_dump())
            targets = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
            model.train()
            self.assertFalse(model.network.coarse.training)
            shapes = []
            hook = model.network.fine.register_forward_hook(
                lambda _m, _args, value: shapes.append(value.shape)
            )
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                prediction, augmented = model.forward_augmented(images, targets)
                loss = criterion(prediction, augmented)
                self.assertTrue(torch.isfinite(prediction).all() and torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(p.grad is None for p in model.network.coarse.parameters()))
                if step == 1:
                    for module in (model.network.fine, model.network.correction):
                        gradients = [p.grad for p in module.parameters()]
                        self.assertTrue(
                            all(g is not None and torch.isfinite(g).all() for g in gradients)
                        )
                        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                optimizer.step()
            hook.remove()
            self.assertTrue(all(tuple(shape) == (2, 64, 48, 33) for shape in shapes))
            for key, value in model.network.coarse.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            model.eval()
            expected = model(images).detach()
            portable = params.model_dump(mode="json")
            portable["data"]["root"] = "Z:/missing-angle-data"
            portable["model"]["coarse_checkpoint"] = "Z:/missing-coarse.pt"
            checkpoint = Path(directory) / "radial-antialias.pt"
            save_checkpoint(
                model,
                checkpoint,
                metadata={
                    "format_version": 1,
                    "input_size": [16, 16],
                    "params": portable,
                },
            )
            ancestor.unlink()
            restored, _, _ = load_angle_predictor(checkpoint, torch.device("cpu"))
            self.assertEqual(
                sum(isinstance(layer, RadialAntialias) for layer in restored.network.fine), 3
            )
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
