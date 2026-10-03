from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch
from pydantic import ValidationError

from angle_predictor.config.angle_experiment import (
    AngleExperimentParams,
    AngleModelConfig,
    PolarRefinementModelConfig,
)
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.experiments.angle import _build_angle_optimizer, _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.losses.angle import build_angle_loss
from angle_predictor.models.polar_refinement import PolarRefinementModel
from tests.unit import test_polar_refinement as fixtures


class FineAngularContextTests(TestCase):
    def test_legacy_metadata_defaults_to_identical_kernel5_weights_and_output(self):
        settings = dict(
            coarse_model=AngleModelConfig(), initialization="imagenet", train_coarse=True
        )
        legacy = PolarRefinementModelConfig.model_validate(settings)
        self.assertEqual(legacy.fine_angular_kernel, 5)
        torch.manual_seed(42)
        implicit = PolarRefinementModel(fixtures.ToyCoarse()).eval()
        torch.manual_seed(42)
        explicit = PolarRefinementModel(fixtures.ToyCoarse(), fine_angular_kernel=5).eval()
        for key, value in implicit.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
        images = torch.randn(2, 3, 16, 36)
        torch.testing.assert_close(implicit(images), explicit(images), rtol=0, atol=0)

    def test_even_or_nonpositive_kernels_are_rejected(self):
        for kernel in (0, -1, 2, 8):
            with self.subTest(kernel=kernel):
                with self.assertRaises(ValidationError):
                    PolarRefinementModelConfig(
                        coarse_model=AngleModelConfig(),
                        initialization="imagenet",
                        train_coarse=True,
                        fine_angular_kernel=kernel,
                    )
                with self.assertRaises(ValueError):
                    PolarRefinementModel(fixtures.ToyCoarse(), fine_angular_kernel=kernel)

    @patch(
        "angle_predictor.models.line_angle.LineAngleModel",
        side_effect=lambda **_: fixtures.ToyCoarse(),
    )
    def test_kernel9_preserves_profile_frozen_training_and_portable_inference(self, _mock):
        torch.manual_seed(42)
        with TemporaryDirectory() as directory:
            original, source, ancestor = fixtures.PolarRefinementTests().make_source(directory)
            settings = original.model_dump(mode="json")
            settings["model"].update(
                crop_size=[384, 33],
                fine_channels=[16, 32, 64],
                fine_angular_kernel=9,
                refinement_hidden=128,
                window_deg=4.0,
            )
            params = AngleExperimentParams.model_validate(settings)
            model = _build_model(params, (16, 16)).eval()
            for layer in model.network.fine:
                if isinstance(layer, torch.nn.Conv2d):
                    self.assertEqual(layer.kernel_size, (5, 9))
                    self.assertEqual(layer.stride, (2, 1))
                    self.assertEqual(layer.padding, (2, 4))
            self.assertEqual(model.network.correction[0].in_features, 4226)
            images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
            torch.testing.assert_close(model(images), source(images), rtol=0, atol=0)
            before = {k: v.clone() for k, v in model.network.coarse.state_dict().items()}
            optimizer = _build_angle_optimizer(model.network, params)
            optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            self.assertEqual(optimizer_ids, {id(p) for p in model.parameters() if p.requires_grad})
            self.assertFalse(optimizer_ids & {id(p) for p in model.network.coarse.parameters()})
            criterion = build_angle_loss(**params.loss.model_dump())
            targets = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
            model.train()
            self.assertFalse(model.network.coarse.training)
            captured = []
            hook = model.network.fine.register_forward_hook(
                lambda _m, _args, value: captured.append(value.shape)
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
            self.assertTrue(all(tuple(shape) == (2, 64, 48, 33) for shape in captured))
            for key, value in model.network.coarse.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            model.eval()
            expected = model(images).detach()
            portable = params.model_dump(mode="json")
            portable["data"]["root"] = "Z:/missing-angle-data"
            portable["model"]["coarse_checkpoint"] = "Z:/missing-coarse.pt"
            checkpoint = Path(directory) / "angular9.pt"
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
            self.assertEqual(restored.network.fine[0].kernel_size, (5, 9))
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
