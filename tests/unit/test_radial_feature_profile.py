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
from angle_predictor.models.polar_refinement import PolarRefinementModel, radial_feature_profile
from tests.unit import test_polar_refinement as fixtures


class RadialProfileTests(TestCase):
    def test_half_profiles_preserve_location_hidden_by_global_pooling(self):
        features = torch.tensor(
            [[[[1.0, 2.0, 3.0], [3.0, 4.0, 5.0], [8.0, 7.0, 6.0], [6.0, 5.0, 4.0]]]],
            requires_grad=True,
        )
        expected = torch.tensor(
            [[[2.0, 3.0, 4.0], [7.0, 6.0, 5.0], [3.0, 4.0, 5.0], [8.0, 7.0, 6.0]]]
        )
        profile = radial_feature_profile(features, 2)
        torch.testing.assert_close(profile, expected, rtol=0, atol=0)
        swapped = features.flip(-2)
        torch.testing.assert_close(
            radial_feature_profile(features), radial_feature_profile(swapped), rtol=0, atol=0
        )
        self.assertFalse(torch.equal(profile, radial_feature_profile(swapped, 2)))
        profile.sum().backward()
        self.assertTrue(torch.isfinite(features.grad).all())
        self.assertTrue((features.grad > 0).all())

    def test_legacy_one_bin_is_bitwise_identical_for_float_and_bfloat16(self):
        settings = dict(
            coarse_model=AngleModelConfig(), initialization="imagenet", train_coarse=True
        )
        self.assertEqual(PolarRefinementModelConfig.model_validate(settings).radial_pool_bins, 1)
        for dtype in (torch.float32, torch.bfloat16):
            features = torch.randn(2, 64, 48, 33).to(dtype)
            old = torch.cat((features.float().mean(-2), features.float().amax(-2)), dim=1)
            torch.testing.assert_close(radial_feature_profile(features), old, rtol=0, atol=0)

    def test_bins_must_fit_feature_radial_rows(self):
        for bins in (0, -1, 49):
            with self.subTest(bins=bins):
                with self.assertRaises(ValidationError):
                    PolarRefinementModelConfig(
                        coarse_model=AngleModelConfig(),
                        initialization="imagenet",
                        train_coarse=True,
                        crop_size=(384, 33),
                        radial_pool_bins=bins,
                    )
                with self.assertRaises(ValueError):
                    PolarRefinementModel(
                        fixtures.ToyCoarse(), crop_size=(384, 33), radial_pool_bins=bins
                    )

    def test_half_profiles_work_with_all_supported_correction_heads(self):
        images = torch.randn(2, 3, 16, 36)
        target = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
        for head in ("mlp", "local_mlp", "heatmap"):
            with self.subTest(head=head):
                model = PolarRefinementModel(
                    fixtures.ToyCoarse(),
                    crop_size=(16, 5),
                    fine_channels=(4, 8),
                    refinement_hidden=8,
                    refinement_head=head,
                    radial_pool_bins=2,
                ).train()
                optimizer = torch.optim.SGD(
                    (p for p in model.parameters() if p.requires_grad), lr=0.1
                )
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    prediction = model(images)
                    self.assertEqual(tuple(prediction.shape), (2, 2))
                    self.assertTrue(torch.isfinite(prediction).all())
                    (prediction - target).square().mean().backward()
                    if step == 1:
                        gradients = [p.grad for p in model.fine.parameters()]
                        self.assertTrue(
                            all(g is not None and torch.isfinite(g).all() for g in gradients)
                        )
                        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                    optimizer.step()

    @patch(
        "angle_predictor.models.line_angle.LineAngleModel",
        side_effect=lambda **_: fixtures.ToyCoarse(),
    )
    def test_radial_halves_preserve_frozen_training_and_portable_inference(self, _mock):
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
                radial_pool_bins=2,
            )
            params = AngleExperimentParams.model_validate(settings)
            model = _build_model(params, (16, 16)).eval()
            for layer in model.network.fine:
                if isinstance(layer, torch.nn.Conv2d):
                    self.assertEqual(layer.kernel_size, (5, 9))
                    self.assertEqual(layer.stride, (2, 1))
                    self.assertEqual(layer.padding, (2, 4))
            self.assertEqual(model.network.correction[0].in_features, 8450)
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
            self.assertEqual(restored.network.radial_pool_bins, 2)
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
