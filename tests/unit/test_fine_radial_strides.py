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
from angle_predictor.models.polar_refinement import PolarRefinementModel, RadialAntialias
from tests.unit import test_polar_refinement as fixtures


class FineRadialStrideTests(TestCase):
    def test_legacy_metadata_defaults_to_identical_stride2_weights_and_output(self):
        settings = dict(
            coarse_model=AngleModelConfig(), initialization="imagenet", train_coarse=True
        )
        legacy = PolarRefinementModelConfig.model_validate(settings)
        self.assertIsNone(legacy.fine_radial_strides)
        torch.manual_seed(42)
        implicit = PolarRefinementModel(fixtures.ToyCoarse()).eval()
        torch.manual_seed(42)
        explicit = PolarRefinementModel(fixtures.ToyCoarse(), fine_radial_strides=(2, 2, 2)).eval()
        for key, value in implicit.state_dict().items():
            torch.testing.assert_close(value, explicit.state_dict()[key], rtol=0, atol=0)
        images = torch.randn(2, 3, 16, 36)
        torch.testing.assert_close(implicit(images), explicit(images), rtol=0, atol=0)

    def test_invalid_stride_lengths_values_and_pool_geometry_are_rejected(self):
        settings = dict(
            coarse_model=AngleModelConfig(), initialization="imagenet", train_coarse=True
        )
        for strides in ((), (2, 2), (2, 2, 0), (2, 2, 3)):
            with self.subTest(strides=strides):
                with self.assertRaises(ValidationError):
                    PolarRefinementModelConfig(**settings, fine_radial_strides=strides)
                with self.assertRaises(ValueError):
                    PolarRefinementModel(fixtures.ToyCoarse(), fine_radial_strides=strides)
        config = PolarRefinementModelConfig(
            **settings, crop_size=(16, 33), fine_radial_strides=(2, 2, 1), radial_pool_bins=4
        )
        self.assertEqual(config.radial_pool_bins, 4)
        with self.assertRaises(ValidationError):
            PolarRefinementModelConfig(**settings, crop_size=(16, 33), radial_pool_bins=4)
        with self.assertRaises(ValueError):
            PolarRefinementModel(fixtures.ToyCoarse(), crop_size=(16, 33), radial_pool_bins=4)

    @patch(
        "angle_predictor.models.line_angle.LineAngleModel",
        side_effect=lambda **_: fixtures.ToyCoarse(),
    )
    def test_stride221_factory_frozen_gradients_and_portable_inference(self, _mock):
        torch.manual_seed(42)
        with TemporaryDirectory() as directory:
            original, source, ancestor = fixtures.PolarRefinementTests().make_source(directory)
            settings = original.model_dump(mode="json")
            settings["model"].update(
                crop_size=[384, 65],
                fine_channels=[16, 32, 64],
                fine_radial_strides=[2, 2, 1],
                fine_radial_antialias=True,
                fine_angular_kernel=17,
                refinement_hidden=128,
                window_deg=4.0,
            )
            params = AngleExperimentParams.model_validate(settings)
            model = _build_model(params, (16, 16)).eval()
            convolutions = [
                layer for layer in model.network.fine if isinstance(layer, torch.nn.Conv2d)
            ]
            self.assertEqual([layer.stride for layer in convolutions], [(2, 1), (2, 1), (1, 1)])
            for layer in model.network.fine:
                if isinstance(layer, torch.nn.Conv2d):
                    self.assertEqual(layer.kernel_size, (5, 17))
                    self.assertEqual(layer.padding, (2, 8))
            self.assertEqual(model.network.correction[0].in_features, 8322)
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
            self.assertTrue(all(tuple(shape) == (2, 64, 96, 65) for shape in captured))
            for key, value in model.network.coarse.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            model.eval()
            expected = model(images).detach()
            portable = params.model_dump(mode="json")
            portable["data"]["root"] = "Z:/missing-angle-data"
            portable["model"]["coarse_checkpoint"] = "Z:/missing-coarse.pt"
            checkpoint = Path(directory) / "stride221.pt"
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
                [
                    layer.stride
                    for layer in restored.network.fine
                    if isinstance(layer, torch.nn.Conv2d)
                ],
                [(2, 1), (2, 1), (1, 1)],
            )
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)

    def test_last_stride_preserves_weights_parameter_count_and_local_field(self):
        def receptive_field(network):
            field, jump = 1, 1
            for layer in network.fine:
                if isinstance(layer, RadialAntialias):
                    field += 2 * jump
                elif isinstance(layer, torch.nn.Conv2d):
                    field += (layer.kernel_size[0] - 1) * jump
                    jump *= layer.stride[0]
            return field, jump

        for height in (383, 384):
            settings = dict(
                crop_size=(height, 65),
                fine_channels=(16, 32, 64),
                fine_angular_kernel=17,
                fine_radial_antialias=True,
            )
            torch.manual_seed(42)
            original = PolarRefinementModel(fixtures.ToyCoarse(), **settings)
            torch.manual_seed(42)
            dense = PolarRefinementModel(
                fixtures.ToyCoarse(), fine_radial_strides=(2, 2, 1), **settings
            )
            self.assertEqual(receptive_field(original), (43, 8))
            self.assertEqual(receptive_field(dense), (43, 4))
            self.assertEqual(
                sum(p.numel() for p in original.parameters()),
                sum(p.numel() for p in dense.parameters()),
            )
            self.assertEqual(dense.correction[0].in_features, original.correction[0].in_features)
            for key, value in original.state_dict().items():
                torch.testing.assert_close(value, dense.state_dict()[key], rtol=0, atol=0)
            features = torch.randn(2, 5, height, 65)
            self.assertEqual(original.fine(features).shape, (2, 64, 48, 65))
            self.assertEqual(dense.fine(features).shape, (2, 64, 96, 65))
