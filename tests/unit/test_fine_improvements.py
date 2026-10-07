import hashlib
import math
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch
from pydantic import ValidationError

from angle_predictor.config.angle_experiment import PolarRefinementModelConfig
from angle_predictor.data.angle_transforms import AngleBatchPreprocessor
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.experiments.angle import _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.models.line_angle import AnglePredictor
from angle_predictor.models.polar_refinement import (
    PolarRefinementModel,
    crop_cartesian_polar,
    radial_attention_profile,
    radial_feature_profile,
)
from tests.unit import test_polar_refinement as fixtures

ToyCoarse = fixtures.ToyCoarse


class FineImprovementsTests(TestCase):
    def test_direct_crop_matches_cartesian_coordinates_and_signed_seam(self):
        for height, width in ((16, 16), (18, 26)):
            y, x = torch.meshgrid(torch.arange(height), torch.arange(width), indexing="ij")
            image = torch.stack((x, y, x + 2 * y)).float()[None].expand(3, -1, -1, -1)
            angles = torch.tensor([0.0, math.pi / 2, math.pi])
            actual = crop_cartesian_polar(image, angles, crop_size=(12, 9), window_deg=4)
            theta = angles[:, None, None] + torch.linspace(-math.radians(4), math.radians(4), 9)
            radius = torch.linspace(-(height - 1) / 2, (height - 1) / 2, 12)[None, :, None]
            expected_x = (width - 1) / 2 + radius * theta.cos()
            expected_y = (height - 1) / 2 + radius * theta.sin()
            expected = torch.stack((expected_x, expected_y, expected_x + 2 * expected_y), dim=1)
            torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)
            torch.testing.assert_close(actual[2], actual[0].flip(-2), atol=1e-5, rtol=1e-5)

    def test_direct_crop_has_finite_nonzero_image_and_angle_gradients(self):
        image = torch.randn(2, 3, 20, 20, requires_grad=True)
        angle = torch.tensor([0.13, 1.6], requires_grad=True)
        crop_cartesian_polar(
            image, angle, crop_size=(16, 9), window_deg=4
        ).square().sum().backward()
        for value in (image, angle):
            self.assertIsNotNone(value.grad)
            self.assertTrue(torch.isfinite(value.grad).all())
            self.assertGreater(float(value.grad.abs().sum()), 0)

    def test_two_views_use_one_augmentation_and_preserve_coarse_and_labels_bitwise(self):
        preprocessor = AngleBatchPreprocessor(input_size=(16, 16), output_size=(24, 36))
        images = torch.randint(0, 256, (4, 3, 16, 16), dtype=torch.uint8)
        labels = torch.nn.functional.normalize(torch.randn(4, 2), dim=-1)
        torch.manual_seed(73)
        old_polar, old_labels = preprocessor(images, labels, augment=True)
        old_rng = torch.random.get_rng_state()
        torch.manual_seed(73)
        polar, cartesian, adjusted = preprocessor.forward_with_cartesian(
            images, labels, augment=True
        )
        self.assertTrue(torch.equal(old_polar, polar))
        self.assertTrue(torch.equal(old_labels, adjusted))
        self.assertTrue(torch.equal(old_rng, torch.random.get_rng_state()))
        self.assertEqual(cartesian.shape, images.shape)
        torch.manual_seed(73)
        rgb, prepared_labels = preprocessor._prepare(images, labels, augment=True)
        self.assertTrue(torch.equal(adjusted, prepared_labels))
        self.assertTrue(torch.equal(cartesian, (rgb - preprocessor.mean) / preprocessor.std))

    def test_attention_initializes_uniform_and_can_select_radial_support(self):
        features = torch.tensor([1.0, 2.0, 5.0]).view(1, 1, 3, 1).expand(1, 1, 3, 4)
        uniform = radial_attention_profile(features, torch.zeros(1))
        torch.testing.assert_close(uniform, radial_feature_profile(features))
        attended = radial_attention_profile(features, torch.tensor([20.0]))
        torch.testing.assert_close(attended[:, 0], features.amax(-2)[:, 0])
        self.assertTrue(torch.equal(attended[:, 1], features.amax(-2)[:, 0]))
        query = torch.zeros(1, requires_grad=True)
        radial_attention_profile(features, query).sum().backward()
        self.assertTrue(torch.isfinite(query.grad).all())
        self.assertGreater(float(query.grad.abs().sum()), 0)

    def test_default_weights_and_initial_fine_weights_stay_compatible(self):
        torch.manual_seed(19)
        legacy = PolarRefinementModel(ToyCoarse(), fine_channels=(4, 8))
        torch.manual_seed(19)
        attention = PolarRefinementModel(
            ToyCoarse(), fine_channels=(4, 8), radial_pool_mode="attention_max"
        )
        for key, value in legacy.state_dict().items():
            self.assertTrue(torch.equal(value, attention.state_dict()[key]))
        self.assertEqual(set(attention.state_dict()) - set(legacy.state_dict()), {"radius_query"})
        self.assertEqual(legacy.fine_crop_source, "polar")
        self.assertNotIn("radius_query", legacy.state_dict())

    def test_direct_crop_requires_original_rgb(self):
        model = PolarRefinementModel(ToyCoarse(), fine_crop_source="cartesian").eval()
        with self.assertRaisesRegex(ValueError, "Cartesian image"):
            model(torch.rand(2, 3, 16, 36))

    def test_factory_frozen_training_and_portable_inference_for_all_three_variants(self):
        for direct, attention in ((True, False), (False, True), (True, True)):
            with (
                self.subTest(direct=direct, attention=attention),
                TemporaryDirectory() as directory,
            ):
                params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
                params.model = PolarRefinementModelConfig.model_validate(
                    {
                        **params.model.model_dump(),
                        "fine_crop_source": "cartesian" if direct else "polar",
                        "radial_pool_mode": "attention_max" if attention else "mean_max",
                    }
                )
                with patch(
                    "angle_predictor.models.line_angle.LineAngleModel",
                    side_effect=lambda **_: ToyCoarse(),
                ):
                    model = _build_model(params, (16, 16)).train()
                    before = deepcopy(model.network.coarse.state_dict())
                    images = torch.randint(0, 256, (4, 3, 16, 16), dtype=torch.uint8)
                    targets = torch.nn.functional.normalize(torch.randn(4, 2), dim=-1)
                    optimizer = torch.optim.AdamW(
                        (p for p in model.parameters() if p.requires_grad), lr=0.01
                    )
                    for _ in range(3):
                        optimizer.zero_grad()
                        output, labels = model.forward_augmented(images, targets)
                        (output - labels).square().mean().backward()
                        optimizer.step()
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertFalse(model.network.coarse.training)
                    self.assertTrue(all(p.grad is None for p in model.network.coarse.parameters()))
                    for key, value in model.network.coarse.state_dict().items():
                        self.assertTrue(torch.equal(before[key], value))
                    if attention:
                        query = model.network.radius_query
                        self.assertTrue(torch.isfinite(query.grad).all())
                        self.assertGreater(float(query.grad.abs().sum()), 0)
                    model.eval()
                    expected = model(images)
                    metadata = {
                        "format_version": 1,
                        "run_id": "2" * 32,
                        "input_size": [16, 16],
                        "params": params.model_dump(mode="json"),
                        "dataset_meta_sha256": hashlib.sha256(
                            (Path(directory) / "meta.json").read_bytes()
                        ).hexdigest(),
                    }
                    metadata["params"]["data"]["root"] = "missing-data"
                    metadata["params"]["model"]["coarse_checkpoint"] = "missing-coarse.pt"
                    path = Path(directory) / "portable.pt"
                    save_checkpoint(model, path, metadata=metadata)
                    restored, _, _ = load_angle_predictor(path, torch.device("cpu"))
                    self.assertTrue(torch.equal(restored(images), expected))
                    self.assertIsInstance(restored, AnglePredictor)

    def test_attention_rejects_multiple_radial_bins(self):
        with TemporaryDirectory() as directory:
            params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
            with self.assertRaises(ValidationError):
                PolarRefinementModelConfig.model_validate(
                    {
                        **params.model.model_dump(),
                        "radial_pool_mode": "attention_max",
                        "radial_pool_bins": 2,
                    }
                )
