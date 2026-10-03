from __future__ import annotations

import math
import tempfile
import unittest
from pathlib import Path

import torch
from torch import nn

from angle_predictor.config.angle_experiment import AngleExperimentParams
from angle_predictor.data.angle_transforms import AngleBatchPreprocessor
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.experiments.angle import _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.models.cartesian import (
    CartesianConvNeXt,
    CartesianRefinement,
    crop_cartesian_strip,
)


class ConstantCoarse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vector = nn.Parameter(torch.tensor([0.0, 1.0]))

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return self.vector.expand(len(image), -1)


class CartesianModelsTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(2)

    def test_preprocessor_preserves_pixels_and_matching_augmentation(self) -> None:
        options = {"input_size": (32, 32), "output_size": (32, 32)}
        cartesian = AngleBatchPreprocessor(**options, projection="cartesian")
        polar = AngleBatchPreprocessor(**options)
        images = torch.randint(0, 256, (2, 3, 32, 32), dtype=torch.uint8)
        targets = torch.tensor([[0.6, 0.8], [0.0, 1.0]])
        expected = (images.float() / 255 - 0.5) / 0.5
        actual, _ = cartesian(images)
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        torch.manual_seed(42)
        c_images, c_labels = cartesian(images, targets, augment=True)
        torch.manual_seed(42)
        p_images, p_labels = polar(images, targets, augment=True)
        torch.testing.assert_close(c_labels, p_labels, rtol=0, atol=0)
        torch.testing.assert_close(polar.polar((c_images + 1) / 2) * 2 - 1, p_images)
        self.assertIsInstance(cartesian.polar, nn.Identity)

    def test_strip_is_affine_cartesian_and_rotates_axes(self) -> None:
        axis = torch.linspace(-1, 1, 33)
        y, x = torch.meshgrid(axis, axis, indexing="ij")
        image = torch.stack((x, y, x + y)).unsqueeze(0)
        for angle in (0.0, math.pi / 2, 0.4):
            crop = crop_cartesian_strip(image, torch.tensor([angle]), (17, 9), 8)
            along = torch.linspace(-1, 1, 17)[:, None]
            across = torch.linspace(-0.25, 0.25, 9)[None, :]
            expected_x = along * math.cos(angle) - across * math.sin(angle)
            expected_y = along * math.sin(angle) + across * math.cos(angle)
            torch.testing.assert_close(crop[0, 0], expected_x.clamp(-1, 1), atol=2e-6, rtol=0)
            torch.testing.assert_close(crop[0, 1], expected_y.clamp(-1, 1), atol=2e-6, rtol=0)

    def test_frozen_coarse_and_fine_updates(self) -> None:
        coarse = ConstantCoarse()
        model = CartesianRefinement(coarse, crop_size=(32, 17), fine_channels=(8, 8))
        image = torch.randn(2, 3, 32, 32)
        model.train()
        self.assertFalse(coarse.training)
        torch.testing.assert_close(model(image), coarse(image), rtol=0, atol=0)
        original = coarse.vector.detach().clone()
        optimizer = torch.optim.SGD(model.parameters(), lr=0.1)
        for _ in range(2):
            optimizer.zero_grad()
            loss = (model(image) - torch.tensor([0.2, 0.98])).square().mean()
            loss.backward()
            optimizer.step()
        self.assertIsNone(coarse.vector.grad)
        torch.testing.assert_close(coarse.vector, original, rtol=0, atol=0)
        self.assertTrue(
            any(p.grad is not None and p.grad.abs().sum() > 0 for p in model.fine.parameters())
        )
        self.assertTrue(torch.isfinite(model(image)).all())

    def test_stock_head_and_portable_full_checkpoint(self) -> None:
        params = AngleExperimentParams.model_validate(
            {
                "data": {"root": "missing-data"},
                "model": {"kind": "cartesian_convnext", "pretrained": False},
                "preprocessing": {"projection": "cartesian", "output_size": [256, 256]},
                "training": {"precision": "fp32"},
            }
        )
        model = _build_model(params, (256, 256)).eval()
        self.assertIsInstance(model.network, CartesianConvNeXt)
        self.assertEqual(model.network.backbone.head.fc.out_features, 2)
        image = torch.randint(0, 256, (1, 3, 256, 256), dtype=torch.uint8)
        with torch.inference_mode():
            expected = model(image)
            torch.testing.assert_close(
                expected,
                nn.functional.normalize(
                    model.network.backbone(model.preprocessor(image)[0]), dim=1
                ),
            )
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "model.pt"
            save_checkpoint(
                model,
                path,
                step=1,
                metadata={
                    "format_version": 1,
                    "input_size": [256, 256],
                    "params": params.model_dump(mode="json"),
                },
            )
            restored, _, _ = load_angle_predictor(path, torch.device("cpu"))
            with torch.inference_mode():
                torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)

    def test_config_rejects_mismatched_projection(self) -> None:
        with self.assertRaises(ValueError):
            AngleExperimentParams.model_validate(
                {"data": {"root": "."}, "model": {"kind": "cartesian_convnext"}}
            )


if __name__ == "__main__":
    unittest.main()
