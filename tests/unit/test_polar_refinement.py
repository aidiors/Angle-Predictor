import hashlib
import json
import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch
from pydantic import ValidationError
from torch import nn
from torch.nn import functional as F

from angle_predictor.config.angle_experiment import (
    AngleExperimentParams,
    AngleModelConfig,
    PolarRefinementModelConfig,
)
from angle_predictor.data.angle_transforms import AngleBatchPreprocessor
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.engine.optimizers import AdamWConfig, build_optimizer
from angle_predictor.experiments.angle import _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.models.line_angle import AnglePredictor
from angle_predictor.models.polar_refinement import (
    PolarRefinementModel,
    crop_signed_polar,
    rotate_double_angle,
)


class ToyCoarse(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.norm = nn.BatchNorm1d(3)
        self.projection = nn.Linear(3, 2)
        self.dropout = nn.Dropout(0.5)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.projection(self.dropout(self.norm(image.mean((-2, -1))))), dim=-1)


class PolarRefinementTests(TestCase):
    def make_source(self, directory: str):
        root = Path(directory)
        meta = root / "meta.json"
        meta.write_text(json.dumps({"images_shape": [4, 16, 16, 3]}), encoding="utf-8")
        source_params = AngleExperimentParams.model_validate(
            {
                "data": {"root": root},
                "model": AngleModelConfig(pretrained=False).model_dump(),
                "preprocessing": {"output_size": [16, 36]},
                "training": {"precision": "fp32"},
            }
        )
        source = AnglePredictor(
            AngleBatchPreprocessor(input_size=(16, 16), output_size=(16, 36)), ToyCoarse()
        ).eval()
        checkpoint = root / "coarse.pt"
        save_checkpoint(
            source,
            checkpoint,
            metadata={
                "format_version": 1,
                "run_id": "1" * 32,
                "dataset_meta_sha256": hashlib.sha256(meta.read_bytes()).hexdigest(),
                "input_size": (16, 16),
                "params": source_params.model_dump(mode="json"),
            },
        )
        config = PolarRefinementModelConfig(
            coarse_model=source_params.model,
            coarse_checkpoint=checkpoint,
            coarse_run_id="1" * 32,
            coarse_split_seed=42,
            crop_size=(16, 9),
            fine_channels=(4, 8),
            refinement_hidden=8,
        )
        return source_params.model_copy(update={"model": config}), source, checkpoint

    def test_crop_zero_crossing_reflects_radius(self) -> None:
        image = torch.arange(48).float().reshape(1, 1, 4, 12)
        actual = crop_signed_polar(image, torch.tensor([0.0]), crop_size=(4, 5), window_deg=30)
        expected = torch.cat((image[..., -2:].flip(-2), image[..., :3]), dim=-1)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)

    def test_crop_pi_crossing_reflects_radius(self) -> None:
        image = torch.arange(48).float().reshape(1, 1, 4, 12)
        actual = crop_signed_polar(
            image, torch.tensor([math.radians(165)]), crop_size=(4, 5), window_deg=30
        )
        expected = torch.cat((image[..., 9:], image[..., :2].flip(-2)), dim=-1)
        torch.testing.assert_close(actual, expected, atol=1e-4, rtol=1e-5)

    def test_continuous_crop_resolves_subcolumn_position(self) -> None:
        image = torch.arange(12).float().view(1, 1, 1, 12).expand(1, 1, 4, 12)
        actual = crop_signed_polar(
            image, torch.tensor([math.radians(67.5)]), crop_size=(4, 3), window_deg=15
        )
        expected = torch.tensor([3.5, 4.5, 5.5]).view(1, 1, 1, 3).expand_as(actual)
        torch.testing.assert_close(actual, expected, atol=1e-5, rtol=1e-5)

    def test_rotation_sign_and_undirected_wrap(self) -> None:
        angle = math.radians(179)
        vector = torch.tensor([[math.sin(2 * angle), math.cos(2 * angle)]])
        actual = rotate_double_angle(vector, torch.tensor([math.radians(2)]))
        expected = torch.tensor([[math.sin(math.radians(2)), math.cos(math.radians(2))]])
        torch.testing.assert_close(actual, expected)
        torch.testing.assert_close(actual.norm(dim=-1), torch.ones(1))

    def test_initial_prediction_equals_coarse_and_training_keeps_it_frozen(self) -> None:
        torch.manual_seed(42)
        coarse = ToyCoarse()
        model = PolarRefinementModel(
            coarse, crop_size=(16, 9), fine_channels=(4, 8), refinement_hidden=8
        ).train()
        self.assertFalse(coarse.training)
        self.assertTrue(model.fine.training)
        self.assertTrue(all(not p.requires_grad for p in coarse.parameters()))
        image = torch.randn(2, 3, 32, 36)
        expected = coarse(image).detach()
        torch.testing.assert_close(model(image), expected)
        before = {name: value.clone() for name, value in coarse.state_dict().items()}
        optimizer = build_optimizer(model, AdamWConfig(lr=0.01))
        coarse_ids = {id(p) for p in coarse.parameters()}
        self.assertTrue(
            all(id(p) not in coarse_ids for g in optimizer.param_groups for p in g["params"])
        )
        model(image)[:, 0].sum().backward()
        self.assertGreater(model.correction[-1].weight.grad.abs().sum().item(), 0)
        self.assertTrue(all(p.grad is None for p in coarse.parameters()))
        optimizer.step()
        for name, value in coarse.state_dict().items():
            torch.testing.assert_close(value, before[name], rtol=0, atol=0)
        self.assertFalse(torch.equal(model(image), expected))

    @patch("angle_predictor.models.line_angle.LineAngleModel", side_effect=lambda **_: ToyCoarse())
    def test_training_initialization_and_portable_checkpoint(self, _mock) -> None:
        with TemporaryDirectory() as directory:
            params, source, coarse_checkpoint = self.make_source(directory)
            model = _build_model(params, (16, 16)).eval()
            images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
            torch.testing.assert_close(model(images), source(images))
            torch.nn.init.normal_(model.network.correction[-1].weight, std=0.01)
            expected = model(images).detach()
            composite_checkpoint = Path(directory) / "composite.pt"
            save_checkpoint(
                model,
                composite_checkpoint,
                metadata={
                    "format_version": 1,
                    "input_size": [16, 16],
                    "params": params.model_dump(mode="json"),
                },
            )
            coarse_checkpoint.unlink()
            restored, size, precision = load_angle_predictor(
                composite_checkpoint, torch.device("cpu")
            )
            self.assertEqual(size, (16, 16))
            self.assertEqual(precision, "fp32")
            torch.testing.assert_close(restored(images), expected)

    def test_source_split_mismatch_is_rejected_before_model_construction(self) -> None:
        with TemporaryDirectory() as directory:
            params, _source, checkpoint = self.make_source(directory)
            payload = torch.load(checkpoint, weights_only=True)
            payload["metadata"]["params"]["data"]["split_seed"] = 43
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "different split seed"):
                _build_model(params, (16, 16))

    def test_source_preprocessing_and_dataset_fingerprint_are_checked(self) -> None:
        with TemporaryDirectory() as directory:
            params, _source, checkpoint = self.make_source(directory)
            payload = torch.load(checkpoint, weights_only=True)
            payload["metadata"]["params"]["preprocessing"]["mean"] = [0.4, 0.5, 0.5]
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "preprocessing differs"):
                _build_model(params, (16, 16))
            payload["metadata"]["params"]["preprocessing"]["mean"] = [0.5, 0.5, 0.5]
            payload["metadata"]["dataset_meta_sha256"] = "invalid"
            torch.save(payload, checkpoint)
            with self.assertRaisesRegex(ValueError, "fingerprint differs"):
                _build_model(params, (16, 16))

    def test_config_rejects_split_mismatch_and_invalid_crop(self) -> None:
        with TemporaryDirectory() as directory:
            params, _source, _checkpoint = self.make_source(directory)
            options = params.model_dump(mode="json")
            options["model"]["coarse_split_seed"] = 43
            with self.assertRaises(ValidationError):
                AngleExperimentParams.model_validate(options)
            options["model"]["coarse_split_seed"] = 42
            options["model"]["crop_size"] = [16, 8]
            with self.assertRaises(ValidationError):
                AngleExperimentParams.model_validate(options)
