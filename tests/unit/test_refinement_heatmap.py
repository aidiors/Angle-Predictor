import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch
from torch import nn

from angle_predictor.config.angle_experiment import AngleExperimentParams
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.experiments.angle import _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.models.polar_refinement import PolarRefinementModel, angular_heatmap_offset
from tests.unit.test_end_to_end_models import Coarse


class AngleCoarse(nn.Module):
    def __init__(self):
        super().__init__()
        self.angles = nn.Parameter(torch.tensor([0.0001, math.pi - 0.0001, 0.7, 2.7]))

    def forward(self, image):
        return torch.stack((torch.sin(2 * self.angles), torch.cos(2 * self.angles)), -1)


class HeatmapTests(TestCase):
    def test_uniform_scores_are_exactly_zero_in_fp32_and_bf16(self):
        offsets = torch.linspace(0, math.radians(4), 17)[1:]
        for dtype in (torch.float32, torch.bfloat16):
            scores = torch.zeros(3, 33, dtype=dtype, requires_grad=True)
            result = angular_heatmap_offset(scores, offsets)
            self.assertEqual(result.dtype, torch.float32)
            torch.testing.assert_close(result, torch.zeros(3), rtol=0, atol=0)
            result.sum().backward()
            self.assertTrue(torch.isfinite(scores.grad).all())
            self.assertGreater(scores.grad.abs().sum().item(), 0)

    def test_decoder_sign_bounds_symmetry_and_continuous_subcolumn_position(self):
        offsets = torch.linspace(0, math.radians(4), 17)[1:]
        scores = torch.full((1, 33), -100.0)
        scores[0, 17:19] = torch.tensor([math.log(0.75), math.log(0.25)])
        actual = angular_heatmap_offset(scores, offsets)
        torch.testing.assert_close(actual, torch.tensor([math.radians(0.3125)]))
        torch.testing.assert_close(angular_heatmap_offset(scores.flip(-1), offsets), -actual)
        for index, expected in ((0, -4), (32, 4)):
            peaked = torch.full((1, 33), -100.0)
            peaked[0, index] = 100
            value = angular_heatmap_offset(peaked, offsets)
            torch.testing.assert_close(value, torch.tensor([math.radians(expected)]))
            self.assertLessEqual(value.abs().item(), offsets[-1].item())

    def test_initial_heatmap_is_exact_coarse_and_mlp_state_layout_is_unchanged(self):
        coarse = Coarse()
        options = dict(crop_size=(16, 9), fine_channels=(4, 8), refinement_hidden=8)
        heatmap = PolarRefinementModel(coarse, **options, refinement_head="heatmap").eval()
        image = torch.randn(2, 3, 16, 36)
        torch.testing.assert_close(heatmap(image), coarse(image), rtol=0, atol=0)
        legacy = PolarRefinementModel(Coarse(), **options)
        self.assertNotIn("positive_offsets", legacy.state_dict())
        self.assertEqual(legacy.correction[0].weight.shape, (8, 146))
        restored = PolarRefinementModel(Coarse(), **options, refinement_head="mlp")
        restored.load_state_dict(legacy.state_dict(), strict=True)
        torch.testing.assert_close(legacy(image), restored(image), rtol=0, atol=0)

    def test_final_loss_gradients_cross_coarse_fine_and_score_head_near_wrap(self):
        torch.manual_seed(42)
        model = PolarRefinementModel(
            AngleCoarse(),
            crop_size=(16, 9),
            fine_channels=(4, 8),
            refinement_hidden=8,
            refinement_head="heatmap",
            train_coarse=True,
        )
        nn.init.normal_(model.correction[-1].weight, std=0.02)
        output = model(torch.randn(4, 3, 24, 36))
        torch.testing.assert_close(output.norm(dim=-1), torch.ones(4))
        output[:, 0].sum().backward()
        for module in (model.coarse, model.fine, model.correction):
            gradients = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)

    @patch("angle_predictor.models.line_angle.LineAngleModel", side_effect=lambda **_: Coarse())
    def test_heatmap_official_loader_ignores_missing_data_and_preserves_predictions(self, mock):
        params = AngleExperimentParams.model_validate(
            {
                "data": {"root": "Z:/missing-angle-data"},
                "model": {
                    "kind": "polar_refinement",
                    "coarse_model": {"pretrained": True},
                    "initialization": "imagenet",
                    "train_coarse": True,
                    "refinement_head": "heatmap",
                    "crop_size": [16, 9],
                    "fine_channels": [4, 8],
                    "refinement_hidden": 8,
                },
                "preprocessing": {"output_size": [16, 36]},
                "training": {"precision": "fp32"},
            }
        )
        model = _build_model(params, (16, 16)).eval()
        nn.init.normal_(model.network.correction[-1].weight, std=0.02)
        images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
        expected = model(images)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "heatmap.pt"
            save_checkpoint(
                model,
                path,
                metadata={
                    "format_version": 1,
                    "input_size": [16, 16],
                    "params": params.model_dump(mode="json"),
                },
            )
            restored, _, _ = load_angle_predictor(path, torch.device("cpu"))
            self.assertFalse(mock.call_args.kwargs["pretrained"])
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
