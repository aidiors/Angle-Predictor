from unittest import TestCase

import torch

from angle_predictor.models.polar_refinement import PolarRefinementModel
from tests.unit.test_polar_refinement import ToyCoarse


class StagedHeatmapComparisonTests(TestCase):
    def test_selected_encoder_and_frozen_coarse_work_with_geometric_head(self):
        options = dict(
            crop_size=(384, 65),
            window_deg=4,
            fine_channels=(16, 32, 128),
            fine_angular_kernel=23,
            fine_radial_antialias=True,
            refinement_hidden=128,
        )
        torch.manual_seed(57)
        baseline = PolarRefinementModel(ToyCoarse(), **options)
        torch.manual_seed(57)
        candidate = PolarRefinementModel(ToyCoarse(), **options, refinement_head="heatmap")
        self.assertEqual(
            sum(p.numel() for p in baseline.parameters())
            - sum(p.numel() for p in candidate.parameters()),
            1966336,
        )
        for name, value in baseline.fine.state_dict().items():
            torch.testing.assert_close(candidate.fine.state_dict()[name], value, rtol=0, atol=0)
        self.assertEqual(candidate.positive_offsets.shape, (32,))
        image = torch.randn(2, 3, 16, 36)
        candidate.train()
        coarse_before = {
            name: value.clone() for name, value in candidate.coarse.state_dict().items()
        }
        torch.testing.assert_close(candidate(image), candidate.coarse(image), rtol=0, atol=0)
        torch.nn.init.normal_(candidate.correction[-1].weight, std=0.001)
        output = candidate(image)
        self.assertTrue(torch.isfinite(output).all())
        output[:, 0].sum().backward()
        for branch in (candidate.fine, candidate.correction):
            gradients = [p.grad for p in branch.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
        self.assertFalse(candidate.coarse.training)
        self.assertTrue(all(p.grad is None for p in candidate.coarse.parameters()))
        for name, value in candidate.coarse.state_dict().items():
            torch.testing.assert_close(value, coarse_before[name], rtol=0, atol=0)
