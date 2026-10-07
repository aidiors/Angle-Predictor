from unittest import TestCase

import torch

from angle_predictor.models.polar_refinement import PolarRefinementModel
from tests.unit.test_polar_refinement import ToyCoarse


class FineStemComparisonTests(TestCase):
    def test_wider_stem_preserves_output_and_head_dimensions(self):
        options = dict(
            crop_size=(24, 9),
            refinement_hidden=8,
            fine_radial_antialias=True,
            fine_angular_kernel=23,
        )
        baseline = PolarRefinementModel(ToyCoarse(), fine_channels=(16, 32, 128), **options)
        candidate = PolarRefinementModel(ToyCoarse(), fine_channels=(32, 64, 128), **options)
        self.assertIsNone(candidate.detail_projection)
        self.assertEqual(baseline.correction[0].in_features, candidate.correction[0].in_features)
        self.assertEqual(
            sum(p.numel() for p in candidate.parameters())
            - sum(p.numel() for p in baseline.parameters()),
            657024,
        )
        image = torch.randn(2, 3, 32, 36)
        candidate.train()
        with torch.no_grad():
            candidate.correction[-1].weight.normal_(std=0.001)
        output = candidate(image)
        self.assertEqual(output.shape, (2, 2))
        self.assertTrue(torch.isfinite(output).all())
        output[:, 0].sum().backward()
        self.assertFalse(candidate.coarse.training)
        self.assertTrue(all(p.grad is None for p in candidate.coarse.parameters()))
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all()
                for p in candidate.fine.parameters()
            )
        )
