import math
from unittest import TestCase

import torch

from angle_predictor.models.polar_refinement import PolarRefinementModel, crop_signed_polar
from tests.unit.test_polar_refinement import ToyCoarse


class Angular129ComparisonTests(TestCase):
    def test_nested_sampling_grid_keeps_original_samples_near_wrap(self):
        image = torch.randn(4, 3, 48, 360)
        angles = torch.tensor([0, math.pi - 0.00001, math.pi / 2, 0.7])
        original = crop_signed_polar(image, angles, crop_size=(48, 65), window_deg=4)
        denser = crop_signed_polar(image, angles, crop_size=(48, 129), window_deg=4)
        torch.testing.assert_close(denser[..., ::2], original, rtol=0, atol=0)

    def test_angular_context_and_parameter_increase_match_the_design(self):
        old_span = 3 * (23 - 1) * 8 / (65 - 1)
        new_span = 3 * (45 - 1) * 8 / (129 - 1)
        self.assertEqual(old_span, new_span)
        self.assertEqual(new_span, 8.25)
        options = dict(
            fine_channels=(16, 32, 128), refinement_hidden=128, fine_radial_antialias=True
        )
        baseline = PolarRefinementModel(
            ToyCoarse(), **options, crop_size=(384, 65), fine_angular_kernel=23
        )
        candidate = PolarRefinementModel(
            ToyCoarse(), **options, crop_size=(384, 129), fine_angular_kernel=45
        )
        self.assertEqual(
            sum(p.numel() for p in candidate.parameters())
            - sum(p.numel() for p in baseline.parameters()),
            2612832,
        )
        self.assertEqual(candidate.correction[0].in_features, 33026)
        self.assertEqual(candidate.fine(torch.randn(1, 5, 32, 129)).shape, (1, 128, 4, 129))
