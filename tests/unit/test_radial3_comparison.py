from unittest import TestCase

import torch

from angle_predictor.models.polar_refinement import PolarRefinementModel
from tests.unit.test_polar_refinement import ToyCoarse


class Radial3ComparisonTests(TestCase):
    def test_radial_support_and_parameters_change_without_new_sampling(self):
        options = dict(
            fine_channels=(16, 32, 128),
            refinement_hidden=128,
            fine_radial_antialias=True,
            crop_size=(384, 65),
            fine_angular_kernel=23,
        )
        baseline = PolarRefinementModel(ToyCoarse(), **options, fine_radial_kernel=5)
        candidate = PolarRefinementModel(ToyCoarse(), **options, fine_radial_kernel=3)
        self.assertEqual(
            sum(p.numel() for p in candidate.parameters())
            - sum(p.numel() for p in baseline.parameters()),
            -215648,
        )
        self.assertEqual(candidate.correction[0].in_features, baseline.correction[0].in_features)
        self.assertEqual(candidate.fine(torch.randn(1, 5, 384, 65)).shape, (1, 128, 48, 65))
        for model, expected in ((baseline, 43), (candidate, 29)):
            support, jump = 1, 1
            for layer in model.fine:
                if isinstance(layer, torch.nn.Conv2d):
                    support += (layer.kernel_size[0] - 1) * jump
                    jump *= layer.stride[0]
                elif type(layer).__name__ == "RadialAntialias":
                    support += 2 * jump
            self.assertEqual(support, expected)
            self.assertEqual(jump, 8)

    def test_narrower_encoder_has_finite_gradients_and_frozen_coarse(self):
        model = PolarRefinementModel(
            ToyCoarse(),
            crop_size=(48, 17),
            fine_channels=(16, 32, 128),
            fine_radial_kernel=3,
            fine_angular_kernel=23,
            fine_radial_antialias=True,
        ).train()
        image = torch.randn(2, 3, 48, 36)
        with torch.no_grad():
            model.correction[-1].weight.normal_(std=0.001)
        output = model(image)
        self.assertTrue(torch.isfinite(output).all())
        output[:, 0].sum().backward()
        self.assertFalse(model.coarse.training)
        self.assertTrue(all(p.grad is None for p in model.coarse.parameters()))
        self.assertTrue(
            all(
                p.grad is not None and torch.isfinite(p.grad).all() for p in model.fine.parameters()
            )
        )
