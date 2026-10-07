import math
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch

from angle_predictor.config.angle_experiment import PolarRefinementModelConfig
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.experiments.angle import _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.models.polar_refinement import PolarRefinementModel, radial_feature_profile
from tests.unit import test_polar_refinement as fixtures


class FourViewProjectionTests(TestCase):
    def model(self):
        return PolarRefinementModel(
            fixtures.ToyCoarse(),
            crop_size=(24, 9),
            fine_channels=(4, 8),
            fine_radial_antialias=True,
            refinement_hidden=8,
            window_deg=4,
            fine_angular_antisymmetry=True,
            fine_radial_reflection=True,
            fine_reflection_readout="offset",
        )

    def test_group_identities_for_learned_readout(self):
        torch.manual_seed(21)
        model = self.model().eval()
        torch.nn.init.normal_(model.correction[-1].weight, std=0.2)
        torch.nn.init.constant_(model.correction[-1].bias, 0.3)
        crop = torch.randn(3, 3, 24, 9)
        coarse = torch.nn.functional.normalize(torch.randn(3, 2), dim=-1)
        normal = model.refine_crop(crop, coarse)
        self.assertGreater(normal.abs().sum().item(), 0)
        torch.testing.assert_close(normal, model.refine_crop(crop.flip(-2), coarse), rtol=0, atol=0)
        torch.testing.assert_close(
            normal, -model.refine_crop(crop.flip(-1), coarse), rtol=0, atol=0
        )
        torch.testing.assert_close(
            normal, -model.refine_crop(crop.flip((-2, -1)), coarse), rtol=0, atol=0
        )
        symmetric = (crop + crop.flip(-1)) * 0.5
        torch.testing.assert_close(
            model.refine_crop(symmetric, coarse), torch.zeros(3), rtol=0, atol=0
        )
        self.assertTrue(torch.isfinite(normal).all())
        self.assertLessEqual(normal.abs().max().item(), math.radians(4))

    def test_manual_four_view_readout_matches_projection(self):
        model = self.model().eval()
        torch.nn.init.normal_(model.correction[-1].weight, std=0.1)
        crop = torch.randn(2, 3, 24, 9)
        coarse = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
        views = torch.cat((crop, crop.flip(-1), crop.flip(-2), crop.flip((-2, -1))))
        radius = model.radius.expand(len(views), -1, -1, 9)
        features = model.fine(torch.cat((views, radius, radius.abs()), dim=1))
        profile = radial_feature_profile(features).flatten(1)
        scores = model.correction(torch.cat((profile, coarse.repeat(4, 1)), dim=1)).squeeze(-1)
        c, a, r, ar = (math.radians(4) * scores.tanh()).chunk(4)
        expected = ((c + r) * 0.5 - (a + ar) * 0.5) * 0.5
        torch.testing.assert_close(model.refine_crop(crop, coarse), expected, rtol=0, atol=0)

    def test_all_views_receive_gradients_and_keep_fixed_radius_coordinates(self):
        model = self.model().train()
        torch.nn.init.normal_(model.correction[-1].weight, std=0.03)
        source = deepcopy(model.coarse.state_dict())
        captured = []

        def retain(_module, inputs, output):
            output.retain_grad()
            captured.append((inputs[0], output))

        handle = model.fine.register_forward_hook(retain)
        model(torch.randn(2, 3, 32, 36))[:, 0].sum().backward()
        handle.remove()
        values, features = captured[0]
        self.assertEqual(values.shape, (8, 5, 24, 9))
        original, angular, radial, both = values.chunk(4)
        for transformed, dimensions in ((angular, -1), (radial, -2), (both, (-2, -1))):
            torch.testing.assert_close(
                transformed[:, :3], original[:, :3].flip(dimensions), rtol=0, atol=0
            )
            torch.testing.assert_close(transformed[:, 3:], original[:, 3:], rtol=0, atol=0)
        for gradient in features.grad.chunk(4):
            self.assertTrue(torch.isfinite(gradient).all())
            self.assertGreater(gradient.abs().sum().item(), 0)
        self.assertFalse(model.coarse.training)
        self.assertTrue(all(p.grad is None for p in model.coarse.parameters()))
        for key, value in source.items():
            self.assertTrue(torch.equal(value, model.coarse.state_dict()[key]))

    def test_zero_initialization_preserves_source_and_parameter_keys(self):
        torch.manual_seed(5)
        candidate = self.model().eval()
        rng = torch.get_rng_state()
        torch.manual_seed(5)
        baseline = PolarRefinementModel(
            fixtures.ToyCoarse(),
            crop_size=(24, 9),
            fine_channels=(4, 8),
            fine_radial_antialias=True,
            refinement_hidden=8,
            window_deg=4,
            fine_angular_antisymmetry=True,
        ).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(candidate.state_dict()), list(baseline.state_dict()))
        candidate.load_state_dict(baseline.state_dict(), strict=True)
        images = torch.randn(2, 3, 32, 36)
        torch.testing.assert_close(candidate(images), candidate.coarse(images), rtol=0, atol=0)

    def test_learned_checkpoint_is_portable_without_data_or_sources(self):
        with TemporaryDirectory() as directory:
            params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
            params.model = PolarRefinementModelConfig.model_validate(
                {
                    **params.model.model_dump(),
                    "fine_angular_antisymmetry": True,
                    "fine_radial_reflection": True,
                    "fine_reflection_readout": "offset",
                }
            )
            with patch(
                "angle_predictor.models.line_angle.LineAngleModel",
                side_effect=lambda **_: fixtures.ToyCoarse(),
            ):
                model = _build_model(params, (16, 16)).eval()
                torch.nn.init.normal_(model.network.correction[-1].weight, std=0.04)
                images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
                expected = model(images)
                metadata = {
                    "format_version": 1,
                    "input_size": [16, 16],
                    "params": params.model_dump(mode="json"),
                }
                metadata["params"]["data"]["root"] = "missing-data"
                metadata["params"]["model"]["coarse_checkpoint"] = "missing-coarse.pt"
                path = Path(directory) / "portable.pt"
                save_checkpoint(model, path, metadata=metadata)
                restored, _, _ = load_angle_predictor(path, torch.device("cpu"))
                self.assertTrue(restored.network.fine_angular_antisymmetry)
                self.assertTrue(restored.network.fine_radial_reflection)
                torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
