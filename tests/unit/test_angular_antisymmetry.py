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
from angle_predictor.models.polar_refinement import (
    PolarRefinementModel,
    crop_signed_polar,
    radial_feature_profile,
    rotate_double_angle,
)
from tests.unit import test_polar_refinement as fixtures


class AngularAntisymmetryTests(TestCase):
    def options(self):
        return dict(
            crop_size=(24, 9),
            fine_channels=(4, 8),
            fine_radial_antialias=True,
            refinement_hidden=8,
            window_deg=4.0,
        )

    def test_legacy_state_rng_nonzero_predictions_and_zero_source_are_preserved(self):
        torch.manual_seed(17)
        legacy = PolarRefinementModel(fixtures.ToyCoarse(), **self.options()).eval()
        rng = torch.get_rng_state()
        torch.manual_seed(17)
        candidate = PolarRefinementModel(
            fixtures.ToyCoarse(), **self.options(), fine_angular_antisymmetry=True
        ).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        self.assertEqual(list(legacy.state_dict()), list(candidate.state_dict()))
        candidate.load_state_dict(legacy.state_dict(), strict=True)
        images = torch.randn(2, 3, 32, 36)
        torch.testing.assert_close(candidate(images), candidate.coarse(images), rtol=0, atol=0)
        torch.nn.init.normal_(legacy.correction[-1].weight, std=0.02)
        coarse = legacy.coarse(images).float()
        angle = (0.5 * torch.atan2(coarse[:, 0], coarse[:, 1])).remainder(math.pi)
        crop = crop_signed_polar(
            images, angle, crop_size=legacy.crop_size, window_deg=legacy.window_deg
        )
        radius = legacy.radius.expand(len(crop), -1, -1, 9)
        features = legacy.fine(torch.cat((crop, radius, radius.abs()), dim=1))
        profile = radial_feature_profile(features).flatten(1)
        score = legacy.correction(torch.cat((profile, coarse), dim=1)).squeeze(-1)
        expected = rotate_double_angle(coarse, math.radians(4) * score.tanh())
        torch.testing.assert_close(legacy(images), expected, rtol=0, atol=0)

    def test_learned_conditional_readout_is_odd_bounded_and_zero_on_symmetric_crops(self):
        model = PolarRefinementModel(
            fixtures.ToyCoarse(), **self.options(), fine_angular_antisymmetry=True
        ).eval()
        torch.nn.init.normal_(model.correction[-1].weight, std=0.4)
        torch.nn.init.constant_(model.correction[-1].bias, 0.25)
        crop = torch.randn(3, 3, 24, 9)
        coarse = torch.nn.functional.normalize(torch.randn(3, 2), dim=-1)
        normal = model.refine_crop(crop, coarse)
        mirrored = model.refine_crop(crop.flip(-1), coarse)
        self.assertGreater(normal.abs().sum().item(), 0)
        torch.testing.assert_close(normal, -mirrored, rtol=0, atol=0)
        self.assertTrue(torch.isfinite(normal).all())
        self.assertLessEqual(normal.abs().max().item(), math.radians(4))
        symmetric = (crop + crop.flip(-1)) * 0.5
        torch.testing.assert_close(
            model.refine_crop(symmetric, coarse), torch.zeros(3), rtol=0, atol=0
        )

    def test_both_views_keep_radius_coordinates_and_receive_finite_gradients(self):
        model = PolarRefinementModel(
            fixtures.ToyCoarse(), **self.options(), fine_angular_antisymmetry=True
        ).train()
        torch.nn.init.normal_(model.correction[-1].weight, std=0.02)
        source = deepcopy(model.coarse.state_dict())
        captured = []

        def retain(_module, inputs, output):
            output.retain_grad()
            captured.append((inputs[0], output))

        handle = model.fine.register_forward_hook(retain)
        prediction = model(torch.randn(2, 3, 32, 36))
        prediction[:, 0].sum().backward()
        handle.remove()
        values, features = captured[0]
        self.assertEqual(values.shape, (4, 5, 24, 9))
        torch.testing.assert_close(values[2:, :3], values[:2, :3].flip(-1), rtol=0, atol=0)
        torch.testing.assert_close(values[2:, 3:], values[:2, 3:], rtol=0, atol=0)
        for view in features.grad.chunk(2):
            self.assertTrue(torch.isfinite(view).all())
            self.assertGreater(view.abs().sum().item(), 0)
        self.assertFalse(model.coarse.training)
        self.assertTrue(all(p.grad is None for p in model.coarse.parameters()))
        for key, value in source.items():
            self.assertTrue(torch.equal(value, model.coarse.state_dict()[key]))

    def test_factory_restores_learned_constraint_without_ancestors(self):
        with TemporaryDirectory() as directory:
            params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
            params.model = PolarRefinementModelConfig.model_validate(
                {**params.model.model_dump(), "fine_angular_antisymmetry": True}
            )
            with patch(
                "angle_predictor.models.line_angle.LineAngleModel",
                side_effect=lambda **_: fixtures.ToyCoarse(),
            ):
                model = _build_model(params, (16, 16)).eval()
                torch.nn.init.normal_(model.network.correction[-1].weight, std=0.03)
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
                torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)

    def test_incompatible_paths_rejected_by_config_and_constructor(self):
        for extra in (
            {"refinement_head": "heatmap"},
            {"radial_pool_bins": 2},
            {"radial_pool_mode": "attention_max"},
            {"fine_detail_skip": True},
            {"fine_radial_reflection": True},
            {"fine_geometric_residual": True},
        ):
            with self.assertRaisesRegex(ValueError, "Angular antisymmetry"):
                PolarRefinementModel(fixtures.ToyCoarse(), fine_angular_antisymmetry=True, **extra)
            with TemporaryDirectory() as directory:
                params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
                with self.assertRaisesRegex(ValueError, "Angular antisymmetry"):
                    PolarRefinementModelConfig.model_validate(
                        {
                            **params.model.model_dump(),
                            "fine_angular_antisymmetry": True,
                            **extra,
                        }
                    )
