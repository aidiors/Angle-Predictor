import math
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch
from pydantic import ValidationError

from angle_predictor.config.angle_experiment import PolarRefinementModelConfig
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.experiments.angle import _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.models.line_moments import WeightedLineMoment, weighted_line_offset
from angle_predictor.models.polar_refinement import PolarRefinementModel
from tests.unit import test_polar_refinement as fixtures


class LineMomentTests(TestCase):
    def options(self):
        return {"crop_size": (24, 9), "fine_channels": (4, 8), "refinement_hidden": 8}

    def test_origin_constrained_fit_recovers_each_angular_column(self):
        offsets = torch.linspace(0, math.radians(4), 5)[1:]
        radius = torch.tensor([1.0, 0.25, 0.0, 0.25, 1.0]).view(1, 1, 5, 1)
        weights = torch.zeros(9, 1, 5, 9)
        for column in range(9):
            weights[column, 0, 0, column] = 2
            weights[column, 0, -1, column] = 3
        actual = weighted_line_offset(weights, radius, offsets)
        torch.testing.assert_close(actual, torch.linspace(-math.radians(4), math.radians(4), 9))

    def test_radial_leverage_changes_fit_and_matches_second_moment_solution(self):
        offsets = torch.linspace(0, math.radians(4), 5)[1:]
        radius = torch.tensor([1.0, 0.01]).view(1, 1, 2, 1)
        weights = torch.zeros(1, 1, 2, 9)
        weights[0, 0, 0, 8] = 1
        weights[0, 0, 1, 0] = 1
        actual = weighted_line_offset(weights, radius, offsets)
        theta = math.radians(4)
        expected = 0.5 * math.atan2(0.99 * math.sin(2 * theta), 1.01 * math.cos(2 * theta))
        self.assertAlmostEqual(actual.item(), expected, places=7)
        self.assertGreater(actual.item(), math.radians(3.9))
        reverse = weighted_line_offset(weights.flip(-1), radius, offsets)
        torch.testing.assert_close(reverse, -actual, rtol=0, atol=0)

    def test_uniform_support_is_exactly_zero_and_extreme_logits_stay_finite(self):
        model = WeightedLineMoment(4, (24, 9), 4, 4)
        features = torch.randn(2, 4, 6, 9, requires_grad=True)
        torch.testing.assert_close(model(features), torch.zeros(2), rtol=0, atol=0)
        for bias in (-1000.0, 1000.0):
            with torch.no_grad():
                model.projection.bias.fill_(bias)
            value = model(features)
            self.assertTrue(torch.isfinite(value).all())
            gradients = torch.autograd.grad(value.sum(), features, retain_graph=True)[0]
            self.assertTrue(torch.isfinite(gradients).all())
        with torch.no_grad():
            model.projection.weight.normal_()
            model.projection.bias.zero_()
        value = model(features)
        self.assertTrue((value.abs() <= math.radians(4) + 1e-7).all())
        self.assertTrue(torch.isfinite(torch.autograd.grad(value.sum(), features)[0]).all())

    def test_zero_branch_preserves_legacy_rng_weights_and_nonzero_predictions(self):
        torch.manual_seed(57)
        baseline = PolarRefinementModel(fixtures.ToyCoarse(), **self.options()).eval()
        rng = torch.get_rng_state()
        torch.manual_seed(57)
        candidate = PolarRefinementModel(
            fixtures.ToyCoarse(), **self.options(), fine_geometric_residual=True
        ).eval()
        self.assertTrue(torch.equal(rng, torch.get_rng_state()))
        for name, value in baseline.state_dict().items():
            self.assertTrue(torch.equal(value, candidate.state_dict()[name]))
        torch.nn.init.normal_(baseline.correction[-1].weight, std=0.02)
        candidate.correction.load_state_dict(baseline.correction.state_dict())
        image = torch.randn(3, 3, 32, 36)
        torch.testing.assert_close(candidate(image), baseline(image), rtol=0, atol=0)

    def test_both_heads_and_fine_learn_without_updating_coarse(self):
        model = PolarRefinementModel(
            fixtures.ToyCoarse(), **self.options(), fine_geometric_residual=True
        ).train()
        source = deepcopy(model.coarse.state_dict())
        optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=0.01)
        image = torch.randn(3, 3, 32, 36)
        target = torch.nn.functional.normalize(torch.randn(3, 2), dim=-1)
        for _ in range(4):
            optimizer.zero_grad()
            output = model(image)
            (output - target).square().mean().backward()
            optimizer.step()
        for branch in (model.fine, model.correction, model.geometric_readout):
            gradients = [p.grad for p in branch.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertTrue(any(g.abs().sum() > 0 for g in gradients))
        self.assertFalse(model.coarse.training)
        self.assertTrue(all(p.grad is None for p in model.coarse.parameters()))
        for name, value in model.coarse.state_dict().items():
            self.assertTrue(torch.equal(value, source[name]))

    def test_factory_checkpoint_is_portable_with_learned_geometry(self):
        with TemporaryDirectory() as directory:
            params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
            params.model = PolarRefinementModelConfig.model_validate(
                {**params.model.model_dump(), "fine_geometric_residual": True}
            )
            with patch(
                "angle_predictor.models.line_angle.LineAngleModel",
                side_effect=lambda **_: fixtures.ToyCoarse(),
            ):
                model = _build_model(params, (16, 16)).eval()
                torch.nn.init.normal_(model.network.geometric_readout.projection.weight, std=0.01)
                image = torch.randint(0, 256, (3, 3, 16, 16), dtype=torch.uint8)
                expected = model(image)
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
                torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)

    def test_incompatible_combinations_rejected_by_config_and_model(self):
        combinations = (
            {"refinement_head": "heatmap"},
            {"fine_radial_reflection": True},
            {"fine_detail_skip": True},
            {"radial_pool_bins": 2},
            {"window_deg": 45},
        )
        with TemporaryDirectory() as directory:
            params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
            for extra in combinations:
                with self.assertRaisesRegex(ValueError, "Geometric residual"):
                    PolarRefinementModel(
                        fixtures.ToyCoarse(), fine_geometric_residual=True, **extra
                    )
                with self.assertRaisesRegex(ValidationError, "Geometric residual"):
                    PolarRefinementModelConfig.model_validate(
                        {**params.model.model_dump(), "fine_geometric_residual": True, **extra}
                    )
