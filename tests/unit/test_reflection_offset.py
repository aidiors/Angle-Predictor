import math
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
from angle_predictor.models.polar_refinement import PolarRefinementModel, rotate_double_angle
from tests.unit import test_fine_radial_reflection as reflection
from tests.unit import test_polar_refinement as fixtures


class ReflectionOffsetTests(TestCase):
    def make_model(self, readout="offset"):
        return PolarRefinementModel(
            fixtures.ToyCoarse(),
            **reflection.RadialReflectionTests().options(),
            fine_radial_reflection=True,
            fine_reflection_readout=readout,
        )

    def test_profile_default_preserves_existing_parameters_rng_and_output(self):
        torch.manual_seed(7)
        implicit = PolarRefinementModel(
            fixtures.ToyCoarse(),
            **reflection.RadialReflectionTests().options(),
            fine_radial_reflection=True,
        ).eval()
        implicit_rng = torch.get_rng_state()
        torch.manual_seed(7)
        explicit = self.make_model("profile").eval()
        self.assertTrue(torch.equal(implicit_rng, torch.get_rng_state()))
        for key, value in implicit.state_dict().items():
            self.assertTrue(torch.equal(value, explicit.state_dict()[key]))
        images = torch.randn(2, 3, 32, 36)
        torch.testing.assert_close(implicit(images), explicit(images), rtol=0, atol=0)
        late = self.make_model().eval()
        self.assertEqual(set(late.state_dict()), set(implicit.state_dict()))
        late.load_state_dict(implicit.state_dict())
        torch.testing.assert_close(late(images), implicit(images), rtol=0, atol=0)

    def test_offset_is_mean_of_bounded_per_view_corrections_with_shared_conditioning(self):
        model = self.make_model().eval()
        torch.nn.init.normal_(model.correction[-1].weight, std=0.1)
        inputs, logits, coarse = [], [], []
        hooks = [
            model.correction.register_forward_pre_hook(lambda _m, x: inputs.append(x[0].detach())),
            model.correction.register_forward_hook(lambda _m, _x, y: logits.append(y.detach())),
            model.coarse.register_forward_hook(lambda _m, _x, y: coarse.append(y.detach().float())),
        ]
        try:
            actual = model(torch.randn(3, 3, 32, 36))
        finally:
            for hook in hooks:
                hook.remove()
        self.assertEqual(inputs[0].shape[0], 6)
        torch.testing.assert_close(inputs[0][:, -2:], coarse[0].repeat(2, 1), rtol=0, atol=0)
        bounded = logits[0].squeeze(-1).tanh() * math.radians(model.window_deg)
        a, b = bounded.chunk(2)
        torch.testing.assert_close(
            actual, rotate_double_angle(coarse[0], (a + b) * 0.5), rtol=0, atol=0
        )
        self.assertTrue(((a + b).abs() * 0.5 <= math.radians(model.window_deg)).all())

    def test_conditional_reflection_invariance_and_gradients_through_both_views(self):
        model = self.make_model().train()
        torch.nn.init.normal_(model.correction[-1].weight, std=0.02)
        image = torch.randn(2, 3, 32, 36)
        crop = torch.randn(2, 3, *model.crop_size)
        with patch("angle_predictor.models.polar_refinement.crop_signed_polar", return_value=crop):
            a = model(image)
        with patch(
            "angle_predictor.models.polar_refinement.crop_signed_polar", return_value=crop.flip(-2)
        ):
            b = model(image)
        torch.testing.assert_close(a, b, rtol=0, atol=1e-7)
        captured = []

        def retain(_m, _x, y):
            y.retain_grad()
            captured.append(y)

        hook = model.fine.register_forward_hook(retain)
        try:
            model(image)[:, 0].sum().backward()
        finally:
            hook.remove()
        for grad in captured[0].grad.chunk(2):
            self.assertTrue(torch.isfinite(grad).all())
            self.assertGreater(grad.abs().sum().item(), 0)
        self.assertFalse(model.coarse.training)
        self.assertTrue(all(p.grad is None for p in model.coarse.parameters()))

    def test_factory_and_portable_checkpoint_preserve_late_readout(self):
        with TemporaryDirectory() as directory:
            params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
            params.model = PolarRefinementModelConfig.model_validate(
                {
                    **params.model.model_dump(),
                    "fine_radial_reflection": True,
                    "fine_reflection_readout": "offset",
                }
            )
            with patch(
                "angle_predictor.models.line_angle.LineAngleModel",
                side_effect=lambda **_: fixtures.ToyCoarse(),
            ):
                model = _build_model(params, (16, 16)).eval()
                torch.nn.init.normal_(model.network.correction[-1].weight, std=0.02)
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
                self.assertEqual(restored.network.fine_reflection_readout, "offset")
                torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)

    def test_disabled_reflection_cannot_silently_enable_offset_readout(self):
        with self.assertRaisesRegex(ValueError, "requires radial reflection"):
            PolarRefinementModel(fixtures.ToyCoarse(), fine_reflection_readout="offset")
        with self.assertRaisesRegex(ValueError, "Unknown reflection readout"):
            self.make_model("bad")
        with TemporaryDirectory() as directory:
            params, _, _ = fixtures.PolarRefinementTests().make_source(directory)
            with self.assertRaisesRegex(ValidationError, "requires radial reflection"):
                PolarRefinementModelConfig.model_validate(
                    {**params.model.model_dump(), "fine_reflection_readout": "offset"}
                )
