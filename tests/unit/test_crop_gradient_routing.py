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
from angle_predictor.losses.angle import vector_charbonnier_angle_loss
from angle_predictor.models.polar_refinement import PolarRefinementModel, rotate_double_angle
from tests.unit.test_end_to_end_models import Coarse
from tests.unit.test_refinement_heatmap import AngleCoarse


class CropGradientRoutingTests(TestCase):
    def make_model(self, detach=False):
        return PolarRefinementModel(
            AngleCoarse(),
            crop_size=(16, 9),
            fine_channels=(4, 8),
            refinement_hidden=8,
            refinement_head="local_mlp",
            train_coarse=True,
            window_deg=4,
            detach_crop_angle=detach,
        )

    def test_same_weights_keep_identical_outputs_and_legacy_state_shapes(self):
        torch.manual_seed(42)
        original = self.make_model().eval()
        detached = self.make_model(True).eval()
        nn.init.normal_(original.correction[-1].weight, std=0.2)
        detached.load_state_dict(original.state_dict(), strict=True)
        self.assertFalse(original.detach_crop_angle)
        self.assertEqual(original.state_dict().keys(), detached.state_dict().keys())
        image = torch.randn(4, 3, 24, 36)
        for bf16 in (False, True):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                expected, actual = original(image), detached(image)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)

    def test_coarse_gradient_is_direct_rotation_derivative_without_sampling_feedback(self):
        torch.manual_seed(43)
        model = self.make_model(True).eval()
        nn.init.normal_(model.correction[-1].weight, std=0.2)
        image = torch.randn(4, 3, 24, 36)
        captured = []
        handle = model.correction.register_forward_hook(
            lambda _module, _inputs, value: captured.append(value)
        )
        output = model(image)
        handle.remove()
        offset = torch.tanh(captured[0].squeeze(-1)).detach() * math.radians(4)
        direct = rotate_double_angle(model.coarse(image), offset)
        weight = torch.tensor([[0.3, -0.7], [0.5, 0.9], [-0.2, 0.8], [0.9, 0.4]])
        actual = torch.autograd.grad((output * weight).sum(), model.coarse.angles)[0]
        expected = torch.autograd.grad((direct * weight).sum(), model.coarse.angles)[0]
        torch.testing.assert_close(actual, expected, rtol=0, atol=0)
        self.assertGreater(actual.abs().sum().item(), 0)
        model.detach_crop_angle = False
        feedback = torch.autograd.grad((model(image) * weight).sum(), model.coarse.angles)[0]
        self.assertGreater((feedback - expected).abs().max().item(), 1e-5)

    def test_final_loss_keeps_coarse_fine_correction_and_image_gradients(self):
        for bf16 in (False, True):
            torch.manual_seed(42)
            model = self.make_model(True).train()
            optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
            image = torch.randn(4, 3, 24, 36, requires_grad=True)
            angles = model.coarse.angles.detach() + math.radians(0.5)
            targets = torch.stack((torch.sin(2 * angles), torch.cos(2 * angles)), -1)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                image.grad = None
                with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                    output = model(image)
                    loss = vector_charbonnier_angle_loss(output, targets, epsilon=0.1)
                self.assertTrue(torch.isfinite(output).all() and torch.isfinite(loss))
                loss.backward()
                if step == 1:
                    for module in (model.coarse, model.fine, model.correction):
                        gradients = [p.grad for p in module.parameters()]
                        self.assertTrue(
                            all(g is not None and torch.isfinite(g).all() for g in gradients)
                        )
                        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                    self.assertTrue(torch.isfinite(image.grad).all())
                    self.assertGreater(image.grad.abs().sum().item(), 0)
                optimizer.step()

    @patch("angle_predictor.models.line_angle.LineAngleModel", side_effect=lambda **_: Coarse())
    def test_factory_and_portable_checkpoint_preserve_routing_option(self, mock):
        params = AngleExperimentParams.model_validate(
            {
                "data": {"root": "Z:/missing-angle-data"},
                "model": {
                    "kind": "polar_refinement",
                    "coarse_model": {"pretrained": True},
                    "initialization": "imagenet",
                    "train_coarse": True,
                    "refinement_head": "local_mlp",
                    "detach_crop_angle": True,
                    "crop_size": [16, 9],
                    "fine_channels": [4, 8],
                    "refinement_hidden": 8,
                },
                "preprocessing": {"output_size": [16, 36]},
                "training": {"precision": "fp32"},
            }
        )
        model = _build_model(params, (16, 16)).eval()
        self.assertTrue(mock.call_args.kwargs["pretrained"])
        self.assertTrue(model.network.detach_crop_angle)
        images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
        expected = model(images).detach()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "routing.pt"
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
            self.assertTrue(restored.network.detach_crop_angle)
            self.assertFalse(mock.call_args.kwargs["pretrained"])
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
