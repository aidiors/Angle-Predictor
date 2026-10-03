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
from angle_predictor.models.polar_refinement import PolarRefinementModel
from tests.unit.test_end_to_end_models import Coarse
from tests.unit.test_refinement_heatmap import AngleCoarse


class LocalMLPTests(TestCase):
    def make_model(self):
        return PolarRefinementModel(
            AngleCoarse(),
            crop_size=(16, 9),
            fine_channels=(4, 8),
            refinement_hidden=8,
            refinement_head="local_mlp",
            train_coarse=True,
            window_deg=4,
        )

    def test_zero_initialized_local_correction_preserves_coarse_at_the_seam(self):
        model = self.make_model().eval()
        image = torch.randn(4, 3, 24, 36)
        expected = model.coarse(image)
        for bf16 in (False, True):
            with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                actual = model(image)
            self.assertEqual(actual.dtype, torch.float32)
            torch.testing.assert_close(actual, expected, rtol=0, atol=0)
            torch.testing.assert_close(actual.norm(dim=-1), torch.ones(4))

    def test_same_local_crop_has_same_correction_input_despite_different_global_angles(self):
        model = self.make_model().eval()
        nn.init.normal_(model.correction[-1].weight, std=0.02)
        captured = []
        handle = model.correction[0].register_forward_pre_hook(
            lambda _module, inputs: captured.append(inputs[0].detach().clone())
        )
        fixed_crop = torch.randn(4, 3, 16, 9)
        image = torch.randn(4, 3, 24, 36)
        with patch(
            "angle_predictor.models.polar_refinement.crop_signed_polar", return_value=fixed_crop
        ):
            first = model(image).detach()
            with torch.no_grad():
                model.coarse.angles.add_(0.4)
            second = model(image).detach()
        handle.remove()
        self.assertEqual(captured[0].shape, (4, 144))
        torch.testing.assert_close(captured[0], captured[1], rtol=0, atol=0)
        self.assertFalse(torch.equal(first, second))

    def test_actual_final_loss_reaches_all_branches_after_zero_init_first_update(self):
        for bf16 in (False, True):
            with self.subTest(bf16=bf16):
                torch.manual_seed(42)
                model = self.make_model().train()
                optimizer = torch.optim.SGD(model.parameters(), lr=0.01)
                image = torch.randn(4, 3, 24, 36)
                target_angles = model.coarse.angles.detach() + math.radians(0.5)
                targets = torch.stack(
                    (torch.sin(2 * target_angles), torch.cos(2 * target_angles)), -1
                )
                for step in range(2):
                    optimizer.zero_grad(set_to_none=True)
                    with torch.autocast("cpu", dtype=torch.bfloat16, enabled=bf16):
                        output = model(image)
                        loss = vector_charbonnier_angle_loss(output, targets, epsilon=0.1)
                    self.assertTrue(torch.isfinite(output).all())
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    if step == 1:
                        for module in (model.coarse, model.fine, model.correction):
                            gradients = [parameter.grad for parameter in module.parameters()]
                            self.assertTrue(
                                all(g is not None and torch.isfinite(g).all() for g in gradients)
                            )
                            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                    optimizer.step()

    @patch("angle_predictor.models.line_angle.LineAngleModel", side_effect=lambda **_: Coarse())
    def test_local_head_uses_imagenet_factory_and_portable_checkpoint_without_data(self, mock):
        params = AngleExperimentParams.model_validate(
            {
                "data": {"root": "Z:/missing-angle-data"},
                "model": {
                    "kind": "polar_refinement",
                    "coarse_model": {"pretrained": True},
                    "initialization": "imagenet",
                    "train_coarse": True,
                    "refinement_head": "local_mlp",
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
        nn.init.normal_(model.network.correction[-1].weight, std=0.02)
        images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
        expected = model(images).detach()
        with TemporaryDirectory() as directory:
            path = Path(directory) / "local.pt"
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
            self.assertEqual(restored.network.refinement_head, "local_mlp")
