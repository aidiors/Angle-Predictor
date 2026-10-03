import math
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch

from angle_predictor.config.angle_experiment import AngleExperimentParams
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.experiments.angle import _build_angle_optimizer, _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.losses.angle import build_angle_loss
from angle_predictor.models.polar_refinement import crop_signed_polar
from tests.unit import test_polar_refinement as fixtures


class FrozenRadialDetailTests(TestCase):
    def test_full_radial_crop_preserves_alternating_rows_lost_by_downsampling(self):
        rows = (torch.arange(384) % 2).float()
        image = rows.view(1, 1, 384, 1).expand(1, 3, 384, 360)
        angle = torch.tensor([math.pi / 2])
        full = crop_signed_polar(image, angle, crop_size=(384, 33), window_deg=4)
        half = crop_signed_polar(image, angle, crop_size=(192, 33), window_deg=4)
        torch.testing.assert_close(full[..., 16], image[..., 180], atol=4e-5, rtol=0)
        self.assertGreater(((half > 0.05) & (half < 0.95)).float().mean().item(), 0.8)

    @patch(
        "angle_predictor.models.line_angle.LineAngleModel",
        side_effect=lambda **_: fixtures.ToyCoarse(),
    )
    def test_fresh_radial_fine_trains_with_frozen_coarse_and_portable_checkpoint(self, _mock):
        torch.manual_seed(42)
        with TemporaryDirectory() as directory:
            original, source, ancestor = fixtures.PolarRefinementTests().make_source(directory)
            settings = original.model_dump(mode="json")
            settings["model"]["crop_size"] = [384, 33]
            params = AngleExperimentParams.model_validate(settings)
            model = _build_model(params, (16, 16)).eval()
            images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
            torch.testing.assert_close(model(images), source(images), rtol=0, atol=0)
            before = {k: v.clone() for k, v in model.network.coarse.state_dict().items()}
            optimizer = _build_angle_optimizer(model.network, params)
            targets = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
            criterion = build_angle_loss(**params.loss.model_dump())
            model.train()
            self.assertFalse(model.network.coarse.training)
            for step in range(2):
                optimizer.zero_grad(set_to_none=True)
                prediction, augmented = model.forward_augmented(images, targets)
                loss = criterion(prediction, augmented)
                self.assertTrue(torch.isfinite(prediction).all() and torch.isfinite(loss))
                loss.backward()
                self.assertTrue(all(p.grad is None for p in model.network.coarse.parameters()))
                if step == 1:
                    for module in (model.network.fine, model.network.correction):
                        gradients = [p.grad for p in module.parameters()]
                        self.assertTrue(
                            all(g is not None and torch.isfinite(g).all() for g in gradients)
                        )
                        self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                optimizer.step()
            for k, v in model.network.coarse.state_dict().items():
                torch.testing.assert_close(v, before[k], rtol=0, atol=0)
            model.eval()
            expected = model(images).detach()
            checkpoint = Path(directory) / "radial384.pt"
            portable = params.model_dump(mode="json")
            portable["data"]["root"] = "Z:/missing-angle-data"
            save_checkpoint(
                model,
                checkpoint,
                metadata={"format_version": 1, "input_size": [16, 16], "params": portable},
            )
            ancestor.unlink()
            restored, _, _ = load_angle_predictor(checkpoint, torch.device("cpu"))
            self.assertEqual(restored.network.crop_size, (384, 33))
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
