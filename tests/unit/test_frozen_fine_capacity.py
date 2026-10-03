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
from tests.unit import test_polar_refinement as fixtures


class FrozenFineCapacityTests(TestCase):
    @patch(
        "angle_predictor.models.line_angle.LineAngleModel",
        side_effect=lambda **_: fixtures.ToyCoarse(),
    )
    def test_fine64_factory_updates_fine_only_and_is_portable_without_ancestors(self, _mock):
        torch.manual_seed(42)
        with TemporaryDirectory() as directory:
            original, source, ancestor = fixtures.PolarRefinementTests().make_source(directory)
            settings = original.model_dump(mode="json")
            settings["model"].update(
                crop_size=[384, 33],
                fine_channels=[16, 32, 64],
                refinement_hidden=128,
                window_deg=4.0,
            )
            params = AngleExperimentParams.model_validate(settings)
            model = _build_model(params, (16, 16)).eval()
            self.assertEqual(model.network.fine[6].out_channels, 64)
            self.assertEqual(model.network.correction[0].in_features, 2 * 64 * 33 + 2)
            images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
            torch.testing.assert_close(model(images), source(images), rtol=0, atol=0)
            before = {k: v.clone() for k, v in model.network.coarse.state_dict().items()}
            optimizer = _build_angle_optimizer(model.network, params)
            optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            self.assertFalse(optimizer_ids & {id(p) for p in model.network.coarse.parameters()})
            self.assertEqual(optimizer_ids, {id(p) for p in model.parameters() if p.requires_grad})
            criterion = build_angle_loss(**params.loss.model_dump())
            targets = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
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
            for key, value in model.network.coarse.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            model.eval()
            expected = model(images).detach()
            portable = params.model_dump(mode="json")
            portable["data"]["root"] = "Z:/missing-angle-data"
            portable["model"]["coarse_checkpoint"] = "Z:/missing-coarse.pt"
            checkpoint = Path(directory) / "fine64.pt"
            save_checkpoint(
                model,
                checkpoint,
                metadata={
                    "format_version": 1,
                    "input_size": [16, 16],
                    "params": portable,
                },
            )
            ancestor.unlink()
            restored, _, _ = load_angle_predictor(checkpoint, torch.device("cpu"))
            self.assertEqual(restored.network.fine[6].out_channels, 64)
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)

    @patch(
        "angle_predictor.models.line_angle.LineAngleModel",
        side_effect=lambda **_: fixtures.ToyCoarse(),
    )
    def test_head256_fine64_factory_updates_fine_only_and_is_portable_without_ancestors(
        self, _mock
    ):
        torch.manual_seed(42)
        with TemporaryDirectory() as directory:
            original, source, ancestor = fixtures.PolarRefinementTests().make_source(directory)
            settings = original.model_dump(mode="json")
            settings["model"].update(
                crop_size=[384, 33],
                fine_channels=[16, 32, 64],
                refinement_hidden=256,
                window_deg=4.0,
            )
            params = AngleExperimentParams.model_validate(settings)
            model = _build_model(params, (16, 16)).eval()
            self.assertEqual(model.network.fine[6].out_channels, 64)
            self.assertEqual(model.network.correction[0].in_features, 2 * 64 * 33 + 2)
            self.assertEqual(model.network.correction[0].out_features, 256)
            self.assertEqual(model.network.correction[-1].in_features, 256)
            images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
            torch.testing.assert_close(model(images), source(images), rtol=0, atol=0)
            before = {k: v.clone() for k, v in model.network.coarse.state_dict().items()}
            optimizer = _build_angle_optimizer(model.network, params)
            optimizer_ids = {id(p) for group in optimizer.param_groups for p in group["params"]}
            self.assertFalse(optimizer_ids & {id(p) for p in model.network.coarse.parameters()})
            self.assertEqual(optimizer_ids, {id(p) for p in model.parameters() if p.requires_grad})
            criterion = build_angle_loss(**params.loss.model_dump())
            targets = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
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
            for key, value in model.network.coarse.state_dict().items():
                torch.testing.assert_close(value, before[key], rtol=0, atol=0)
            model.eval()
            expected = model(images).detach()
            portable = params.model_dump(mode="json")
            portable["data"]["root"] = "Z:/missing-angle-data"
            portable["model"]["coarse_checkpoint"] = "Z:/missing-coarse.pt"
            checkpoint = Path(directory) / "fine64.pt"
            save_checkpoint(
                model,
                checkpoint,
                metadata={
                    "format_version": 1,
                    "input_size": [16, 16],
                    "params": portable,
                },
            )
            ancestor.unlink()
            restored, _, _ = load_angle_predictor(checkpoint, torch.device("cpu"))
            self.assertEqual(restored.network.fine[6].out_channels, 64)
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
