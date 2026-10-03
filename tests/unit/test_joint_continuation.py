from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch
from torch import nn

from angle_predictor.config.angle_experiment import AngleExperimentParams
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.experiments.angle import _build_angle_optimizer, _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.losses.angle import build_angle_loss
from tests.unit import test_polar_refinement as fixtures


class JointContinuationTests(TestCase):
    @patch(
        "angle_predictor.models.line_angle.LineAngleModel",
        side_effect=lambda **_: fixtures.ToyCoarse(),
    )
    def test_radial_warmstart_unfreezes_coarse_with_scaled_lr_and_portable_inference(self, _mock):
        torch.manual_seed(42)
        with TemporaryDirectory() as directory:
            params, _, ancestor = fixtures.PolarRefinementTests().make_source(directory)
            settings = params.model_dump(mode="json")
            settings["model"]["crop_size"] = [384, 33]
            frozen_params = AngleExperimentParams.model_validate(settings)
            source = _build_model(frozen_params, (16, 16)).eval()
            nn.init.normal_(source.network.correction[-1].weight, std=0.01)
            full = Path(directory) / "frozen.pt"
            metadata = {
                **torch.load(ancestor, weights_only=True)["metadata"],
                "run_id": "2" * 32,
                "params": frozen_params.model_dump(mode="json"),
            }
            save_checkpoint(source, full, metadata=metadata)
            settings["model"]["train_coarse"] = True
            settings["training"].update(
                initialization_checkpoint=str(full),
                initialization_run_id="2" * 32,
                coarse_lr_scale=0.1,
            )
            target = AngleExperimentParams.model_validate(settings)
            model = _build_model(target, (16, 16)).eval()
            images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
            torch.testing.assert_close(model(images), source(images), rtol=0, atol=0)
            before = {k: v.clone() for k, v in model.network.coarse.state_dict().items()}
            optimizer = _build_angle_optimizer(model.network, target)
            coarse_ids = {id(p) for p in model.network.coarse.parameters()}
            coarse_members = set()
            for group in optimizer.param_groups:
                owned = {id(p) for p in group["params"]}
                is_coarse = bool(owned & coarse_ids)
                self.assertFalse(is_coarse and bool(owned - coarse_ids))
                self.assertAlmostEqual(group["lr"], target.optimizer.lr * (0.1 if is_coarse else 1))
                coarse_members |= owned & coarse_ids
            self.assertEqual(coarse_members, coarse_ids)
            model.train()
            self.assertTrue(model.network.coarse.training)
            criterion = build_angle_loss(**target.loss.model_dump())
            targets = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                prediction, augmented = model.forward_augmented(images, targets)
                loss = criterion(prediction, augmented)
                self.assertTrue(torch.isfinite(prediction).all() and torch.isfinite(loss))
                loss.backward()
                for branch in (model.network.coarse, model.network.fine, model.network.correction):
                    gradients = [p.grad for p in branch.parameters()]
                    self.assertTrue(
                        all(g is not None and torch.isfinite(g).all() for g in gradients)
                    )
                    self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                optimizer.step()
            self.assertTrue(
                any(
                    not torch.equal(v, before[k])
                    for k, v in model.network.coarse.state_dict().items()
                )
            )
            model.eval()
            expected = model(images).detach()
            portable = Path(directory) / "joint.pt"
            portable_params = target.model_dump(mode="json")
            portable_params["data"]["root"] = "Z:/missing-angle-data"
            save_checkpoint(model, portable, metadata={**metadata, "params": portable_params})
            ancestor.unlink()
            full.unlink()
            restored, _, _ = load_angle_predictor(portable, torch.device("cpu"))
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
