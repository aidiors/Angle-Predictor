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
from tests.unit import test_polar_refinement as refinement_fixtures
from tests.unit.test_end_to_end_models import Coarse


class FrozenContinuationTests(TestCase):
    def fixture(self, directory):
        params, _, coarse_path = refinement_fixtures.PolarRefinementTests().make_source(directory)
        payload = torch.load(coarse_path, weights_only=True)
        payload["model_state_dict"] = {
            f"network.{name}": value for name, value in Coarse().state_dict().items()
        }
        torch.save(payload, coarse_path)
        original = _build_model(params, (16, 16)).eval()
        nn.init.normal_(original.network.correction[-1].weight, std=0.01)
        fine_path = Path(directory) / "fine.pt"
        metadata = {
            **payload["metadata"],
            "run_id": "2" * 32,
            "params": params.model_dump(mode="json"),
        }
        save_checkpoint(original, fine_path, metadata=metadata)
        settings = params.model_dump(mode="json")
        settings["training"].update(
            initialization_checkpoint=str(fine_path), initialization_run_id="2" * 32
        )
        target = AngleExperimentParams.model_validate(settings)
        return original, target, metadata, fine_path, coarse_path

    @patch("angle_predictor.models.line_angle.LineAngleModel", side_effect=lambda **_: Coarse())
    def test_actual_warmstart_updates_fine_without_coarse_gradients_or_weights(self, _mock):
        torch.manual_seed(42)
        with TemporaryDirectory() as directory:
            original, params, _, _, _ = self.fixture(directory)
            model = _build_model(params, (16, 16)).eval()
            images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
            torch.testing.assert_close(model(images), original(images), rtol=0, atol=0)
            before = {n: v.clone() for n, v in model.network.coarse.state_dict().items()}
            model.train()
            self.assertFalse(model.network.coarse.training)
            optimizer = _build_angle_optimizer(model.network, params)
            coarse_ids = {id(p) for p in model.network.coarse.parameters()}
            owned = {id(p) for group in optimizer.param_groups for p in group["params"]}
            self.assertFalse(coarse_ids & owned)
            criterion = build_angle_loss(**params.loss.model_dump())
            targets = torch.nn.functional.normalize(torch.randn(2, 2), dim=-1)
            for _ in range(2):
                optimizer.zero_grad(set_to_none=True)
                output, augmented = model.forward_augmented(images, targets)
                loss = criterion(output, augmented)
                self.assertTrue(torch.isfinite(loss) and torch.isfinite(output).all())
                loss.backward()
                self.assertTrue(all(p.grad is None for p in model.network.coarse.parameters()))
                for module in (model.network.fine, model.network.correction):
                    gradients = [p.grad for p in module.parameters()]
                    self.assertTrue(
                        all(g is not None and torch.isfinite(g).all() for g in gradients)
                    )
                    self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
                optimizer.step()
            for n, v in model.network.coarse.state_dict().items():
                torch.testing.assert_close(v, before[n], rtol=0, atol=0)

    @patch("angle_predictor.models.line_angle.LineAngleModel", side_effect=lambda **_: Coarse())
    def test_frozen_continuation_checkpoint_is_portable_without_ancestor_files(self, _mock):
        with TemporaryDirectory() as directory:
            _, params, metadata, fine_path, coarse_path = self.fixture(directory)
            model = _build_model(params, (16, 16)).eval()
            images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
            expected = model(images).detach()
            portable_params = params.model_dump(mode="json")
            portable_params["data"]["root"] = "Z:/missing-angle-data"
            portable = Path(directory) / "continued.pt"
            save_checkpoint(
                model,
                portable,
                metadata={**metadata, "run_id": "3" * 32, "params": portable_params},
            )
            fine_path.unlink()
            coarse_path.unlink()
            restored, _, _ = load_angle_predictor(portable, torch.device("cpu"))
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
            self.assertFalse(restored.network.train_coarse)
