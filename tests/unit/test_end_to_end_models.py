import math
from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
from unittest import TestCase
from unittest.mock import patch

import torch
from torch import nn
from torch.nn import functional as F

from angle_predictor.config.angle_experiment import (
    AngleExperimentParams,
    AngleModelConfig,
    PolarRefinementModelConfig,
    SharedPolarModelConfig,
)
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.experiments.angle import _build_angle_optimizer, _build_model, _scheduler
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.models.line_angle import build_angle_network
from angle_predictor.models.polar_refinement import PolarRefinementModel, crop_signed_polar


class Coarse(nn.Module):
    def __init__(self):
        super().__init__()
        self.projection = nn.Linear(3, 2)

    def forward(self, image):
        return F.normalize(self.projection(image.mean((-2, -1))), dim=-1)


class Backbone(nn.Module):
    def __init__(self):
        super().__init__()
        self.early = nn.Conv2d(3, 4, 3, padding=1)
        self.deep = nn.Conv2d(4, 8, 3, padding=1)
        self.feature_info = SimpleNamespace(channels=lambda: [4, 8])

    def forward(self, image):
        early = self.early(image)
        return [early, self.deep(F.avg_pool2d(early, 4))]


class EndToEndTests(TestCase):
    @patch("angle_predictor.models.line_angle.LineAngleModel", side_effect=lambda **_: Coarse())
    def test_imagenet_refinement_uses_no_task_weights_and_has_portable_inference(self, mock):
        config = PolarRefinementModelConfig(
            coarse_model=AngleModelConfig(pretrained=True),
            initialization="imagenet",
            train_coarse=True,
            crop_size=(16, 9),
            fine_channels=(4, 8),
            refinement_hidden=8,
        )
        with patch("torch.load", side_effect=AssertionError("Task weights must not be read")):
            model = build_angle_network(config)
            self.assertTrue(mock.call_args.kwargs["pretrained"])
            restored = build_angle_network(config, load_pretrained=False)
            self.assertFalse(mock.call_args.kwargs["pretrained"])
        restored.load_state_dict(model.state_dict(), strict=True)
        image = torch.randn(2, 3, 16, 36)
        torch.testing.assert_close(model.eval()(image), restored.eval()(image), rtol=0, atol=0)
        self.assertTrue(all(p.requires_grad for p in model.parameters()))
        model(image)[:, 0].sum().backward()
        self.assertGreater(model.coarse.projection.weight.grad.abs().sum().item(), 0)

    def test_imagenet_refinement_rejects_frozen_random_coarse_and_task_lineage(self):
        from pydantic import ValidationError

        options = {
            "coarse_model": AngleModelConfig().model_dump(),
            "initialization": "imagenet",
            "train_coarse": True,
        }
        for override in (
            {"train_coarse": False},
            {"coarse_run_id": "1" * 32},
            {"coarse_model": AngleModelConfig(pretrained=False).model_dump()},
        ):
            with self.assertRaises(ValidationError):
                PolarRefinementModelConfig(**(options | override))
        with self.assertRaises(ValidationError):
            PolarRefinementModelConfig(coarse_model=AngleModelConfig())

    @patch("angle_predictor.models.line_angle.LineAngleModel", side_effect=lambda **_: Coarse())
    def test_joint_warm_start_is_exact_and_inference_needs_no_source_checkpoint(self, _mock):
        from tests.unit.test_polar_refinement import PolarRefinementTests

        with TemporaryDirectory() as directory:
            fixture = PolarRefinementTests()
            params, _, coarse_checkpoint = fixture.make_source(directory)
            # The fixture's original toy network has BN/dropout; replace its coarse weights.
            coarse_payload = torch.load(coarse_checkpoint, weights_only=True)
            coarse_payload["model_state_dict"] = {
                f"network.{name}": value for name, value in Coarse().state_dict().items()
            }
            torch.save(coarse_payload, coarse_checkpoint)
            original = _build_model(params, (16, 16)).eval()
            nn.init.normal_(original.network.correction[-1].weight, std=0.01)
            source_path = Path(directory) / "fine.pt"
            metadata = {
                **coarse_payload["metadata"],
                "run_id": "2" * 32,
                "params": params.model_dump(mode="json"),
            }
            save_checkpoint(original, source_path, metadata=metadata)
            settings = params.model_dump(mode="json")
            settings["model"]["train_coarse"] = True
            settings["training"].update(
                initialization_checkpoint=str(source_path),
                initialization_run_id="2" * 32,
                coarse_lr_scale=0.1,
            )
            joint_params = AngleExperimentParams.model_validate(settings)
            joint = _build_model(joint_params, (16, 16)).eval()
            images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
            expected = original(images).detach()
            torch.testing.assert_close(joint(images), expected, rtol=0, atol=0)
            portable = Path(directory) / "joint.pt"
            save_checkpoint(
                joint,
                portable,
                metadata={
                    **metadata,
                    "run_id": "3" * 32,
                    "params": joint_params.model_dump(mode="json"),
                },
            )
            source_path.unlink()
            coarse_checkpoint.unlink()
            restored, _, _ = load_angle_predictor(portable, torch.device("cpu"))
            torch.testing.assert_close(restored(images), expected, rtol=0, atol=0)
            self.assertTrue(all(p.requires_grad for p in restored.network.coarse.parameters()))

    def test_joint_final_loss_updates_both_branches(self):
        torch.manual_seed(42)
        model = PolarRefinementModel(
            Coarse(),
            crop_size=(16, 9),
            fine_channels=(4, 8),
            refinement_hidden=8,
            train_coarse=True,
        ).train()
        nn.init.normal_(model.correction[-1].weight, std=0.02)
        model(torch.randn(3, 3, 24, 36))[:, 0].sum().backward()
        self.assertTrue(model.coarse.training)
        for module in (model.coarse, model.fine, model.correction):
            self.assertGreater(sum(p.grad.abs().sum().item() for p in module.parameters()), 0)
        with torch.inference_mode():
            self.assertFalse(model(torch.randn(3, 3, 24, 36)).requires_grad)

    def test_crop_gradient_crosses_angular_wrap(self):
        image = torch.randn(4, 3, 16, 36)
        angles = torch.tensor([0.0001, math.pi - 0.0001, 0.7, 2.7], requires_grad=True)
        crop = crop_signed_polar(image, angles, crop_size=(16, 9), window_deg=4)
        crop.square().sum().backward()
        self.assertTrue(torch.isfinite(angles.grad).all())
        self.assertTrue((angles.grad.abs() > 0).all())

    def test_joint_optimizer_owns_all_weights_with_scaled_coarse_lr(self):
        config = PolarRefinementModelConfig(
            coarse_model=AngleModelConfig(pretrained=False),
            coarse_checkpoint="unused.pt",
            coarse_run_id="1" * 32,
            coarse_split_seed=42,
            train_coarse=True,
            crop_size=(16, 9),
            fine_channels=(4, 8),
            refinement_hidden=8,
        )
        params = AngleExperimentParams.model_validate(
            {
                "data": {"root": "."},
                "model": config.model_dump(),
                "optimizer": {"kind": "musgd", "lr": 0.001},
                "training": {"coarse_lr_scale": 0.1},
            }
        )
        model = PolarRefinementModel(
            Coarse(),
            crop_size=(16, 9),
            fine_channels=(4, 8),
            refinement_hidden=8,
            train_coarse=True,
        )
        optimizer = _build_angle_optimizer(model, params)
        coarse_ids = {id(p) for p in model.coarse.parameters()}
        seen = []
        for group in optimizer.param_groups:
            for parameter in group["params"]:
                seen.append(id(parameter))
                self.assertAlmostEqual(
                    group["lr"], 0.0001 if id(parameter) in coarse_ids else 0.001
                )
        self.assertEqual(len(seen), len(set(seen)))
        self.assertEqual(set(seen), {id(p) for p in model.parameters()})
        model(torch.randn(3, 3, 24, 36))[:, 0].sum().backward()
        optimizer.step()

    def test_scaled_lr_reaches_both_chained_optimizers_and_scheduler(self):
        params = AngleExperimentParams.model_validate(
            {
                "data": {"root": "."},
                "model": {
                    "kind": "polar_refinement",
                    "coarse_model": {"pretrained": False},
                    "coarse_checkpoint": "unused.pt",
                    "coarse_run_id": "1" * 32,
                    "coarse_split_seed": 42,
                    "train_coarse": True,
                },
                "optimizer": {"kind": "muon_adamw", "muon_lr": 0.001, "adamw_lr": 0.001},
                "training": {"coarse_lr_scale": 0.1, "warmup_steps": 0},
            }
        )
        model = PolarRefinementModel(
            Coarse(),
            crop_size=(16, 9),
            fine_channels=(4, 8),
            refinement_hidden=8,
            train_coarse=True,
        )
        optimizer = _build_angle_optimizer(model, params)
        coarse_ids = {id(p) for p in model.coarse.parameters()}
        actual_groups = [g for part in optimizer.optimizers for g in part.param_groups]
        self.assertEqual({id(g) for g in actual_groups}, {id(g) for g in optimizer.param_groups})
        for group in actual_groups:
            for parameter in group["params"]:
                self.assertAlmostEqual(
                    group["lr"], 0.0001 if id(parameter) in coarse_ids else 0.001
                )
        scheduler = _scheduler(optimizer, params)
        model(torch.randn(3, 3, 24, 36))[:, 0].sum().backward()
        optimizer.step()
        scheduler.step()
        self.assertEqual([g["lr"] for g in actual_groups], scheduler.get_last_lr())

    @patch(
        "angle_predictor.models.line_angle.timm.create_model",
        side_effect=lambda *_, **__: Backbone(),
    )
    def test_shared_model_uses_one_backbone_with_both_stage_gradients(self, mock):
        model = build_angle_network(
            SharedPolarModelConfig(pretrained=False, fusion_channels=4, head_hidden=8),
            load_pretrained=False,
        )
        output = model(torch.randn(3, 3, 16, 36))
        torch.testing.assert_close(output.norm(dim=-1), torch.ones(3))
        output[:, 0].sum().backward()
        for module in (model.backbone.early, model.backbone.deep, model.detail, model.context):
            self.assertGreater(module.weight.grad.abs().sum().item(), 0)
        self.assertEqual(mock.call_count, 1)
        self.assertEqual(mock.call_args.kwargs["out_indices"], (0, 3))
