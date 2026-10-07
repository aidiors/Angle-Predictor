import hashlib
import json
from copy import deepcopy
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from pydantic import ValidationError
from torch import nn

from angle_predictor.config.angle_experiment import AngleExperimentParams, AngleModelConfig
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.experiments.angle import _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.models.feedback import StageFeedback
from angle_predictor.models.line_angle import LineAngleModel, load_initialization_weights


class FeedbackTests(TestCase):
    def test_default_and_zero_feedback_preserve_actual_convnext_and_rng(self):
        torch.set_num_threads(2)
        torch.manual_seed(42)
        original = LineAngleModel(pretrained=False).eval()
        weights = original.state_dict()
        image = torch.rand(2, 3, 64, 68)
        with torch.inference_mode():
            expected = original(image)
        for mode in ("fixed", "gated", "routed"):
            with self.subTest(mode=mode):
                torch.manual_seed(42)
                model = LineAngleModel(pretrained=False, feedback_mode=mode).eval()
                state_after_feedback_init = torch.get_rng_state()
                torch.manual_seed(42)
                LineAngleModel(pretrained=False)
                self.assertTrue(torch.equal(state_after_feedback_init, torch.get_rng_state()))
                for key, value in weights.items():
                    self.assertTrue(torch.equal(value, model.state_dict()[key]), key)
                with torch.inference_mode():
                    torch.testing.assert_close(model(image), expected, rtol=0, atol=0)
        self.assertFalse(any(key.startswith("feedback.") for key in weights))

    def test_feedback_matches_saved_odd_shape_and_has_gradients(self):
        for mode in ("fixed", "gated", "routed"):
            with self.subTest(mode=mode):
                module = StageFeedback(4, 8, mode).train()
                stage = nn.Conv2d(4, 8, 2, stride=2)
                early = torch.randn(6, 4, 8, 9, requires_grad=True)
                deep = stage(early)
                optimizer = torch.optim.AdamW(module.parameters(), lr=0.03)
                for _ in range(5):
                    optimizer.zero_grad()
                    deep = stage(early)
                    output = module(early, deep, stage)
                    output.square().mean().backward()
                    optimizer.step()
                self.assertEqual(module.update(early, deep).shape, early.shape)
                self.assertEqual(output.shape, deep.shape)
                for name, parameter in module.named_parameters():
                    self.assertIsNotNone(parameter.grad, name)
                    self.assertTrue(torch.isfinite(parameter.grad).all(), name)
                self.assertGreater(float(module.projection.weight.grad.abs().sum()), 0)
                if module.gate is not None:
                    self.assertGreater(float(module.gate[1].weight.grad.abs().sum()), 0)
                if module.router is not None:
                    self.assertGreater(float(module.router[-1].weight.grad.abs().sum()), 0)

    def test_router_skips_actual_stage_in_eval_and_keeps_augmentation_rng(self):
        module = StageFeedback(4, 8, "routed")
        stage = nn.Conv2d(4, 8, 2, stride=2)
        early = torch.randn(5, 4, 8, 8)
        deep = stage(early)
        calls = []
        handle = stage.register_forward_hook(lambda _, args, output: calls.append(len(args[0])))
        module.eval()
        with torch.inference_mode():
            self.assertTrue(torch.equal(module(early, deep, stage), deep))
        self.assertEqual(calls, [])
        with torch.no_grad():
            module.router[-1].bias[1] = 1
            module.projection.weight.fill_(0.1)
        with torch.inference_mode():
            repeated = module(early, deep, stage)
        self.assertEqual(calls, [5])
        self.assertFalse(torch.equal(repeated, deep))
        module.train()
        before = torch.get_rng_state()
        module(early, deep, stage)
        self.assertTrue(torch.equal(torch.get_rng_state(), before))
        handle.remove()

    def test_feedback_transfer_is_strict_and_completed_model_is_portable(self):
        torch.set_num_threads(2)
        with TemporaryDirectory() as directory:
            root = Path(directory)
            meta = root / "meta.json"
            meta.write_text(json.dumps({"images_shape": [2, 64, 64, 3]}))
            source_params = AngleExperimentParams.model_validate(
                {
                    "data": {"root": str(root)},
                    "model": {"pretrained": False},
                    "preprocessing": {"output_size": [64, 68]},
                    "training": {"precision": "fp32"},
                }
            )
            source = _build_model(source_params, (64, 64)).eval()
            path = root / "coarse.pt"
            save_checkpoint(
                source,
                path,
                metadata={
                    "format_version": 1,
                    "run_id": "1" * 32,
                    "input_size": [64, 64],
                    "params": source_params.model_dump(mode="json"),
                    "dataset_meta_sha256": hashlib.sha256(meta.read_bytes()).hexdigest(),
                },
            )
            options = source_params.model_dump(mode="json")
            options["model"]["feedback_mode"] = "gated"
            options["training"].update(
                initialization_mode="feedback_base",
                initialization_checkpoint=str(path),
                initialization_run_id="1" * 32,
            )
            params = AngleExperimentParams.model_validate(options)
            model = _build_model(params, (64, 64)).eval()
            image = torch.randint(0, 256, (1, 3, 64, 64), dtype=torch.uint8)
            with torch.inference_mode():
                expected = model(image)
                torch.testing.assert_close(expected, source(image), rtol=0, atol=0)
            broken = deepcopy(source.state_dict())
            broken.pop(next(iter(broken)))
            with self.assertRaisesRegex(ValueError, "every original"):
                load_initialization_weights(model, broken, params)
            portable_params = params.model_dump(mode="json")
            portable_params["data"]["root"] = "missing-data"
            portable_params["training"]["initialization_checkpoint"] = "missing-source.pt"
            portable = root / "portable.pt"
            save_checkpoint(
                model,
                portable,
                metadata={
                    "format_version": 1,
                    "input_size": [64, 64],
                    "params": portable_params,
                },
            )
            path.unlink()
            meta.unlink()
            restored, _, _ = load_angle_predictor(portable, torch.device("cpu"))
            with torch.inference_mode():
                torch.testing.assert_close(restored(image), expected, rtol=0, atol=0)

    def test_config_rejects_wrong_backbone_or_partial_hierarchy(self):
        for changes in (
            {"backbone_name": "efficientvit_b2.r224_in1k"},
            {"out_index": 2},
            {"final_stage_blocks": 2},
            {"penultimate_stage_blocks": 8},
        ):
            with self.subTest(changes=changes), self.assertRaises(ValidationError):
                AngleModelConfig(feedback_mode="fixed", **changes)
