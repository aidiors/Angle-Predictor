from unittest import TestCase

import torch
from pydantic import ValidationError

from angle_predictor.config.angle_experiment import AngleModelConfig
from angle_predictor.models.line_angle import build_angle_network


class StagePruningTests(TestCase):
    def test_pruning_preserves_existing_weights_and_output_interface(self) -> None:
        torch.manual_seed(42)
        full = build_angle_network(AngleModelConfig(pretrained=False))
        torch.manual_seed(42)
        pruned = build_angle_network(AngleModelConfig(pretrained=False, final_stage_blocks=2))
        full_weights = full.state_dict()
        for name, weight in pruned.state_dict().items():
            torch.testing.assert_close(weight, full_weights[name], rtol=0, atol=0)
        self.assertLess(
            sum(p.numel() for p in pruned.parameters()), sum(p.numel() for p in full.parameters())
        )
        with torch.inference_mode():
            prediction = pruned.eval()(torch.zeros(1, 3, 64, 64))
        self.assertEqual(prediction.shape, (1, 2))
        self.assertTrue(torch.isfinite(prediction).all())

    def test_pruning_config_rejects_unsupported_stages_and_counts(self) -> None:
        for options in (
            {"final_stage_blocks": 0},
            {"final_stage_blocks": 4},
            {"final_stage_blocks": 2, "out_index": 2},
            {"final_stage_blocks": 2, "backbone_name": "resnet18"},
            {"penultimate_stage_blocks": 0},
            {"penultimate_stage_blocks": 10},
            {"penultimate_stage_blocks": 6, "out_index": 1},
        ):
            with self.subTest(options=options), self.assertRaises(ValidationError):
                AngleModelConfig(**options)
