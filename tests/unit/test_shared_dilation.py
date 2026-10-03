from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase
from unittest.mock import patch

import torch
from pydantic import ValidationError
from torch import nn

from angle_predictor.config.angle_experiment import AngleExperimentParams, SharedPolarModelConfig
from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.experiments.angle import _build_model
from angle_predictor.inference.angle import load_angle_predictor
from angle_predictor.models.line_angle import PolarAngleRegressionHead, build_angle_network
from tests.unit.test_end_to_end_models import Backbone


class SharedDilationTests(TestCase):
    def test_config_rejects_nonpositive_or_wrong_count(self):
        for values in ((0, 2, 4), (1, -2, 4), (1, 2), (1, 2, 4, 8)):
            with self.assertRaises(ValidationError):
                SharedPolarModelConfig(angular_dilations=values)
        self.assertEqual(SharedPolarModelConfig().angular_dilations, (1, 1, 1))
        with self.assertRaises(ValueError):
            PolarAngleRegressionHead(4, angular_dilations=(1, 0, 4))

    def test_default_and_explicit_default_preserve_state_layout_and_predictions(self):
        original = PolarAngleRegressionHead(4, hidden=8, dropout=0).eval()
        explicit = PolarAngleRegressionHead(4, hidden=8, dropout=0, angular_dilations=(1, 1, 1))
        dilated = PolarAngleRegressionHead(4, hidden=8, angular_dilations=(1, 2, 4))
        explicit.load_state_dict(original.state_dict(), strict=True)
        dilated.load_state_dict(original.state_dict(), strict=True)
        self.assertEqual(
            sum(p.numel() for p in original.parameters()),
            sum(p.numel() for p in dilated.parameters()),
        )
        image = torch.randn(2, 4, 12, 90)
        torch.testing.assert_close(original(image), explicit.eval()(image), rtol=0, atol=0)

    def test_impulse_support_expands_to_21_columns_with_circular_wrap(self):
        for dilations, radius in (((1, 1, 1), 5), ((1, 2, 4), 10)):
            head = PolarAngleRegressionHead(1, hidden=1, angular_dilations=dilations)
            for layer in head.angle_conv:
                if isinstance(layer, nn.Conv1d):
                    nn.init.ones_(layer.weight)
                    nn.init.zeros_(layer.bias)
            for center in (0, 45):
                profile = torch.zeros(1, 5, 90)
                profile[0, 0, center] = 1
                result = head.angle_conv(profile)
                self.assertEqual(result.shape, (1, 1, 90))
                support = (result[0, 0] > 0).nonzero().flatten().tolist()
                expected = sorted((center + offset) % 90 for offset in range(-radius, radius + 1))
                self.assertEqual(support, expected)

    @patch(
        "angle_predictor.models.line_angle.timm.create_model",
        side_effect=lambda *_, **__: Backbone(),
    )
    def test_shared_dilation_gradients_and_official_portable_loader(self, mock):
        config = SharedPolarModelConfig(
            pretrained=True, fusion_channels=4, head_hidden=8, angular_dilations=(1, 2, 4)
        )
        params = AngleExperimentParams.model_validate(
            {
                "data": {"root": "Z:/missing-angle-data"},
                "model": config.model_dump(),
                "preprocessing": {"output_size": [16, 36]},
                "training": {"precision": "fp32"},
            }
        )
        network = build_angle_network(config)
        output = network(torch.randn(3, 3, 16, 36))
        torch.testing.assert_close(output.norm(dim=-1), torch.ones(3))
        output[:, 0].sum().backward()
        for module in (
            network.backbone.early,
            network.backbone.deep,
            network.detail,
            network.context,
            network.head,
        ):
            gradients = [p.grad for p in module.parameters()]
            self.assertTrue(all(g is not None and torch.isfinite(g).all() for g in gradients))
            self.assertGreater(sum(g.abs().sum().item() for g in gradients), 0)
        model = _build_model(params, (16, 16)).eval()
        images = torch.randint(0, 256, (2, 3, 16, 16), dtype=torch.uint8)
        expected = model(images)
        with TemporaryDirectory() as directory:
            path = Path(directory) / "shared.pt"
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
