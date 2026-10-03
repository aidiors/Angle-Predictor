from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import torch
from torch import nn, optim

from angle_predictor.engine.checkpoint import save_checkpoint
from angle_predictor.engine.optimizers import (
    AdamWConfig,
    MuonAdamWConfig,
    UltralyticsMuSGDConfig,
    build_optimizer,
)
from angle_predictor.engine.optimizers._muon import MuSGD, muon_update
from angle_predictor.engine.optimizers.factory import ChainedOptimizer, MuonOptimizer


class TinyModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.conv = nn.Conv2d(3, 4, 3)
        self.norm = nn.BatchNorm2d(4)
        self.head = nn.Linear(4, 2)

    def forward(self, image: torch.Tensor) -> torch.Tensor:
        features = self.norm(self.conv(image)).mean(dim=(-2, -1))
        return self.head(features)


def _parameter_groups(optimizer: optim.Optimizer) -> dict[int, dict]:
    return {id(param): group for group in optimizer.param_groups for param in group["params"]}


class OptimizerTests(TestCase):
    def test_adamw_decay_groups_cover_every_trainable_parameter_once(self) -> None:
        model = TinyModel()
        optimizer = build_optimizer(model, AdamWConfig())
        self.assertIsInstance(optimizer, optim.AdamW)
        groups = _parameter_groups(optimizer)
        self.assertEqual(set(groups), {id(param) for param in model.parameters()})
        self.assertEqual(groups[id(model.conv.weight)]["weight_decay"], 0.01)
        self.assertEqual(groups[id(model.head.weight)]["weight_decay"], 0.01)
        self.assertEqual(groups[id(model.norm.weight)]["weight_decay"], 0.0)
        self.assertEqual(groups[id(model.head.bias)]["weight_decay"], 0.0)

    def test_muon_adamw_partitions_parameters_and_resumes(self) -> None:
        model = TinyModel()
        optimizer = build_optimizer(model, MuonAdamWConfig())
        self.assertIsInstance(optimizer, ChainedOptimizer)
        muon, adamw = optimizer.optimizers
        self.assertIsInstance(muon, MuonOptimizer)
        self.assertIsInstance(adamw, optim.AdamW)
        muon_ids = set(_parameter_groups(muon))
        adamw_ids = set(_parameter_groups(adamw))
        self.assertEqual(muon_ids, {id(model.conv.weight), id(model.head.weight)})
        self.assertFalse(muon_ids & adamw_ids)
        self.assertEqual(muon_ids | adamw_ids, {id(param) for param in model.parameters()})

        before = model.head.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(2, 3, 8, 8)).square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.head.weight))

        restored_model = TinyModel()
        restored_optimizer = build_optimizer(restored_model, MuonAdamWConfig())
        restored_optimizer.load_state_dict(optimizer.state_dict())
        self.assertEqual(len(restored_optimizer.optimizers[0].state), len(muon.state))
        self.assertEqual(len(restored_optimizer.optimizers[1].state), len(adamw.state))
        self.assertIs(
            restored_optimizer.param_groups[0], restored_optimizer.optimizers[0].param_groups[0]
        )

        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.5)
        before_lrs = [group["lr"] for group in optimizer.param_groups]
        optimizer.step()
        scheduler.step()
        after_lrs = [group["lr"] for group in optimizer.param_groups]
        for before_lr, after_lr in zip(before_lrs, after_lrs, strict=True):
            self.assertAlmostEqual(after_lr, before_lr * 0.5)

        with TemporaryDirectory() as temp_dir:
            path = Path(temp_dir) / "checkpoint.pt"
            save_checkpoint(model, path, optimizer=optimizer)
            payload = torch.load(path, weights_only=False)
        self.assertEqual(len(payload["optimizer_state_dict"]["optimizers"]), 2)

    def test_ultralytics_musgd_uses_local_class_and_routing(self) -> None:
        model = TinyModel()
        config = UltralyticsMuSGDConfig(boosted_parameter_names=frozenset({"head.weight"}))
        optimizer = build_optimizer(model, config)
        self.assertIsInstance(optimizer, MuSGD)
        self.assertEqual(optimizer.muon, 0.2)
        self.assertEqual(optimizer.sgd, 1.0)
        groups = _parameter_groups(optimizer)
        self.assertEqual(set(groups), {id(param) for param in model.parameters()})
        self.assertTrue(groups[id(model.conv.weight)]["use_muon"])
        self.assertTrue(groups[id(model.head.weight)]["use_muon"])
        self.assertFalse(groups[id(model.norm.weight)]["use_muon"])
        self.assertEqual(groups[id(model.norm.weight)]["weight_decay"], 0.0)
        self.assertEqual(groups[id(model.head.bias)]["weight_decay"], 0.0)
        self.assertAlmostEqual(groups[id(model.head.weight)]["lr"], 3 * config.lr)

        before = model.conv.weight.detach().clone()
        optimizer.zero_grad(set_to_none=True)
        model(torch.randn(2, 3, 8, 8)).square().mean().backward()
        optimizer.step()
        self.assertFalse(torch.equal(before, model.conv.weight))

    def test_muon_adamw_update_uses_local_orthogonalization(self) -> None:
        param = nn.Parameter(torch.arange(12, dtype=torch.float32).reshape(3, 4) / 10)
        gradient = torch.arange(1, 13, dtype=torch.float32).reshape(3, 4) / 10
        reference_buffer = torch.zeros_like(param)
        update = muon_update([gradient.clone()], [reference_buffer], beta=0.95, nesterov=True)[0]
        expected = param.detach().clone() * (1 - 0.02 * 0.1) - 0.02 * update

        optimizer = MuonOptimizer([param], lr=0.02, momentum=0.95, weight_decay=0.1)
        param.grad = gradient.clone()
        optimizer.step()
        torch.testing.assert_close(param, expected)

    def test_musgd_matches_ultralytics_reference_step(self) -> None:
        weight = nn.Parameter(torch.tensor([[0.1, -0.2, 0.3], [0.4, -0.5, 0.6]]))
        bias = nn.Parameter(torch.tensor([0.2, -0.4]))
        optimizer = MuSGD(
            [
                {
                    "params": [weight],
                    "lr": 0.01,
                    "momentum": 0.937,
                    "weight_decay": 0.0005,
                    "nesterov": True,
                    "use_muon": True,
                },
                {
                    "params": [bias],
                    "lr": 0.01,
                    "momentum": 0.937,
                    "weight_decay": 0.0,
                    "nesterov": True,
                    "use_muon": False,
                },
            ],
            muon=0.2,
            sgd=1.0,
        )
        weight.grad = weight.detach().clone()
        bias.grad = torch.tensor([0.3, -0.2])
        optimizer.step()
        torch.testing.assert_close(
            weight,
            torch.tensor(
                [
                    [0.09918702393770218, -0.19600394368171692, 0.29285016655921936],
                    [0.3911231458187103, -0.4895250201225281, 0.5878937244415283],
                ]
            ),
        )
        torch.testing.assert_close(bias, torch.tensor([0.19418899714946747, -0.39612600207328796]))

    def test_invalid_configs_fail_clearly(self) -> None:
        model = TinyModel()
        with self.assertRaises(ValueError):
            build_optimizer(model, AdamWConfig(lr=0))
        with self.assertRaises(ValueError):
            build_optimizer(model, MuonAdamWConfig(muon_momentum=1))
        with self.assertRaises(ValueError):
            build_optimizer(
                model,
                UltralyticsMuSGDConfig(boosted_parameter_names=frozenset({"missing"})),
            )
