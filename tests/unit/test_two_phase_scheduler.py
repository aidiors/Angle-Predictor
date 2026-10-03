import copy
import math
from unittest import TestCase

import torch
from pydantic import ValidationError

from angle_predictor.config.angle_experiment import AngleExperimentParams, AngleTrainingConfig
from angle_predictor.engine.optimizers import build_optimizer
from angle_predictor.experiments.angle import _scheduler


def parameters(**overrides):
    settings = {
        "data": {"root": "."},
        "scheduler": {"kind": "cosine_two_phase"},
        "training": {"max_steps": 15600, "warmup_steps": 200},
    }
    settings.update(overrides)
    return AngleExperimentParams.model_validate(settings)


class TwoPhaseSchedulerTests(TestCase):
    def test_actual_update_rates_cover_warmup_fast_decay_and_tail_without_restart(self):
        base = 0.0007975423028925935
        params = parameters()
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=base)
        scheduler = _scheduler(optimizer, params)
        rates = [optimizer.param_groups[0]["lr"]]
        for _ in range(15602):
            optimizer.step()
            scheduler.step()
            rates.append(optimizer.param_groups[0]["lr"])
        self.assertAlmostEqual(rates[0], base / 200, places=15)
        self.assertAlmostEqual(rates[200], base, places=15)
        self.assertAlmostEqual(rates[2600], base * 0.505, places=15)
        self.assertAlmostEqual(rates[5000], base * 0.01, places=15)
        self.assertAlmostEqual(rates[10300], (base * 0.01 + 1e-8) / 2, places=15)
        self.assertEqual(rates[15600:], [1e-8] * 3)
        self.assertTrue(all(math.isfinite(lr) and lr > 0 for lr in rates))
        self.assertTrue(all(a <= b for a, b in zip(rates[:200], rates[1:201], strict=True)))
        self.assertTrue(all(a >= b for a, b in zip(rates[200:-1], rates[201:], strict=True)))
        # Both cosine derivatives vanish at the join; no abrupt drop/restart at5000.
        self.assertLess(rates[4999] - rates[5000], base * 1e-6)
        self.assertLess(rates[5000] - rates[5001], base * 1e-6)

    def test_zero_warmup_starts_at_peak(self):
        params = parameters(
            training={"max_steps": 20, "warmup_steps": 0},
            scheduler={
                "kind": "cosine_two_phase",
                "fast_decay_steps": 8,
            },
        )
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.001)
        scheduler = _scheduler(optimizer, params)
        self.assertEqual(scheduler.get_last_lr(), [0.001])

    def test_old_cosine_matches_torch_schedule_at_every_step(self):
        params = parameters(scheduler={}, training={"max_steps": 30, "warmup_steps": 3})
        actual = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.001)
        expected = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.001)
        scheduler = _scheduler(actual, params)
        warmup = torch.optim.lr_scheduler.LinearLR(
            expected, start_factor=1 / 3, end_factor=1.0, total_iters=3
        )
        cosine = torch.optim.lr_scheduler.CosineAnnealingLR(expected, T_max=27, eta_min=1e-8)
        reference = torch.optim.lr_scheduler.SequentialLR(
            expected, schedulers=[warmup, cosine], milestones=[3]
        )
        for _ in range(31):
            self.assertEqual(scheduler.get_last_lr(), reference.get_last_lr())
            actual.step()
            scheduler.step()
            expected.step()
            reference.step()

    def test_all_scaled_groups_in_musgd_and_chained_optimizer_receive_the_schedule(self):
        for kind in ("musgd", "muon_adamw"):
            with self.subTest(kind=kind):
                params = parameters(
                    optimizer={"kind": kind},
                    scheduler={"kind": "cosine_two_phase", "fast_decay_steps": 8},
                    training={"max_steps": 20, "warmup_steps": 2},
                )
                model = torch.nn.Sequential(torch.nn.Linear(4, 3), torch.nn.LayerNorm(3))
                optimizer = build_optimizer(model, params.to_optimizer_config())
                for i, group in enumerate(optimizer.param_groups):
                    if i % 2 == 0:
                        group["lr"] *= 0.1
                bases = [group["lr"] for group in optimizer.param_groups]
                scheduler = _scheduler(optimizer, params)
                for step in range(1, 21):
                    optimizer.zero_grad()
                    loss = model(torch.ones(2, 4)).square().mean()
                    self.assertTrue(torch.isfinite(loss))
                    loss.backward()
                    optimizer.step()
                    scheduler.step()
                    if step == 8:
                        for group, base in zip(optimizer.param_groups, bases, strict=True):
                            self.assertAlmostEqual(group["lr"], base * 0.01, places=15)
                self.assertEqual(scheduler.get_last_lr(), [1e-8] * len(bases))
                if hasattr(optimizer, "optimizers"):
                    actual_groups = [g for part in optimizer.optimizers for g in part.param_groups]
                    self.assertEqual(
                        [id(g) for g in actual_groups], [id(g) for g in optimizer.param_groups]
                    )
                    self.assertEqual([g["lr"] for g in actual_groups], scheduler.get_last_lr())

    def test_checkpoint_resume_has_identical_next_update_lr_in_each_phase(self):
        params = parameters(
            scheduler={"kind": "cosine_two_phase", "fast_decay_steps": 8},
            training={"max_steps": 20, "warmup_steps": 2},
        )
        for saved_step in (1, 7, 8, 13, 20):
            with self.subTest(step=saved_step):

                def make():
                    optimizer = torch.optim.SGD(
                        [
                            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 0.001},
                            {"params": [torch.nn.Parameter(torch.ones(1))], "lr": 0.0001},
                        ]
                    )
                    return optimizer, _scheduler(optimizer, params)

                original, scheduler = make()
                for _ in range(saved_step):
                    original.step()
                    scheduler.step()
                restored, resumed = make()
                restored.load_state_dict(copy.deepcopy(original.state_dict()))
                resumed.load_state_dict(copy.deepcopy(scheduler.state_dict()))
                self.assertEqual(resumed.get_last_lr(), scheduler.get_last_lr())
                original.step()
                scheduler.step()
                restored.step()
                resumed.step()
                self.assertEqual(resumed.get_last_lr(), scheduler.get_last_lr())

    def test_invalid_phase_boundaries_and_tail_rates_are_rejected(self):
        for settings in (
            {"fast_decay_steps": 200},
            {"fast_decay_steps": 15600},
            {"tail_start_factor": 0},
            {"tail_start_factor": 1},
            {"tail_start_factor": float("nan")},
        ):
            with self.subTest(settings=settings), self.assertRaises(ValidationError):
                parameters(scheduler={"kind": "cosine_two_phase", **settings})
        params = parameters(scheduler={"kind": "cosine_two_phase", "eta_min": 1e-5})
        optimizer = torch.optim.SGD([torch.nn.Parameter(torch.ones(1))], lr=0.001)
        with self.assertRaisesRegex(ValueError, "tail LR"):
            _scheduler(optimizer, params)

    def test_validation_milestone_union_includes_final_step_once(self):
        config = AngleTrainingConfig(
            max_steps=8, warmup_steps=0, validate_every_steps=3, validate_at_steps=(6, 5)
        )
        self.assertEqual(config.validation_steps(), (3, 5, 6, 8))
        self.assertEqual(AngleTrainingConfig().validation_steps(), (1300, 2600, 3900, 5200))
        for milestones in ((0,), (9,), (5, 5)):
            with self.subTest(milestones=milestones), self.assertRaises(ValidationError):
                AngleTrainingConfig(max_steps=8, warmup_steps=0, validate_at_steps=milestones)
