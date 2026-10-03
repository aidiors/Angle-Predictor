import math
from unittest import TestCase

import torch

from angle_predictor.losses.angle import (
    ANGLE_LOSS_KINDS,
    angular_huber_loss,
    angular_mae_loss,
    build_angle_loss,
    mse_angle_loss,
)


def _label(angle_degrees: float) -> torch.Tensor:
    doubled = 2 * math.radians(angle_degrees)
    return torch.tensor([[math.sin(doubled), math.cos(doubled)]])


class AngleLossTests(TestCase):
    def test_default_selects_vector_charbonnier(self) -> None:
        pred = torch.tensor([[0.5, 1.0]])
        target = torch.tensor([[0.0, 1.0]])
        self.assertAlmostEqual(
            build_angle_loss()(pred, target).item(),
            build_angle_loss("vector_charbonnier")(pred, target).item(),
        )

    def test_factory_selects_function_once_and_binds_parameters(self) -> None:
        self.assertIs(build_angle_loss("mse"), mse_angle_loss)
        self.assertIs(build_angle_loss("angular_mae"), angular_mae_loss)
        self.assertNotEqual(
            build_angle_loss("angular_huber", angular_huber_beta_deg=0.1)(
                _label(0.2), _label(0)
            ).item(),
            build_angle_loss("angular_huber", angular_huber_beta_deg=1.0)(
                _label(0.2), _label(0)
            ).item(),
        )
        pred = torch.tensor([[0.5, 1.0]])
        target = torch.tensor([[0.0, 1.0]])
        self.assertNotEqual(
            build_angle_loss("smooth_l1", smooth_l1_beta=0.1)(pred, target).item(),
            build_angle_loss("smooth_l1", smooth_l1_beta=1.0)(pred, target).item(),
        )
        self.assertNotEqual(
            build_angle_loss("vector_charbonnier", charbonnier_epsilon=0.1)(pred, target).item(),
            build_angle_loss("vector_charbonnier", charbonnier_epsilon=1.0)(pred, target).item(),
        )

    def test_all_variants_vanish_for_matching_unit_vectors(self) -> None:
        target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        for kind in ANGLE_LOSS_KINDS:
            with self.subTest(kind=kind):
                self.assertAlmostEqual(build_angle_loss(kind)(target, target).item(), 0.0, places=6)

    def test_angular_mae_uses_undirected_wrap_at_180_degrees(self) -> None:
        actual = build_angle_loss("angular_mae")(_label(179), _label(1))
        self.assertAlmostEqual(actual.item(), math.radians(2), places=6)

    def test_angular_huber_uses_undirected_wrap_and_linear_tail(self) -> None:
        actual = build_angle_loss("angular_huber", angular_huber_beta_deg=0.1)(
            _label(179), _label(1)
        )
        self.assertAlmostEqual(actual.item(), math.radians(1.95), places=6)

    def test_angular_huber_is_quadratic_inside_beta(self) -> None:
        delta = math.radians(0.05)
        actual = build_angle_loss("angular_huber", angular_huber_beta_deg=0.1)(
            _label(0.05), _label(0)
        )
        expected = 0.5 * delta**2 / math.radians(0.1)
        self.assertAlmostEqual(actual.item(), expected, places=8)

    def test_angular_huber_penalizes_degenerate_prediction(self) -> None:
        prediction = torch.zeros(1, 2, requires_grad=True)
        loss = build_angle_loss("angular_huber")(prediction, _label(20))
        self.assertAlmostEqual(loss.item(), 1.0, places=6)
        loss.backward()
        assert prediction.grad is not None
        self.assertTrue(torch.isfinite(prediction.grad).all())
        self.assertGreater(prediction.grad.abs().sum().item(), 0.0)

    def test_angular_mae_zero_vector_degeneracy_is_visible(self) -> None:
        prediction = torch.zeros(1, 2)
        target = _label(0)
        self.assertEqual(build_angle_loss("angular_mae")(prediction, target).item(), 0.0)
        self.assertGreater(build_angle_loss("mse")(prediction, target).item(), 0.0)

    def test_vector_charbonnier_depends_on_vector_error_magnitude(self) -> None:
        target = torch.tensor([[0.0, 1.0]])
        first = torch.tensor([[1.0, 1.0]])
        second = torch.tensor([[0.0, 2.0]])
        criterion = build_angle_loss("vector_charbonnier")
        self.assertAlmostEqual(criterion(first, target).item(), criterion(second, target).item())

    def test_all_variants_have_finite_gradients_away_from_singularities(self) -> None:
        target = torch.tensor([[0.0, 1.0], [1.0, 0.0]])
        for kind in ANGLE_LOSS_KINDS:
            with self.subTest(kind=kind):
                prediction = torch.tensor([[0.2, 0.8], [0.7, -0.1]], requires_grad=True)
                loss = build_angle_loss(kind)(prediction, target)
                loss.backward()
                self.assertTrue(torch.isfinite(loss))
                assert prediction.grad is not None
                self.assertTrue(torch.isfinite(prediction.grad).all())
                self.assertGreater(prediction.grad.abs().sum().item(), 0.0)

    def test_rejects_unknown_kind_and_invalid_shapes(self) -> None:
        with self.assertRaises(ValueError):
            build_angle_loss("unknown")
        with self.assertRaises(ValueError):
            build_angle_loss("smooth_l1", smooth_l1_beta=0)
        with self.assertRaises(ValueError):
            build_angle_loss("vector_charbonnier", charbonnier_epsilon=float("nan"))
        with self.assertRaises(ValueError):
            build_angle_loss("angular_huber", angular_huber_beta_deg=0)
        with self.assertRaises(ValueError):
            angular_huber_loss(torch.zeros(1, 2), torch.ones(1, 2), beta_rad=float("inf"))
        with self.assertRaises(ValueError):
            build_angle_loss("mse")(torch.zeros(1, 2), torch.zeros(2, 2))
        with self.assertRaises(ValueError):
            build_angle_loss("mse")(torch.zeros(0, 2), torch.zeros(0, 2))
