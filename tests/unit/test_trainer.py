from unittest import TestCase

import torch
from torch import nn, optim
from torch.utils.data import DataLoader, TensorDataset

from dl_template.engine.trainer import train_epochs


class TrainerTests(TestCase):
    def test_train_epochs_updates_model(self) -> None:
        torch.manual_seed(1)
        features = torch.randn(32, 4)
        targets = (features.sum(dim=1, keepdim=True) > 0).float()
        loader = DataLoader(TensorDataset(features, targets), batch_size=8, shuffle=False)
        model = nn.Linear(4, 1)
        optimizer = optim.SGD(model.parameters(), lr=0.1)
        loss_fn = nn.BCEWithLogitsLoss()
        before = model.weight.detach().clone()

        def loss_step(module: nn.Module, batch: object) -> torch.Tensor:
            batch_features, batch_targets = batch  # type: ignore[misc]
            return loss_fn(module(batch_features), batch_targets)

        losses = train_epochs(model, loader, optimizer, loss_step, epochs=2)

        self.assertEqual(len(losses), 2)
        self.assertTrue(all(loss >= 0 for loss in losses))
        self.assertFalse(torch.equal(before, model.weight.detach()))
