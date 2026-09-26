"""Small reusable PyTorch training-loop primitives."""

from __future__ import annotations

from collections.abc import Callable, Iterable
from typing import Any

import torch
from torch import nn, optim

LossStep = Callable[[nn.Module, Any], torch.Tensor]


def train_epochs(
    model: nn.Module,
    batches: Iterable[Any],
    optimizer: optim.Optimizer,
    loss_step: LossStep,
    *,
    epochs: int,
) -> list[float]:
    """Train for multiple epochs and return the mean batch loss per epoch.

    Batch structure and device transfer deliberately remain task-specific and
    live inside ``loss_step``. This keeps the engine usable for CV, NLP and
    other PyTorch tasks without teaching it a particular batch schema.
    """
    epoch_losses: list[float] = []

    for _ in range(epochs):
        model.train()
        total_loss = 0.0
        batch_count = 0

        for batch in batches:
            optimizer.zero_grad(set_to_none=True)
            loss = loss_step(model, batch)
            loss.backward()
            optimizer.step()
            total_loss += float(loss.detach().item())
            batch_count += 1

        if batch_count == 0:
            raise ValueError("Training batches must not be empty")
        epoch_losses.append(total_loss / batch_count)

    return epoch_losses
