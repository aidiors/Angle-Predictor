from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterable
from typing import overload

import torch
from torch import nn, optim


def _orthogonalize(matrices: torch.Tensor, eps: float = 1e-7) -> torch.Tensor:
    original_shape = matrices.shape
    x = matrices.reshape(-1, *original_shape[-2:]).bfloat16()
    x = x / (x.norm(dim=(-2, -1), keepdim=True) + eps)
    transposed = original_shape[-2] > original_shape[-1]
    if transposed:
        x = x.transpose(-2, -1)
    for _ in range(5):
        gram = x @ x.transpose(-2, -1)
        polynomial = torch.baddbmm(gram, gram, gram, beta=-4.7750, alpha=2.0315)
        x = torch.baddbmm(x, polynomial, x, beta=3.4445)
    if transposed:
        x = x.transpose(-2, -1)
    return x.reshape(original_shape)


def muon_update(
    gradients: list[torch.Tensor],
    buffers: list[torch.Tensor],
    *,
    beta: float = 0.95,
    nesterov: bool = True,
) -> list[torch.Tensor]:
    if len(gradients) != len(buffers):
        raise ValueError("Each gradient must have a momentum buffer")
    if any(gradient.ndim < 2 for gradient in gradients):
        raise ValueError("Muon requires parameters with at least two dimensions")
    if not gradients:
        return []

    torch._foreach_mul_(buffers, beta)
    torch._foreach_add_(buffers, gradients, alpha=1 - beta)
    if nesterov:
        updates = list(torch._foreach_mul(buffers, beta))
        torch._foreach_add_(updates, gradients, alpha=1 - beta)
    else:
        updates = list(buffers)

    buckets: dict[
        tuple[int, float, torch.device, torch.dtype],
        list[tuple[int, torch.Tensor, tuple[int, ...] | None]],
    ] = defaultdict(list)
    for index, update in enumerate(updates):
        standard = update.is_contiguous()
        channels_last = (
            not standard
            and update.ndim == 4
            and update.is_contiguous(memory_format=torch.channels_last)
        )
        matrix = (
            update.permute(0, 2, 3, 1).flatten(1)
            if channels_last
            else update.flatten(1)
            if update.ndim > 2
            else update.contiguous()
        )
        scaling = max(1, gradients[index].size(-2) / gradients[index].size(-1)) ** 0.5
        shape_key = (matrix.size(1), scaling, matrix.device, matrix.dtype)
        buckets[shape_key].append(
            (index, matrix, update.stride() if standard or channels_last else None)
        )

    for (columns, scaling, device, dtype), entries in buckets.items():
        entries.sort(key=lambda entry: -entry[1].size(0))
        start = 0
        for end in range(1, len(entries) + 1):
            if end < len(entries) and entries[start][1].size(0) <= 16 * entries[end][1].size(0):
                continue
            block = entries[start:end]
            padded = torch.zeros(
                len(block), block[0][1].size(0), columns, device=device, dtype=dtype
            )
            torch._foreach_add_(
                [padded[row, : matrix.size(0)] for row, (_, matrix, _) in enumerate(block)],
                [matrix for _, matrix, _ in block],
            )
            result = _orthogonalize(padded).contiguous().to(gradients[block[0][0]].dtype)
            result.mul_(scaling)
            for row, (index, matrix, stride) in enumerate(block):
                values = result[row, : matrix.size(0)]
                updates[index] = (
                    values.as_strided(gradients[index].shape, stride)
                    if stride is not None
                    else values.reshape(gradients[index].shape)
                )
            start = end
    return updates


class MuSGD(optim.Optimizer):
    """Combine Muon and SGD updates using the Ultralytics 8.4.163 step equations."""

    def __init__(
        self,
        params: Iterable[nn.Parameter] | Iterable[dict],
        lr: float = 1e-3,
        momentum: float = 0.0,
        weight_decay: float = 0.0,
        nesterov: bool = False,
        use_muon: bool = False,
        muon: float = 0.5,
        sgd: float = 0.5,
    ) -> None:
        defaults = {
            "lr": lr,
            "momentum": momentum,
            "weight_decay": weight_decay,
            "nesterov": nesterov,
            "use_muon": use_muon,
        }
        super().__init__(params, defaults)
        self.muon = muon
        self.sgd = sgd

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    @torch.no_grad()
    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()

        for group in self.param_groups:
            params = [param for param in group["params"] if param.grad is not None]
            if not params:
                continue
            momentum = group["momentum"]
            for param in params:
                state = self.state[param]
                if not state:
                    state["momentum_buffer"] = torch.zeros_like(param)
                    if group["use_muon"]:
                        state["momentum_buffer_SGD"] = torch.zeros_like(param)

            if group["use_muon"]:
                orthogonal_updates = muon_update(
                    [param.grad for param in params],
                    [self.state[param]["momentum_buffer"] for param in params],
                    beta=momentum,
                    nesterov=group["nesterov"],
                )
                torch._foreach_add_(params, orthogonal_updates, alpha=-group["lr"] * self.muon)
                sgd_buffers = [self.state[param]["momentum_buffer_SGD"] for param in params]
                sgd_lr = group["lr"] * self.sgd
            else:
                sgd_buffers = [self.state[param]["momentum_buffer"] for param in params]
                sgd_lr = group["lr"]

            gradients = [param.grad for param in params]
            if group["weight_decay"]:
                gradients = torch._foreach_add(gradients, params, alpha=group["weight_decay"])
            torch._foreach_mul_(sgd_buffers, momentum)
            torch._foreach_add_(sgd_buffers, gradients)
            sgd_updates = (
                torch._foreach_add(gradients, sgd_buffers, alpha=momentum)
                if group["nesterov"]
                else sgd_buffers
            )
            torch._foreach_add_(params, sgd_updates, alpha=-sgd_lr)
        return loss
