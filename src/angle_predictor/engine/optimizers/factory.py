from __future__ import annotations

import math
from collections.abc import Callable, Iterable, Iterator
from dataclasses import dataclass, field
from typing import Any, overload

import torch
from torch import nn, optim

from angle_predictor.engine.optimizers._muon import MuSGD, muon_update


@dataclass(frozen=True, slots=True)
class AdamWConfig:
    lr: float = 3e-4
    weight_decay: float = 0.01
    betas: tuple[float, float] = (0.9, 0.999)


@dataclass(frozen=True, slots=True)
class MuonAdamWConfig:
    muon_lr: float = 0.02
    muon_momentum: float = 0.95
    muon_weight_decay: float = 0.1
    adamw_lr: float = 3e-4
    adamw_weight_decay: float = 0.01
    adamw_betas: tuple[float, float] = (0.9, 0.999)


@dataclass(frozen=True, slots=True)
class UltralyticsMuSGDConfig:
    lr: float = 0.01
    momentum: float = 0.937
    weight_decay: float = 0.0005
    nesterov: bool = True
    muon: float = 0.2
    sgd: float = 1.0
    boosted_parameter_names: frozenset[str] = field(default_factory=frozenset)


OptimizerConfig = AdamWConfig | MuonAdamWConfig | UltralyticsMuSGDConfig


class MuonOptimizer(optim.Optimizer):
    """Apply Muon updates with decoupled weight decay."""

    def __init__(
        self,
        params: Iterable[nn.Parameter],
        *,
        lr: float,
        momentum: float,
        weight_decay: float,
    ) -> None:
        super().__init__(
            params,
            defaults={"lr": lr, "momentum": momentum, "weight_decay": weight_decay},
        )

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        with torch.no_grad():
            for group in self.param_groups:
                params = [param for param in group["params"] if param.grad is not None]
                if not params:
                    continue
                buffers = []
                for param in params:
                    state = self.state[param]
                    if "momentum_buffer" not in state:
                        state["momentum_buffer"] = torch.zeros_like(param)
                    buffers.append(state["momentum_buffer"])
                updates = muon_update(
                    [param.grad for param in params],
                    buffers,
                    beta=group["momentum"],
                    nesterov=True,
                )
                if group["weight_decay"]:
                    torch._foreach_mul_(params, 1 - group["lr"] * group["weight_decay"])
                torch._foreach_add_(params, updates, alpha=-group["lr"])
        return loss


class ChainedOptimizer(optim.Optimizer):
    """Expose two disjoint optimizers as one object for training and checkpoints."""

    def __init__(self, first: optim.Optimizer, second: optim.Optimizer) -> None:
        groups = [*first.param_groups, *second.param_groups]
        super().__init__(groups, defaults={})
        self.optimizers = (first, second)
        self.param_groups = groups

    @overload
    def step(self, closure: None = None) -> None: ...

    @overload
    def step(self, closure: Callable[[], float]) -> float: ...

    def step(self, closure: Callable[[], float] | None = None) -> float | None:
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for optimizer in self.optimizers:
            optimizer.step()
        return loss

    def zero_grad(self, set_to_none: bool = True) -> None:
        for optimizer in self.optimizers:
            optimizer.zero_grad(set_to_none=set_to_none)

    def state_dict(self) -> dict[str, Any]:
        return {"optimizers": [optimizer.state_dict() for optimizer in self.optimizers]}

    def load_state_dict(self, state_dict: dict[str, Any]) -> None:
        states = state_dict.get("optimizers")
        if not isinstance(states, list) or len(states) != len(self.optimizers):
            raise ValueError("Expected one state dictionary per chained optimizer")
        for optimizer, state in zip(self.optimizers, states, strict=True):
            optimizer.load_state_dict(state)
        self.param_groups = [
            group for optimizer in self.optimizers for group in optimizer.param_groups
        ]


_NORM_TYPES = tuple(
    value
    for name, value in vars(nn).items()
    if "Norm" in name and isinstance(value, type) and issubclass(value, nn.Module)
)


def _owned_parameters(model: nn.Module) -> Iterator[tuple[str, nn.Module, nn.Parameter]]:
    seen: set[int] = set()
    for module_name, module in model.named_modules():
        for local_name, param in module.named_parameters(recurse=False):
            if not param.requires_grad or id(param) in seen:
                continue
            seen.add(id(param))
            name = f"{module_name}.{local_name}" if module_name else local_name
            yield name, module, param


def _no_decay(name: str, module: nn.Module) -> bool:
    return "bias" in name or isinstance(module, _NORM_TYPES) or "logit_scale" in name


def _adamw_groups(
    parameters: list[tuple[str, nn.Module, nn.Parameter]], weight_decay: float
) -> list[dict[str, Any]]:
    decay: list[nn.Parameter] = []
    no_decay: list[nn.Parameter] = []
    for name, module, param in parameters:
        (no_decay if _no_decay(name, module) else decay).append(param)
    groups = []
    if decay:
        groups.append({"params": decay, "weight_decay": weight_decay, "param_group": "weight"})
    if no_decay:
        groups.append({"params": no_decay, "weight_decay": 0.0, "param_group": "no_decay"})
    return groups


def _validate(config: OptimizerConfig) -> None:
    if isinstance(config, AdamWConfig):
        rates = (config.lr,)
        decays = (config.weight_decay,)
        betas = config.betas
    elif isinstance(config, MuonAdamWConfig):
        rates = (config.muon_lr, config.adamw_lr)
        decays = (config.muon_weight_decay, config.adamw_weight_decay)
        betas = config.adamw_betas
        if not math.isfinite(config.muon_momentum) or not 0 <= config.muon_momentum < 1:
            raise ValueError("muon_momentum must be in [0, 1)")
    elif isinstance(config, UltralyticsMuSGDConfig):
        rates = (config.lr,)
        decays = (config.weight_decay,)
        betas = None
        if not math.isfinite(config.momentum) or not 0 <= config.momentum < 1:
            raise ValueError("momentum must be in [0, 1)")
        if any(not math.isfinite(value) or value < 0 for value in (config.muon, config.sgd)):
            raise ValueError("Muon and SGD factors must be finite and nonnegative")
        if config.muon == 0 and config.sgd == 0:
            raise ValueError("At least one of Muon and SGD factors must be positive")
    else:
        raise TypeError("Unsupported optimizer configuration")
    if any(not math.isfinite(rate) or rate <= 0 for rate in rates):
        raise ValueError("Learning rates must be finite and positive")
    if any(not math.isfinite(decay) or decay < 0 for decay in decays):
        raise ValueError("Weight decay must be finite and nonnegative")
    if betas is not None and (
        len(betas) != 2 or any(not math.isfinite(beta) or not 0 <= beta < 1 for beta in betas)
    ):
        raise ValueError("AdamW betas must be finite and in [0, 1)")


def build_optimizer(model: nn.Module, config: OptimizerConfig) -> optim.Optimizer:
    """Build an optimizer after moving the model to its training device."""
    _validate(config)
    parameters = list(_owned_parameters(model))
    if not parameters:
        raise ValueError("Model has no trainable parameters")

    if isinstance(config, AdamWConfig):
        groups = _adamw_groups(parameters, config.weight_decay)
        return optim.AdamW(
            groups,
            lr=config.lr,
            betas=config.betas,
            fused=all(param.is_cuda for _, _, param in parameters),
        )

    if isinstance(config, MuonAdamWConfig):
        muon_items = [
            item
            for item in parameters
            if item[2].ndim in {2, 4}
            and not _no_decay(item[0], item[1])
            and not isinstance(item[1], (nn.Embedding, nn.EmbeddingBag))
        ]
        muon_ids = {id(param) for _, _, param in muon_items}
        muon_params = [param for _, _, param in muon_items]
        adamw_params = [item for item in parameters if id(item[2]) not in muon_ids]
        if not muon_params or not adamw_params:
            raise ValueError("Muon+AdamW requires both Muon and AdamW trainable parameters")
        muon = MuonOptimizer(
            muon_params,
            lr=config.muon_lr,
            momentum=config.muon_momentum,
            weight_decay=config.muon_weight_decay,
        )
        adamw = optim.AdamW(
            _adamw_groups(adamw_params, config.adamw_weight_decay),
            lr=config.adamw_lr,
            betas=config.adamw_betas,
            fused=all(param.is_cuda for _, _, param in adamw_params),
        )
        return ChainedOptimizer(muon, adamw)

    if not isinstance(config, UltralyticsMuSGDConfig):
        raise TypeError("Unsupported optimizer configuration")
    groups: dict[str, dict[str, nn.Parameter]] = {
        "weight": {},
        "bn": {},
        "bias": {},
        "muon": {},
    }
    for name, module, param in parameters:
        if param.ndim in {2, 4}:
            group = "muon"
        elif "bias" in name:
            group = "bias"
        elif isinstance(module, _NORM_TYPES) or "logit_scale" in name:
            group = "bn"
        else:
            group = "weight"
        groups[group][name] = param
    if not groups["muon"]:
        raise ValueError("Ultralytics MuSGD requires 2D or 4D trainable parameters")
    unknown_boosted = config.boosted_parameter_names - {
        name for group in groups.values() for name in group
    }
    if unknown_boosted:
        raise ValueError(f"Unknown boosted parameters: {sorted(unknown_boosted)}")

    param_groups: list[dict[str, Any]] = []
    for group_name in ("weight", "bn", "bias", "muon"):
        named = groups[group_name]
        if not named:
            continue
        weight_decay = config.weight_decay if group_name in {"weight", "muon"} else 0.0
        normal = [
            param for name, param in named.items() if name not in config.boosted_parameter_names
        ]
        boosted = [param for name, param in named.items() if name in config.boosted_parameter_names]
        for params, lr in ((boosted, config.lr * 3), (normal, config.lr)):
            if params:
                param_groups.append(
                    {
                        "params": params,
                        "lr": lr,
                        "momentum": config.momentum,
                        "nesterov": config.nesterov,
                        "weight_decay": weight_decay,
                        "use_muon": group_name == "muon",
                        "param_group": group_name,
                    }
                )
    return MuSGD(param_groups, muon=config.muon, sgd=config.sgd)
