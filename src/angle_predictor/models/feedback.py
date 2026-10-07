from __future__ import annotations

from typing import Literal

import torch
from torch import nn
from torch.nn import functional as F
from torch.utils.checkpoint import checkpoint

FeedbackMode = Literal["fixed", "gated", "routed"]


class StageFeedback(nn.Module):
    def __init__(self, early_channels: int, deep_channels: int, mode: FeedbackMode) -> None:
        super().__init__()
        self.mode = mode
        self.projection = nn.Conv2d(deep_channels, early_channels, 1)
        nn.init.zeros_(self.projection.weight)
        assert self.projection.bias is not None
        nn.init.zeros_(self.projection.bias)
        self.gate = None
        self.router = None
        self._generator: torch.Generator | None = None
        if mode in {"gated", "routed"}:
            self.gate = nn.Sequential(
                nn.LayerNorm(2 * early_channels),
                nn.Linear(2 * early_channels, early_channels),
                nn.Sigmoid(),
            )
            layer = self.gate[1]
            assert isinstance(layer, nn.Linear) and layer.bias is not None
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)
        if mode == "routed":
            self.router = nn.Sequential(
                nn.LayerNorm(deep_channels),
                nn.Linear(deep_channels, 128),
                nn.GELU(),
                nn.Linear(128, 2),
            )
            layer = self.router[-1]
            assert isinstance(layer, nn.Linear) and layer.bias is not None
            nn.init.zeros_(layer.weight)
            nn.init.zeros_(layer.bias)

    def update(self, early: torch.Tensor, deep: torch.Tensor) -> torch.Tensor:
        message = F.interpolate(
            self.projection(deep), size=early.shape[-2:], mode="bilinear", align_corners=False
        )
        if self.gate is not None:
            with torch.autocast(early.device.type, enabled=False):
                pooled = torch.cat(
                    (early.float().mean((-2, -1)), message.float().mean((-2, -1))), 1
                )
                gate = self.gate(pooled).to(message.dtype)
            message = message * gate[:, :, None, None]
        return early + message

    def routing_logits(self, deep: torch.Tensor) -> torch.Tensor:
        if self.router is None:
            raise ValueError("Routing logits require a learned router")
        with torch.autocast(deep.device.type, enabled=False):
            return self.router(deep.float().mean((-2, -1)))

    def repeat(self, stage: nn.Module, early: torch.Tensor) -> torch.Tensor:
        if self.training and torch.is_grad_enabled():
            return checkpoint(stage, early, use_reentrant=False)
        return stage(early)

    def forward(self, early: torch.Tensor, deep: torch.Tensor, stage: nn.Module) -> torch.Tensor:
        if self.router is None:
            return self.repeat(stage, self.update(early, deep))
        logits = self.routing_logits(deep)
        if self.training:
            if self._generator is None or self._generator.device != deep.device:
                self._generator = torch.Generator(device=deep.device).manual_seed(
                    torch.initial_seed()
                )
            uniform = torch.rand(
                logits.shape, generator=self._generator, device=deep.device
            ).clamp_(1e-6, 1 - 1e-6)
            probability = F.softmax(logits - (-uniform.log()).log(), dim=-1)
            hard = F.one_hot(probability.argmax(-1), 2).to(probability.dtype)
            choice = hard - probability.detach() + probability
            repeated = self.repeat(stage, self.update(early, deep))
            refine = choice[:, 1, None, None, None].to(deep.dtype)
            return deep + refine * (repeated - deep)
        selected = logits.argmax(-1) == 1
        if not bool(selected.any()):
            return deep
        updated = stage(self.update(early[selected], deep[selected]))
        result = deep.clone()
        result[selected] = updated
        return result


class ConvNeXtFeedback(nn.Module):
    def __init__(self, mode: FeedbackMode) -> None:
        super().__init__()
        self.stage3_to_2 = StageFeedback(192, 384, mode)
        self.stage4_to_3 = StageFeedback(384, 768, mode)

    def forward(self, backbone: nn.Module, image: torch.Tensor) -> torch.Tensor:
        first = backbone.get_submodule("stages_0")(
            backbone.get_submodule("stem_1")(backbone.get_submodule("stem_0")(image))
        )
        second = backbone.get_submodule("stages_1")(first)
        third = backbone.get_submodule("stages_2")(second)
        third = self.stage3_to_2(second, third, backbone.get_submodule("stages_2"))
        fourth = backbone.get_submodule("stages_3")(third)
        return self.stage4_to_3(third, fourth, backbone.get_submodule("stages_3"))
