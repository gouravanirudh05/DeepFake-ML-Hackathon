"""Minimal LoRA (Hu et al., 2021) for nn.Linear layers.

Written in-house (no peft dependency) so that adapters can be merged back into
plain nn.Linear weights after training. The merged model is an ordinary timm
model: inference needs no LoRA code and pays zero extra latency.
"""
from __future__ import annotations

import math
from typing import Iterable

import torch
import torch.nn as nn


class LoRALinear(nn.Module):
    """y = W x + b + (alpha / r) * B A dropout(x), with W, b frozen."""

    def __init__(self, base: nn.Linear, r: int, alpha: float, dropout: float = 0.0):
        super().__init__()
        self.base = base
        self.r = r
        self.scaling = alpha / r
        self.lora_A = nn.Parameter(torch.empty(r, base.in_features))
        self.lora_B = nn.Parameter(torch.zeros(base.out_features, r))  # zero init -> starts as identity
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.dropout = nn.Dropout(dropout) if dropout > 0 else nn.Identity()
        for p in self.base.parameters():
            p.requires_grad = False

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        lora = (self.dropout(x) @ self.lora_A.t()) @ self.lora_B.t()
        return self.base(x) + lora * self.scaling

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        out = nn.Linear(self.base.in_features, self.base.out_features,
                        bias=self.base.bias is not None)
        delta = (self.lora_B.float() @ self.lora_A.float()) * self.scaling
        out.weight.copy_(self.base.weight.float() + delta)
        if self.base.bias is not None:
            out.bias.copy_(self.base.bias.float())
        return out.to(self.base.weight.device)


def _set_submodule(root: nn.Module, name: str, new: nn.Module) -> None:
    parent_name, _, child = name.rpartition(".")
    parent = root.get_submodule(parent_name) if parent_name else root
    setattr(parent, child, new)


def inject_lora(model: nn.Module, targets: Iterable[str], r: int, alpha: float,
                dropout: float = 0.0) -> list[str]:
    """Wrap every nn.Linear whose qualified name ends with one of `targets`."""
    targets = tuple(targets)
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.endswith(targets)]
    for n in names:
        _set_submodule(model, n, LoRALinear(model.get_submodule(n), r, alpha, dropout))
    return names


def merge_lora(model: nn.Module) -> int:
    """Replace every LoRALinear with an equivalent plain nn.Linear (in place)."""
    names = [n for n, m in model.named_modules() if isinstance(m, LoRALinear)]
    for n in names:
        _set_submodule(model, n, model.get_submodule(n).merged())
    return len(names)
