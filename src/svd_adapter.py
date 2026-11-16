"""Effort-style orthogonal-subspace fine-tuning (Yan et al., "Orthogonal Subspace Decomposition for
Generalizable AI-Generated Image Detection", ICML 2025).

Each target weight is split by SVD, W = U S V^T:
  * principal part  W_main = U[:, :k] S[:k] V[:k]  (top singular directions) -> frozen. It holds the
    pretrained knowledge, so the model keeps its general-purpose features and generalises better to
    unseen forgery methods;
  * residual part   U_r diag(S_r) V_r             (the `rank` smallest directions) -> trainable.
    It learns the forgery-specific signal in a subspace orthogonal to the pretrained one.

Two regularisers keep the decomposition well-behaved:
  * orthogonality: [U_main U_r] and [V_main V_r] stay orthonormal (residual stays orthogonal to the
    frozen principal subspace);
  * norm preservation: ||W||_F stays close to the pretrained ||W0||_F.

After training, `merge_svd` turns every layer back into a plain nn.Linear (zero inference overhead).
"""
from __future__ import annotations

from typing import Iterable

import torch
import torch.nn as nn
import torch.nn.functional as F

from .lora import _set_submodule


class SVDResidualLinear(nn.Module):
    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        W = base.weight.detach().float()
        U, S, Vh = torch.linalg.svd(W, full_matrices=False)
        k = S.numel() - rank
        self.register_buffer("weight_main", (U[:, :k] * S[:k]) @ Vh[:k])
        self.register_buffer("U_main", U[:, :k].contiguous())
        self.register_buffer("V_main", Vh[:k].contiguous())
        self.register_buffer("w0_sq", W.pow(2).sum())
        self.U_r = nn.Parameter(U[:, k:].contiguous())
        self.S_r = nn.Parameter(S[k:].clone())
        self.V_r = nn.Parameter(Vh[k:].contiguous())
        self.bias = None if base.bias is None else nn.Parameter(base.bias.detach().clone(),
                                                                requires_grad=False)
        self.in_features, self.out_features = base.in_features, base.out_features

    def weight(self) -> torch.Tensor:
        return self.weight_main + (self.U_r * self.S_r) @ self.V_r

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.linear(x, self.weight(), self.bias)

    def reg_loss(self) -> torch.Tensor:
        U_r, V_r = self.U_r.float(), self.V_r.float()
        eye = torch.eye(U_r.shape[1], device=U_r.device)
        # ||[U_main U_r]^T [U_main U_r] - I||_F^2 = 2||U_main^T U_r||^2 + ||U_r^T U_r - I||^2
        orth = (2 * (self.U_main.t() @ U_r).pow(2).sum() + (U_r.t() @ U_r - eye).pow(2).sum()
                + 2 * (self.V_main @ V_r.t()).pow(2).sum() + (V_r @ V_r.t() - eye).pow(2).sum())
        keep = (self.weight().pow(2).sum() - self.w0_sq).abs() / self.w0_sq
        return orth + keep

    @torch.no_grad()
    def merged(self) -> nn.Linear:
        out = nn.Linear(self.in_features, self.out_features, bias=self.bias is not None)
        out.weight.copy_(self.weight().float())
        if self.bias is not None:
            out.bias.copy_(self.bias.float())
        return out.to(self.weight_main.device)


def inject_svd(model: nn.Module, targets: Iterable[str], rank: int) -> list[str]:
    targets = tuple(targets)
    names = [n for n, m in model.named_modules()
             if isinstance(m, nn.Linear) and n.endswith(targets)]
    for n in names:
        _set_submodule(model, n, SVDResidualLinear(model.get_submodule(n), rank))
    return names


def svd_reg_loss(model: nn.Module) -> torch.Tensor | None:
    terms = [m.reg_loss() for m in model.modules() if isinstance(m, SVDResidualLinear)]
    return torch.stack(terms).mean() if terms else None


def merge_svd(model: nn.Module) -> int:
    names = [n for n, m in model.named_modules() if isinstance(m, SVDResidualLinear)]
    for n in names:
        _set_submodule(model, n, model.get_submodule(n).merged())
    return len(names)
