"""Detector = timm backbone (CLIP / DINOv2 / SigLIP / ConvNeXt ...) + linear head.

Two feature modes:
  * feat_layers=None  -> the backbone's own pooled output (final CLS token after the last norm).
  * feat_layers=[...] -> CLS tokens taken after several transformer blocks, each LayerNorm-ed, then
    concatenated. Intermediate CLIP layers keep low-level texture/blending cues that the last,
    semantics-oriented layer discards (cf. RINE, ECCV 2024), which matters for forgery detection.
"""
from __future__ import annotations

import timm
import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from .lora import inject_lora
from .svd_adapter import inject_svd

# Suffixes of the nn.Linear layers that receive LoRA adapters (timm ViT naming).
LORA_TARGETS_ATTN = ("attn.qkv", "attn.proj")
LORA_TARGETS_ALL = LORA_TARGETS_ATTN + ("mlp.fc1", "mlp.fc2")


def parse_feat_layers(spec: str | None, depth: int) -> list[int] | None:
    """'' / None -> None; 'auto' -> every 4th block ending at the last; '7,15,23' -> explicit."""
    if not spec:
        return None
    if spec == "auto":
        return list(range(depth - 1, -1, -4))[::-1]
    layers = sorted({int(s) % depth for s in spec.split(",")})
    return layers


class Detector(nn.Module):
    def __init__(self, arch: str, pretrained: bool = True, drop: float = 0.1,
                 img_size: int | None = None, feat_layers: list[int] | str | None = None):
        super().__init__()
        kw = {"img_size": img_size} if img_size else {}  # e.g. run DINOv2 at 224 instead of 518
        self.backbone = timm.create_model(arch, pretrained=pretrained, num_classes=0, **kw)
        if isinstance(feat_layers, str):
            feat_layers = parse_feat_layers(feat_layers, len(self.backbone.blocks))
        self.feat_layers = list(feat_layers) if feat_layers else None
        self.grad_ckpt = False
        dim = self.backbone.num_features
        if self.feat_layers:
            self.layer_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in self.feat_layers)
            dim *= len(self.feat_layers)
        self.head = nn.Sequential(nn.Dropout(drop), nn.Linear(dim, 1))

    def features(self, x: torch.Tensor) -> torch.Tensor:
        if not self.feat_layers:
            return self.backbone(x)
        b = self.backbone  # mirrors timm VisionTransformer / Eva forward_features
        pe = b._pos_embed(b.patch_embed(x))
        if isinstance(pe, tuple):  # Eva family (DINOv3, PE): RoPE embeddings passed to every block
            x, kw = b.norm_pre(pe[0]), {"rope": pe[1]}
        else:
            x, kw = b.norm_pre(b.patch_drop(pe)), {}
        wanted, cls = set(self.feat_layers), []
        for i, blk in enumerate(b.blocks):
            if self.grad_ckpt and self.training:
                x = checkpoint(blk, x, use_reentrant=False, **kw)
            else:
                x = blk(x, **kw)
            if i in wanted:
                cls.append(x[:, 0])
        return torch.cat([ln(c) for ln, c in zip(self.layer_norms, cls)], dim=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.features(x)).squeeze(1)  # logit of P(fake)

    def set_grad_checkpointing(self, enable: bool = True):
        self.grad_ckpt = enable
        self.backbone.set_grad_checkpointing(enable)


def build_detector(arch: str, pretrained: bool, adapter: str = "lora", r: int = 16,
                   alpha: float = 32, lora_dropout: float = 0.05, svd_rank: int = 4,
                   targets=LORA_TARGETS_ALL, train_norms: bool = True,
                   head_drop: float = 0.1, img_size: int | None = None,
                   feat_layers: str | None = None) -> tuple[Detector, dict]:
    """Freeze the backbone, add adapters (LoRA or Effort-style SVD residuals), keep head
    (+ optionally LayerNorms) trainable. Returns the model and its optimiser param groups."""
    model = Detector(arch, pretrained=pretrained, drop=head_drop, img_size=img_size,
                     feat_layers=feat_layers)
    for p in model.backbone.parameters():
        p.requires_grad = False
    if adapter == "lora":
        wrapped = inject_lora(model.backbone, targets, r=r, alpha=alpha, dropout=lora_dropout)
        tag = "lora_"
    elif adapter == "svd":
        wrapped = inject_svd(model.backbone, targets, rank=svd_rank)
        tag = ("U_r", "S_r", "V_r")
    else:
        raise ValueError(adapter)
    if not wrapped:
        raise ValueError(f"No adapter target layers matched in {arch}; check `targets`.")

    groups = {"adapter": [], "norm": [], "head": list(model.head.parameters())}
    if model.feat_layers:
        groups["head"] += list(model.layer_norms.parameters())
    for n, p in model.backbone.named_parameters():
        if (n.rsplit(".", 1)[-1] in tag) if adapter == "svd" else (tag in n):
            p.requires_grad = True
            groups["adapter"].append(p)
        elif train_norms and ".norm" in f".{n}" and p.ndim == 1:
            p.requires_grad = True
            groups["norm"].append(p)
    return model, groups


def data_config(model: Detector, img_size: int | None = None) -> dict:
    cfg = timm.data.resolve_data_config({}, model=model.backbone)
    return {"mean": list(cfg["mean"]), "std": list(cfg["std"]),
            "img_size": int(img_size or cfg["input_size"][-1])}
