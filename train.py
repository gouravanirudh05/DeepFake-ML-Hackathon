"""LoRA fine-tuning of a frozen foundation backbone (default: CLIP ViT-L/14) for deepfake detection.

One run = one fold. Outputs in <out_dir>/<run_name>/fold<k>/:
    member.pt      merged (LoRA folded into weights) fp16 model, ready for the inference bundle
    oof.csv        out-of-fold P(fake) on the clean validation fold
    robustness.json  per-degradation validation AUC (proxy for the hidden test variants)
    history.csv    per-epoch loss / AUC (training curves for the report)

Example (Kaggle / Colab T4):
    python train.py --fold 0 --arch vit_large_patch14_clip_224.openai --bs 32 --epochs 10
"""
from __future__ import annotations

import argparse
import json
import math
import random
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from torch.utils.data import DataLoader

from src.data import EvalDataset, TrainDataset
from src.degradations import VARIANTS
from src.lora import merge_lora
from src.model import LORA_TARGETS_ALL, LORA_TARGETS_ATTN, build_detector, data_config
from src.svd_adapter import merge_svd, svd_reg_loss


def parse_args():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds_csv", default="splits/folds.csv")
    ap.add_argument("--fold", type=int, default=0)
    ap.add_argument("--data_dir", default=None,
                    help="dataset root containing train/real, train/fake (overrides paths in folds_csv)")
    ap.add_argument("--full", action="store_true", help="train on ALL data (no validation)")
    ap.add_argument("--arch", default="vit_large_patch14_clip_224.openai")
    ap.add_argument("--img_size", type=int, default=None)
    ap.add_argument("--no_pretrained", action="store_true", help="random init (smoke tests only)")
    ap.add_argument("--feat_layers", default="",
                    help="'' = final pooled token; 'auto' = CLS of every 4th block; or '7,15,23'")
    # adapters
    ap.add_argument("--adapter", choices=["lora", "svd"], default="lora",
                    help="svd = Effort-style orthogonal subspace fine-tuning (ICML 2025)")
    ap.add_argument("--svd_rank", type=int, default=4, help="trainable smallest singular directions")
    ap.add_argument("--svd_reg", type=float, default=1.0, help="weight of orthogonality + norm loss")
    ap.add_argument("--r", type=int, default=16)
    ap.add_argument("--alpha", type=float, default=32)
    ap.add_argument("--lora_dropout", type=float, default=0.05)
    ap.add_argument("--targets", choices=["attn", "all"], default="all")
    ap.add_argument("--no_train_norms", action="store_true")
    # optimisation
    ap.add_argument("--epochs", type=int, default=10)
    ap.add_argument("--bs", type=int, default=32)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=3e-4, help="LoRA + LayerNorm LR")
    ap.add_argument("--head_lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=float, default=0.05, help="fraction of steps")
    ap.add_argument("--label_smoothing", type=float, default=0.05, help="BCE only")
    # label-noise robustness
    ap.add_argument("--loss", choices=["bce", "gce"], default="bce",
                    help="gce = Generalized Cross-Entropy (Zhang & Sabuncu 2018), noise-tolerant")
    ap.add_argument("--gce_q", type=float, default=0.7)
    ap.add_argument("--drop_rate", type=float, default=0.0,
                    help="small-loss trick: drop this fraction of highest-loss samples per batch")
    ap.add_argument("--drop_warmup", type=float, default=1.0,
                    help="epochs over which drop_rate ramps up linearly from 0")
    ap.add_argument("--head_drop", type=float, default=0.1)
    ap.add_argument("--aug_strength", type=float, default=1.0)
    ap.add_argument("--sbi_prob", type=float, default=0.0,
                    help="prob. a real train image becomes a self-blended pseudo-fake (src/sbi.py)")
    ap.add_argument("--grad_ckpt", action="store_true")
    # misc
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--limit", type=int, default=0, help="subsample N train/val images (smoke test)")
    ap.add_argument("--out_dir", default="outputs")
    ap.add_argument("--run_name", default="clip_l14_lora")
    return ap.parse_args()


def seed_everything(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@torch.no_grad()
def predict(model, loader, device, amp) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    probs, labels = [], []
    for x, y in loader:
        x = x.to(device, non_blocking=True)
        with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
            logit = model(x) + model(torch.flip(x, dims=[3]))  # hflip TTA (same as inference)
        probs.append(torch.sigmoid(logit.float() / 2).cpu().numpy())
        labels.append(y.numpy())
    return np.concatenate(probs), np.concatenate(labels)


def per_sample_loss(logits, y, args):
    if args.loss == "gce":
        p_true = torch.sigmoid(logits * (2 * y - 1)).clamp_min(1e-6)  # prob. of the given label
        return (1 - p_true.pow(args.gce_q)) / args.gce_q
    y_s = y * (1 - args.label_smoothing) + 0.5 * args.label_smoothing
    return F.binary_cross_entropy_with_logits(logits, y_s, reduction="none")


def reduce_loss(losses, rate):
    """Average the (1 - rate) smallest per-sample losses (likely-clean samples)."""
    if rate <= 0:
        return losses.mean()
    keep = max(1, int(round(losses.numel() * (1 - rate))))
    return torch.topk(losses, keep, largest=False).values.mean()


def safe_auc(y, p):
    return float(roc_auc_score(y, p)) if len(np.unique(y)) == 2 else float("nan")


def main():
    args = parse_args()
    seed_everything(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    amp = device == "cuda"
    out = Path(args.out_dir) / args.run_name / ("full" if args.full else f"fold{args.fold}")
    out.mkdir(parents=True, exist_ok=True)
    (out / "args.json").write_text(json.dumps(vars(args), indent=2))

    # ── data ──
    df = pd.read_csv(args.folds_csv)
    if args.data_dir:
        sub = df.label.map({0: "real", 1: "fake"})
        df["path"] = [str(Path(args.data_dir) / "train" / s / f"{i}.jpg") for s, i in zip(sub, df.image_id)]
    tr_df = df if args.full else df[df.fold != args.fold]
    va_df = df.iloc[:0] if args.full else df[df.fold == args.fold]
    if args.limit:
        tr_df = tr_df.sample(min(args.limit, len(tr_df)), random_state=args.seed)
        va_df = va_df.sample(min(args.limit, len(va_df)), random_state=args.seed)

    # ── model ──
    targets = LORA_TARGETS_ALL if args.targets == "all" else LORA_TARGETS_ATTN
    model, groups = build_detector(
        args.arch, pretrained=not args.no_pretrained, adapter=args.adapter, r=args.r,
        alpha=args.alpha, lora_dropout=args.lora_dropout, svd_rank=args.svd_rank, targets=targets,
        train_norms=not args.no_train_norms,
        head_drop=args.head_drop, img_size=args.img_size, feat_layers=args.feat_layers)
    if args.grad_ckpt:
        model.set_grad_checkpointing(True)
    model.to(device)
    dcfg = data_config(model, args.img_size)
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_total = sum(p.numel() for p in model.parameters())
    print(f"device={device} arch={args.arch} img={dcfg['img_size']} "
          f"trainable={n_train/1e6:.2f}M / {n_total/1e6:.1f}M ({100*n_train/n_total:.2f}%) "
          f"train={len(tr_df)} val={len(va_df)} feat_layers={model.feat_layers} "
          f"adapter={args.adapter} loss={args.loss} drop_rate={args.drop_rate} sbi={args.sbi_prob}")

    tr_dl = DataLoader(TrainDataset(tr_df, dcfg["img_size"], dcfg["mean"], dcfg["std"],
                                    args.aug_strength, args.sbi_prob),
                       batch_size=args.bs, shuffle=True, num_workers=args.workers,
                       pin_memory=amp, drop_last=True, persistent_workers=args.workers > 0)

    def eval_loader(variant):
        ds = EvalDataset(va_df, dcfg["img_size"], dcfg["mean"], dcfg["std"], variant, args.seed)
        return DataLoader(ds, batch_size=args.bs * 2, shuffle=False, num_workers=args.workers,
                          pin_memory=amp)

    # ── optimiser / schedule ──
    opt = torch.optim.AdamW([
        {"params": groups["adapter"], "lr": args.lr,
         "weight_decay": args.wd if args.adapter == "lora" else 0.0},  # no decay on singular vectors
        {"params": groups["norm"], "lr": args.lr, "weight_decay": 0.0},
        {"params": groups["head"], "lr": args.head_lr, "weight_decay": args.wd},
    ])
    total = max(1, args.epochs * len(tr_dl) // args.accum)
    warm = max(1, int(args.warmup * total))
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warm) * 0.5 * (1 + math.cos(math.pi * min(1.0, s / total))))
    scaler = torch.amp.GradScaler("cuda", enabled=amp)
    trainable = {n for n, p in model.named_parameters() if p.requires_grad}

    best, best_row, history = -1.0, {}, []
    for epoch in range(args.epochs):
        model.train()
        t0, losses = time.time(), []
        opt.zero_grad(set_to_none=True)
        for step, (x, y) in enumerate(tr_dl):
            x, y = x.to(device, non_blocking=True), y.to(device, non_blocking=True)
            rate = args.drop_rate * min(1.0, (epoch + step / len(tr_dl)) / max(args.drop_warmup, 1e-8))
            with torch.autocast("cuda", dtype=torch.float16, enabled=amp):
                logits = model(x).float()
            loss = reduce_loss(per_sample_loss(logits, y, args), rate)
            if args.adapter == "svd" and args.svd_reg > 0:
                loss = loss + args.svd_reg * svd_reg_loss(model.backbone)
            scaler.scale(loss / args.accum).backward()
            if (step + 1) % args.accum == 0:
                scaler.unscale_(opt)
                torch.nn.utils.clip_grad_norm_([p for g in opt.param_groups for p in g["params"]], 1.0)
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                sched.step()
            losses.append(loss.item())

        row = {"epoch": epoch, "train_loss": float(np.mean(losses)), "lr": sched.get_last_lr()[0]}
        if len(va_df):
            p, y = predict(model, eval_loader("clean"), device, amp)
            pm, ym = predict(model, eval_loader("mixed"), device, amp)
            row.update(val_loss=float(F.binary_cross_entropy(torch.tensor(p).float().clamp(1e-6, 1 - 1e-6),
                                                             torch.tensor(y).float())),
                       val_auc=safe_auc(y, p), val_auc_degraded=safe_auc(ym, pm))
            row["score"] = 0.5 * (row["val_auc"] + row["val_auc_degraded"])
        else:
            row["score"] = epoch  # full-data run: keep the last epoch
        row["sec"] = round(time.time() - t0, 1)
        history.append(row)
        pd.DataFrame(history).to_csv(out / "history.csv", index=False)
        print(json.dumps({k: (round(v, 4) if isinstance(v, float) else v) for k, v in row.items()}))
        if row["score"] > best:
            best, best_row = row["score"], dict(row)
            torch.save({n: p.detach().cpu() for n, p in model.state_dict().items() if n in trainable},
                       out / "best_trainable.pt")

    # ── final: reload best adapters, robustness report, OOF preds ──
    model.load_state_dict(torch.load(out / "best_trainable.pt"), strict=False)
    if len(va_df):
        report = {}
        for v in VARIANTS:
            p, y = predict(model, eval_loader(v), device, amp)
            report[v] = safe_auc(y, p)
            if v == "clean":
                pd.DataFrame({"image_id": va_df.image_id.values, "label": y.astype(int),
                              "confidence": p, "fold": args.fold}).to_csv(out / "oof.csv", index=False)
        deg = [a for k, a in report.items() if k != "clean"]
        report["mean_degraded"], report["worst_degraded"] = float(np.mean(deg)), float(np.min(deg))
        (out / "robustness.json").write_text(json.dumps(report, indent=2))
        print("robustness:", json.dumps({k: round(v, 4) for k, v in report.items()}))

    n_merged = merge_lora(model.backbone) + merge_svd(model.backbone)
    model.set_grad_checkpointing(False)
    torch.save({
        "arch": args.arch, "img_size": dcfg["img_size"], "mean": dcfg["mean"], "std": dcfg["std"],
        "head_drop": args.head_drop, "feat_layers": model.feat_layers,
        "state_dict": {k: v.detach().cpu().half() for k, v in model.state_dict().items()},
        "metrics": best_row,
    }, out / "member.pt")
    print(f"merged {n_merged} adapter layers -> {out / 'member.pt'}")


if __name__ == "__main__":
    main()
