"""
Deepfake ML Hackathon — inference script.

    python inference_script.py --test_dir ./test/images/ --weights ./model_weights.pt --output predictions.csv

Self-contained: needs only torch, torchvision, timm, pandas, Pillow (no repo imports, no downloads).
`--weights` is a bundle produced by build_bundle.py: one or more fine-tuned backbones whose LoRA
adapters were merged into the weights, plus the OOF-selected decision threshold.
"""
import argparse
import os
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.data import DataLoader, Dataset

IMG_EXTS = {".jpg", ".jpeg", ".png"}
LOAD_SIZE = 224  # test images are 224x224; anything else is resized once on load


class Detector(nn.Module):
    """Same architecture as training (src/model.py), minus LoRA (already merged into weights)."""

    def __init__(self, arch, img_size, feat_layers=None):
        super().__init__()
        self.backbone = timm.create_model(arch, pretrained=False, num_classes=0, img_size=img_size)
        self.feat_layers = list(feat_layers) if feat_layers else None
        dim = self.backbone.num_features
        if self.feat_layers:  # multi-layer CLS features
            self.layer_norms = nn.ModuleList(nn.LayerNorm(dim) for _ in self.feat_layers)
            dim *= len(self.feat_layers)
        self.head = nn.Sequential(nn.Dropout(0.0), nn.Linear(dim, 1))

    def features(self, x):
        if not self.feat_layers:
            return self.backbone(x)
        b = self.backbone
        pe = b._pos_embed(b.patch_embed(x))
        if isinstance(pe, tuple):  # Eva family (DINOv3, PE): RoPE passed to every block
            x, kw = b.norm_pre(pe[0]), {"rope": pe[1]}
        else:
            x, kw = b.norm_pre(b.patch_drop(pe)), {}
        wanted, cls = set(self.feat_layers), []
        for i, blk in enumerate(b.blocks):
            x = blk(x, **kw)
            if i in wanted:
                cls.append(x[:, 0])
        return torch.cat([ln(c) for ln, c in zip(self.layer_norms, cls)], dim=1)

    def forward(self, x):
        return self.head(self.features(x)).squeeze(1)


class TestDataset(Dataset):
    def __init__(self, test_dir):
        self.paths = sorted(p for p in Path(test_dir).iterdir() if p.suffix.lower() in IMG_EXTS)

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        img = Image.open(self.paths[i]).convert("RGB")
        if img.size != (LOAD_SIZE, LOAD_SIZE):
            img = img.resize((LOAD_SIZE, LOAD_SIZE), Image.BICUBIC)
        x = torch.from_numpy(np.asarray(img, dtype=np.uint8).copy()).permute(2, 0, 1)
        return x, self.paths[i].stem


def load_members(bundle, device, dtype):
    members = []
    for m in bundle["members"]:
        net = Detector(m["arch"], m["img_size"], m.get("feat_layers"))
        net.load_state_dict({k: v.float() for k, v in m["state_dict"].items()})
        net = net.to(device=device, dtype=dtype).eval()
        mean = torch.tensor(m["mean"], device=device, dtype=dtype).view(1, 3, 1, 1)
        std = torch.tensor(m["std"], device=device, dtype=dtype).view(1, 3, 1, 1)
        members.append((net, m["img_size"], mean, std, float(m.get("weight", 1.0))))
    return members


@torch.inference_mode()
def predict(members, loader, device, dtype, tta_hflip):
    ids, probs = [], []
    wsum = sum(w for *_, w in members)
    for x, batch_ids in loader:
        x = x.to(device, non_blocking=True).to(dtype).div_(255.0)
        logit = torch.zeros(x.shape[0], device=device, dtype=torch.float32)
        for net, size, mean, std, w in members:
            xi = x if size == x.shape[-1] else F.interpolate(
                x, size=(size, size), mode="bicubic", align_corners=False, antialias=True)
            xi = (xi - mean) / std
            out = net(xi).float()
            if tta_hflip:
                out = 0.5 * (out + net(torch.flip(xi, dims=[3])).float())
            logit += w * out
        probs.append(torch.sigmoid(logit / wsum).cpu().numpy())
        ids.extend(batch_ids)
    return ids, np.concatenate(probs) if probs else np.zeros(0)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test_dir", required=True, help="Path to test images folder")
    parser.add_argument("--weights", required=True, help="Path to model weight file")
    parser.add_argument("--output", default="predictions.csv", help="Output CSV path")
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--num_workers", type=int, default=min(8, os.cpu_count() or 1))
    args = parser.parse_args()

    torch.backends.cudnn.benchmark = False
    device = "cuda" if torch.cuda.is_available() else "cpu"
    dtype = torch.float16 if device == "cuda" else torch.float32

    bundle = torch.load(args.weights, map_location="cpu", weights_only=False)
    members = load_members(bundle, device, dtype)
    threshold = float(bundle.get("threshold", 0.5))
    print(f"device={device} members={[m['arch'] for m in bundle['members']]} threshold={threshold:.4f}")
    del bundle

    ds = TestDataset(args.test_dir)
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=device == "cuda")
    ids, conf = predict(members, loader, device, dtype, tta_hflip=True)

    conf = np.clip(conf, 0.0, 1.0)
    out = pd.DataFrame({"image_id": ids, "label": (conf >= threshold).astype(int),
                        "confidence": np.round(conf, 6)})
    assert len(out) == len(ds) and out.image_id.is_unique
    out.to_csv(args.output, index=False)
    print(f"wrote {len(out)} predictions -> {args.output} (predicted fake: {out.label.mean():.3f})")


if __name__ == "__main__":
    main()
