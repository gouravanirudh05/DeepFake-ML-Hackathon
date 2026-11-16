"""Pack trained members (folds and/or architectures) into ONE weights file for inference_script.py.

The decision threshold is chosen on out-of-fold predictions (Youden's J), never on test data.

    python build_bundle.py --members "outputs/clip_l14_lora/fold*/member.pt" \
                           --oof "outputs/clip_l14_lora/fold*/oof.csv" --out model_weights.pt
"""
from __future__ import annotations

import argparse
import glob

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score, roc_auc_score, roc_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--members", nargs="+", required=True, help="member.pt paths or globs")
    ap.add_argument("--weights", nargs="*", type=float, default=None, help="per-member weights")
    ap.add_argument("--oof", nargs="*", default=[], help="oof.csv paths or globs (for threshold)")
    ap.add_argument("--out", default="model_weights.pt")
    args = ap.parse_args()

    paths = sorted(p for g in args.members for p in glob.glob(g))
    assert paths, "no member files matched"
    w = args.weights or [1.0] * len(paths)
    assert len(w) == len(paths), "need one weight per member"

    members = []
    for p, wi in zip(paths, w):
        m = torch.load(p, map_location="cpu", weights_only=False)
        m["weight"] = wi
        members.append(m)
        print(f"+ {p}  arch={m['arch']}  weight={wi}  metrics={m.get('metrics', {})}")

    threshold, oof_report = 0.5, {}
    oof_paths = sorted(p for g in args.oof for p in glob.glob(g))
    if oof_paths:
        oof = pd.concat([pd.read_csv(p) for p in oof_paths])
        oof = oof.groupby("image_id").agg(label=("label", "first"), confidence=("confidence", "mean"))
        y, p = oof.label.values, oof.confidence.values
        fpr, tpr, thr = roc_curve(y, p)
        threshold = float(np.clip(thr[np.argmax(tpr - fpr)], 0.0, 1.0))
        oof_report = {"oof_auc": float(roc_auc_score(y, p)),
                      "oof_f1": float(f1_score(y, p >= threshold)),
                      "oof_acc": float(((p >= threshold) == y).mean()),
                      "threshold": threshold, "n": int(len(y))}
        print("OOF:", {k: round(v, 4) if isinstance(v, float) else v for k, v in oof_report.items()})

    torch.save({"format": "dfbundle-v1", "members": members, "threshold": threshold,
                "tta_hflip": True, "oof": oof_report}, args.out)
    print(f"wrote {args.out} ({len(members)} members, threshold={threshold:.4f})")


if __name__ == "__main__":
    main()
