"""Data audit + leakage-free fold assignment.

1. Perceptual-hash (pHash) every TRAIN image (test images are never touched).
2. Link images whose hashes are within `--max_dist` bits (near-duplicates,
   e.g. a real photo and its face-swapped version share background/framing).
3. Union-find -> groups; StratifiedGroupKFold so a group never spans folds.
4. Report exact duplicates and groups containing BOTH labels (label-noise hint).

Usage:
    python audit_and_split.py --data_dir Data --out splits/folds.csv
"""
from __future__ import annotations

import argparse
import hashlib
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image
from sklearn.model_selection import StratifiedGroupKFold


def _dct_matrix(n: int) -> np.ndarray:
    k = np.arange(n)[:, None]
    m = np.cos(np.pi * (2 * np.arange(n)[None, :] + 1) * k / (2 * n))
    m[0] *= 1 / np.sqrt(2)
    return m * np.sqrt(2 / n)


_D32 = _dct_matrix(32)


def phash(path: Path) -> np.uint64:
    g = np.asarray(Image.open(path).convert("L").resize((32, 32), Image.LANCZOS), np.float64)
    low = (_D32 @ g @ _D32.T)[:8, :8].flatten()
    bits = low > np.median(low[1:])
    return np.uint64(int("".join("1" if b else "0" for b in bits), 2))


class UnionFind:
    def __init__(self, n):
        self.p = list(range(n))

    def find(self, a):
        while self.p[a] != a:
            self.p[a] = self.p[self.p[a]]
            a = self.p[a]
        return a

    def union(self, a, b):
        self.p[self.find(a)] = self.find(b)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data_dir", default="Data")
    ap.add_argument("--out", default="splits/folds.csv")
    ap.add_argument("--n_folds", type=int, default=5)
    ap.add_argument("--max_dist", type=int, default=6, help="pHash Hamming radius for near-dups")
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    data = Path(args.data_dir)
    meta = pd.read_csv(data / "train_metadata.csv")
    sub = {0: "real", 1: "fake"}
    meta["path"] = [str(data / "train" / sub[l] / f"{i}.jpg") for i, l in zip(meta.image_id, meta.label)]
    missing = [p for p in meta.path if not Path(p).exists()]
    assert not missing, f"{len(missing)} metadata rows have no image, e.g. {missing[:3]}"
    print(f"{len(meta)} images | label counts: {meta.label.value_counts().to_dict()}")

    meta["md5"] = [hashlib.md5(Path(p).read_bytes()).hexdigest() for p in meta.path]
    hashes = np.array([phash(Path(p)) for p in meta.path], dtype=np.uint64)

    n = len(meta)
    uf = UnionFind(n)
    for md5, idx in meta.groupby("md5").groups.items():
        idx = list(idx)
        for j in idx[1:]:
            uf.union(idx[0], j)
    n_pairs, dist_hist = 0, np.zeros(65, int)
    for i in range(n):
        d = np.bitwise_count(hashes[i + 1:] ^ hashes[i]).astype(int)
        dist_hist += np.bincount(d, minlength=65)
        for j in np.nonzero(d <= args.max_dist)[0]:
            uf.union(i, i + 1 + j)
            n_pairs += 1
    meta["group"] = [uf.find(i) for i in range(n)]

    exact = meta.md5.duplicated(keep=False).sum()
    gsize = meta.group.map(meta.group.value_counts())
    mixed = meta.groupby("group").label.nunique()
    mixed_groups = mixed[mixed > 1].index
    print(f"exact-duplicate files: {exact}")
    print(f"near-dup pairs (pHash dist <= {args.max_dist}): {n_pairs}")
    print("pHash distance histogram (0..12):", dist_hist[:13].tolist())
    print(f"groups: {meta.group.nunique()} | images in multi-image groups: {(gsize > 1).sum()}")
    print(f"groups containing BOTH real and fake: {len(mixed_groups)} "
          f"({meta.group.isin(mixed_groups).sum()} images)")

    sgkf = StratifiedGroupKFold(n_splits=args.n_folds, shuffle=True, random_state=args.seed)
    meta["fold"] = -1
    for k, (_, va) in enumerate(sgkf.split(meta, meta.label, meta.group)):
        meta.loc[meta.index[va], "fold"] = k
    print(meta.groupby("fold").label.agg(["count", "mean"]).round(3).to_string())

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    meta["phash"] = [f"{int(h):016x}" for h in hashes]
    meta[["image_id", "label", "fold", "group", "phash", "md5", "path"]].to_csv(out, index=False)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
