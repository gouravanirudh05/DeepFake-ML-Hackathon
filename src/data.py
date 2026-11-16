"""Datasets: augmented training set and deterministic (optionally degraded) eval set."""
from __future__ import annotations

import random

import pandas as pd
import torch
from PIL import Image
from torch.utils.data import Dataset
from torchvision.transforms import functional as TF

from .degradations import VARIANTS, random_degrade
from .sbi import self_blend


def to_tensor(img: Image.Image, img_size: int, mean, std) -> torch.Tensor:
    if img.size != (img_size, img_size):
        img = img.resize((img_size, img_size), Image.BICUBIC)
    return TF.normalize(TF.to_tensor(img), mean, std)


class TrainDataset(Dataset):
    def __init__(self, df: pd.DataFrame, img_size: int, mean, std, aug_strength: float = 1.0,
                 sbi_prob: float = 0.0):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype("float32").tolist()
        self.img_size, self.mean, self.std = img_size, mean, std
        self.aug_strength, self.sbi_prob = aug_strength, sbi_prob

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        # torch.initial_seed() differs per worker and per epoch -> fresh but reproducible aug.
        rng = random.Random((torch.initial_seed() + i * 7919) % (2**63))
        img = Image.open(self.paths[i]).convert("RGB")
        label = self.labels[i]
        if label == 0 and self.sbi_prob > 0 and rng.random() < self.sbi_prob:
            img, label = self_blend(img, rng), 1.0  # self-blended pseudo-fake (src/sbi.py)
        if rng.random() < 0.5:
            img = img.transpose(Image.FLIP_LEFT_RIGHT)
        if rng.random() < 0.5:  # mild random-resized crop (faces are already centred)
            w, h = img.size
            s = rng.uniform(0.85, 1.0)
            cw, ch = int(w * s), int(h * s)
            x, y = rng.randint(0, w - cw), rng.randint(0, h - ch)
            img = img.crop((x, y, x + cw, y + ch))
        img = random_degrade(img, rng, self.aug_strength)
        return to_tensor(img, self.img_size, self.mean, self.std), torch.tensor(label)


class EvalDataset(Dataset):
    """Deterministic: the same image always receives the same degradation."""

    def __init__(self, df: pd.DataFrame, img_size: int, mean, std, variant: str = "clean",
                 seed: int = 0):
        self.paths = df["path"].tolist()
        self.labels = df["label"].astype("float32").tolist()
        self.img_size, self.mean, self.std = img_size, mean, std
        self.variant, self.seed = variant, seed

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        rng = random.Random(self.seed * 100003 + i)
        img = Image.open(self.paths[i]).convert("RGB")
        if self.variant == "mixed":  # one random named variant per image
            name = rng.choice([k for k in VARIANTS if k != "clean"])
        else:
            name = self.variant
        img = VARIANTS[name](img, rng)
        return to_tensor(img, self.img_size, self.mean, self.std), torch.tensor(self.labels[i])
