"""Self-Blended Images (Shiohara & Yamasaki, CVPR 2022), adapted to 224x224 face crops.

A *real* training image is blended with a slightly altered copy of itself inside a soft, irregular
face-shaped mask, then labelled fake. This synthesises the generic traces of face swapping (blending
boundary, colour / sharpness mismatch, small misalignment) without any external data or generator,
which pushes the detector toward method-agnostic cues and improves generalisation to unseen
deepfake methods. Uses only the provided real images, so it is allowed by the rules.

No landmark detector is available offline, so the mask is a randomly sized, jittered and
elastically deformed ellipse around the (centred) face instead of a landmark convex hull.
"""
from __future__ import annotations

import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def _source_aug(img: Image.Image, rng: random.Random) -> Image.Image:
    """Make the 'source' face differ from the 'target' in colour, sharpness and position."""
    if rng.random() < 0.7:
        img = ImageEnhance.Brightness(img).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.7:
        img = ImageEnhance.Contrast(img).enhance(rng.uniform(0.85, 1.15))
    if rng.random() < 0.5:
        img = ImageEnhance.Color(img).enhance(rng.uniform(0.8, 1.2))
    if rng.random() < 0.5:
        a = np.asarray(img, dtype=np.float32) + np.array([rng.uniform(-12, 12) for _ in range(3)])
        img = Image.fromarray(np.clip(a, 0, 255).astype(np.uint8))
    r = rng.random()
    if r < 0.25:
        img = img.filter(ImageFilter.GaussianBlur(rng.uniform(0.4, 1.2)))
    elif r < 0.5:
        img = img.filter(ImageFilter.UnsharpMask(radius=2, percent=rng.randint(60, 160)))
    elif r < 0.65:
        w, h = img.size
        s = rng.uniform(0.4, 0.75)
        img = img.resize((int(w * s), int(h * s)), Image.BILINEAR).resize((w, h), Image.BILINEAR)
    # small scale + translation (landmark misalignment of a swapped face)
    w, h = img.size
    s = rng.uniform(0.95, 1.05)
    tx, ty = rng.uniform(-0.03, 0.03) * w, rng.uniform(-0.03, 0.03) * h
    cx, cy = w / 2, h / 2
    return img.transform((w, h), Image.AFFINE,
                         (1 / s, 0, cx - (cx + tx) / s, 0, 1 / s, cy - (cy + ty) / s),
                         resample=Image.BILINEAR)


def _mask(w: int, h: int, rng: random.Random) -> np.ndarray:
    nrng = np.random.default_rng(rng.getrandbits(32))
    cx, cy = w * rng.uniform(0.42, 0.58), h * rng.uniform(0.42, 0.60)
    ax, ay = w * rng.uniform(0.16, 0.30), h * rng.uniform(0.22, 0.36)
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float32)
    d = ((xx - cx) / ax) ** 2 + ((yy - cy) / ay) ** 2
    # elastic deformation: low-frequency noise added to the ellipse distance field
    g = nrng.normal(0, rng.uniform(0.1, 0.35), (6, 6)).astype(np.float32)
    noise = np.asarray(Image.fromarray(g, mode="F").resize((w, h), Image.BICUBIC))
    m = (d + noise < 1.0).astype(np.float32)
    m = np.asarray(Image.fromarray((m * 255).astype(np.uint8)).filter(
        ImageFilter.GaussianBlur(rng.uniform(2, 10))), dtype=np.float32) / 255.0
    return m * rng.choice([0.5, 0.75, 1.0, 1.0, 1.0])  # blend ratio


def self_blend(img: Image.Image, rng: random.Random) -> Image.Image:
    target = np.asarray(img, dtype=np.float32)
    source = np.asarray(_source_aug(img, rng), dtype=np.float32)
    m = _mask(img.width, img.height, rng)[..., None]
    return Image.fromarray(np.clip(source * m + target * (1 - m), 0, 255).astype(np.uint8))
