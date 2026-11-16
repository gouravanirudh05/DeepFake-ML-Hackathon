"""Image degradations shared by training augmentation and the robustness harness.

The hidden test "variants" are undisclosed, so we train with (and validate
against) the common real-world corruptions: re-compression, blur, sensor noise,
resolution loss, colour shifts, grayscale and occlusion. The training images
already contain noise / cutout / colour-block corruptions in BOTH classes, so
these are label-independent nuisances the model must learn to ignore.
"""
from __future__ import annotations

import io
import random

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter


def jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=int(quality))
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def blur(img: Image.Image, sigma: float) -> Image.Image:
    return img.filter(ImageFilter.GaussianBlur(radius=sigma))


def noise(img: Image.Image, std: float, rng: random.Random) -> Image.Image:
    arr = np.asarray(img, dtype=np.float32)
    g = np.random.default_rng(rng.randrange(2**31)).normal(0.0, std, arr.shape)
    return Image.fromarray(np.clip(arr + g, 0, 255).astype(np.uint8))


def downscale(img: Image.Image, factor: float) -> Image.Image:
    w, h = img.size
    small = img.resize((max(8, int(w * factor)), max(8, int(h * factor))), Image.BILINEAR)
    return small.resize((w, h), Image.BILINEAR)


def color(img: Image.Image, b: float, c: float, s: float) -> Image.Image:
    img = ImageEnhance.Brightness(img).enhance(b)
    img = ImageEnhance.Contrast(img).enhance(c)
    return ImageEnhance.Color(img).enhance(s)


def grayscale(img: Image.Image) -> Image.Image:
    return img.convert("L").convert("RGB")


def cutout(img: Image.Image, frac: float, rng: random.Random) -> Image.Image:
    w, h = img.size
    cw, ch = int(w * frac), int(h * frac)
    x, y = rng.randrange(0, w - cw + 1), rng.randrange(0, h - ch + 1)
    arr = np.array(img)
    arr[y:y + ch, x:x + cw] = rng.randrange(0, 256)
    return Image.fromarray(arr)


# ── Training-time random corruption ───────────────────────────────────────────
def random_degrade(img: Image.Image, rng: random.Random, strength: float = 1.0) -> Image.Image:
    """Apply a random subset of corruptions in random order."""
    ops = []
    if rng.random() < 0.30 * strength:
        ops.append(lambda im: jpeg(im, rng.randint(30, 95)))
    if rng.random() < 0.20 * strength:
        ops.append(lambda im: blur(im, rng.uniform(0.2, 2.0)))
    if rng.random() < 0.20 * strength:
        ops.append(lambda im: noise(im, rng.uniform(2.0, 15.0), rng))
    if rng.random() < 0.20 * strength:
        ops.append(lambda im: downscale(im, rng.uniform(0.35, 0.9)))
    if rng.random() < 0.30 * strength:
        ops.append(lambda im: color(im, rng.uniform(0.7, 1.3), rng.uniform(0.7, 1.3),
                                    rng.uniform(0.6, 1.4)))
    if rng.random() < 0.05 * strength:
        ops.append(grayscale)
    if rng.random() < 0.15 * strength:
        ops.append(lambda im: cutout(im, rng.uniform(0.1, 0.3), rng))
    rng.shuffle(ops)
    for op in ops:
        img = op(img)
    return img


# ── Fixed variants for the robustness harness (proxy for hidden test variants) ─
VARIANTS = {
    "clean":      lambda im, rng: im,
    "jpeg70":     lambda im, rng: jpeg(im, 70),
    "jpeg40":     lambda im, rng: jpeg(im, 40),
    "blur1.5":    lambda im, rng: blur(im, 1.5),
    "noise10":    lambda im, rng: noise(im, 10.0, rng),
    "down0.5":    lambda im, rng: downscale(im, 0.5),
    "color":      lambda im, rng: color(im, 1.2, 0.8, 1.3),
    "gray":       lambda im, rng: grayscale(im),
    "cutout0.2":  lambda im, rng: cutout(im, 0.2, rng),
    "combo":      lambda im, rng: jpeg(downscale(blur(im, 0.8), 0.6), 60),
}
