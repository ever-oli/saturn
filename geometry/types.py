"""Shared types and small image helpers."""

from __future__ import annotations

from typing import TypedDict

import numpy as np
from PIL import Image

from config import FACE_BACKGROUND


class CubeFaces(TypedDict):
    """Six cube faces. Each image is a square RGB panel, +U right, +V down."""

    top: Image.Image
    bottom: Image.Image
    front: Image.Image
    back: Image.Image
    left: Image.Image
    right: Image.Image


FACE_ORDER: tuple[str, ...] = ("top", "bottom", "front", "back", "left", "right")


def flatten_rgb(
    image: Image.Image,
    background: tuple[int, int, int] = FACE_BACKGROUND,
) -> Image.Image:
    """Return an opaque RGB image. Alpha is composited onto ``background``.

    Never drops the alpha channel (which leaves leftover RGB in transparent
    pixels) and never bakes a checkerboard — Gradio shows unresolved alpha as
    a gray/white grid.
    """
    if image.mode == "RGB":
        return image
    if image.mode == "P":
        image = image.convert("RGBA") if "transparency" in image.info else image.convert("RGB")
        if image.mode == "RGB":
            return image
    if image.mode in {"RGBA", "LA"}:
        rgba = image.convert("RGBA")
        bg = Image.new("RGB", rgba.size, background)
        bg.paste(rgba, mask=rgba.split()[-1])
        return bg
    return image.convert("RGB")


def to_rgb(image: Image.Image) -> Image.Image:
    """Back-compat alias: flatten onto the Black Cube fill, not white."""
    return flatten_rgb(image, FACE_BACKGROUND)


def as_opaque_rgba(
    image: Image.Image,
    background: tuple[int, int, int] = FACE_BACKGROUND,
) -> Image.Image:
    """RGB flatten then restore a fully-opaque alpha channel."""
    rgb = flatten_rgb(image, background)
    return rgb.convert("RGBA")


def has_useful_alpha(image: Image.Image) -> bool:
    """True when the image has a real cutout (opaque subject + transparent field)."""
    if image.mode == "P" and "transparency" in image.info:
        image = image.convert("RGBA")
    if image.mode not in {"RGBA", "LA"}:
        return False
    alpha = np.asarray(image.split()[-1], dtype=np.uint8)
    transparent = float((alpha < 16).mean())
    opaque = float((alpha > 240).mean())
    return transparent > 0.08 and opaque > 0.12


def center_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)


def checker_score(image: Image.Image) -> float:
    """Heuristic for a baked transparency-checker (alternating gray tiles).

    Returns a score in ``[0, 1]``. Real fabric/starfields stay low; an 8px
    gray/white checker that Gradio uses for unresolved alpha scores high.
    """
    rgb = np.asarray(image.convert("RGB"), dtype=np.int16)
    if rgb.size == 0:
        return 0.0
    gray = rgb.mean(axis=2)
    sat = rgb.max(axis=2) - rgb.min(axis=2)
    # Classic checkers live in desaturated mid-grays (not near-black cloth).
    mask = (sat < 12) & (gray > 130) & (gray < 245)
    if float(mask.mean()) < 0.04:
        return 0.0
    ys, xs = np.nonzero(mask)
    if xs.size < 128:
        return 0.0
    parity = ((xs // 8) + (ys // 8)) % 2
    luma = gray[mask]
    even = luma[parity == 0]
    odd = luma[parity == 1]
    if even.size < 32 or odd.size < 32:
        return 0.0
    # Two parity populations with a clear luma split ⇒ checker.
    split = abs(float(even.mean()) - float(odd.mean())) / 255.0
    return float(split * mask.mean())
