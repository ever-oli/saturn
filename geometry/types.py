"""Shared types and small image helpers."""

from __future__ import annotations

from typing import TypedDict

from PIL import Image


class CubeFaces(TypedDict):
    """Six cube faces. Each image is a square RGB panel, +U right, +V down."""

    top: Image.Image
    bottom: Image.Image
    front: Image.Image
    back: Image.Image
    left: Image.Image
    right: Image.Image


FACE_ORDER: tuple[str, ...] = ("top", "bottom", "front", "back", "left", "right")


def to_rgb(image: Image.Image) -> Image.Image:
    if image.mode == "RGB":
        return image
    if image.mode in {"RGBA", "LA"}:
        bg = Image.new("RGB", image.size, (255, 255, 255))
        bg.paste(image, mask=image.split()[-1])
        return bg
    return image.convert("RGB")


def center_square(image: Image.Image) -> Image.Image:
    w, h = image.size
    side = min(w, h)
    left = (w - side) // 2
    top = (h - side) // 2
    return image.crop((left, top, left + side, top + side))


def resize_square(image: Image.Image, size: int) -> Image.Image:
    return image.resize((size, size), Image.Resampling.LANCZOS)
