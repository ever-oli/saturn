"""CPU helpers for pairing Flux outputs (no GPU)."""

from __future__ import annotations

from PIL import Image

from config import MOCKUP_BACKGROUND


def side_by_side(left: Image.Image, right: Image.Image, gap: int = 48) -> Image.Image:
    """Place two RGB mockups on a studio-gray canvas."""
    a = left.convert("RGB")
    b = right.convert("RGB")
    h = max(a.height, b.height)
    if a.height != h:
        a = a.resize((round(a.width * h / a.height), h), Image.Resampling.LANCZOS)
    if b.height != h:
        b = b.resize((round(b.width * h / b.height), h), Image.Resampling.LANCZOS)
    canvas = Image.new("RGB", (a.width + b.width + gap, h), MOCKUP_BACKGROUND)
    canvas.paste(a, (0, 0))
    canvas.paste(b, (a.width + gap, 0))
    return canvas
