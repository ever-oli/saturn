"""Place cube + Latin-cross net onto blank-tee canvases (front / back / combined)."""

from __future__ import annotations

from dataclasses import dataclass

from PIL import Image, ImageFilter

from config import MOCKUP_BACKGROUND
from geometry.tees import load_tee


@dataclass(frozen=True)
class ShirtMockup:
    combined: Image.Image
    front: Image.Image
    back: Image.Image


def composite_shirts(
    cube: Image.Image,
    net: Image.Image,
    background: tuple[int, int, int] = MOCKUP_BACKGROUND,
) -> ShirtMockup:
    """Chest cube on the front tee, unfolded net on the back, side-by-side sheet."""
    tee_front = load_tee("front")
    tee_back = load_tee("back")

    front = _place_on_tee(
        tee_front,
        _crop_alpha(cube, pad=12),
        width_ratio=0.36,
        center_xy=(0.50, 0.44),
    )
    back = _place_on_tee(
        tee_back,
        _crop_alpha(net, pad=8),
        width_ratio=0.40,
        center_xy=(0.50, 0.50),
    )
    combined = _side_by_side(front, back, background)
    return ShirtMockup(combined=combined, front=front, back=back)


def _crop_alpha(image: Image.Image, pad: int = 8) -> Image.Image:
    """Trim transparent margins so placement ratios use the graphic, not padding."""
    rgba = image.convert("RGBA")
    bbox = rgba.getbbox()
    if bbox is None:
        return rgba
    x0, y0, x1, y1 = bbox
    x0 = max(0, x0 - pad)
    y0 = max(0, y0 - pad)
    x1 = min(rgba.width, x1 + pad)
    y1 = min(rgba.height, y1 + pad)
    return rgba.crop((x0, y0, x1, y1))


def _place_on_tee(
    tee: Image.Image,
    graphic: Image.Image,
    width_ratio: float,
    center_xy: tuple[float, float],
) -> Image.Image:
    shirt = tee.convert("RGBA")
    tw, th = shirt.size
    gw, gh = graphic.size
    target_w = max(1, int(tw * width_ratio))
    target_h = max(1, int(target_w * gh / max(gw, 1)))
    scaled = graphic.convert("RGBA").resize((target_w, target_h), Image.Resampling.LANCZOS)
    # Soft contact shadow so the print sits on the fabric.
    shadow = _drop_shadow(scaled, offset=(0, max(4, target_h // 40)))
    cx, cy = int(tw * center_xy[0]), int(th * center_xy[1])
    box = (cx - target_w // 2, cy - target_h // 2)
    layer = Image.new("RGBA", shirt.size, (0, 0, 0, 0))
    # Shadow canvas is padded by 8px on each side (see _drop_shadow).
    layer.paste(shadow, (box[0] - 8, box[1] - 8), shadow)
    layer.paste(scaled, box, scaled)
    return Image.alpha_composite(shirt, layer)


def _drop_shadow(graphic: Image.Image, offset: tuple[int, int]) -> Image.Image:
    alpha = graphic.split()[-1]
    shadow = Image.new("RGBA", graphic.size, (0, 0, 0, 0))
    shadow.putalpha(alpha.point(lambda a: int(a * 0.22)))
    shadow = shadow.filter(ImageFilter.GaussianBlur(radius=8))
    padded = Image.new(
        "RGBA",
        (graphic.size[0] + abs(offset[0]) + 16, graphic.size[1] + abs(offset[1]) + 16),
        (0, 0, 0, 0),
    )
    padded.paste(shadow, (8 + offset[0], 8 + offset[1]), shadow)
    return padded


def _side_by_side(
    front: Image.Image,
    back: Image.Image,
    background: tuple[int, int, int],
    pad: int = 72,
    gap: int = 48,
) -> Image.Image:
    # Match the height of the two tees, keep aspect.
    h = max(front.height, back.height)
    front_r = _fit_height(front, h)
    back_r = _fit_height(back, h)
    width = pad * 2 + front_r.width + gap + back_r.width
    height = pad * 2 + h
    sheet = Image.new("RGB", (width, height), background)
    sheet.paste(front_r, (pad, pad), front_r)
    sheet.paste(back_r, (pad + front_r.width + gap, pad), back_r)
    return sheet


def _fit_height(image: Image.Image, height: int) -> Image.Image:
    if image.height == height:
        return image
    w = max(1, int(image.width * height / image.height))
    return image.resize((w, height), Image.Resampling.LANCZOS)
