"""Derive six cube-face textures from a single uploaded image (PIL, CPU).

v1 does not run multi-view diffusion. Faces come from crop / tile / wrap:

- ``single`` — center-square crop, same texture on all six faces (starfield-style).
- ``wrap`` — treat the source as a horizontal equator so a band (e.g. gold trim)
  continues LEFT → FRONT → RIGHT → BACK; top/bottom are upper/lower crops.
- ``grid`` — split the source into a 2×3 crop grid, one cell per face.
- ``auto`` — ``wrap`` when the image is wide (aspect ≥ 1.4), else ``single``.
"""

from __future__ import annotations

from PIL import Image, ImageEnhance

from geometry.types import CubeFaces, center_square, resize_square, to_rgb

VALID_MODES = ("single", "wrap", "grid", "auto")


def derive_faces(
    image: Image.Image,
    panel_size: int,
    mode: str = "single",
) -> CubeFaces:
    rgb = to_rgb(image)
    resolved = _resolve_mode(rgb, mode)
    if resolved == "wrap":
        return _wrap_faces(rgb, panel_size)
    if resolved == "grid":
        return _grid_faces(rgb, panel_size)
    return _single_faces(rgb, panel_size)


def _resolve_mode(image: Image.Image, mode: str) -> str:
    key = (mode or "single").strip().lower()
    if key not in VALID_MODES:
        key = "single"
    if key == "auto":
        w, h = image.size
        return "wrap" if h > 0 and (w / h) >= 1.4 else "single"
    return key


def _single_faces(image: Image.Image, size: int) -> CubeFaces:
    panel = resize_square(center_square(image), size)
    return CubeFaces(
        top=_tweak(panel, brightness=1.06, contrast=1.02),
        bottom=_tweak(panel, brightness=0.78, contrast=0.95),
        front=panel,
        back=_tweak(panel, brightness=0.9),
        left=_tweak(panel, brightness=0.88),
        right=_tweak(panel, brightness=0.94),
    )


def _wrap_faces(image: Image.Image, size: int) -> CubeFaces:
    """Map a horizontal band around the four side faces for trim continuity."""
    equator = _equator_strip(image, size)  # LEFT | FRONT | RIGHT | BACK
    left = equator.crop((0, 0, size, size))
    front = equator.crop((size, 0, 2 * size, size))
    right = equator.crop((2 * size, 0, 3 * size, size))
    back = equator.crop((3 * size, 0, 4 * size, size))
    # Roof / floor stay off the equator so a wrap band does not stripe the top face.
    top = resize_square(center_square(_horizontal_band(image, 0.0, 0.16)), size)
    bottom = resize_square(center_square(_horizontal_band(image, 0.84, 1.0)), size)
    return CubeFaces(
        top=_tweak(top, brightness=1.05),
        bottom=_tweak(bottom, brightness=0.8),
        front=front,
        back=back,
        left=left,
        right=right,
    )


def _grid_faces(image: Image.Image, size: int) -> CubeFaces:
    w, h = image.size
    cols, rows = 3, 2
    cw, rh = w / cols, h / rows
    cells: list[Image.Image] = []
    for r in range(rows):
        for c in range(cols):
            box = (
                int(c * cw),
                int(r * rh),
                int((c + 1) * cw),
                int((r + 1) * rh),
            )
            cells.append(resize_square(center_square(image.crop(box)), size))
    # 2×3 reading order → cube faces
    top, bottom, front = cells[0], cells[1], cells[2]
    back, left, right = cells[3], cells[4], cells[5]
    return CubeFaces(
        top=top,
        bottom=bottom,
        front=front,
        back=back,
        left=left,
        right=right,
    )


def _equator_strip(image: Image.Image, size: int) -> Image.Image:
    """Build a 4×1 panorama used as LEFT|FRONT|RIGHT|BACK."""
    mid = _horizontal_band(image, 0.15, 0.85)
    mw, mh = mid.size
    new_w = max(1, round(mw * size / max(mh, 1)))
    scaled = mid.resize((new_w, size), Image.Resampling.LANCZOS)
    target = 4 * size
    if scaled.width >= target:
        x0 = (scaled.width - target) // 2
        return scaled.crop((x0, 0, x0 + target, size))
    # Square-ish sources: repeat the center square around the cube.
    tile = resize_square(center_square(image), size)
    canvas = Image.new("RGB", (target, size))
    for i in range(4):
        canvas.paste(tile, (i * size, 0))
    return canvas


def _horizontal_band(image: Image.Image, y0: float, y1: float) -> Image.Image:
    w, h = image.size
    top = max(0, min(h - 1, int(h * y0)))
    bot = max(top + 1, min(h, int(h * y1)))
    return image.crop((0, top, w, bot))


def _tweak(
    image: Image.Image,
    brightness: float = 1.0,
    contrast: float = 1.0,
) -> Image.Image:
    out = image
    if brightness != 1.0:
        out = ImageEnhance.Brightness(out).enhance(brightness)
    if contrast != 1.0:
        out = ImageEnhance.Contrast(out).enhance(contrast)
    return out
