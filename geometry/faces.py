"""Derive six cube-face textures from a single uploaded image (PIL, CPU).

v1 does not run multi-view diffusion. Faces come from crop / wrap / emblem:

- ``single`` — treat the upload as albedo/material. Center-square crop, same
  cloth on every face, with distinct per-face lighting so it reads as a cube
  rather than six identical stickers.
- ``wrap`` — horizontal equator around LEFT → FRONT → RIGHT → BACK so a trim
  (gold kiswah, horizon) stays continuous; top/bottom are fabric crops.
- ``grid`` — split the source into a 2×3 crop grid, one cell per face.
- ``emblem`` — subject-centered print: optional rembg / existing alpha, then
  place the subject on solid Black Cube fill (not edge-to-edge photos).
- ``auto`` — transparency/cutout → emblem; wide or gold/chroma band → wrap;
  else single material.
"""

from __future__ import annotations

import numpy as np
from PIL import Image, ImageEnhance, ImageOps

from config import FACE_BACKGROUND
from geometry.bg import try_remove_background
from geometry.types import (
    CubeFaces,
    center_square,
    flatten_rgb,
    has_useful_alpha,
    resize_square,
)

VALID_MODES = ("single", "wrap", "grid", "emblem", "auto")
WRAP_ASPECT = 1.4
_MODE_ALIASES = {
    "single material": "single",
    "wrap band": "wrap",
    "emblem on black": "emblem",
    "grid 2×3": "grid",
    "grid 2x3": "grid",
}

# Per-face lighting (isometric three-quarter: light from above-right).
_FACE_LIGHT = {
    "top": {"brightness": 1.14, "contrast": 1.04, "color": 0.94, "hue": 2.0},
    "bottom": {"brightness": 0.66, "contrast": 0.94, "color": 0.86, "hue": -4.0},
    "front": {"brightness": 0.94, "contrast": 1.02, "color": 1.00, "hue": 0.0},
    "back": {"brightness": 0.80, "contrast": 0.98, "color": 0.95, "hue": -2.0},
    "left": {"brightness": 0.72, "contrast": 0.97, "color": 0.90, "hue": -8.0},
    "right": {"brightness": 1.04, "contrast": 1.02, "color": 1.03, "hue": 6.0},
}


def derive_faces(
    image: Image.Image,
    panel_size: int,
    mode: str = "auto",
    remove_bg: bool = False,
) -> tuple[CubeFaces, str, bool]:
    """Return ``(faces, resolved_mode, removed_bg)``. Faces are opaque RGB."""
    resolved = resolve_face_mode(image, mode)
    src = image
    removed = False
    if remove_bg and resolved == "emblem":
        src, removed = try_remove_background(image)

    if resolved == "emblem":
        faces = _emblem_faces(src, panel_size)
    elif resolved == "wrap":
        faces = _wrap_faces(flatten_rgb(src), panel_size)
    elif resolved == "grid":
        faces = _grid_faces(flatten_rgb(src), panel_size)
    else:
        faces = _single_faces(flatten_rgb(src), panel_size)
    return _ensure_opaque_faces(faces), resolved, removed


def resolve_face_mode(image: Image.Image, mode: str) -> str:
    key = (mode or "auto").strip().lower()
    key = _MODE_ALIASES.get(key, key)
    if key not in VALID_MODES:
        key = "auto"
    if key != "auto":
        return key
    if has_useful_alpha(image):
        return "emblem"
    w, h = image.size
    if h > 0 and (w / h) >= WRAP_ASPECT:
        return "wrap"
    if find_horizontal_band(image) is not None:
        return "wrap"
    if _looks_like_cutout(image):
        return "emblem"
    return "single"


def _ensure_opaque_faces(faces: CubeFaces) -> CubeFaces:
    return CubeFaces(
        top=flatten_rgb(faces["top"]),
        bottom=flatten_rgb(faces["bottom"]),
        front=flatten_rgb(faces["front"]),
        back=flatten_rgb(faces["back"]),
        left=flatten_rgb(faces["left"]),
        right=flatten_rgb(faces["right"]),
    )


def _single_faces(image: Image.Image, size: int) -> CubeFaces:
    """Same albedo on all faces, lit as a cube — not six perspective stickers."""
    material = resize_square(center_square(flatten_rgb(image)), size)
    return CubeFaces(
        top=_finish_face(material, "top"),
        bottom=_finish_face(material, "bottom"),
        front=_finish_face(material, "front"),
        back=_finish_face(material, "back"),
        left=_finish_face(material, "left"),
        right=_finish_face(material, "right"),
    )


def _wrap_faces(image: Image.Image, size: int) -> CubeFaces:
    """Map a horizontal band around the four side faces for trim continuity."""
    rgb = flatten_rgb(image)
    band = find_horizontal_band(rgb)
    if band is not None:
        lo, hi = band
        cy = (lo + hi) / 2.0
        y0 = max(0.0, cy - 0.38)
        y1 = min(1.0, cy + 0.42)
        top_hi = max(0.06, lo * 0.9)
        bot_lo = min(0.94, hi + 0.04)
    else:
        y0, y1 = 0.12, 0.88
        top_hi, bot_lo = 0.16, 0.84

    equator = _equator_strip(rgb, size, y0, y1)
    left = equator.crop((0, 0, size, size))
    front = equator.crop((size, 0, 2 * size, size))
    right = equator.crop((2 * size, 0, 3 * size, size))
    back = equator.crop((3 * size, 0, 4 * size, size))
    # If panning collapsed to one tile, mirror the back so the net isn't four clones.
    if np.array_equal(np.asarray(left), np.asarray(back)):
        back = ImageOps.mirror(left)

    top = _fabric_face(rgb, size, 0.0, top_hi)
    bottom = _fabric_face(rgb, size, bot_lo, 1.0)
    return CubeFaces(
        top=_finish_face(top, "top"),
        bottom=_finish_face(bottom, "bottom"),
        front=_finish_face(front, "front"),
        back=_finish_face(back, "back"),
        left=_finish_face(left, "left"),
        right=_finish_face(right, "right"),
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
    top, bottom, front = cells[0], cells[1], cells[2]
    back, left, right = cells[3], cells[4], cells[5]
    return CubeFaces(
        top=_finish_face(top, "top"),
        bottom=_finish_face(bottom, "bottom"),
        front=_finish_face(front, "front"),
        back=_finish_face(back, "back"),
        left=_finish_face(left, "left"),
        right=_finish_face(right, "right"),
    )


def _emblem_faces(image: Image.Image, size: int) -> CubeFaces:
    """Subject scaled on solid black — print-style, not a full-bleed sticker."""
    emblem = _subject_on_black(image, size)
    return CubeFaces(
        top=_finish_face(emblem, "top"),
        bottom=_finish_face(emblem, "bottom"),
        front=_finish_face(emblem, "front"),
        back=_finish_face(emblem, "back"),
        left=_finish_face(emblem, "left"),
        right=_finish_face(emblem, "right"),
    )


def _subject_on_black(image: Image.Image, size: int, margin: float = 0.14) -> Image.Image:
    rgba = image.convert("RGBA") if has_useful_alpha(image) else None
    if rgba is not None:
        bbox = rgba.getbbox()
        if bbox is None:
            return Image.new("RGB", (size, size), FACE_BACKGROUND)
        subject = rgba.crop(bbox)
    else:
        subject = resize_square(center_square(flatten_rgb(image)), size).convert("RGBA")

    sw, sh = subject.size
    inner = max(1, int(size * (1.0 - 2.0 * margin)))
    scale = min(inner / max(sw, 1), inner / max(sh, 1))
    nw, nh = max(1, int(sw * scale)), max(1, int(sh * scale))
    fitted = subject.resize((nw, nh), Image.Resampling.LANCZOS).convert("RGBA")
    canvas = Image.new("RGBA", (size, size), (*FACE_BACKGROUND, 255))
    canvas.alpha_composite(fitted, ((size - nw) // 2, (size - nh) // 2))
    return flatten_rgb(canvas, FACE_BACKGROUND)


def _equator_strip(image: Image.Image, size: int, y0: float, y1: float) -> Image.Image:
    """Build a 4×1 panorama used as LEFT|FRONT|RIGHT|BACK.

    Wide sources are center-cropped into four sequential panels. Narrow or
    square sources pan four overlapping windows left→right so a 3/4 photo's
    left wall and right wall land on different faces instead of tiling the
    same sticker four times.
    """
    mid = _horizontal_band(image, y0, y1)
    mw, mh = mid.size
    new_w = max(1, round(mw * size / max(mh, 1)))
    scaled = mid.resize((new_w, size), Image.Resampling.LANCZOS)
    target = 4 * size
    if scaled.width >= target:
        x0 = (scaled.width - target) // 2
        return scaled.crop((x0, 0, x0 + target, size))

    if scaled.width < size:
        scaled = scaled.resize((size, size), Image.Resampling.LANCZOS)

    canvas = Image.new("RGB", (target, size), FACE_BACKGROUND)
    max_off = max(0, scaled.width - size)
    for i, t in enumerate((0.0, 0.34, 0.68, 1.0)):
        x0 = int(round(max_off * t))
        tile = scaled.crop((x0, 0, x0 + size, size))
        canvas.paste(tile, (i * size, 0))
    return canvas


def _fabric_face(image: Image.Image, size: int, y0: float, y1: float) -> Image.Image:
    """Roof/floor from a horizontal crop, or grain-matched solid cloth."""
    band = _horizontal_band(image, y0, y1)
    if band.height >= 8 and min(band.size) >= 8:
        return resize_square(center_square(band), size)
    return _solid_fabric(image, size)


def _solid_fabric(image: Image.Image, size: int) -> Image.Image:
    arr = np.asarray(flatten_rgb(image), dtype=np.float32)
    # Median of the darker half — Black Cube cloth, not highlights.
    luma = arr.mean(axis=2)
    cutoff = np.median(luma)
    dark = arr[luma <= cutoff]
    if dark.size == 0:
        color = arr.reshape(-1, 3).mean(axis=0)
    else:
        color = np.median(dark, axis=0)
    rgb = np.clip(color, 0, 255).astype(np.uint8)
    canvas = np.zeros((size, size, 3), dtype=np.uint8)
    canvas[:, :] = rgb
    rng = np.random.default_rng(0x5A7A12)
    grain = rng.integers(-10, 11, (size, size, 1), dtype=np.int16)
    return Image.fromarray(np.clip(canvas.astype(np.int16) + grain, 0, 255).astype(np.uint8))


def _horizontal_band(image: Image.Image, y0: float, y1: float) -> Image.Image:
    w, h = image.size
    top = max(0, min(h - 1, int(h * y0)))
    bot = max(top + 1, min(h, int(h * y1)))
    return image.crop((0, top, w, bot))


def find_horizontal_band(image: Image.Image) -> tuple[float, float] | None:
    """Return ``(y0, y1)`` fractions of a saturated equatorial stripe, if any.

    Tuned for a gold kiswah belt (high R+G, low B) but also fires on any
    high-chroma horizontal run that spans most of the width.
    """
    rgb = flatten_rgb(image).resize((64, 64), Image.Resampling.BOX)
    arr = np.asarray(rgb, dtype=np.float32)
    r, g, b = arr[:, :, 0], arr[:, :, 1], arr[:, :, 2]
    chroma = arr.max(axis=2) - arr.min(axis=2)
    gold = (r > 120) & (g > 90) & (b < 0.88 * g) & ((r + g) > (2.0 * b + 30))
    row_chroma = chroma.mean(axis=1)
    row_gold = gold.mean(axis=1)
    chroma_med = float(np.median(row_chroma))
    gold_med = float(np.median(row_gold))
    hot = (row_chroma > max(chroma_med * 1.65, 18.0)) | (row_gold > max(gold_med * 2.2, 0.12))
    run = _longest_true_run(hot)
    if run is None:
        return None
    i0, i1 = run
    length = i1 - i0
    if length < 3 or length > 28:
        return None
    # Band should be a stripe, not the whole frame, and sit away from a single edge pixel.
    return i0 / 64.0, i1 / 64.0


def _longest_true_run(mask: np.ndarray) -> tuple[int, int] | None:
    best: tuple[int, int] | None = None
    start: int | None = None
    for i, flag in enumerate(list(mask) + [False]):
        if flag and start is None:
            start = i
        elif not flag and start is not None:
            run = (start, i)
            if best is None or (run[1] - run[0]) > (best[1] - best[0]):
                best = run
            start = None
    return best


def _looks_like_cutout(image: Image.Image) -> bool:
    """RGB heuristic: uniform corners + a distinct center subject."""
    if has_useful_alpha(image):
        return True
    rgb = np.asarray(flatten_rgb(image).resize((64, 64), Image.Resampling.BOX), dtype=np.float32)
    corners = [rgb[:8, :8], rgb[:8, -8:], rgb[-8:, :8], rgb[-8:, -8:]]
    means = np.stack([c.mean(axis=(0, 1)) for c in corners])
    if float(means.std()) > 18.0:
        return False
    corner_std = float(np.mean([c.std() for c in corners]))
    if corner_std > 24.0:
        return False
    cm = means.mean(axis=0)
    center = rgb[16:48, 16:48].mean(axis=(0, 1))
    if float(np.linalg.norm(cm - center)) < 45.0:
        return False
    return True


def _finish_face(panel: Image.Image, name: str) -> Image.Image:
    params = _FACE_LIGHT[name]
    out = flatten_rgb(panel)
    out = _shade_gradient(out, name)
    out = ImageEnhance.Brightness(out).enhance(params["brightness"])
    out = ImageEnhance.Contrast(out).enhance(params["contrast"])
    out = ImageEnhance.Color(out).enhance(params["color"])
    out = _hue_shift(out, params["hue"])
    return flatten_rgb(out)


def _hue_shift(image: Image.Image, degrees: float) -> Image.Image:
    if abs(degrees) < 0.5:
        return image
    hsv = image.convert("HSV")
    h, s, v = hsv.split()
    delta = int(round(degrees * 255.0 / 360.0))
    h = h.point(lambda x, d=delta: (x + d) % 256)
    return Image.merge("HSV", (h, s, v)).convert("RGB")


def _shade_gradient(image: Image.Image, name: str) -> Image.Image:
    """Soft directional light so identical albedo still reads as a cube."""
    arr = np.asarray(flatten_rgb(image), dtype=np.float32)
    h, w = arr.shape[:2]
    yy, xx = np.mgrid[0:h, 0:w]
    yn = yy / max(h - 1, 1)
    xn = xx / max(w - 1, 1)
    if name == "top":
        light = 1.06 - 0.08 * yn - 0.04 * ((xn - 0.5) ** 2)
    elif name == "bottom":
        light = 0.90 + 0.06 * yn
    elif name == "left":
        light = 0.86 + 0.16 * xn - 0.06 * yn
    elif name == "right":
        light = 1.04 - 0.10 * xn - 0.04 * yn
    elif name == "front":
        light = 0.94 + 0.10 * (1.0 - yn)
    else:  # back
        light = 0.88 + 0.04 * (1.0 - yn)
    arr *= light[:, :, None]
    return Image.fromarray(np.clip(arr, 0, 255).astype(np.uint8))
