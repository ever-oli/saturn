"""Procedural oversized blank-tee templates (front / back)."""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFilter

from config import TEE_BACK_PATH, TEE_FABRIC, TEE_FRONT_PATH

TEE_SIZE = (1000, 1200)


def ensure_tee_templates(
    front_path: Path | None = None,
    back_path: Path | None = None,
    size: tuple[int, int] = TEE_SIZE,
) -> tuple[Path, Path]:
    """Write placeholder tees if they are missing. Returns (front, back) paths."""
    front = front_path or TEE_FRONT_PATH
    back = back_path or TEE_BACK_PATH
    front.parent.mkdir(parents=True, exist_ok=True)
    back.parent.mkdir(parents=True, exist_ok=True)
    if not front.exists():
        render_blank_tee("front", size).save(front)
    if not back.exists():
        render_blank_tee("back", size).save(back)
    return front, back


def load_tee(view: str) -> Image.Image:
    ensure_tee_templates()
    path = TEE_FRONT_PATH if view == "front" else TEE_BACK_PATH
    return Image.open(path).convert("RGBA")


def render_blank_tee(
    view: str = "front",
    size: tuple[int, int] = TEE_SIZE,
    fabric: tuple[int, int, int] = TEE_FABRIC,
) -> Image.Image:
    """Draw a drop-shoulder oversized tee on a transparent canvas."""
    scale = 2
    w, h = size[0] * scale, size[1] * scale
    mask = _tee_mask(w, h, view)
    # Soften the silhouette, then keep a crisp alpha.
    mask = mask.filter(ImageFilter.GaussianBlur(radius=2.4))

    rgb = np.zeros((h, w, 3), dtype=np.uint8)
    rgb[:, :] = np.array(fabric, dtype=np.uint8)
    grain = np.random.default_rng(7).integers(0, 18, (h, w, 1), dtype=np.uint8)
    rgb = np.clip(rgb.astype(np.int16) - grain + 6, 0, 255).astype(np.uint8)

    alpha = np.array(mask, dtype=np.uint8)
    rgba = np.dstack([rgb, alpha])
    tee = Image.fromarray(rgba, "RGBA")

    overlay = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    _draw_collar(draw, w, h, view)
    _draw_seams(draw, w, h, view)
    tee = Image.alpha_composite(tee, overlay)
    tee = tee.resize(size, Image.Resampling.LANCZOS)
    return tee


def _p(x: float, y: float, w: int, h: int) -> tuple[float, float]:
    return x * w / 1000.0, y * h / 1200.0


def _tee_mask(w: int, h: int, view: str) -> Image.Image:
    mask = Image.new("L", (w, h), 0)
    d = ImageDraw.Draw(mask)

    body = [
        _p(318, 238, w, h),
        _p(188, 248, w, h),
        _p(62, 328, w, h),
        _p(28, 418, w, h),
        _p(148, 492, w, h),
        _p(228, 448, w, h),
        _p(198, 1072, w, h),
        _p(802, 1072, w, h),
        _p(772, 448, w, h),
        _p(852, 492, w, h),
        _p(972, 418, w, h),
        _p(938, 328, w, h),
        _p(812, 248, w, h),
        _p(682, 238, w, h),
    ]
    d.polygon(body, fill=255)

    # Collar cut (deeper on front).
    if view == "front":
        d.ellipse(
            [*_p(412, 188, w, h), *_p(588, 292, w, h)],
            fill=0,
        )
    else:
        d.ellipse(
            [*_p(418, 204, w, h), *_p(582, 258, w, h)],
            fill=0,
        )
        d.rectangle([*_p(410, 236, w, h), *_p(590, 280, w, h)], fill=255)
    return mask


def _draw_collar(draw: ImageDraw.ImageDraw, w: int, h: int, view: str) -> None:
    color = (210, 210, 206, 90)
    if view == "front":
        draw.arc(
            [*_p(414, 190, w, h), *_p(586, 290, w, h)],
            start=12,
            end=168,
            fill=color,
            width=7,
        )
    else:
        draw.arc(
            [*_p(420, 206, w, h), *_p(580, 256, w, h)],
            start=200,
            end=340,
            fill=color,
            width=6,
        )


def _draw_seams(draw: ImageDraw.ImageDraw, w: int, h: int, view: str) -> None:
    seam = (200, 200, 196, 36)
    draw.line([_p(228, 448, w, h), _p(198, 1072, w, h)], fill=seam, width=3)
    draw.line([_p(772, 448, w, h), _p(802, 1072, w, h)], fill=seam, width=3)
    draw.arc(
        [*_p(198, 1044, w, h), *_p(802, 1100, w, h)],
        start=200,
        end=340,
        fill=seam,
        width=3,
    )
    _ = view


def ensure_example_starfield(path: Path | None = None, size: int = 768) -> Path:
    """Procedural grainy starfield used as the Gradio example texture."""
    from config import EXAMPLES_DIR

    out = path or (EXAMPLES_DIR / "starfield.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    rng = np.random.default_rng(saturn_seed())
    yy, xx = np.mgrid[0:size, 0:size]
    vignette = 1.0 - 0.35 * ((xx - size / 2) ** 2 + (yy - size / 2) ** 2) / (size**2)
    grain = rng.normal(48, 22, (size, size))
    img = np.clip(grain * vignette, 0, 255)
    stars = rng.random((size, size))
    img = np.where(stars > 0.9965, rng.uniform(170, 255, (size, size)), img)
    img = np.where(stars > 0.982, np.maximum(img, rng.uniform(90, 160, (size, size))), img)
    img = np.clip(img, 0, 255).astype(np.uint8)
    rgb = np.stack(
        [img, img, np.clip(img.astype(np.int16) + 8, 0, 255).astype(np.uint8)],
        axis=-1,
    )
    Image.fromarray(rgb, "RGB").save(out)
    return out


def ensure_example_band(path: Path | None = None, size: int = 768) -> Path:
    """Dark cube with a gold equatorial band — good demo for wrap mode."""
    from config import EXAMPLES_DIR

    out = path or (EXAMPLES_DIR / "gold-band.png")
    out.parent.mkdir(parents=True, exist_ok=True)
    if out.exists():
        return out
    rng = np.random.default_rng(saturn_seed() + 1)
    cloth = rng.integers(6, 22, (size, size, 3), dtype=np.uint8)
    y0, y1 = int(size * 0.22), int(size * 0.38)
    gold = np.zeros((y1 - y0, size, 3), dtype=np.uint8)
    gold[:, :, 0] = rng.integers(168, 210, (y1 - y0, size))
    gold[:, :, 1] = rng.integers(128, 168, (y1 - y0, size))
    gold[:, :, 2] = rng.integers(48, 78, (y1 - y0, size))
    cloth[y0:y1] = gold
    Image.fromarray(cloth, "RGB").save(out)
    return out


def saturn_seed() -> int:
    return 0x5A7A12
