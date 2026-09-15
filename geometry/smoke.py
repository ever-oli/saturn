"""Headless smoke: generate mockups and fail if transparency-checker bleed remains.

Run::

    python -m geometry.smoke
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw

from config import DEMO_DIR, PANEL_SIZE
from geometry.faces import resolve_face_mode
from geometry.net import FACE_CELLS, NET_COLS, NET_GAP, NET_ROWS
from geometry.pipeline import generate
from geometry.tees import ensure_example_band, ensure_example_starfield, ensure_tee_templates
from geometry.types import checker_score, flatten_rgb


# Keep smoke snappy; production still uses SATURN_PANEL_SIZE.
_SMOKE_PANEL = min(256, PANEL_SIZE)
_CHECKER_LIMIT = 0.025


def run_smoke(output_dir: Path | None = None) -> Path:
    ensure_tee_templates()
    dest = Path(output_dir or DEMO_DIR)
    dest.mkdir(parents=True, exist_ok=True)

    starfield = Image.open(ensure_example_starfield()).convert("RGB")
    band = Image.open(ensure_example_band()).convert("RGB")
    kaaba = _synthetic_kaaba_like(640)
    cutout = _synthetic_cutout_subject(640)

    kaaba.save(dest / "source_kaaba_like.png")
    cutout.save(dest / "source_cutout.png")

    _assert_mode(starfield, "single", "starfield")
    _assert_mode(band, "wrap", "gold-band")
    _assert_mode(kaaba, "wrap", "kaaba-like")
    _assert_mode(cutout, "emblem", "cutout")

    cases = [
        ("starfield", starfield, "auto", False),
        ("kaaba_like", kaaba, "auto", False),
        ("cutout", cutout, "auto", False),
    ]
    scores: list[str] = []
    for name, src, mode, rembg in cases:
        result = generate(src, face_mode=mode, panel_size=_SMOKE_PANEL, remove_bg=rembg)
        bundle = {
            "combined": result.combined,
            "front": result.front,
            "back": result.back,
            "cube": result.cube,
            "net": result.net,
        }
        for key, im in bundle.items():
            if im.mode != "RGB":
                raise AssertionError(f"{name} {key} is {im.mode}, expected opaque RGB")
            path = dest / f"{name}_{key}.png"
            im.save(path)
            score = checker_score(im)
            scores.append(f"{name}_{key}: mode={im.mode} checker={score:.4f} face_mode={result.face_mode}")
            if score > _CHECKER_LIMIT:
                raise AssertionError(
                    f"Checkerboard-like region in {path} (score={score:.4f} > {_CHECKER_LIMIT})"
                )

        _assert_faces_opaque_dark_or_lit(result.faces, name)

    # Before/after for a transparent subject: old sticker-net vs new emblem net.
    _naive_sticker_net(cutout, _SMOKE_PANEL).save(dest / "cutout_before_net.png")
    generate(cutout, face_mode="emblem", panel_size=_SMOKE_PANEL).net.save(
        dest / "cutout_after_net.png"
    )

    summary = dest / "smoke.txt"
    summary.write_text("\n".join(scores) + "\n", encoding="utf-8")
    return dest / "starfield_combined.png"


def _assert_mode(image: Image.Image, expected: str, label: str) -> None:
    got = resolve_face_mode(image, "auto")
    if got != expected:
        raise AssertionError(f"auto mode for {label}: expected {expected}, got {got}")


def _assert_faces_opaque_dark_or_lit(faces: dict, label: str) -> None:
    for name, face in faces.items():
        rgb = flatten_rgb(face)
        if rgb.mode != "RGB":
            raise AssertionError(f"{label} face {name} is {rgb.mode}")
        arr = np.asarray(rgb)
        # No large mid-gray checker field: corners of emblem faces should be near black.
        score = checker_score(rgb)
        if score > _CHECKER_LIMIT:
            raise AssertionError(f"{label} face {name} checker score {score:.4f}")


def _naive_sticker_net(image: Image.Image, size: int) -> Image.Image:
    """Old look: same perspective cutout stamped on every panel, alpha unresolved.

    Composited onto a baked gray/white checker so the PNG itself documents the
    Gradio transparency-grid failure (GitHub cannot show RGBA checkers).
    """
    sticker = image.convert("RGBA").resize((size, size), Image.Resampling.LANCZOS)
    pad = NET_GAP
    step = size + pad
    canvas_w = NET_COLS * size + (NET_COLS - 1) * pad
    canvas_h = NET_ROWS * size + (NET_ROWS - 1) * pad
    net = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    for _name, (col, row) in FACE_CELLS.items():
        net.paste(sticker, (col * step, row * step), sticker)
    return _flatten_onto_checker(net)


def _flatten_onto_checker(image: Image.Image, tile: int = 8) -> Image.Image:
    rgba = image.convert("RGBA")
    w, h = rgba.size
    yy, xx = np.indices((h, w))
    chk = ((xx // tile) + (yy // tile)) % 2
    bg = np.empty((h, w, 3), dtype=np.uint8)
    bg[chk == 0] = (192, 192, 192)
    bg[chk == 1] = (255, 255, 255)
    base = Image.fromarray(bg)
    base.paste(rgba, mask=rgba.split()[-1])
    return base


def _synthetic_kaaba_like(size: int = 640) -> Image.Image:
    """Dark cloth + gold equatorial band (kiswah-like) for wrap-mode smoke."""
    rng = np.random.default_rng(0x5A7A12 + 7)
    cloth = rng.integers(6, 24, (size, size, 3), dtype=np.uint8)
    # Faint warm embroidery specks.
    spec = rng.random((size, size)) > 0.992
    cloth[spec] = (40, 32, 12)
    y0, y1 = int(size * 0.30), int(size * 0.42)
    gold = np.zeros((y1 - y0, size, 3), dtype=np.uint8)
    gold[:, :, 0] = rng.integers(168, 214, (y1 - y0, size))
    gold[:, :, 1] = rng.integers(126, 172, (y1 - y0, size))
    gold[:, :, 2] = rng.integers(42, 78, (y1 - y0, size))
    cloth[y0:y1] = gold
    return Image.fromarray(cloth)


def _synthetic_cutout_subject(size: int = 640) -> Image.Image:
    """3/4 cube-like glyph on a transparent field — the live-Space failure case."""
    img = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(img, "RGBA")
    cx, cy = size // 2, int(size * 0.52)
    s = int(size * 0.28)
    # Isometric hex: top / left / right diamonds of a cube.
    top = [
        (cx, cy - s),
        (cx + int(s * 0.92), cy - int(s * 0.42)),
        (cx, cy + int(s * 0.08)),
        (cx - int(s * 0.92), cy - int(s * 0.42)),
    ]
    left = [
        (cx - int(s * 0.92), cy - int(s * 0.42)),
        (cx, cy + int(s * 0.08)),
        (cx, cy + int(s * 1.12)),
        (cx - int(s * 0.92), cy + int(s * 0.62)),
    ]
    right = [
        (cx + int(s * 0.92), cy - int(s * 0.42)),
        (cx, cy + int(s * 0.08)),
        (cx, cy + int(s * 1.12)),
        (cx + int(s * 0.92), cy + int(s * 0.62)),
    ]
    draw.polygon(top, fill=(28, 28, 28, 255))
    draw.polygon(left, fill=(12, 12, 12, 255))
    draw.polygon(right, fill=(22, 22, 22, 255))
    # Gold belt across the two walls.
    band_y = cy + int(s * 0.22)
    draw.polygon(
        [
            (cx - int(s * 0.92), band_y),
            (cx, band_y + int(s * 0.12)),
            (cx, band_y + int(s * 0.28)),
            (cx - int(s * 0.92), band_y + int(s * 0.16)),
        ],
        fill=(196, 150, 62, 255),
    )
    draw.polygon(
        [
            (cx + int(s * 0.92), band_y),
            (cx, band_y + int(s * 0.12)),
            (cx, band_y + int(s * 0.28)),
            (cx + int(s * 0.92), band_y + int(s * 0.16)),
        ],
        fill=(210, 164, 72, 255),
    )
    return img


if __name__ == "__main__":
    path = run_smoke()
    print(path)
