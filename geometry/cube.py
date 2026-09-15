"""Isometric cube renderer — three visible faces + white wireframe + ring shadow."""

from __future__ import annotations

import math

from PIL import Image, ImageDraw, ImageFilter

from config import CUBE_EDGE, FACE_BACKGROUND
from geometry.raster import paste_parallelogram
from geometry.types import CubeFaces, as_opaque_rgba

_SQRT3 = math.sqrt(3.0)


def render_isometric_cube(
    faces: CubeFaces,
    edge: int = 360,
    padding: int = 48,
) -> Image.Image:
    """Draw a 30° isometric cube (top / front / right) onto an RGBA canvas.

    Faces are flattened opaque before sampling so the Gradio checker cannot
    punch through. Surrounding pixels stay transparent for tee compositing;
    the pipeline flattens the display copy onto an opaque background.
    """
    shadow_h = int(edge * 0.48)
    width = int(edge * _SQRT3) + padding * 2
    height = 2 * edge + shadow_h + padding * 2
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    origin = (width / 2.0, padding + edge)

    def iso(x: float, y: float, z: float) -> tuple[float, float]:
        sx = (x - y) * (_SQRT3 / 2.0) * edge
        sy = ((x + y) * 0.5 - z) * edge
        return origin[0] + sx, origin[1] + sy

    e = iso(0, 0, 1)
    f = iso(1, 0, 1)
    g = iso(1, 1, 1)
    h = iso(0, 1, 1)
    b = iso(1, 0, 0)
    c = iso(1, 1, 0)
    d = iso(0, 1, 0)

    _draw_shadow(canvas, c, edge)

    paste_parallelogram(
        canvas,
        as_opaque_rgba(faces["top"], FACE_BACKGROUND),
        origin=e,
        u_vec=(f[0] - e[0], f[1] - e[1]),
        v_vec=(h[0] - e[0], h[1] - e[1]),
        shade=1.10,
    )
    paste_parallelogram(
        canvas,
        as_opaque_rgba(faces["front"], FACE_BACKGROUND),
        origin=h,
        u_vec=(g[0] - h[0], g[1] - h[1]),
        v_vec=(d[0] - h[0], d[1] - h[1]),
        shade=0.80,
    )
    paste_parallelogram(
        canvas,
        as_opaque_rgba(faces["right"], FACE_BACKGROUND),
        origin=g,
        u_vec=(f[0] - g[0], f[1] - g[1]),
        v_vec=(c[0] - g[0], c[1] - g[1]),
        shade=0.94,
    )

    _draw_wireframe(canvas, [e, f, b, c, d, h, g])
    return canvas


def _draw_shadow(canvas: Image.Image, front_bottom: tuple[float, float], edge: int) -> None:
    """Soft filled ellipse under the cube (Saturn-ring / mockup pedestal)."""
    w, h = canvas.size
    layer = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(layer)
    cx = front_bottom[0]
    cy = front_bottom[1] + edge * 0.22
    rx, ry = edge * 0.78, edge * 0.20
    draw.ellipse((cx - rx, cy - ry, cx + rx, cy + ry), fill=255)
    # Inner contact umbra.
    draw.ellipse(
        (cx - rx * 0.42, cy - ry * 0.38, cx + rx * 0.42, cy + ry * 0.38),
        fill=255,
    )
    layer = layer.filter(ImageFilter.GaussianBlur(radius=max(10, edge // 16)))
    shadow = Image.new("RGBA", (w, h), (0, 0, 0, 0))
    alpha = layer.point(lambda p: int(p * 0.48))
    shadow.putalpha(alpha)
    canvas.alpha_composite(shadow)


def _draw_wireframe(
    canvas: Image.Image,
    pts: list[tuple[float, float]],
) -> None:
    e, f, b, c, d, h, g = pts
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    edges = [
        (e, f),
        (f, g),
        (g, h),
        (h, e),
        (h, d),
        (d, c),
        (c, g),
        (f, b),
        (b, c),
    ]
    # Dark understroke then bright edge so seams stay crisp on busy textures.
    for a, z in edges:
        draw.line([a, z], fill=(20, 20, 20, 160), width=5)
    for a, z in edges:
        draw.line([a, z], fill=CUBE_EDGE, width=3)
    canvas.alpha_composite(overlay)
