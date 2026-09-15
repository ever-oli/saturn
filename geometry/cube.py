"""Isometric cube renderer — three visible faces + white wireframe + ring shadow."""

from __future__ import annotations

import math

from PIL import Image, ImageDraw

from config import CUBE_EDGE
from geometry.raster import paste_parallelogram
from geometry.types import CubeFaces

_SQRT3 = math.sqrt(3.0)


def render_isometric_cube(
    faces: CubeFaces,
    edge: int = 360,
    padding: int = 48,
) -> Image.Image:
    """Draw a 30° isometric cube (top / front / right) onto a transparent canvas.

    Geometry is approximate on purpose: v1 is PIL affine mapping, not a mesh
    renderer. Multi-view diffusion / SF3D can replace this later.
    """
    # Bounding box of the hexagon: width = edge * √3, height = 2 * edge
    shadow_h = int(edge * 0.42)
    width = int(edge * _SQRT3) + padding * 2
    height = 2 * edge + shadow_h + padding * 2
    canvas = Image.new("RGBA", (width, height), (0, 0, 0, 0))

    origin = (width / 2.0, padding + edge)

    def iso(x: float, y: float, z: float) -> tuple[float, float]:
        sx = (x - y) * (_SQRT3 / 2.0) * edge
        sy = ((x + y) * 0.5 - z) * edge
        return origin[0] + sx, origin[1] + sy

    # Cube [0,1]^3. Camera sees TOP (z=1), FRONT (y=1), RIGHT (x=1).
    e = iso(0, 0, 1)
    f = iso(1, 0, 1)
    g = iso(1, 1, 1)
    h = iso(0, 1, 1)
    b = iso(1, 0, 0)
    c = iso(1, 1, 0)
    d = iso(0, 1, 0)

    _draw_shadow(canvas, c, edge)

    # TOP: origin E, +u → F, +v → H  (front of top image = H–G, matches net)
    paste_parallelogram(
        canvas,
        faces["top"],
        origin=e,
        u_vec=(f[0] - e[0], f[1] - e[1]),
        v_vec=(h[0] - e[0], h[1] - e[1]),
        shade=1.12,
    )
    # FRONT (visible left): origin H, +u → G, +v → D
    paste_parallelogram(
        canvas,
        faces["front"],
        origin=h,
        u_vec=(g[0] - h[0], g[1] - h[1]),
        v_vec=(d[0] - h[0], d[1] - h[1]),
        shade=0.62,
    )
    # RIGHT (visible right): origin G, +u → F, +v → C
    paste_parallelogram(
        canvas,
        faces["right"],
        origin=g,
        u_vec=(f[0] - g[0], f[1] - g[1]),
        v_vec=(c[0] - g[0], c[1] - g[1]),
        shade=0.88,
    )

    _draw_wireframe(canvas, [e, f, b, c, d, h, g])
    return canvas


def _draw_shadow(canvas: Image.Image, front_bottom: tuple[float, float], edge: int) -> None:
    """Concentric ellipse under the cube (Saturn-ring / mockup pedestal)."""
    overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)
    cx, cy = front_bottom[0], front_bottom[1] + edge * 0.20
    rx, ry = edge * 0.58, edge * 0.15
    rings = 16
    for i in range(rings, 0, -1):
        t = i / rings
        alpha = int(55 * t * t)
        draw.ellipse(
            (cx - rx * t, cy - ry * t, cx + rx * t, cy + ry * t),
            outline=(0, 0, 0, max(alpha, 18)),
            width=max(2, int(3 * t)),
        )
    draw.ellipse(
        (cx - rx * 0.22, cy - ry * 0.22, cx + rx * 0.22, cy + ry * 0.22),
        fill=(0, 0, 0, 40),
    )
    canvas.alpha_composite(overlay)


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
        (h, e),  # top
        (h, d),
        (d, c),
        (c, g),  # front
        (f, b),
        (b, c),  # right
    ]
    for a, z in edges:
        draw.line([a, z], fill=CUBE_EDGE, width=2)
    canvas.alpha_composite(overlay)
