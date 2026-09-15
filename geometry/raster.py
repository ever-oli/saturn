"""Affine texture mapping onto isometric cube parallelograms."""

from __future__ import annotations

import numpy as np
from PIL import Image

from config import FACE_BACKGROUND
from geometry.types import as_opaque_rgba


def paste_parallelogram(
    canvas: Image.Image,
    texture: Image.Image,
    origin: tuple[float, float],
    u_vec: tuple[float, float],
    v_vec: tuple[float, float],
    shade: float = 1.0,
) -> None:
    """Map a rectangular texture onto the dest parallelogram (origin, +u, +v).

    ``origin`` is the texture's (0, 0) corner. ``u_vec`` and ``v_vec`` run to
    the (1, 0) and (0, 1) corners. Inverse-affine sampled with bilinear filter.
    The texture is flattened onto an opaque fill first so face interiors never
    sample unresolved alpha (Gradio's checker).
    """
    opaque = as_opaque_rgba(texture, FACE_BACKGROUND)
    tex = np.asarray(opaque, dtype=np.float32)
    tex[:, :, 3] = 255.0
    th, tw = tex.shape[:2]
    if th < 2 or tw < 2:
        return

    o = np.array(origin, dtype=np.float64)
    u = np.array(u_vec, dtype=np.float64)
    v = np.array(v_vec, dtype=np.float64)
    matrix = np.column_stack((u, v))
    try:
        inverse = np.linalg.inv(matrix)
    except np.linalg.LinAlgError:
        return

    corners = np.stack((o, o + u, o + v, o + u + v))
    minx = int(np.floor(corners[:, 0].min()))
    maxx = int(np.ceil(corners[:, 0].max()))
    miny = int(np.floor(corners[:, 1].min()))
    maxy = int(np.ceil(corners[:, 1].max()))

    cw, ch = canvas.size
    minx, miny = max(0, minx), max(0, miny)
    maxx, maxy = min(cw, maxx), min(ch, maxy)
    if maxx - minx < 1 or maxy - miny < 1:
        return

    ys, xs = np.mgrid[miny:maxy, minx:maxx]
    pts = np.stack((xs.ravel() + 0.5, ys.ravel() + 0.5), axis=1) - o
    uv = pts @ inverse.T
    uu, vv = uv[:, 0], uv[:, 1]
    # Slightly expand the hit test so adjacent faces share a pixel and hide cracks.
    inside = (uu >= -0.003) & (uu <= 1.003) & (vv >= -0.003) & (vv <= 1.003)
    if not np.any(inside):
        return

    uu = np.clip(uu[inside], 0.0, 1.0)
    vv = np.clip(vv[inside], 0.0, 1.0)
    xs_i = xs.ravel()[inside]
    ys_i = ys.ravel()[inside]

    src_x = uu * (tw - 1)
    src_y = vv * (th - 1)
    x0 = np.floor(src_x).astype(np.int32)
    y0 = np.floor(src_y).astype(np.int32)
    x1 = np.clip(x0 + 1, 0, tw - 1)
    y1 = np.clip(y0 + 1, 0, th - 1)
    x0 = np.clip(x0, 0, tw - 1)
    y0 = np.clip(y0, 0, th - 1)
    dx = (src_x - x0).astype(np.float32)[:, None]
    dy = (src_y - y0).astype(np.float32)[:, None]

    c00 = tex[y0, x0]
    c10 = tex[y0, x1]
    c01 = tex[y1, x0]
    c11 = tex[y1, x1]
    sample = (
        c00 * (1 - dx) * (1 - dy)
        + c10 * dx * (1 - dy)
        + c01 * (1 - dx) * dy
        + c11 * dx * dy
    )
    if shade != 1.0:
        sample = sample.copy()
        sample[:, :3] = np.clip(sample[:, :3] * shade, 0, 255)
    sample[:, 3] = 255.0

    dest = np.array(canvas, dtype=np.float32)
    dst_px = dest[ys_i, xs_i]
    src_a = (sample[:, 3:4] / 255.0)
    dst_a = (dst_px[:, 3:4] / 255.0)
    out_a = src_a + dst_a * (1.0 - src_a)
    out_rgb = sample[:, :3] * src_a + dst_px[:, :3] * dst_a * (1.0 - src_a)
    with np.errstate(divide="ignore", invalid="ignore"):
        out_rgb = np.divide(out_rgb, out_a, out=np.zeros_like(out_rgb), where=out_a > 1e-6)
    blended = np.concatenate([out_rgb, out_a * 255.0], axis=1)
    dest[ys_i, xs_i] = blended
    canvas.paste(Image.fromarray(np.clip(dest, 0, 255).astype(np.uint8)))
