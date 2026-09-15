"""Placeholder for multi-view diffusion / SF3D cube-face inference.

v1 is CPU-only. This module is the seam where Zero123++, SV3D, or SF3D will
plug in on a later GPU Space. It must stay import-safe with no torch.
"""

from __future__ import annotations

from PIL import Image

from geometry.types import CubeFaces


def infer_cube_faces(
    image: Image.Image,
    panel_size: int,
) -> CubeFaces | None:
    """Return six consistent cube faces, or None to fall back to PIL geometry.

    Future implementation sketch (not wired):
        1. Run a multi-view diffusion model to synthesize 6 orthographic faces.
        2. Optionally reconstruct a mesh with SF3D and bake textures.
        3. Return ``CubeFaces`` with the same orientation contract as
           ``geometry.faces.derive_faces``.
    """
    _ = (image, panel_size)
    return None
