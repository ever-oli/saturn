"""Upload → 6 faces → isometric cube + Latin-cross net → t-shirt mockups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from config import DEMO_DIR, ENABLE_MULTIVIEW, MOCKUP_BACKGROUND, PANEL_SIZE
from geometry.cube import render_isometric_cube
from geometry.faces import derive_faces
from geometry.mockup import composite_shirts
from geometry.multiview import infer_cube_faces, pipeline_loaded
from geometry.net import build_latin_cross_net
from geometry.tees import ensure_tee_templates
from geometry.types import CubeFaces, flatten_rgb


@dataclass(frozen=True)
class SaturnResult:
    combined: Image.Image
    front: Image.Image
    back: Image.Image
    cube: Image.Image
    net: Image.Image
    faces: CubeFaces
    used_multiview: bool
    face_mode: str
    removed_bg: bool


def generate(
    image: Image.Image,
    face_mode: str = "auto",
    panel_size: int | None = None,
    enable_multiview: bool | None = None,
    remove_bg: bool = False,
) -> SaturnResult:
    """PIL face mapping + optional Zero123++ faces, then cube/net/shirts."""
    ensure_tee_templates()
    size = panel_size or PANEL_SIZE
    used_mv = False
    faces: CubeFaces | None = None
    resolved_mode = (face_mode or "auto").strip().lower()
    removed_bg = False

    want_mv = ENABLE_MULTIVIEW if enable_multiview is None else enable_multiview
    if want_mv and pipeline_loaded():
        faces = infer_cube_faces(image, size, remove_bg=remove_bg)
        if faces is not None:
            used_mv = True
            resolved_mode = "zero123++"
            removed_bg = bool(remove_bg)

    if faces is None:
        faces, resolved_mode, removed_bg = derive_faces(
            image, size, mode=face_mode, remove_bg=remove_bg
        )

    cube_rgba = render_isometric_cube(faces, edge=max(220, size * 3 // 4))
    net_rgba = build_latin_cross_net(faces, panel_size=size)
    shirts = composite_shirts(cube_rgba, net_rgba)
    return SaturnResult(
        combined=shirts.combined,
        front=shirts.front,
        back=shirts.back,
        cube=flatten_rgb(cube_rgba, MOCKUP_BACKGROUND),
        net=flatten_rgb(net_rgba, MOCKUP_BACKGROUND),
        faces=faces,
        used_multiview=used_mv,
        face_mode=resolved_mode,
        removed_bg=removed_bg,
    )


def write_demo(output_dir: Path | None = None) -> Path:
    """Generate demo mockups (starfield + photo-like) and assert no checker bleed."""
    from geometry.smoke import run_smoke

    return run_smoke(output_dir or DEMO_DIR)
