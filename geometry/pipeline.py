"""Upload → 6 faces → isometric cube + Latin-cross net → t-shirt mockups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from config import DEMO_DIR, ENABLE_MULTIVIEW, MOCKUP_BACKGROUND, PANEL_SIZE
from geometry.cube import render_isometric_cube
from geometry.faces import derive_faces
from geometry.mockup import ShirtMockup, composite_shirts
from geometry.multiview import infer_cube_faces
from geometry.net import build_latin_cross_net
from geometry.tees import ensure_example_starfield, ensure_tee_templates
from geometry.types import CubeFaces, checker_score, flatten_rgb


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
    """v1 generate(): PIL face mapping + geometry + shirt compositing."""
    ensure_tee_templates()
    size = panel_size or PANEL_SIZE
    used_mv = False
    faces: CubeFaces | None = None
    resolved_mode = (face_mode or "auto").strip().lower()
    removed_bg = False

    want_mv = ENABLE_MULTIVIEW if enable_multiview is None else enable_multiview
    if want_mv:
        # Hook only — infer_cube_faces is a stub and returns None on CPU.
        faces = infer_cube_faces(image, size)
        used_mv = faces is not None

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
