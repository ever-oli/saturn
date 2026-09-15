"""Upload → 6 faces → isometric cube + Latin-cross net → t-shirt mockups."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from PIL import Image

from config import DEMO_DIR, ENABLE_MULTIVIEW, PANEL_SIZE
from geometry.cube import render_isometric_cube
from geometry.faces import derive_faces
from geometry.mockup import ShirtMockup, composite_shirts
from geometry.multiview import infer_cube_faces
from geometry.net import build_latin_cross_net
from geometry.tees import ensure_example_starfield, ensure_tee_templates
from geometry.types import CubeFaces


@dataclass(frozen=True)
class SaturnResult:
    combined: Image.Image
    front: Image.Image
    back: Image.Image
    cube: Image.Image
    net: Image.Image
    faces: CubeFaces
    used_multiview: bool


def generate(
    image: Image.Image,
    face_mode: str = "single",
    panel_size: int | None = None,
    enable_multiview: bool | None = None,
) -> SaturnResult:
    """v1 generate(): PIL face mapping + geometry + shirt compositing."""
    ensure_tee_templates()
    size = panel_size or PANEL_SIZE
    used_mv = False
    faces: CubeFaces | None = None

    want_mv = ENABLE_MULTIVIEW if enable_multiview is None else enable_multiview
    if want_mv:
        # Hook only — infer_cube_faces is a stub and returns None on CPU.
        faces = infer_cube_faces(image, size)
        used_mv = faces is not None

    if faces is None:
        faces = derive_faces(image, size, mode=face_mode)

    cube = render_isometric_cube(faces, edge=max(220, size * 3 // 4))
    net = build_latin_cross_net(faces, panel_size=size)
    shirts = composite_shirts(cube, net)
    return SaturnResult(
        combined=shirts.combined,
        front=shirts.front,
        back=shirts.back,
        cube=cube,
        net=net,
        faces=faces,
        used_multiview=used_mv,
    )


def write_demo(output_dir: Path | None = None) -> Path:
    """Generate a committed-style demo mockup from the bundled starfield."""
    ensure_tee_templates()
    example = ensure_example_starfield()
    result = generate(Image.open(example), face_mode="single")
    dest = output_dir or DEMO_DIR
    dest.mkdir(parents=True, exist_ok=True)
    path = dest / "mockup.png"
    result.combined.save(path)
    result.front.save(dest / "front.png")
    result.back.save(dest / "back.png")
    result.cube.save(dest / "cube.png")
    result.net.save(dest / "net.png")
    return path
