"""Legacy Zero123++ multi-view backend (not loaded by the Gradio UI).

The product path is FLUX.2-klein-4B in ``app.py``. This module remains for
geometry smoke tests and optional ``SATURN_ENABLE_MULTIVIEW`` callers. Weights
are CC-BY-NC 4.0 — not for a commercial Printify pipeline.

Official load (SUDO-AI-3D/zero123plus README, v1.2 usage unchanged from v1.1)::

    DiffusionPipeline.from_pretrained(
        "sudo-ai/zero123plus-v1.2",
        custom_pipeline="sudo-ai/zero123plus-pipeline",
        torch_dtype=torch.float16,
    )

The pipeline class is ``Zero123PlusPipeline`` in that custom module. It returns
one 640×960 PIL image: a **2×3** grid of 320×320 tiles (row-major, left→right,
top→bottom), matching ``gradio_app.py``::

    side_len = image.width // 2
    tiles = [crop...] for y in rows for x in cols

v1.2 cameras (azimuth relative to the input view, elevation absolute)::

    index  row,col   az     el     Saturn face
    0      0,0       30°   +20°    FRONT
    1      0,1       90°   −10°    RIGHT
    2      1,0      150°   +20°    TOP     (leftover +20° view; not a true +Z)
    3      1,1      210°   −10°    BACK    (≈180° from FRONT's 30°)
    4      2,0      270°   +20°    LEFT
    5      2,1      330°   −10°    BOTTOM  (leftover −10° view; not a true −Z)

Zero123++ does **not** emit orthographic cube unwraps. These six perspective
views are assigned 1:1 into Saturn face slots so the Latin-cross net and
isometric cube stay view-consistent.

This module is import-safe without torch. ``load_pipeline()`` must run after
``import spaces`` if you ever load weights on ZeroGPU.
"""

from __future__ import annotations

import os
import traceback
from typing import Any

from PIL import Image

from config import (
    FACE_BACKGROUND,
    ZERO123_CUSTOM_PIPELINE,
    ZERO123_GUIDANCE,
    ZERO123_MODEL,
    ZERO123_STEPS,
)
from geometry.bg import try_remove_background
from geometry.types import CubeFaces, flatten_rgb, has_useful_alpha, resize_square

# Gray used by Zero123++ for RGBA flatten and square padding (pipeline to_rgb_image).
CONDITION_GRAY: tuple[int, int, int] = (127, 127, 127)

# Grid index → Saturn face. See module docstring.
VIEW_TO_FACE: tuple[str, ...] = (
    "front",
    "right",
    "top",
    "back",
    "left",
    "bottom",
)

# (azimuth_deg, elevation_deg) per grid index — Zero123++ v1.2.
VIEW_CAMERAS: tuple[tuple[int, int], ...] = (
    (30, 20),
    (90, -10),
    (150, 20),
    (210, -10),
    (270, 20),
    (330, -10),
)

_PIPE: Any = None
_LOAD_ERROR: str | None = None


def pipeline_loaded() -> bool:
    return _PIPE is not None


def load_error() -> str | None:
    return _LOAD_ERROR


def load_pipeline() -> bool:
    """Load Zero123++ at module scope. Returns False on failure (PIL still works).

    Must be called from ``app.py`` after ``import spaces``.
    """
    global _PIPE, _LOAD_ERROR
    if _PIPE is not None:
        return True
    try:
        import torch
        from diffusers import DiffusionPipeline, EulerAncestralDiscreteScheduler

        device = "cuda" if torch.cuda.is_available() else "cpu"
        if device == "cuda" and os.getenv("SPACES_ZERO_GPU"):
            dtype = torch.bfloat16
        elif device == "cuda":
            dtype = torch.float16
        else:
            dtype = torch.float32

        pipe = DiffusionPipeline.from_pretrained(
            ZERO123_MODEL,
            custom_pipeline=ZERO123_CUSTOM_PIPELINE,
            torch_dtype=dtype,
        )
        pipe.scheduler = EulerAncestralDiscreteScheduler.from_config(
            pipe.scheduler.config, timestep_spacing="trailing"
        )
        # ZeroGPU: string "cuda" only — never cuda:0 / device index.
        pipe.to(device)
        if hasattr(pipe, "prepare"):
            pipe.prepare()
        _PIPE = pipe
        _LOAD_ERROR = None
        return True
    except Exception as exc:
        _PIPE = None
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        print(f"Saturn: Zero123++ load failed ({_LOAD_ERROR}); PIL fallback stays available.")
        return False


def infer_cube_faces(
    image: Image.Image,
    panel_size: int,
    remove_bg: bool = False,
) -> CubeFaces | None:
    """Run Zero123++ and map the 6-view grid into CubeFaces, or None to fall back."""
    if _PIPE is None:
        return None
    try:
        cond, _removed = prepare_condition_image(image, remove_bg=remove_bg)
        result = _PIPE(
            cond,
            prompt="",
            guidance_scale=ZERO123_GUIDANCE,
            num_inference_steps=ZERO123_STEPS,
            width=640,
            height=960,
        )
        grid = result.images[0]
        tiles = split_zero123_grid(grid)
        return grid_to_cube_faces(tiles, panel_size)
    except Exception:
        traceback.print_exc()
        return None


def prepare_condition_image(
    image: Image.Image,
    remove_bg: bool = False,
) -> tuple[Image.Image, bool]:
    """Square, object-centric RGB on Zero123++ gray. Optional rembg first."""
    src = image
    removed = False
    if remove_bg:
        src, removed = try_remove_background(image)

    if has_useful_alpha(src) or src.mode in {"RGBA", "LA"}:
        rgba = src.convert("RGBA")
        square = _expand2square(rgba, (*CONDITION_GRAY, 0))
        bg = Image.new("RGB", square.size, CONDITION_GRAY)
        bg.paste(square, mask=square.split()[-1])
        out = bg
    else:
        out = _expand2square(flatten_rgb(src, CONDITION_GRAY), CONDITION_GRAY)

    w, h = out.size
    if min(w, h) < 320:
        scale = 320.0 / min(w, h)
        out = out.resize(
            (max(320, round(w * scale)), max(320, round(h * scale))),
            Image.Resampling.LANCZOS,
        )
    return out, removed


def split_zero123_grid(image: Image.Image) -> list[Image.Image]:
    """Split the 640×960 (2×3) Zero123++ output into six square tiles.

    Order matches the official Gradio demo: row-major, two columns.
    """
    rgb = flatten_rgb(image, CONDITION_GRAY)
    side = rgb.width // 2
    if side < 1 or rgb.height < 3 * side:
        raise ValueError(
            f"Zero123++ grid expected ~640×960 (2×3 of squares); got {rgb.size}"
        )
    tiles: list[Image.Image] = []
    for y in range(0, 3 * side, side):
        for x in range(0, 2 * side, side):
            tiles.append(rgb.crop((x, y, x + side, y + side)))
    if len(tiles) != 6:
        raise ValueError(f"Expected 6 Zero123++ tiles, got {len(tiles)}")
    return tiles


def grid_to_cube_faces(
    tiles: list[Image.Image],
    panel_size: int,
) -> CubeFaces:
    """Assign six tiles to Saturn faces and flatten to opaque RGB squares."""
    if len(tiles) != 6:
        raise ValueError(f"Need 6 tiles, got {len(tiles)}")
    mapped: dict[str, Image.Image] = {}
    for tile, name in zip(tiles, VIEW_TO_FACE, strict=True):
        mapped[name] = resize_square(flatten_rgb(tile, FACE_BACKGROUND), panel_size)
    return CubeFaces(
        top=mapped["top"],
        bottom=mapped["bottom"],
        front=mapped["front"],
        back=mapped["back"],
        left=mapped["left"],
        right=mapped["right"],
    )


def _expand2square(
    pil_img: Image.Image,
    background_color: tuple[int, ...],
) -> Image.Image:
    """Official Zero123++ square pad (see gradio_app.expand2square)."""
    width, height = pil_img.size
    if width == height:
        return pil_img
    if width > height:
        result = Image.new(pil_img.mode, (width, width), background_color)
        result.paste(pil_img, (0, (width - height) // 2))
        return result
    result = Image.new(pil_img.mode, (height, height), background_color)
    result.paste(pil_img, ((height - width) // 2, 0))
    return result
