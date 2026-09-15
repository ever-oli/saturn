"""Latin-cross cube net: vertical column of 4 + left/right wings on row 2."""

from __future__ import annotations

from PIL import Image, ImageDraw

from config import FACE_BACKGROUND, NET_BORDER, NET_GAP, NET_SEPARATOR
from geometry.types import CubeFaces, flatten_rgb

# Net occupancy (col, row) with row 0 at the top.
# Column of 4: top, front, bottom, back. Wings on the second square from the top.
FACE_CELLS: dict[str, tuple[int, int]] = {
    "top": (1, 0),
    "left": (0, 1),
    "front": (1, 1),
    "right": (2, 1),
    "bottom": (1, 2),
    "back": (1, 3),
}

NET_COLS = 3
NET_ROWS = 4


def build_latin_cross_net(
    faces: CubeFaces,
    panel_size: int | None = None,
    gap: int | None = None,
    border: tuple[int, int, int, int] = NET_BORDER,
) -> Image.Image:
    """Composite six square faces into a Latin-cross net.

    Panels are fully opaque RGB flattened onto the Black Cube fill. Thin light
    separators sit in the gaps. Pixels outside the cross stay transparent for
    tee compositing; the pipeline flattens the Gradio copy onto an opaque
    background so the checker never shows.
    """
    sample = flatten_rgb(faces["front"])
    size = panel_size or sample.size[0]
    pad = NET_GAP if gap is None else gap
    step = size + pad
    canvas_w = NET_COLS * size + (NET_COLS - 1) * pad
    canvas_h = NET_ROWS * size + (NET_ROWS - 1) * pad
    net = Image.new("RGBA", (canvas_w, canvas_h), (0, 0, 0, 0))
    draw = ImageDraw.Draw(net)

    # Opaque separator pads behind each panel (and in the gaps they share).
    for _name, (col, row) in FACE_CELLS.items():
        x, y = col * step, row * step
        draw.rectangle(
            (
                max(0, x - pad),
                max(0, y - pad),
                min(canvas_w - 1, x + size + pad - 1),
                min(canvas_h - 1, y + size + pad - 1),
            ),
            fill=NET_SEPARATOR,
        )

    for name, (col, row) in FACE_CELLS.items():
        face = flatten_rgb(faces[name], FACE_BACKGROUND).resize(
            (size, size), Image.Resampling.LANCZOS
        )
        outlined = _outline_panel(face.convert("RGBA"), border)
        x, y = col * step, row * step
        net.paste(outlined, (x, y), outlined)

    return net


def _outline_panel(panel: Image.Image, border: tuple[int, int, int, int]) -> Image.Image:
    out = panel.copy()
    if out.mode != "RGBA":
        out = out.convert("RGBA")
        # Force fully opaque after flatten.
        r, g, b, _a = out.split()
        opaque = Image.new("L", out.size, 255)
        out = Image.merge("RGBA", (r, g, b, opaque))
    else:
        r, g, b, _a = out.split()
        out = Image.merge("RGBA", (r, g, b, Image.new("L", out.size, 255)))
    draw = ImageDraw.Draw(out)
    w, h = out.size
    draw.rectangle((0, 0, w - 1, h - 1), outline=border, width=2)
    return out


def net_size(panel_size: int, gap: int | None = None) -> tuple[int, int]:
    pad = NET_GAP if gap is None else gap
    return (
        NET_COLS * panel_size + (NET_COLS - 1) * pad,
        NET_ROWS * panel_size + (NET_ROWS - 1) * pad,
    )
