"""Saturn — Black Cube of Saturn streetwear customizer.

Hugging Face Spaces entrypoint. PIL geometry on CPU by default.
Zero123++ runs on ZeroGPU when enabled; Printify is not called.
"""

from __future__ import annotations

import os
from pathlib import Path

import spaces  # MUST be before torch / diffusers (ZeroGPU monkey-patch)

import gradio as gr
from PIL import Image

from config import (
    DEFAULT_FACE_MODE,
    ENABLE_MULTIVIEW,
    ENABLE_PRINTIFY,
    ZERO123_GPU_DURATION,
    ZERO123_MODEL,
    ZERO123_STEPS,
)
from geometry.bg import rembg_available
from geometry.multiview import load_error, load_pipeline, pipeline_loaded
from geometry.pipeline import generate
from geometry.tees import ensure_example_band, ensure_example_starfield, ensure_tee_templates

ensure_tee_templates()
EXAMPLE_STARFIELD = ensure_example_starfield()
EXAMPLE_BAND = ensure_example_band()

if ENABLE_MULTIVIEW:
    load_pipeline()

CUSTOM_CSS = """
@import url("https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500&family=Space+Grotesk:wght@400;500;600;700&display=swap");

:root {
  --saturn-bg: #090909;
  --saturn-panel: #111111;
  --saturn-line: #2a2a2a;
  --saturn-gold: #c4a574;
  --saturn-text: #e8e4dc;
  --saturn-dim: #8a8580;
}

.gradio-container {
  font-family: "Space Grotesk", system-ui, sans-serif !important;
  background: var(--saturn-bg) !important;
  max-width: 1280px !important;
}

footer { opacity: 0.45; }

#saturn-hero h1 {
  font-family: "Space Grotesk", system-ui, sans-serif;
  font-weight: 600;
  letter-spacing: 0.28em;
  font-size: 1.05rem !important;
  text-transform: uppercase;
  color: var(--saturn-text) !important;
  margin-bottom: 0.35rem;
}

#saturn-hero p,
#saturn-hero li {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.78rem !important;
  color: var(--saturn-dim) !important;
}

#saturn-hero em {
  color: var(--saturn-gold);
  font-style: normal;
}

button.primary {
  background: var(--saturn-gold) !important;
  color: #111 !important;
  border: none !important;
  letter-spacing: 0.12em;
  text-transform: uppercase;
  font-weight: 600 !important;
}

.stub-note {
  font-family: "IBM Plex Mono", ui-monospace, monospace;
  font-size: 0.75rem;
  color: var(--saturn-dim);
}
"""

HEAD = """
<meta name="theme-color" content="#090909" />
"""

THEME = gr.themes.Base(
    primary_hue="amber",
    secondary_hue="zinc",
    neutral_hue="zinc",
    font=gr.themes.GoogleFont("Space Grotesk"),
    font_mono=gr.themes.GoogleFont("IBM Plex Mono"),
).set(
    body_background_fill="#090909",
    body_background_fill_dark="#090909",
    body_text_color="#e8e4dc",
    body_text_color_dark="#e8e4dc",
    background_fill_primary="#111111",
    background_fill_primary_dark="#111111",
    background_fill_secondary="#161616",
    background_fill_secondary_dark="#161616",
    border_color_primary="#2a2a2a",
    border_color_primary_dark="#2a2a2a",
    block_background_fill="#111111",
    block_background_fill_dark="#111111",
    block_border_color="#2a2a2a",
    block_border_color_dark="#2a2a2a",
    block_label_text_color="#c4a574",
    block_title_text_color="#e8e4dc",
    button_primary_background_fill="#c4a574",
    button_primary_text_color="#111111",
    button_secondary_background_fill="#1a1a1a",
    input_background_fill="#161616",
)



ENGINE_PIL = "pil"
ENGINE_ZERO123 = "zero123++"


def _pack(result) -> tuple:
    engine = (
        f"Zero123++ (`{ZERO123_MODEL}`, {ZERO123_STEPS} steps)"
        if result.used_multiview
        else "PIL geometry"
    )
    rembg_note = "on" if result.removed_bg else "off"
    status = (
        f"**Path:** {engine} · **Face mode:** `{result.face_mode}` · "
        f"**rembg:** {rembg_note}\n\n"
        "Front is an isometric cube on the chest. Back is the 6-face Latin-cross net "
        "(column of 4; wings on the second square from the top)."
    )
    return result.combined, result.front, result.back, result.cube, result.net, status


def generate_pil(
    image: Image.Image | None,
    face_mode: str,
    remove_bg: bool,
) -> tuple:
    if image is None:
        raise gr.Error("Upload an image — Saturn maps it onto a cube and unfolds the net.")
    return _pack(generate(image, face_mode=face_mode, remove_bg=bool(remove_bg), enable_multiview=False))


@spaces.GPU(duration=ZERO123_GPU_DURATION)
def generate_multiview(
    image: Image.Image | None,
    face_mode: str,
    remove_bg: bool,
) -> tuple:
    """ZeroGPU worker: Zero123++ faces, then CPU cube/net/tee. Falls back to PIL."""
    if image is None:
        raise gr.Error("Upload an image — Saturn maps it onto a cube and unfolds the net.")
    return _pack(generate(image, face_mode=face_mode, remove_bg=bool(remove_bg), enable_multiview=True))


def generate_auto(
    image: Image.Image | None,
    face_mode: str,
    remove_bg: bool,
    engine: str,
) -> tuple:
    """Gradio default `/generate`. PIL stays on the web process (no GPU quota)."""
    key = (engine or ENGINE_PIL).strip().lower()
    if key in {ENGINE_ZERO123, "zero123plus", "multiview"} and ENABLE_MULTIVIEW and pipeline_loaded():
        return generate_multiview(image, face_mode, remove_bg)
    return generate_pil(image, face_mode, remove_bg)


with gr.Blocks(title="Saturn", theme=THEME, css=CUSTOM_CSS, head=HEAD) as demo:
    with gr.Column(elem_id="saturn-hero"):
        gr.Markdown(
            """
# Saturn
Black Cube of Saturn · *streetwear customizer*

Upload any image. Default path is **PIL geometry** (CPU, no GPU quota).
Enable **Multi-view / Zero123++** on a ZeroGPU Space for six view-consistent
faces. Faces are opaque materials or emblems (never unresolved alpha).
            """
        )

    with gr.Row():
        with gr.Column(scale=4):
            image_in = gr.Image(
                label="Source texture",
                type="pil",
                image_mode="RGBA",
                sources=["upload", "clipboard"],
                height=360,
            )
            _modes = ("auto", "single", "wrap", "emblem", "grid")
            face_mode = gr.Radio(
                choices=[
                    ("Auto", "auto"),
                    ("Single material", "single"),
                    ("Wrap band", "wrap"),
                    ("Emblem on black", "emblem"),
                    ("Grid 2×3", "grid"),
                ],
                value=DEFAULT_FACE_MODE if DEFAULT_FACE_MODE in _modes else "auto",
                label="Face mapping",
                info=(
                    "auto = cutout→emblem, wide/gold band→wrap, else material · "
                    "single = albedo + per-face light · wrap = kiswah/equator · "
                    "emblem = subject on black · grid = 2×3 crops"
                ),
            )
            _rembg = rembg_available()
            _mv_ready = ENABLE_MULTIVIEW and pipeline_loaded()
            _mv_info = (
                f"Zero123++ ready (`{ZERO123_MODEL}`, {ZERO123_STEPS} steps, "
                f"@{ZERO123_GPU_DURATION}s GPU). PIL stays default (no quota)."
                if _mv_ready
                else (
                    f"Zero123++ load failed: `{load_error()}`. PIL fallback only."
                    if ENABLE_MULTIVIEW and load_error()
                    else "Set Space variable `SATURN_ENABLE_MULTIVIEW=true` (on by default under ZeroGPU) to load Zero123++."
                )
            )
            engine = gr.Radio(
                choices=[
                    ("PIL geometry", ENGINE_PIL),
                    ("Multi-view / Zero123++", ENGINE_ZERO123),
                ],
                value=ENGINE_PIL,
                label="Cube engine",
                info=_mv_info,
                interactive=True,
            )
            remove_bg = gr.Checkbox(
                label="Remove background (rembg)",
                value=_rembg,
                info=(
                    "Cuts the subject out (u2netp, CPU) before emblem mapping or Zero123++. "
                    "Textures in single/wrap skip rembg. Falls back to flatten if rembg is unavailable."
                    if _rembg
                    else "rembg is not installed — alpha is flattened onto #0a0a0a instead."
                ),
                interactive=_rembg,
            )
            generate_btn = gr.Button("Generate mockup", variant="primary")
            status = gr.Markdown("Upload a texture, then generate.", elem_classes=["stub-note"])
        with gr.Column(scale=6):
            combined_out = gr.Image(
                label="Front + back mockup", type="pil", image_mode="RGB", height=520
            )

    with gr.Row():
        front_out = gr.Image(label="Front tee", type="pil", image_mode="RGB")
        back_out = gr.Image(label="Back tee", type="pil", image_mode="RGB")

    with gr.Accordion("Geometry intermediates", open=False):
        with gr.Row():
            cube_out = gr.Image(label="Isometric cube", type="pil", image_mode="RGB")
            net_out = gr.Image(label="Latin-cross net", type="pil", image_mode="RGB")

    with gr.Accordion("Multi-view / Zero123++", open=False):
        gr.Markdown(
            f"""
**Model:** `{ZERO123_MODEL}` via diffusers `Zero123PlusPipeline`
(`custom_pipeline=sudo-ai/zero123plus-pipeline`).

The pipeline emits a **640×960** image: **2 columns × 3 rows** of 320×320 tiles
(row-major). v1.2 cameras and Saturn face slots:

| Tile | Azimuth | Elevation | Face |
| --- | --- | --- | --- |
| 0 (r0c0) | 30° | +20° | FRONT |
| 1 (r0c1) | 90° | −10° | RIGHT |
| 2 (r1c0) | 150° | +20° | TOP |
| 3 (r1c1) | 210° | −10° | BACK |
| 4 (r2c0) | 270° | +20° | LEFT |
| 5 (r2c1) | 330° | −10° | BOTTOM |

TOP/BOTTOM are the leftover +20°/−10° views — Zero123++ does not output true
orthographic +Z/−Z. Flag `SATURN_ENABLE_MULTIVIEW` is **{'on' if ENABLE_MULTIVIEW else 'off'}**;
weights loaded: **{'yes' if pipeline_loaded() else 'no'}**.

**License:** Zero123++ code is Apache 2.0; **weights are CC-BY-NC 4.0** (no
commercial product pipeline). The Hub checkpoint is not gated. PIL geometry
does not use this model.
            """,
            elem_classes=["stub-note"],
        )

    with gr.Accordion("Printify (coming soon)", open=False):
        gr.Markdown(
            """
`printify/client.py` exposes `PrintifyClient.create_product` against
`POST /v1/shops/{shop_id}/products.json`. It reads `PRINTIFY_API_TOKEN` and
`PRINTIFY_SHOP_ID` and raises if they are missing. **The generate button does
not call Printify.** Wire it here in v2 after print-area mapping exists.
            """,
            elem_classes=["stub-note"],
        )
        gr.Button("Create Printify product", interactive=False)
        gr.Markdown(
            f"Flag `SATURN_ENABLE_PRINTIFY` is currently **{'on' if ENABLE_PRINTIFY else 'off'}**.",
            elem_classes=["stub-note"],
        )

    examples = []
    if Path(EXAMPLE_STARFIELD).exists():
        examples.append([str(EXAMPLE_STARFIELD), "auto", False, ENGINE_PIL])
    if Path(EXAMPLE_BAND).exists():
        examples.append([str(EXAMPLE_BAND), "wrap", False, ENGINE_PIL])
    if examples:
        gr.Examples(
            examples=examples,
            inputs=[image_in, face_mode, remove_bg, engine],
            label="Example textures",
        )

    _outputs = [combined_out, front_out, back_out, cube_out, net_out, status]
    generate_btn.click(
        fn=generate_auto,
        inputs=[image_in, face_mode, remove_bg, engine],
        outputs=_outputs,
        api_name="generate",
    )
    # Registered so ZeroGPU's startup scan sees a @spaces.GPU handler.
    # Direct API: /generate_multiview. The visible button uses generate_auto,
    # which calls this only when the engine is Zero123++ (PIL stays off-GPU).
    with gr.Row(visible=False):
        _mv_btn = gr.Button(visible=False)
        _mv_btn.click(
            fn=generate_multiview,
            inputs=[image_in, face_mode, remove_bg],
            outputs=_outputs,
            api_name="generate_multiview",
        )


if __name__ == "__main__":
    # On Spaces, let the platform bind the port; local default stays 7865.
    demo.queue()
    kwargs = {"show_error": True}
    if not os.getenv("SPACE_ID"):
        kwargs.update(
            server_name="0.0.0.0",
            server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7865")),
        )
    demo.launch(**kwargs)
