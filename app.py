"""Saturn — Black Cube of Saturn streetwear customizer.

Hugging Face Spaces entrypoint. CPU Basic: PIL geometry only.
Multi-view diffusion / SF3D and Printify live calls are stubbed.
"""

from __future__ import annotations

import os
from pathlib import Path

import spaces  # ZeroGPU requires ≥1 @spaces.GPU fn; PIL work stays outside it
import gradio as gr
from PIL import Image

from config import DEFAULT_FACE_MODE, ENABLE_MULTIVIEW, ENABLE_PRINTIFY
from geometry.bg import rembg_available
from geometry.pipeline import generate
from geometry.tees import ensure_example_band, ensure_example_starfield, ensure_tee_templates

ensure_tee_templates()
EXAMPLE_STARFIELD = ensure_example_starfield()
EXAMPLE_BAND = ensure_example_band()

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



@spaces.GPU(duration=1)
def _noop_zerogpu() -> None:
    """ZeroGPU startup scan requires ≥1 decorated function; never called."""
    return None


def _run(
    image: Image.Image | None,
    face_mode: str,
    remove_bg: bool,
) -> tuple[Image.Image | None, Image.Image | None, Image.Image | None, Image.Image | None, Image.Image | None, str]:
    if image is None:
        raise gr.Error("Upload an image — Saturn maps it onto a cube and unfolds the net.")

    result = generate(image, face_mode=face_mode, remove_bg=bool(remove_bg))
    engine = "multi-view diffusion (stub hit — unexpected)" if result.used_multiview else "PIL geometry"
    rembg_note = "on" if result.removed_bg else "off"
    status = (
        f"**Path:** {engine} · **Face mode:** `{face_mode}` → `{result.face_mode}` · "
        f"**rembg:** {rembg_note}\n\n"
        "Front is an isometric cube on the chest. Back is the 6-face Latin-cross net "
        "(column of 4; wings on the second square from the top)."
    )
    return result.combined, result.front, result.back, result.cube, result.net, status


with gr.Blocks(title="Saturn", theme=THEME, css=CUSTOM_CSS, head=HEAD) as demo:
    with gr.Column(elem_id="saturn-hero"):
        gr.Markdown(
            """
# Saturn
Black Cube of Saturn · *streetwear customizer*

Upload any image. v1 builds a chest cube and unfolds it into a Latin-cross net,
then composites both onto blank oversized tees. Faces are opaque materials or
emblems (never unresolved alpha). v1 is PIL on ZeroGPU hosting (no GPU quota burned).
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
            remove_bg = gr.Checkbox(
                label="Remove background (rembg)",
                value=_rembg,
                info=(
                    "Cuts the subject out (u2netp, CPU) when Auto picks emblem or you "
                    "choose Emblem. Textures skip rembg. Falls back to flatten if rembg is unavailable."
                    if _rembg
                    else "rembg is not installed in this runtime — alpha is flattened onto #0a0a0a instead."
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

    with gr.Accordion("Multi-view diffusion (coming soon)", open=False):
        gr.Markdown(
            """
This is the plug-in point for **Zero123++ / SV3D / SF3D** so six cube faces stay
view-consistent. v1 never loads those models — `geometry/multiview.py` returns
`None` and the PIL path runs instead. Set `SATURN_ENABLE_MULTIVIEW=true` later
when a GPU Space is attached; the flag is ignored until the stub is replaced.
            """,
            elem_classes=["stub-note"],
        )
        gr.Dropdown(
            choices=["PIL geometry (v1)", "Multi-view / SF3D (stub)"],
            value="PIL geometry (v1)",
            label="Cube engine",
            interactive=False,
        )
        gr.Markdown(
            f"Flag `SATURN_ENABLE_MULTIVIEW` is currently **{'on' if ENABLE_MULTIVIEW else 'off'}**.",
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
        examples.append([str(EXAMPLE_STARFIELD), "auto", False])
    if Path(EXAMPLE_BAND).exists():
        examples.append([str(EXAMPLE_BAND), "wrap", False])
    if examples:
        gr.Examples(
            examples=examples,
            inputs=[image_in, face_mode, remove_bg],
            label="Example textures",
        )

    generate_btn.click(
        fn=_run,
        inputs=[image_in, face_mode, remove_bg],
        outputs=[combined_out, front_out, back_out, cube_out, net_out, status],
        api_name="generate",
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
