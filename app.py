"""Saturn — Black Cube of Saturn streetwear customizer.

Primary path: FLUX.2-klein-9B product mockups on ZeroGPU (text-to-image and
reference-conditioned design). Optional upload is passed as Klein ``image=``
plus an optional BLIP caption — never stamped as six stickers. PIL geometry
lives under Experimental. Printify is a stub. 9B is FLUX Non-Commercial.
"""

from __future__ import annotations

import os

# ZeroGPU: expandable segments before torch; spaces before any CUDA-touching import.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import spaces  # noqa: E402  MUST be before torch / diffusers / transformers

import gradio as gr  # noqa: E402
from PIL import Image  # noqa: E402

from config import (  # noqa: E402
    DEFAULT_FACE_MODE,
    ENABLE_FLUX,
    ENABLE_PRINTIFY,
    FLUX_MODEL_ID,
    FLUX_STEPS,
    SKIP_MODEL_LOAD,
)
from flux.caption import caption_image, caption_load_error, captioner_loaded, load_captioner  # noqa: E402
from flux.compose import side_by_side  # noqa: E402
from flux.engine import (  # noqa: E402
    flux_load_error,
    flux_loaded,
    load_flux_pipeline,
    run_flux,
)
from flux.prompts import (  # noqa: E402
    LAYOUT_SEPARATE,
    LAYOUT_SIDE,
    PromptJob,
    build_job,
    format_job_markdown,
)
from geometry.bg import rembg_available  # noqa: E402
from geometry.pipeline import generate as generate_geometry  # noqa: E402

load_captioner()
load_flux_pipeline()

# Official Klein example is 1024²; BFL Space caps at 1024.
PAIR_SIZE = (1024, 1024)
TEE_SIZE = (1024, 1024)
PRINT_SIZE = (1024, 1024)

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


def estimate_gpu_duration(
    image,
    notes,
    layout,
    print_assets,
    seed,
    randomize_seed,
    num_inference_steps,
    *args,
    **kwargs,
) -> int:
    """60–90s. Official BFL Klein Space uses 85s for one distilled call."""
    n = 2 if (layout or LAYOUT_SIDE) == LAYOUT_SEPARATE else 1
    if print_assets:
        n += 2
    if n <= 1:
        return 85
    return 90


def _engine_banner() -> str:
    if flux_loaded():
        flux = f"FLUX.2-klein-9B ready (`{FLUX_MODEL_ID}`)"
    elif SKIP_MODEL_LOAD:
        flux = "Flux skipped (`SATURN_SKIP_MODEL_LOAD`)"
    elif not ENABLE_FLUX:
        flux = "Flux disabled (`SATURN_ENABLE_FLUX=false`)"
    else:
        err = flux_load_error() or "not loaded"
        flux = (
            f"Flux not loaded ({err}). Set Space secret **HF_TOKEN** and accept the "
            f"[model license](https://huggingface.co/{FLUX_MODEL_ID})."
        )
    cap = (
        "BLIP captioner ready"
        if captioner_loaded()
        else f"captioner off ({caption_load_error() or 'not loaded'}) — notes still apply"
    )
    return f"{flux}. {cap}."


def _job_from_inputs(
    image: Image.Image | None,
    notes: str,
    layout: str,
) -> tuple[PromptJob, str | None]:
    caption = caption_image(image) if image is not None else None
    return (
        build_job(
            notes=notes,
            caption=caption,
            layout=layout,
            has_reference=image is not None,
        ),
        caption,
    )


def preview_prompt(
    image: Image.Image | None,
    notes: str,
    layout: str,
    print_assets: bool,
) -> str:
    job, _caption = _job_from_inputs(image, notes, layout)
    md = format_job_markdown(job, print_assets=bool(print_assets))
    if image is not None and not job.caption:
        md += "\n\n**Caption:** (none — upload was not described; notes/default material used)"
    return md


@spaces.GPU(duration=estimate_gpu_duration)
def generate_flux(
    image: Image.Image | None,
    notes: str,
    layout: str,
    print_assets: bool,
    seed: int,
    randomize_seed: bool,
    num_inference_steps: int,
    progress=gr.Progress(track_tqdm=True),
) -> tuple:
    """ZeroGPU handler: prompt (+ optional Klein ``image=`` ref) → mockup(s)."""
    if not flux_loaded():
        raise gr.Error(
            "FLUX.2-klein-9B is not loaded. On the Space: Settings → Secrets → "
            f"HF_TOKEN, and accept the FLUX Non-Commercial license at "
            f"huggingface.co/{FLUX_MODEL_ID}. "
            f"Detail: {flux_load_error() or 'load was skipped'}."
        )

    import random

    job, caption = _job_from_inputs(image, notes, layout)
    if randomize_seed:
        seed = int(random.randint(0, 2_147_483_647))
    seed = int(seed)
    steps = max(1, int(num_inference_steps or FLUX_STEPS))
    reference = image.convert("RGB") if image is not None else None

    progress(0.05, desc="Building prompt")
    front_im: Image.Image | None = None
    back_im: Image.Image | None = None
    print_front: Image.Image | None = None
    print_back: Image.Image | None = None

    if job.layout == LAYOUT_SEPARATE:
        progress(0.15, desc="Front tee")
        front_im = run_flux(
            job.front_prompt, width=TEE_SIZE[0], height=TEE_SIZE[1],
            num_inference_steps=steps, seed=seed, reference=reference,
        )
        progress(0.45, desc="Back tee")
        back_im = run_flux(
            job.back_prompt, width=TEE_SIZE[0], height=TEE_SIZE[1],
            num_inference_steps=steps, seed=seed + 1, reference=reference,
        )
        combined = side_by_side(front_im, back_im)
    else:
        progress(0.2, desc="Front + back mockup")
        combined = run_flux(
            job.mockup_prompt, width=PAIR_SIZE[0], height=PAIR_SIZE[1],
            num_inference_steps=steps, seed=seed, reference=reference,
        )

    if print_assets:
        progress(0.7, desc="Print-ready cube")
        print_front = run_flux(
            job.print_front_prompt, width=PRINT_SIZE[0], height=PRINT_SIZE[1],
            num_inference_steps=steps, seed=seed + 2, reference=reference,
        )
        progress(0.85, desc="Print-ready net")
        print_back = run_flux(
            job.print_back_prompt, width=PRINT_SIZE[0], height=PRINT_SIZE[1],
            num_inference_steps=steps, seed=seed + 3, reference=reference,
        )

    if reference is not None:
        ref_bit = "Klein `image=[upload]` reference conditioning"
        if caption:
            ref_bit += f" · caption `{caption}`"
    else:
        ref_bit = "text-to-image (no upload)"
    n_calls = 1 if job.layout == LAYOUT_SIDE else 2
    if print_assets:
        n_calls += 2
    status = (
        f"**Engine:** FLUX.2-klein-9B (`{FLUX_MODEL_ID}`) · **steps:** {steps} · "
        f"**guidance:** 1.0 · **seed:** `{seed}` · **calls:** {n_calls} · {ref_bit}\n\n"
        "Front: isometric cube chest graphic. Back: 6-panel Latin-cross cube net. "
        "**License:** FLUX Non-Commercial — demo/research only, not paid merch."
    )
    prompt_md = format_job_markdown(job, print_assets=bool(print_assets))
    return combined, front_im, back_im, print_front, print_back, status, prompt_md


def generate_pil_experimental(
    image: Image.Image | None,
    face_mode: str,
    remove_bg: bool,
) -> tuple:
    if image is None:
        raise gr.Error("Experimental PIL path still needs an upload.")
    result = generate_geometry(
        image, face_mode=face_mode, remove_bg=bool(remove_bg), enable_multiview=False
    )
    status = (
        f"**Path:** experimental PIL geometry · **Face mode:** `{result.face_mode}` · "
        f"**rembg:** {'on' if result.removed_bg else 'off'}\n\n"
        "This is the legacy compositor (not the Flux product look)."
    )
    return result.combined, result.front, result.back, result.cube, result.net, status


with gr.Blocks(title="Saturn", theme=THEME, css=CUSTOM_CSS, head=HEAD) as demo:
    with gr.Column(elem_id="saturn-hero"):
        gr.Markdown(
            f"""
# Saturn
Black Cube of Saturn · *streetwear*

Primary engine is **FLUX.2 [klein] 9B** — text-to-image and multi-reference
editing in one model. Saturn writes a product prompt: isometric cube on the
**front**, Latin-cross cube net on the **back**, white oversized tee, catalog
lighting. Optional upload is Klein **`image=`** reference conditioning (plus a
short caption), never tiled as six stickers. Zero123++ is not a user-facing engine.

**License: [FLUX Non-Commercial](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B).**
Demo and research on this Space are OK. **Do not use 9B outputs for paid Printify
merch.** A later commercial swap is [FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)
(Apache-2.0) — not loaded here.

{_engine_banner()}
            """
        )

    with gr.Row():
        with gr.Column(scale=4):
            image_in = gr.Image(
                label="Reference (optional)",
                type="pil",
                image_mode="RGB",
                sources=["upload", "clipboard"],
                height=280,
            )
            notes = gr.Textbox(
                label="Notes (optional)",
                placeholder="e.g. matte black kiswah, thin gold band, no calligraphy",
                lines=3,
            )
            layout = gr.Radio(
                choices=[
                    ("Side-by-side front + back", LAYOUT_SIDE),
                    ("Separate front and back", LAYOUT_SEPARATE),
                ],
                value=LAYOUT_SIDE,
                label="Mockup layout",
            )
            print_assets = gr.Checkbox(
                label="Also generate print-ready cube + net graphics",
                value=False,
                info="Two extra Klein calls (isolated graphics, no shirt).",
            )
            generate_btn = gr.Button("Generate mockup", variant="primary")
            preview_btn = gr.Button("Preview prompt", variant="secondary")
            status = gr.Markdown(
                "Optional upload + notes, then generate. Klein 9B is the primary action.",
                elem_classes=["stub-note"],
            )
        with gr.Column(scale=6):
            combined_out = gr.Image(
                label="Product mockup", type="pil", image_mode="RGB", height=520
            )

    with gr.Row():
        front_out = gr.Image(label="Front tee (separate layout)", type="pil", image_mode="RGB")
        back_out = gr.Image(label="Back tee (separate layout)", type="pil", image_mode="RGB")

    with gr.Accordion("Print-ready graphics", open=False):
        with gr.Row():
            print_front_out = gr.Image(label="Front cube graphic", type="pil", image_mode="RGB")
            print_back_out = gr.Image(label="Back net graphic", type="pil", image_mode="RGB")

    with gr.Accordion("Built prompt", open=False):
        prompt_out = gr.Markdown(elem_classes=["stub-note"])

    with gr.Accordion("Advanced", open=False):
        seed = gr.Slider(label="Seed", minimum=0, maximum=2_147_483_647, step=1, value=0)
        randomize_seed = gr.Checkbox(label="Randomize seed", value=True)
        num_inference_steps = gr.Slider(
            label="Inference steps (Klein distilled: 4)",
            minimum=1,
            maximum=8,
            step=1,
            value=FLUX_STEPS,
        )

    with gr.Accordion("Experimental — PIL geometry (legacy)", open=False):
        gr.Markdown(
            """
Off by default. Maps an upload onto cube faces with Pillow (the old sticker/compositor
path). **Zero123++ is not offered here** — that engine is demoted and is not loaded.
Use Klein generate for the streetwear product look.
            """,
            elem_classes=["stub-note"],
        )
        _modes = ("auto", "single", "wrap", "emblem", "grid")
        exp_face_mode = gr.Radio(
            choices=[
                ("Auto", "auto"),
                ("Single material", "single"),
                ("Wrap band", "wrap"),
                ("Emblem on black", "emblem"),
                ("Grid 2×3", "grid"),
            ],
            value=DEFAULT_FACE_MODE if DEFAULT_FACE_MODE in _modes else "auto",
            label="Face mapping",
        )
        _rembg = rembg_available()
        exp_remove_bg = gr.Checkbox(
            label="Remove background (rembg)",
            value=False,
            interactive=_rembg,
        )
        exp_btn = gr.Button("Run experimental PIL", variant="secondary")
        exp_status = gr.Markdown("", elem_classes=["stub-note"])
        with gr.Row():
            exp_combined = gr.Image(label="PIL combined", type="pil", image_mode="RGB")
        with gr.Row():
            exp_front = gr.Image(label="PIL front", type="pil", image_mode="RGB")
            exp_back = gr.Image(label="PIL back", type="pil", image_mode="RGB")
        with gr.Row():
            exp_cube = gr.Image(label="Isometric cube", type="pil", image_mode="RGB")
            exp_net = gr.Image(label="Latin-cross net", type="pil", image_mode="RGB")

    with gr.Accordion("Printify (coming soon)", open=False):
        gr.Markdown(
            """
`printify/client.py` exposes `PrintifyClient.create_product` against
`POST /v1/shops/{shop_id}/products.json`. It reads `PRINTIFY_API_TOKEN` and
`PRINTIFY_SHOP_ID` and raises if they are missing. **Generate does not call
Printify.**

**Do not sell merch from this 9B Space.** FLUX.2-klein-9B is
[FLUX Non-Commercial](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B).
For a later paid Printify path, swap the checkpoint to
`black-forest-labs/FLUX.2-klein-4B` (Apache-2.0) — not implemented here.
            """,
            elem_classes=["stub-note"],
        )
        gr.Button("Create Printify product", interactive=False)
        gr.Markdown(
            f"Flag `SATURN_ENABLE_PRINTIFY` is currently **{'on' if ENABLE_PRINTIFY else 'off'}**.",
            elem_classes=["stub-note"],
        )

    gr.Examples(
        examples=[
            [None, "matte black cube, thin gold equatorial band, high contrast", LAYOUT_SIDE, False],
            [None, "dark cosmic stone cube, faint stars, soft oval shadow under the cube", LAYOUT_SIDE, False],
            [None, "black cube, gold band, no calligraphy, oversized boxy tee", LAYOUT_SEPARATE, False],
        ],
        inputs=[image_in, notes, layout, print_assets],
        label="Example notes",
        cache_examples=False,
    )

    generate_btn.click(
        fn=generate_flux,
        inputs=[
            image_in, notes, layout, print_assets,
            seed, randomize_seed, num_inference_steps,
        ],
        outputs=[
            combined_out, front_out, back_out,
            print_front_out, print_back_out, status, prompt_out,
        ],
        api_name="generate",
    )
    preview_btn.click(
        fn=preview_prompt,
        inputs=[image_in, notes, layout, print_assets],
        outputs=prompt_out,
        api_name="preview_prompt",
    )
    exp_btn.click(
        fn=generate_pil_experimental,
        inputs=[image_in, exp_face_mode, exp_remove_bg],
        outputs=[exp_combined, exp_front, exp_back, exp_cube, exp_net, exp_status],
        api_name="generate_pil",
    )


if __name__ == "__main__":
    # On Spaces, let the platform bind the port; local default stays 7865.
    # GRADIO_SSR_MODE=false is the Space env; also pass ssr_mode=False so a
    # local launch matches.
    demo.queue()
    kwargs = {"show_error": True, "ssr_mode": False}
    if not os.getenv("SPACE_ID"):
        kwargs.update(
            server_name="0.0.0.0",
            server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7865")),
        )
    demo.launch(**kwargs)
