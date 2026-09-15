---
title: Saturn
emoji: ⬛
colorFrom: gray
colorTo: yellow
sdk: gradio
sdk_version: 6.27.0
app_file: app.py
python_version: "3.12"
suggested_hardware: zero-a10g
startup_duration_timeout: 1h
short_description: Klein-4B cube + Latin-cross tee mockups
---

# Saturn

Open-source streetwear customizer for the **Black Cube of Saturn** motif.
The primary engine is **[FLUX.2-klein-4B](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B)**
on a **ZeroGPU** Space. Klein unifies **text-to-image** and **multi-reference
image editing** — upload a texture/ref, get a designed front+back tee mockup
(not a Zero123++ sticker grid).

- **Front:** isometric / 3D cube graphic on a white oversized tee chest
- **Back:** unfolded 6-panel **Latin-cross** cube net (column of 4, wings on the second square from the top)
- High-contrast minimalist esoteric streetwear — side-by-side front+back **or** separate 1024² images

## License

**Default weights are Apache-2.0** (`FLUX.2-klein-4B`, `gated: false`).
Commercial use and a later Printify path are OK. Saturn itself is MIT.

Gated **[FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)** (FLUX Non-Commercial) is **optional** via Space variable `SATURN_FLUX_MODEL` only if you already have Hub access. **Do not set 9B as default** — the Space must boot without a license click.

If a previous deploy set `SATURN_FLUX_MODEL=black-forest-labs/FLUX.2-klein-9B`, **delete that Space variable** so the 4B default applies. Otherwise you will still get `GatedRepoError` 403.

PIL geometry (the old compositor / sticker path) is under **Experimental**, off by default.
**Zero123++ is not a user-facing engine.** Printify remains an importable stub only and is not called.

## Generate path (Klein 4B)

1. Optional texture / reference photo + short notes.
2. If an image is present:
   - It is passed to `Flux2KleinPipeline` as **`image=[upload]`** — official reference *conditioning* (not img2img, **no `strength`**).
   - BLIP (`Salesforce/blip-image-captioning-base`) may also name materials/colors for the prompt. The photo is **not** stamped onto six cube faces.
3. `flux/prompts.py` builds a product prompt: cube-front + Latin-cross-back, white oversized tee.
4. `Flux2KleinPipeline` on ZeroGPU: `guidance_scale=1.0`, `num_inference_steps=4`, **1024×1024** (Hub example).
5. Optional extra calls: isolated print-ready cube graphic + net graphic (same reference).

```python
from diffusers import Flux2KleinPipeline
import torch

pipe = Flux2KleinPipeline.from_pretrained(
    "black-forest-labs/FLUX.2-klein-4B",
    torch_dtype=torch.bfloat16,
)
pipe.to("cuda")  # ZeroGPU: module scope, string "cuda" only

image = pipe(
    prompt=prompt,
    height=1024,
    width=1024,
    guidance_scale=1.0,
    num_inference_steps=4,
    generator=torch.Generator(device="cuda").manual_seed(0),
    # only when the user uploaded a ref:
    image=[reference_pil],
).images[0]
```

`printify/client.py` is not called from Generate.

## Diffusers install

`Flux2KleinPipeline` is **not** in older PyPI wheels (e.g. `diffusers==0.36.0` only has `Flux2Pipeline`).
The [4B model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) installs **from git**:

```
git+https://github.com/huggingface/diffusers.git
```

That line is in this repo's `requirements.txt`. If a later PyPI release ships Klein, you can pin it instead.

## ZeroGPU / Spaces

Hardware card: **~13GB VRAM** for 4B bf16. ZeroGPU **large** is ~48GB, so 4B fits easily.
Load at **module scope** with `.to("cuda")`. Do not `enable_model_cpu_offload` on ZeroGPU.

`import spaces` is the first CUDA-touching import. The Gradio **Generate mockup** handler is `@spaces.GPU` with duration **60s** for one 4-step call and **85s** when separate tees / print assets stack extra calls.

Do **not** pin `torch`, `gradio`, or `spaces` in `requirements.txt`. Set Space variable `GRADIO_SSR_MODE=false`. `app.py` does not bind a port when `SPACE_ID` is set.

**HF_TOKEN is not required** for 4B. A token is optional (rate limits / private cache). The Space must be usable when 4B loads.

Live Space: [eyov/saturn](https://huggingface.co/spaces/eyov/saturn) (`zero-a10g`).

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install gradio==6.27.0 spaces   # provided by Spaces; needed locally
cp .env.example .env
# Optional: skip Klein + BLIP downloads while hacking UI / prompts
export SATURN_SKIP_MODEL_LOAD=1
python app.py
```

Open [http://127.0.0.1:7865](http://127.0.0.1:7865). Full Klein 4B locally needs a GPU and ~13GB VRAM at bf16. No Hub license click.

Headless smokes (no Flux VRAM):

```bash
python -m unittest discover -s tests -v
python -m geometry.smoke
```

## Space test steps (after this lands on `eyov/saturn`)

1. Confirm Gradio + ZeroGPU (`hf spaces info eyov/saturn --expand runtime`).
2. **Unset** `SATURN_FLUX_MODEL` if it still points at 9B. 4B must be the default.
3. Confirm env `GRADIO_SSR_MODE=false`. HF_TOKEN is optional.
4. Logs: `Flux2KleinPipeline.from_pretrained("black-forest-labs/FLUX.2-klein-4B")` succeeds — **no `GatedRepoError`**.
5. Open the app: **Zero123++ / PIL face-mode must not be the default path**. Primary control is **Generate mockup**. Hero must show **Apache-2.0** / 4B.
6. Empty upload + example notes → catalog pair at 1024², not a checkerboard cube.
7. Upload a reference. Prompt accordion should mention **`image=[upload]`**.
8. Optional: print-ready graphics (two extra GPU calls).
9. Experimental accordion still runs PIL; Printify button stays disabled.

## Layout

| Path | Role |
| --- | --- |
| `app.py` | Gradio UI; `@spaces.GPU` on Klein generate |
| `flux/prompts.py` | Cube-front + Latin-cross-back prompt builder |
| `flux/caption.py` | Optional BLIP caption (CPU); Klein `image=` is the real ref path |
| `flux/engine.py` | `Flux2KleinPipeline` load + infer |
| `geometry/` | Experimental PIL cube/net/tee (legacy) |
| `geometry/multiview.py` | Dead Zero123++ backend (not loaded by UI) |
| `printify/` | `create_product` stub |
| `assets/` | Blank tees, example textures, demo outputs |

## License

Saturn is MIT. **FLUX.2-klein-4B weights are Apache-2.0.** BLIP is BSD.
Optional 9B weights remain FLUX Non-Commercial if `SATURN_FLUX_MODEL` is overridden.
Legacy Zero123++ weights remain CC-BY-NC 4.0 if that code is ever invoked outside the UI.
