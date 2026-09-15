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
short_description: Klein-9B cube + Latin-cross tee mockups
---

# Saturn

Open-source streetwear customizer for the **Black Cube of Saturn** motif.
The primary engine is **[FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B)**
on a **ZeroGPU** Space. Klein unifies **text-to-image** and **multi-reference
image editing** in one architecture — upload a texture/ref, get a designed
front+back tee mockup (not a Zero123++ sticker grid).

- **Front:** isometric / 3D cube graphic on a white oversized tee chest
- **Back:** unfolded 6-panel **Latin-cross** cube net (column of 4, wings on the second square from the top)
- High-contrast minimalist esoteric streetwear — side-by-side front+back **or** separate 1024² images

## License (read this)

**FLUX.2-klein-9B is [FLUX Non-Commercial](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B).**
This Space is for **demo and research**. **Do not use 9B outputs for paid Printify merch or other commercial product pipelines.**

A later commercial swap (not implemented here) is
[`black-forest-labs/FLUX.2-klein-4B`](https://huggingface.co/black-forest-labs/FLUX.2-klein-4B) (**Apache-2.0**).
Saturn itself remains MIT.

PIL geometry (the old compositor / sticker path) is under **Experimental**, off by default.
**Zero123++ is not a user-facing engine.** Printify remains an importable stub only and is not called.

## Generate path (Klein 9B)

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
    "black-forest-labs/FLUX.2-klein-9B",
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
The [model card](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) and the official
[BFL Space](https://huggingface.co/spaces/black-forest-labs/FLUX.2-klein-9B) install **from git**:

```
git+https://github.com/huggingface/diffusers.git
```

That line is in this repo's `requirements.txt`. If a later PyPI release ships Klein, you can pin it instead.

## ZeroGPU / Spaces

Hardware card: **~29GB VRAM** for 9B bf16. ZeroGPU **large** is ~48GB, so full 9B should fit.
Load at **module scope** with `.to("cuda")`. Do not `enable_model_cpu_offload` on ZeroGPU.

If 9B still OOMs, set Space variable `SATURN_FLUX_MODEL=black-forest-labs/FLUX.2-klein-9b-fp8` (fp8 variant). Default stays the full 9B.

`import spaces` is the first CUDA-touching import. The Gradio **Generate mockup** handler is `@spaces.GPU` with duration **85s** for one call (matches the official Klein Space) and **90s** when separate tees / print assets stack extra calls.

Do **not** pin `torch`, `gradio`, or `spaces` in `requirements.txt`. Set Space variable `GRADIO_SSR_MODE=false`. `app.py` does not bind a port when `SPACE_ID` is set.

### Gated model + `HF_TOKEN`

The 9B repo is **gated (auto)**. For the Space to download weights:

1. Visit [black-forest-labs/FLUX.2-klein-9B](https://huggingface.co/black-forest-labs/FLUX.2-klein-9B) while logged in as the Space owner and **Accept** the FLUX Non-Commercial license.
2. Create a Hub token that can read that repo.
3. Add Space secret **`HF_TOKEN`** (or `HUGGING_FACE_HUB_TOKEN`).

Without this, the UI boots but Generate explains that Klein did not load.

Live Space: [eyov/saturn](https://huggingface.co/spaces/eyov/saturn) (`zero-a10g`).

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install gradio==6.27.0 spaces   # provided by Spaces; needed locally
cp .env.example .env
# Optional: skip 9B + BLIP downloads while hacking UI / prompts
export SATURN_SKIP_MODEL_LOAD=1
python app.py
```

Open [http://127.0.0.1:7865](http://127.0.0.1:7865). Full Klein locally needs a GPU, Hub login (accepted license), and ~29GB VRAM at bf16.

Headless smokes (no Flux VRAM):

```bash
python -m unittest discover -s tests -v
python -m geometry.smoke
```

## Space test steps (after this lands on `eyov/saturn`)

1. Confirm Gradio + ZeroGPU (`hf spaces info eyov/saturn --expand runtime`).
2. Confirm secret `HF_TOKEN` is set and the owner **accepted the 9B license**.
3. Confirm env `GRADIO_SSR_MODE=false`.
4. Open the app: **Zero123++ / PIL face-mode must not be the default path**. Primary control is **Generate mockup**. Hero must show **FLUX Non-Commercial**.
5. Empty upload + example notes → catalog pair at 1024², not a checkerboard cube.
6. Upload a reference (fabric / Kaaba-like). Prompt accordion should mention **`image=[upload]`**. Cube should follow materials/colors, not a 6-up sticker sheet.
7. Optional: print-ready graphics (two extra GPU calls).
8. `hf spaces logs eyov/saturn --tail 200` — Klein load at startup, no CPU fallback, generate inside `@spaces.GPU`. If OOM, switch to the fp8 repo via `SATURN_FLUX_MODEL`.
9. Experimental accordion still runs PIL; Printify button stays disabled.

Measure wall time of a 4-step 1024² call (with and without a ref) and tighten duration if 85s is generous.

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

Saturn is MIT. **FLUX.2-klein-9B weights are FLUX Non-Commercial.** BLIP is BSD.
Legacy Zero123++ weights remain CC-BY-NC 4.0 if that code is ever invoked outside the UI.
