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
short_description: Cube-to-cross streetwear t-shirt mockups
---

# Saturn

Open-source streetwear customizer for the **Black Cube of Saturn** motif. Upload any image; Saturn maps it onto a cube and unfolds that cube into a 6-panel **Latin-cross net** (vertical column of 4 squares, left/right wings on the second square from the top), then composites both onto oversized tee mockups.

**PIL geometry** is the default (CPU, no ZeroGPU quota). **Zero123++** (`sudo-ai/zero123plus-v1.2`) is the first real multi-view backend: it runs only when you pick *Multi-view / Zero123++* and the weights are loaded. Optional `rembg` (u2netp, CPU) cuts the subject out for emblem mode and as Zero123++ preprocess.

## Generate path

1. **Faces**
   - PIL: `auto` / `single` / `wrap` / `emblem` / `grid` in `geometry/faces.py` (opaque, no Gradio checker).
   - Zero123++: `geometry/multiview.py` runs `Zero123PlusPipeline` and maps the 6-view grid into cube slots. On failure or when disabled, PIL is used.
2. Latin-cross net with opaque panels (`geometry/net.py`).
3. Isometric cube with white edges and a soft pedestal shadow (`geometry/cube.py`).
4. Chest cube + back net on blank tees (`geometry/mockup.py`). Returned images are RGB.
5. Side-by-side mockup plus front/back crops.

Printify is an importable stub only (`printify/client.py`); the UI does not call it.

## Zero123++ (ZeroGPU)

Load (official API, v1.2 usage is the same as v1.1):

```python
DiffusionPipeline.from_pretrained(
    "sudo-ai/zero123plus-v1.2",
    custom_pipeline="sudo-ai/zero123plus-pipeline",
    torch_dtype=torch.float16,  # bfloat16 on ZeroGPU
)
```

Then `EulerAncestralDiscreteScheduler` with `timestep_spacing="trailing"`, `.to("cuda")` (never `cuda:0`). Default **36** steps, CFG **4.0**, `@spaces.GPU(duration=90)`. Official notes ~28 steps for general objects and 75–100 for delicate detail.

`SATURN_ENABLE_MULTIVIEW` defaults **off** on CPU and **on** when `SPACES_ZERO_GPU` is set. Explicit `true`/`false` always wins. Weights load at `app.py` module scope after `import spaces`. The Gradio-bound PIL path is **not** GPU-decorated; `/generate_multiview` is.

### Grid → cube faces

The pipeline returns one **640×960** image: **2×3** tiles of 320×320, row-major (same split as `SUDO-AI-3D/zero123plus/gradio_app.py`). v1.2 cameras and Saturn slots:

| Tile | Azimuth (rel. input) | Elevation | Face |
| --- | --- | --- | --- |
| 0 (r0 c0) | 30° | +20° | FRONT |
| 1 (r0 c1) | 90° | −10° | RIGHT |
| 2 (r1 c0) | 150° | +20° | TOP |
| 3 (r1 c1) | 210° | −10° | BACK |
| 4 (r2 c0) | 270° | +20° | LEFT |
| 5 (r2 c1) | 330° | −10° | BOTTOM |

These are perspective object views, not an orthographic cube unwrap. TOP/BOTTOM are the leftover +20°/−10° tiles (Zero123++ has no true +Z/−Z). Input is rembg’d when the checkbox is on, then squared onto gray `(127,127,127)` like the official demo.

### License / access

- **Code** (custom pipeline): Apache 2.0.
- **Weights** (`sudo-ai/zero123plus-v1.2`): **CC-BY-NC 4.0** — not for a commercial product pipeline; outputs may still be used freely. You are accountable for generated images.
- Checkpoint is **not gated** (no HF token required to download).
- PIL path does not load or depend on this model.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
pip install gradio==6.27.0 spaces   # provided by Spaces; needed locally
cp .env.example .env   # optional
python app.py
```

Then open [http://127.0.0.1:7865](http://127.0.0.1:7865). First launch writes blank tee placeholders and example textures under `assets/` if they are missing.

Headless smoke (PIL + Zero123++ **grid mapping**, no GPU/weights):

```bash
python -m geometry.smoke
# or
python -c "from geometry.pipeline import write_demo; print(write_demo())"
```

Live Zero123++ inference is not part of smoke (needs a GPU and ~5GB VRAM). On a Space, pick *Multi-view / Zero123++* after the model has loaded.

## Push to a Hugging Face Space

Create a **Gradio + ZeroGPU** Space (`zero-a10g`). `SATURN_ENABLE_MULTIVIEW` defaults on under ZeroGPU. To force it:

```
hf spaces secrets set <you>/saturn SATURN_ENABLE_MULTIVIEW=true
```

Do not pin `torch` / `gradio` / `spaces` in `requirements.txt` (ZeroGPU-managed). Optional Printify secrets stay unused.

## Face mapping (PIL)

| Mode | Behavior |
| --- | --- |
| `auto` (default) | Transparency or cutout → `emblem`. Aspect ≥ 1.4 **or** a horizontal gold/chroma band (kiswah) → `wrap`. Else `single` material. |
| `single` | Center-square crop as albedo on all six faces, with distinct per-face lighting. |
| `wrap` | Horizontal equator around LEFT → FRONT → RIGHT → BACK; top/bottom are fabric crops. |
| `emblem` | Subject scaled on solid `#0a0a0a`. Optional rembg. |
| `grid` | 2×3 crop grid, one cell per face. |

## Layout

| Path | Role |
| --- | --- |
| `app.py` | Gradio Blocks UI; `@spaces.GPU` on Zero123++ only |
| `config.py` | Env-based settings and feature flags |
| `geometry/multiview.py` | Zero123++ load, grid split, face map |
| `geometry/` | Faces, net, isometric cube, tee templates, compositing |
| `printify/` | `create_product` stub |
| `assets/` | Blank tees, example textures, demo outputs |

## License

Saturn is MIT. Zero123++ weights remain CC-BY-NC 4.0 when that engine is used.
