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

v1 is **Pillow geometry** hosted on a **ZeroGPU** Space slot (free-account path). Generation stays on CPU — a no-op `@spaces.GPU` satisfies ZeroGPU startup without burning quota. Optional `rembg` (u2netp, CPU) can cut a subject out for emblem mode; if it is missing, alpha is flattened onto `#0a0a0a`.

## v1 generate path

1. Derive six **opaque** face textures from the upload (`auto` / `single` / `wrap` / `emblem` / `grid`) — see `geometry/faces.py`. Unresolved alpha is never left for Gradio’s transparency checker.
2. Build a Latin-cross net with opaque panels and thin crisp separators (`geometry/net.py`).
3. Render an approximate isometric cube with the same faces, white edges, and a soft elliptical pedestal shadow (`geometry/cube.py`).
4. Place the cube on the chest and the net on the back of blank tee templates (`geometry/mockup.py`). Graphics are alpha-composited onto the fabric; returned images are RGB.
5. Return a side-by-side mockup plus separate front/back crops.

`geometry/multiview.py` is the seam for Zero123++ / SV3D / SF3D. It returns `None` today so the PIL path always runs. Printify is an importable stub only (`printify/client.py`); the UI does not call it.

## Local run

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # optional
python app.py
```

Then open [http://127.0.0.1:7865](http://127.0.0.1:7865). First launch writes blank tee placeholders and a starfield example under `assets/` if they are missing.

Headless smoke test (no Gradio) — writes `assets/demo/` PNGs and fails if checker-like regions remain:

```bash
python -m geometry.smoke
# or
python -c "from geometry.pipeline import write_demo; print(write_demo())"
```

## Push to a Hugging Face Space

1. Create a Gradio Space on CPU Basic: [huggingface.co/new-space](https://huggingface.co/new-space) (SDK: Gradio, hardware: CPU basic).
2. Add this repo as a remote and push:

```bash
huggingface-cli login
git remote add space https://huggingface.co/spaces/<you>/saturn
git push space main
```

Or with the `hf` CLI:

```bash
hf auth login
hf repos create <you>/saturn --type space --space-sdk gradio --public
git remote add space https://huggingface.co/spaces/<you>/saturn
git push space main
```

Spaces reads the YAML frontmatter in this README (`sdk: gradio`, `app_file: app.py`, `python_version: 3.12`). Do not attach a GPU for v1. Optional secrets (`PRINTIFY_API_TOKEN`, `PRINTIFY_SHOP_ID`) are unused until Printify is wired.

## Face mapping

| Mode | Behavior |
| --- | --- |
| `auto` (default) | Transparency or cutout → `emblem`. Aspect ≥ 1.4 **or** a horizontal gold/chroma band (kiswah) → `wrap`. Else `single` material. |
| `single` | Center-square crop as albedo on all six faces, with distinct per-face lighting (top bright, left dark, right mid). |
| `wrap` | Horizontal equator around LEFT → FRONT → RIGHT → BACK; top/bottom are fabric crops. |
| `emblem` | Subject scaled on solid `#0a0a0a` (print-style). Optional rembg when the checkbox is on. |
| `grid` | 2×3 crop grid, one cell per face. |

## Layout

| Path | Role |
| --- | --- |
| `app.py` | Gradio Blocks UI |
| `config.py` | Env-based settings and feature flags |
| `geometry/` | Faces, net, isometric cube, tee templates, compositing |
| `printify/` | `create_product` stub (`POST /v1/shops/{shop_id}/products.json`) |
| `assets/` | Blank tees, example texture, demo outputs |

## License

MIT
