"""FLUX.2-klein-9B load + inference (ZeroGPU-friendly).

``import spaces`` must happen in ``app.py`` *before* this module is imported so
the CUDA monkey-patch is in place. Load at module scope with ``.to(device)``.
Do not ``enable_model_cpu_offload`` (fights ZeroGPU packing).

Official call (Hub model card + BFL Space): ``Flux2KleinPipeline``,
``guidance_scale=1.0``, ``num_inference_steps=4``, 1024². Uploads go in
``image=`` as a list of PIL images — reference conditioning, not img2img
``strength`` (that kwarg does not exist on this pipeline).
"""

from __future__ import annotations

import os
import traceback
from typing import Any

from PIL import Image

from config import (
    ENABLE_FLUX,
    FLUX_GUIDANCE,
    FLUX_MODEL_ID,
    SKIP_MODEL_LOAD,
)

_PIPE: Any = None
_LOAD_ERROR: str | None = None
_DEVICE: str = "cpu"

_GIT_DIFFUSERS = "pip install git+https://github.com/huggingface/diffusers.git"


def flux_loaded() -> bool:
    return _PIPE is not None


def flux_load_error() -> str | None:
    return _LOAD_ERROR


def flux_device() -> str:
    return _DEVICE


def hf_token() -> str | None:
    return os.getenv("HF_TOKEN") or os.getenv("HUGGING_FACE_HUB_TOKEN") or None


def _import_klein_pipeline():
    try:
        from diffusers import Flux2KleinPipeline
    except ImportError as exc:
        raise ImportError(
            "Flux2KleinPipeline is not in this diffusers install. Install from "
            f"source: `{_GIT_DIFFUSERS}` (required until a PyPI release ships Klein)."
        ) from exc
    return Flux2KleinPipeline


def load_flux_pipeline() -> bool:
    """Load Flux2KleinPipeline at module scope. Returns False on skip/failure."""
    global _PIPE, _LOAD_ERROR, _DEVICE
    if SKIP_MODEL_LOAD or not ENABLE_FLUX:
        _LOAD_ERROR = (
            "SATURN_SKIP_MODEL_LOAD=1"
            if SKIP_MODEL_LOAD
            else "SATURN_ENABLE_FLUX=false"
        )
        return False
    if _PIPE is not None:
        return True
    try:
        import torch

        Flux2KleinPipeline = _import_klein_pipeline()
        device = "cuda" if torch.cuda.is_available() else "cpu"
        dtype = torch.bfloat16 if device == "cuda" else torch.float32
        kwargs: dict[str, Any] = {"torch_dtype": dtype}
        token = hf_token()
        if token:
            kwargs["token"] = token
        pipe = Flux2KleinPipeline.from_pretrained(FLUX_MODEL_ID, **kwargs)
        # ZeroGPU: string "cuda" only — never cuda:0 / device index.
        pipe.to(device)
        _PIPE = pipe
        _DEVICE = device
        _LOAD_ERROR = None
        return True
    except Exception as exc:
        _PIPE = None
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        print(
            "Saturn: FLUX.2-klein-9B load failed "
            f"({_LOAD_ERROR}). Set Space secret HF_TOKEN and accept the "
            f"FLUX Non-Commercial license at huggingface.co/{FLUX_MODEL_ID}. "
            f"If the class is missing: {_GIT_DIFFUSERS}."
        )
        return False


def build_call_kwargs(
    prompt: str,
    *,
    width: int,
    height: int,
    num_inference_steps: int,
    seed: int,
    reference: Image.Image | None = None,
    device: str | None = None,
) -> dict[str, Any]:
    """Kwargs matching the official Flux2KleinPipeline example + optional ``image``.

    ``image`` is a list of RGB PIL refs (BFL Space). Do not pass ``strength``.
    """
    import torch

    steps = max(1, int(num_inference_steps))
    gen_device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    generator = torch.Generator(device=gen_device).manual_seed(int(seed) & 0x7FFFFFFF)
    kwargs: dict[str, Any] = {
        "prompt": prompt,
        "height": int(height),
        "width": int(width),
        "num_inference_steps": steps,
        "guidance_scale": FLUX_GUIDANCE,
        "generator": generator,
    }
    if reference is not None:
        kwargs["image"] = [reference.convert("RGB")]
    return kwargs


def run_flux(
    prompt: str,
    *,
    width: int,
    height: int,
    num_inference_steps: int,
    seed: int,
    reference: Image.Image | None = None,
) -> Image.Image:
    if _PIPE is None:
        raise RuntimeError(
            "FLUX.2-klein-9B is not loaded. Set Space secret HF_TOKEN, accept the "
            f"FLUX Non-Commercial license ({FLUX_MODEL_ID}), and keep "
            f"SATURN_ENABLE_FLUX=true. Last error: {_LOAD_ERROR or 'not attempted'}."
        )
    kwargs = build_call_kwargs(
        prompt,
        width=width,
        height=height,
        num_inference_steps=num_inference_steps,
        seed=seed,
        reference=reference,
        device=_DEVICE,
    )
    image = _PIPE(**kwargs).images[0]
    return image.convert("RGB")
