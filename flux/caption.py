"""Lightweight BLIP captioner — materials/colors from an optional upload.

Runs on CPU so Flux keeps the GPU. Do not stamp the source image onto cube faces;
the caption is prompt text only.
"""

from __future__ import annotations

import traceback
from typing import Any

from PIL import Image

from config import CAPTION_MODEL_ID, SKIP_MODEL_LOAD

_PROCESSOR: Any = None
_MODEL: Any = None
_LOAD_ERROR: str | None = None

_MATERIALS_PREFIX = "the materials and colors of this object are"


def captioner_loaded() -> bool:
    return _MODEL is not None and _PROCESSOR is not None


def caption_load_error() -> str | None:
    return _LOAD_ERROR


def load_captioner() -> bool:
    """Load BLIP at module scope on CPU. Safe to call when Flux is skipped."""
    global _PROCESSOR, _MODEL, _LOAD_ERROR
    if SKIP_MODEL_LOAD:
        _LOAD_ERROR = "SATURN_SKIP_MODEL_LOAD=1"
        return False
    if _MODEL is not None:
        return True
    try:
        from transformers import BlipForConditionalGeneration, BlipProcessor

        processor = BlipProcessor.from_pretrained(CAPTION_MODEL_ID)
        model = BlipForConditionalGeneration.from_pretrained(CAPTION_MODEL_ID)
        model.eval()
        _PROCESSOR = processor
        _MODEL = model
        _LOAD_ERROR = None
        return True
    except Exception as exc:
        _PROCESSOR = None
        _MODEL = None
        _LOAD_ERROR = f"{type(exc).__name__}: {exc}"
        traceback.print_exc()
        print(f"Saturn: BLIP captioner load failed ({_LOAD_ERROR}); Flux still runs without it.")
        return False


def caption_image(image: Image.Image | None) -> str | None:
    """Return a short materials/colors caption, or None if unavailable."""
    if image is None or _MODEL is None or _PROCESSOR is None:
        return None
    try:
        import torch

        rgb = image.convert("RGB")
        inputs = _PROCESSOR(images=rgb, text=_MATERIALS_PREFIX, return_tensors="pt")
        with torch.inference_mode():
            out = _MODEL.generate(**inputs, max_new_tokens=32)
        text = _PROCESSOR.decode(out[0], skip_special_tokens=True)
        text = " ".join(text.split())
        if text.lower().startswith(_MATERIALS_PREFIX):
            text = text[len(_MATERIALS_PREFIX) :].strip(" :,-")
        return text or None
    except Exception:
        traceback.print_exc()
        return None
