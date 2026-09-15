"""Optional CPU background removal (rembg / u2netp) with a hard fallback."""

from __future__ import annotations

import importlib.util

from PIL import Image

_SESSION = None
_AVAILABLE: bool | None = None
# Lightweight U²-Net portable (~4.4 MB). Never use bria-rmbg (1 GB, non-MIT).
_MODEL = "u2netp"


def rembg_available() -> bool:
    """True when the rembg package is importable (model may still fail later).

    Uses ``find_spec`` so Gradio startup does not load onnxruntime.
    """
    global _AVAILABLE
    if _AVAILABLE is None:
        _AVAILABLE = importlib.util.find_spec("rembg") is not None
    return _AVAILABLE


def try_remove_background(image: Image.Image) -> tuple[Image.Image, bool]:
    """Return ``(image, True)`` after rembg, or the original and ``False``.

    Failures (missing package, missing onnxruntime, model download, etc.)
    never raise — the PIL flatten / emblem path still runs.
    """
    if not rembg_available():
        return image, False
    global _SESSION
    try:
        from rembg import new_session, remove

        if _SESSION is None:
            _SESSION = new_session(_MODEL)
        out = remove(image, session=_SESSION)
        if not isinstance(out, Image.Image):
            out = Image.open(out).convert("RGBA")
        else:
            out = out.convert("RGBA")
        return out, True
    except Exception:
        return image, False
