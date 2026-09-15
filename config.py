"""Env-based settings for Saturn. Safe to import without a .env file."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

ROOT = Path(__file__).resolve().parent
ASSETS_DIR = ROOT / "assets"
EXAMPLES_DIR = ASSETS_DIR / "examples"
DEMO_DIR = ASSETS_DIR / "demo"


def _bool(name: str, default: bool = False) -> bool:
    raw = os.getenv(name)
    if raw is None:
        return default
    return raw.strip().lower() in {"1", "true", "yes", "on"}


def _int(name: str, default: int) -> int:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    return int(raw)


def _path(name: str, default: Path) -> Path:
    raw = os.getenv(name)
    if raw is None or raw.strip() == "":
        return default
    path = Path(raw)
    return path if path.is_absolute() else ROOT / path


PANEL_SIZE: int = _int("SATURN_PANEL_SIZE", 512)
NET_GAP: int = _int("SATURN_NET_GAP", 3)
DEFAULT_FACE_MODE: str = os.getenv("SATURN_FACE_MODE", "auto").strip().lower()

TEE_FRONT_PATH: Path = _path("SATURN_TEE_FRONT", ASSETS_DIR / "tee_front.png")
TEE_BACK_PATH: Path = _path("SATURN_TEE_BACK", ASSETS_DIR / "tee_back.png")

# Primary engine: FLUX.2-klein-4B (Apache-2.0, ungated). Default on.
# Optional: SATURN_FLUX_MODEL=black-forest-labs/FLUX.2-klein-9B if the Space
# owner has accepted that gated Non-Commercial license. Do not set 9B as default.
ENABLE_FLUX: bool = _bool("SATURN_ENABLE_FLUX", True)
SKIP_MODEL_LOAD: bool = _bool("SATURN_SKIP_MODEL_LOAD", False)
DEFAULT_FLUX_MODEL: str = "black-forest-labs/FLUX.2-klein-4B"
FLUX_MODEL_ID: str = os.getenv("SATURN_FLUX_MODEL", DEFAULT_FLUX_MODEL).strip() or DEFAULT_FLUX_MODEL
FLUX_STEPS: int = _int("SATURN_FLUX_STEPS", 4)
FLUX_GUIDANCE: float = float(os.getenv("SATURN_FLUX_GUIDANCE", "1.0") or "1.0")
FLUX_MAX_SEQ: int = _int("SATURN_FLUX_MAX_SEQ", 512)
# 4B is ~13GB / 4-step; one call 60s, stacked calls up to 85s.
FLUX_GPU_DURATION: int = _int("SATURN_FLUX_DURATION", 60)


def is_gated_klein_model(model_id: str | None = None) -> bool:
    """True for the gated 9B / 9b-fp8 family. 4B is ungated Apache-2.0."""
    mid = (model_id if model_id is not None else FLUX_MODEL_ID).lower()
    return "9b" in mid.replace("_", "-")

CAPTION_MODEL_ID: str = os.getenv(
    "SATURN_CAPTION_MODEL", "Salesforce/blip-image-captioning-base"
).strip()

# Legacy Zero123++ / PIL geometry. Off by default — not the product path.
ENABLE_MULTIVIEW: bool = _bool("SATURN_ENABLE_MULTIVIEW", False)
ENABLE_PRINTIFY: bool = _bool("SATURN_ENABLE_PRINTIFY", False)

ZERO123_MODEL: str = os.getenv("SATURN_ZERO123_MODEL", "sudo-ai/zero123plus-v1.2").strip()
ZERO123_CUSTOM_PIPELINE: str = os.getenv(
    "SATURN_ZERO123_PIPELINE", "sudo-ai/zero123plus-pipeline"
).strip()
ZERO123_STEPS: int = _int("SATURN_ZERO123_STEPS", 36)
ZERO123_GUIDANCE: float = float(os.getenv("SATURN_ZERO123_GUIDANCE", "4.0") or "4.0")
ZERO123_GPU_DURATION: int = _int("SATURN_ZERO123_DURATION", 90)

PRINTIFY_API_TOKEN: str | None = os.getenv("PRINTIFY_API_TOKEN") or None
PRINTIFY_SHOP_ID: str | None = os.getenv("PRINTIFY_SHOP_ID") or None
PRINTIFY_API_BASE: str = os.getenv("PRINTIFY_API_BASE", "https://api.printify.com").rstrip("/")

MOCKUP_BACKGROUND: tuple[int, int, int] = (214, 214, 214)
TEE_FABRIC: tuple[int, int, int] = (244, 244, 241)
# Opaque fill for cutouts / emblem grounds / unresolved alpha (Black Cube motif).
FACE_BACKGROUND: tuple[int, int, int] = (10, 10, 10)
CUBE_EDGE: tuple[int, int, int, int] = (248, 248, 248, 255)
NET_BORDER: tuple[int, int, int, int] = (255, 255, 255, 255)
NET_SEPARATOR: tuple[int, int, int, int] = (236, 236, 236, 255)
