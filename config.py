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
DEFAULT_FACE_MODE: str = os.getenv("SATURN_FACE_MODE", "single").strip().lower()

TEE_FRONT_PATH: Path = _path("SATURN_TEE_FRONT", ASSETS_DIR / "tee_front.png")
TEE_BACK_PATH: Path = _path("SATURN_TEE_BACK", ASSETS_DIR / "tee_back.png")

ENABLE_MULTIVIEW: bool = _bool("SATURN_ENABLE_MULTIVIEW", False)
ENABLE_PRINTIFY: bool = _bool("SATURN_ENABLE_PRINTIFY", False)

PRINTIFY_API_TOKEN: str | None = os.getenv("PRINTIFY_API_TOKEN") or None
PRINTIFY_SHOP_ID: str | None = os.getenv("PRINTIFY_SHOP_ID") or None
PRINTIFY_API_BASE: str = os.getenv("PRINTIFY_API_BASE", "https://api.printify.com").rstrip("/")

MOCKUP_BACKGROUND: tuple[int, int, int] = (214, 214, 214)
TEE_FABRIC: tuple[int, int, int] = (244, 244, 241)
CUBE_EDGE: tuple[int, int, int, int] = (236, 236, 236, 230)
NET_BORDER: tuple[int, int, int, int] = (255, 255, 255, 255)
