"""Printify API client skeleton.

Endpoint (create product):
    POST {base}/v1/shops/{shop_id}/products.json

Auth:
    Authorization: Bearer $PRINTIFY_API_TOKEN

This module is importable without credentials. ``create_product`` raises a
clear ``PrintifyConfigError`` when ``PRINTIFY_API_TOKEN`` or
``PRINTIFY_SHOP_ID`` is missing. It is not invoked from ``app.py`` in v1.
"""

from __future__ import annotations

from typing import Any

import requests

from config import PRINTIFY_API_BASE, PRINTIFY_API_TOKEN, PRINTIFY_SHOP_ID

# Re-read via constructor defaults so tests can pass overrides without env.


class PrintifyError(RuntimeError):
    """Raised when the Printify API returns an error response."""


class PrintifyConfigError(PrintifyError):
    """Raised when shop id / token are missing before a live call."""


class PrintifyClient:
    def __init__(
        self,
        token: str | None = None,
        shop_id: str | None = None,
        base_url: str | None = None,
        session: requests.Session | None = None,
        timeout: float = 30.0,
    ) -> None:
        self.token = token if token is not None else PRINTIFY_API_TOKEN
        self.shop_id = shop_id if shop_id is not None else PRINTIFY_SHOP_ID
        self.base_url = (base_url or PRINTIFY_API_BASE).rstrip("/")
        self.session = session or requests.Session()
        self.timeout = timeout

    def create_product(self, payload: dict[str, Any]) -> dict[str, Any]:
        """POST /v1/shops/{shop_id}/products.json

        ``payload`` should follow Printify's catalog schema (title, blueprint_id,
        print_provider_id, variants, print_areas). This v1 skeleton sends the
        JSON body as-is; it does not upload mockup images or map print areas.
        """
        token, shop_id = self._require_credentials()
        url = f"{self.base_url}/v1/shops/{shop_id}/products.json"
        headers = {
            "Authorization": f"Bearer {token}",
            "Content-Type": "application/json; charset=utf-8",
            "User-Agent": "saturn-streetwear/0.1",
        }
        response = self.session.post(
            url,
            json=payload,
            headers=headers,
            timeout=self.timeout,
        )
        if response.status_code >= 400:
            raise PrintifyError(
                f"Printify create_product failed ({response.status_code}): "
                f"{response.text[:500]}"
            )
        data = response.json()
        if not isinstance(data, dict):
            raise PrintifyError("Printify create_product returned a non-object JSON body.")
        return data

    def product_payload_skeleton(
        self,
        title: str = "Saturn Cube Tee",
        description: str = "Black Cube of Saturn — isometric cube front, Latin-cross net back.",
    ) -> dict[str, Any]:
        """Documented placeholder body. Fill blueprint / variant IDs before posting."""
        return {
            "title": title,
            "description": description,
            "blueprint_id": None,
            "print_provider_id": None,
            "tags": ["saturn", "streetwear", "cube", "esoteric"],
            "variants": [],
            "print_areas": [
                {
                    "variant_ids": [],
                    "placeholders": [
                        {"position": "front", "images": []},
                        {"position": "back", "images": []},
                    ],
                }
            ],
        }

    def _require_credentials(self) -> tuple[str, str]:
        missing: list[str] = []
        if not self.token:
            missing.append("PRINTIFY_API_TOKEN")
        if not self.shop_id:
            missing.append("PRINTIFY_SHOP_ID")
        if missing:
            raise PrintifyConfigError(
                "Printify credentials are not configured. Set "
                + " and ".join(missing)
                + " in the environment (see .env.example). "
                "The Saturn UI does not call Printify in v1."
            )
        return str(self.token), str(self.shop_id)
