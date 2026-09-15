"""Import app without downloading Flux/BLIP; UI must not default to Zero123++."""

from __future__ import annotations

import os
import unittest
from pathlib import Path

os.environ["SATURN_SKIP_MODEL_LOAD"] = "1"

import app  # noqa: E402
from config import ENABLE_FLUX, ENABLE_MULTIVIEW, FLUX_MODEL_ID, SKIP_MODEL_LOAD  # noqa: E402
from flux.engine import flux_loaded  # noqa: E402


class AppImportTests(unittest.TestCase):
    def test_skip_model_load_does_not_hold_flux(self) -> None:
        self.assertTrue(SKIP_MODEL_LOAD)
        self.assertTrue(ENABLE_FLUX)
        self.assertFalse(ENABLE_MULTIVIEW)
        self.assertFalse(flux_loaded())
        self.assertEqual(FLUX_MODEL_ID, "black-forest-labs/FLUX.2-klein-4B")

    def test_primary_api_is_flux_generate(self) -> None:
        self.assertTrue(callable(app.generate_flux))
        self.assertTrue(callable(app.generate_pil_experimental))
        src = Path(app.__file__).read_text(encoding="utf-8")
        self.assertIn('api_name="generate"', src)
        self.assertIn("generate_flux", src)
        self.assertIn("FLUX.2-klein-4B", src)
        self.assertIn("Apache-2.0", src)
        self.assertNotIn("Primary engine is **FLUX.2 [klein] 9B**", src)
        self.assertNotIn("FLUX.1-schnell", src)
        self.assertNotIn("ENGINE_ZERO123", src)
        self.assertNotIn("Cube engine", src)
        self.assertIn("Experimental — PIL geometry", src)
        self.assertIn("Printify", src)
        self.assertNotIn('api_name="generate_multiview"', src)

    def test_launch_kwargs_avoid_spaces_port(self) -> None:
        src = Path(app.__file__).read_text(encoding="utf-8")
        self.assertIn("SPACE_ID", src)
        self.assertIn("ssr_mode", src)


if __name__ == "__main__":
    unittest.main()
