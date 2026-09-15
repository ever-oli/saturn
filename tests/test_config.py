"""Default Klein checkpoint is ungated 4B."""

from __future__ import annotations

import unittest

from config import DEFAULT_FLUX_MODEL, FLUX_MODEL_ID, is_gated_klein_model


class ConfigTests(unittest.TestCase):
    def test_default_model_is_ungated_4b(self) -> None:
        self.assertEqual(DEFAULT_FLUX_MODEL, "black-forest-labs/FLUX.2-klein-4B")
        self.assertEqual(FLUX_MODEL_ID, DEFAULT_FLUX_MODEL)
        self.assertFalse(is_gated_klein_model())
        self.assertFalse(is_gated_klein_model("black-forest-labs/FLUX.2-klein-4B"))

    def test_9b_family_is_treated_as_gated(self) -> None:
        self.assertTrue(is_gated_klein_model("black-forest-labs/FLUX.2-klein-9B"))
        self.assertTrue(is_gated_klein_model("black-forest-labs/FLUX.2-klein-9b-fp8"))


if __name__ == "__main__":
    unittest.main()
