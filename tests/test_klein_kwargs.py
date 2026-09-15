"""Flux2KleinPipeline call kwargs: official image= list, never strength."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("SATURN_SKIP_MODEL_LOAD", "1")

try:
    import torch  # noqa: F401
except ImportError:
    torch = None  # type: ignore


@unittest.skipUnless(torch is not None, "torch not installed (VRAM-free smoke skips Klein call kwargs)")
class KleinKwargsTests(unittest.TestCase):
    def test_text_to_image_omits_image_and_strength(self) -> None:
        from flux.engine import build_call_kwargs

        kw = build_call_kwargs(
            "a cube on a tee",
            width=1024,
            height=1024,
            num_inference_steps=4,
            seed=0,
            reference=None,
            device="cpu",
        )
        self.assertEqual(kw["guidance_scale"], 1.0)
        self.assertEqual(kw["num_inference_steps"], 4)
        self.assertEqual(kw["width"], 1024)
        self.assertEqual(kw["height"], 1024)
        self.assertNotIn("image", kw)
        self.assertNotIn("strength", kw)

    def test_upload_is_list_of_rgb_pil(self) -> None:
        from PIL import Image

        from flux.engine import build_call_kwargs

        ref = Image.new("RGBA", (32, 32), (10, 20, 30, 255))
        kw = build_call_kwargs(
            "use the reference for cube materials",
            width=1024,
            height=1024,
            num_inference_steps=4,
            seed=1,
            reference=ref,
            device="cpu",
        )
        self.assertIn("image", kw)
        self.assertIsInstance(kw["image"], list)
        self.assertEqual(len(kw["image"]), 1)
        self.assertEqual(kw["image"][0].mode, "RGB")
        self.assertNotIn("strength", kw)


if __name__ == "__main__":
    unittest.main()
