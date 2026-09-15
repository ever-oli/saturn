"""CPU compose helper for separate front/back Flux outputs."""

from __future__ import annotations

import unittest

from PIL import Image

from flux.compose import side_by_side


class ComposeTests(unittest.TestCase):
    def test_side_by_side_rgb(self) -> None:
        a = Image.new("RGB", (40, 50), (10, 10, 10))
        b = Image.new("RGB", (30, 50), (200, 200, 200))
        out = side_by_side(a, b, gap=10)
        self.assertEqual(out.mode, "RGB")
        self.assertEqual(out.size, (80, 50))


if __name__ == "__main__":
    unittest.main()
