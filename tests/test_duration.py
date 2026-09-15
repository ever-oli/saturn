"""Duration estimator stays in the 60–90s ZeroGPU window."""

from __future__ import annotations

import os
import unittest

os.environ.setdefault("SATURN_SKIP_MODEL_LOAD", "1")

from app import estimate_gpu_duration  # noqa: E402
from flux.prompts import LAYOUT_SEPARATE, LAYOUT_SIDE  # noqa: E402


class DurationTests(unittest.TestCase):
    def test_side_by_side_four_steps_is_official_window(self) -> None:
        d = estimate_gpu_duration(None, "", LAYOUT_SIDE, False, 0, True, 4)
        self.assertEqual(d, 85)
        self.assertGreaterEqual(d, 60)
        self.assertLessEqual(d, 90)

    def test_separate_plus_print_caps_at_ninety(self) -> None:
        d = estimate_gpu_duration(object(), "notes", LAYOUT_SEPARATE, True, 0, True, 8)
        self.assertEqual(d, 90)


if __name__ == "__main__":
    unittest.main()
