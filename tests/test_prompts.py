"""Prompt builder: cube-front + Latin-cross-back, caption as materials not stickers."""

from __future__ import annotations

import unittest

from flux.prompts import (
    LAYOUT_SEPARATE,
    LAYOUT_SIDE,
    build_job,
    clip_prompt,
    format_job_markdown,
    material_from_caption,
)


def _net_blobs(job) -> str:
    return " ".join(
        (
            job.mockup_prompt,
            job.back_prompt,
            job.print_back_prompt,
        )
    ).lower()


class PromptBuilderTests(unittest.TestCase):
    def test_default_encodes_cube_front_and_latin_cross_back(self) -> None:
        job = build_job()
        blob = job.mockup_prompt.lower()
        self.assertEqual(job.layout, LAYOUT_SIDE)
        self.assertIn("isometric", blob)
        self.assertIn("cube", blob)
        self.assertIn("latin-cross", blob)
        self.assertIn("net", blob)
        self.assertIn("front", blob)
        self.assertIn("back", blob)
        self.assertIn("oversized", blob)
        self.assertIn("tee", blob)
        self.assertIn("no sticker sheet", blob)
        self.assertIn("checkerboard", blob)
        self.assertLessEqual(len(job.mockup_prompt), 1600)

    def test_back_is_literal_unfolded_cube_net(self) -> None:
        job = build_job()
        for blob in (
            job.mockup_prompt.lower(),
            job.back_prompt.lower(),
            job.print_back_prompt.lower(),
        ):
            self.assertIn("laid completely flat", blob)
            self.assertIn("cut along its edges", blob)
            self.assertIn("six", blob)
            self.assertIn("net", blob)
            self.assertIn("latin-cross", blob)
            self.assertIn("generic cross", blob)
            self.assertIn("religious", blob)
            self.assertLessEqual(len(blob), 1600)

    def test_net_negatives_reject_decorative_crosses(self) -> None:
        blob = _net_blobs(build_job())
        self.assertIn("wireframe hexagon", blob)
        self.assertIn("peeled 3d glyph", blob)
        self.assertIn("sacred geometry", blob)
        self.assertIn("no sticker sheet", blob)
        self.assertIn("alpha grid", blob)
        self.assertIn("unfolded net of the exact cube", blob)

    def test_notes_and_caption_are_materials_not_paste(self) -> None:
        job = build_job(
            notes="thin gold band, no calligraphy",
            caption="a black draped cloth shrine with gold trim",
        )
        self.assertIn("gold band", job.material)
        self.assertIn("do not paste the photo", job.material.lower())
        self.assertIn("black draped cloth", job.material)
        self.assertIn("preserve this cube's faces", job.material.lower())
        self.assertNotIn("materials only", job.material.lower())
        self.assertIn(job.material.split(":")[-1].strip()[:20], job.front_prompt)

    def test_separate_layout_uses_front_and_back_prompts(self) -> None:
        job = build_job(layout=LAYOUT_SEPARATE)
        self.assertEqual(job.layout, LAYOUT_SEPARATE)
        self.assertIn("front", job.front_prompt.lower())
        self.assertIn("latin-cross", job.back_prompt.lower())
        self.assertIn("isometric", job.front_prompt.lower())
        self.assertIn("laid completely flat", job.back_prompt.lower())
        self.assertNotIn("two tees side by side", job.mockup_prompt.lower())

    def test_print_ready_prompts_have_no_shirt(self) -> None:
        job = build_job()
        self.assertIn("no t-shirt", job.print_front_prompt.lower())
        self.assertIn("latin-cross", job.print_back_prompt.lower())
        self.assertIn("isometric", job.print_front_prompt.lower())
        self.assertIn("laid completely flat", job.print_back_prompt.lower())
        self.assertIn("six", job.print_back_prompt.lower())

    def test_clip_prompt_respects_budget(self) -> None:
        long = "cube " * 400
        out = clip_prompt(long, max_chars=80)
        self.assertLessEqual(len(out), 80)

    def test_reference_prompt_preserves_cube_faces(self) -> None:
        job = build_job(
            notes="thin gold band",
            caption="black cloth with gold trim",
            has_reference=True,
        )
        self.assertTrue(job.has_reference)
        self.assertIn("reference image is attached", job.material.lower())
        self.assertIn("do not paste", job.material.lower())
        self.assertIn("preserve this cube's faces", job.material.lower())
        self.assertIn("face content", job.material.lower())
        self.assertNotIn("materials only", job.material.lower())
        self.assertNotIn("use it only for", job.material.lower())
        self.assertIn("preserve this cube's faces", job.mockup_prompt.lower())
        md = format_job_markdown(job)
        self.assertIn("image=[upload]", md)

    def test_markdown_includes_caption(self) -> None:
        job = build_job(caption="obsidian stone")
        md = format_job_markdown(job, print_assets=True)
        self.assertIn("obsidian stone", md)
        self.assertIn("Print-ready", md)

    def test_material_from_caption_default(self) -> None:
        self.assertIn("Black Cube of Saturn", material_from_caption(None, None))


if __name__ == "__main__":
    unittest.main()
