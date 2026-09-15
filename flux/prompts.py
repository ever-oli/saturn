"""Black Cube of Saturn streetwear prompts for FLUX.2-klein-9B.

Klein's Qwen3 encoder allows ``max_sequence_length=512``. Uploads are passed
to the pipeline as ``image=`` reference conditioning; prompts tell the model
to treat the ref as materials/colors, never as six stickers.
"""

from __future__ import annotations

from dataclasses import dataclass

_MAX_CHARS = 1600

LAYOUT_SIDE = "side-by-side"
LAYOUT_SEPARATE = "separate"

DEFAULT_MATERIAL = (
    "matte black cube with a thin gold equatorial band, obsidian / kiswah cloth, "
    "esoteric Black Cube of Saturn"
)

_QUALITY = (
    "professional streetwear product photograph, oversized drop-shoulder heavyweight "
    "white cotton tee, garment-dyed, catalog lighting on a light gray seamless, "
    "photoreal fabric, editorial lookbook quality, high-contrast minimalist esoteric "
    "streetwear, no model, no hanger hardware, no watermark, no UI chrome"
)

_ANTI_STICKER = (
    "designed as original merch artwork, not a photo collage, not six identical "
    "stickers, no checkerboard, no PNG transparency grid, no alpha holes"
)

_CUBE_FRONT = (
    "FRONT chest graphic: a single isometric 3D cube (three visible faces of one "
    "solid object), centered, modest scale, soft oval contact shadow under the cube"
)

_NET_BACK = (
    "BACK graphic: unfolded 6-panel Latin-cross cube net — column of four squares "
    "with left and right wings on the second square from the top; each panel is a "
    "different face of the same cube, paper-craft unfolding with crisp seams"
)

_REF_INSTRUCTION = (
    "A reference image is attached: use it only for the cube's materials, colors, "
    "and surface language. Redesign as original streetwear artwork — do not paste "
    "the photo onto the faces"
)


@dataclass(frozen=True)
class PromptJob:
    layout: str
    mockup_prompt: str
    front_prompt: str
    back_prompt: str
    print_front_prompt: str
    print_back_prompt: str
    material: str
    caption: str | None
    notes: str
    has_reference: bool


def clip_prompt(text: str, max_chars: int = _MAX_CHARS) -> str:
    cleaned = " ".join((text or "").split())
    if len(cleaned) <= max_chars:
        return cleaned
    return cleaned[: max_chars - 1].rsplit(" ", 1)[0] + "…"


def material_from_caption(
    caption: str | None,
    notes: str | None = None,
    has_reference: bool = False,
) -> str:
    """Turn a VLM caption + user notes into cube-surface language."""
    cap = " ".join((caption or "").split())
    extra = " ".join((notes or "").split())
    if has_reference:
        material = _REF_INSTRUCTION
        if cap:
            material = f"{material}. Reference reads as: {cap}"
        if extra:
            material = f"{material}. Design notes: {extra}"
        return material
    if cap:
        material = (
            f"cube surface inspired by this reference (materials and colors only, "
            f"do not paste the photo onto the faces): {cap}"
        )
        if extra:
            material = f"{material}. Design notes: {extra}"
        return material
    if extra:
        return f"{DEFAULT_MATERIAL}. Design notes: {extra}"
    return DEFAULT_MATERIAL


def build_job(
    notes: str | None = None,
    caption: str | None = None,
    layout: str = LAYOUT_SIDE,
    has_reference: bool = False,
) -> PromptJob:
    layout_key = (layout or LAYOUT_SIDE).strip().lower()
    if layout_key not in {LAYOUT_SIDE, LAYOUT_SEPARATE}:
        layout_key = LAYOUT_SIDE
    note_text = " ".join((notes or "").split())
    cap_text = " ".join((caption or "").split()) or None
    material = material_from_caption(
        cap_text, note_text, has_reference=has_reference
    )

    cube = f"{_CUBE_FRONT}, {material}, {_ANTI_STICKER}"
    net = f"{_NET_BACK}, same material as the cube, {_ANTI_STICKER}"

    pair = clip_prompt(
        f"{_QUALITY}. Two tees side by side: left is the front of the shirt, "
        f"right is the back of the matching shirt. {cube}. {net}."
    )
    front = clip_prompt(
        f"{_QUALITY}. Single tee, front view, white shirt filling the frame. {cube}."
    )
    back = clip_prompt(
        f"{_QUALITY}. Single tee, back view, white shirt filling the frame. {net}."
    )
    print_front = clip_prompt(
        f"print-ready merch graphic on a plain white background, no t-shirt, "
        f"no mockup, isolated isometric 3D cube, {material}, {_ANTI_STICKER}, "
        f"centered, high detail, catalog icon"
    )
    print_back = clip_prompt(
        f"print-ready merch graphic on a plain white background, no t-shirt, "
        f"no mockup, isolated Latin-cross cube net (four stacked squares, wings "
        f"on the second row), six unique faces of one cube, {material}, "
        f"{_ANTI_STICKER}, centered, high detail"
    )

    mockup = pair if layout_key == LAYOUT_SIDE else front
    return PromptJob(
        layout=layout_key,
        mockup_prompt=mockup,
        front_prompt=front,
        back_prompt=back,
        print_front_prompt=print_front,
        print_back_prompt=print_back,
        material=material,
        caption=cap_text,
        notes=note_text,
        has_reference=has_reference,
    )


def format_job_markdown(job: PromptJob, *, print_assets: bool = False) -> str:
    lines = [
        f"**Layout:** `{job.layout}`",
        f"**Material:** {job.material}",
    ]
    if job.has_reference:
        lines.append(
            "**Reference:** passed to Klein as `image=[upload]` (conditioning, not stickers)."
        )
    if job.caption:
        lines.append(f"**Caption (from upload):** {job.caption}")
    lines.append("\n**Mockup prompt**\n")
    lines.append(job.mockup_prompt)
    if job.layout == LAYOUT_SEPARATE:
        lines.append("\n**Front tee**\n")
        lines.append(job.front_prompt)
        lines.append("\n**Back tee**\n")
        lines.append(job.back_prompt)
    if print_assets:
        lines.append("\n**Print-ready front**\n")
        lines.append(job.print_front_prompt)
        lines.append("\n**Print-ready back net**\n")
        lines.append(job.print_back_prompt)
    return "\n".join(lines)
