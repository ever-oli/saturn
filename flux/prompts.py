"""Black Cube of Saturn streetwear prompts for FLUX.2-klein.

Klein's Qwen3 encoder allows ``max_sequence_length=512``. Uploads are passed
to the pipeline as ``image=`` reference conditioning; prompts tell the model
to preserve this cube's face identity when unfolding — never a decorative
cross, hexagon, or sticker sheet.
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
    "premium contemporary streetwear product photo, oversized heavyweight white "
    "cotton tee, minimal white shirt, monochrome/limited-color graphic, realistic "
    "fabric and screen-print texture, studio lighting, highly accurate geometry, "
    "no model"
)

_ANTI_STICKER = (
    "no sticker sheet, no checkerboard, no alpha grid, not a photo collage; "
    "do not redesign the cube into a generic cross, wireframe hexagon, peeled "
    "3D glyph, or abstract sacred geometry; do not add text, logos, symbols, "
    "religious imagery, photographs, or unrelated graphics; the cross must be "
    "the unfolded net of the exact cube shown on the front"
)

_CUBE_FRONT = (
    "FRONT chest graphic: refined centered 3D isometric cube (three visible faces "
    "of one solid object); crisp geometric edges, subtle texture, consistent "
    "lighting, premium screen-print ready, soft oval contact shadow"
)

# Canonical net matching geometry/net.py: column of 4, wings on row 2.
# Five-square plus + sixth square extending down = Latin cross, not a plus logo.
_LATIN_CROSS_NET = (
    "6-face Latin-cross cube net: column of four squares with left/right wings on "
    "the second square from the top; center square plus four adjacent squares, "
    "sixth square extending down from that plus so it is a cube net, not a plus logo"
)

_NET_BACK = (
    "BACK graphic: the same cube physically unfolded into a "
    f"{_LATIN_CROSS_NET}. Think of the back graphic as the cube literally being "
    "cut along its edges and laid completely flat on the shirt. Each of the six "
    "squares contains the corresponding original cube-face artwork — same cube "
    "opened flat, not a separate cross. Continuity across adjacent panels. "
    "Large, centered between shoulders and lower back; clean thin separation "
    "lines between the six faces. Preserve this cube's faces when unfolding"
)

_REF_INSTRUCTION = (
    "A reference image is attached: it is this cube's identity (materials and "
    "face content). Preserve this cube's faces when unfolding. Use the upload "
    "for the front isometric cube and every back-net face — redesign for print "
    "quality, do not paste the photo as a collage, do not invent a different motif"
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


def _with_tail(core: str, tail: str, max_chars: int = _MAX_CHARS) -> str:
    """Keep geometry in ``core``; append material only if the clip budget allows."""
    base = clip_prompt(core, max_chars=max_chars)
    extra = " ".join((tail or "").split())
    if not extra or base.endswith("…"):
        return base
    room = max_chars - len(base) - 1
    if room < 24:
        return base
    if len(extra) <= room:
        return f"{base} {extra}"
    return clip_prompt(f"{base} {extra}", max_chars=max_chars)


def material_from_caption(
    caption: str | None,
    notes: str | None = None,
    has_reference: bool = False,
) -> str:
    """Turn a VLM caption + user notes into cube-surface and face-identity language."""
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
            f"cube surface and face artwork inspired by this reference "
            f"(preserve this cube's faces when unfolding; do not paste the photo "
            f"onto the faces): {cap}"
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

    # Geometry + unfolding language first; material fills leftover clip budget.
    pair = _with_tail(
        f"{_QUALITY}. Two tees side by side: left is the front of the shirt, "
        f"right is the back of the matching shirt. {_CUBE_FRONT}. {_NET_BACK}. "
        f"{_ANTI_STICKER}.",
        material,
    )
    front = clip_prompt(
        f"{_QUALITY}. Single tee, front view, white shirt filling the frame. "
        f"{_CUBE_FRONT}, {material}, {_ANTI_STICKER}."
    )
    back = _with_tail(
        f"{_QUALITY}. Single tee, back view, white shirt filling the frame. "
        f"{_NET_BACK}, same cube identity and face artwork as the front, "
        f"{_ANTI_STICKER}.",
        material,
    )
    print_front = clip_prompt(
        f"print-ready merch graphic on a plain white background, no t-shirt, "
        f"no mockup, isolated refined 3D isometric cube, crisp geometric edges, "
        f"subtle texture, consistent lighting, premium screen-print ready, "
        f"{material}, {_ANTI_STICKER}, centered, high detail, catalog icon"
    )
    print_back = clip_prompt(
        f"print-ready merch graphic on a plain white background, no t-shirt, "
        f"no mockup, isolated {_LATIN_CROSS_NET}. Think of the back graphic as "
        f"the cube literally being cut along its edges and laid completely flat. "
        f"Each of the six squares contains the corresponding artwork from the "
        f"original cube face; continuity across adjacent panels; clean thin "
        f"separation lines. {material}, {_ANTI_STICKER}, centered, high detail"
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
            "**Reference:** passed to Klein as `image=[upload]` (cube identity "
            "for front isometric and back net faces, not stickers)."
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
