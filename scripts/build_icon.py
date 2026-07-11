"""Build crisp Synapse desktop/tray icons at all Windows sizes."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
BG = (12, 16, 24, 255)
TEAL = (0, 200, 140, 255)
TEAL_SOFT = (0, 200, 140, 90)


def _rounded_mask(size: int, radius_ratio: float = 0.22) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    radius = max(4, int(size * radius_ratio))
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def render_icon(size: int) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    base = Image.new("RGBA", (size, size), BG)
    base.putalpha(_rounded_mask(size))

    glow = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    glow_draw = ImageDraw.Draw(glow)
    cx = cy = size // 2
    glow_r = int(size * 0.36)
    glow_draw.ellipse(
        (cx - glow_r, cy - glow_r, cx + glow_r, cy + glow_r),
        fill=TEAL_SOFT,
    )
    if size >= 48:
        glow = glow.filter(ImageFilter.GaussianBlur(radius=max(1, size // 32)))

    ring = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(ring)
    outer = int(size * 0.30)
    inner = int(size * 0.17)
    if size <= 32:
        outer = int(size * 0.38)
        inner = int(size * 0.16)
    if size <= 20:
        outer = int(size * 0.42)
        inner = int(size * 0.14)
    draw.ellipse(
        (cx - outer, cy - outer, cx + outer, cy + outer),
        fill=TEAL,
    )
    draw.ellipse(
        (cx - inner, cy - inner, cx + inner, cy + inner),
        fill=BG[:3] + (255,),
    )

    canvas = Image.alpha_composite(canvas, base)
    canvas = Image.alpha_composite(canvas, glow)
    canvas = Image.alpha_composite(canvas, ring)
    return canvas


def build() -> None:
    ASSETS.mkdir(parents=True, exist_ok=True)
    sizes = [256, 128, 64, 48, 32, 24, 16]
    images = [render_icon(size) for size in sizes]

    png_path = ASSETS / "synapse_icon.png"
    ico_path = ASSETS / "synapse.ico"
    images[0].save(png_path, format="PNG")
    images[0].save(
        ico_path,
        format="ICO",
        sizes=[(image.width, image.height) for image in images],
        append_images=images[1:],
    )

    verify = Image.open(ico_path)
    frame_sizes = []
    for index in range(getattr(verify, "n_frames", 1)):
        verify.seek(index)
        frame_sizes.append(verify.size)
    print(f"Wrote {png_path} ({images[0].size[0]}px)")
    print(f"Wrote {ico_path} ({ico_path.stat().st_size} bytes, frames: {frame_sizes})")


if __name__ == "__main__":
    build()
