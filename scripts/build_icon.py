"""Build Synapse desktop/tray icons — orange/purple gradient S."""

from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageDraw, ImageFilter, ImageFont

ROOT = Path(__file__).resolve().parents[1]
ASSETS = ROOT / "assets"
HERO_PATH = ASSETS / "synapse-hero.png"

BG = (4, 4, 10, 255)
ORANGE = (255, 118, 24)
MAGENTA = (228, 42, 168)


def _rounded_mask(size: int, radius_ratio: float = 0.2) -> Image.Image:
    mask = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(mask)
    radius = max(4, int(size * radius_ratio))
    draw.rounded_rectangle((0, 0, size - 1, size - 1), radius=radius, fill=255)
    return mask


def _load_font(size: int) -> ImageFont.FreeTypeFont | ImageFont.ImageFont:
    candidates = [
        "C:/Windows/Fonts/segoeuib.ttf",
        "C:/Windows/Fonts/arialbd.ttf",
        "C:/Windows/Fonts/calibrib.ttf",
    ]
    for path in candidates:
        if Path(path).exists():
            return ImageFont.truetype(path, size=size)
    return ImageFont.load_default()


def _gradient_letter_mask(size: int) -> Image.Image:
    letter = Image.new("L", (size, size), 0)
    draw = ImageDraw.Draw(letter)
    font = _load_font(int(size * 0.62))
    bbox = draw.textbbox((0, 0), "S", font=font)
    text_w = bbox[2] - bbox[0]
    text_h = bbox[3] - bbox[1]
    x = (size - text_w) // 2 - bbox[0]
    y = (size - text_h) // 2 - bbox[1] - int(size * 0.02)
    draw.text((x, y), "S", fill=255, font=font)
    return letter


def _horizontal_gradient(size: int) -> Image.Image:
    gradient = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    draw = ImageDraw.Draw(gradient)
    for x in range(size):
        t = x / max(1, size - 1)
        color = (
            int(ORANGE[0] * (1 - t) + MAGENTA[0] * t),
            int(ORANGE[1] * (1 - t) + MAGENTA[1] * t),
            int(ORANGE[2] * (1 - t) + MAGENTA[2] * t),
            255,
        )
        draw.line((x, 0, x, size), fill=color)
    return gradient


def render_icon(size: int) -> Image.Image:
    canvas = Image.new("RGBA", (size, size), (0, 0, 0, 0))
    base = Image.new("RGBA", (size, size), BG)
    base.putalpha(_rounded_mask(size))

    letter_mask = _gradient_letter_mask(size)
    gradient = _horizontal_gradient(size)
    letter = Image.composite(
        gradient,
        Image.new("RGBA", (size, size), (0, 0, 0, 0)),
        letter_mask,
    )

    canvas = Image.alpha_composite(canvas, base)

    if size >= 48:
        glow_mask = letter_mask.filter(ImageFilter.GaussianBlur(radius=max(1, size // 18)))
        glow = Image.composite(
            gradient,
            Image.new("RGBA", (size, size), (0, 0, 0, 0)),
            glow_mask,
        )
        glow = Image.blend(
            Image.new("RGBA", (size, size), (0, 0, 0, 0)),
            glow,
            alpha=0.4,
        )
        canvas = Image.alpha_composite(canvas, glow)

    canvas = Image.alpha_composite(canvas, letter)
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

    print(f"Wrote {png_path} ({images[0].size[0]}px)")
    print(f"Wrote {ico_path} ({ico_path.stat().st_size} bytes)")
    if HERO_PATH.exists():
        print(f"Hero asset: {HERO_PATH} ({HERO_PATH.stat().st_size} bytes)")


if __name__ == "__main__":
    build()
