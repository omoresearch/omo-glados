"""Compose the README render sheets from the sidecar's own preview renders of our procedural model.

src/*.png are unedited outputs of sidecar/render_preview.py (track_low, angry, sleep_side: the real MuJoCo sim
driven through each behaviour, in-process) and sidecar/design_check.py (design_*: the design pose; *_flat = face
normals, the look Iris shader packs give in-game). This script only lays them out and adds captions.
    python3 docs/media/make_sheets.py path/to/fonts
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fonts"
INK = (26, 16, 8)
CREAM = (245, 240, 232)
SAND = (196, 184, 168)
ORANGE = (226, 104, 36)
TW, TH, LH, GAP = 480, 640, 108, 12

SHEETS = {
    "render-modes.png": [
        ("track_low.png", "!glados track", "her eye follows you"),
        ("angry.png", "!glados angry", "red eye, sharp lunges"),
        ("sleep_side.png", "!glados sleep", "droops, eye dims"),
        ("design_head.png", "the eye", "one glowing lens"),
    ],
    "render-design.png": [
        ("design_front.png", "front", "15,800 triangles"),
        ("design_side.png", "side", "hangs 7.8 m from the ceiling"),
        ("design_player_flat.png", "from the floor", "flat shading, as with Iris"),
    ],
}


def font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def sheet(out, tiles):
    img = Image.new("RGB", (TW * len(tiles) + GAP * (len(tiles) - 1), TH + LH), INK)
    d = ImageDraw.Draw(img)
    mono, ital = font("IBMPlexMono-Medium.ttf", 26), font("CrimsonText-Italic.ttf", 31)
    for i, (src, title, sub) in enumerate(tiles):
        x = i * (TW + GAP)
        im = Image.open(HERE / "src" / src).convert("RGB")
        s = max(TW / im.width, TH / im.height)
        im = im.resize((round(im.width * s), round(im.height * s)), Image.LANCZOS)
        left, top = (im.width - TW) // 2, (im.height - TH) // 2
        img.paste(im.crop((left, top, left + TW, top + TH)), (x, 0))
        d.rectangle((x + 24, TH + 12, x + 64, TH + 14), fill=ORANGE)
        d.text((x + 24, TH + 22), title, font=mono, fill=CREAM)
        d.text((x + 24, TH + 54), sub, font=ital, fill=SAND)
    img.save(HERE / out, optimize=True)
    print("wrote", HERE / out, img.size)


if __name__ == "__main__":
    for name, tiles in SHEETS.items():
        sheet(name, tiles)
