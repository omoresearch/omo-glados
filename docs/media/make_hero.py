"""Compose the README hero (light + dark) and the GitHub social preview from a real render of our own model.

The picture is src/design_three_quarter.png: the original procedural GLaDOS-style model (sidecar/glados_model.py)
in its design pose, drawn by the sidecar's own software renderer (sidecar/design_check.py). It is not in-game
footage and nothing in it is painted or generated: this script only frames it and sets the type.
Fonts: Crimson Text + IBM Plex Mono (SIL OFL). Run from the repo root:
    python3 docs/media/make_hero.py path/to/fonts [path/to/omo-mark.png]
"""
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
FONTS = Path(sys.argv[1]) if len(sys.argv) > 1 else HERE / "fonts"
MARK = Path(sys.argv[2]) if len(sys.argv) > 2 else None

CREAM = (245, 240, 232)
INK = (26, 16, 8)
BROWN = (74, 59, 47)
LINE = (212, 205, 192)
SAND = (196, 184, 168)
ORANGE = (226, 104, 36)
EYE = (255, 210, 74)

W, H = 2400, 960
PAD = 48
THEMES = {
    "light": dict(bg=CREAM, fg=INK, sub=BROWN, pill_bg=INK, pill_fg=CREAM, frame=LINE),
    "dark": dict(bg=(18, 12, 7), fg=CREAM, sub=SAND, pill_bg=CREAM, pill_fg=INK, frame=BROWN),
}
SRC = HERE / "src" / "design_three_quarter.png"


def font(name, size):
    return ImageFont.truetype(str(FONTS / name), size)


def plate(img, d, box, c, radius=28, caption="SIDECAR RENDER  ·  OUR OWN MODEL", cap_size=22):
    """Paste the render as a rounded photo plate filling `box` (cover-crop), with an honest caption pill."""
    x0, y0, x1, y1 = box
    pw, ph = x1 - x0, y1 - y0
    src = Image.open(SRC).convert("RGB")
    s = max(pw / src.width, ph / src.height)
    r = src.resize((round(src.width * s), round(src.height * s)), Image.LANCZOS)
    left = (r.width - pw) // 2
    top = (r.height - ph) // 2
    shot = r.crop((left, top, left + pw, top + ph))
    mask = Image.new("L", (pw, ph), 0)
    ImageDraw.Draw(mask).rounded_rectangle((0, 0, pw - 1, ph - 1), radius=radius, fill=255)
    img.paste(shot, (x0, y0), mask)
    d.rounded_rectangle((x0 - 1, y0 - 1, x1, y1), radius=radius + 1, outline=c["frame"], width=3)
    cf = font("IBMPlexMono-Medium.ttf", cap_size)
    cw = d.textlength(caption, font=cf)
    cx, cy = x0 + 24, y0 + 24
    d.rounded_rectangle((cx, cy, cx + cw + 2 * cap_size, cy + 2 * cap_size), radius=cap_size, fill=INK)
    d.text((cx + cap_size, cy + cap_size), caption, font=cf, fill=CREAM, anchor="lm")


def mark(img, theme, x, y, size):
    if not (MARK and MARK.exists()):
        return 0
    m = Image.open(MARK).convert("RGBA").resize((size, size), Image.LANCZOS)   # the app-icon tile, as-is
    img.paste(m, (x, y), m)
    return size + 20


def build(theme):
    c = THEMES[theme]
    img = Image.new("RGB", (W, H), c["bg"])
    d = ImageDraw.Draw(img)
    ph = H - 2 * PAD
    pw = round(ph * 0.75)
    plate(img, d, (W - PAD - pw, PAD, W - PAD, PAD + ph), c)
    lx = 140

    eb = font("IBMPlexMono-Medium.ttf", 32)
    label = "OMO RESEARCH  ·  GLADOS"
    tw = d.textlength(label, font=eb)
    d.rounded_rectangle((lx - 4, 150, lx + tw + 44, 208), radius=29, fill=c["pill_bg"])
    d.text((lx + 20, 159), label, font=eb, fill=c["pill_fg"])

    serif = font("CrimsonText-SemiBold.ttf", 146)
    d.text((lx, 248), "A ceiling AI that", font=serif, fill=c["fg"])
    d.text((lx, 410), "watches you back.", font=serif, fill=c["fg"])
    d.text((lx, 590), "An original model on real physics, hanging in your Minecraft lab.",
           font=font("CrimsonText-Italic.ttf", 52), fill=c["sub"])

    d.rectangle((lx, 694, lx + 120, 699), fill=ORANGE)
    d.ellipse((lx + 132, 688, lx + 150, 706), fill=EYE)
    d.text((lx, 726), "MuJoCo  ·  procedural model  ·  Fabric 26.2",
           font=font("IBMPlexMono-Regular.ttf", 33), fill=c["fg"])

    fy = 852
    tx = lx + mark(img, theme, lx, fy - 12, 60)
    d.text((tx, fy), "omo.sh", font=font("IBMPlexMono-Medium.ttf", 30), fill=c["sub"])
    d.text((W - PAD - pw - 60, fy + 4), "Fan project. Not affiliated with Valve.", anchor="ra",
           font=font("IBMPlexMono-Regular.ttf", 24), fill=c["sub"])

    d.rectangle((0, 0, W - 1, H - 1), outline=c["frame"], width=4)
    out = img.resize((1600, 640), Image.LANCZOS)
    name = "hero.png" if theme == "light" else "hero-dark.png"
    out.save(HERE / name, optimize=True)
    print("wrote", HERE / name, out.size)


def social():
    """1280x640 GitHub social preview (dark theme, readable as a small card)."""
    c = THEMES["dark"]
    S = 2
    img = Image.new("RGB", (1280 * S, 640 * S), c["bg"])
    d = ImageDraw.Draw(img)
    pad = 40 * S
    ph = 640 * S - 2 * pad
    pw = round(ph * 0.75)
    plate(img, d, (1280 * S - pad - pw, pad, 1280 * S - pad, pad + ph), c, radius=24 * S,
          caption="OUR OWN MODEL", cap_size=16 * S)
    lx = 70 * S
    eb = font("IBMPlexMono-Medium.ttf", 22 * S)
    label = "OMO RESEARCH"
    tw = d.textlength(label, font=eb)
    d.rounded_rectangle((lx - 4, 92 * S, lx + tw + 30 * S, 132 * S), radius=20 * S, fill=c["pill_bg"])
    d.text((lx + 14 * S, 112 * S), label, font=eb, fill=c["pill_fg"], anchor="lm")
    d.text((lx, 160 * S), "Omo GLaDOS", font=font("CrimsonText-SemiBold.ttf", 118 * S), fill=c["fg"])
    d.text((lx, 300 * S), "A GLaDOS-style ceiling AI,", font=font("CrimsonText-Italic.ttf", 46 * S), fill=c["sub"])
    d.text((lx, 352 * S), "alive in Minecraft.", font=font("CrimsonText-Italic.ttf", 46 * S), fill=c["sub"])
    d.rectangle((lx, 440 * S, lx + 80 * S, 443 * S), fill=ORANGE)
    d.ellipse((lx + 90 * S, 435 * S, lx + 103 * S, 448 * S), fill=EYE)
    d.text((lx, 470 * S), "MuJoCo  ·  Fabric 26.2  ·  MIT", font=font("IBMPlexMono-Regular.ttf", 24 * S), fill=c["fg"])
    tx = lx + mark(img, "dark", lx, 528 * S, 40 * S)
    d.text((tx, 536 * S), "omo.sh", font=font("IBMPlexMono-Medium.ttf", 22 * S), fill=c["sub"])
    out = img.resize((1280, 640), Image.LANCZOS)
    out.save(HERE / "social-preview.png", optimize=True)
    print("wrote", HERE / "social-preview.png", out.size)


if __name__ == "__main__":
    for t in THEMES:
        build(t)
    social()
