#!/usr/bin/env python3
"""Design-pose geometry check (no physics): preview/design_*.png + preview/design_sheet.png.

  design_check.py           smooth normals (what the mod shows without a shader pack)
  design_check.py --flat    flat face normals (what Iris/Complementary shows in-game)
"""
import os
import sys

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import glados_model as gm  # noqa: E402
import render  # noqa: E402

flat = "--flat" in sys.argv
suffix = "_flat" if flat else ""
m = gm.model_json()
wp = gm.body_world_pos()
pose = {"bodies": {k: [*v, 1, 0, 0, 0] for k, v in wp.items()}, "eye": [1, 0.824, 0.29, 1]}
os.makedirs(os.path.join(HERE, "preview"), exist_ok=True)
head_c = wp["head"] + gm.head_center()
mid = (0.6, 0, -3.9)
views = {
    "front": ((14.0, -3.0, -4.6), mid, 40),
    "side": ((0.8, -15.0, -4.2), mid, 40),
    "three_quarter": ((9.5, -9.5, -6.8), (0.8, 0, -4.2), 44),
    "head": (tuple(head_c + np.array([3.4, -2.6, -0.4])), tuple(head_c + [-0.3, 0, 0.2]), 40),
    "player": ((7.0, -3.0, -11.4), (0.9, 0, -4.6), 62),     # what a player standing on the lab floor sees
}
paths = []
for name, (cam, tgt, fov) in views.items():
    img = render.render(m, pose, cam, tgt, size=(480, 640), fov=fov, extra=[render.ceiling_quad(half=8)], flat=flat)
    p = os.path.join(HERE, "preview", f"design_{name}{suffix}.png")
    img.save(p)
    paths.append(p)
    print(p)

# side-by-side sheet of our views
tiles = [Image.open(p).convert("RGB") for p in paths]
H = 520
tiles = [t.resize((int(t.width * H / t.height), H)) for t in tiles]
sheet = Image.new("RGB", (sum(t.width for t in tiles) + 6 * (len(tiles) - 1), H + 24), (20, 20, 22))
x = 0
d = ImageDraw.Draw(sheet)
labels = list(views)
for t, lab in zip(tiles, labels):
    sheet.paste(t, (x, 24))
    d.text((x + 4, 6), lab, fill=(230, 230, 230))
    x += t.width + 6
sp = os.path.join(HERE, "preview", f"design_sheet{suffix}.png")
sheet.save(sp)
print(sp)
