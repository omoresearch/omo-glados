#!/usr/bin/env python3
"""preview/v3 pose gallery: software renders (render_v3) of an asset driven by the real sim through behaviours
(skinned wires follow the bodies). Writes preview/v3/poses_<model>.png (one row: rest, track, dance, angry, sleep).

    python3 tools/pose_gallery_v3.py [model]
"""
import io
import json
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
from PIL import Image, ImageDraw  # noqa: E402

import render_v3 as rv  # noqa: E402

model_name = sys.argv[1] if len(sys.argv) > 1 else "procedural"
os.environ["GLADOS_STATE"] = os.path.join(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")), "state.json")
os.environ["GLADOS_MODEL"] = model_name
import server  # noqa: E402

server.log = lambda *a: None
svc = server.Service()
model = json.loads(svc.model_bytes)
atlas = Image.open(io.BytesIO(svc.atlas_bytes))
P = [6.0, -4.0, -11.4]
CAM = dict(pos=[10.5, -9.5, -9.0], target=[0.6, -1.0, -4.6], fov=50.0)


def run(secs, cmd=None, player=True):
    with svc.lock:
        if cmd:
            svc.sim.command(cmd)
        for k in range(int(secs * 60)):
            if player and k % 12 == 0:
                svc.sim.set_player(*P)
            svc.sim.step()
        svc._snapshot()
    return json.loads(svc.pose_bytes)


shots = [("rest", run(0, "reset", player=False)), ("track player", run(6.0)), ("dance t=3s", (run(3.0, "dance"))),
         ("angry t=1.2s", (run(5.5) and run(1.2, "angry"))), ("sleep", (run(4) and run(5.0, "sleep")))]
W, H = 420, 560
sheet = Image.new("RGB", (W * len(shots), H), (0, 0, 0))
for i, (name, pose) in enumerate(shots):
    im = rv.render(model, pose, atlas, CAM["pos"], CAM["target"], size=(W, H), fov=CAM["fov"])
    d = ImageDraw.Draw(im)
    d.rectangle([0, 0, W, 20], fill=(0, 0, 0))
    d.text((6, 4), f"{model_name}: {name} (mode {pose['mode']})", fill=(255, 255, 255))
    sheet.paste(im, (i * W, 0))
out = os.path.join(HERE, "preview", "v3", f"poses_{model_name}.png")
sheet.save(out)
print(out)
