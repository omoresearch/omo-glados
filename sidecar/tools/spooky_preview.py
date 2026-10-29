#!/usr/bin/env python3
"""preview/spooky: the spooky dance rendered with render_v3 (textured v3 renderer) from the real sim.

    python3 tools/spooky_preview.py [model ...]       (default: procedural)

Per model: preview/spooky/<model>_<NN>_<label>.png (third-person frames + the player's POV at the climax) and
preview/spooky/contact_<model>.png (contact sheet). The player stands at P (eye height), drawn as a stick figure.
"""
import io
import json
import math
import os
import sys
import tempfile

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402
from PIL import Image, ImageDraw  # noqa: E402

import render_v3 as rv  # noqa: E402

P = [5.0, 1.0, -9.4]          # player eye (sim frame): ~5 m out from under the mount, standing 2 m above the floor
SHOTS = [(2.2, "1 power-down sway"), (4.4, "1 slow head tilt"), (7.0, "2 glitch dance"), (9.7, "2 glitch snap"),
         (13.0, "3 swoop"), (14.9, "3 climax stare"), (16.9, "3 pull back"), (20.5, "back to track")]
POV_AT = (13.6, 14.9, 15.6)
OUT = os.path.join(HERE, "preview", "spooky")


def project(cam, target, fov, W, H, pts):
    R = rv.look_at(np.asarray(cam, float), target)
    f = 0.5 * H / math.tan(math.radians(fov) / 2)
    out = []
    for p in pts:
        c = (np.asarray(p, float) - cam) @ R.T
        z = -c[2]
        out.append(None if z < 0.05 else (W / 2 + f * c[0] / z, H / 2 - f * c[1] / z))
    return out


def stick(img, cam, target, fov):
    W, H = img.size
    feet = np.array(P) - [0, 0, 1.62]
    pts = project(cam, target, fov, W, H, [P, feet, np.array(P) - [0, 0, 0.35], feet + [0, 0, 0.9]])
    if any(p is None for p in pts):
        return
    d = ImageDraw.Draw(img)
    (hx, hy), (fx, fy), (nx, ny), (wx, wy) = pts
    r = max(3, abs(fy - hy) * 0.12)
    d.line([nx, ny, wx, wy], fill=(90, 200, 255), width=3)
    d.line([wx, wy, fx - r, fy], fill=(90, 200, 255), width=3)
    d.line([wx, wy, fx + r, fy], fill=(90, 200, 255), width=3)
    d.ellipse([hx - r, hy - r, hx + r, hy + r], outline=(90, 200, 255), width=3)


def main():
    import server  # noqa: E402
    server.log = lambda *a: None
    names = sys.argv[1:] or ["procedural"]
    os.makedirs(OUT, exist_ok=True)
    sheets = []
    for name in names:
        os.environ["GLADOS_STATE"] = os.path.join(tempfile.mkdtemp(dir=os.environ.get("TMPDIR")), "state.json")
        os.environ["GLADOS_MODEL"] = name
        svc = server.Service()
        model = json.loads(svc.model_bytes)
        atlas = Image.open(io.BytesIO(svc.atlas_bytes))
        sim = svc.sim
        for k in range(5 * 60):
            if k % 12 == 0:
                sim.set_player(*P)
            sim.step()
        sim.command("spooky")
        t0 = sim.t
        want = sorted([(t, "3p", lab) for t, lab in SHOTS] + [(t, "pov", f"POV t={t:.1f}s") for t in POV_AT] +
                      [(15.2, "ots", "over the shoulder t=15.2s")])
        frames = []
        k = 0
        dmin = 1e9
        for tau_w, kind, label in want:
            while sim.t - t0 < tau_w:
                if k % 12 == 0:
                    sim.set_player(*P)
                sim.step()
                k += 1
                if gs_window(sim.t - t0):
                    dmin = min(dmin, float(np.linalg.norm(sim.eye_pos() - P)))
            svc._snapshot()
            pose = json.loads(svc.pose_bytes)
            eye = sim.eye_pos()
            dist = float(np.linalg.norm(eye - P))
            if kind == "3p":
                a = math.atan2(P[1], P[0]) - math.pi / 2
                mid = np.array([P[0] * 0.5, P[1] * 0.5, -6.6])
                cam = mid + 13.5 * np.array([math.cos(a), math.sin(a), 0.0]) + [0, 0, 0.6]
                fov, size = 58.0, (420, 520)
                img = rv.render(model, pose, atlas, cam, mid, size=size, fov=fov)
                stick(img, cam, mid, fov)
            elif kind == "ots":
                back = np.array([P[0], P[1], 0.0]) / math.hypot(P[0], P[1])
                side = np.array([-back[1], back[0], 0.0])
                cam = np.array(P) + 3.2 * back + 1.0 * side + [0, 0, 0.5]
                fov, size = 60.0, (640, 400)
                tgt = 0.5 * (eye + np.array(P))
                img = rv.render(model, pose, atlas, cam, tgt, size=size, fov=fov)
                stick(img, cam, tgt, fov)
            else:
                cam, fov, size = np.array(P), 70.0, (640, 400)
                img = rv.render(model, pose, atlas, cam, eye, size=size, fov=fov)
            d = ImageDraw.Draw(img)
            d.rectangle([0, 0, img.size[0], 30], fill=(0, 0, 0))
            d.text((6, 3), f"{name}: {label}", fill=(255, 255, 255))
            d.text((6, 16), f"mode {pose['mode']}  eye {np.round(pose['eye'], 2).tolist()}  eye->player {dist:.2f} m",
                   fill=(255, 220, 120))
            fn = os.path.join(OUT, f"{name}_{len(frames):02d}_{kind}_t{tau_w:04.1f}.png")
            img.save(fn)
            frames.append((kind, img))
            print(fn, f"dist {dist:.2f} m", flush=True)
        # contact sheet: third-person frames in a row of 8, the POV shots under them
        tp = [im for kd, im in frames if kd == "3p"]
        pv = [im for kd, im in frames if kd in ("pov", "ots")]
        W = sum(im.size[0] for im in tp[:4])
        H = tp[0].size[1] * 2 + pv[0].size[1]
        sheet = Image.new("RGB", (W, H), (0, 0, 0))
        for i, im in enumerate(tp):
            sheet.paste(im, ((i % 4) * im.size[0], (i // 4) * im.size[1]))
        x = 0
        for im in pv:
            sheet.paste(im.resize((W // len(pv), int(im.size[1] * (W / len(pv)) / im.size[0]))),
                        (x, tp[0].size[1] * 2))
            x += W // len(pv)
        fn = os.path.join(OUT, f"contact_{name}.png")
        sheet.save(fn)
        sheets.append(fn)
        print(fn, f"min eye->player during the climax {dmin:.2f} m", flush=True)
    print("\n".join(sheets))


def gs_window(tau):
    import glados_sim as gs
    return gs.SPOOKY_T["swoop"] <= tau <= gs.SPOOKY_T["back"] + 0.5


if __name__ == "__main__":
    main()
