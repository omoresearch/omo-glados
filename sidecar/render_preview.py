#!/usr/bin/env python3
"""Preview PNGs -> preview/*.png, rendered from the contract payloads (/model.json + /pose) with Lambert lighting.

  render_preview.py                 # in-process sim (deterministic), same payload code as the server
  render_preview.py --url http://127.0.0.1:8794   # from the live service (drives it with /cmd + /player)
"""
import argparse
import json
import os
import sys
import time
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import render  # noqa: E402

OUT = os.path.join(HERE, "preview")
LOW_TARGET = [4.0, -2.5, -11.4]   # player eye standing on the lab floor (13 m below the ceiling)
FLOOR_Z = -13.0


class Local:
    def __init__(self):
        import glados_model as gm
        import glados_sim as gs
        self.sim = gs.GladosSim(seed=3)
        self.model = gm.model_json()

    def run(self, secs):
        for _ in range(int(secs * 60)):
            self.sim.step()

    def cmd(self, c, **kw):
        print("cmd", c, self.sim.command(c, **kw))

    def player(self, p):
        self.sim.set_player(*p)

    def pose(self):
        return self.sim.pose_dict()


class Remote:
    def __init__(self, url):
        self.url = url.rstrip("/")
        self.model = self._get("/model.json")

    def _get(self, p):
        with urllib.request.urlopen(self.url + p, timeout=10) as r:
            return json.loads(r.read())

    def _post(self, p, o):
        req = urllib.request.Request(self.url + p, data=json.dumps(o).encode(), method="POST",
                                     headers={"Content-Type": "application/json"})
        with urllib.request.urlopen(req, timeout=10) as r:
            return json.loads(r.read())

    def run(self, secs):
        end = time.time() + secs
        while time.time() < end:
            if getattr(self, "_p", None):
                self._post("/player", dict(zip("xyz", self._p)))
            time.sleep(0.2)

    def cmd(self, c, **kw):
        print("cmd", c, self._post("/cmd", {"cmd": c, "from": "mc", **kw}))

    def player(self, p):
        self._p = p
        self._post("/player", dict(zip("xyz", p)))

    def pose(self):
        return self._get("/pose")


def eye_world(pose):
    import glados_model as gm
    h = pose["bodies"]["head"]
    ep, _ = gm.eye_site()
    return np.array(h[:3]) + render.quat_mat(h[3:7]) @ ep


def shot(src, name, cam, target=(0.4, 0, -3.9), marker=None, fov=40, floor=False):
    pose = src.pose()
    extra = [render.ceiling_quad()]
    if floor:
        extra.append(render.floor_quad(FLOOR_Z))
    line = (eye_world(pose), marker) if marker is not None else None
    img = render.render(src.model, pose, cam, target, size=(600, 800), fov=fov, extra=extra, marker=marker,
                        line=line)
    p = os.path.join(OUT, name)
    img.save(p)
    print("wrote", p, "mode", pose["mode"], "eye", pose["eye"])


def main():
    ap = argparse.ArgumentParser()
    global OUT
    ap.add_argument("--url")
    ap.add_argument("--out", default=OUT)
    a = ap.parse_args()
    OUT = a.out
    os.makedirs(OUT, exist_ok=True)
    src = Remote(a.url) if a.url else Local()
    if a.url:
        src.cmd("reset")
    src.player([60.0, 0.0, -11.4])  # far away -> idle
    src.cmd("idle")
    src.run(4.0)
    if isinstance(src, Local):  # face front for the reference shots
        src.sim.glance = src.sim.eye_pos() + np.array([8.0, 0, -2.5])
        src.run(2.0)
    shot(src, "idle_front.png", (13.5, -4.0, -3.2))
    shot(src, "idle_side.png", (0.6, -14.5, -3.8))
    src.player(LOW_TARGET)
    src.cmd("track")
    src.run(4.0)
    shot(src, "track_low.png", (11.5, 10.5, -6.5), target=(1.8, -0.8, -6.3), marker=LOW_TARGET, fov=58, floor=True)
    src.player([5.0, 3.0, -11.4])
    src.run(3.0)
    src.cmd("angry")
    src.run(1.0)
    shot(src, "angry.png", (10.0, 6.5, -6.5), target=(0.8, 0.5, -5.0), marker=[5.0, 3.0, -11.4], fov=50)
    src.run(4.0)
    src.cmd("sleep")
    src.run(4.0)
    shot(src, "sleep_side.png", (0.6, -14.5, -3.8))


if __name__ == "__main__":
    main()
