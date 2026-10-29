#!/usr/bin/env python3
"""Tiny stand-in for the glados-sim sidecar (sidecar/CONTRACT.md API incl. v3), used only by the mod's harness.

A fake hanging chain of boxes: mount (root, first) -> arm1 -> arm2 -> torso -> neck -> head (+ an "eye" mesh), ~7.5 m
from the ceiling (sim origin) down, swaying slowly in -z. /pose carries talk + eye (+ "rev" in v3 mode); POST /cmd
implements the contract command set (twitch may not reset / switch models: 403; look needs a valid dir; talk sets the
level); POST /player and /config are recorded and readable at GET /_last.

v3 (default): two switchable models,
  * "boxes"      version 3: top-level meshes with "body", per-vertex uv into a 32x32 /atlas.png (4 colour patches),
                 tags eye / emissive / glass, one linear-blend skinned cable (arm2 -> torso, verts in the world rest
                 frame), bodies with parent / joint_pos / rest, eye + voice blocks, credit;
  * "procedural" version 1 style nested meshes (no uv, no atlas) but with rev/name/credit like the real v3 sidecar.
  POST /cmd {"cmd":"model","name":...} switches and bumps "rev" (in /model.json, /pose and the atlas URL);
  GET /models lists them. Each rev has a different atlas (patch 0 colour changes) so a refetch is observable.
--v1: the pre-v3 sidecar exactly (no rev, no /models, no /atlas.png, no model/dance cmds) for backwards-compat tests.
Usage: python3 fake_sidecar.py [port] [--v1]   (default 8789, so it never collides with the real sidecar on 8794)
"""
import json
import math
import struct
import sys
import time
import zlib
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

ARGS = [a for a in sys.argv[1:] if not a.startswith("--")]
V1 = "--v1" in sys.argv
PORT = int(ARGS[0]) if ARGS else 8789
T0 = time.time()
LAST = {"cmd": None, "player": None, "player_count": 0, "config": None, "model_gets": 0, "atlas_gets": 0}
STATE = {"mode": "idle", "talk": 0.0, "talk_until": 0.0, "angry_until": 0.0, "sleep": False, "dance_until": 0.0,
         "model": "boxes", "rev": 1}
CMDS = {"idle", "track", "angry", "scan", "nod", "shake", "bow", "sleep", "wake", "reset", "look", "talk"}
if not V1:
    CMDS |= {"dance", "model"}
DIRS = {"left", "right", "up", "down", "player"}


def box(cx, cy, cz, hx, hy, hz):
    v = []
    for z in (-hz, hz):
        for x, y in ((-hx, -hy), (hx, -hy), (hx, hy), (-hx, hy)):
            v += [cx + x, cy + y, cz + z]
    t = [0, 2, 1, 0, 3, 2, 4, 5, 6, 4, 6, 7, 0, 1, 5, 0, 5, 4, 1, 2, 6, 1, 6, 5, 2, 3, 7, 2, 7, 6, 3, 0, 4, 3, 4, 7]
    return v, t


# atlas patches (16x16 px each in a 32x32 atlas): uv rectangle (u0, v0, u1, v1), v=0 = top row
PATCH = {"white": (0.0, 0.0, 0.5, 0.5), "checker": (0.5, 0.0, 1.0, 0.5), "eye": (0.0, 0.5, 0.5, 1.0),
         "glass": (0.5, 0.5, 1.0, 1.0)}


def mesh(rgba, *boxes, tag=None, normals=False, patch=None):
    verts, tris, uv = [], [], []
    for b in boxes:
        v, t = box(*b)
        off = len(verts) // 3
        verts += v
        tris += [i + off for i in t]
        if patch:
            u0, v0, u1, v1 = PATCH[patch]
            for k in range(8):                       # corners of the patch, inset half a texel
                uu = (u0 if k % 4 in (0, 3) else u1)
                vv = (v0 if k < 4 else v1)
                uv += [round(uu + (0.01 if uu == u0 else -0.01), 4), round(vv + (0.01 if vv == v0 else -0.01), 4)]
    m = {"rgba": rgba, "verts": [round(x, 4) for x in verts], "tris": tris}
    if normals:  # radial per-vertex normals from each box centre (exercises the optional key)
        n = []
        for b in boxes:
            v, _ = box(*b)
            for i in range(0, len(v), 3):
                d = [v[i] - b[0], v[i + 1] - b[1], v[i + 2] - b[2]]
                ln = math.sqrt(sum(x * x for x in d)) or 1
                n += [round(x / ln, 4) for x in d]
        m["normals"] = n
    if patch:
        m["uv"] = uv
    if tag:
        m["tag"] = tag
    return m


SHELL, MECH, DARK, RING = [0.91, 0.91, 0.9, 1], [0.23, 0.24, 0.25, 1], [0.16, 0.17, 0.18, 1], [0.08, 0.08, 0.08, 1]
EYE = [1.0, 0.82, 0.29, 1]
NAMES = ["mount", "arm1", "arm2", "torso", "neck", "head"]
OFFS = [0.0, 0.2, 1.5, 1.5, 2.5, 0.7]          # joint offset below the parent origin (links hang in -z)
# body-local meshes (v1 nested and v3 top-level share them)
LOCAL = [
    [mesh(DARK, (0, 0, -0.1, 0.8, 0.8, 0.1)), mesh(MECH, (0.5, 0.3, -0.5, 0.05, 0.05, 0.4))],
    [mesh(MECH, (0, 0, -0.75, 0.18, 0.18, 0.75)), mesh(RING, (0, 0, -1.5, 0.22, 0.22, 0.06))],
    [mesh(MECH, (0, 0, -0.75, 0.16, 0.16, 0.75)), mesh(SHELL, (0.12, 0, -0.75, 0.06, 0.14, 0.5))],
    [mesh(SHELL, (0, 0, -1.25, 0.6, 0.55, 1.25), normals=True), mesh(DARK, (-0.62, 0, -1.25, 0.05, 0.2, 1.0))],
    [mesh(MECH, (0, 0, -0.35, 0.14, 0.14, 0.35))],
    [mesh(SHELL, (0.45, 0, -0.35, 0.7, 0.3, 0.3)), mesh(DARK, (1.16, 0, -0.35, 0.02, 0.25, 0.25)),
     mesh(EYE, (1.19, 0, -0.35, 0.02, 0.1, 0.1), tag="eye")],
]
MODEL_V1 = {"version": 1, "bodies": [{"name": n, "meshes": m} for n, m in zip(NAMES, LOCAL)]}

REST_Z = []
z = 0.0
for o in OFFS:
    z -= o
    REST_Z.append(z)


def cable():
    """Square tube from arm2 (world z -2.0) to torso (z -3.6) at x 0.4, 6 rings: skinned arm2 (w 1) -> torso (w 0)."""
    verts, normals, uv, j, w, tris = [], [], [], [], [], []
    rings, r = 6, 0.06
    corners = [(1, 1), (-1, 1), (-1, -1), (1, -1)]
    for k in range(rings):
        f = k / (rings - 1)
        zz = -2.0 - 1.6 * f
        for cx, cy in corners:
            verts += [0.4 + r * cx, r * cy, zz]
            ln = math.sqrt(2)
            normals += [round(cx / ln, 4), round(cy / ln, 4), 0.0]
            uv += [0.55 + 0.4 * (cx > 0), 0.05 + 0.4 * f]
            j += [0, 1]
            w.append(round(1 - f, 4))
    for k in range(rings - 1):
        for c in range(4):
            a, b = k * 4 + c, k * 4 + (c + 1) % 4
            tris += [a, b, b + 4, a, b + 4, a + 4]
    return {"body": "arm2", "name": "cable", "rgba": [1, 1, 1, 1], "verts": [round(x, 4) for x in verts],
            "normals": normals, "uv": [round(x, 4) for x in uv], "tris": tris, "tag": None,
            "skin": {"bodies": ["arm2", "torso"], "j": j, "w": w}}


def model_boxes():
    meshes = []
    for name, ms in zip(NAMES, LOCAL):
        for i, m in enumerate(ms):
            m3 = dict(m)
            m3["body"] = name
            m3["name"] = f"{name}_{i}"
            if m.get("tag") == "eye":
                m3["uv"] = mesh(EYE, (1.19, 0, -0.35, 0.02, 0.1, 0.1), patch="eye")["uv"]
                m3["rgba"] = [1, 1, 1, 1]
            elif name == "torso" and i == 0:
                m3["uv"] = mesh(SHELL, (0, 0, -1.25, 0.6, 0.55, 1.25), patch="checker")["uv"]
                m3["rgba"] = [1, 1, 1, 1]
            elif name in ("mount", "arm1") and i == 0:
                m3["uv"] = mesh(DARK, (0, 0, 0, 1, 1, 1), patch="white")["uv"]
            meshes.append(m3)
    meshes.append(dict(mesh([1, 1, 1, 0.35], (0.45, 0, -0.35, 0.72, 0.32, 0.05), patch="glass", tag="glass"),
                       body="head", name="head_visor"))
    meshes.append(dict(mesh([0.3, 0.9, 1.0, 1], (-0.3, 0, -0.1, 0.05, 0.05, 0.05), tag="emissive"),
                       body="head", name="head_led"))
    meshes.append(cable())
    bodies = []
    for i, n in enumerate(NAMES):
        bodies.append({"name": n, "parent": NAMES[i - 1] if i else None, "joint_pos": [0, 0, -OFFS[i]],
                       "rest": [0, 0, REST_Z[i], 1, 0, 0, 0]})
    tri = sum(len(m["tris"]) // 3 for m in meshes)
    return {"version": 3, "name": "boxes", "title": "Fake box GLaDOS", "credit": "test fixture, CC0",
            "bodies": bodies, "meshes": meshes, "eye": {"body": "head", "pos": [1.21, 0, -0.35], "axis": [1, 0, 0]},
            "voice": {"body": "head", "pos": [0.5, 0, -0.35]}, "atlas": "atlas.png", "triangles": tri}


def model_procedural():
    m = dict(MODEL_V1)
    m.update({"name": "procedural", "title": "Procedural GLaDOS-style (fake)", "credit": "original, procedural",
              "triangles": sum(len(x["tris"]) // 3 for ms in LOCAL for x in ms)})
    return m


MODELS = {"boxes": model_boxes(), "procedural": model_procedural()}


def png(w, h, px):
    """px(x, y) -> (r, g, b, a); stdlib-only PNG encoder (RGBA8, filter 0)."""
    raw = b"".join(b"\0" + b"".join(bytes(px(x, y)) for x in range(w)) for y in range(h))

    def chunk(t, d):
        return struct.pack(">I", len(d)) + t + d + struct.pack(">I", zlib.crc32(t + d) & 0xFFFFFFFF)
    return (b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", struct.pack(">IIBBBBB", w, h, 8, 6, 0, 0, 0))
            + chunk(b"IDAT", zlib.compress(raw)) + chunk(b"IEND", b""))


def atlas(rev):
    tint = (255, 255 - (rev * 16) % 128, 255)              # differs per rev: a refetch is observable

    def px(x, y):
        if x < 16 and y < 16:
            return tint + (255,)
        if y < 16:
            return (220, 40, 40, 255) if (x // 4 + y // 4) % 2 else (40, 60, 220, 255)
        if x < 16:
            return (255, 210, 74, 255)
        return (180, 230, 255, 120)
    return png(32, 32, px)


def active():
    m = dict(MODELS[STATE["model"]])
    m["rev"] = STATE["rev"]
    if m.get("atlas"):
        m["atlas"] = "/atlas.png?rev=%d" % STATE["rev"]
    return m


def qmul(a, b):
    w1, x1, y1, z1 = a
    w2, x2, y2, z2 = b
    return [w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2, w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2, w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2]


def rot(q, v):
    w, x, y, z = q
    return qmul(qmul(q, [0] + list(v)), [w, -x, -y, -z])[1:]


def axq(axis, ang):
    s = math.sin(ang / 2)
    return [math.cos(ang / 2), axis[0] * s, axis[1] * s, axis[2] * s]


def pose():
    now = time.time()
    t = now - T0
    talk = STATE["talk"] if now < STATE["talk_until"] else 0.0
    angry = now < STATE["angry_until"]
    dance = now < STATE["dance_until"]
    out = {}
    p = [0.0, 0.0, 0.0]
    q = axq((0, 0, 1), 0.2 * math.sin(0.3 * t) + (0.3 * math.sin(4 * t) if dance else 0))   # slow yaw at the mount
    for i, name in enumerate(NAMES):
        p = [a + b for a, b in zip(p, rot(q, (0, 0, -OFFS[i])))]
        if 0 < i < 5:                                       # gentle pendulum sway on the arm joints
            q = qmul(q, axq((0, 1, 0), 0.05 * math.sin(0.7 * t + i) + (0.15 * math.sin(9 * t) if angry else 0)))
        if name == "head":
            q = qmul(q, axq((0, 1, 0), -0.25 + 0.1 * talk))
        out[name] = [round(x, 5) for x in p + q]
    eye = [1.0, 0.82, 0.29, 1.0]
    if angry:
        eye = [1.0, 0.35, 0.1, 1.0]
    elif STATE["sleep"]:
        eye = [1.0, 0.82, 0.29, 0.15]
    mode = "angry" if angry else "dance" if dance else STATE["mode"]
    r = {"seq": int(t * 60), "t": round(t, 3), "mode": mode, "talk": round(talk, 3), "eye": eye, "bodies": out}
    if not V1:
        r["rev"] = STATE["rev"]
    return r


class H(BaseHTTPRequestHandler):
    def log_message(self, *a):
        pass

    def _raw(self, b, ctype, code=200):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(b)))
        self.end_headers()
        self.wfile.write(b)

    def _send(self, obj, code=200):
        self._raw(json.dumps(obj).encode(), "application/json", code)

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            self._send({"ok": True, "mode": STATE["mode"], "t": time.time() - T0, "hz": 60, "fake": True,
                        "fake_v1": V1})
        elif path == "/model.json":
            LAST["model_gets"] += 1
            self._send(MODEL_V1 if V1 else active())
        elif path == "/atlas.png" and not V1:
            LAST["atlas_gets"] += 1
            if not MODELS[STATE["model"]].get("atlas"):
                return self._send({"ok": False, "error": "model has no atlas"}, 404)
            self._raw(atlas(STATE["rev"]), "image/png")
        elif path == "/models" and not V1:
            self._send({"active": STATE["model"], "models": [
                {"name": k, "title": m["title"], "credit": m["credit"], "triangles": m["triangles"]}
                for k, m in MODELS.items()]})
        elif path == "/pose":
            self._send(pose())
        elif path == "/_last":
            self._send(LAST)
        else:
            self._send({"ok": False, "error": "not found"}, 404)

    def do_POST(self):
        n = int(self.headers.get("Content-Length", "0"))
        try:
            body = json.loads(self.rfile.read(n) or b"{}")
        except ValueError:
            return self._send({"ok": False, "error": "bad json"}, 400)
        if self.path == "/cmd":
            LAST["cmd"] = body
            c = body.get("cmd")
            if c not in CMDS:
                return self._send({"ok": False, "error": "unknown cmd"})
            twitch = str(body.get("from", "")).startswith("twitch:")
            if c in ("reset", "model") and twitch:
                return self._send({"ok": False, "error": "forbidden"}, 403)
            if c == "look" and body.get("dir") not in DIRS:
                return self._send({"ok": False, "error": "bad dir"})
            now = time.time()
            if c == "model":
                name = body.get("name")
                if name not in MODELS:
                    return self._send({"ok": False, "error": "unknown model '%s' (have %s)" % (name, ", ".join(MODELS))})
                if name != STATE["model"]:
                    STATE["model"] = name
                    STATE["rev"] += 1
                return self._send({"ok": True, "mode": STATE["mode"], "model": name, "rev": STATE["rev"]})
            if c == "talk":
                on = body.get("on", True)
                STATE["talk"] = float(body.get("level", 0.7)) if on else 0.0
                STATE["talk_until"] = now + float(body.get("duration", 3600 if on else 0))
            elif c == "angry":
                STATE["angry_until"] = now + 4
            elif c == "dance":
                STATE["dance_until"] = now + 8
            elif c == "sleep":
                STATE["sleep"] = True
                STATE["mode"] = "sleep"
            elif c in ("wake", "reset", "idle"):
                STATE["sleep"] = False
                STATE["mode"] = "idle"
            elif c == "track":
                STATE["mode"] = "track"
            mode = "angry" if c == "angry" else "dance" if c == "dance" else STATE["mode"]
            self._send({"ok": True, "mode": mode})
        elif self.path == "/player":
            LAST["player"] = body
            LAST["player_count"] += 1
            self._send({"ok": True})
        elif self.path == "/config":
            LAST["config"] = body
            self._send({"ok": True})
        else:
            self._send({"ok": False, "error": "not found"}, 404)


ThreadingHTTPServer(("127.0.0.1", PORT), H).serve_forever()
