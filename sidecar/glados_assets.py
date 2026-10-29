#!/usr/bin/env python3
"""CONTRACT v3 asset library: list / load / validate `assets/<name>/{asset.json, atlas.png}`, export the procedural
model as an asset, and derive MuJoCo mass geoms + chain geometry from any asset.

    python3 glados_assets.py export-procedural      # (re)write assets/procedural from glados_model.py
    python3 glados_assets.py check [name ...]       # validate assets, print stats
"""

import io
import json
import math
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ASSET_DIR = os.environ.get("GLADOS_ASSETS", os.path.join(HERE, "assets"))
BODY_ORDER = ["mount", "arm1", "arm2", "torso", "neck", "head"]
PARENT = {"arm1": "mount", "arm2": "arm1", "torso": "arm2", "neck": "torso", "head": "neck"}
TAGS = (None, "eye", "emissive", "glass")
MAX_TRIS = 40000


def list_assets():
    """Names of asset dirs that contain an asset.json (sorted, procedural first)."""
    if not os.path.isdir(ASSET_DIR):
        return []
    out = [d for d in sorted(os.listdir(ASSET_DIR)) if os.path.isfile(os.path.join(ASSET_DIR, d, "asset.json"))]
    return sorted(out, key=lambda n: (n != "procedural", n))


class Asset:
    """A loaded asset: `.data` (the asset.json dict), `.atlas_png` (bytes), helpers for the sim."""

    def __init__(self, name, data, atlas_png):
        self.name = name
        self.data = data
        self.atlas_png = atlas_png
        self.bodies = {b["name"]: b for b in data["bodies"]}
        self.triangles = int(data.get("triangles") or sum(len(m["tris"]) // 3 for m in data["meshes"]))

    @property
    def title(self):
        return self.data.get("title", self.name)

    @property
    def credit(self):
        return self.data.get("credit", "")

    def summary(self):
        return {"name": self.name, "title": self.title, "credit": self.credit, "triangles": self.triangles}

    def rest(self, body):
        r = self.bodies[body]["rest"]
        return np.array(r[:3], float), np.array(r[3:7], float)

    def body_points(self, body, min_w=0.5):
        """Body-local points of everything that moves with `body` (rigid meshes + dominant skinned verts)."""
        pts = []
        p0, q0 = self.rest(body)
        R0 = quat_mat(q0)
        for m in self.data["meshes"]:
            V = np.asarray(m["verts"], float).reshape(-1, 3)
            sk = m.get("skin")
            if not sk:
                if m["body"] == body:
                    pts.append(V)
                continue
            if body not in sk["bodies"]:
                continue
            bi = sk["bodies"].index(body)
            j = np.asarray(sk["j"], int).reshape(-1, 2)
            w = np.asarray(sk["w"], float)
            sel = ((j[:, 0] == bi) & (w >= min_w)) | ((j[:, 1] == bi) & (1 - w >= min_w))
            if sel.any():
                pts.append((V[sel] - p0) @ R0)          # world rest -> body-local
        return np.vstack(pts) if pts else np.zeros((0, 3))

    def model_json(self, rev):
        """The /model.json payload: asset keys + rev + atlas URL."""
        d = dict(self.data)
        d["rev"] = rev
        d["atlas"] = f"/atlas.png?rev={rev}"
        return d


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def load_asset(name):
    if name == "procedural" and not os.path.isfile(os.path.join(ASSET_DIR, name, "asset.json")):
        export_procedural()
    d = os.path.join(ASSET_DIR, name)
    with open(os.path.join(d, "asset.json")) as f:
        data = json.load(f)
    with open(os.path.join(d, data.get("atlas") or "atlas.png"), "rb") as f:
        png = f.read()
    probs = validate(data)
    if probs:
        raise ValueError(f"asset {name}: " + "; ".join(probs[:5]))
    return Asset(name, data, png)


def validate(a):
    """List of problems (empty = valid CONTRACT v3 asset)."""
    p = []
    if a.get("version") != 3:
        p.append("version != 3")
    names = [b.get("name") for b in a.get("bodies", [])]
    if names != BODY_ORDER:
        p.append(f"bodies {names} != {BODY_ORDER}")
    for b in a.get("bodies", []):
        if b.get("parent") != PARENT.get(b.get("name")):
            p.append(f"{b.get('name')}: parent {b.get('parent')}")
        if len(b.get("joint_pos", [])) != 3 or len(b.get("rest", [])) != 7:
            p.append(f"{b.get('name')}: joint_pos/rest shape")
    tot = 0
    for m in a.get("meshes", []):
        nv = len(m.get("verts", [])) // 3
        t = m.get("tris", [])
        nm = m.get("name", "?")
        if len(m.get("verts", [])) % 3 or len(t) % 3 or not t:
            p.append(f"{nm}: verts/tris length")
            continue
        if min(t) < 0 or max(t) >= nv:
            p.append(f"{nm}: tri index out of range")
        if len(m.get("normals", [])) != 3 * nv:
            p.append(f"{nm}: normals length")
        uv = m.get("uv")
        if uv is None or len(uv) != 2 * nv or (uv and (min(uv) < 0 or max(uv) > 1)):
            p.append(f"{nm}: uv missing/out of range")
        if m.get("body") not in BODY_ORDER:
            p.append(f"{nm}: body {m.get('body')}")
        if m.get("tag") not in TAGS:
            p.append(f"{nm}: tag {m.get('tag')}")
        if len(m.get("rgba", [])) != 4:
            p.append(f"{nm}: rgba")
        sk = m.get("skin")
        if sk:
            if any(b not in BODY_ORDER for b in sk.get("bodies", [])):
                p.append(f"{nm}: skin bodies")
            if len(sk.get("j", [])) != 2 * nv or len(sk.get("w", [])) != nv:
                p.append(f"{nm}: skin j/w length")
            elif sk["j"] and (min(sk["j"]) < 0 or max(sk["j"]) >= len(sk["bodies"])):
                p.append(f"{nm}: skin j range")
        tot += len(t) // 3
    if tot > MAX_TRIS:
        p.append(f"{tot} triangles > {MAX_TRIS}")
    if a.get("triangles") not in (None, tot):
        p.append(f"triangles field {a.get('triangles')} != {tot}")
    e = a.get("eye") or {}
    if e.get("body") not in BODY_ORDER or len(e.get("pos", [])) != 3 or len(e.get("axis", [])) != 3:
        p.append("eye")
    if not any(m.get("tag") == "eye" for m in a.get("meshes", [])):
        p.append("no eye-tagged mesh")
    return p


# ------------------------------------------------------------------------------------------- physics helpers
def mass_geoms(asset, body, mass):
    """Invisible mass geoms for `body` of a non-procedural asset: one PCA ellipsoid fitted to the body's points
    (body-local) carrying `mass` kg; a body without geometry (pure pivot) gets a thin capsule from its joint to
    its child's joint. Returns dicts in the glados_model.mass_geoms format."""
    pts = asset.body_points(body)
    if len(pts) >= 8:
        c = pts.mean(0)
        cov = np.cov((pts - c).T)
        ev, evec = np.linalg.eigh(cov)
        semi = np.sqrt(np.maximum(ev, 1e-6) * 3.0)          # surface points: var = a^2/3 per axis
        semi = np.clip(semi, 0.08, None)
        vol = 4 / 3 * math.pi * float(np.prod(semi))
        R = evec.copy()
        if np.linalg.det(R) < 0:
            R[:, 0] = -R[:, 0]
        return [dict(type="ellipsoid", size=[float(x) for x in semi], pos=[float(x) for x in c],
                     quat=mat_quat(R), density=mass / vol)]
    kids = [b for b, p in PARENT.items() if p == body]
    end = np.array(asset.bodies[kids[0]]["joint_pos"], float) if kids else np.array([0, 0, -0.5])
    r = 0.25
    L = float(np.linalg.norm(end))
    vol = math.pi * r * r * L + 4 / 3 * math.pi * r ** 3
    return [dict(type="capsule", fromto=[0, 0, 0, *[float(x) for x in end]], size=[r, 0, 0], density=mass / vol)]


def mat_quat(R):
    """Rotation matrix -> quat [w,x,y,z]."""
    tr = R[0, 0] + R[1, 1] + R[2, 2]
    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2
        q = [0.25 * s, (R[2, 1] - R[1, 2]) / s, (R[0, 2] - R[2, 0]) / s, (R[1, 0] - R[0, 1]) / s]
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2
        q = [(R[2, 1] - R[1, 2]) / s, 0.25 * s, (R[0, 1] + R[1, 0]) / s, (R[0, 2] + R[2, 0]) / s]
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2
        q = [(R[0, 2] - R[2, 0]) / s, (R[0, 1] + R[1, 0]) / s, 0.25 * s, (R[1, 2] + R[2, 1]) / s]
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2
        q = [(R[1, 0] - R[0, 1]) / s, (R[0, 2] + R[2, 0]) / s, (R[1, 2] + R[2, 1]) / s, 0.25 * s]
    n = math.sqrt(sum(x * x for x in q))
    return [float(x / n) for x in q]


# ------------------------------------------------------------------------------------------- procedural export
def _png_bytes(rgba_u8):
    from PIL import Image
    buf = io.BytesIO()
    Image.fromarray(rgba_u8, "RGBA").save(buf, format="PNG")
    return buf.getvalue()


def procedural_asset():
    """The glados_model.py geometry in the v3 asset format (white 16x16 atlas, colour in `rgba`)."""
    sys.path.insert(0, HERE)
    import glados_model as gm
    m1 = gm.model_json()
    rel = gm.body_pos()
    wp = gm.body_world_pos()
    bodies = []
    for b in gm.BODY_ORDER:
        bodies.append({"name": b, "parent": gm.PARENT.get(b), "joint_pos": [round(float(x), 5) for x in rel[b]],
                       "rest": [round(float(x), 5) for x in wp[b]] + [1.0, 0.0, 0.0, 0.0]})
    meshes = []
    for b in m1["bodies"]:
        for me in b["meshes"]:
            nv = len(me["verts"]) // 3
            meshes.append({"body": b["name"], "name": me["name"], "verts": me["verts"], "normals": me["normals"],
                           "uv": [0.5] * (2 * nv), "tris": me["tris"], "rgba": me["rgba"],
                           "tag": me.get("tag"), "skin": None})
    ep, ea = gm.eye_site()
    hc = gm.head_center()
    return {"version": 3, "name": "procedural", "title": "Classic (procedural, original geometry)",
            "credit": "original procedural geometry generated by glados_model.py (no Valve assets)",
            "bodies": bodies, "meshes": meshes,
            "eye": {"body": "head", "pos": [round(float(x), 4) for x in ep],
                    "axis": [round(float(x), 4) for x in ea / np.linalg.norm(ea)]},
            "voice": {"body": "head", "pos": [round(float(x), 4) for x in hc]},
            "atlas": "atlas.png", "triangles": m1["triangles"]}


def export_procedural(out=None):
    out = out or os.path.join(ASSET_DIR, "procedural")
    os.makedirs(out, exist_ok=True)
    a = procedural_asset()
    with open(os.path.join(out, "asset.json"), "w") as f:
        json.dump(a, f, separators=(",", ":"))
    with open(os.path.join(out, "atlas.png"), "wb") as f:
        f.write(_png_bytes(np.full((16, 16, 4), 255, np.uint8)))
    return a


def main():
    if len(sys.argv) > 1 and sys.argv[1] == "export-procedural":
        a = export_procedural()
        print(f"wrote assets/procedural: {a['triangles']} tris, {len(a['meshes'])} meshes")
        return
    names = sys.argv[2:] if len(sys.argv) > 2 else list_assets()
    bad = 0
    for n in names:
        with open(os.path.join(ASSET_DIR, n, "asset.json")) as f:
            a = json.load(f)
        p = validate(a)
        bad += bool(p)
        sk = sum(1 for m in a["meshes"] if m.get("skin"))
        tags = sorted({m.get("tag") for m in a["meshes"] if m.get("tag")})
        print(f"{n:12s} {'OK ' if not p else 'BAD'} {a.get('triangles')} tris, {len(a['meshes'])} meshes "
              f"({sk} skinned), tags {tags}, eye {a['eye']}")
        for x in p:
            print("   -", x)
    sys.exit(1 if bad else 0)


if __name__ == "__main__":
    main()
