#!/usr/bin/env python3
"""Textured numpy software rasteriser for CONTRACT v3 /model.json + /pose (+ /atlas.png) - what the mod should draw.

- rigid meshes: body-local verts -> pose transform; skinned meshes: v = w*T_a*R_a^-1*v + (1-w)*T_b*R_b^-1*v (R = rest)
- per-pixel perspective-correct uv -> bilinear atlas sample (v=0 = top row) x rgba tint
- tags: eye = fullbright x pose.eye colour/glow; emissive = fullbright; glass = alpha blended (rgba[3])
- lighting in linear space: ambient + key sun (with a shadow map) + fill sun, Lambert; output sRGB. The light rig is
  LIGHTS (shared with tools/blender_ref.py so the Blender reference is lit the same way).
"""

import math

import numpy as np
from PIL import Image

LIGHTS = {
    "key_dir": [0.55, -0.65, 0.30],       # direction TOWARD the light (sim frame): front-left, slightly above
    "key": 7.5,                            # sun strength (Blender W/m^2: a diffuse surface gets key/pi)
    "fill_dir": [-0.7, 0.5, -0.15],
    "fill": 1.8,
    "ambient": [0.10, 0.105, 0.115],       # world colour (strength 1)
    "bg": [0.10, 0.105, 0.115],
}


EYE_DEFAULT = np.array([1.0, 0.82, 0.29])   # = GladosMath.EYE_DEFAULT in the mod
EYE_BASE = 1.0                               # the mod uses 0.75 (talk-pulse headroom); 1.0 = the Blender look


def srgb2lin(c):
    c = np.asarray(c, np.float32)
    return np.where(c <= 0.04045, c / 12.92, ((c + 0.055) / 1.055) ** 2.4)


def lin2srgb(c):
    c = np.clip(np.asarray(c, np.float32), 0, None)
    return np.where(c <= 0.0031308, 12.92 * c, 1.055 * np.power(c, 1 / 2.4) - 0.055)


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def look_at(eye, target, up=(0, 0, 1)):
    f = np.asarray(target, float) - eye
    f /= np.linalg.norm(f)
    r = np.cross(f, up)
    r /= np.linalg.norm(r)
    u = np.cross(r, f)
    return np.stack([r, u, -f])  # rows: camera x, y, z (camera looks along -z)


def posed_meshes(model, pose):
    """[(V (n,3), N (n,3), UV (n,2), tris (m,3), rgba, tag)] in the sim world frame."""
    bodies = {b["name"]: b for b in model["bodies"]}
    T = {}
    for name, tr in pose["bodies"].items():
        T[name] = (np.array(tr[:3], float), quat_mat(tr[3:7]))
    rest = {n: (np.array(b["rest"][:3], float), quat_mat(b["rest"][3:7])) for n, b in bodies.items()}
    out = []
    for m in model["meshes"]:
        V = np.asarray(m["verts"], float).reshape(-1, 3)
        N = np.asarray(m["normals"], float).reshape(-1, 3)
        UV = np.asarray(m.get("uv") or [0.5] * (2 * len(V)), float).reshape(-1, 2)
        t = np.asarray(m["tris"], np.int64).reshape(-1, 3)
        sk = m.get("skin")
        if not sk:
            if m["body"] not in T:
                continue
            p, R = T[m["body"]]
            Vw, Nw = V @ R.T + p, N @ R.T
        else:
            j = np.asarray(sk["j"], np.int64).reshape(-1, 2)
            w = np.asarray(sk["w"], float)[:, None]
            Vw = np.zeros_like(V)
            Nw = np.zeros_like(N)
            for k, wk in ((0, w), (1, 1 - w)):
                for bi, bn in enumerate(sk["bodies"]):
                    sel = j[:, k] == bi
                    if not sel.any():
                        continue
                    p, R = T[bn]
                    p0, R0 = rest[bn]
                    M = R @ R0.T
                    Vw[sel] += wk[sel] * ((V[sel] - p0) @ M.T + p)
                    Nw[sel] += wk[sel] * (N[sel] @ M.T)
        Nw /= np.maximum(np.linalg.norm(Nw, axis=1, keepdims=True), 1e-12)
        out.append((Vw, Nw, UV, t, np.array(m.get("rgba", [1, 1, 1, 1]), float), m.get("tag")))
    return out


def _raster(sx, sy, z, W, H, persp=True):
    """Yield (i, y0, y1, x0, x1, win-mask, a, b, c, depth) for every triangle i (barycentrics perspective-correct)."""
    for i in range(len(sx)):
        xs, ys, zs = sx[i], sy[i], z[i]
        x0, x1 = max(int(math.floor(xs.min())), 0), min(int(math.ceil(xs.max())), W - 1)
        y0, y1 = max(int(math.floor(ys.min())), 0), min(int(math.ceil(ys.max())), H - 1)
        if x0 > x1 or y0 > y1:
            continue
        den = (ys[1] - ys[2]) * (xs[0] - xs[2]) + (xs[2] - xs[1]) * (ys[0] - ys[2])
        if abs(den) < 1e-12:
            continue
        px, py = np.meshgrid(np.arange(x0, x1 + 1) + 0.5, np.arange(y0, y1 + 1) + 0.5)
        a = ((ys[1] - ys[2]) * (px - xs[2]) + (xs[2] - xs[1]) * (py - ys[2])) / den
        b = ((ys[2] - ys[0]) * (px - xs[2]) + (xs[0] - xs[2]) * (py - ys[2])) / den
        c = 1 - a - b
        ins = (a >= -1e-6) & (b >= -1e-6) & (c >= -1e-6)
        if not ins.any():
            continue
        if persp:
            iz = a / zs[0] + b / zs[1] + c / zs[2]
            depth = 1.0 / iz
            a, b, c = a / zs[0] * depth, b / zs[1] * depth, c / zs[2] * depth
        else:
            depth = a * zs[0] + b * zs[1] + c * zs[2]
        yield i, y0, y1, x0, x1, ins, a, b, c, depth


class ShadowMap:
    """Orthographic depth map along the key light."""

    def __init__(self, meshes, light_dir, res=1536):
        L = np.asarray(light_dir, float)
        L /= np.linalg.norm(L)
        up = (0, 0, 1) if abs(L[2]) < 0.95 else (1, 0, 0)
        self.R = look_at(np.zeros(3), -L, up)        # camera looking along -L (from the light)
        allv = np.vstack([m[0] for m in meshes])
        pc = allv @ self.R.T
        self.lo, self.hi = pc[:, :2].min(0) - 0.1, pc[:, :2].max(0) + 0.1
        self.res = res
        self.s = (res - 1) / float((self.hi - self.lo).max())
        self.z = np.full((res, res), np.inf)
        for V, N, UV, t, rgba, tag in meshes:
            if tag in ("eye", "emissive", "glass"):
                continue
            P = V[t] @ self.R.T
            sx = (P[..., 0] - self.lo[0]) * self.s
            sy = (self.hi[1] - P[..., 1]) * self.s
            d = -P[..., 2]
            for i, y0, y1, x0, x1, ins, a, b, c, depth in _raster(sx, sy, d, res, res, persp=False):
                sub = self.z[y0:y1 + 1, x0:x1 + 1]
                w = ins & (depth < sub)
                sub[w] = depth[w]

    def lit(self, pw, n_dot_l):
        P = pw @ self.R.T
        x = np.clip(((P[:, 0] - self.lo[0]) * self.s).astype(int), 0, self.res - 1)
        y = np.clip(((self.hi[1] - P[:, 1]) * self.s).astype(int), 0, self.res - 1)
        bias = 0.03 + 0.05 * np.sqrt(np.clip(1 - n_dot_l ** 2, 0, 1))
        # 3x3 PCF
        acc = np.zeros(len(P))
        for dy in (-1, 0, 1):
            for dx in (-1, 0, 1):
                zz = self.z[np.clip(y + dy, 0, self.res - 1), np.clip(x + dx, 0, self.res - 1)]
                acc += (-P[:, 2] <= zz + bias)
        return acc / 9.0


def sample(atlas, uv):
    """Bilinear sample, atlas (H,W,4) float linear-rgb + alpha, uv (n,2) with v=0 at the top row."""
    H, W = atlas.shape[:2]
    x = np.clip(uv[:, 0] * W - 0.5, 0, W - 1)
    y = np.clip(uv[:, 1] * H - 0.5, 0, H - 1)
    x0, y0 = np.floor(x).astype(int), np.floor(y).astype(int)
    x1, y1 = np.minimum(x0 + 1, W - 1), np.minimum(y0 + 1, H - 1)
    fx, fy = (x - x0)[:, None], (y - y0)[:, None]
    return ((atlas[y0, x0] * (1 - fx) + atlas[y0, x1] * fx) * (1 - fy) +
            (atlas[y1, x0] * (1 - fx) + atlas[y1, x1] * fx) * fy)


def load_atlas(img):
    a = np.asarray(img.convert("RGBA"), np.float32) / 255.0
    a[..., :3] = srgb2lin(a[..., :3])
    return a


def render(model, pose, atlas_img, cam_pos, target, size=(640, 800), fov=40.0, lights=None, shadows=True):
    """fov = vertical field of view (deg). Returns a PIL image."""
    Lt = dict(LIGHTS, **(lights or {}))
    W, H = size
    cam_pos = np.asarray(cam_pos, float)
    Rc = look_at(cam_pos, target)
    f = 0.5 * H / math.tan(math.radians(fov) / 2)
    atlas = load_atlas(atlas_img)
    meshes = posed_meshes(model, pose)
    key = np.asarray(Lt["key_dir"], float)
    key /= np.linalg.norm(key)
    fill = np.asarray(Lt["fill_dir"], float)
    fill /= np.linalg.norm(fill)
    amb = np.asarray(Lt["ambient"], float)
    sm = ShadowMap(meshes, key) if shadows else None
    img = np.zeros((H, W, 3), np.float32)
    img[:] = np.asarray(Lt["bg"], float)
    zbuf = np.full((H, W), np.inf)
    eye = pose.get("eye", [1.0, 0.824, 0.29, 1.0])
    glass = []
    for mi, (V, N, UV, t, rgba, tag) in enumerate(meshes):
        if tag == "glass":
            glass.append(mi)
            continue
        _draw(img, zbuf, V, N, UV, t, rgba, tag, cam_pos, Rc, f, W, H, atlas, key, fill, amb, Lt, sm, eye)
    for mi in glass:
        V, N, UV, t, rgba, tag = meshes[mi]
        _draw(img, zbuf, V, N, UV, t, rgba, tag, cam_pos, Rc, f, W, H, atlas, key, fill, amb, Lt, sm, eye,
              blend=True)
    out = Image.fromarray((np.clip(lin2srgb(img), 0, 1) * 255 + 0.5).astype(np.uint8))
    out.info["mask"] = np.isfinite(zbuf)
    return out


def _draw(img, zbuf, V, N, UV, t, rgba, tag, cam_pos, Rc, f, W, H, atlas, key, fill, amb, Lt, sm, eye,
          blend=False):
    tint = srgb2lin(rgba[:3])
    pc = (V[t] - cam_pos) @ Rc.T
    z = -pc[..., 2]
    ok = (z > 0.05).all(axis=1)
    tt = t[ok]
    pc, z = pc[ok], z[ok]
    sx = W / 2 + f * pc[..., 0] / z
    sy = H / 2 - f * pc[..., 1] / z
    Vt, Nt, UVt = V[tt], N[tt], UV[tt]
    for i, y0, y1, x0, x1, ins, a, b, c, depth in _raster(sx, sy, z, W, H):
        sub = zbuf[y0:y1 + 1, x0:x1 + 1]
        win = ins & (depth < sub)
        if not win.any():
            continue
        wa, wb, wc = a[win][:, None], b[win][:, None], c[win][:, None]
        uv = wa * UVt[i, 0] + wb * UVt[i, 1] + wc * UVt[i, 2]
        tex = sample(atlas, uv)
        alb = tex[:, :3] * tint
        if tag == "eye":
            # like the mod (GladosMath.eyeTint): texels hold the normal look, tinted by pose.eye / EYE_DEFAULT
            # x (0.35 + 0.65*glow) x EYE_BASE, multiplied in sRGB (vertex colour x texture)
            k = (0.35 + 0.65 * min(max(eye[3], 0), 1)) * EYE_BASE
            ratio = np.maximum(np.asarray(eye[:3], float), 0) / EYE_DEFAULT * k
            col = srgb2lin(np.clip(lin2srgb(alb) * ratio, 0, 1))
        elif tag == "emissive":
            col = alb
        else:
            n = wa * Nt[i, 0] + wb * Nt[i, 1] + wc * Nt[i, 2]
            n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
            pw = wa * Vt[i, 0] + wb * Vt[i, 1] + wc * Vt[i, 2]
            vd = cam_pos - pw
            vd /= np.linalg.norm(vd, axis=1, keepdims=True)
            n = np.where(((n * vd).sum(1) < 0)[:, None], -n, n)       # two-sided
            nk = n @ key
            sh = sm.lit(pw, nk) if sm is not None else 1.0
            irr = amb[None] + (Lt["key"] / math.pi * np.clip(nk, 0, 1) * sh)[:, None] + \
                (Lt["fill"] / math.pi * np.clip(n @ fill, 0, 1))[:, None]    # Blender sun: E/pi on diffuse
            col = alb * irr
        if blend:
            al = rgba[3] * tex[:, 3:4]
            dst = img[y0:y1 + 1, x0:x1 + 1][win]
            img[y0:y1 + 1, x0:x1 + 1][win] = al * col + (1 - al) * dst
        else:
            sub[win] = depth[win]
            img[y0:y1 + 1, x0:x1 + 1][win] = col
