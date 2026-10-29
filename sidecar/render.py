#!/usr/bin/env python3
"""Tiny numpy software rasteriser for previews: renders a contract /model.json + /pose with Lambert lighting.

No GL needed. Per-pixel interpolated vertex normals, z-buffer, key + fill + ambient light, unlit glowing eye
(colour/glow from pose["eye"]), a dark ceiling plane at z=0 for context.
"""

import math

import numpy as np
from PIL import Image


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


def gather(model, pose, extra=()):
    """World-space triangles: list of (V (n,3,3), N (n,3,3), rgb, emissive)."""
    out = []
    eye = pose.get("eye", [1.0, 0.824, 0.29, 1.0])
    for b in model["bodies"]:
        tr = pose["bodies"].get(b["name"])
        if tr is None:
            continue
        p = np.array(tr[:3])
        R = quat_mat(tr[3:7])
        for m in b["meshes"]:
            v = np.array(m["verts"], float).reshape(-1, 3) @ R.T + p
            t = np.array(m["tris"], np.int64).reshape(-1, 3)
            if "normals" in m:
                n = np.array(m["normals"], float).reshape(-1, 3) @ R.T
            else:
                fn = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])
                n = np.zeros_like(v)
                for k in range(3):
                    np.add.at(n, t[:, k], fn)
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-12)
            if m.get("tag") == "eye":
                rgb = np.array(eye[:3]) * (0.35 + 0.65 * eye[3]) + 0.25 * eye[3]
                out.append((v[t], n[t], np.clip(rgb, 0, 1), True))
            else:
                out.append((v[t], n[t], np.array(m["rgba"][:3]), False))
    out.extend(extra)
    return out


def ceiling_quad(half=6.0, z=0.003):
    v = np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]])
    t = np.array([[0, 2, 1], [0, 3, 2]])
    n = np.tile([0, 0, -1.0], (4, 1))
    return (v[t], n[t], np.array([0.30, 0.31, 0.33]), False)


def floor_quad(z, half=8.0):
    v = np.array([[-half, -half, z], [half, -half, z], [half, half, z], [-half, half, z]])
    t = np.array([[0, 1, 2], [0, 2, 3]])
    n = np.tile([0, 0, 1.0], (4, 1))
    return (v[t], n[t], np.array([0.42, 0.42, 0.40]), False)


def render(model, pose, cam_pos, target, size=(600, 800), fov=40.0, extra=(), marker=None, line=None,
           bg_top=(0.55, 0.60, 0.66), bg_bot=(0.20, 0.22, 0.25), flat=False):
    """flat=True shades every triangle with its face normal (what Iris shader packs do in-game)."""
    W, H = size
    cam_pos = np.asarray(cam_pos, float)
    Rc = look_at(cam_pos, target)
    f = 0.5 * H / math.tan(math.radians(fov) / 2)
    img = np.zeros((H, W, 3))
    g = np.linspace(0, 1, H)[:, None, None]
    img[:] = (1 - g) * np.array(bg_top) + g * np.array(bg_bot)
    zbuf = np.full((H, W), np.inf)
    key = np.array([0.55, -0.45, 0.70])
    key /= np.linalg.norm(key)
    fill = np.array([-0.6, 0.5, -0.2])
    fill /= np.linalg.norm(fill)
    view_dir_world = None
    for V, N, rgb, emis in gather(model, pose, extra):
        if flat:
            fn = np.cross(V[:, 1] - V[:, 0], V[:, 2] - V[:, 0])
            fn /= np.maximum(np.linalg.norm(fn, axis=1, keepdims=True), 1e-12)
            N = np.repeat(fn[:, None, :], 3, axis=1)
        pc = (V - cam_pos) @ Rc.T          # (n,3,3) camera coords
        z = -pc[..., 2]
        ok = (z > 0.05).all(axis=1)
        pc, z, Vn, Nn = pc[ok], z[ok], V[ok], N[ok]
        sx = W / 2 + f * pc[..., 0] / z
        sy = H / 2 - f * pc[..., 1] / z
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
            # perspective-correct depth
            iz = a / zs[0] + b / zs[1] + c / zs[2]
            depth = 1.0 / iz
            sub = zbuf[y0:y1 + 1, x0:x1 + 1]
            win = ins & (depth < sub)
            if not win.any():
                continue
            sub[win] = depth[win]
            if emis:
                col = np.broadcast_to(rgb, (int(win.sum()), 3))
            else:
                wa, wb, wc = (a / zs[0] * depth)[win], (b / zs[1] * depth)[win], (c / zs[2] * depth)[win]
                n = wa[:, None] * Nn[i, 0] + wb[:, None] * Nn[i, 1] + wc[:, None] * Nn[i, 2]
                n /= np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)
                pw = wa[:, None] * Vn[i, 0] + wb[:, None] * Vn[i, 1] + wc[:, None] * Vn[i, 2]
                vd = cam_pos - pw
                vd /= np.linalg.norm(vd, axis=1, keepdims=True)
                n = np.where(((n * vd).sum(1) < 0)[:, None], -n * 0.6, n)  # back faces: darker, two-sided
                lam = 0.28 + 0.72 * np.clip(n @ key, 0, 1) + 0.22 * np.clip(n @ fill, 0, 1)
                refl = 2 * (n @ key)[:, None] * n - key
                spec = 0.12 * np.clip((refl * vd).sum(1), 0, 1) ** 24
                rim = 0.10 * (1 - np.clip((n * vd).sum(1), 0, 1)) ** 3
                col = np.clip(rgb[None] * lam[:, None] + (rim + spec)[:, None], 0, 1)
            img[y0:y1 + 1, x0:x1 + 1][win] = col
    if line is not None:  # dashed red sight line between two world points (eye -> target), drawn on top
        a, b = (np.asarray(x, float) for x in line)
        for k, u in enumerate(np.linspace(0, 1, 400)):
            if (k // 8) % 2:
                continue
            pc = (a + u * (b - a) - cam_pos) @ Rc.T
            if -pc[2] <= 0:
                continue
            mx, my = int(W / 2 + f * pc[0] / -pc[2]), int(H / 2 - f * pc[1] / -pc[2])
            if 1 <= mx < W - 1 and 1 <= my < H - 1:
                img[my - 1:my + 1, mx - 1:mx + 1] = (0.95, 0.15, 0.1)
    if marker is not None:  # draw a red dot at a world point (e.g. the tracked target)
        pc = (np.asarray(marker, float) - cam_pos) @ Rc.T
        if -pc[2] > 0:
            mx, my = W / 2 + f * pc[0] / -pc[2], H / 2 - f * pc[1] / -pc[2]
            yy, xx = np.ogrid[:H, :W]
            img[(xx - mx) ** 2 + (yy - my) ** 2 < 49] = (0.9, 0.1, 0.1)
    return Image.fromarray((np.clip(img, 0, 1) ** (1 / 1.15) * 255).astype(np.uint8))
