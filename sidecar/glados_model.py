#!/usr/bin/env python3
"""Procedural, ORIGINAL GLaDOS-style geometry (numpy only): lathe / sweep surfaces -> body-local meshes.

Sim frame: z-up, metres, origin = ceiling mount point, she hangs in -z and faces +x.
All bodies have identity orientation in the design pose (all joints 0); BODY_POS gives each body's
position relative to its parent. Mesh verts are body-local.

Silhouette (v2, classic hanging ceiling-AI look, all original geometry):
  mount  big dark ceiling hub plate + heavy outer ring + drooping cables
  arm1   a bundle of many thick black cables converging from the hub into the body (yaw + swing joint at the hub)
  arm2   the dark "shoulder" ring cluster where the cables enter the body (big hoops + clamps)
  torso  white inverted torso/hip shell (wide at the top, tapering down, curved), split into left/right panels
         with a dark cable channel down the front and a ribbed dark spine down the back
  neck   dark articulated neck: down out of the torso bottom, curving forward and up (U)
  head   big horizontal white "lamp" helmet (upper/lower shells split by a dark band), dark visor ring on the
         front face around ONE glowing eye; face end points forward/down
"""

import math

import numpy as np

# ---------------------------------------------------------------- palette (contract)
SHELL = (0.910, 0.910, 0.902, 1.0)      # #E8E8E6
MECH = (0.227, 0.235, 0.251, 1.0)       # #3A3C40
MECH_DARK = (0.165, 0.169, 0.180, 1.0)  # #2A2B2E
RING = (0.082, 0.082, 0.082, 1.0)       # #151515
EYE = (1.0, 0.824, 0.290, 1.0)          # #FFD24A

# ---------------------------------------------------------------- layout (design pose)
HUB_R = 1.80            # ceiling plate radius
HUB_DEPTH = 0.55        # ceiling -> yaw/swing joint (bottom of the hub)
ARM1_LEN = 1.95         # cable bundle
ARM2_LEN = 0.75         # shoulder ring cluster
TORSO_LEN = 3.10
# torso centre line (top -> bottom, body-local): bulges back (-x), bottom swings forward (+x)
TORSO_CTRL = np.array([[0.0, 0.0, 0.0], [-0.45, 0.0, -1.00], [-0.40, 0.0, -2.20], [0.32, 0.0, -3.10]])
# neck: out of the torso bottom, down, forward and up into the back of the head
NECK_CTRL = np.array([[0.0, 0.0, 0.0], [0.08, 0.0, -0.62], [0.78, 0.0, -0.85], [1.08, 0.0, -0.28]])
NECK_END = NECK_CTRL[3].copy()
# head spine (head-local; the head joint = origin sits at the upper back of the helmet)
HEAD_CTRL = np.array([[0.02, 0.0, -0.38], [0.75, 0.0, -0.34], [1.45, 0.0, -0.44], [2.00, 0.0, -0.76]])
HEAD_H, HEAD_W = 0.54, 0.47      # max half height / half width of the helmet section
HEAD_SCALE = 1.0                 # kept for API compatibility (geometry is built at final size)
TORSO_U = np.array([0.0, 0.0, -1.0])   # compatibility: nominal hang direction


def torso_profile(t):
    """(half width y, half depth x) of the torso section at fraction t (0 = top collar, 1 = bottom joint)."""
    # inverted figure: broad hips at the top, a waist, slightly wider shoulders, then the neck joint
    w = [(0.0, 0.48), (0.03, 0.80), (0.08, 1.00), (0.18, 1.07), (0.30, 0.99), (0.44, 0.78), (0.57, 0.64),
         (0.71, 0.68), (0.83, 0.69), (0.93, 0.55), (1.0, 0.34)]
    d = [(0.0, 0.46), (0.03, 0.70), (0.08, 0.84), (0.18, 0.90), (0.30, 0.85), (0.44, 0.68), (0.57, 0.58),
         (0.71, 0.61), (0.83, 0.61), (0.93, 0.49), (1.0, 0.32)]
    return _smooth_interp(w, t), _smooth_interp(d, t)


def head_profile(t):
    """Radius factor (0..1 of HEAD_H/HEAD_W) along the head spine: rounded back, full body, slight front taper."""
    pts = [(0.0, 0.0), (0.015, 0.36), (0.05, 0.62), (0.11, 0.82), (0.20, 0.94), (0.34, 1.0), (0.55, 0.99),
           (0.75, 0.95), (0.90, 0.91), (0.97, 0.88), (1.0, 0.86)]
    return _smooth_interp(pts, t)


# ---------------------------------------------------------------- small math helpers
def _n(v):
    v = np.asarray(v, float)
    return v / np.linalg.norm(v)


def _smooth_interp(pts, t):
    """Catmull-Rom through (x, y) control points, evaluated at t (array or scalar)."""
    p = np.asarray(pts, float)
    t = np.atleast_1d(np.asarray(t, float))
    out = np.empty_like(t)
    xs = p[:, 0]
    for k, tv in enumerate(t):
        i = int(np.clip(np.searchsorted(xs, tv) - 1, 0, len(xs) - 2))
        x0, x1 = xs[i], xs[i + 1]
        u = 0.0 if x1 == x0 else (tv - x0) / (x1 - x0)
        y0 = p[max(i - 1, 0), 1]
        y1, y2 = p[i, 1], p[i + 1, 1]
        y3 = p[min(i + 2, len(p) - 1), 1]
        # tangents scaled by local spacing (non-uniform Catmull-Rom)
        xa, xb = xs[max(i - 1, 0)], xs[min(i + 2, len(p) - 1)]
        m1 = (y2 - y0) / max(x1 - xa, 1e-9) * (x1 - x0)
        m2 = (y3 - y1) / max(xb - x0, 1e-9) * (x1 - x0)
        h00 = 2 * u ** 3 - 3 * u ** 2 + 1
        h10 = u ** 3 - 2 * u ** 2 + u
        h01 = -2 * u ** 3 + 3 * u ** 2
        h11 = u ** 3 - u ** 2
        out[k] = h00 * y1 + h10 * m1 + h01 * y2 + h11 * m2
    return out if out.size > 1 else float(out[0])


def bezier(p0, p1, p2, p3, n):
    t = np.linspace(0, 1, n)[:, None]
    p0, p1, p2, p3 = (np.asarray(x, float) for x in (p0, p1, p2, p3))
    return (1 - t) ** 3 * p0 + 3 * (1 - t) ** 2 * t * p1 + 3 * (1 - t) * t ** 2 * p2 + t ** 3 * p3


def bezier_tangent(p0, p1, p2, p3, t):
    p0, p1, p2, p3 = (np.asarray(x, float) for x in (p0, p1, p2, p3))
    return _n(3 * (1 - t) ** 2 * (p1 - p0) + 6 * (1 - t) * t * (p2 - p1) + 3 * t ** 2 * (p3 - p2))


def quat_from_x_to(d):
    """Quaternion [w,x,y,z] rotating +x onto direction d."""
    a = np.array([1.0, 0, 0])
    d = _n(d)
    c = float(a @ d)
    if c > 0.999999:
        return np.array([1.0, 0, 0, 0])
    ax = np.cross(a, d)
    s = np.linalg.norm(ax)
    ang = math.atan2(s, c)
    ax /= s
    return np.array([math.cos(ang / 2), *(ax * math.sin(ang / 2))])


# ---------------------------------------------------------------- mesh parts
class Part:
    def __init__(self, verts, tris, rgba, tag=None):
        self.v = np.asarray(verts, float).reshape(-1, 3)
        self.t = np.asarray(tris, np.int64).reshape(-1, 3)
        self.rgba = tuple(rgba)
        self.tag = tag

    def normals(self):
        v, t = self.v, self.t
        fn = np.cross(v[t[:, 1]] - v[t[:, 0]], v[t[:, 2]] - v[t[:, 0]])  # area-weighted
        n = np.zeros_like(v)
        for k in range(3):
            np.add.at(n, t[:, k], fn)
        ln = np.linalg.norm(n, axis=1, keepdims=True)
        ln[ln < 1e-12] = 1.0
        return n / ln


def sweep(path, radii, seg=24, ref=(1, 0, 0), theta=None, rfac=None, ax=1.0, ay=1.0,
          closed_path=False, cap0=False, cap1=False, rgba=MECH, tag=None):
    """Generalised lathe: a (possibly elliptical / partial) circle of radius radii[i] swept along path.

    theta=None -> full closed ring of `seg` columns; otherwise explicit column angles (open sector) with optional
    per-column radius factors rfac (used for panel lips). theta measured from the transported normal N (starts as
    `ref` projected) toward B = T x N. Rings with radius 0 collapse to a pole vertex.
    """
    path = np.asarray(path, float)
    radii = np.broadcast_to(np.asarray(radii, float), (len(path),)).astype(float)
    ax = np.broadcast_to(np.asarray(ax, float), (len(path),)).astype(float)
    ay = np.broadcast_to(np.asarray(ay, float), (len(path),)).astype(float)
    if theta is not None:  # open sectors can't collapse to a pole: drop zero-radius end rings
        keep = radii > 1e-7
        path, radii, ax, ay = path[keep], radii[keep], ax[keep], ay[keep]
    K = len(path)
    if closed_path:
        T = np.roll(path, -1, 0) - np.roll(path, 1, 0)
    else:
        T = np.gradient(path, axis=0)
    ln = np.linalg.norm(T, axis=1)
    good = np.where(ln > 1e-9)[0]
    if len(good) == 0:
        raise ValueError("degenerate sweep path")
    for i in np.where(ln <= 1e-9)[0]:  # repeated path points (flat lathe steps): borrow the nearest tangent
        T[i] = T[good[np.argmin(abs(good - i))]]
    T /= np.linalg.norm(T, axis=1, keepdims=True)
    ref = _n(ref)
    N = np.zeros_like(path)
    n0 = ref - (ref @ T[0]) * T[0]
    if np.linalg.norm(n0) < 1e-6:
        alt = np.array([0, 0, 1.0]) if abs(T[0][2]) < 0.9 else np.array([1.0, 0, 0])
        n0 = alt - (alt @ T[0]) * T[0]
    N[0] = _n(n0)
    for i in range(1, K):
        n = N[i - 1] - (N[i - 1] @ T[i]) * T[i]
        N[i] = _n(n)
    B = np.cross(T, N)
    closed_ring = theta is None
    th = np.linspace(0, 2 * np.pi, seg, endpoint=False) if closed_ring else np.asarray(theta, float)
    rf = np.ones(len(th)) if rfac is None else np.asarray(rfac, float)
    C = len(th)
    dirs = (np.cos(th)[None, :, None] * ax[:, None, None] * N[:, None, :]
            + np.sin(th)[None, :, None] * ay[:, None, None] * B[:, None, :])
    ring = path[:, None, :] + radii[:, None, None] * rf[None, :, None] * dirs
    verts, starts, pole = [], [], []
    for i in range(K):
        starts.append(len(verts))
        if closed_ring and radii[i] < 1e-7 and (i == 0 or i == K - 1):
            verts.append(path[i])
            pole.append(True)
        else:
            verts.extend(ring[i])
            pole.append(False)
    tris = []
    ncol = C if closed_ring else C - 1
    nring = K if closed_path else K - 1
    for i in range(nring):
        a, b = i, (i + 1) % K
        for j in range(ncol):
            j2 = (j + 1) % C
            if pole[a]:
                tris.append((starts[a], starts[b] + j, starts[b] + j2))
            elif pole[b]:
                tris.append((starts[a] + j, starts[b], starts[a] + j2))
            else:
                tris.append((starts[a] + j, starts[b] + j, starts[b] + j2))
                tris.append((starts[a] + j, starts[b] + j2, starts[a] + j2))
    verts = np.array(verts)
    tris = np.array(tris, np.int64)
    # orient outward: compare face normals with the radial direction from the path
    fn = np.cross(verts[tris[:, 1]] - verts[tris[:, 0]], verts[tris[:, 2]] - verts[tris[:, 0]])
    cen = verts[tris].mean(axis=1)
    near = np.argmin(((cen[:, None, :] - path[None, :, :]) ** 2).sum(-1), axis=1)
    if float(np.einsum("ij,ij->", fn, cen - path[near])) < 0:
        tris = tris[:, ::-1].copy()
    vlist = [verts]
    tlist = [tris]
    nv = len(verts)
    for end, cap in ((0, cap0), (K - 1, cap1)):
        if not cap or pole[end] or not closed_ring:
            continue
        rim = ring[end]
        cv = np.vstack([path[end][None], rim])
        ct = np.array([(nv, nv + 1 + j, nv + 1 + (j + 1) % C) for j in range(C)], np.int64)
        out = -T[0] if end == 0 else T[-1]
        f = np.cross(cv[ct[0, 1] - nv] - cv[0], cv[ct[0, 2] - nv] - cv[0])
        if f @ out < 0:
            ct = ct[:, ::-1].copy()
        vlist.append(cv)
        tlist.append(ct)
        nv += len(cv)
    return Part(np.vstack(vlist), np.vstack(tlist), rgba, tag)


def lathe(origin, axis, prof, seg=24, ref=None, rgba=MECH, tag=None, **kw):
    """prof: list of (s, r) along unit `axis` from `origin`."""
    axis = _n(axis)
    if ref is None:
        ref = (1, 0, 0) if abs(axis[0]) < 0.9 else (0, 0, 1)
    prof = np.asarray(prof, float)
    path = np.asarray(origin, float)[None] + prof[:, 0:1] * axis[None]
    return sweep(path, prof[:, 1], seg=seg, ref=ref, rgba=rgba, tag=tag, **kw)


def cylinder(p0, p1, r, seg=20, rgba=MECH, caps=True):
    return sweep(np.array([p0, p1], float), r, seg=seg, cap0=caps, cap1=caps, rgba=rgba,
                 ref=(1, 0, 0) if abs(_n(np.subtract(p1, p0))[0]) < 0.9 else (0, 0, 1))


def torus(center, axis, R, r, seg=32, tseg=10, rgba=RING):
    axis = _n(axis)
    u = _n(np.cross(axis, [1, 0, 0] if abs(axis[0]) < 0.9 else [0, 0, 1]))
    v = np.cross(axis, u)
    a = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    path = np.asarray(center)[None] + R * (np.cos(a)[:, None] * u + np.sin(a)[:, None] * v)
    return sweep(path, r, seg=tseg, ref=axis, closed_path=True, rgba=rgba)


def box(center, axes, half, rgba=MECH):
    """Oriented box: axes = 3 unit vectors (rows), half = half extents. Flat-shaded (separate face verts)."""
    c = np.asarray(center, float)
    A = np.asarray(axes, float)
    h = np.asarray(half, float)
    verts, tris = [], []
    for k in range(3):
        for sgn in (-1, 1):
            n = A[k] * sgn
            u, w = A[(k + 1) % 3] * h[(k + 1) % 3], A[(k + 2) % 3] * h[(k + 2) % 3]
            base = c + n * h[k]
            i0 = len(verts)
            verts += [base - u - w, base + u - w, base + u + w, base - u + w]
            t = [(i0, i0 + 1, i0 + 2), (i0, i0 + 2, i0 + 3)]
            f = np.cross(verts[i0 + 1] - verts[i0], verts[i0 + 2] - verts[i0])
            if f @ n < 0:
                t = [tt[::-1] for tt in t]
            tris += t
    return Part(verts, tris, rgba)


def sector_cols(center_deg, half_deg, n, lip=0.0):
    """Column angles for a panel sector centred at center_deg (+- half_deg), with lip columns at both edges."""
    th = np.radians(np.linspace(center_deg - half_deg, center_deg + half_deg, n))
    if lip <= 0:
        return th, np.ones(n)
    th = np.concatenate([[th[0]], th, [th[-1]]])
    rf = np.concatenate([[lip], np.ones(n), [lip]])
    return th, rf


def ring_band(path_fn, t0, t1, rfun, seg, k=3, grow=0.008, ref=(1, 0, 0), rgba=RING, theta=None):
    ts = np.linspace(t0, t1, k)
    pts = np.array([path_fn(t) for t in ts])
    rr = np.array([rfun(t) for t in ts]) + grow
    return sweep(pts, rr, seg=seg, ref=ref, rgba=rgba, theta=theta)


# ---------------------------------------------------------------- v2 helpers
def bez_at(ctrl, t):
    """Cubic Bezier through 4 control points at parameter(s) t."""
    c = np.asarray(ctrl, float)
    t = np.atleast_1d(np.asarray(t, float))[:, None]
    return (1 - t) ** 3 * c[0] + 3 * (1 - t) ** 2 * t * c[1] + 3 * (1 - t) * t ** 2 * c[2] + t ** 3 * c[3]


def bez_tan(ctrl, t):
    return bezier_tangent(*np.asarray(ctrl, float), float(t))


def superellipse(seg, p=2.6):
    """Column angles + radius factors of a rounded-rectangle section (|c|^p + |s|^p = 1)."""
    th = np.linspace(0, 2 * np.pi, seg, endpoint=False)
    c, s = np.abs(np.cos(th)), np.abs(np.sin(th))
    return th, (c ** p + s ** p) ** (-1.0 / p)


def _xz_front(T):
    """Unit normal in the x-z plane perpendicular to tangent T, pointing to +x (front)."""
    n = np.array([-T[2], 0.0, T[0]])
    if n[0] < 0:
        n = -n
    return n / max(np.linalg.norm(n), 1e-9)


# ---------------------------------------------------------------- mount (fixed ceiling hub)
def build_mount():
    parts = []
    # top face 2 cm below the ceiling underside: flush (z=0) it z-fights with the ceiling block's bottom face
    prof = [(-0.02, 0.0), (-0.02, HUB_R), (-0.06, HUB_R + 0.02), (-0.15, HUB_R), (-0.19, HUB_R - 0.14),
            (-0.22, 1.00), (-0.26, 0.92), (-0.48, 0.92), (-0.52, 0.86), (-HUB_DEPTH, 0.80), (-HUB_DEPTH, 0.0)]
    path = np.array([[0, 0, s] for s, _ in prof])
    parts.append(sweep(path, [r for _, r in prof], seg=36, ref=(1, 0, 0), rgba=MECH_DARK))
    parts.append(torus((0, 0, -0.19), (0, 0, 1), HUB_R - 0.20, 0.085, seg=36, tseg=6, rgba=RING))
    parts.append(torus((0, 0, -0.40), (0, 0, 1), 0.93, 0.05, seg=32, tseg=5, rgba=RING))
    for k in range(12):                                   # brackets under the rim
        a = 2 * math.pi * k / 12 + 0.13
        u = np.array([math.cos(a), math.sin(a), 0.0])
        parts.append(box(u * (HUB_R - 0.42) + [0, 0, -0.24], [u, [-u[1], u[0], 0], [0, 0, 1]],
                         [0.12, 0.05, 0.05], rgba=MECH))
    # drooping service cables between rim points (on the fixed hub, so they never move)
    for k, (a0, a1, sag, r) in enumerate([(10, 70, 0.75, 0.06), (95, 150, 1.05, 0.07), (170, 215, 0.55, 0.05),
                                          (235, 300, 0.95, 0.065), (310, 352, 0.65, 0.05)]):
        A0, A1 = math.radians(a0), math.radians(a1)
        p0 = np.array([1.12 * math.cos(A0), 1.12 * math.sin(A0), -0.18])
        p3 = np.array([1.12 * math.cos(A1), 1.12 * math.sin(A1), -0.18])
        out = _n([(p0 + p3)[0], (p0 + p3)[1], 0]) * 0.25
        parts.append(sweep(bezier(p0, p0 + [0, 0, -sag] + out, p3 + [0, 0, -sag] + out, p3, 9), r, seg=6,
                           ref=(0, 0, 1), rgba=RING if k % 2 else MECH_DARK))
    return parts


# ---------------------------------------------------------------- arm1: the cable bundle (yaw + swing at the hub)
CABLES = [  # start radius, start angle (deg), bulge, twist (deg), cable radius, colour
    (0.68, 0, 0.08, 30, 0.150, RING), (0.59, 30, 0.04, 26, 0.119, MECH_DARK), (0.71, 58, 0.10, 34, 0.156, RING),
    (0.56, 88, 0.05, 22, 0.112, RING), (0.68, 115, 0.08, 32, 0.144, MECH_DARK), (0.61, 145, 0.06, 28, 0.125, RING),
    (0.73, 172, 0.10, 36, 0.156, RING), (0.58, 200, 0.04, 24, 0.112, MECH_DARK), (0.70, 228, 0.09, 33, 0.150, RING),
    (0.61, 256, 0.06, 27, 0.125, RING), (0.71, 284, 0.10, 35, 0.156, MECH_DARK), (0.58, 312, 0.05, 23, 0.119, RING),
    (0.68, 338, 0.08, 31, 0.144, RING), (0.34, 20, 0.02, 18, 0.112, RING), (0.32, 140, 0.02, 20, 0.112, MECH_DARK),
    (0.36, 260, 0.02, 16, 0.112, RING),
    (0.50, 75, 0.04, 20, 0.130, MECH_DARK), (0.52, 190, 0.04, 24, 0.125, RING), (0.48, 300, 0.04, 21, 0.130, RING),
    (0.62, 45, 0.06, 28, 0.110, RING), (0.60, 160, 0.06, 30, 0.110, MECH), (0.62, 280, 0.06, 26, 0.110, RING),
]


def build_arm1():
    parts = []
    L = ARM1_LEN
    parts.append(cylinder((0, 0, 0.25), (0, 0, -L - 0.05), 0.24, seg=14, rgba=MECH_DARK, caps=False))
    for r0, a, bulge, tw, rc, col in CABLES:
        A, Tw = math.radians(a), math.radians(tw)

        def at(r, ang, z):
            return np.array([r * math.cos(ang), r * math.sin(ang), z])
        p0 = at(r0 * 0.85, A, 0.20)
        p1 = at(r0 * 0.85 + bulge, A + 0.25 * Tw, -0.50)
        p2 = at(0.40 * min(1.0, r0 / 0.7), A + 0.75 * Tw, -1.20)
        p3 = at(0.27 * min(1.0, r0 / 0.7), A + Tw, -L - 0.08)
        parts.append(sweep(bezier(p0, p1, p2, p3, 10), rc, seg=5, ref=(0, 0, 1), rgba=col))
    for z, R in ((-1.30, 0.50), (-1.62, 0.42)):            # clamp bands holding the bundle together
        parts.append(torus((0, 0, z), (0, 0, 1), R, 0.05, seg=24, tseg=5, rgba=MECH))
    for k in range(4):
        a = math.pi / 2 * k + 0.4
        u = np.array([math.cos(a), math.sin(a), 0.0])
        parts.append(box(u * 0.52 + [0, 0, -1.30], [u, [-u[1], u[0], 0], [0, 0, 1]], [0.06, 0.09, 0.11],
                         rgba=MECH_DARK))
    return parts


# ---------------------------------------------------------------- arm2: shoulder ring cluster
def build_arm2():
    parts = []
    L = ARM2_LEN
    parts.append(lathe((0, 0, 0), (0, 0, -1), [(-0.12, 0.0), (-0.12, 0.40), (-0.02, 0.47), (0.25, 0.50),
                                               (0.42, 0.45), (0.58, 0.38), (L + 0.06, 0.34), (L + 0.06, 0.0)],
                       seg=28, rgba=MECH_DARK))
    parts.append(torus((0, 0, -0.06), (0, 0, 1), 0.47, 0.06, seg=28, tseg=6, rgba=RING))
    # big hoops around the joint (the dark "ring cluster")
    parts.append(torus((0.0, 0, -0.32), (1, 0, 0), 0.80, 0.10, seg=32, tseg=7, rgba=RING))
    parts.append(torus((-0.05, 0, -0.30), _n([0.25, 1, 0]), 0.66, 0.08, seg=28, tseg=6, rgba=MECH))
    for sg in (1, -1):                                    # clamps where the hoops meet the body
        parts.append(box((0, sg * 0.50, -0.32), [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.16, 0.09, 0.16], rgba=MECH))
        parts.append(box((0, sg * 0.80, -0.32), [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.13, 0.06, 0.13],
                         rgba=MECH_DARK))
    parts.append(box((0.42, 0, -0.30), [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.08, 0.20, 0.14], rgba=MECH))
    return parts


# ---------------------------------------------------------------- torso: inverted hip shell
TORSO_PANELS = ((0.012, 0.270), (0.286, 0.560), (0.576, 0.955))   # shell bands (t ranges), gaps = seams
FRONT_GAP, BACK_GAP = 17.0, 13.0                                    # deg: cable channel / spine slot


def torso_frame(t):
    p = bez_at(TORSO_CTRL, t)[0]
    T = bez_tan(TORSO_CTRL, t)
    return p, T, _xz_front(T)


def build_torso():
    parts = []
    tt = np.concatenate([[0.0, 0.012], np.linspace(0.03, 0.97, 24), [0.99, 1.0]])
    path = bez_at(TORSO_CTRL, tt)
    w, d = torso_profile(tt)
    # dark inner core (shows through the panel seams, the front channel and the spine slot)
    parts.append(sweep(path, 1.0, seg=18, ref=(1, 0, 0), ax=d * 0.90, ay=w * 0.90, rgba=MECH_DARK))
    # white shell panels: left/right halves, 3 bands each, folded lips into the core
    for t0, t1 in TORSO_PANELS:
        ts = np.linspace(t0, t1, max(4, int(round((t1 - t0) * 34))))
        pth = bez_at(TORSO_CTRL, ts)
        ww, dd = torso_profile(ts)
        for lo, hi in ((FRONT_GAP, 180 - BACK_GAP), (-180 + BACK_GAP, -FRONT_GAP)):
            n = 12
            th = np.radians(np.concatenate([[lo], np.linspace(lo, hi, n), [hi]]))
            rf = np.concatenate([[0.91], np.ones(n), [0.91]])
            parts.append(sweep(pth, 1.0, ref=(1, 0, 0), theta=th, rfac=rf, ax=dd, ay=ww, rgba=SHELL))
    # front cable channel
    for k, (yo, rc, col) in enumerate(((-0.11, 0.055, RING), (0.0, 0.065, MECH), (0.11, 0.055, RING))):
        ts = np.linspace(0.05, 0.96, 16)
        pts = []
        for x in ts:
            p, T, N = torso_frame(x)
            _, dd = torso_profile(x)
            pts.append(p + N * (float(dd) * 0.92) + np.array([0, yo, 0]))
        parts.append(sweep(np.array(pts), rc, seg=6, ref=(1, 0, 0), rgba=col))
    # ribbed spine down the back
    ts = np.linspace(0.04, 0.97, 18)
    back = []
    for x in ts:
        p, T, N = torso_frame(x)
        _, dd = torso_profile(x)
        back.append(p - N * (float(dd) * 0.97))
    parts.append(sweep(np.array(back), 0.075, seg=8, ref=(1, 0, 0), cap0=True, cap1=True, rgba=MECH_DARK))
    for i, x in enumerate(np.linspace(0.07, 0.93, 13)):
        p, T, N = torso_frame(x)
        _, dd = torso_profile(x)
        c = p - N * (float(dd) * 0.98 + 0.03)
        half = [0.045, 0.075, 0.17 if i % 2 == 0 else 0.12]
        parts.append(box(c, [T, N, [0, 1, 0]], half, rgba=MECH if i % 2 == 0 else RING))
    # top collar (black joint where the shoulder cluster attaches) + bottom joint housing (neck socket)
    parts.append(lathe((0, 0, 0.12), (0, 0, -1), [(0.0, 0.0), (0.0, 0.42), (0.08, 0.50), (0.20, 0.50),
                                                  (0.30, 0.42), (0.34, 0.0)], seg=28, rgba=RING))
    pb, Tb, _ = torso_frame(1.0)
    parts.append(lathe(pb - Tb * 0.30, Tb, [(0.0, 0.0), (0.0, 0.25), (0.10, 0.31), (0.26, 0.31), (0.36, 0.24),
                                           (0.40, 0.0)], seg=24, rgba=MECH_DARK))
    parts.append(torus(pb - Tb * 0.12, Tb, 0.31, 0.045, seg=24, tseg=5, rgba=RING))
    return parts


# ---------------------------------------------------------------- neck: dark articulated U
def build_neck():
    parts = []
    path = bez_at(NECK_CTRL, np.linspace(0, 1, 20))
    parts.append(sweep(path, 0.22, seg=14, ref=(0, 1, 0), rgba=MECH_DARK))
    for i, t in enumerate(np.linspace(0.08, 0.92, 6)):        # segment collars (articulated look)
        p = bez_at(NECK_CTRL, t)[0]
        T = bez_tan(NECK_CTRL, t)
        parts.append(lathe(p - T * 0.09, T, [(0.0, 0.24), (0.03, 0.29), (0.15, 0.29), (0.18, 0.24)],
                           seg=16, rgba=RING if i % 2 == 0 else MECH))
    for sg in (1, -1):                                         # side pistons
        pts = bez_at(NECK_CTRL, np.linspace(0.05, 0.95, 10)) + np.array([0, sg * 0.25, 0])
        parts.append(sweep(pts, 0.04, seg=6, ref=(0, 0, 1), rgba=MECH))
    # ball joints at both ends
    parts.append(lathe((0, 0, 0.18), (0, 0, -1), [(0, 0), (0.04, 0.20), (0.18, 0.30), (0.32, 0.20), (0.36, 0)],
                       seg=16, rgba=MECH))
    return parts


# ---------------------------------------------------------------- head: big horizontal lamp helmet
HEAD_SEAM = 5.0      # deg: half width of the dark band between the upper and lower shells (at the sides)
HEAD_P = 2.7         # superellipse exponent of the helmet section


def head_frame(t):
    p = bez_at(HEAD_CTRL, t)[0]
    T = bez_tan(HEAD_CTRL, t)
    return p, T


def build_head():
    parts = []
    tt = np.concatenate([[0.0, 0.006, 0.02, 0.04], np.linspace(0.065, 1.0, 22)])
    path = bez_at(HEAD_CTRL, tt)
    f = head_profile(tt)
    seg = 28
    th, sq = superellipse(seg, HEAD_P)
    # dark core (inside the shells; visible in the side band and the seams)
    thc_, sqc = superellipse(20, HEAD_P)
    parts.append(sweep(path, f, seg=20, ref=(0, 0, 1), rfac=sqc * 0.93, ax=HEAD_H, ay=HEAD_W, rgba=MECH_DARK))
    # upper and lower white shells (front band + back band split by a thin seam at t ~ 0.36)
    for t0, t1 in ((0.012, 0.345), (0.36, 0.975)):
        ts = np.concatenate([[t0], np.linspace(max(t0, 0.03), t1, max(5, int((t1 - t0) * 26)))])
        ts = np.unique(ts)
        pth = bez_at(HEAD_CTRL, ts)
        ff = head_profile(ts)
        for lo, hi in ((-90 + HEAD_SEAM, 90 - HEAD_SEAM), (90 + HEAD_SEAM, 270 - HEAD_SEAM)):
            n = 15
            a = np.radians(np.linspace(lo, hi, n))
            c, s = np.abs(np.cos(a)), np.abs(np.sin(a))
            r = (c ** HEAD_P + s ** HEAD_P) ** (-1.0 / HEAD_P)
            thc = np.concatenate([[a[0]], a, [a[-1]]])
            rf = np.concatenate([[r[0] * 0.93], r, [r[-1] * 0.93]])
            parts.append(sweep(pth, ff, ref=(0, 0, 1), theta=thc, rfac=rf, ax=HEAD_H, ay=HEAD_W, rgba=SHELL))
    # front: dark visor rim + recessed face dish + eye ring + eye
    fc, fa = head_frame(1.0)
    f1 = float(head_profile(1.0))
    _, sqr = superellipse(seg, HEAD_P)
    rim_scale = f1 * 1.0
    # rim: closed superellipse loop at the face edge
    N0 = _n(np.array([0, 0, 1.0]) - (np.array([0, 0, 1.0]) @ fa) * fa)
    B0 = np.cross(fa, N0)
    loop = np.array([fc - fa * 0.01 + rim_scale * sqr[k] * (math.cos(th[k]) * HEAD_H * N0 + math.sin(th[k]) * HEAD_W * B0)
                     for k in range(seg)])
    parts.append(sweep(loop, 0.04, seg=6, ref=fa, closed_path=True, rgba=RING))
    dish = np.array([fc - fa * 0.0, fc - fa * 0.05, fc - fa * 0.10, fc - fa * 0.11])
    parts.append(sweep(dish, np.array([f1 * 0.97, f1 * 0.85, f1 * 0.50, 0.0]), seg=seg, ref=(0, 0, 1),
                       rfac=sqr, ax=HEAD_H, ay=HEAD_W, rgba=RING))
    ec = fc - fa * 0.08
    parts.append(lathe(ec, fa, [(0.0, 0.0), (0.0, 0.215), (0.05, 0.225), (0.10, 0.20), (0.12, 0.16),
                                (0.12, 0.0)], seg=28, ref=(0, 0, 1), rgba=MECH_DARK))
    parts.append(torus(ec + fa * 0.11, fa, 0.17, 0.04, seg=24, tseg=5, rgba=RING))
    parts.append(lathe(ec + fa * 0.085, fa, [(0.0, 0.14), (0.035, 0.134), (0.06, 0.106), (0.077, 0.055),
                                             (0.082, 0.0)], seg=24, ref=(0, 0, 1), rgba=EYE, tag="eye"))
    # side discs over the seam band (hinge "ears")
    for sg in (1, -1):
        p, _ = head_frame(0.30)
        wy = HEAD_W * float(head_profile(0.30))
        parts.append(lathe(p + [0, sg * (wy - 0.06), 0], (0, sg, 0),
                           [(0, 0), (0, 0.19), (0.07, 0.19), (0.10, 0.14), (0.11, 0.0)], seg=20, rgba=MECH))
        parts.append(torus(p + [0, sg * (wy + 0.03), 0], (0, sg, 0), 0.14, 0.025, seg=18, tseg=5, rgba=RING))
    # joint housing at the head origin (neck attaches here, upper back of the helmet)
    parts.append(lathe((0, -0.24, 0), (0, 1, 0), [(0, 0), (0, 0.15), (0.04, 0.17), (0.44, 0.17), (0.48, 0.15),
                                                  (0.48, 0)], seg=16, rgba=MECH_DARK))
    parts.append(box((0.06, 0, -0.13), [[1, 0, 0], [0, 1, 0], [0, 0, 1]], [0.12, 0.13, 0.13], rgba=MECH_DARK))
    return parts


def head_face():
    """(centre, axis) of the face plate = front end of the head spine."""
    return head_frame(1.0)


def eye_site():
    """Eye position (head-local) and look axis."""
    fc, fa = head_face()
    return fc + fa * 0.09, fa


def head_center():
    """Head-local centre of the helmet (voice position / focus target)."""
    return bez_at(HEAD_CTRL, 0.5)[0]


def mass_geoms(name):
    """Invisible MuJoCo mass geoms approximating each body: list of dicts (type, size, pos, quat?, fromto?, density)."""
    if name == "mount":
        return [dict(type="cylinder", size=[1.2, 0.2, 0], pos=[0, 0, -0.25], density=500)]
    if name == "arm1":
        return [dict(type="capsule", fromto=[0, 0, -0.15, 0, 0, -ARM1_LEN + 0.1], size=[0.38, 0, 0], density=550)]
    if name == "arm2":
        return [dict(type="ellipsoid", size=[0.55, 0.75, 0.38], pos=[0, 0, -0.35], density=300)]
    if name == "torso":
        top, bot = TORSO_CTRL[0], TORSO_CTRL[3]
        c = bez_at(TORSO_CTRL, 0.42)[0]
        ax = _n(bot - top)
        return [dict(type="ellipsoid", size=[0.55, 0.78, 0.5 * float(np.linalg.norm(bot - top)) * 0.95],
                     pos=list(c), zaxis=list(ax), density=230)]
    if name == "neck":
        mid = bez_at(NECK_CTRL, 0.5)[0]
        return [dict(type="capsule", fromto=[0, 0, 0, *mid], size=[0.17, 0, 0], density=900),
                dict(type="capsule", fromto=[*mid, *NECK_END], size=[0.17, 0, 0], density=900)]
    if name == "head":
        c = head_center()
        return [dict(type="ellipsoid", size=[0.80, HEAD_W, HEAD_H], pos=list(c), density=320)]
    return []


# ---------------------------------------------------------------- assembly
BODY_ORDER = ["mount", "arm1", "arm2", "torso", "neck", "head"]
PARENT = {"arm1": "mount", "arm2": "arm1", "torso": "arm2", "neck": "torso", "head": "neck"}


def body_pos():
    """Body position relative to parent (design pose; all body frames axis-aligned)."""
    return {
        "mount": np.zeros(3),
        "arm1": np.array([0, 0, -HUB_DEPTH]),
        "arm2": np.array([0, 0, -ARM1_LEN]),
        "torso": np.array([0, 0, -ARM2_LEN]),
        "neck": TORSO_CTRL[3].copy(),
        "head": NECK_END.copy(),
    }


def body_world_pos():
    rel = body_pos()
    out = {}
    for b in BODY_ORDER:
        out[b] = rel[b] + (out[PARENT[b]] if b in PARENT else 0)
    return out


BUILDERS = {"mount": build_mount, "arm1": build_arm1, "arm2": build_arm2, "torso": build_torso,
            "neck": build_neck, "head": build_head}


def build_parts():
    return {b: BUILDERS[b]() for b in BODY_ORDER}


def model_json(parts=None, ndig=4):
    """Contract /model.json: merge parts per body by colour (+ eye separate), per-vertex normals."""
    parts = parts or build_parts()
    bodies = []
    total = 0
    for b in BODY_ORDER:
        groups = {}
        for p in parts[b]:
            groups.setdefault((p.rgba, p.tag), []).append(p)
        meshes = []
        for (rgba, tag), ps in groups.items():
            vs, ts, ns, off = [], [], [], 0
            for p in ps:
                vs.append(p.v)
                ns.append(p.normals())
                ts.append(p.t + off)
                off += len(p.v)
            v, t, nn = np.vstack(vs), np.vstack(ts), np.vstack(ns)
            total += len(t)
            m = {"name": f"{b}_{tag or _cname(rgba)}", "rgba": [round(c, 4) for c in rgba],
                 "verts": [round(float(x), ndig) for x in v.ravel()], "tris": [int(x) for x in t.ravel()],
                 "normals": [round(float(x), 3) for x in nn.ravel()]}
            if tag:
                m["tag"] = tag
            meshes.append(m)
        bodies.append({"name": b, "meshes": meshes})
    return {"version": 1, "bodies": bodies, "triangles": total}


def _cname(rgba):
    return {SHELL: "shell", MECH: "mech", MECH_DARK: "mechdark", RING: "ring", EYE: "eye"}.get(tuple(rgba), "x")


if __name__ == "__main__":
    import json
    import time
    t0 = time.time()
    m = model_json()
    s = json.dumps(m, separators=(",", ":"))
    print(f"{len(m['bodies'])} bodies, {m['triangles']} tris, {len(s)} bytes, {time.time() - t0:.2f}s")
    for b in m["bodies"]:
        print(" ", b["name"], [(me["name"], len(me["tris"]) // 3) for me in b["meshes"]])
    wp = body_world_pos()
    for k, v in wp.items():
        print(f"  {k:6s} world {np.round(v, 3)}")
    lo = min(min(me["verts"][2::3]) + wp[b["name"]][2] for b in m["bodies"] for me in b["meshes"])
    print("lowest point z", round(lo, 3))
    ep, ea = eye_site()
    print("eye site head-local", np.round(ep, 3), "axis", np.round(ea, 3), "world", np.round(ep + wp["head"], 3))
