#!/usr/bin/env python3
"""GLaDOS-style hanging robot: MuJoCo physics (MjSpec chain from the ceiling) + scripted behaviour layer.

No HTTP here (see server.py). One `GladosSim.step()` = one 60 Hz control tick = SUBSTEPS physics steps.
Sim frame: z-up metres, origin = ceiling mount point, she hangs in -z, default facing +x.
"""

import math
import os
import random
import threading

os.environ.setdefault("OMP_NUM_THREADS", "1")

import mujoco  # noqa: E402
import numpy as np  # noqa: E402

import glados_assets as ga  # noqa: E402
import glados_model as gm  # noqa: E402

CONTROL_HZ = 60
CONTROL_DT = 1.0 / CONTROL_HZ
SUBSTEPS = 8
PHYS_DT = CONTROL_DT / SUBSTEPS
TRACK_RANGE = 12.0      # auto-track when the player eye is this close to her head (m)
PLAYER_STALE = 5.0      # /player older than this (s) = unknown

D = math.radians
# name, body, axis, range (deg) or None, group
JOINTS = [
    ("yaw", "arm1", (0, 0, 1), None, "yaw"),
    ("arm1_x", "arm1", (1, 0, 0), (-18, 18), "arm"),
    ("arm1_y", "arm1", (0, 1, 0), (-18, 18), "arm"),
    ("arm2_x", "arm2", (1, 0, 0), (-22, 22), "arm"),
    ("arm2_y", "arm2", (0, 1, 0), (-22, 22), "arm"),
    ("torso_x", "torso", (1, 0, 0), (-20, 20), "torso"),
    ("torso_y", "torso", (0, 1, 0), (-25, 25), "torso"),
    ("torso_z", "torso", (0, 0, 1), (-35, 35), "torso"),
    ("neck_y", "neck", (0, 1, 0), (-55, 75), "neck"),
    ("neck_z", "neck", (0, 0, 1), (-60, 60), "neck"),
    ("head_y", "head", (0, 1, 0), (-65, 70), "head"),
    ("head_z", "head", (0, 0, 1), (-55, 55), "head"),
    ("head_x", "head", (1, 0, 0), (-30, 30), "head"),
]
JN = [j[0] for j in JOINTS]
JI = {n: i for i, n in enumerate(JN)}
# PD design per group: natural frequency (Hz), damping ratio, gravity-compensation fraction, target rate limit (rad/s)
GROUP = {
    "yaw": (0.40, 0.95, 1.0, 0.8),
    "arm": (0.45, 0.45, 0.5, 0.8),
    "torso": (0.60, 0.40, 0.7, 1.2),
    "neck": (1.30, 0.80, 1.0, 2.5),
    "head": (1.80, 0.80, 1.0, 3.5),
}
IK_W = {"neck_y": 0.40, "neck_z": 0.40, "head_y": 0.60, "head_z": 0.60}
CMDS = ("idle", "track", "look", "angry", "scan", "nod", "shake", "bow", "sleep", "wake", "talk", "reset", "dance",
        "spooky")
CMD_ALIASES = {"spookydance": "spooky", "spooky_dance": "spooky", "creepy": "spooky"}
# spooky dance phases (s from the start): power-down sway | glitch dance | swoop | hold (stare) | pull back
SPOOKY_T = {"glitch": 5.5, "swoop": 12.0, "hold": 13.9, "back": 15.8, "end": 18.6}
SPOOKY_STANDOFF = 1.5      # eye -> player-eye distance at the climax (m = blocks at scale 1)
SPOOKY_MIN_DIST = 1.2      # never closer than this (camera clipping)
SPOOKY_ABOVE_FEET = 0.5    # no part of the head below the player's feet + this
PLAYER_EYE_H = 1.62        # player eye height above the feet (m)
ACTION_LEN = {"look": 2.2, "angry": 4.0, "scan": 5.0, "nod": 1.8, "shake": 1.8, "bow": 3.0, "wake": 2.2, "dance": 8.0,
              "spooky": SPOOKY_T["end"]}
SPOOKY_IK = ("arm1_x", "arm1_y", "arm2_x", "arm2_y", "torso_x", "torso_y", "torso_z", "neck_y", "neck_z",
             "head_y", "head_z")
DANCE_BPM = 112.0
EYE_NORMAL = np.array([1.0, 0.824, 0.290, 1.0])
EYE_ANGRY = np.array([1.0, 0.27, 0.06, 1.0])
EYE_SLEEP = np.array([0.55, 0.42, 0.14, 0.12])
EYE_SPOOKY = np.array([1.0, 0.20, 0.03, 1.0])   # climax flare: hot red-orange, full glow


def _quat_z_to(d):
    """quat [w,x,y,z] rotating +z onto d."""
    a = np.array([0, 0, 1.0])
    d = np.asarray(d, float) / np.linalg.norm(d)
    c = float(a @ d)
    if c > 0.999999:
        return [1.0, 0, 0, 0]
    ax = np.cross(a, d)
    s = np.linalg.norm(ax)
    ang = math.atan2(s, c)
    ax /= s
    return [math.cos(ang / 2), *(ax * math.sin(ang / 2))]


_PROC_MASS = {}


def procedural_masses():
    """Per-body masses (kg) of the procedural model: assets reuse them so the PD tuning carries over."""
    if not _PROC_MASS:
        m = build_model(None)
        for name in gm.BODY_ORDER:
            _PROC_MASS[name] = float(m.body_mass[mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)])
    return dict(_PROC_MASS)


def neck_scale(asset):
    """Neck articulation scale for an asset: the behaviour layer was tuned on the procedural neck (~1.1 m from the
    neck joint to the head joint); a longer neck lever (e.g. an imported model with a ~2.2 m neck) swings the head much further for the
    same angle and over-stretches the skinned neck plates, so its neck limits/gesture offsets are scaled down."""
    if asset is None or asset.name == "procedural":
        return 1.0
    L = float(np.linalg.norm(asset.bodies["head"]["joint_pos"]))
    L0 = float(np.linalg.norm(gm.body_pos()["head"]))
    return float(min(1.0, max(0.4, L0 / max(L, 1e-6))))


def build_model(asset=None):
    """MjSpec chain. asset=None or the procedural asset -> glados_model geometry (v2 behaviour, bit-identical);
    otherwise a glados_assets.Asset: bodies/joint anchors from asset.bodies, mass geoms fitted to its meshes."""
    if asset is not None and asset.name == "procedural":
        asset = None
    ns = neck_scale(asset)
    spec = mujoco.MjSpec()
    spec.modelname = "glados"
    spec.compiler.degree = False  # all angles below are radians
    spec.option.timestep = PHYS_DT
    spec.option.integrator = mujoco.mjtIntegrator.mjINT_IMPLICITFAST
    masses = procedural_masses() if asset is not None else None
    bodies = {}
    G = mujoco.mjtGeom
    for name in gm.BODY_ORDER:
        parent = spec.worldbody if name == "mount" else bodies[gm.PARENT[name]]
        if asset is None:
            b = parent.add_body(name=name, pos=list(gm.body_pos()[name]))
        else:
            p, q = asset.rest(name)
            if name == "mount":
                pp, pq = np.zeros(3), np.array([1.0, 0, 0, 0])
            else:
                pp, pq = asset.rest(gm.PARENT[name])
            Rp = ga.quat_mat(pq)
            rel_p = Rp.T @ (p - pp)
            rel_q = ga.mat_quat(Rp.T @ ga.quat_mat(q))
            b = parent.add_body(name=name, pos=[float(x) for x in rel_p], quat=rel_q)
        bodies[name] = b
        for jn, jb, axis, rng, grp in JOINTS:
            if jb != name:
                continue
            kw = dict(name=jn, type=mujoco.mjtJoint.mjJNT_HINGE, axis=list(axis), armature=2.0, damping=1.0)
            if rng is not None:
                k = ns if grp == "neck" else 1.0
                kw.update(range=[D(rng[0]) * k, D(rng[1]) * k], limited=mujoco.mjtLimited.mjLIMITED_TRUE)
            else:
                kw.update(limited=mujoco.mjtLimited.mjLIMITED_FALSE)
            b.add_joint(**kw)
        # invisible, non-colliding mass geoms approximating each part (density in kg/m^3)
        mg = dict(contype=0, conaffinity=0, group=3)
        gtypes = {"cylinder": G.mjGEOM_CYLINDER, "capsule": G.mjGEOM_CAPSULE, "ellipsoid": G.mjGEOM_ELLIPSOID}
        geoms = gm.mass_geoms(name) if asset is None else ga.mass_geoms(asset, name, masses[name])
        for g in geoms:                        # geometry-matched mass approximations
            kw = dict(type=gtypes[g["type"]], size=list(g["size"]), density=g["density"], **mg)
            if "fromto" in g:
                kw["fromto"] = list(g["fromto"])
            else:
                kw["pos"] = list(g["pos"])
            if "zaxis" in g:
                kw["quat"] = _quat_z_to(g["zaxis"])
            if "quat" in g:
                kw["quat"] = list(g["quat"])
            b.add_geom(**kw)
        eye_body = "head" if asset is None else asset.data["eye"]["body"]
        if name == eye_body:
            if asset is None:
                ep, ea = gm.eye_site()
            else:
                ep, ea = np.array(asset.data["eye"]["pos"], float), np.array(asset.data["eye"]["axis"], float)
            q = gm.quat_from_x_to(ea / np.linalg.norm(ea))
            b.add_site(name="eye", pos=[float(x) for x in ep], quat=[float(x) for x in q], size=[0.05, 0, 0])
    return spec.compile()


_HEAD_BOX = {}


def _head_box(asset):
    """8 corners of the head's body-local bounding box (rigid head meshes + verts skinned mostly to the head)."""
    key = "procedural" if asset is None or asset.name == "procedural" else asset.name
    if key not in _HEAD_BOX:
        try:
            a = asset if key != "procedural" else ga.Asset("procedural", ga.procedural_asset(), b"")
            P = a.body_points("head")
        except Exception:  # noqa: BLE001 - never fail a sim build over the clamp helper
            P = np.zeros((0, 3))
        if len(P) == 0:
            P = np.array([[-0.3, -0.5, -0.8], [2.4, 0.5, 0.4]])
        lo, hi = P.min(0), P.max(0)
        _HEAD_BOX[key] = np.array([[x, y, z] for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])])
    return _HEAD_BOX[key].copy()


def _wrap(a):
    return (a + math.pi) % (2 * math.pi) - math.pi


def _smooth01(x):
    x = min(max(x, 0.0), 1.0)
    return x * x * (3 - 2 * x)


def _bump(t, a, b):
    """0 outside [a,b], smooth sin^2 bump inside."""
    if t <= a or t >= b:
        return 0.0
    return math.sin(math.pi * (t - a) / (b - a)) ** 2


class GladosSim:
    def __init__(self, log=print, seed=None, asset=None):
        self.log = log
        self.lock = threading.RLock()
        self.asset = asset
        self.asset_name = asset.name if asset is not None else "procedural"
        self.model = build_model(asset)
        self.data = mujoco.MjData(self.model)
        self.kin = mujoco.MjData(self.model)  # kinematic scratch for the look-at IK
        m = self.model
        self.jid = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in JN])
        self.qadr = m.jnt_qposadr[self.jid]
        self.dadr = m.jnt_dofadr[self.jid]
        self.limited = np.array([j[3] is not None for j in JOINTS])
        self.neck_scale = neck_scale(asset)
        nk = np.array([self.neck_scale if j[4] == "neck" else 1.0 for j in JOINTS])
        self.nk = nk
        self.lo = np.array([D(j[3][0]) if j[3] else -1e9 for j in JOINTS]) * nk
        self.hi = np.array([D(j[3][1]) if j[3] else 1e9 for j in JOINTS]) * nk
        self.eye_sid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_SITE, "eye")
        self.head_bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "head")
        self.head_box = _head_box(asset)      # head-local AABB corners (8,3): floor clamp of the spooky swoop
        self.body_names = list(gm.BODY_ORDER)
        self.body_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, n) for n in self.body_names]
        # PD gains from the effective inertia (mass-matrix diagonal) at the design pose
        mujoco.mj_forward(m, self.data)
        Mfull = np.zeros((m.nv, m.nv))
        mujoco.mj_fullM(m, self.data, Mfull)
        diag = np.diag(Mfull)[self.dadr] + m.dof_armature[self.dadr]
        self.kp0 = np.zeros(len(JN))
        self.kd0 = np.zeros(len(JN))
        self.comp = np.zeros(len(JN))
        self.rate = np.zeros(len(JN))
        self.grp = [j[4] for j in JOINTS]
        for i, g in enumerate(self.grp):
            f, z, c, r = GROUP[g]
            w = 2 * math.pi * f
            self.kp0[i] = diag[i] * w * w
            self.kd0[i] = 2 * z * diag[i] * w
            self.comp[i] = c
            self.rate[i] = r
        self.tau_max = self.kp0 * 1.5 + 4.0 * np.abs(self._gravity_torque()) + 50
        self.rng = random.Random(seed)
        self.mass = float(sum(m.body_mass))
        self.nan_resets = 0
        self.resets = 0
        self.player = None
        self.player_t = -1e9
        self.scale_hint = None
        self._reset_state()

    # ------------------------------------------------------------------ state
    def _gravity_torque(self):
        d = mujoco.MjData(self.model)
        mujoco.mj_forward(self.model, d)
        return d.qfrc_bias[self.dadr].copy()

    def _reset_state(self):
        mujoco.mj_resetData(self.model, self.data)
        mujoco.mj_forward(self.model, self.data)
        self.t = 0.0 if not hasattr(self, "t") else self.t
        n = len(JN)
        self.q_cmd = np.zeros(n)
        self.ik_q = np.zeros(n)
        self.yaw_goal = 0.0
        self.gaze = self.eye_pos() + 8 * self.eye_dir()
        self.glance = self.gaze.copy()
        self.next_glance = self.t + 2.0
        self.action = None
        self.action_t0 = 0.0
        self.action_end = 0.0
        self.action_args = {}
        self.sleeping = False
        self.droop = 0.0
        self.idle_hold_until = -1e9
        self.track_forced = False
        self.talk_on = False
        self.talk_until = 0.0
        self.talk_in = 0.0
        self.talk_in_t = -1e9
        self.talk = 0.0
        self.eye = EYE_NORMAL.copy()
        self.jitter = np.zeros(n)
        self.next_jitter = 0.0
        self.gain_mult = np.ones(n)
        self.damp_mult = np.ones(n)
        self.tilt = 0.0           # curious head roll while tracking (rad)
        self.tilt_goal = 0.0
        self.tilt_anchor = None   # player position when the current tilt was chosen
        self._spk = None          # spooky-dance scratch state (reach solution, glitch targets); never carried over
        self.comp_boost = 0.0     # 0..1: extra gravity compensation (spooky swoop holds the pose against gravity)

    # ------------------------------------------------------------------ queries
    def q(self):
        return self.data.qpos[self.qadr].copy()

    def eye_pos(self):
        return self.data.site_xpos[self.eye_sid].copy()

    def eye_dir(self):
        return self.data.site_xmat[self.eye_sid].reshape(3, 3)[:, 0].copy()

    def head_pos(self):
        return self.data.xpos[self.head_bid].copy()

    def player_fresh(self):
        return self.player is not None and self.t - self.player_t < PLAYER_STALE

    def player_dist(self):
        return float(np.linalg.norm(self.player - self.head_pos())) if self.player_fresh() else float("inf")

    def base_mode(self):
        if self.sleeping:
            return "sleep"
        if self.player_fresh():
            d = self.player_dist()
            if self.track_forced and d < 30.0:
                return "track"
            if d < TRACK_RANGE and self.t >= self.idle_hold_until:
                return "track"
        return "idle"

    def public_mode(self):
        return self.action or self.base_mode()

    # ------------------------------------------------------------------ commands
    def set_player(self, x, y, z):
        self.player = np.array([x, y, z], float)
        self.player_t = self.t

    def command(self, cmd, **kw):
        """Returns (ok, error)."""
        cmd = (cmd or "").strip().lower()
        cmd = CMD_ALIASES.get(cmd, cmd)
        if cmd not in CMDS:
            return False, f"unknown cmd {cmd!r} (valid: {', '.join(CMDS)})"
        if cmd == "talk":
            if "level" in kw and kw["level"] is not None:
                self.talk_in = float(min(max(kw["level"], 0.0), 1.0))
                self.talk_in_t = self.t
                if not self.talk_on or self.talk_until < self.t + 0.6:
                    self.talk_on, self.talk_until = True, max(self.talk_until, self.t + 0.6)
            if "duration" in kw and kw["duration"] is not None:
                dur = float(min(max(kw["duration"], 0.0), 120.0))
                self.talk_on, self.talk_until = dur > 0, self.t + dur
            if "on" in kw and kw["on"] is not None:
                if kw["on"]:
                    self.talk_on, self.talk_until = True, self.t + 60.0
                else:
                    self.talk_on, self.talk_until = False, self.t
            if not any(k in kw and kw[k] is not None for k in ("level", "duration", "on")):
                self.talk_on, self.talk_until = True, self.t + 3.0
            return True, None
        if cmd == "reset":
            self.resets += 1
            self._reset_state()
            return True, None
        if cmd == "sleep":
            self.sleeping = True
            self.action = None
            return True, None
        # every other command wakes her up
        was_asleep = self.sleeping or self.droop > 0.3
        self.sleeping = False
        if cmd == "wake":
            self._start("wake" if was_asleep else None)
            return True, None
        if cmd == "idle":
            self.action = None
            self.track_forced = False
            self.idle_hold_until = self.t + 20.0
            self.next_glance = self.t
            return True, None
        if cmd == "track":
            self.action = None
            self.track_forced = True
            self.idle_hold_until = -1e9
            return True, None
        if cmd == "look":
            d = str(kw.get("dir", "player")).lower()
            if d == "me":
                d = "player"
            if d not in ("left", "right", "up", "down", "player"):
                return False, "look needs \"dir\":\"left\"|\"right\"|\"up\"|\"down\"|\"player\""
            self._start("look", dir=d, point=self._look_point(d))
            return True, None
        self._start(cmd)
        return True, None

    def _start(self, name, **args):
        if name is None:
            return
        self.action = name
        self.action_t0 = self.t
        self.action_end = self.t + ACTION_LEN[name]
        self.action_args = args
        self._spk = None
        if name in ("angry", "bow"):
            self.action_args["point"] = self._player_or_forward()

    def _frame(self):
        yaw = self.q_cmd[JI["yaw"]]
        F = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        L = np.array([-math.sin(yaw), math.cos(yaw), 0.0])
        return F, L

    def _player_or_forward(self):
        if self.player_fresh():
            return self.player.copy()
        F, _ = self._frame()
        return self.head_pos() + 7 * F + np.array([0, 0, -3.5])

    def _look_point(self, d):
        F, L = self._frame()
        h = self.head_pos()
        z = np.array([0, 0, 1.0])
        if d == "player":
            return self._player_or_forward()
        v = {"left": 0.35 * F + L - 0.2 * z, "right": 0.35 * F - L - 0.2 * z, "up": F + 1.1 * z,
             "down": 0.35 * F - z}[d]
        return h + 8 * v / np.linalg.norm(v)

    # ------------------------------------------------------------------ behaviour
    def _behaviour(self):
        t = self.t
        n = len(JN)
        if self.action and t >= self.action_end:
            if self.action == "spooky":
                self.track_forced = True   # she pulls back up into watching the player
            self.action = None
        if self.track_forced and (not self.player_fresh() or self.player_dist() >= 30.0):
            self.track_forced = False  # the player left: back to idle
        base = self.base_mode()
        act = self.action
        tau = t - self.action_t0
        F, L = self._frame()
        off = np.zeros(n)
        gm_ = np.ones(n)
        gaze_tau = 0.35
        yaw_follow, yaw_dead = True, D(20)
        use_ik = True
        # --- gaze point
        if base == "track":
            raw = self.player.copy()
            gaze_tau = 0.18
        else:
            if t >= self.next_glance:
                self.next_glance = t + self.rng.uniform(3.0, 8.0)
                if self.player_fresh() and self.rng.random() < 0.35:
                    self.glance = self.player.copy()
                else:
                    # glance around the player's side of the room (or her default facing), so idle glances
                    # don't random-walk the unlimited mount yaw round and round
                    if self.player_fresh():
                        base_az = math.atan2(self.player[1], self.player[0])
                    else:
                        base_az = 0.0
                    a = base_az + self.rng.uniform(-1.1, 1.1)
                    h = self.head_pos()
                    self.glance = np.array([h[0] + 7 * math.cos(a), h[1] + 7 * math.sin(a),
                                            h[2] + self.rng.uniform(-4.0, 0.8)])
            raw = self.glance
            yaw_dead = D(50)
            gaze_tau = 0.6          # idle glances: slow, deliberate head moves
        if act == "look":
            raw = self.action_args["point"]
            gaze_tau, yaw_follow = 0.15, self.action_args["dir"] == "player"
        elif act in ("angry", "bow"):
            raw = self.player.copy() if self.player_fresh() else self.action_args["point"]
            gaze_tau = 0.08 if act == "angry" else 0.25
        elif act == "scan":
            T = ACTION_LEN["scan"]
            ph = D(70) * math.cos(math.pi * min(tau / T, 1.0) * 2)  # +70 -> -70 -> +70 deg
            if tau < 0.6:
                ph = D(70)
            v = math.cos(ph) * F + math.sin(ph) * L + np.array([0, 0, -0.35])
            raw = self.head_pos() + 8 * v / np.linalg.norm(v)
            gaze_tau, yaw_follow = 0.25, False
        elif act == "spooky":
            raw = self._player_or_forward()
            if tau < SPOOKY_T["glitch"]:
                gaze_tau = 1.2          # power-down: a slow, heavy stare that lags behind
            elif tau < SPOOKY_T["swoop"]:
                gaze_tau = 0.08         # glitch dance: snaps
            else:
                gaze_tau, yaw_follow = 0.1, tau >= SPOOKY_T["back"]
        if self.sleeping:
            use_ik, yaw_follow = False, False
        a = 1 - math.exp(-CONTROL_DT / max(gaze_tau, 1e-3))
        self.gaze = self.gaze + a * (np.asarray(raw, float) - self.gaze)

        # --- lazy yaw at the mount
        if yaw_follow:
            rel = self.gaze[:2]
            if np.hypot(*rel) > 1.5:
                az = math.atan2(rel[1], rel[0])
                err = _wrap(az - self.yaw_goal)
                if abs(err) > yaw_dead or getattr(self, "_yaw_moving", False):
                    self._yaw_moving = abs(err) > D(4)
                    self.yaw_goal += max(-0.6 * CONTROL_DT, min(0.6 * CONTROL_DT, 1.5 * err * CONTROL_DT))
        # nearest equivalent of yaw_goal to the current yaw
        qy = self.data.qpos[self.qadr[0]]
        yaw_t = qy + _wrap(self.yaw_goal - qy)

        # --- look-at IK (on a kinematic copy: real arm/torso sway, commanded yaw, IK neck/head)
        if use_ik:
            self._ik(self.gaze, yaw_t)
        else:
            self.ik_q *= 0.97

        # --- procedural offsets
        br = 2 * math.pi * t
        off[JI["torso_y"]] += 0.03 * math.sin(br / 6.0)
        off[JI["torso_z"]] += 0.05 * math.sin(br / 11.3)
        off[JI["arm1_x"]] += 0.02 * math.sin(br / 9.3)
        off[JI["head_x"]] += 0.04 * math.sin(br / 7.1)
        # expressive tracking: lean the whole body toward a close player + a curious head tilt that changes
        # whenever the player has moved (event-driven, so a still player gets a still GLaDOS)
        if base == "track" and act is None and not self.sleeping and self.player_fresh():
            near = max(0.0, min(1.0, (TRACK_RANGE - self.player_dist()) / 6.0))
            off += self._lean(self.player, 0.06 * near)
            if self.tilt_anchor is None or np.linalg.norm(self.player - self.tilt_anchor) > 1.5:
                self.tilt_anchor = self.player.copy()
                self.tilt_goal = self.rng.choice((-1, 1)) * self.rng.uniform(0.10, 0.22) \
                    if self.rng.random() < 0.75 else 0.0
        else:
            self.tilt_goal, self.tilt_anchor = 0.0, None
        self.tilt += (1 - math.exp(-CONTROL_DT / 0.7)) * (self.tilt_goal - self.tilt)
        off[JI["head_x"]] += self.tilt
        # sleep droop
        target_droop = 1.0 if self.sleeping else 0.0
        if act == "wake":
            target_droop = 0.0
        rate = CONTROL_DT / (2.5 if self.sleeping else 1.8)
        self.droop += max(-rate, min(rate, target_droop - self.droop))
        dr = _smooth01(self.droop)
        off[JI["neck_y"]] += 1.0 * dr
        off[JI["head_y"]] += 0.85 * dr
        off[JI["head_x"]] += 0.22 * dr
        off[JI["torso_y"]] += 0.08 * dr
        if self.sleeping:
            off[JI["torso_y"]] += 0.03 * math.sin(br / 9.0) * dr
        if act == "nod":
            T = ACTION_LEN["nod"]
            off[JI["head_y"]] += 0.30 * math.sin(2 * math.pi * 1.7 * tau) * math.sin(math.pi * tau / T)
            off[JI["neck_y"]] += 0.08 * math.sin(2 * math.pi * 1.7 * tau) * math.sin(math.pi * tau / T)
        elif act == "shake":
            T = ACTION_LEN["shake"]
            off[JI["head_z"]] += 0.38 * math.sin(2 * math.pi * 1.7 * tau) * math.sin(math.pi * tau / T)
            off[JI["neck_z"]] += 0.10 * math.sin(2 * math.pi * 1.7 * tau) * math.sin(math.pi * tau / T)
        elif act == "wake":
            if tau > 1.3:
                off[JI["head_z"]] += 0.15 * math.sin(2 * math.pi * 2.2 * (tau - 1.3)) * math.sin(
                    math.pi * min((tau - 1.3) / 0.9, 1))
        elif act == "bow":
            T = ACTION_LEN["bow"]
            env = _smooth01(tau / 0.8) * _smooth01((T - tau) / 0.8)
            off[JI["neck_y"]] += 0.55 * env
            off[JI["head_y"]] += 0.45 * env
            off += self._lean(raw, 0.07 * env)
        elif act == "angry":
            env = _bump(tau, 0.4, 1.4) + _bump(tau, 1.9, 2.9) + 0.5 * _bump(tau, 3.0, 3.8)
            off += self._lean(raw, 0.16 * env)
            off[JI["neck_y"]] += 0.2 * env
            if t >= self.next_jitter:
                self.next_jitter = t + self.rng.uniform(0.07, 0.14)
                for k, amp in (("head_z", 0.14), ("head_y", 0.10), ("head_x", 0.12), ("neck_z", 0.06)):
                    self.jitter[JI[k]] = self.rng.uniform(-amp, amp)
            fade = _smooth01((ACTION_LEN["angry"] - tau) / 0.5)
            off += self.jitter * fade
            for k in ("neck_y", "neck_z", "head_y", "head_z", "head_x"):
                gm_[JI[k]] = 1.6
        elif act == "dance":
            # ~8 s groove at DANCE_BPM: head bob on every beat, head roll + neck/torso sway on 2/4-beat phrases,
            # the hanging chain circles once per 4 beats, a slow yaw swing over the whole routine
            T = ACTION_LEN["dance"]
            env = _smooth01(tau / 0.6) * _smooth01((T - tau) / 0.8)
            f = DANCE_BPM / 60.0
            ph = 2 * math.pi * f * tau
            off[JI["head_y"]] += env * (0.16 * math.sin(ph) - 0.05)
            off[JI["neck_y"]] += env * 0.06 * math.sin(ph - 0.6)
            off[JI["head_x"]] += env * 0.24 * math.sin(ph / 2)
            off[JI["neck_z"]] += env * 0.22 * math.sin(ph / 4)
            off[JI["head_z"]] += env * 0.12 * math.sin(ph / 2 + 1.0)
            off[JI["torso_z"]] += env * 0.22 * math.sin(ph / 4 + math.pi)
            off[JI["torso_x"]] += env * 0.08 * math.sin(ph / 2)
            off[JI["yaw"]] += env * 0.30 * math.sin(2 * math.pi * tau / T * 2)
            yaw_c = self.data.qpos[self.qadr[0]]
            a4 = ph / 4
            circ = np.array([math.cos(yaw_c + a4), math.sin(yaw_c + a4), 0.0]) * 5.0
            off += self._lean(circ, 0.07 * env)
            for k in ("neck_y", "neck_z", "head_y", "head_z", "head_x", "torso_z", "torso_x"):
                gm_[JI[k]] = 1.4
        elif act == "spooky":
            self._spooky_offsets(tau, off, gm_, F, L)
        # talking: mouthless head bob driven by the speech level
        self._talk_update()
        off[JI["head_y"]] += -0.07 * self.talk + 0.035 * self.talk * math.sin(2 * math.pi * 3.3 * t)
        off[JI["neck_y"]] += 0.02 * self.talk * math.sin(2 * math.pi * 1.7 * t)
        if self.sleeping:
            gm_ *= 0.8
        self.gain_mult = gm_
        self.damp_mult = np.ones(n)
        if act == "spooky" and tau >= SPOOKY_T["swoop"]:
            # near-critical damping on the heavy chain while she reaches out: no pendulum overshoot into the camera
            w = self._spooky_w(tau)
            for k in ("arm1_x", "arm1_y", "arm2_x", "arm2_y", "torso_x", "torso_y", "torso_z"):
                self.damp_mult[JI[k]] = 1.0 + 1.0 * w

        # --- final joint targets, clamped + rate limited
        if self.neck_scale != 1.0:
            # long-necked assets: gestures move the neck less; the head joint takes the remainder of the pitch
            for nn, hn in (("neck_y", "head_y"), ("neck_z", "head_z")):
                o = off[JI[nn]]
                off[JI[nn]] = o * self.neck_scale
                off[JI[hn]] += 0.5 * o * (1 - self.neck_scale)
        des = self.ik_q + off
        des[0] = yaw_t + off[0]
        self.comp_boost = 0.0
        if act == "spooky":
            des = self._spooky_blend(tau, des)
        lim = self.limited
        des[lim] = np.clip(des[lim], self.lo[lim] + 0.02, self.hi[lim] - 0.02)
        rl = self.rate * CONTROL_DT
        if act == "angry":
            rl[[JI[k] for k in ("neck_y", "neck_z", "head_y", "head_z", "head_x")]] *= 2.0
        elif act == "dance":
            rl[[JI[k] for k in ("head_y", "head_x", "head_z", "torso_z", "torso_x")]] *= 1.6
        elif act == "spooky" and SPOOKY_T["glitch"] <= tau < SPOOKY_T["swoop"]:
            rl[[JI[k] for k in ("neck_y", "neck_z", "head_y", "head_z", "head_x")]] *= 3.0   # head snaps
            rl[[JI[k] for k in ("torso_z", "torso_x", "torso_y")]] *= 1.8
        self.q_cmd += np.clip(des - self.q_cmd, -rl, rl)

        # --- eye colour
        if act == "angry":
            tgt = EYE_ANGRY
        elif self.droop > 0.5 and (self.sleeping or act == "wake"):
            tgt = EYE_SLEEP
        else:
            tgt = EYE_NORMAL.copy()
            tgt[3] = 0.85 + 0.15 * self.talk
            if act == "dance":
                tgt[3] = 0.75 + 0.25 * abs(math.sin(math.pi * DANCE_BPM / 60.0 * tau))
        if self.sleeping:
            tgt = EYE_NORMAL + (EYE_SLEEP - EYE_NORMAL) * _smooth01(self.droop * 1.5)
        ea = 1 - math.exp(-CONTROL_DT / 0.15)
        if act == "spooky":
            tgt, eye_tau = self._spooky_eye(tau)
            ea = 1 - math.exp(-CONTROL_DT / eye_tau)
        self.eye = self.eye + ea * (tgt - self.eye)

    # ------------------------------------------------------------------ spooky dance
    def _spk_state(self):
        if self._spk is None:
            self._spk = {"roll_from": 0.0, "roll_to": 0.0, "roll_t0": -9.0, "next_roll": 0.0,
                         "glitch": np.zeros(len(JN)), "next_glitch": SPOOKY_T["glitch"], "flash": 0.0,
                         "flicker": 1.0, "next_flicker": 0.25, "glow2": 1.0, "red": 0.0,
                         "fallback": self._player_or_forward(), "q": None, "elev": None,
                         "idx": np.array([JI[n] for n in SPOOKY_IK])}
        return self._spk

    @staticmethod
    def _spooky_w(tau):
        """Swoop blend weight: 0 before the swoop, eases to 1 (arrive), holds, eases back to 0 (pull back)."""
        T = SPOOKY_T
        if tau < T["swoop"]:
            return 0.0
        if tau < T["hold"]:
            x = (tau - T["swoop"]) / (T["hold"] - T["swoop"])
            return _smooth01(x) ** 0.8          # quick dive, decelerating into her stop
        if tau < T["back"]:
            return 1.0
        return 1.0 - _smooth01((tau - T["back"]) / (T["end"] - T["back"] - 0.3))

    def _spooky_offsets(self, tau, off, gm_, F, L):
        s = self._spk_state()
        T = SPOOKY_T
        rng = self.rng
        if tau < T["glitch"]:
            # phase 1, power down: the head sags, the body swings like a pendulum with growing amplitude, slow
            # unnatural head rolls (held, then rolling over to the other side) with a counter-twisting neck
            env = _smooth01(tau / 1.2) * _smooth01((T["glitch"] - tau) / 0.7)
            off[JI["neck_y"]] += 0.30 * env
            off[JI["head_y"]] += 0.18 * env
            amp = 0.12 * _smooth01(tau / 2.5) * _smooth01((T["glitch"] - tau) / 0.9)
            off += self._lean(L * 5.0, amp * math.sin(2 * math.pi * tau / 3.6))
            off += self._lean(F * 5.0, 0.35 * amp * math.sin(2 * math.pi * tau / 3.6 + 1.2))
            if tau >= s["next_roll"]:
                s["roll_from"] = s["roll_to"] if tau - s["roll_t0"] > 1.3 else s["roll_from"]
                sign = -1.0 if s["roll_to"] > 0 else 1.0
                s["roll_to"] = sign * rng.uniform(0.32, 0.48)
                s["roll_t0"] = tau
                s["next_roll"] = tau + rng.uniform(1.5, 2.1)
            roll = s["roll_from"] + (s["roll_to"] - s["roll_from"]) * _smooth01((tau - s["roll_t0"]) / 1.3)
            off[JI["head_x"]] += roll * env
            off[JI["neck_z"]] += -0.45 * roll * env
            off[JI["head_z"]] += 0.12 * env * math.sin(2 * math.pi * tau / 4.7)
            for k in ("neck_y", "neck_z", "head_y", "head_z", "head_x"):
                gm_[JI[k]] = 0.9                 # heavy, sluggish
        elif tau < T["swoop"]:
            # phase 2, glitch dance: a wide swinging groove at DANCE_BPM, off-beat twitches that snap the head into
            # random held poses, strobing eye
            u = tau - T["glitch"]
            T2 = T["swoop"] - T["glitch"]
            env = _smooth01(u / 0.4) * _smooth01((T2 - u) / 0.7)
            beat = 60.0 / DANCE_BPM
            ph = 2 * math.pi * u / beat
            yaw_c = self.data.qpos[self.qadr[0]]
            a4 = ph / 4
            circ = np.array([math.cos(yaw_c + a4), math.sin(yaw_c + a4), 0.0]) * 5.0
            off += self._lean(circ, 0.16 * env)
            off[JI["torso_z"]] += env * 0.32 * math.sin(ph / 4 + math.pi)
            off[JI["torso_x"]] += env * 0.12 * math.sin(ph / 2)
            off[JI["yaw"]] += env * 0.30 * math.sin(2 * math.pi * u / T2 * 1.5)
            off[JI["head_y"]] += env * 0.10 * math.sin(ph)
            if u >= s["next_glitch"] - T["glitch"]:
                # next twitch on an off-beat (the "and" of a beat), sometimes skipping one: syncopated
                k_beat = math.floor(u / beat)
                nxt = (k_beat + 1.5) * beat if rng.random() < 0.7 else (k_beat + 2.5) * beat
                s["next_glitch"] = T["glitch"] + nxt
                g = np.zeros(len(JN))
                if rng.random() < 0.85:
                    for name, amp in (("head_y", 0.32), ("head_z", 0.40), ("head_x", 0.45), ("neck_z", 0.25),
                                      ("neck_y", 0.15), ("torso_x", 0.08)):
                        g[JI[name]] = rng.uniform(-amp, amp)
                s["glitch"] = g
                s["glow2"] = rng.choice((1.0, 1.0, 0.15, 0.45, 0.05))
                s["red"] = 1.0 if rng.random() < 0.18 else 0.0     # a teaser flash of red now and then
            off += s["glitch"] * env
            if u > T2 - 1.6:                          # line up the slow mount yaw for the swoop
                self.yaw_goal = self._spooky_yaw(self._spooky_target()[0])
            for k in ("neck_y", "neck_z", "head_y", "head_z", "head_x"):
                gm_[JI[k]] = 2.0
            for k in ("torso_z", "torso_x", "torso_y"):
                gm_[JI[k]] = 1.5
        else:
            w = self._spooky_w(tau)
            gm_[JI["yaw"]] = 1.0 + 0.5 * w
            for k in ("arm1_x", "arm1_y", "arm2_x", "arm2_y", "torso_x", "torso_y", "torso_z"):
                gm_[JI[k]] = 1.0 + 0.6 * w
            for k in ("neck_y", "neck_z", "head_y", "head_z"):
                gm_[JI[k]] = 1.0 + 0.3 * w

    def _spooky_yaw(self, P):
        """Mount yaw facing the player's azimuth, the equivalent nearest to the current yaw."""
        yc = float(self.data.qpos[self.qadr[0]])
        if np.hypot(P[0], P[1]) < 0.3:
            return yc
        return yc + _wrap(math.atan2(P[1], P[0]) - yc)

    def _spooky_goal(self, P, elev, zmin, standoff=SPOOKY_STANDOFF):
        F, _ = self._frame()
        v = -np.asarray(P[:2], float)                 # from the player toward the mount axis (where she hangs)
        if np.hypot(*v) < 0.3:
            v = -F[:2]
        v = v / np.linalg.norm(v)
        dvec = np.array([v[0] * math.cos(elev), v[1] * math.cos(elev), math.sin(elev)])
        g = np.asarray(P, float) + standoff * dvec
        g[2] = max(g[2], zmin + 0.6)
        return g

    def _reach_ik(self, goal, look, q, zmin, iters, max_step=None):
        """Whole-chain damped least squares (kinematic copy): eye site -> goal, eye axis -> look, the head's
        bounding box above zmin, joint limits respected. Returns the full joint vector. max_step (rad) bounds the
        total change per call, so per-tick use traces a continuous trajectory instead of jumping IK branches."""
        m, k = self.model, self.kin
        idx = self._spk_state()["idx"]
        cols = self.dadr[idx]
        lo, hi = self.lo[idx] + 0.05, self.hi[idx] - 0.05
        q = np.array(q, float)
        q_in = q[idx].copy()
        jp, jr = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        jp2, jr2 = np.zeros((3, m.nv)), np.zeros((3, m.nv))
        wr = 1.2
        for _ in range(iters):
            k.qpos[self.qadr] = q
            mujoco.mj_kinematics(m, k)
            mujoco.mj_comPos(m, k)
            e = k.site_xpos[self.eye_sid].copy()
            f = k.site_xmat[self.eye_sid].reshape(3, 3)[:, 0]
            mujoco.mj_jacSite(m, k, jp, jr, self.eye_sid)
            rows, errs = [jp[:, cols]], [np.asarray(goal, float) - e]
            dv = np.asarray(look, float) - e
            nd = np.linalg.norm(dv)
            if nd > 1e-6:
                dv /= nd
                ax = np.cross(f, dv)
                sn = np.linalg.norm(ax)
                if sn > 1e-9:
                    rows.append(wr * jr[:, cols])
                    errs.append(wr * ax / sn * math.atan2(sn, float(f @ dv)))
            R = k.xmat[self.head_bid].reshape(3, 3)
            C = self.head_box @ R.T + k.xpos[self.head_bid]
            for i in np.nonzero(C[:, 2] < zmin + 0.1)[0]:
                mujoco.mj_jac(m, k, jp2, jr2, np.ascontiguousarray(C[i]), self.head_bid)
                rows.append(4.0 * jp2[2:3, cols].copy())
                errs.append(4.0 * np.array([zmin + 0.15 - C[i, 2]]))
            J = np.vstack(rows)
            err = np.concatenate(errs)
            # joints near a limit get a smaller weight (they move less), so the solution doesn't limit-cycle
            # against the clip with several joints chasing each other
            wj = np.clip(np.minimum(q[idx] - lo, hi - q[idx]) / 0.25, 0.08, 1.0)
            Jw = J * wj
            dq = wj * (Jw.T @ np.linalg.solve(Jw @ Jw.T + 0.09 * np.eye(len(err)), err))
            q[idx] = np.clip(q[idx] + np.clip(dq, -0.12, 0.12), lo, hi)
            if max_step is not None:
                q[idx] = np.clip(q[idx], q_in - max_step, q_in + max_step)
        return q

    def _spooky_target(self):
        """Player eye (or the point picked at the start when nobody is around) and the floor clamp height."""
        s = self._spk_state()
        P = self.player.copy() if self.player_fresh() else s["fallback"]
        zmin = P[2] - PLAYER_EYE_H + SPOOKY_ABOVE_FEET
        return P, zmin

    def _spooky_tilt(self, tau):
        """Slow, unnatural head roll while she studies the test subject up close (rad)."""
        T = SPOOKY_T
        s = self._spk_state()
        d = s.setdefault("tilt_dir", self.rng.choice((-1.0, 1.0)))
        return d * 0.30 * _smooth01((tau - T["hold"] + 0.2) / 1.0) * (1.0 - _smooth01((tau - T["back"]) / 1.2))

    def _spooky_plan(self):
        """Climax pose (fully converged, from the current pose): the approach elevation is the lowest (face-to-face)
        one she can reach without pinning joints against their limits, else the one getting closest."""
        s = self._spk_state()
        P, zmin = self._spooky_target()
        q0 = self.q()
        q0[0] = self._spooky_yaw(P)
        q0[JI["head_x"]] = self._spooky_tilt(SPOOKY_T["back"] - 0.01)   # plan with the full tilt
        best = None
        for deg in (0, 15, 30, 45, 60, 75, 88):
            e = math.radians(deg)
            g = self._spooky_goal(P, e, zmin)
            q = self._reach_ik(g, P, q0, zmin, 40)
            k = self.kin
            k.qpos[self.qadr] = q
            mujoco.mj_kinematics(self.model, k)
            err = float(np.linalg.norm(k.site_xpos[self.eye_sid] - g))
            idx = s["idx"]
            margin = np.minimum(q[idx] - self.lo[idx], self.hi[idx] - q[idx])
            strain = float(np.clip(1.0 - margin / 0.3, 0.0, 1.0).sum())   # joints pinned near a limit: twitchy
            score = err + 0.12 * e + 0.15 * strain
            if best is None or score < best[0]:
                best = (score, e, q, err)
        s["elev"], s["q"], s["plan_err"] = best[1], best[2], best[3]
        s["P_solved"] = P.copy()
        return s

    def _spooky_blend(self, tau, des):
        w = self._spooky_w(tau)
        if w <= 0.0:
            return des
        T = SPOOKY_T
        s = self._spk_state()
        P, zmin = self._spooky_target()
        if s["q"] is None:
            q0 = self.q()
            q0[0] = self._spooky_yaw(P)
            s["q0"] = q0
            self._spooky_plan()
            s["x"] = 0.0
        # reactive keep-out: if the real (swinging) eye gets closer than the minimum, back the goal off
        d_now = float(np.linalg.norm(self.eye_pos() - P))
        push = s.get("push", 0.0)
        push = push + 0.06 * (SPOOKY_MIN_DIST + 0.2 - d_now) if d_now < SPOOKY_MIN_DIST + 0.2 else push * 0.985
        s["push"] = push = min(max(push, 0.0), 1.5)
        so = SPOOKY_STANDOFF + push
        # the player moved (or she had to back off): re-solve the climax pose incrementally (continuous, no jumps)
        if float(np.linalg.norm(P - s["P_solved"])) > 0.12 or push > 0.05:
            s["q"][0] = self._spooky_yaw(P)
            s["q"] = self._reach_ik(self._spooky_goal(P, s["elev"], zmin, so), P, s["q"], zmin, 4, max_step=0.03)
            if push <= 0.05:
                s["P_solved"] = P.copy()
        if tau < T["hold"]:
            # swoop: joint-space glide from where she was to the climax pose (monotone, so no wandering through the
            # redundant chain); the progress stalls if the commanded eye would enter the keep-out sphere
            x = _smooth01((tau - T["swoop"]) / (T["hold"] - T["swoop"] - 0.15))
            qx = s["q0"] + (s["q"] - s["q0"]) * x
            k = self.kin
            k.qpos[self.qadr] = qx
            mujoco.mj_kinematics(self.model, k)
            if float(np.linalg.norm(k.site_xpos[self.eye_sid] - P)) < so - 0.05 and x > s["x"]:
                x = s["x"]
                qx = s["q0"] + (s["q"] - s["q0"]) * x
            s["x"] = x
            qr = qx
            wb = _smooth01((tau - T["swoop"]) / 0.35)
        else:
            qr = s["q"].copy()
            wb = w
        qr = qr.copy()
        qr[JI["head_x"]] = self._spooky_tilt(tau)
        # gaze lock: the reach IK assumes the arm sits exactly at its target, the real (heavy, lagging) arm doesn't;
        # re-aim the head joints at the player on the REAL body pose (like the normal look-at IK), blended in
        # from late in the swoop, so she really stares into the camera
        ga_w = _smooth01((tau - T["hold"] + 0.8) / 0.6) * (1.0 - _smooth01((tau - T["back"]) / 1.0))
        if ga_w > 0.0:
            k = self.kin
            hj = [JI["head_y"], JI["head_z"]]
            k.qpos[:] = self.data.qpos
            k.qpos[self.qadr[JI["head_x"]]] = qr[JI["head_x"]]
            hv = qr[hj].copy()
            for _ in range(4):
                k.qpos[self.qadr[hj]] = hv
                mujoco.mj_kinematics(self.model, k)
                f = k.site_xmat[self.eye_sid].reshape(3, 3)[:, 0]
                dv = P - k.site_xpos[self.eye_sid]
                dv /= max(float(np.linalg.norm(dv)), 1e-6)
                ax = np.cross(f, dv)
                sn = float(np.linalg.norm(ax))
                if sn < 1e-6:
                    break
                err = ax / sn * math.atan2(sn, float(f @ dv))
                hv = np.clip(hv + np.clip(0.6 * (k.xaxis[self.jid[hj]] @ err), -0.15, 0.15),
                             self.lo[hj] + 0.05, self.hi[hj] - 0.05)
            gl = s.get("gl")
            gl = hv if gl is None else gl + 0.2 * (hv - gl)     # low-pass: no tick-to-tick flipping
            s["gl"] = gl
            qr[hj] += ga_w * (gl - qr[hj])
        else:
            s["gl"] = None
        if T["hold"] - 0.3 <= tau < T["back"] + 0.4:   # tiny tremble while she stares
            t = self.t
            qr[JI["head_x"]] += 0.022 * math.sin(2 * math.pi * 8.3 * t) + 0.012 * math.sin(2 * math.pi * 13.1 * t)
            qr[JI["head_z"]] += 0.014 * math.sin(2 * math.pi * 6.7 * t + 1.0)
            qr[JI["head_y"]] += 0.010 * math.sin(2 * math.pi * 11.3 * t + 2.0)
        yaw_c = self.data.qpos[self.qadr[0]]
        qr[0] = yaw_c + _wrap(qr[0] - yaw_c)
        if wb > 0.5:
            self.yaw_goal = qr[0]                   # track resumes from where she is facing
        self.comp_boost = w
        return (1.0 - wb) * des + wb * qr

    def _spooky_eye(self, tau):
        """(target eye rgba, smoothing time constant) for the spooky dance."""
        s = self._spk_state()
        T = SPOOKY_T
        rng = self.rng
        if tau < T["glitch"]:
            if tau >= s["next_flicker"]:            # failing power: random dropouts of varying depth
                if s["flicker"] < 1.0:
                    s["flicker"], s["next_flicker"] = 1.0, tau + rng.uniform(0.15, 0.9)
                else:
                    s["flicker"], s["next_flicker"] = rng.uniform(0.0, 0.45), tau + rng.uniform(0.04, 0.16)
            g = 0.85 - 0.62 * _smooth01(tau / 1.6)
            if 0.35 < tau < 0.6:
                g = 0.02                            # the "power cut" blink
            col = EYE_NORMAL + (np.array([1.0, 0.62, 0.18, 1.0]) - EYE_NORMAL) * 0.4
            tgt = col.copy()
            tgt[3] = g * s["flicker"]
            return tgt, 0.03
        if tau < T["swoop"]:
            tgt = EYE_SPOOKY.copy() if s["red"] else EYE_NORMAL.copy()
            tgt[3] = s["glow2"] * (0.75 if not s["red"] else 1.0)
            return tgt, 0.025
        w = self._spooky_w(tau)
        if tau < T["back"] + 0.2:
            tgt = EYE_NORMAL + (EYE_SPOOKY - EYE_NORMAL) * min(1.0, 1.4 * w)
            tgt[3] = 1.0 if tau < T["hold"] else 0.93 + 0.07 * math.sin(2 * math.pi * 6.0 * self.t)
            return tgt, 0.06
        k = _smooth01((tau - T["back"] - 0.2) / 1.4)
        tgt = EYE_SPOOKY + (EYE_NORMAL - EYE_SPOOKY) * k
        tgt[3] = 1.0 - 0.15 * k
        return tgt, 0.2

    def _lean(self, point, ang):
        """Joint offsets swinging the hanging chain so her lower body leans toward `point` by ~ang rad."""
        off = np.zeros(len(JN))
        if ang == 0:
            return off
        yaw = self.data.qpos[self.qadr[0]]
        v = np.asarray(point, float)[:2]
        if np.hypot(*v) < 0.5:
            return off
        c, s = math.cos(-yaw), math.sin(-yaw)
        dx, dy = c * v[0] - s * v[1], s * v[0] + c * v[1]
        nrm = math.hypot(dx, dy)
        dx, dy = dx / nrm, dy / nrm
        for jx, jy, w in (("arm1_x", "arm1_y", 1.0), ("arm2_x", "arm2_y", 0.9), ("torso_x", "torso_y", 0.6)):
            off[JI[jx]] += ang * w * dy
            off[JI[jy]] += -ang * w * dx
        return off

    def _ik(self, target, yaw_t, iters=3):
        k = self.kin
        k.qpos[:] = self.data.qpos
        k.qpos[self.qadr[0]] = yaw_t
        idx = [JI[n] for n in IK_W]
        w = np.array(list(IK_W.values()))
        for _ in range(iters):
            k.qpos[self.qadr[idx]] = self.ik_q[idx]
            mujoco.mj_kinematics(self.model, k)
            e = k.site_xpos[self.eye_sid]
            f = k.site_xmat[self.eye_sid].reshape(3, 3)[:, 0]
            d = np.asarray(target, float) - e
            nd = np.linalg.norm(d)
            if nd < 0.3:
                return
            d /= nd
            ax = np.cross(f, d)
            s = np.linalg.norm(ax)
            ang = math.atan2(s, float(f @ d))
            if s < 1e-9:
                return
            err = ax / s * ang
            axes = k.xaxis[self.jid[idx]]
            dq = w * (axes @ err)
            self.ik_q[idx] = np.clip(self.ik_q[idx] + dq, self.lo[idx] + 0.02, self.hi[idx] - 0.02)
        # mild preference for the head over the neck (keeps the neck from winding up against a limit)
        for nn, hn in (("neck_z", "head_z"), ("neck_y", "head_y")):
            i, j = JI[nn], JI[hn]
            mv = 0.004 * self.ik_q[i]
            room = (self.hi[j] - 0.1 - self.ik_q[j]) if mv > 0 else (self.ik_q[j] - self.lo[j] - 0.1)
            if room > abs(mv):
                self.ik_q[i] -= mv
                self.ik_q[j] += mv

    def _gravity(self):
        """Gravity-only joint torques at the current configuration (zero velocity), once per control tick."""
        k = self.kin
        k.qpos[:] = self.data.qpos
        k.qvel[:] = 0
        mujoco.mj_kinematics(self.model, k)
        mujoco.mj_comPos(self.model, k)
        out = np.zeros(self.model.nv)
        mujoco.mj_rne(self.model, k, 0, out)
        return out[self.dadr]

    def _talk_update(self):
        active = self.talk_on and self.t < self.talk_until
        if not active:
            self.talk_on = False
            raw = 0.0
        elif self.t - self.talk_in_t < 0.6:
            raw = self.talk_in
        else:  # no levels from the voice backend: synthetic syllables
            raw = 0.45 + 0.35 * math.sin(2 * math.pi * 2.9 * self.t) * math.sin(2 * math.pi * 0.7 * self.t)
        tau = 0.05 if raw > self.talk else 0.15
        self.talk += (1 - math.exp(-CONTROL_DT / tau)) * (raw - self.talk)
        self.talk = min(max(self.talk, 0.0), 1.0)

    # ------------------------------------------------------------------ physics
    def step(self):
        self._behaviour()
        d = self.data
        kp = self.kp0 * self.gain_mult ** 2
        # the D term lives in MuJoCo's joint damping: integrated implicitly (implicitfast), so it stays stable
        # even though the mass-matrix diagonal overestimates the effective inertia of the coupled chain
        self.model.dof_damping[self.dadr] = self.kd0 * self.gain_mult * self.damp_mult + 1.0
        comp = self.comp + self.comp_boost * (1.0 - self.comp)
        grav = comp * self._gravity()
        for _ in range(SUBSTEPS):
            q = d.qpos[self.qadr]
            tau = kp * (self.q_cmd - q) + grav
            d.qfrc_applied[self.dadr] = np.clip(tau, -self.tau_max, self.tau_max)
            mujoco.mj_step(self.model, d)
        self.t += CONTROL_DT
        if not (np.all(np.isfinite(d.qpos)) and np.all(np.isfinite(d.qvel))):
            self.nan_resets += 1
            self.log(f"non-finite state at t={self.t:.2f}: reset")
            self._reset_state()

    # ------------------------------------------------------------------ model switch
    def carry_state_from(self, old):
        """Adopt the behavioural state of another sim (live model switch: same mode, player, talk, yaw)."""
        self.t = old.t
        for k in ("player", "player_t", "scale_hint", "sleeping", "droop", "track_forced", "idle_hold_until",
                  "talk_on", "talk_until", "talk_in", "talk_in_t", "talk", "action", "action_t0", "action_end",
                  "action_args", "next_glance", "resets", "nan_resets"):
            v = getattr(old, k)
            setattr(self, k, v.copy() if isinstance(v, (np.ndarray, dict)) else v)
        self.eye = old.eye.copy()
        self.yaw_goal = old.yaw_goal
        self.data.qpos[self.qadr[0]] = old.data.qpos[old.qadr[0]]
        self.q_cmd[0] = old.q_cmd[0]
        if self.droop > 0:
            for k, a in (("neck_y", 1.0 * self.neck_scale), ("head_y", 0.85), ("head_x", 0.22), ("torso_y", 0.08)):
                self.data.qpos[self.qadr[JI[k]]] = a * _smooth01(self.droop)
            self.q_cmd[:] = self.data.qpos[self.qadr]
        mujoco.mj_forward(self.model, self.data)
        self.gaze = self.eye_pos() + 8 * self.eye_dir()

    # ------------------------------------------------------------------ contract payloads
    def pose_dict(self, seq=0, rev=None):
        d = self.data
        bodies = {}
        for name, b in zip(self.body_names, self.body_ids):
            p, q = d.xpos[b], d.xquat[b]
            bodies[name] = [round(float(p[0]), 5), round(float(p[1]), 5), round(float(p[2]), 5),
                            round(float(q[0]), 5), round(float(q[1]), 5), round(float(q[2]), 5), round(float(q[3]), 5)]
        out = {"seq": seq, "t": round(self.t, 3), "mode": self.public_mode(), "talk": round(self.talk, 3),
               "eye": [round(float(x), 3) for x in self.eye], "bodies": bodies}
        if rev is not None:
            out["rev"] = rev
        return out

    def aim_error_deg(self, target):
        d = np.asarray(target, float) - self.eye_pos()
        d /= np.linalg.norm(d)
        return math.degrees(math.acos(max(-1.0, min(1.0, float(self.eye_dir() @ d)))))

    def limit_violation_deg(self):
        q = self.q()
        lim = self.limited
        over = np.maximum(q - self.hi, self.lo - q)[lim]
        return float(np.degrees(max(0.0, over.max())))
