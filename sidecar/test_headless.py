#!/usr/bin/env python3
"""Headless checks (no HTTP), ~40x real time. PASS/FAIL summary; exit code 1 on any failure."""

import math
import os
import random
import sys
import time

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import glados_model as gm  # noqa: E402
import glados_sim as gs  # noqa: E402

results = []
WORST_LIMIT = [0.0, ""]


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def run(s, secs, tag=""):
    nan = False
    for _ in range(int(round(secs * 60))):
        s.step()
        if not np.all(np.isfinite(s.data.qpos)):
            nan = True
        v = s.limit_violation_deg()
        if v > WORST_LIMIT[0]:
            WORST_LIMIT[:] = [v, f"{tag} t={s.t:.1f}"]
    return nan


def feed_player(s, p, secs, tag=""):
    for _ in range(int(secs * 5)):
        s.set_player(*p)
        run(s, 0.2, tag)


def rss_mb():
    with open("/proc/self/status") as f:
        for line in f:
            if line.startswith("VmRSS:"):
                return int(line.split()[1]) / 1024
    return -1


def _spooky_run(asset, P, seed, head_pts, interrupt_at=None):
    """Run one spooky dance with the player standing at eye position P. Returns per-phase measurements."""
    s = gs.GladosSim(log=lambda *a: None, seed=seed, asset=asset)
    feed_player(s, P, 4.0, "spooky pre")
    ok, _ = s.command("creepy")                       # alias
    T = gs.SPOOKY_T
    t0 = s.t
    zmin = P[2] - gs.PLAYER_EYE_H + gs.SPOOKY_ABOVE_FEET
    d0 = float(np.linalg.norm(s.eye_pos() - P))
    m = dict(alias_ok=ok and s.action == "spooky", d0=d0, zmin=zmin, dmin=1e9, d_hold=[], aim_hold=[], zlow=1e9,
             nan=False, p1_glow=[], p1_roll=0.0, p2_twitches=0, p2_swing=0.0, p3_eye=None, modes=[], hmin=1e9)
    last_glitch = None
    steps = int((T["end"] + 3.0) * 60) if interrupt_at is None else int((interrupt_at + 3.0) * 60)
    for k in range(steps):
        if k % 12 == 0:
            s.set_player(*P)
        tau = s.t - t0
        if interrupt_at is not None and "interrupt" not in m and tau >= interrupt_at:
            m["interrupt"] = s.command("nod")
            m["mode_after_cmd"] = s.public_mode()
        nan = run(s, 1 / 60, f"spooky {s.asset_name}")
        tau = s.t - t0
        m["nan"] |= nan or not np.all(np.isfinite(s.data.qpos))
        m["modes"].append(s.public_mode())
        if tau < T["glitch"]:
            m["p1_glow"].append(float(s.eye[3]))
            m["p1_roll"] = max(m["p1_roll"], abs(math.degrees(s.q()[gs.JI["head_x"]])))
        elif tau < T["swoop"] and s.action == "spooky":
            g = s._spk["glitch"] if s._spk else None
            if g is not None and (last_glitch is None or not np.array_equal(g, last_glitch)):
                m["p2_twitches"] += 1
                last_glitch = g.copy()
            m["p2_swing"] = max(m["p2_swing"], float(np.degrees(np.abs(s.q()[1:5]).max())))
        if T["swoop"] <= tau <= T["back"] + 0.3 and s.action == "spooky":
            R = s.data.xmat[s.head_bid].reshape(3, 3)
            V = head_pts @ R.T + s.data.xpos[s.head_bid]
            m["zlow"] = min(m["zlow"], float(V[:, 2].min()))
            m["hmin"] = min(m["hmin"], float(np.linalg.norm(V - P, axis=1).min()))
            d = float(np.linalg.norm(s.eye_pos() - P))
            m["dmin"] = min(m["dmin"], d)
            if T["hold"] + 0.5 <= tau < T["back"]:
                m["d_hold"].append(d)
                m["aim_hold"].append(s.aim_error_deg(P))
                m["p3_eye"] = s.eye.copy()
    m["end_mode"] = s.public_mode()
    m["end_eye"] = s.eye.copy()
    m["nan_resets"] = s.nan_resets
    m["comp_boost"] = s.comp_boost
    return m


def spooky_checks(name):
    import glados_assets as ga
    a = None if name == "procedural" else ga.load_asset(name)
    pa = a if a is not None else ga.Asset("procedural", ga.procedural_asset(), b"")
    H = pa.body_points("head")
    H = H[:: max(1, len(H) // 500)]
    T = gs.SPOOKY_T
    rows, ok_all, worst = [], True, []
    # reachable: ~5 m out from under the mount, standing on a ledge ~2 m above the lab floor (eye 9.2-9.4 m below
    # the ceiling); she hangs ~7.6 m, so a player on the 13 m-deep floor itself is out of reach (lunge case below)
    for P, seed in (([5.0, 1.0, -9.4], 3), ([4.0, -3.0, -9.2], 4), ([6.0, 0.0, -8.0], 5)):
        P = np.array(P)
        m = _spooky_run(a, P, seed, H)
        p1 = np.array(m["p1_glow"])
        phases = (p1.min() < 0.2 and m["p1_roll"] > 10 and m["p2_twitches"] >= 6 and m["p2_swing"] > 5
                  and m["p3_eye"] is not None and m["p3_eye"][1] < 0.35 and m["p3_eye"][3] > 0.85)
        dist_ok = 1.2 <= m["dmin"] <= 2.2 and max(m["d_hold"]) <= 2.2
        floor_ok = m["zlow"] >= m["zmin"] - 0.05
        aim = float(np.median(m["aim_hold"]))
        back = m["end_mode"] == "track" and m["end_eye"][1] > 0.7 and m["comp_boost"] == 0.0
        stable = not m["nan"] and m["nan_resets"] == 0
        ok = m["alias_ok"] and phases and dist_ok and floor_ok and back and stable and aim < 15
        ok_all &= ok
        worst.append(m["dmin"])
        rows.append(f"P={P.tolist()}: start {m['d0']:.1f} m -> min {m['dmin']:.2f} m, hold "
                    f"{min(m['d_hold']):.2f}-{max(m['d_hold']):.2f} m, head mesh >= {m['hmin']:.2f} m, aim median "
                    f"{aim:.1f} deg, head low {m['zlow']:.2f} >= clamp {m['zmin']:.2f}; p1 glow min {p1.min():.2f} "
                    f"roll {m['p1_roll']:.0f} deg, p2 {m['p2_twitches']} twitches swing {m['p2_swing']:.0f} deg, "
                    f"p3 eye {np.round(m['p3_eye'], 2).tolist()}, end {m['end_mode']}")
    check(f"[{name}] spooky dance phases + climax distance + floor clamp + back to track", ok_all,
          " | ".join(rows))
    SPOOKY_DIST[name] = worst
    # out of reach (player on the 13 m-deep floor, 10 m out): she lunges as far as she can, still >= 1.2 m
    P = np.array([10.0, 0.0, -11.4])
    m = _spooky_run(a, P, 6, H)
    check(f"[{name}] spooky lunge when out of reach", m["dmin"] < m["d0"] - 0.5 and m["dmin"] >= 1.2 and
          not m["nan"] and m["end_mode"] == "track" and m["zlow"] >= m["zmin"] - 0.05,
          f"player 10 m out on the floor: eye {m['d0']:.1f} m -> {m['dmin']:.2f} m (as far as the chain reaches)")
    # interrupt mid-swoop with nod
    P = np.array([5.0, 1.0, -9.4])
    m = _spooky_run(a, P, 7, H, interrupt_at=T["swoop"] + 1.0)
    after = m["modes"][-1]
    check(f"[{name}] spooky interrupted by nod", m["interrupt"] == (True, None) and m["mode_after_cmd"] == "nod"
          and after == "track" and not m["nan"] and m["nan_resets"] == 0 and m["comp_boost"] == 0.0 and
          m["end_eye"][1] > 0.7, f"nod at tau {T['swoop'] + 1.0:.1f} s -> {m['mode_after_cmd']}, 3 s later {after}, "
          f"eye {np.round(m['end_eye'], 2).tolist()}")
    # the player walks around her (1 m/s on a 5 m circle) during the whole dance: stays stable, never closer than
    # the keep-out distance, ends in track
    s = gs.GladosSim(log=lambda *a: None, seed=8, asset=a)
    pl = lambda t: np.array([5.0 * math.cos(0.2 * t), 5.0 * math.sin(0.2 * t), -9.2])  # noqa: E731
    feed_player(s, pl(0), 3.0, "spooky walk pre")
    s.command("spooky")
    t0, dmin, nan = s.t, 1e9, False
    for k in range(int((T["end"] + 2.0) * 60)):
        tau = s.t - t0
        if k % 12 == 0:
            s.set_player(*pl(tau))
        nan |= run(s, 1 / 60, f"spooky walk {name}")
        if T["swoop"] <= tau <= T["back"]:
            dmin = min(dmin, float(np.linalg.norm(s.eye_pos() - pl(tau))))
    check(f"[{name}] spooky with a walking player", not nan and s.nan_resets == 0 and dmin >= 1.2 and
          s.public_mode() == "track", f"closest eye approach {dmin:.2f} m while the player circles at 1 m/s, "
          f"end {s.public_mode()}")


SPOOKY_DIST = {}


def main():
    t_wall = time.time()
    # ---------------------------------------------------------------- model
    m = gm.model_json()
    names = [b["name"] for b in m["bodies"]]
    meshes = [me for b in m["bodies"] for me in b["meshes"]]
    ntris = sum(len(me["tris"]) // 3 for me in meshes)
    idx_ok = all(max(me["tris"]) < len(me["verts"]) // 3 and min(me["tris"]) >= 0 for me in meshes)
    nrm_ok = all(len(me["normals"]) == len(me["verts"]) and
                 np.allclose(np.linalg.norm(np.array(me["normals"]).reshape(-1, 3), axis=1), 1, atol=0.01)
                 for me in meshes)
    eyes = [(b["name"], me["name"]) for b in m["bodies"] for me in b["meshes"] if me.get("tag") == "eye"]
    check("model.json", names[0] == "mount" and ntris <= 16000 and idx_ok and nrm_ok and eyes == [("head", "head_eye")],
          f"bodies {names}, {ntris} tris, eye mesh {eyes}, normals unit {nrm_ok}")
    wp = gm.body_world_pos()
    lo = min(min(me["verts"][2::3]) + wp[b["name"]][2] for b in m["bodies"] for me in b["meshes"])
    hub = max(math.hypot(x, y) for me in m["bodies"][0]["meshes"]
              for x, y in zip(me["verts"][0::3], me["verts"][1::3]))
    check("size", 7.0 <= -lo <= 8.2 and 1.5 <= hub <= 2.0, f"ceiling->lowest {-lo:.2f} m, hub radius {hub:.2f} m")

    s = gs.GladosSim(log=print, seed=7)
    check("joints", len(gs.JN) == s.model.njnt and all(s.limited[1:]) and not s.limited[0],
          f"{s.model.njnt} hinge joints, {s.model.nbody - 1} bodies, mass {s.mass:.0f} kg, yaw unlimited, rest limited")

    # ---------------------------------------------------------------- idle + settle + speed
    t0 = time.perf_counter()
    nan = run(s, 20.0, "idle")
    ms = (time.perf_counter() - t0) / 1200 * 1000
    check("idle 20 s", not nan and s.public_mode() == "idle", f"mode {s.public_mode()}, {ms:.3f} ms per 60 Hz tick "
          f"({ms / 16.67 * 100:.1f}% of real time budget)")
    # settle: track a fixed target, then measure residual motion
    s.command("reset")
    feed_player(s, [6.0, 2.0, -11.4], 8.0, "settle")
    q0 = s.q()
    feed_player(s, [6.0, 2.0, -11.4], 2.0, "settle")
    vmax = float(np.abs(s.data.qvel).max())
    drift = float(np.degrees(np.abs(gs.np.array([gs._wrap(a) for a in s.q() - q0])).max()))
    check("settles", vmax < 0.05 and drift < 5.0, f"after 8 s: max |qvel| {vmax:.4f} rad/s, max joint drift over 2 s "
          f"{drift:.2f} deg (breathing sway only)")

    # ---------------------------------------------------------------- track aim accuracy
    errs = []
    for tgt in ([6.0, 0.0, -11.4], [3.0, 4.0, -11.4], [-4.0, 3.0, -11.4], [1.0, -6.0, -9.0], [8.0, 2.0, -5.0],
                [-6.0, -6.0, -11.4], [2.0, 0.5, -11.4], [0.0, -9.0, -8.0]):
        feed_player(s, tgt, 6.0, f"track {tgt}")
        errs.append((tgt, s.aim_error_deg(tgt), s.public_mode()))
    worst = max(e for _, e, _ in errs)
    check("track aims the eye", worst < 10.0 and all(md == "track" for _, _, md in errs),
          "errors " + ", ".join(f"{t}:{e:.1f}" for t, e, _ in errs) + " deg")

    # ---------------------------------------------------------------- auto track / leave / stale
    s.command("reset")
    feed_player(s, [5.0, 0.0, -11.4], 1.0, "auto")
    m1 = s.public_mode()
    feed_player(s, [25.0, 0.0, -11.4], 1.0, "auto")
    m2 = s.public_mode()
    feed_player(s, [5.0, 0.0, -11.4], 1.0, "auto")
    run(s, 6.0, "stale")
    m3 = s.public_mode()
    s.command("track")
    feed_player(s, [20.0, 0.0, -11.4], 1.0, "forced")
    m4 = s.public_mode()
    feed_player(s, [40.0, 0.0, -11.4], 1.0, "forced")
    m5 = s.public_mode()
    s.set_player(5.0, 0.0, -11.4)
    s.command("idle")
    run(s, 1.0, "idle hold")
    m6 = s.public_mode()
    check("auto track", (m1, m2, m3, m4, m5, m6) == ("track", "idle", "idle", "track", "idle", "idle"),
          f"near {m1}, 25 m {m2}, stale {m3}, forced@20 m {m4}, forced@40 m {m5}, explicit idle near {m6}")

    # ---------------------------------------------------------------- every command
    s.command("reset")
    P = [4.0, -3.0, -11.4]
    cmd_rows = []
    all_ok = True
    for cmd, kw in [("idle", {}), ("track", {}), ("look", {"dir": "left"}), ("look", {"dir": "right"}),
                    ("look", {"dir": "up"}), ("look", {"dir": "down"}), ("look", {"dir": "player"}),
                    ("angry", {}), ("scan", {}), ("nod", {}), ("shake", {}), ("bow", {}), ("sleep", {}),
                    ("wake", {}), ("talk", {"duration": 1.5}), ("talk", {"on": True}), ("talk", {"on": False}),
                    ("reset", {})]:
        s.set_player(*P)
        ok, err = s.command(cmd, **kw)
        q_before = s.q()
        mid_mode = None
        maxmove = 0.0
        dur = gs.ACTION_LEN.get(cmd, 1.5) + 2.0
        nan = False
        for k in range(int(dur * 5)):
            s.set_player(*P)
            nan |= run(s, 0.2, f"cmd {cmd}")
            if k == 2:
                mid_mode = s.public_mode()
            maxmove = max(maxmove, float(np.degrees(np.abs(s.q()[1:] - q_before[1:]).max())))
        end_mode = s.public_mode()
        cmd_rows.append(f"{cmd}{('/' + str(list(kw.values())[0])) if kw else ''}:{mid_mode}->{end_mode}"
                        f"({maxmove:.0f}deg)")
        all_ok &= ok and not nan
        if cmd in gs.ACTION_LEN and cmd != "wake":
            all_ok &= mid_mode == cmd and end_mode == "track" and maxmove > 3.0
    check("every cmd runs", all_ok, "; ".join(cmd_rows))
    bad = [s.command("fly")[0], s.command("look", dir="sideways")[0]]
    check("bad cmds rejected", bad == [False, False], f"fly -> {s.command('fly')[1]!r}")

    # ---------------------------------------------------------------- eye colour: angry / sleep / wake
    s.command("reset")
    feed_player(s, P, 2.0, "eye")
    e0 = s.eye.copy()
    s.command("angry")
    feed_player(s, P, 1.0, "angry")
    e1 = s.eye.copy()
    feed_player(s, P, 4.5, "angry")
    e2 = s.eye.copy()
    check("angry eye red", e0[0] > 0.95 and e0[1] > 0.75 and e1[0] > 0.9 and e1[1] < 0.4 and e1[2] < 0.15 and
          e2[1] > 0.75, f"before {np.round(e0, 2).tolist()}, angry {np.round(e1, 2).tolist()}, "
          f"after {np.round(e2, 2).tolist()}")
    s.command("sleep")
    feed_player(s, P, 4.0, "sleep")
    e3, q3, m3 = s.eye.copy(), s.q(), s.public_mode()
    s.command("wake")
    feed_player(s, P, 3.0, "wake")
    e4, m4 = s.eye.copy(), s.public_mode()
    check("sleep/wake", m3 == "sleep" and e3[3] < 0.3 and math.degrees(q3[gs.JI["neck_y"]]) > 40 and e4[3] > 0.8
          and m4 == "track", f"sleep eye {np.round(e3, 2).tolist()} neck_y {math.degrees(q3[gs.JI['neck_y']]):.0f} deg, "
          f"woken eye {np.round(e4, 2).tolist()} mode {m4}")

    # ---------------------------------------------------------------- talk levels
    s.command("reset")
    lv = []
    for k in range(30):  # 3 s of ~10 Hz levels
        s.command("talk", level=0.9 if k % 4 else 0.2)
        run(s, 0.1, "talk")
        lv.append(s.talk)
    s.command("talk", on=False)
    run(s, 1.0, "talk")
    t_off = s.talk
    s.command("talk", duration=1.0)
    run(s, 0.5, "talk")
    t_dur = s.talk
    run(s, 1.5, "talk")
    check("talk", max(lv) > 0.6 and min(lv[5:]) > 0.1 and t_off < 0.02 and t_dur > 0.1 and s.talk < 0.05,
          f"level-driven talk {min(lv[5:]):.2f}..{max(lv):.2f}, after off {t_off:.3f}, duration mid {t_dur:.2f}, "
          f"after {s.talk:.3f}")

    # ---------------------------------------------------------------- physics sway reacts to the head
    s.command("reset")
    feed_player(s, [6.0, 0.0, -11.4], 4.0, "sway")
    s.command("angry")
    sw = []
    for _ in range(25):
        s.set_player(6.0, 0.0, -11.4)
        run(s, 0.2, "sway")
        sw.append(float(np.degrees(np.abs(s.q()[1:5]).max())))
    feed_player(s, [6.0, 0.0, -11.4], 6.0, "sway")
    rest = float(np.degrees(np.abs(s.data.qvel[s.dadr[1:5]]).max()))
    check("arm sways + settles", max(sw) > 3.0 and rest < 1.0,
          f"arm swing during angry up to {max(sw):.1f} deg, arm joint speed 6 s later {rest:.2f} deg/s")

    # ---------------------------------------------------------------- reset
    s.command("reset")
    run(s, 0.05)
    check("reset", float(np.abs(s.q()).max()) < 0.03 and s.action is None and not s.sleeping,
          f"max |q| {float(np.abs(s.q()).max()):.4f} rad 3 ticks after reset, mode {s.public_mode()}")

    # ---------------------------------------------------------------- expressive tracking: lean + head tilt
    near = [4.0, 3.0, -11.4]
    sims = []
    for lean_on in (True, False):          # same seed, with and without the lean -> isolates its effect
        sl = gs.GladosSim(log=print, seed=9)
        if not lean_on:
            sl._lean = lambda point, ang, _n=len(gs.JN): np.zeros(_n)
        feed_player(sl, near, 7.0, "lean")
        sims.append(sl)
    ni = gs.gm.BODY_ORDER.index("neck")
    dvec = np.array(near[:2]) / np.linalg.norm(near[:2])
    toward = float(dvec @ (sims[0].data.xpos[sims[0].body_ids[ni]][:2] - sims[1].data.xpos[sims[1].body_ids[ni]][:2]))
    tilt = abs(math.degrees(sims[0].tilt))
    aim = sims[0].aim_error_deg(near)
    check("lean + tilt while tracking", toward > 0.15 and aim < 8.0 and (tilt > 4.0 or sims[0].tilt_goal == 0.0),
          f"lean moves the torso bottom {toward:.2f} m toward a player ~5 m away, head tilt {tilt:.1f} deg, "
          f"aim {aim:.1f} deg")

    # ---------------------------------------------------------------- idle: yaw stays put, no high-frequency jitter
    s2 = gs.GladosSim(log=print, seed=5)
    yaws, heads = [], []
    for k in range(120 * 60):        # 2 sim-minutes of idle with nobody around
        s2.step()
        yaws.append(s2.q()[0])
        heads.append(s2.eye_pos())
    yaws, heads = np.array(yaws), np.array(heads)
    hf = heads[2:] - 2 * heads[1:-1] + heads[:-2]          # 2nd difference per 60 Hz tick
    hf_mm = 1000 * float(np.sqrt((np.linalg.norm(hf, axis=1) ** 2).mean()))
    check("idle stays oriented, smooth", float(np.degrees(np.abs(yaws).max())) < 120 and hf_mm < 2.0
          and s2.nan_resets == 0, f"max |yaw| {np.degrees(np.abs(yaws).max()):.0f} deg over 2 min idle, eye 2nd-diff "
          f"RMS {hf_mm:.2f} mm/tick^2")

    # ---------------------------------------------------------------- random stress
    rnd = random.Random(11)
    s.command("reset")
    pl = [5.0, 0.0, -11.4]
    nan = False
    cmds = 0
    for k in range(600):  # 2 sim-minutes
        if k % 25 == 0:
            c = rnd.choice(["idle", "track", "look", "angry", "scan", "nod", "shake", "bow", "sleep", "wake", "talk"])
            s.command(c, dir=rnd.choice(["left", "right", "up", "down", "player"]), duration=rnd.uniform(0.5, 3))
            cmds += 1
        pl = [pl[0] + rnd.uniform(-0.6, 0.6), pl[1] + rnd.uniform(-0.6, 0.6), -11.4]
        pl = [max(-14, min(14, pl[0])), max(-11, min(11, pl[1])), -11.4]
        s.set_player(*pl)
        nan |= run(s, 0.2, "stress")
    check("stress 2 min random cmds", not nan and s.nan_resets == 0,
          f"{cmds} cmds, nan_resets {s.nan_resets}, yaw {math.degrees(gs._wrap(s.q()[0])):.0f} deg")

    # ---------------------------------------------------------------- spooky dance, every model
    r = rss_mb()        # (the service footprint: measured before this test loads the two big v3 assets too)
    import glados_assets as ga
    names = ["procedural"] + [n for n in ga.list_assets() if n != "procedural"]
    for name in names:          # extra models only if you imported your own into assets/
        spooky_checks(name)

    check("joint limits", WORST_LIMIT[0] < 1.5,
          f"worst violation over the whole run {WORST_LIMIT[0]:.2f} deg ({WORST_LIMIT[1]})")
    check("RSS", r < 250, f"{r:.0f} MB")
    n = len(results)
    print(f"\n{sum(results)}/{n} PASS  ({time.time() - t_wall:.0f} s wall)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
