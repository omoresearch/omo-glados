#!/usr/bin/env python3
"""CONTRACT v3 checks: assets (format, size, frame, tags, atlas), the sim built from an asset's joints (track,
every cmd incl. dance, limits, stability), live model switching in-process, and the HTTP additions against a side
instance (own port + own state file, never the user's service). PASS/FAIL summary; exit 1 on any failure.

    python3 test_v3.py [asset ...]          (default: every asset in assets/ except procedural for the sim part)
"""

import io
import json
import math
import os
import random
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402
from PIL import Image  # noqa: E402

def _ensure_second_model():
    """The model-switch checks need two models. A clean clone ships only `procedural`, so unless you imported your
    own model into assets/ (or set GLADOS_ASSETS), run against a temp asset dir holding procedural + a renamed copy."""
    if os.environ.get("GLADOS_ASSETS"):
        return
    src = os.path.join(HERE, "assets")
    names = [d for d in os.listdir(src) if os.path.isfile(os.path.join(src, d, "asset.json"))]
    if any(n != "procedural" for n in names):
        return
    import shutil
    tmp = tempfile.mkdtemp(prefix="glados-assets-", dir=os.environ.get("TMPDIR"))
    for name in ("procedural", "procedural-copy"):
        shutil.copytree(os.path.join(src, "procedural"), os.path.join(tmp, name))
    path = os.path.join(tmp, "procedural-copy", "asset.json")
    with open(path) as f:
        d = json.load(f)
    d["name"], d["title"] = "procedural-copy", "Test copy of the procedural model"
    with open(path, "w") as f:
        json.dump(d, f)
    os.environ["GLADOS_ASSETS"] = tmp
    import atexit
    atexit.register(shutil.rmtree, tmp, True)
    print(f"only the procedural model is installed: model-switch checks use a temp copy ({tmp})", flush=True)


_ensure_second_model()

import glados_assets as ga  # noqa: E402
import glados_sim as gs  # noqa: E402

OTHER = next((n for n in ga.list_assets() if n != "procedural"), "procedural")

PY = sys.executable
results = []
WORST = [0.0, ""]


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def run(s, secs, tag=""):
    nan = False
    for _ in range(int(round(secs * 60))):
        s.step()
        nan |= not np.all(np.isfinite(s.data.qpos))
        v = s.limit_violation_deg()
        if v > WORST[0]:
            WORST[:] = [v, f"{tag} t={s.t:.1f}"]
    return nan


def feed(s, p, secs, tag=""):
    nan = False
    for _ in range(int(secs * 5)):
        s.set_player(*p)
        nan |= run(s, 0.2, tag)
    return nan


def world_points(a):
    """All verts of an asset at its rest pose in the sim frame."""
    out = []
    for m in a.data["meshes"]:
        V = np.asarray(m["verts"], float).reshape(-1, 3)
        if not m.get("skin"):
            p, q = a.rest(m["body"])
            V = V @ ga.quat_mat(q).T + p
        out.append(V)
    return np.vstack(out)


# ------------------------------------------------------------------------------------------- assets
def test_asset(name):
    a = ga.load_asset(name)                       # raises on validation problems
    d = a.data
    probs = ga.validate(d)
    P = world_points(a)
    low = -float(P[:, 2].min())
    top = float(P[:, 2].max())
    mount = np.vstack([np.asarray(m["verts"], float).reshape(-1, 3) for m in d["meshes"]
                       if m["body"] == "mount" and not m.get("skin")])
    hub_c = np.hypot(*((mount[:, :2].max(0) + mount[:, :2].min(0)) / 2))
    ax = np.asarray(d["eye"]["axis"], float)
    ax /= np.linalg.norm(ax)
    check(f"[{name}] asset format", not probs and a.triangles <= 40000 and
          [b["name"] for b in d["bodies"]] == ga.BODY_ORDER,
          f"{a.triangles} tris, {len(d['meshes'])} meshes, problems {probs[:3]}")
    check(f"[{name}] size + frame", 7.5 <= low <= 8.0 and top <= 0.01 and hub_c < 0.15 and
          d["bodies"][0]["rest"][:3] == [0, 0, 0] and ax[0] > 0.85,
          f"ceiling->lowest {low:.2f} m, top {top:.3f}, hub centre off-axis {hub_c:.3f} m, eye axis "
          f"{np.round(ax, 3).tolist()}")
    img = Image.open(io.BytesIO(a.atlas_png))
    tags = {}
    for m in d["meshes"]:
        tags[m.get("tag")] = tags.get(m.get("tag"), 0) + len(m["tris"]) // 3
    eye_bodies = {m["body"] for m in d["meshes"] if m.get("tag") == "eye"}
    check(f"[{name}] atlas + tags", img.format == "PNG" and img.mode == "RGBA" and max(img.size) <= 4096 and
          eye_bodies == {"head"}, f"atlas {img.size} {img.mode}, tris per tag {tags}")
    skins = [m for m in d["meshes"] if m.get("skin")]
    sk_ok = all(0 <= min(m["skin"]["w"]) and max(m["skin"]["w"]) <= 1 for m in skins)
    check(f"[{name}] skinned meshes", sk_ok, f"{len(skins)} skinned meshes: " +
          ", ".join(f"{m['name']}->{m['skin']['bodies']}" for m in skins))
    return a


# ------------------------------------------------------------------------------------------- sim on an asset
def test_sim(a):
    name = a.name
    s = gs.GladosSim(log=lambda *x: None, seed=3, asset=a)
    ok = True
    worst = 0.0
    for b, bid in zip(s.body_names, s.body_ids):
        p, q = a.rest(b)
        worst = max(worst, float(np.abs(s.data.xpos[bid] - p).max()))
    ep = a.rest("head")[0] + ga.quat_mat(a.rest("head")[1]) @ np.array(a.data["eye"]["pos"])
    eye_err = float(np.abs(s.eye_pos() - ep).max())
    ok = worst < 1e-6 and eye_err < 1e-6
    check(f"[{name}] chain from asset joints", ok and s.model.njnt == len(gs.JN),
          f"rest body pos max err {worst:.2e} m, eye site err {eye_err:.2e} m, {s.model.njnt} joints, "
          f"mass {s.mass:.0f} kg")
    t0 = time.perf_counter()
    nan = run(s, 10.0, "idle")
    ms = (time.perf_counter() - t0) / 600 * 1000
    check(f"[{name}] idle 10 s", not nan and s.public_mode() == "idle", f"{ms:.3f} ms per 60 Hz tick")
    s.command("reset")
    feed(s, [6.0, 2.0, -11.4], 8.0, "settle")
    vmax = float(np.abs(s.data.qvel).max())
    check(f"[{name}] settles", vmax < 0.06, f"max |qvel| {vmax:.4f} rad/s after 8 s tracking a fixed target")
    errs = []
    for tgt in ([6.0, 0.0, -11.4], [3.0, 4.0, -11.4], [-4.0, 3.0, -11.4], [1.0, -6.0, -9.0], [8.0, 2.0, -5.0],
                [-6.0, -6.0, -11.4], [0.0, -9.0, -8.0]):
        feed(s, tgt, 6.0, f"track {tgt}")
        errs.append(s.aim_error_deg(tgt))
    check(f"[{name}] track aims the eye", max(errs) < 10.0, "errors " + ", ".join(f"{e:.1f}" for e in errs) + " deg")
    s.command("reset")
    P = [4.0, -3.0, -11.4]
    rows, all_ok = [], True
    for cmd, kw in [("look", {"dir": "left"}), ("angry", {}), ("scan", {}), ("nod", {}), ("shake", {}), ("bow", {}),
                    ("dance", {}), ("spooky", {}), ("sleep", {}), ("wake", {}), ("talk", {"duration": 1.5}), ("reset", {})]:
        s.set_player(*P)
        okc, err = s.command(cmd, **kw)
        q0 = s.q()
        mid, mv, nan = None, 0.0, False
        for k in range(int((gs.ACTION_LEN.get(cmd, 1.5) + 2.0) * 5)):
            s.set_player(*P)
            nan |= run(s, 0.2, f"cmd {cmd}")
            if k == 2:
                mid = s.public_mode()
            mv = max(mv, float(np.degrees(np.abs(s.q()[1:] - q0[1:]).max())))
        end = s.public_mode()
        rows.append(f"{cmd}:{mid}->{end}({mv:.0f}deg)")
        all_ok &= okc and not nan
        if cmd in gs.ACTION_LEN and cmd != "wake":
            all_ok &= mid == cmd and end == "track" and mv > 3.0
    check(f"[{name}] every cmd runs", all_ok, "; ".join(rows))
    # stress
    rnd = random.Random(5)
    s.command("reset")
    pl = [5.0, 0.0, -11.4]
    nan = False
    for k in range(300):
        if k % 25 == 0:
            s.command(rnd.choice(["idle", "track", "look", "angry", "scan", "nod", "shake", "bow", "sleep", "wake",
                                  "talk", "dance", "spooky"]), dir=rnd.choice(["left", "right", "up", "down", "player"]))
        pl = [max(-14, min(14, pl[0] + rnd.uniform(-0.6, 0.6))), max(-11, min(11, pl[1] + rnd.uniform(-0.6, 0.6))),
              -11.4]
        s.set_player(*pl)
        nan |= run(s, 0.2, "stress")
    check(f"[{name}] stress 1 min random cmds", not nan and s.nan_resets == 0, f"nan_resets {s.nan_resets}")


def test_dance(a):
    s = gs.GladosSim(log=lambda *x: None, seed=4, asset=a)
    P = [5.0, 1.0, -11.4]
    feed(s, P, 4.0, "pre-dance")
    s.command("dance")
    hy, modes, eye = [], [], []
    for k in range(int(8.0 * 60)):
        if k % 12 == 0:
            s.set_player(*P)
        run(s, 1 / 60, "dance")
        hy.append(s.q()[gs.JI["head_y"]])
        modes.append(s.public_mode())
        eye.append(s.eye[3])
    feed(s, P, 2.0, "post-dance")
    x = np.array(hy[60:420]) - np.mean(hy[60:420])
    spec = np.abs(np.fft.rfft(x * np.hanning(len(x))))
    fr = np.fft.rfftfreq(len(x), 1 / 60)
    band = (fr > 0.5) & (fr < 4)
    peak = float(fr[band][spec[band].argmax()])
    amp = float(np.degrees(x.max() - x.min()))
    beat = gs.DANCE_BPM / 60
    check(f"[{a.name if a else 'procedural'}] dance", all(m == "dance" for m in modes[:470]) and
          s.public_mode() == "track" and abs(peak - beat) < 0.35 and amp > 8 and max(eye) - min(eye) > 0.1,
          f"head bob {amp:.0f} deg p-p at {peak:.2f} Hz (beat {beat:.2f} Hz), eye glow pulse "
          f"{min(eye):.2f}..{max(eye):.2f}, after: {s.public_mode()}")


# ------------------------------------------------------------------------------------------- service in-process
def test_service_inproc(state_path):
    os.environ["GLADOS_STATE"] = state_path
    os.environ.pop("GLADOS_MODEL", None)
    import importlib
    import server
    importlib.reload(server)
    server.log = lambda *x: None
    svc = server.Service()
    a0, r0 = svc.asset.name, svc.rev
    with svc.lock:
        svc.sim.command("sleep")
        for _ in range(120):
            svc.sim.step()
    other = OTHER if a0 != OTHER else "procedural"
    t = time.time()
    ok1, _ = svc.switch_model(other)
    dt = time.time() - t
    mode = svc.sim.public_mode()
    with svc.lock:
        for _ in range(60):
            svc.sim.step()
    m = json.loads(svc.model_bytes)
    pose = json.loads(svc.pose_bytes)
    ok2, err = svc.switch_model("nope")
    st = json.load(open(state_path))
    check("model switch in-process", ok1 and svc.asset.name == other and svc.rev == r0 + 1 and mode == "sleep" and
          m["name"] == other and m["rev"] == svc.rev and m["atlas"] == f"/atlas.png?rev={svc.rev}" and
          pose["rev"] == svc.rev and not ok2 and st.get("model") == other,
          f"{a0}(rev {r0}) -> {other}(rev {svc.rev}) in {dt:.2f}s, mode kept {mode}, unknown -> {err!r}, "
          f"state {st}")


# ------------------------------------------------------------------------------------------- HTTP side instance
def http(base, path, obj=None):
    req = urllib.request.Request(base + path, data=None if obj is None else json.dumps(obj).encode(),
                                 headers={"Content-Type": "application/json"}, method="GET" if obj is None else "POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as r:
            return r.status, r.headers.get("Content-Type"), r.read()
    except urllib.error.HTTPError as e:
        return e.code, e.headers.get("Content-Type"), e.read()


def start_side(port, state_path, model=None):
    env = dict(os.environ, GLADOS_PORT=str(port), GLADOS_STATE=state_path, OMP_NUM_THREADS="1")
    env.pop("GLADOS_MODEL", None)
    if model:
        env["GLADOS_MODEL"] = model
    p = subprocess.Popen([PY, os.path.join(HERE, "server.py")], env=env, stdout=subprocess.DEVNULL,
                         stderr=subprocess.DEVNULL)
    base = f"http://127.0.0.1:{port}"
    for _ in range(150):
        time.sleep(0.2)
        try:
            if http(base, "/health")[0] == 200:
                return p, base
        except OSError:
            pass
    p.terminate()
    raise RuntimeError("side instance did not start")


def free_port():
    import socket
    with socket.socket() as s:
        s.bind(("127.0.0.1", 0))
        return s.getsockname()[1]


def test_http_side(state_path, port=None):
    port = port or free_port()
    with open(state_path, "w") as f:
        json.dump({"model": "procedural"}, f)
    p, base = start_side(port, state_path)
    try:
        h = json.loads(http(base, "/health")[2])
        st, ct, b = http(base, "/models")
        ms = json.loads(b)
        check("GET /models", st == 200 and ms["active"] == "procedural" and h["model"] == "procedural" and
              {m["name"] for m in ms["models"]} >= {"procedural", OTHER} and
              all({"name", "title", "credit", "triangles"} <= set(m) for m in ms["models"]),
              json.dumps([(m["name"], m["triangles"]) for m in ms["models"]]))
        rev0 = json.loads(http(base, "/pose")[2])["rev"]
        codes = {
            "twitch model": http(base, "/cmd", {"cmd": "model", "name": OTHER, "from": "twitch:x"})[0],
            "model no name": http(base, "/cmd", {"cmd": "model", "from": "mc"})[0],
            "model unknown": http(base, "/cmd", {"cmd": "model", "name": "nope", "from": "mc"})[0],
            "twitch dance": http(base, "/cmd", {"cmd": "dance", "from": "twitch:x"})[0],
        }
        check("model cmd errors / twitch 403", codes == {"twitch model": 403, "model no name": 400,
                                                         "model unknown": 400, "twitch dance": 200}, json.dumps(codes))
        t = time.time()
        st, _, b = http(base, "/cmd", {"cmd": "model", "name": OTHER, "from": "mc"})
        r = json.loads(b)
        dt = time.time() - t
        pose = json.loads(http(base, "/pose")[2])
        st2, ct2, mb = http(base, "/model.json")
        m = json.loads(mb)
        st3, ct3, png = http(base, "/atlas.png?rev=" + str(m["rev"]))
        img = Image.open(io.BytesIO(png))
        check(f"switch -> {OTHER} over HTTP", st == 200 and r["model"] == OTHER and r["mode"] == "dance" and
              pose["rev"] == r["rev"] != rev0 and m["version"] == 3 and m["name"] == OTHER and
              m["rev"] == r["rev"] and m["atlas"] == f"/atlas.png?rev={r['rev']}" and "meshes" in m and
              st3 == 200 and ct3 == "image/png" and png[:8] == b"\x89PNG\r\n\x1a\n",
              f"{dt:.2f}s, rev {rev0}->{r['rev']}, mode kept {r['mode']}, model.json {len(mb) / 1e6:.1f} MB "
              f"{m['triangles']} tris, atlas {img.size}")
    finally:
        p.terminate()
        p.wait(10)
    p, base = start_side(port, state_path)          # persistence: comes back as OTHER
    try:
        h = json.loads(http(base, "/health")[2])
        check("model persists across restart", h["model"] == OTHER and
              json.load(open(state_path))["model"] == OTHER, f"after restart: {h['model']}, rev {h['rev']}")
    finally:
        p.terminate()
        p.wait(10)
    p, base = start_side(port, state_path, model="procedural")   # env GLADOS_MODEL wins
    try:
        h = json.loads(http(base, "/health")[2])
        check("GLADOS_MODEL env overrides", h["model"] == "procedural", h["model"])
    finally:
        p.terminate()
        p.wait(10)


def main():
    t_wall = time.time()
    names = sys.argv[1:] or ga.list_assets()
    assets = {n: test_asset(n) for n in names}
    for n, a in assets.items():
        if n != "procedural":
            test_sim(a)
            test_dance(a)
    test_dance(None)
    check("joint limits", WORST[0] < 1.5, f"worst violation {WORST[0]:.2f} deg ({WORST[1]})")
    with tempfile.TemporaryDirectory(dir=os.environ.get("TMPDIR")) as td:
        test_service_inproc(os.path.join(td, "state_inproc.json"))
        test_http_side(os.path.join(td, "state_http.json"))
    print(f"\n{sum(results)}/{len(results)} PASS  ({time.time() - t_wall:.0f} s wall)")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
