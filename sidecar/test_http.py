#!/usr/bin/env python3
"""HTTP checks against the running sidecar (default http://127.0.0.1:8794). ~60 s. Exit 1 on failure."""

import json
import math
import os
import subprocess
import sys
import time
import urllib.error
import urllib.request

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import numpy as np  # noqa: E402

import glados_model as gm  # noqa: E402

BASE = sys.argv[1] if len(sys.argv) > 1 else "http://127.0.0.1:8794"
results = []
EYE_SITE = list(gm.eye_site())      # head-local eye (pos, axis); replaced by the v3 model's "eye"


def check(name, ok, detail=""):
    results.append(bool(ok))
    print(f"{'PASS' if ok else 'FAIL'}  {name}: {detail}", flush=True)


def get(path):
    try:
        with urllib.request.urlopen(BASE + path, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def post(path, obj, raw=None):
    data = raw if raw is not None else json.dumps(obj).encode()
    req = urllib.request.Request(BASE + path, data=data, headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, json.loads(r.read())
    except urllib.error.HTTPError as e:
        return e.code, json.loads(e.read())


def quat_mat(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def aim_error(pose, target):
    """Eye aim error computed only from /pose (head transform) + the model's head-local eye site."""
    h = pose["bodies"]["head"]
    R = quat_mat(h[3:7])
    ep, ea = EYE_SITE
    e = np.array(h[:3]) + R @ ep
    f = R @ ea
    d = np.asarray(target, float) - e
    d /= np.linalg.norm(d)
    return math.degrees(math.acos(max(-1, min(1, float(f @ d)))))


def hold_player(p, secs):
    end = time.time() + secs
    while time.time() < end:
        post("/player", dict(zip("xyz", p)))
        time.sleep(0.2)


def proc_stats(pid):
    with open(f"/proc/{pid}/stat") as f:
        parts = f.read().rsplit(")", 1)[1].split()
    ticks = int(parts[11]) + int(parts[12])
    with open(f"/proc/{pid}/status") as f:
        rss = next(int(line.split()[1]) for line in f if line.startswith("VmRSS:")) / 1024
    with open(f"/proc/{pid}/status") as f:
        thr = next(int(line.split()[1]) for line in f if line.startswith("Threads:"))
    return ticks, rss, thr


def main():
    st, h = get("/health")
    check("GET /health", st == 200 and h.get("ok") is True and h.get("hz") == 60 and "mode" in h and "t" in h,
          json.dumps({k: h.get(k) for k in ("ok", "mode", "t", "hz", "overruns", "step_ms")}))
    t0 = time.time()
    st, m = get("/model.json")
    names = [b["name"] for b in m["bodies"]]
    if m.get("version", 1) >= 3:          # v3: top-level meshes with a body name each
        meshes = m["meshes"]
        eye = [(me["body"], me["rgba"]) for me in meshes if me.get("tag") == "eye"]
        EYE_SITE[:] = [np.array(m["eye"]["pos"], float), np.array(m["eye"]["axis"], float)]
    else:
        meshes = [me for b in m["bodies"] for me in b["meshes"]]
        eye = [(b["name"], me["rgba"]) for b in m["bodies"] for me in b["meshes"] if me.get("tag") == "eye"]
    ntris = sum(len(me["tris"]) // 3 for me in meshes)
    idx_ok = all(max(me["tris"]) < len(me["verts"]) // 3 for me in meshes)
    check("GET /model.json", st == 200 and m.get("version") in (1, 3) and names[0] == "mount" and ntris <= 40000
          and idx_ok and len(eye) == 1 and eye[0][0] == "head", f"v{m.get('version')} model {m.get('name')}, "
          f"{len(names)} bodies {names}, {ntris} tris, eye {eye}, {time.time() - t0:.2f}s")

    post("/cmd", {"cmd": "reset", "from": "mc"})
    post("/player", {"x": 80.0, "y": 0.0, "z": -11.4})
    st, p1 = get("/pose")
    tw = time.time()
    time.sleep(2.0)
    st2, p2 = get("/pose")
    dw = time.time() - tw
    rate = (p2["seq"] - p1["seq"]) / dw
    check("GET /pose", st == 200 and set(p2["bodies"]) == set(names) and len(p2["eye"]) == 4 and "talk" in p2
          and all(len(v) == 7 for v in p2["bodies"].values()) and 55 <= rate <= 62
          and abs((p2["t"] - p1["t"]) / dw - 1) < 0.05, f"{rate:.1f} snapshots/s, sim/wall {(p2['t'] - p1['t']) / dw:.3f},"
          f" head {p2['bodies']['head']}")
    q = p2["bodies"]["head"][3:7]
    check("pose quats unit + origin", abs(sum(x * x for x in q) - 1) < 1e-3 and p2["bodies"]["mount"][:3] == [0, 0, 0],
          f"mount {p2['bodies']['mount']}")

    # track a low target (player standing on the lab floor)
    tgt = [4.0, -2.5, -11.4]
    hold_player(tgt, 5.0)
    st, p = get("/pose")
    err = aim_error(p, tgt)
    check("auto-track aims eye (from /pose)", p["mode"] == "track" and err < 10, f"mode {p['mode']}, aim {err:.1f} deg")
    tgt2 = [-3.0, 5.0, -11.4]
    hold_player(tgt2, 6.0)
    st, p = get("/pose")
    err2 = aim_error(p, tgt2)
    check("track behind/side", err2 < 10, f"target {tgt2}: aim {err2:.1f} deg")

    # every command
    rows = []
    ok_all = True
    for c, extra in [("idle", {}), ("track", {}), ("look", {"dir": "left"}), ("look", {"dir": "right"}),
                     ("look", {"dir": "up"}), ("look", {"dir": "down"}), ("look", {"dir": "player"}), ("angry", {}),
                     ("scan", {}), ("nod", {}), ("shake", {}), ("bow", {}), ("sleep", {}), ("wake", {}),
                     ("talk", {"duration": 1.0}), ("talk", {"on": True}), ("talk", {"on": False}), ("reset", {})]:
        st, r = post("/cmd", {"cmd": c, "from": "mc", **extra})
        ok_all &= st == 200 and r.get("ok") is True
        rows.append(f"{c}{'/' + str(list(extra.values())[0]) if extra else ''}={st}:{r.get('mode')}")
        hold_player(tgt, 0.4)
    check("POST /cmd all commands", ok_all, ", ".join(rows))

    st, r = post("/cmd", {"cmd": "angry", "from": "mc"})
    hold_player(tgt, 1.0)
    st, p = get("/pose")
    e = p["eye"]
    check("angry: eye red-orange", p["mode"] == "angry" and e[0] > 0.9 and e[1] < 0.45 and e[2] < 0.2,
          f"mode {p['mode']} eye {e}")
    hold_player(tgt, 4.0)
    st, p = get("/pose")
    check("angry ends", p["mode"] == "track" and p["eye"][1] > 0.7, f"mode {p['mode']} eye {p['eye']}")
    post("/cmd", {"cmd": "sleep", "from": "voice"})
    hold_player(tgt, 3.5)
    st, p = get("/pose")
    check("sleep: eye dim", p["mode"] == "sleep" and p["eye"][3] < 0.3, f"eye {p['eye']}")
    post("/cmd", {"cmd": "wake", "from": "voice"})
    hold_player(tgt, 3.0)
    st, p = get("/pose")
    check("wake", p["mode"] == "track" and p["eye"][3] > 0.8, f"mode {p['mode']} eye {p['eye']}")

    # talk levels from the voice backend (~10 Hz)
    lv = []
    for k in range(25):
        post("/cmd", {"cmd": "talk", "level": 0.9 if k % 3 else 0.3, "from": "voice"})
        time.sleep(0.1)
        lv.append(get("/pose")[1]["talk"])
    post("/cmd", {"cmd": "talk", "on": False, "from": "voice"})
    time.sleep(1.0)
    off = get("/pose")[1]["talk"]
    check("talk levels", max(lv) > 0.6 and off < 0.05, f"talk {min(lv[5:]):.2f}..{max(lv):.2f}, after off {off:.3f}")

    # errors / permissions
    codes = {
        "unknown cmd": post("/cmd", {"cmd": "fly"})[0],
        "look bad dir": post("/cmd", {"cmd": "look", "dir": "sideways"})[0],
        "twitch reset": post("/cmd", {"cmd": "reset", "from": "twitch:someone"})[0],
        "twitch talk": post("/cmd", {"cmd": "talk", "from": "twitch:someone"})[0],
        "twitch angry": post("/cmd", {"cmd": "angry", "from": "twitch:someone"})[0],
        "twitch look": post("/cmd", {"cmd": "look", "dir": "up", "from": "twitch:someone"})[0],
        "bad json": post("/cmd", None, raw=b"{nope")[0],
        "player missing z": post("/player", {"x": 1, "y": 2})[0],
        "player NaN": post("/player", None, raw=b'{"x":NaN,"y":0,"z":0}')[0],
        "config": post("/config", {"scale_hint": 1.0})[0],
        "404": get("/nope")[0],
    }
    want = {"unknown cmd": 400, "look bad dir": 400, "twitch reset": 403, "twitch talk": 403, "twitch angry": 200,
            "twitch look": 200, "bad json": 400, "player missing z": 400, "player NaN": 400, "config": 200, "404": 404}
    check("errors + twitch perms", codes == want, json.dumps(codes))

    # spooky dance: aliases, twitch per-mode cooldown (60 s), interruptible
    sp = {
        "twitch spooky": post("/cmd", {"cmd": "spooky", "from": "twitch:someone"}),
        "twitch spooky again": post("/cmd", {"cmd": "spooky", "from": "twitch:other"}),
        "twitch alias spookydance": post("/cmd", {"cmd": "spookydance", "from": "twitch:other"}),
        "mc alias creepy": post("/cmd", {"cmd": "creepy", "from": "mc"}),
    }
    hold_player(tgt, 0.6)
    mid = get("/pose")[1]["mode"]
    nod = post("/cmd", {"cmd": "nod", "from": "mc"})
    got = {k: (v[0], v[1].get("mode") or v[1].get("error")) for k, v in sp.items()}
    check("spooky: aliases + twitch 60 s cooldown + nod interrupts",
          [v[0] for v in sp.values()] == [200, 429, 429, 200] and sp["mc alias creepy"][1]["mode"] == "spooky"
          and mid == "spooky" and nod[0] == 200 and nod[1]["mode"] == "nod", f"{json.dumps(got)}, mid {mid}, "
          f"nod -> {nod[1].get('mode')}")

    # leave -> idle
    hold_player([40.0, 0.0, -11.4], 2.5)  # (lets the twitch `look up` above finish)
    check("player leaves -> idle", get("/health")[1]["mode"] == "idle", get("/health")[1]["mode"])

    # bind address
    ss = subprocess.run(["ss", "-ltnH"], capture_output=True, text=True).stdout
    port = BASE.rstrip("/").rsplit(":", 1)[1]      # works against a side instance too (GLADOS_PORT=8799)
    binds = [line.split()[3] for line in ss.splitlines() if line.split()[3].endswith(":" + port)]
    check("bound to localhost only", binds == ["127.0.0.1:" + port], str(binds))

    # CPU / RSS of the service process
    pid = subprocess.run(["systemctl", "--user", "show", "-p", "MainPID", "--value", "glados-sim"],
                         capture_output=True, text=True).stdout.strip()
    if pid and pid != "0":
        hz = os.sysconf("SC_CLK_TCK")
        t1, rss, thr = proc_stats(pid)
        time.sleep(10)
        t2, rss, thr = proc_stats(pid)
        cpu = (t2 - t1) / hz / 10 * 100
        nice = int(open(f"/proc/{pid}/stat").read().rsplit(")", 1)[1].split()[16])
        env = open(f"/proc/{pid}/environ", "rb").read().split(b"\0")
        omp = [x.decode() for x in env if x.startswith(b"OMP_NUM_THREADS")]
        check("service CPU/RSS/nice", cpu < 25 and rss < 250 and nice == 10 and omp == ["OMP_NUM_THREADS=1"],
              f"pid {pid}: CPU {cpu:.1f}% of one core (idle mode), RSS {rss:.0f} MB, threads {thr}, nice {nice}, {omp}")
    else:
        check("service CPU/RSS/nice", False, "glados-sim service not running")
    h = get("/health")[1]
    check("real time", h["overruns"] == 0, f"overruns {h['overruns']}, step {h['step_ms']} ms of 16.7 ms")
    post("/cmd", {"cmd": "reset", "from": "mc"})
    print(f"\n{sum(results)}/{len(results)} PASS")
    sys.exit(0 if all(results) else 1)


if __name__ == "__main__":
    main()
