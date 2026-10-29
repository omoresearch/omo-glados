#!/usr/bin/env python3
"""GLaDOS sim sidecar - HTTP API on 127.0.0.1:8794 (see CONTRACT.md). Same design as the Omo Microduck sidecar."""

import argparse
import json
import math
import os
import sys
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

os.environ.setdefault("OMP_NUM_THREADS", "1")
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)

import glados_assets as ga  # noqa: E402
import glados_model as gm  # noqa: E402
import glados_sim as gs  # noqa: E402

HOST = "127.0.0.1"
PORT = int(os.environ.get("GLADOS_PORT", "8794"))
TWITCH_ALLOWED = {"idle", "track", "angry", "scan", "nod", "shake", "bow", "sleep", "wake", "look", "dance", "spooky"}
TWITCH_MODE_COOLDOWN = {"spooky": 60.0}   # per-mode cooldown for twitch (s), on top of the mod's global 2 s
STATE_FILE = os.environ.get("GLADOS_STATE", os.path.expanduser("~/.local/share/glados-sim/state.json"))
DEFAULT_MODEL = "procedural"


def log(*a):
    print(time.strftime("%H:%M:%S"), *a, flush=True)


def _num(v, name):
    x = float(v)
    if not math.isfinite(x):
        raise ValueError(name)
    return x


def read_state():
    try:
        with open(STATE_FILE) as f:
            s = json.load(f)
        return s if isinstance(s, dict) else {}
    except (OSError, ValueError):
        return {}


def write_state(**kw):
    s = read_state()
    s.update(kw)
    try:
        os.makedirs(os.path.dirname(STATE_FILE), exist_ok=True)
        tmp = STATE_FILE + ".tmp"
        with open(tmp, "w") as f:
            json.dump(s, f, indent=1)
        os.replace(tmp, STATE_FILE)
    except OSError as e:
        log(f"cannot persist state to {STATE_FILE}: {e}")


class Service:
    def __init__(self):
        self.lock = threading.RLock()
        self.switch_lock = threading.Lock()
        self.snap_lock = threading.Lock()
        self.rev = int(time.time()) % 1000000 * 10      # changes across restarts too (mods compare /pose.rev)
        self.assets = {}
        want = os.environ.get("GLADOS_MODEL") or read_state().get("model") or DEFAULT_MODEL
        asset = None
        for name in (want, DEFAULT_MODEL):
            try:
                asset = self._load(name)
                break
            except Exception as e:  # noqa: BLE001 - a broken asset must never take the sidecar down
                log(f"cannot load model {name!r}: {e}; falling back to {DEFAULT_MODEL}")
        self.sim = gs.GladosSim(log=log, asset=asset)
        self._activate(asset)
        self.seq = 0
        self.pose_bytes = b"{}"
        self.health = {}
        self.overruns = 0
        self.step_ms = 0.0
        self.stop = threading.Event()
        self.cmds = 0
        self.twitch_last = {}           # cmd -> monotonic time of the last accepted twitch use (per-mode cooldown)
        self._snapshot()

    def _load(self, name):
        if name not in ga.list_assets() and name != "procedural":
            raise ValueError(f"unknown model {name!r} (have: {', '.join(ga.list_assets())})")
        t = time.time()
        a = ga.load_asset(name)
        log(f"asset {name}: {a.triangles} tris, {len(a.data['meshes'])} meshes, atlas {len(a.atlas_png)} bytes "
            f"({time.time() - t:.2f}s)")
        return a

    def _activate(self, asset):  # with self.lock held (or during __init__)
        self.asset = asset
        self.ntris = asset.triangles
        self.rev += 1
        t = time.time()
        self.model_bytes = json.dumps(asset.model_json(self.rev), separators=(",", ":")).encode()
        self.atlas_bytes = asset.atlas_png
        log(f"active model {asset.name} rev {self.rev}: {len(self.model_bytes)} bytes model.json "
            f"({time.time() - t:.2f}s); mass {self.sim.mass:.0f} kg")

    def models(self):
        out = []
        for n in ga.list_assets():
            path = os.path.join(ga.ASSET_DIR, n, "asset.json")
            try:
                mt = os.path.getmtime(path)
                c = self.assets.get(n)
                if c is None or c[0] != mt:
                    with open(path) as f:
                        a = json.load(f)
                    c = (mt, {"name": n, "title": a.get("title", n), "credit": a.get("credit", ""),
                              "triangles": int(a.get("triangles") or sum(len(m["tris"]) // 3 for m in a["meshes"]))})
                    self.assets[n] = c
            except (OSError, ValueError, KeyError, TypeError):
                continue
            out.append(c[1])
        return {"active": self.asset.name, "rev": self.rev, "models": out}

    def switch_model(self, name):
        """Returns (ok, error). Loads + builds outside the sim lock, swaps atomically, keeps the mode."""
        name = str(name or "").strip().lower()
        with self.switch_lock:
            return self._switch(name)

    def _switch(self, name):
        if name == self.asset.name:
            write_state(model=name)
            return True, None
        try:
            asset = self._load(name)
            new = gs.GladosSim(log=log, asset=asset)
        except Exception as e:  # noqa: BLE001
            return False, str(e)
        with self.lock:
            new.carry_state_from(self.sim)
            self.sim = new
            self._activate(asset)
            self._snapshot()
        write_state(model=name)
        return True, None

    def _snapshot(self):  # called with self.lock held
        s = self.sim
        self.seq += 1
        pose = s.pose_dict(self.seq, rev=self.rev)
        health = {"ok": True, "mode": pose["mode"], "t": pose["t"], "hz": gs.CONTROL_HZ, "seq": self.seq,
                  "talk": pose["talk"], "eye": pose["eye"], "sleeping": s.sleeping,
                  "player": None if s.player is None else [round(float(x), 2) for x in s.player],
                  "player_fresh": s.player_fresh(),
                  "player_dist": None if not s.player_fresh() else round(s.player_dist(), 2),
                  "yaw_deg": round(math.degrees(gs._wrap(float(s.data.qpos[s.qadr[0]]))), 1),
                  "overruns": self.overruns, "step_ms": round(self.step_ms, 3), "resets": s.resets,
                  "nan_resets": s.nan_resets, "cmds": self.cmds, "triangles": self.ntris,
                  "scale_hint": s.scale_hint, "model": self.asset.name, "rev": self.rev}
        b = json.dumps(pose, separators=(",", ":")).encode()
        with self.snap_lock:
            self.pose_bytes = b
            self.health = health

    def run(self):
        dt = gs.CONTROL_DT
        nxt = time.monotonic()
        while not self.stop.is_set():
            t0 = time.perf_counter()
            with self.lock:
                self.sim.step()
                self._snapshot()
            self.step_ms = 0.98 * self.step_ms + 0.02 * (time.perf_counter() - t0) * 1000
            nxt += dt
            now = time.monotonic()
            delay = nxt - now
            if delay > 0:
                time.sleep(delay)
            elif delay < -0.25:  # fell far behind (suspend / CPU starvation): resync, don't spiral
                self.overruns += 1
                nxt = now

    def get_pose(self):
        with self.snap_lock:
            return self.pose_bytes

    def get_health(self):
        with self.snap_lock:
            return dict(self.health)


SERVICE = None


class QuietHTTPServer(ThreadingHTTPServer):
    """A client vanishing mid-request (the game quitting) is normal: one log line instead of a traceback."""

    def handle_error(self, request, client_address):
        import sys
        exc = sys.exc_info()[1]
        if isinstance(exc, (ConnectionResetError, BrokenPipeError, TimeoutError)):
            log(f"client {client_address[0]}:{client_address[1]} went away ({type(exc).__name__})")
            return
        super().handle_error(request, client_address)


class Handler(BaseHTTPRequestHandler):
    server_version = "glados-sim/1.0"
    protocol_version = "HTTP/1.1"

    def log_message(self, fmt, *args):  # /pose is polled at 20+ Hz: keep the journal quiet
        pass

    def _send(self, code, body, ctype="application/json"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, separators=(",", ":")).encode()
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self.end_headers()
        self.wfile.write(body)

    def _json_body(self):
        n = int(self.headers.get("Content-Length") or 0)
        if n > 65536:
            raise ValueError("body too large")
        raw = self.rfile.read(n) if n else b"{}"
        obj = json.loads(raw or b"{}")
        if not isinstance(obj, dict):
            raise ValueError("body must be a JSON object")
        return obj

    def do_GET(self):
        path = self.path.split("?", 1)[0]
        if path == "/health":
            return self._send(200, SERVICE.get_health())
        if path == "/pose":
            return self._send(200, SERVICE.get_pose())
        if path == "/model.json":
            with SERVICE.lock:
                body = SERVICE.model_bytes
            return self._send(200, body)
        if path == "/atlas.png":
            with SERVICE.lock:
                body = SERVICE.atlas_bytes
            return self._send(200, body, "image/png")
        if path == "/models":
            return self._send(200, SERVICE.models())
        return self._send(404, {"ok": False, "error": "not found"})

    def do_POST(self):
        path = self.path.split("?", 1)[0]
        try:
            body = self._json_body()
        except Exception as e:
            return self._send(400, {"ok": False, "error": f"bad json: {e}"})
        if path == "/cmd":
            cmd = str(body.get("cmd", "")).strip().lower()
            cmd = gs.CMD_ALIASES.get(cmd, cmd)
            src = str(body.get("from", "?"))
            if src.startswith("twitch:") and cmd not in TWITCH_ALLOWED:
                return self._send(403, {"ok": False, "error": f"cmd {cmd!r} not allowed from twitch"})
            if src.startswith("twitch:") and cmd in TWITCH_MODE_COOLDOWN:
                now = time.monotonic()
                with SERVICE.lock:
                    left = SERVICE.twitch_last.get(cmd, -1e9) + TWITCH_MODE_COOLDOWN[cmd] - now
                    if left <= 0:
                        SERVICE.twitch_last[cmd] = now
                if left > 0:
                    log(f"cmd {cmd} from={src} -> 429 (twitch cooldown {left:.0f} s)")
                    return self._send(429, {"ok": False, "error": f"{cmd} is cooling down ({left:.0f} s left)",
                                            "retry_after": round(left, 1)})
            if cmd == "model":
                name = body.get("name")
                if not isinstance(name, str) or not name.strip():
                    return self._send(400, {"ok": False, "error": "model needs \"name\" (see GET /models)"})
                t = time.time()
                ok, err = SERVICE.switch_model(name)
                SERVICE.cmds += 1
                log(f"cmd model {name!r} from={src} -> ok={ok} active={SERVICE.asset.name} rev={SERVICE.rev} "
                    f"({time.time() - t:.2f}s){' err=' + err if err else ''}")
                if ok:
                    with SERVICE.lock:
                        mode = SERVICE.sim.public_mode()
                    return self._send(200, {"ok": True, "mode": mode, "model": SERVICE.asset.name,
                                            "rev": SERVICE.rev})
                return self._send(400, {"ok": False, "error": err})
            kw = {}
            try:
                if "dir" in body:
                    kw["dir"] = str(body["dir"]).lower()
                if body.get("level") is not None:
                    kw["level"] = _num(body["level"], "level")
                if body.get("duration") is not None:
                    kw["duration"] = _num(body["duration"], "duration")
                if body.get("on") is not None:
                    kw["on"] = bool(body["on"])
            except (TypeError, ValueError) as e:
                return self._send(400, {"ok": False, "error": f"bad number: {e}"})
            with SERVICE.lock:
                ok, err = SERVICE.sim.command(cmd, **kw)
                mode = SERVICE.sim.public_mode()
            SERVICE.cmds += 1
            if not (cmd == "talk" and set(kw) <= {"level"}):  # level updates arrive ~10 Hz: don't log them
                log(f"cmd {cmd} {kw if kw else ''} from={src} -> ok={ok} mode={mode}{' err=' + err if err else ''}")
            if ok:
                return self._send(200, {"ok": True, "mode": mode})
            return self._send(400, {"ok": False, "error": err})
        if path == "/player":
            try:
                x, y, z = (_num(body[k], k) for k in ("x", "y", "z"))
            except (KeyError, TypeError, ValueError):
                return self._send(400, {"ok": False, "error": "need numeric x, y and z"})
            with SERVICE.lock:
                SERVICE.sim.set_player(x, y, z)
            return self._send(200, {"ok": True})
        if path == "/config":
            sim = SERVICE.sim
            if "scale_hint" in body:
                try:
                    sim.scale_hint = _num(body["scale_hint"], "scale_hint")
                except (TypeError, ValueError):
                    return self._send(400, {"ok": False, "error": "scale_hint must be a number"})
                log(f"config scale_hint={sim.scale_hint} (informational only)")
            return self._send(200, {"ok": True, "scale_hint": sim.scale_hint})
        return self._send(404, {"ok": False, "error": "not found"})


def main():
    global SERVICE
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=PORT)
    args = ap.parse_args()
    SERVICE = Service()
    th = threading.Thread(target=SERVICE.run, name="sim", daemon=True)
    th.start()
    tries = 0
    while True:  # don't crash-loop if the port is taken: keep simulating and retry the bind
        try:
            httpd = QuietHTTPServer((HOST, args.port), Handler)
            break
        except OSError as e:
            if tries % 30 == 0:
                log(f"cannot bind {HOST}:{args.port} ({e}); retrying every 10 s")
            tries += 1
            time.sleep(10)
    httpd.daemon_threads = True
    log(f"listening on http://{HOST}:{args.port} ({gs.CONTROL_HZ} Hz, {gs.SUBSTEPS} substeps)")
    try:
        httpd.serve_forever(poll_interval=0.5)
    except KeyboardInterrupt:
        pass
    finally:
        SERVICE.stop.set()
        httpd.server_close()


if __name__ == "__main__":
    main()
