# Omo GLaDOS sidecar

An original, procedural GLaDOS-style ceiling AI (no Valve assets): numpy lathe/sweep geometry -> a hanging MuJoCo
(MjSpec) hinge chain under gravity, PD-driven by a scripted behaviour layer, served on `127.0.0.1:8794` for the
Fabric mod. API: [`CONTRACT.md`](CONTRACT.md).

## Run

```bash
./setup.sh                                          # .venv with mujoco, numpy, pillow + smoke test
OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python server.py
GLADOS_PORT=8799 .venv/bin/python server.py         # another port (e.g. a test instance)

nice -n 19 .venv/bin/python test_headless.py        # in-process checks, no HTTP
nice -n 19 .venv/bin/python test_v3.py              # asset format, sim on every asset, model switching, HTTP side instance
nice -n 19 .venv/bin/python test_http.py http://127.0.0.1:8799   # HTTP checks against a running instance
nice -n 19 .venv/bin/python render_preview.py       # preview/*.png from /model.json + /pose (in-process sim)
.venv/bin/python glados_model.py                    # triangle counts / body positions
.venv/bin/python design_check.py [--flat]           # design pose (all joints 0) -> preview/design_*.png
.venv/bin/python glados_assets.py check             # validate every asset in assets/
.venv/bin/python glados_assets.py export-procedural # regenerate assets/procedural from glados_model.py
```

Point `test_http.py` at a test instance, not the one your game is using: the game posts your real `/player` at
5 Hz, which overrides the test's player and fails the aim checks, and the test ends with `reset`.

If the port is taken the server keeps simulating and retries the bind every 10 s instead of crash-looping.

| Env | Default | |
|---|---|---|
| `GLADOS_PORT` | `8794` | listen port (always 127.0.0.1) |
| `GLADOS_MODEL` | last `model` cmd, else `procedural` | model to start with |
| `GLADOS_STATE` | `~/.local/share/glados-sim/state.json` | where the last chosen model is remembered |
| `GLADOS_ASSETS` | `./assets` | asset library directory |

## Files

| File | |
|---|---|
| `glados_model.py` | procedural geometry (numpy only): `sweep()` (a generalised lathe along a curve with parallel-transport frames, elliptical sections, open sectors with lips for panels, pole collapse, caps), `lathe`, `torus`, `box`, `cylinder`; body builders; `model_json()` |
| `glados_sim.py` | MjSpec model, PD + gravity compensation, look-at IK, behaviour layer, `/pose` payload |
| `server.py` | stdlib `ThreadingHTTPServer`, one sim thread on an absolute 60 Hz schedule, pre-serialised snapshots |
| `glados_assets.py` | asset library: list/load/validate `assets/<name>/`, mass geoms fitted to any asset, `export-procedural` |
| `render.py` / `render_v3.py` | numpy software rasterisers (Lambert / textured with skinning and a shadow map) |
| `render_preview.py` / `design_check.py` | preview PNGs |
| `test_headless.py` / `test_v3.py` / `test_http.py` | tests |
| `smoke.py` | the one-second smoke test `setup.sh` runs |
| `tools/` | `pose_gallery_v3.py`, `spooky_preview.py` (render sheets), `pose_probe.py` (live /pose jitter stats), `asset_info.py`, `uv_debug.py`, `compare.py` (montage) |

## Model

Six bodies, root first: `mount` -> `arm1` -> `arm2` -> `torso` -> `neck` -> `head`. **15,800 triangles**, about
550 kB of JSON, per-vertex normals, one mesh per colour per body, eye = mesh `head_eye` tagged `"eye"` in `head`.
Colours: shell #E8E8E6, mechanics #3A3C40 / #2A2B2E, rings #151515, eye #FFD24A. Served as the `procedural`
asset: a white 16x16 atlas with the colour in `rgba`.

- `mount` (fixed): a big dark ceiling hub (plate r 1.8 m, top face 2 cm below the ceiling so it can't z-fight with
  the ceiling block), a 0.55 m drum, a heavy outer ring, 12 brackets, 5 drooping service cables.
- `arm1` (yaw + swing at the hub, 1.95 m): a bundle of 22 thick black cables twisting together, two clamp bands.
- `arm2` (0.75 m): the dark "shoulder" ring cluster: collar, two big hoops, clamps.
- `torso` (3.1 m): a white inverted shell, broad at the top, curved, with left/right panels in 3 bands whose gaps are
  the seams (the dark core shows through 4+ cm deeper, so there are no thin seam strips to alias), a dark cable
  channel down the front, a ribbed spine (13 vertebrae) down the back.
- `neck`: a thick dark articulated U with 6 collars and side pistons.
- `head`: a 2.0 m horizontal "lamp" helmet, upper/lower white shells split by a dark band, side discs, a black visor
  rim, recessed face dish, lens housing and a glowing domed eye (r 0.14).
- Design pose: ceiling -> lowest point **7.83 m**; eye at (3.48, 0, -7.44) looking 30° below +x.

## Physics

- MjSpec, `implicitfast`, physics dt 1/480 s, 8 substeps per 60 Hz control tick, no contacts (mass geoms are
  `contype=0`). Total mass 2,691 kg (mount fixed).
- 13 hinge joints: `yaw` (unlimited: she can spin round the hub), `arm1_x/y` ±18°, `arm2_x/y` ±22°, `torso_x` ±20°,
  `torso_y` ±25°, `torso_z` ±35°, `neck_y` -55..75°, `neck_z` ±60°, `head_y` -65..70°, `head_z` ±55°, `head_x` ±30°.
- Per-joint PD: kp from the mass-matrix diagonal and a per-group natural frequency (yaw 0.4 Hz, arm 0.45 Hz, torso
  0.6 Hz, neck 1.3 Hz, head 1.8 Hz). The D term is MuJoCo joint damping (integrated implicitly): explicit damping on
  the heavy arm joints chattered at the substep rate. Gravity-only feed-forward (`mj_rne`): 100 % on yaw/neck/head,
  70 % torso, 50 % arm, so the arm still hangs like a pendulum and sways from the head's reaction torques.
- Targets are rate-limited (arm 0.8 rad/s, neck 2.5, head 3.5) and clamped 0.02 rad inside the limits.
- Look-at IK on a kinematic copy: 3 Jacobian-transpose iterations per tick on the eye-axis error over
  `neck_y/neck_z/head_y/head_z`. The mount yaw follows lazily (20° deadband tracking, 50° idle, <= 0.6 rad/s); the
  head does the rest.
- Expressive tracking: within 12 m she leans the whole hanging body toward you (up to ~0.5 m at the torso bottom)
  and picks a curious head tilt whenever you have moved > 1.5 m, so a still player gets a still GLaDOS.

## Behaviour

See [`CONTRACT.md`](CONTRACT.md#commands-post-cmd) for every command. Two worth knowing:

- **`spooky`** (~18.6 s): 0-5.5 s power-down (the eye dims with dropouts, the head sags, the chain swings like a
  pendulum), 5.5-12 s glitch dance (112 BPM, off-beat twitches into held poses, a strobing eye), 12-13.9 s swoop to
  a climax pose solved by whole-chain damped-least-squares IK (eye ~1.5 m from the player's eye), 13.9-15.8 s hold,
  then pull back and track. Safety: a keep-out push if the real eye comes within 1.4 m, near-critical damping while
  reaching. Any command interrupts it.
- **`dance`** (8 s, 112 BPM): head bob per beat, head roll and torso sway per 2/4 beats, the chain circles.

## Adding your own model

The sidecar can serve any model in the v3 asset format ([`CONTRACT.md`](CONTRACT.md#model-format-v3-asset-assetsnameassetjson--atlaspng)):
drop `assets/<name>/asset.json` + `atlas.png`, run `glados_assets.py check`, then `!glados model <name>`. The sim
builds its joint chain from the asset's `joint_pos` and fits mass ellipsoids to each body's vertices; assets with a
longer neck lever get their neck limits scaled down. A broken or missing asset falls back to `procedural`, never
crashing the service.

**No importer or third-party model ships here.** If you bring a model, check its licence first: most GLaDOS models
online are fan art of a Valve character or rips of Valve's own assets, and their licences rarely allow
redistribution. Keep them out of any repo you publish (`.gitignore` already ignores everything in `assets/` except
`procedural/`).

## Limitations

- No collisions: she can't hit the ceiling or the room; the joint limits keep her compact.
- `yaw` is unlimited, so it accumulates turns (reported wrapped as `yaw_deg`); quaternions are unaffected.
- IK aims the eye axis only; head roll is procedural (breathing, jitter, droop), not levelled.
