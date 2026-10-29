# Omo GLaDOS client mod (Minecraft 26.2, Fabric)

Renders the sidecar's active model (by default the original procedural GLaDOS-style ceiling AI) hanging
articulated from a ceiling, from the local sidecar ([`../sidecar`](../sidecar), HTTP `127.0.0.1:8794`). It sends
the sidecar your eye position, turns `!glados ...` chat lines into sidecar commands (client-side only, never sent to
the server) and, with Simple Voice Chat installed and a voice backend of your own running, lets you talk to her:
your push-to-talk goes to the backend and her reply plays from her head. Contract:
[`../sidecar/CONTRACT.md`](../sidecar/CONTRACT.md). Coexists with Omo Microduck: each mod intercepts only its own
prefix.

## Build / test

```
python3 deps/fetch.py               # optional: Simple Voice Chat API + jar into deps/ (hash-checked; voice plugin + voicetest)
./build.sh                          # javac against the unobfuscated 26.2 client jar + fabric-api nested jars (+ SVC API)
./build.sh test [URL...]            # + harness (test/, not in the jar) against each URL (default http://127.0.0.1:8794)
python3 test/fake_sidecar.py 8789   # stand-in v3 sidecar: a 6-link box chain, models "boxes" (atlas, uv,
                                    #   eye/emissive/glass tags, skinned cable) + "procedural" (v1-style), /models, rev
python3 test/fake_sidecar.py 8788 --v1   # the pre-v3 API exactly (backwards-compat run)
./build.sh test http://127.0.0.1:8789 http://127.0.0.1:8788
./build.sh voicetest                # PCM framing, voice config, mic-focus arbitration, Opus round trip (Concentus + native)
./build.sh voicetest ws://127.0.0.1:8793/svc a.pcm b.pcm   # + live run against your backend (PCM16 LE 48k mono)
python3 test/check_refs.py          # javap: every MC/Fabric/JOML/Gson/SVC member the jar calls exists in the real jars
./build.sh install                  # atomic copy into $MC_MODS (loads on the next launch)
```

Paths come from env vars (`JDK_BIN` or `JAVA_HOME`, `MC_LIBRARIES` or `PRISM`, `MC_MODS`, `FABRIC_API_JAR`); see the
top of [`build.sh`](build.sh). Without `deps/` the mod builds without the voice plugin. SVC is a `suggests`, so the
mod loads fine without it.

## Chat commands (MC chat; intercepted; replies as `[GLaDOS] ...`)

| command | effect |
|---|---|
| `!glados` / `!glados help` | help + the active model's credit line |
| `!glados idle\|track\|angry\|scan\|nod\|shake\|bow\|dance\|spooky\|sleep\|wake\|reset` | `POST /cmd {"cmd":..., "from":"mc"}` (`spookydance` and `creepy` = `spooky`) |
| `!glados look left\|right\|up\|down\|me` | `POST /cmd {"cmd":"look","dir":...}` (`me` -> `"player"`) |
| `!glados here` | scan straight up from your head to the first block with a collision shape; anchor = its underside (a top slab works), x/z = column centre, + dimension; saved. Warns if the room is lower than her height |
| `!glados scale <0.25-4>` | MC blocks per sim metre (she is ~7.5 blocks tall at 1); saved; also `POST /config {"scale_hint":s}` |
| `!glados yaw <deg>` | turns the whole model about the anchor's vertical axis (render and player->sim); saved |
| `!glados models` | `GET /models`: one line per model (`*` = active) with title, triangles and credit |
| `!glados model <name>` | switch the sidecar's model live; the mod refetches on the new `/pose.rev` |

Every line starting with `!glados` is swallowed, also without the space: `!gladoshere`, `!gladosdance` work (the
longest known command word is split off); unknown ones get a usage reply instead of leaking to the server chat.

## Config

`config/glados.properties` in your instance: `anchor.x/y/z` (MC coords of the ceiling mount, default
`26.5 / 33.0 / 271.5`: the ceiling of the Omo Space robot lab, so in your world use `!glados here`), `dim`
(`minecraft:overworld`), `scale` (1.0), `yaw` (0), `port` (8794), `enabled`. Written with defaults on first launch.

`config/glados-voice.properties`: `enabled`, `url` (`ws://127.0.0.1:8793/svc`, loopback only), `hear_range` (16, or
`always`), `player` (blank = you), `voice_distance` (32), `captions`, `head_body` (`head`: the `/model.json` body
whose mesh centre is her voice position and focus target), `look_angle` (30). With no backend running, the link
just retries quietly.

## How it works

* **Network** (`SidecarClient`, one low-priority daemon thread): `/model.json` + its atlas PNG (IHDR-checked
  <= 4096 px) on connect (retry 1 -> 15 s backoff) and again whenever `/pose.rev` differs from the model's `rev` (a
  live model switch; the old model stays on screen until the new one is complete), `/pose` at ~25 Hz,
  `POST /player` = your eye position in sim metres at ~5 Hz while you are in her dimension. Render/tick threads only
  read a volatile snapshot. Sidecar down -> one log line, she disappears, reconnects quietly.
* **Mapping** (`GladosMath`): MC = A + Ry(yaw)·s·C·p, C(x,y,z) = (x,z,-y); rotations Ry(yaw)·C R C^T; player eye ->
  sim = C^T·Ry(-yaw)·(P-A)/s.
* **Model** (`GladosModel`): v1/v2 meshes nested in `bodies[].meshes` and v3 top-level `meshes[]`. Baked once into
  indexed flat arrays per body and part (tag x textured): per-vertex normals (zero/NaN repaired from the adjacent
  faces), `uv` into the atlas, ARGB per vertex from `rgba`. Tags: `eye` (fullbright, tinted by `pose.eye`, pulses
  with `pose.talk`), `emissive` (fullbright), `glass` (translucent, drawn after the cutout parts). Skinned meshes
  are blended per vertex at draw time. The harness measures emitting 40k triangles at ~0.5 ms per frame.
* **Atlas**: decoded off-thread, registered on the client thread as a `DynamicTexture`
  (`glados:dynamic/atlas_<rev>`); the previous atlas is released.
* **Rendering**: `LevelRenderEvents.COLLECT_SUBMITS` -> per body and part `submitCustomGeometry` with
  `RenderTypes.entityCutout(atlas)` and `entityTranslucent(...)` for glass (Sodium/Iris-safe). Per-body light = max
  of the body origin's block and the one below it. Culling uses a bounding box over every body origin padded by the
  largest body extent (she is ~7.5 blocks tall), within 96 blocks of the camera.
* **Voice** (`GladosVoicePlugin`, SVC entrypoint `voicechat`): her reply goes Opus-encoded in 20 ms frames on a
  `LocationalAudioChannel` at her head's mesh centre, which follows the head as she sways (volume category
  "GLaDOS"). Mic packets are forwarded only while GLaDOS owns mic focus.
* **Mic focus** (`GladosFocus`): she owns the mic if your eye is within `hear_range` of her head AND (you look within
  `look_angle` of her head OR you are closer to her head than to an Omo Microduck, found by reflection). While she
  owns it she sets `System.setProperty("omo.voice.focus", "glados:" + now)` every tick; Omo Microduck's plugin
  ignores the mic while that property says `glados` and is < 1 s old.

## In-game checklist

1. `logs/latest.log`: `glados: sidecar http://127.0.0.1:8794, ...`, then `glados: sidecar ... connected (model: 6
   bodies, 15800 triangles ...)`. With SVC: `glados voice: Simple Voice Chat plugin loaded`.
2. Stand under a ceiling at least ~9 blocks high and type `!glados here`. She hangs ~7.5 blocks down; the shells are
   white, the mechanics dark and the eye glows yellow even in the dark. Walking round, she never pops out at the
   screen edge.
3. `!glados track` / `look me` / `angry` (the eye turns red-orange) / `sleep` (the eye dims) / `wake` / `nod` / `scan`
   / `dance` / `spooky`.
4. `!glados yaw 90` turns her a quarter turn and she still tracks you. `!glados scale 0.5`, then `!glados scale 1`.
5. `!glados` lines never reach the server, also `!gladoshere`.

## Uninstall

Close the game and delete `mods/glados-1.0.0+mc26.2.jar` (optionally also the two config files).
