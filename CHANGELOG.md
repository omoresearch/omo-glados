# Changelog

## Unreleased: first public source

The first public source drop of Omo GLaDOS. No release jar yet: build the mod from source.

- **Sidecar**: an original procedural GLaDOS-style model (15,800 triangles, 6 bodies) as a hanging MuJoCo hinge
  chain (13 joints, 2,691 kg) under gravity, PD-driven with gravity compensation and look-at IK. Behaviours: idle,
  track, look, angry, scan, nod, shake, bow, sleep/wake, dance, spooky, talk. Loopback HTTP API on 127.0.0.1:8794.
- **Models**: the v3 asset format with live switching (`!glados model <name>`); only `procedural` ships.
- **Mod** (Fabric, MC 26.2, client-only): renders her articulated from any ceiling (`!glados here`), with
  textured/skinned asset support, eye glow and talk pulse; Sodium/Iris-safe. `!glados` chat commands.
- **Voice** (optional): a Simple Voice Chat plugin with positional replies from her head and mic-focus arbitration
  with Omo Microduck. The voice backend is not included.
- **Twitch-safe**: commands from Twitch are limited to gestures and moods.
