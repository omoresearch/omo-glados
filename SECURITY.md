# Security

Omo GLaDOS is local-first. Nothing in it talks to the internet at runtime.

## The model

- The sidecar listens on **127.0.0.1 only** (port 8794). It has no auth, so do not port-forward or reverse-proxy it.
- The mod is **client-side**. `!glados` chat lines (also `!gladoshere` without the space) are intercepted and
  cancelled on your client; they never reach the server.
- The optional voice link only accepts a **loopback** WebSocket URL.
- Commands tagged `from: "twitch:*"` cannot `reset`, switch models or `talk`; anchor, scale and yaw are MC-chat
  only. `spooky` from Twitch has a 60 s cooldown. Chat can make her glare; it cannot touch your setup.
- Models are **data**: the sidecar validates every asset (`glados_assets.validate`) before using it, and a broken
  asset falls back to the procedural model. The mod checks the atlas PNG header and size before decoding it.

## Reporting

Please report vulnerabilities privately to **harry@omo.space** with the subject "Omo security", not in a public
issue. Include the version, what an attacker needs, and how to reproduce it. We aim to reply within a few days.
