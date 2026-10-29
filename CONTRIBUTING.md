# Contributing

Issues and PRs are welcome, especially renderer fixes, new behaviours and macOS or Windows test reports.

- **Sidecar changes**: run `sidecar/test_headless.py` and `sidecar/test_v3.py` (under `nice`, because they are
  CPU-bound). `test_v3.py` starts its own side instance on a free port with its own state file.
- **Mod changes**: run `mod/build.sh test` against the stand-in sidecars
  (`python3 test/fake_sidecar.py 8789` and `... 8788 --v1`) or a test sidecar
  (`GLADOS_PORT=8799 .venv/bin/python server.py`), plus `mod/build.sh voicetest`. Don't point the harness at a
  sidecar you are playing with: it sends commands and moves the player.
- **New behaviours**: add them in `sidecar/glados_sim.py`, give them a check in `test_headless.py`, and list them in
  `GladosCommand.java` and `sidecar/CONTRACT.md`.
- **API changes**: update `sidecar/CONTRACT.md` in the same PR.
- **Models**: we only accept original work. No Valve assets, no rips, no fan models whose licence doesn't allow
  redistribution, no voice clips.

Keep the sidecar loopback-only and the mod client-side.
