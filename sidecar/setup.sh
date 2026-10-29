#!/usr/bin/env bash
# One-shot setup for the Omo GLaDOS MuJoCo sidecar.
#   ./setup.sh        create .venv, install mujoco + numpy + pillow, run a 1-second smoke test
# Re-running is safe (skips what is already there). Nothing is downloaded except the pip packages:
# the model is generated from glados_model.py (assets/procedural ships with the repo).
set -euo pipefail
HERE="$(cd "$(dirname "$0")" && pwd)"; cd "$HERE"

say() { printf '\033[1;33m>>\033[0m %s\n' "$*"; }

PY="${PYTHON:-python3}"
if command -v uv >/dev/null 2>&1; then
  [ -d .venv ] || uv venv -q --python "${UV_PYTHON:-3.12}" .venv
  say "installing requirements (uv)"
  uv pip install -q --python .venv/bin/python -r requirements.txt
else
  [ -d .venv ] || "$PY" -m venv .venv
  say "installing requirements (pip)"
  .venv/bin/pip install -q -r requirements.txt
fi

say "checking the procedural model"
.venv/bin/python glados_assets.py check

say "smoke test"
OMP_NUM_THREADS=1 .venv/bin/python smoke.py
say "done. Run:  OMP_NUM_THREADS=1 nice -n 10 .venv/bin/python server.py   (127.0.0.1:8794)"
