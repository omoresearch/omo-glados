#!/usr/bin/env python3
"""Smoke test used by setup.sh: build the sim on the procedural model and step it for one simulated second."""
import os

os.environ.setdefault("OMP_NUM_THREADS", "1")
import numpy as np  # noqa: E402

import glados_sim as gs  # noqa: E402

s = gs.GladosSim(log=lambda *a: None, seed=1)
s.set_player(5.0, 0.0, -11.4)
for _ in range(60):
    s.step()
assert np.all(np.isfinite(s.data.qpos)), "non-finite state"
print(f"ok: sim stepped 1 s, mode {s.public_mode()}, {s.model.njnt} joints, mass {s.mass:.0f} kg")
