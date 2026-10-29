"""Sample the live /pose for N seconds and report per-body motion stats (jitter / sway / NaN check).

usage: pose_probe.py [seconds] [base_url]
Jitter metric: RMS of the 2nd difference of the body position and orientation per sim step (high-frequency
content), versus the overall excursion (low-frequency sway).
"""
import json, math, sys, time, urllib.request
import numpy as np

T = float(sys.argv[1]) if len(sys.argv) > 1 else 10
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8794"
rows, seen = [], set()
t_end = time.time() + T
bad = 0
while time.time() < t_end:
    p = json.load(urllib.request.urlopen(BASE + "/pose", timeout=2))
    if p["seq"] in seen:
        time.sleep(0.004)
        continue
    seen.add(p["seq"])
    vals = [v for b in p["bodies"].values() for v in b]
    if not all(math.isfinite(v) for v in vals):
        bad += 1
    rows.append((p["seq"], p["t"], p["mode"], p["bodies"]))
    time.sleep(0.012)
rows.sort()
names = list(rows[0][3].keys())
seqs = np.array([r[0] for r in rows])
print(f"{len(rows)} poses over {rows[-1][1] - rows[0][1]:.2f}s sim, seq gaps>1: {int((np.diff(seqs) > 1).sum())}, "
      f"non-finite: {bad}, modes: {sorted(set(r[2] for r in rows))}")


def qang(q1, q2):
    d = abs(float(np.dot(q1, q2)))
    return 2 * math.degrees(math.acos(min(1.0, d)))


ts = np.array([r[1] for r in rows])
for n in names:
    P = np.array([r[3][n][:3] for r in rows])
    Q = np.array([r[3][n][3:] for r in rows])
    dt = np.diff(ts)
    v = np.linalg.norm(np.diff(P, axis=0), axis=1) / np.maximum(dt, 1e-6)
    w = np.array([qang(Q[i], Q[i + 1]) for i in range(len(Q) - 1)]) / np.maximum(dt, 1e-6)
    # high-frequency: deviation from a 9-sample moving average
    k = 9
    if len(P) > 2 * k:
        ker = np.ones(k) / k
        Ps = np.stack([np.convolve(P[:, i], ker, "same") for i in range(3)], 1)
        hf = np.linalg.norm(P[k:-k] - Ps[k:-k], axis=1)
        hf_mm = 1000 * float(np.sqrt((hf ** 2).mean()))
    else:
        hf_mm = float("nan")
    print(f"  {n:6s} excursion {np.ptp(P, axis=0).round(3)} m, speed max {v.max():.2f} m/s, "
          f"ang-speed max {w.max():.1f} deg/s mean {w.mean():.1f}, hf-jitter {hf_mm:.2f} mm")
