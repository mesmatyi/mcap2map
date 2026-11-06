#!/usr/bin/env python3
"""Reference trajectory for the pose graph.

GNSS/INS base_link positions are smoothed piecewise between 'breaks' (solution-type switches and steps inside INS
segments, where the receiver's solution jumps; RTK epochs weighted by their std), attitude from the INS.
Inside lidar-odometry windows (GNSS single-point / absent: the INS drifts and re-converges by metres within
seconds) positions and yaw come from lo.py runs instead (roll/pitch stay INS).

usage:
  build_ref.py CACHE_DIR --detect OUT_JSON        degraded-GNSS windows (absolute times, snapped to breaks)
  build_ref.py CACHE_DIR [--windows WINDOWS_JSON]  write CACHE_DIR/traj_ref_1khz.npz (WINDOWS_JSON: list of
                                                   {"lo": npz, "ta": t, "tb": t}); without windows: receiver only
"""
import json
import os
import sys

import numpy as np
from scipy.interpolate import CubicSpline, make_smoothing_spline
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import param

C = sys.argv[1]
MODE = sys.argv[2] if len(sys.argv) > 2 else None
g = np.load(os.path.join(C, "gnss_epochs.npz"))
t, p, ty, std = g["t"], g["p_base"], g["type"], g["std"]
std_h = np.hypot(std[:, 0], std[:, 1])
ini = np.load(os.path.join(C, "traj_init_1khz.npz"))
GOOD_TYPES = (50, 56)          # NARROW_INT, INS_RTKFIXED
GOOD_STD = param("rtk_good_std_m", 0.10)
MIN_BAD = param("lo_window_min_s", 5.0)
MERGE_GAP = param("lo_window_merge_gap_s", 15.0)

# breaks: solution-type switches and steps inside INS segments (epoch velocity far from its neighbours)
starts = set((np.flatnonzero(np.diff(ty) != 0) + 1).tolist())
v = np.diff(p, axis=0) / np.diff(t)[:, None]
n_step = 0
for j in range(3, len(v) - 3):
    if ty[j] < 52 or ty[j + 1] != ty[j]:
        continue
    nb = np.r_[v[j - 3:j], v[j + 1:j + 4]]
    if np.linalg.norm(v[j, :2] - np.median(nb[:, :2], axis=0)) > 1.0:
        starts.add(j + 1)
        n_step += 1
starts = sorted(starts)
breaks = np.array([0.5 * (t[s - 1] + t[s]) for s in starts])

if MODE == "--detect":
    good = np.isin(ty, GOOD_TYPES) & (std_h < GOOD_STD)
    runs, k = [], 0
    while k < len(t):
        if good[k]:
            k += 1
            continue
        j = k
        while j < len(t) and not good[j]:
            j += 1
        runs.append([t[k], t[j - 1]])
        k = j
    runs = [r for r in runs if r[1] - r[0] >= MIN_BAD]
    merged = []
    for r in runs:
        if merged and r[0] - merged[-1][1] < MERGE_GAP:
            merged[-1][1] = r[1]
        else:
            merged.append(r)
    wins = []
    for a, b in merged:
        # start: the break where the degradation begins (else exactly there); end: the break where RTK returns
        bb = breaks[(breaks <= a + 0.06) & (breaks >= a - 2.0)]
        ta = float(bb.max()) if len(bb) else (float(t[0]) if a - t[0] < 1.0 else float(a - 0.025))
        be = breaks[(breaks >= b - 0.06) & (breaks <= b + 2.0)]
        tb = float(be.min()) if len(be) else (float(t[-1]) if t[-1] - b < 1.0 else float(b + 0.025))
        wins.append({"ta": ta, "tb": tb, "duration_s": tb - ta, "t_rel": [ta - t[0], tb - t[0]]})
    json.dump(wins, open(sys.argv[3], "w"), indent=1)
    print(f"{len(starts)} breaks; degraded-GNSS windows (good = types {GOOD_TYPES} & std < {GOOD_STD} m, >= {MIN_BAD} s): "
          + (", ".join(f"{w['t_rel'][0]:.2f}-{w['t_rel'][1]:.2f} s" for w in wins) or "none"))
    sys.exit(0)

print(f"{len(starts)} breaks ({len(starts) - n_step} type switches, {n_step} INS steps)")
cuts = [0] + starts + [len(t)]
tg = ini["t0"] + np.arange(len(ini["pos"])) * ini["dt"]
pos = np.zeros_like(ini["pos"])
filled = np.zeros(len(tg), bool)
for a, b in zip(cuts[:-1], cuts[1:]):
    tt, pp = t[a:b], p[a:b]
    lam = 1e-4 if ty[a] == 50 else 1e-6  # RTK-only epochs carry mm noise; INS epochs are smooth already
    lo = 0.5 * (t[a - 1] + t[a]) if a > 0 else tg[0]
    hi = 0.5 * (t[b - 1] + t[b]) if b < len(t) else tg[-1] + 1
    m = (tg >= lo) & (tg < hi)
    if len(tt) >= 5:
        sd_ = np.maximum(std_h[a:b], 0.005)
        w_ = np.clip((np.median(sd_) / sd_) ** 2, 0.01, 10.0)  # degraded epochs (std up to dm) count less
        for k in range(3):
            s = make_smoothing_spline(tt - t[0], pp[:, k], w=w_, lam=lam)
            pos[m, k] = s(np.clip(tg[m], tt[0], tt[-1]) - t[0]) + np.where(tg[m] < tt[0], (tg[m] - tt[0]), 0) * \
                s.derivative()(tt[0] - t[0]) + np.where(tg[m] > tt[-1], (tg[m] - tt[-1]), 0) * s.derivative()(tt[-1] - t[0])
    else:
        for k in range(3):
            pos[m, k] = np.interp(tg[m], tt, pp[:, k])
    filled |= m
assert filled.all()
quat = ini["quat"].copy()

if MODE == "--windows":
    wins = json.load(open(sys.argv[3]))
    for w in wins:
        lo = np.load(w["lo"])
        ta, tb = w["ta"], w["tb"]
        m = (tg >= ta) & (tg < tb)
        tl = lo["t"]
        ps = CubicSpline(tl, lo["pos"])
        tc = np.clip(tg[m], tl[0], tl[-1])
        v0, v1 = ps.derivative()(tl[0]), ps.derivative()(tl[-1])
        pos[m] = ps(tc) + np.where(tg[m] < tl[0], tg[m] - tl[0], 0)[:, None] * v0 + \
            np.where(tg[m] > tl[-1], tg[m] - tl[-1], 0)[:, None] * v1
        y_lo = R.from_quat(lo["quat"]).as_euler("ZYX")[:, 0]
        y_ins = R.from_quat(quat[np.clip(np.round((tl - tg[0]) / ini["dt"]).astype(int), 0, len(tg) - 1)]).as_euler("ZYX")[:, 0]
        dpsi = np.unwrap(y_lo - y_ins)
        quat[m] = (R.from_rotvec(np.column_stack([np.zeros((m.sum(), 2)), np.interp(tg[m], tl, dpsi)])) *
                   R.from_quat(quat[m])).as_quat()
        keep = (breaks <= ta + 1e-6) | (breaks >= tb - 1e-6)
        breaks = breaks[keep]
        for bt in (ta, tb):  # the hand-over points are breaks (unless they are the ends of the drive)
            if tg[0] + 0.5 < bt < tg[-1] - 0.5 and not np.any(np.abs(breaks - bt) < 1e-4):
                breaks = np.sort(np.r_[breaks, bt])
        print(f"lidar odometry replaces the receiver solution {ta - t[0]:.2f}-{tb - t[0]:.2f} s, "
              f"yaw offset odometry-INS {np.degrees(dpsi[0]):+.2f} .. {np.degrees(dpsi[-1]):+.2f} deg")

# position / yaw step of the reference at every break (the correction must cancel it: the vehicle did not jump)
steps, ysteps, svec, ysig = [], [], [], []
for bt in breaks:
    i = int(round((bt - tg[0]) / ini["dt"]))
    i0, i1 = max(i - 3, 0), min(i + 3, len(tg) - 1)
    vv = (pos[i0] - pos[max(i0 - 20, 0)]) / max(i0 - max(i0 - 20, 0), 1)
    pred = pos[i0] + vv * (i1 - i0)
    steps.append(np.linalg.norm(pos[i1] - pred))
    svec.append(pos[i1] - pred)
    ya, yb = (R.from_quat(quat[k]).as_euler("ZYX")[0] for k in (i0, i1))
    ysteps.append(abs(np.angle(np.exp(1j * (yb - ya)))))
    ysig.append(np.angle(np.exp(1j * (yb - ya))))
steps, ysteps, ysig = (np.array(x, float) for x in (steps, ysteps, ysig))
svec = np.array(svec, float).reshape(-1, 3)
d = np.linalg.norm(pos - ini["pos"], axis=1)
print(f"reference vs interpolated GNSS: median {np.median(d)*1000:.1f} mm; {len(breaks)} breaks, largest step "
      f"{(np.max(steps) if len(steps) else 0)*100:.1f} cm")
np.savez(os.path.join(C, "traj_ref_1khz.npz"), t0=ini["t0"], dt=ini["dt"], pos=pos, quat=quat, breaks=breaks,
         break_step=np.atleast_1d(steps), break_yaw_step=np.atleast_1d(ysteps), break_step_vec=svec,
         break_yaw_step_signed=np.atleast_1d(ysig), lever_arm=ini["lever_arm"])
