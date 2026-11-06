#!/usr/bin/env python3
"""Per-scan pose summary from a 1 kHz trajectory: t, x, y, z, yaw, speed, yaw rate, rtk_good flag.
rtk_good: GNSS epochs within +-3 s are all NARROW_INT / INS_RTKFIXED with std_h < 5 cm and no solution type change.
usage: scan_summary.py EXTRACT_DIR TRAJ_NPZ GNSS_NPZ OUT_NPY"""
import os
import sys
import numpy as np
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maplib import ScanCache, Trajectory

sc = ScanCache(sys.argv[1])
tr = Trajectory(sys.argv[2])
g = np.load(sys.argv[3])
t = sc.stamp + 0.05
i = tr.index(t)
p = tr.pos[i]
q = tr.quat[i]
yaw = np.unwrap(np.arctan2(2 * (q[:, 3] * q[:, 2] + q[:, 0] * q[:, 1]), 1 - 2 * (q[:, 1] ** 2 + q[:, 2] ** 2)))
v = np.linalg.norm(tr.pos[tr.index(t + 0.05)] - tr.pos[tr.index(t - 0.05)], axis=1) / 0.1
yr = np.gradient(yaw, t)
gt, gty, gstd = g["t"], g["type"], np.hypot(g["std"][:, 0], g["std"][:, 1])
good_ep = ((gty == 50) | (gty == 56)) & (gstd < 0.05)
chg = np.r_[False, np.diff(gty) != 0]
bad_t = gt[~good_ep | chg]
near_bad = np.zeros(len(t), bool)
if len(bad_t):
    j = np.searchsorted(bad_t, t)
    for jj in (j - 1, j):
        jj = np.clip(jj, 0, len(bad_t) - 1)
        near_bad |= np.abs(bad_t[jj] - t) < 3.0
rtk_good = (~near_bad) & tr.valid(t - 3) & tr.valid(t + 3)
out = np.column_stack([t, p, yaw, v, yr, rtk_good])
np.save(sys.argv[4], out)
print(f"{len(t)} scans, moving (>0.5 m/s) {np.mean(v > 0.5)*100:.0f}%, rtk_good {rtk_good.mean()*100:.0f}%, "
      f"rtk_good & moving {np.mean(rtk_good & (v > 0.5))*100:.0f}%, turns (>8 deg/s) in good: {np.sum(rtk_good & (np.abs(np.degrees(yr)) > 8) & (v > 2))}")
