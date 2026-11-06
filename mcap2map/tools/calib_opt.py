#!/usr/bin/env python3
"""Lidar extrinsic rotation + time offset self-calibration on RTK-fixed stretches.

Cost: GICP corrections of single deskewed scans against local maps built from temporally separated scans of the
same pass (a wrong rotation / time offset makes a scan disagree with its neighbours, most of all in turns).
Parameters: (yaw, pitch, roll) correction in degrees applied in base_link on top of /tf_static, dt_lidar in ms.

usage: calib_opt.py EXTRACT_DIR TRAJ_NPZ SUMMARY_NPY OUT_JSON [eval "y,p,r,dt" ...]
"""
import json
import os
import sys

import numpy as np
import small_gicp
from scipy.optimize import minimize
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import INS_HEADING_OFFSET_DEG, PRIOR
from maplib import EXTRINSIC_TF, ScanCache, Trajectory, deskew_to_map, make_T

EXT, TRAJ, SUMM, OUT = sys.argv[1:5]
sc, tr = ScanCache(EXT), Trajectory(TRAJ)
summ = np.load(SUMM)  # t, x, y, z, yaw, v, yawrate, rtk_good
v, yr, good = summ[:, 5], np.degrees(summ[:, 6]), summ[:, 7] > 0.5
ok_c = good & (np.arange(len(sc)) >= 25) & (np.arange(len(sc)) < len(sc) - 25)
# the whole +-2 s window must be RTK-good
win_ok = np.array([ok_c[i] and good[max(i - 21, 0):i + 22].all() for i in range(len(sc))])
rng = np.random.default_rng(1)
turn = np.flatnonzero(win_ok & (v > 2) & (np.abs(yr) > 8))
straight = np.flatnonzero(win_ok & (v > 4) & (np.abs(yr) < 1))
centres = list(rng.choice(turn, min(24, len(turn)), replace=False)) + \
    list(rng.choice(straight, min(8, len(straight)), replace=False))
print(f"{len(centres)} windows ({len(turn)} turn / {len(straight)} straight candidates)", flush=True)
_pl = PRIOR.get("lidar_correction")  # [yaw, pitch, roll, dt_ms] on top of /tf_static, in the INS frame
X0 = np.array(_pl, float) if _pl else np.array([INS_HEADING_OFFSET_DEG, 0.0, 0.0, 0.0])
START = "vehicle profile prior" if _pl else "/tf_static rotated by the INS heading offset"

pts = {}
for c in centres:
    for j in range(c - 20, c + 21):
        if j not in pts:
            s = sc.get(j)
            r = np.linalg.norm(s["xyz"], axis=1)
            m = (r > 3.0) & (r < 60.0)
            pts[j] = (s["xyz"][m], s["t"][m])


def extrinsic(x):
    yaw, pitch, roll = x[:3]
    return make_T(rotvec=R.from_euler("zyx", [yaw, pitch, roll], degrees=True).as_rotvec()) @ EXTRINSIC_TF


def corrections(x):
    extr, dtl = extrinsic(x), x[3] * 1e-3
    tn, rn = [], []
    for c in centres:
        tgt = np.concatenate([deskew_to_map(pts[j][0][::3], pts[j][1][::3], sc.stamp[j], tr, extr, dtl)
                              for j in list(range(c - 20, c - 5)) + list(range(c + 6, c + 21))])
        src = deskew_to_map(pts[c][0], pts[c][1], sc.stamp[c], tr, extr, dtl)
        off = src.mean(axis=0)
        tc, tt = small_gicp.preprocess_points(tgt - off, downsampling_resolution=0.2, num_threads=16)
        scl, _ = small_gicp.preprocess_points(src - off, downsampling_resolution=0.2, num_threads=16)
        T = small_gicp.align(tc, scl, tt, max_correspondence_distance=0.6, num_threads=16).T_target_source
        tn.append(np.linalg.norm(T[:3, 3]))
        rn.append(np.degrees(np.linalg.norm(R.from_matrix(T[:3, :3]).as_rotvec())))
    return np.array(tn), np.array(rn)


def cost(x):
    tn, rn = corrections(x)
    f = np.mean(tn) * 100 + 10 * np.mean(rn)
    print(f"yaw {x[0]:+.3f} pitch {x[1]:+.3f} roll {x[2]:+.3f} dt {x[3]:+.1f} ms -> "
          f"{f:.3f} (trans {np.mean(tn)*100:.2f} cm, rot {np.mean(rn):.3f} deg)", flush=True)
    return f


if len(sys.argv) > 5 and sys.argv[5] == "eval":
    for cfg in sys.argv[6:]:
        cost(np.array(list(map(float, cfg.split(",")))))
    sys.exit(0)

x0 = X0
if len(centres) < 6:  # not enough RTK-good motion to calibrate: keep the start
    print(f"WARNING: only {len(centres)} RTK-good calibration windows; keeping the start ({START})", flush=True)
    json.dump({"yaw_deg": x0[0], "pitch_deg": x0[1], "roll_deg": x0[2], "dt_lidar_ms": x0[3], "cost": None,
               "start": START, "extrinsic_base_to_lidar": extrinsic(x0).tolist()}, open(OUT, "w"), indent=2)
    sys.exit(0)
print(f"start: {START} {np.round(x0, 3)}", flush=True)
simplex = np.array([x0, x0 + [0.4, 0, 0, 0], x0 + [0, 0.4, 0, 0], x0 + [0, 0, 0.4, 0], x0 + [0, 0, 0, 12]])
res = minimize(cost, x0, method="Nelder-Mead",
               options={"initial_simplex": simplex, "maxfev": 60 if _pl else 90, "xatol": 0.03, "fatol": 0.03})
print("best:", res.x, res.fun)
json.dump({"yaw_deg": res.x[0], "pitch_deg": res.x[1], "roll_deg": res.x[2], "dt_lidar_ms": res.x[3],
           "cost": res.fun, "start": START, "extrinsic_base_to_lidar": extrinsic(res.x).tolist()}, open(OUT, "w"), indent=2)
