#!/usr/bin/env python3
"""Camera extrinsic rotation + time offset by photo-consistency: map points seen from several consecutive frames
should get the same brightness from each of them (robust spread of their normalised brightness, lower = better).

Per camera:
1. start: vehicle-profile prior (base_T_cam, dt) if any, else /tf_static rotated by the INS heading offset
2. time-offset scan with the start rotation (around the prior dt, else -1.5 .. +1.5 s, then fine)
3. Nelder-Mead over (rx, ry, rz) [deg, camera frame] and dt [ms] on training segments
4. held-out segments decide: optimum vs start, and pinhole vs plumb_bob
Camera header stamps can be far from the capture time (0.69 s late on the Lexus3 2026 bags).

usage: cam_calib.py EXTRACT_DIR MAP_PCD TRAJ_NPZ OUT_JSON
"""
import json
import os
import sys

import numpy as np
from scipy.optimize import minimize
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camlib import NCAM, Images, base_T_cam, model, prior_T_cam, project, read_pcd
from config import CAMERAS, INS_HEADING_OFFSET_DEG
from maplib import Trajectory

EXT, PCD, TRAJ, OUT = sys.argv[1:5]
P_ = read_pcd(PCD)
pts = P_[:, :3].astype(np.float64)
kd = cKDTree(pts[:, :2])
tr = Trajectory(TRAJ)
imgs = Images(EXT)
SC = 0.25
speed = lambda t: np.linalg.norm(tr.pos[tr.index(t + 0.1)] - tr.pos[tr.index(t - 0.1)]) / 0.2


def make_segments(cam, seed, n, dt_guess):
    idx = imgs.of(cam)
    if len(idx) < 40:
        return []
    rng = np.random.default_rng(seed)
    st = imgs.stamp[idx] + dt_guess
    cand = [k for k in range(0, len(idx) - 30, 7)
            if tr.valid(st[k] - 1.6) and tr.valid(st[k + 24] + 1.6) and 3 < speed(st[k]) < 10 and 3 < speed(st[k + 21]) < 10]
    if not cand:
        return []
    segs = [list(idx[k:k + 24:3]) for k in rng.choice(cand, min(n, len(cand)), replace=False)]
    data = []
    for sg in segs:
        gs = []
        for i in sg:
            g = imgs.gray(i, 4).astype(np.float32)
            gs.append((g - g.mean()) / max(g.std(), 1e-6))
        tm = imgs.stamp[sg[len(sg) // 2]] + dt_guess
        data.append((sg, gs, pts[kd.query_ball_point(tr.pose(tm)[:2, 3], 34.0)]))
    return data


def spread(cam, data, T0, x, mdl):
    Tbc = T0.copy()
    Tbc[:3, :3] = T0[:3, :3] @ R.from_rotvec(np.radians(x[:3])).as_matrix()
    dt = x[3] * 1e-3
    meds = []
    for sg, gs, P in data:
        vals = np.full((len(sg), len(P)), np.nan, np.float32)
        for f, (i, g) in enumerate(zip(sg, gs)):
            Tmc = tr.pose(imgs.stamp[i] + dt) @ Tbc
            pc = (P - Tmc[:3, 3]) @ Tmc[:3, :3]
            u, v, ok = project(cam, pc, SC, mdl)
            ok &= pc[:, 2] < 25
            if ok.sum() < 100:
                continue
            hs, ws = g.shape
            ui, vi, d = u[ok].astype(int), v[ok].astype(int), pc[ok, 2]
            ui, vi = np.clip(ui, 0, ws - 1), np.clip(vi, 0, hs - 1)
            D = np.full((hs, ws), np.inf)
            np.minimum.at(D, (vi, ui), d)
            vis = d <= D[vi, ui] + 0.3
            vals[f, np.flatnonzero(ok)[vis]] = g[vi[vis], ui[vis]]
        m = np.sum(np.isfinite(vals), axis=0) >= 4
        if m.sum() < 200:
            continue
        v_ = vals[:, m]
        med = np.nanmedian(v_, axis=0)
        meds.append(np.median(np.nanmedian(np.abs(v_ - med), axis=0)))
    return float(np.mean(meds)) if meds else 9.0


result = []
for cam in range(NCAM):
    pr = prior_T_cam(cam)
    if pr is not None:
        T0, dt0, start = pr[0], pr[1], "vehicle profile prior"
    else:
        Rz = np.eye(4)
        Rz[:3, :3] = R.from_euler("z", INS_HEADING_OFFSET_DEG, degrees=True).as_matrix()
        T0, dt0, start = Rz @ base_T_cam(cam), 0.0, "/tf_static rotated by the INS heading offset"
    grid = np.arange(-1500, 1501, 100.0)  # always wide: stamp offsets differ between camera drivers
    mdl = model(cam)
    entry = {"index": cam, "name": CAMERAS[cam]["name"], "model": mdl, "start": start,
             "focal_px": float(CAMERAS[cam]["K"][0])}
    train = make_segments(cam, 21, 16, dt0 / 1e3)
    test = make_segments(cam, 77, 12, dt0 / 1e3)
    if len(train) < 4 or len(test) < 3:
        print(f"camera{cam}: too few moving segments ({len(train)}/{len(test)}); keeping the start", flush=True)
        entry.update(base_T_cam=T0.tolist(), dt_cam_ms=dt0, rot_deg_cam=[0, 0, 0], heldout_spread=None)
        result.append(entry)
        continue
    sg = [spread(cam, train, T0, np.r_[0, 0, 0, d], mdl) for d in grid]
    d0 = grid[int(np.argmin(sg))]
    if True:  # fine scan around the coarse minimum
        g2 = np.arange(d0 - 150, d0 + 151, 25.0)
        s2 = [spread(cam, train, T0, np.r_[0, 0, 0, d], mdl) for d in g2]
        d0 = g2[int(np.argmin(s2))]
    print(f"camera{cam} ({CAMERAS[cam]['name']}): start {start}; time-offset scan best {d0:+.0f} ms", flush=True)
    x0 = np.r_[0.0, 0.0, 0.0, d0]
    simplex = np.array([x0, x0 + [0.4, 0, 0, 0], x0 + [0, 0.4, 0, 0], x0 + [0, 0, 0.4, 0], x0 + [0, 0, 0, 40]])
    res = minimize(lambda x: spread(cam, train, T0, x, mdl), x0, method="Nelder-Mead",
                   options={"initial_simplex": simplex, "maxfev": 70, "xatol": 0.02, "fatol": 1e-4})
    t_start, t_scan, t_opt = (spread(cam, test, T0, x, mdl) for x in (np.r_[0, 0, 0, dt0], x0, res.x))
    xf = res.x if t_opt <= t_scan else x0
    other = "plumb_bob" if mdl == "pinhole" else "pinhole"
    t_other = spread(cam, test, T0, xf, other)
    best_t = min(t_opt, t_scan)
    if t_other < 0.97 * best_t:
        mdl, best_t = other, t_other
    Tbc = T0.copy()
    Tbc[:3, :3] = T0[:3, :3] @ R.from_rotvec(np.radians(xf[:3])).as_matrix()
    entry.update(model=mdl, base_T_cam=Tbc.tolist(), dt_cam_ms=float(xf[3]), rot_deg_cam=xf[:3].tolist(),
                 heldout_spread=float(best_t), heldout_spread_start=float(t_start))
    print(f"camera{cam}: rot {np.round(xf[:3], 3)} deg, dt {xf[3]:+.1f} ms, model {mdl}; held-out spread start "
          f"{t_start:.4f} -> {best_t:.4f}", flush=True)
    result.append(entry)
json.dump({"cameras": result}, open(OUT, "w"), indent=2)
print("saved", OUT)
