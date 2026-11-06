#!/usr/bin/env python3
"""LiDAR odometry (scan-to-local-map, small_gicp) on INS attitude.

Each scan is deskewed to its mid time with the INS attitude (smooth, re-timed 50 Hz stream) and the INS body
velocity, then registered against a local map of the previous scans (incremental Gaussian voxel map, VGICP,
then a GICP refinement against the recent scans), starting from the previous pose plus the INS relative motion.
Roll and pitch are taken from the INS (drift-free); yaw and position come from the registration.
The odometry is started at the GNSS pose of the first scan.

usage: lo.py EXTRACT_DIR TRAJ_INIT_NPZ CALIB_JSON OUT_NPZ [VOXEL] [START_IDX END_IDX]
"""
import json
import os
import sys
import time
from collections import deque

import numpy as np
import small_gicp
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maplib import ScanCache, Trajectory

EXT, TRAJ, CALIB, OUT = sys.argv[1:5]
VOX = float(sys.argv[5]) if len(sys.argv) > 5 else 1.0
calib = json.load(open(CALIB))
EXTR = np.array(calib["extrinsic_base_to_lidar"])
DTL = calib["dt_lidar_ms"] * 1e-3
sc, tr = ScanCache(EXT), Trajectory(TRAJ)
raw = np.load(os.path.join(EXT, "traj_raw.npz"))
odom = raw["odom"]
# INS body velocity (odom twist) -> world, as a function of time
_, iu = np.unique(odom[:, 0], return_index=True)
tw_t, tw_v = odom[iu, 0], odom[iu, 9:12]


def v_world(t):
    vb = np.array([np.interp(t, tw_t, tw_v[:, k]) for k in range(3)])
    return R.from_quat(tr.quat[tr.index(t)]).apply(vb)


# dead-reckoned position: cumulative integral of the INS world velocity at 1 kHz (smooth, jump-free)
_tt = tr.t0 + np.arange(tr.N) * tr.dt
_vb = np.column_stack([np.interp(_tt, tw_t, tw_v[:, k]) for k in range(3)])
_vw = R.from_quat(tr.quat).apply(_vb)
P_DR = np.vstack([np.zeros(3), np.cumsum(0.5 * (_vw[1:] + _vw[:-1]) * tr.dt, axis=0)])
del _tt, _vb, _vw


def R_ins(t):
    return R.from_quat(tr.quat[tr.index(t)])


def deskew(i):
    """Scan i in base_link at its mid time t_k (INS rotation + constant INS velocity within the scan)."""
    s = sc.get(i)
    r = np.linalg.norm(s["xyz"], axis=1)
    m = (r > 3.0) & (r < 80.0)
    xyz, tns = s["xyz"][m].astype(np.float64), s["t"][m]
    tk = sc.stamp[i] + DTL + 0.05
    tp = sc.stamp[i] + DTL + tns * 1e-9
    idx = tr.index(tp)
    ui, inv = np.unique(idx, return_inverse=True)
    Rk = R.from_quat(tr.quat[tr.index(tk)])
    Rrel = (Rk.inv() * R.from_quat(tr.quat[ui])).as_matrix()  # R_k^T R(t_i)
    pb = xyz @ EXTR[:3, :3].T + EXTR[:3, 3]
    q = np.einsum("nij,nj->ni", Rrel[inv], pb)
    vk = Rk.inv().apply(v_world(tk))
    q += (tp - tk)[:, None] * vk[None, :]
    return q, tk


def with_ins_tilt(Rm, t):
    """Keep the yaw of Rm, take roll and pitch from the INS at t."""
    yaw = R.from_matrix(Rm).as_euler("ZYX")[0]
    e_ins = R_ins(t).as_euler("ZYX")
    return R.from_euler("ZYX", [yaw, e_ins[1], e_ins[2]]).as_matrix()


# scans to process: moved >= 0.15 m or turned >= 0.5 deg since the last processed scan (stationary scans skipped)
i0 = int(sys.argv[6]) if len(sys.argv) > 7 else 0
i1 = int(sys.argv[7]) if len(sys.argv) > 7 else len(sc)
sel, last_p, last_y, last_t = [], None, None, -1e9
for i in range(i0, i1):
    t = sc.stamp[i] + DTL + 0.05
    if not (tr.valid(t - 0.06) and tr.valid(t + 0.06)):
        continue
    T = tr.pose(t)
    y = np.arctan2(T[1, 0], T[0, 0])
    if last_p is None or np.linalg.norm(T[:3, 3] - last_p) >= 0.15 or abs(np.angle(np.exp(1j * (y - last_y)))) >= \
            np.radians(0.5) or t - last_t >= 1.0:
        sel.append(i)
        last_p, last_y, last_t = T[:3, 3], y, t
print(f"processing {len(sel)} of {i1 - i0} scans, voxel {VOX} m", flush=True)

vmap = small_gicp.GaussianVoxelMap(VOX)
vmap.set_lru(horizon=120, clear_cycle=10)
recent = deque(maxlen=12)  # (points in map frame) of recent keyframes for the GICP refinement
poses, times, idxs, info = [], [], [], []
T_prev, t_prev = None, None
last_kf_p = None
tgt_dirty, tgt, tgt_tree = True, None, None
tilt_log = []
t_start = time.time()
for n, i in enumerate(sel):
    q, tk = deskew(i)
    src, _ = small_gicp.preprocess_points(q, downsampling_resolution=0.25, num_threads=16)
    if T_prev is None:
        T = tr.pose(tk)
        T[:3, :3] = with_ins_tilt(T[:3, :3], tk)
        res_v = res_g = None
    else:
        # INS relative motion prev -> current, expressed in the previous body frame
        Rp, Rc = R_ins(t_prev), R_ins(tk)
        dR = (Rp.inv() * Rc).as_matrix()
        dp_w = P_DR[tr.index(tk)] - P_DR[tr.index(t_prev)]
        dT = np.eye(4)
        dT[:3, :3] = dR
        dT[:3, 3] = Rp.inv().apply(dp_w)
        T0 = T_prev @ dT
        res_v = small_gicp.align(vmap, src, T0, max_correspondence_distance=2.0 * VOX, num_threads=16, max_iterations=30)
        T = res_v.T_target_source
        # GICP refinement against the recent keyframes (finer than the voxel map)
        if tgt_dirty:
            tgt, tgt_tree = small_gicp.preprocess_points(np.concatenate(list(recent)), downsampling_resolution=0.2,
                                                         num_threads=16)
            tgt_dirty = False
        res_g = small_gicp.align(tgt, src, tgt_tree, T, max_correspondence_distance=0.5, num_threads=16, max_iterations=30)
        if res_g.converged and res_g.num_inliers > 0.4 * src.size():
            T = res_g.T_target_source
        e_reg = R.from_matrix(T[:3, :3]).as_euler("ZYX")
        e_ins = R_ins(tk).as_euler("ZYX")
        tilt_log.append((tk, e_reg[1] - e_ins[1], e_reg[2] - e_ins[2]))
        T[:3, :3] = with_ins_tilt(T[:3, :3], tk)
        jump = np.linalg.norm(T[:3, 3] - T0[:3, 3])
        if jump > 0.5:
            print(f"  WARNING scan {i} t={tk - sc.stamp[0]:.1f}s: registration moved {jump:.2f} m from the INS guess", flush=True)
    pts_map = q @ T[:3, :3].T + T[:3, 3]
    vmap.insert(src, T)  # preprocessed cloud carries the per-point covariances the Gaussian voxels need
    if last_kf_p is None or np.linalg.norm(T[:3, 3] - last_kf_p) > 1.0 or not recent:
        recent.append(small_gicp.voxelgrid_sampling(small_gicp.PointCloud(pts_map), 0.15).points()[:, :3])
        last_kf_p = T[:3, 3]
        tgt_dirty = True
    poses.append(T.copy())
    times.append(tk)
    idxs.append(i)
    info.append((res_v.num_inliers if res_v else 0, res_v.error if res_v else 0, res_g.num_inliers if res_g else 0,
                 res_g.error if res_g else 0, src.size()))
    T_prev, t_prev = T, tk
    if n % 500 == 0:
        print(f"  {n}/{len(sel)} t={tk - sc.stamp[0]:.1f}s  {time.time() - t_start:.0f}s", flush=True)

poses = np.array(poses)
np.savez(OUT, t=np.array(times), scan=np.array(idxs), pos=poses[:, :3, 3],
         quat=R.from_matrix(poses[:, :3, :3]).as_quat(), info=np.array(info), tilt=np.array(tilt_log))
print(f"done: {len(poses)} poses in {time.time() - t_start:.0f}s", flush=True)
