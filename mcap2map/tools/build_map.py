#!/usr/bin/env python3
"""Accumulate deskewed keyframe scans into the map frame, binned into 100 m tiles, then voxelise each tile.

usage: build_map.py EXTRACT_DIR CALIB_JSON WORK_DIR TRAJ_NPZ
writes WORK_DIR/tiles/tile_<ix>_<iy>.npz with 0.1 m voxel centroids + per-voxel scan statistics
"""
import glob
import json
import os
import sys
import time
from collections import defaultdict

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maplib import ScanCache, Trajectory, deskew_to_map

EXT, CALIB, WORK, TRAJ = sys.argv[1:5]
TILE = 100.0
VOX = 0.1
RMIN, RMAX = 3.0, 60.0
RAW = os.path.join(WORK, "raw")
TILES = os.path.join(WORK, "tiles")
os.makedirs(RAW, exist_ok=True)
os.makedirs(TILES, exist_ok=True)

calib = json.load(open(CALIB))
EXTR = np.array(calib["extrinsic_base_to_lidar"])
DT_LIDAR = calib["dt_lidar_ms"] * 1e-3

sc, tr = ScanCache(EXT), Trajectory(TRAJ)

# --- keyframes: moved >= 0.25 m or turned >= 1 deg, or 2 s elapsed
keys = []
last_p, last_yaw, last_t = None, None, -1e9
for i, s in enumerate(sc.stamp):
    if not (tr.valid(s + DT_LIDAR) and tr.valid(s + DT_LIDAR + 0.1)):
        continue
    T = tr.pose(s + DT_LIDAR + 0.05)
    yaw = np.arctan2(T[1, 0], T[0, 0])
    if last_p is None or np.linalg.norm(T[:3, 3] - last_p) >= 0.25 or \
            abs(np.angle(np.exp(1j * (yaw - last_yaw)))) >= np.radians(1.0) or s - last_t >= 2.0:
        keys.append(i)
        last_p, last_yaw, last_t = T[:3, 3], yaw, s
keys = np.array(keys)
np.save(os.path.join(WORK, "keyframes.npy"), keys)
print(f"keyframes {len(keys)} / {len(sc)}", flush=True)

# --- pass 1: deskew + bin into tiles (flush to disk periodically)
buf = defaultdict(list)
nbuf, flush_id = 0, 0


def flush():
    global nbuf, flush_id
    for (ix, iy), parts in buf.items():
        np.save(os.path.join(RAW, f"t_{ix}_{iy}_{flush_id:04d}.npy"), np.concatenate(parts))
    buf.clear()
    nbuf = 0
    flush_id += 1


dt_rec = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("refl", "u2"), ("inten", "u2"), ("scan", "u2")])
t0 = time.time()
for n, i in enumerate(keys):
    s = sc.get(i)
    r = np.linalg.norm(s["xyz"], axis=1)
    m = (r > RMIN) & (r < RMAX)
    p = deskew_to_map(s["xyz"][m], s["t"][m], sc.stamp[i], tr, EXTR, DT_LIDAR)
    rec = np.empty(len(p), dt_rec)
    rec["x"], rec["y"], rec["z"] = p[:, 0], p[:, 1], p[:, 2]
    rec["refl"], rec["inten"], rec["scan"] = s["refl"][m], s["inten"][m], i
    tx = np.floor(p[:, 0] / TILE).astype(np.int32)
    ty = np.floor(p[:, 1] / TILE).astype(np.int32)
    tid = tx.astype(np.int64) * 100000 + ty
    order = np.argsort(tid, kind="stable")
    tid_s = tid[order]
    cuts = np.flatnonzero(np.diff(tid_s)) + 1
    for grp in np.split(order, cuts):
        buf[(int(tx[grp[0]]), int(ty[grp[0]]))].append(rec[grp])
    nbuf += len(p)
    if nbuf > 60_000_000:
        flush()
    if n % 500 == 0:
        print(f"  {n}/{len(keys)} scans, {time.time()-t0:.0f}s", flush=True)
flush()
print(f"pass 1 done in {time.time()-t0:.0f}s", flush=True)

# --- pass 2: voxelise each tile (centroid, mean reflectivity, hit count, distinct scans, scan span)
tiles = sorted({tuple(map(int, os.path.basename(f).split("_")[1:3])) for f in glob.glob(os.path.join(RAW, "t_*.npy"))})
total = 0
for ix, iy in tiles:
    rec = np.concatenate([np.load(f) for f in sorted(glob.glob(os.path.join(RAW, f"t_{ix}_{iy}_*.npy")))])
    p = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
    v = np.floor(p / VOX).astype(np.int64)
    v -= v.min(axis=0)
    span = v.max(axis=0) + 1
    key = (v[:, 0] * span[1] + v[:, 1]) * span[2] + v[:, 2]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    c = np.column_stack([np.bincount(inv, p[:, k], len(uk)) for k in range(3)]) / cnt[:, None]
    refl = np.bincount(inv, rec["refl"].astype(np.float64), len(uk)) / cnt
    inten = np.bincount(inv, rec["inten"].astype(np.float64), len(uk)) / cnt
    pair = np.unique(inv.astype(np.int64) * 65536 + rec["scan"])
    nscan = np.bincount(pair // 65536, minlength=len(uk))
    smin = np.full(len(uk), 65535, np.int64)
    smax = np.zeros(len(uk), np.int64)
    np.minimum.at(smin, inv, rec["scan"].astype(np.int64))
    np.maximum.at(smax, inv, rec["scan"].astype(np.int64))
    np.savez(os.path.join(TILES, f"tile_{ix}_{iy}.npz"), xyz=c.astype(np.float32), refl=refl.astype(np.float32),
             inten=inten.astype(np.float32), hits=cnt.astype(np.int32), nscan=nscan.astype(np.int32),
             smin=smin.astype(np.int32), smax=smax.astype(np.int32))
    total += len(uk)
    print(f"  tile {ix},{iy}: {len(rec)} pts -> {len(uk)} voxels", flush=True)
print(f"done: {total} voxels in {len(tiles)} tiles, {time.time()-t0:.0f}s", flush=True)
