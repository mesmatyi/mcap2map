#!/usr/bin/env python3
"""Remove moving objects from the voxel map by free-space voting (range-image visibility test).

A non-ground map voxel is 'seen through' by a scan if the scan's measured range in that direction is clearly
longer than the voxel's range. Voxels seen through repeatedly (and rarely confirmed) are dynamic.

usage: dynamic_filter.py EXTRACT_DIR CALIB_JSON TRAJ_NPZ MAP_DIR RASTER_NPZ
writes MAP_DIR/tiles/*.npz key 'dynamic' (bool per voxel) into MAP_DIR/dynamic/<tile>.npy
"""
import glob
import json
import os
import sys
import time

import numpy as np
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maplib import ScanCache, Trajectory, deskew_to_map

CACHE, CALIB, TRAJ, MAP, RAS = sys.argv[1:6]
COARSE = 0.3
RMIN, RMAX = 3.0, 35.0
NCOL = 1024
OUTD = os.path.join(MAP, "dynamic")
os.makedirs(OUTD, exist_ok=True)

calib = json.load(open(CALIB))
EXTR = np.array(calib["extrinsic_base_to_lidar"])
DT = calib["dt_lidar_ms"] * 1e-3
sc, tr = ScanCache(CACHE), Trajectory(TRAJ)
keys = np.load(os.path.join(MAP, "keyframes.npy"))
ras = np.load(RAS)
gx0, gy0, gres, ground = float(ras["x0"]), float(ras["y0"]), float(ras["res"]), ras["ground"]

# --- load voxels, mark ground, build coarse test voxels from non-ground ones
files = sorted(glob.glob(os.path.join(MAP, "tiles", "tile_*.npz")))
xyz_all, tile_of, n_of = [], [], []
for f in files:
    xyz = np.load(f)["xyz"]
    xyz_all.append(xyz)
    n_of.append(len(xyz))
xyz_all = np.concatenate(xyz_all)
ix = np.floor((xyz_all[:, 0] - gx0) / gres).astype(np.int64)
iy = np.floor((xyz_all[:, 1] - gy0) / gres).astype(np.int64)
inr = (ix >= 0) & (ix < ground.shape[1]) & (iy >= 0) & (iy < ground.shape[0])
gz = np.full(len(xyz_all), np.nan, np.float32)
gz[inr] = ground[iy[inr], ix[inr]]
nonground = ~(np.isfinite(gz) & (xyz_all[:, 2] < gz + 0.25))
cand = np.flatnonzero(nonground)
cv = np.floor(xyz_all[cand] / COARSE).astype(np.int64)
cv -= cv.min(axis=0)
span = cv.max(axis=0) + 1
ckey = (cv[:, 0] * span[1] + cv[:, 1]) * span[2] + cv[:, 2]
uk, cinv, ccnt = np.unique(ckey, return_inverse=True, return_counts=True)
cxyz = np.column_stack([np.bincount(cinv, xyz_all[cand, k].astype(np.float64), len(uk)) for k in range(3)]) / ccnt[:, None]
print(f"voxels {len(xyz_all)}, non-ground {len(cand)}, coarse test voxels {len(uk)}", flush=True)

# 2-D grid index (5 m) over coarse voxels
G = 5.0
gxy = np.floor(cxyz[:, :2] / G).astype(np.int64)
gkey = gxy[:, 0] * 100000 + gxy[:, 1]
gorder = np.argsort(gkey)
gk_s = gkey[gorder]
gu, gstart = np.unique(gk_s, return_index=True)
gend = np.r_[gstart[1:], len(gk_s)]
gmap = dict(zip(gu.tolist(), zip(gstart.tolist(), gend.tolist())))


def voxels_near(p, r):
    cx, cy = int(np.floor(p[0] / G)), int(np.floor(p[1] / G))
    k = int(np.ceil(r / G))
    out = []
    for a in range(cx - k, cx + k + 1):
        for b in range(cy - k, cy + k + 1):
            v = gmap.get(a * 100000 + b)
            if v:
                out.append(gorder[v[0]:v[1]])
    return np.concatenate(out) if out else np.zeros(0, np.int64)


# --- ring elevation table from a few scans (sensor frame)
NRING = 1 + max(int(sc.get(i)["ring"].max()) for i in keys[:: max(1, len(keys) // 20)])
els = [[] for _ in range(NRING)]
for i in keys[:: max(1, len(keys) // 20)]:
    s = sc.get(i)
    e = np.arctan2(s["xyz"][:, 2], np.hypot(s["xyz"][:, 0], s["xyz"][:, 1]))
    for r_ in range(NRING):
        m = s["ring"] == r_
        if m.any():
            els[r_].append(np.median(e[m]))
ring_el = np.array([np.median(v) if v else np.nan for v in els])
ring_el = np.where(np.isfinite(ring_el), ring_el, np.nanmedian(ring_el))
order_el = np.argsort(ring_el)
el_sorted = ring_el[order_el]
spacing = np.median(np.diff(el_sorted))
print(f"ring elevations {np.degrees(el_sorted[0]):.1f} .. {np.degrees(el_sorted[-1]):.1f} deg, spacing {np.degrees(spacing):.2f}")

free = np.zeros(len(uk), np.int32)
seen = np.zeros(len(uk), np.int32)
t0 = time.time()
test_keys = keys[::3]
for n, i in enumerate(test_keys):
    s = sc.get(i)
    rr = np.linalg.norm(s["xyz"], axis=1)
    m = rr > 1.0
    tm = sc.stamp[i] + DT + 0.05
    Tm = tr.pose(tm) @ EXTR
    pm = deskew_to_map(s["xyz"][m], s["t"][m], sc.stamp[i], tr, EXTR, DT)
    pl = (pm - Tm[:3, 3]) @ Tm[:3, :3]  # deskewed points in the lidar frame at mid-scan
    rng = np.linalg.norm(pl, axis=1)
    az = np.arctan2(pl[:, 1], pl[:, 0])
    col = ((az + np.pi) / (2 * np.pi) * NCOL).astype(np.int64) % NCOL
    row = s["ring"][m].astype(np.int64)
    img = np.full((NRING, NCOL), np.inf)
    np.minimum.at(img, (row, col), rng)
    # conservative neighbourhood: minimum over 3x3 (unfilled pixels stay inf only if all neighbours are)
    pad = np.pad(img, ((1, 1), (1, 1)), mode="edge")
    pad[:, 0], pad[:, -1] = pad[:, -2], pad[:, 1]
    nb = np.min(np.stack([pad[1 + dy:NRING + 1 + dy, 1 + dx:NCOL + 1 + dx] for dy in (-1, 0, 1) for dx in (-1, 0, 1)]), axis=0)
    vid = voxels_near(Tm[:3, 3], RMAX)
    if not len(vid):
        continue
    vl = (cxyz[vid] - Tm[:3, 3]) @ Tm[:3, :3]
    vr = np.linalg.norm(vl, axis=1)
    ve = np.arctan2(vl[:, 2], np.hypot(vl[:, 0], vl[:, 1]))
    j = np.searchsorted(el_sorted, ve)
    j = np.clip(j, 1, NRING - 1)
    jn = np.where(np.abs(el_sorted[j] - ve) < np.abs(el_sorted[j - 1] - ve), j, j - 1)
    inside = (vr > RMIN) & (vr < RMAX) & (np.abs(el_sorted[jn] - ve) < 0.5 * spacing)
    vrow = order_el[jn]
    vcol = ((np.arctan2(vl[:, 1], vl[:, 0]) + np.pi) / (2 * np.pi) * NCOL).astype(np.int64) % NCOL
    meas = nb[vrow, vcol]
    finite = np.isfinite(meas) & inside
    margin = 0.5 + 0.03 * vr
    is_free = finite & (meas > vr + margin)
    is_seen = finite & (meas > vr - margin)
    np.add.at(free, vid[is_free], 1)
    np.add.at(seen, vid[is_seen], 1)
    if n % 200 == 0:
        print(f"  {n}/{len(test_keys)} scans, {time.time()-t0:.0f}s", flush=True)

ratio = free / np.maximum(seen, 1)
# height of each coarse voxel above the ground model
cgx = np.floor((cxyz[:, 0] - gx0) / gres).astype(np.int64)
cgy = np.floor((cxyz[:, 1] - gy0) / gres).astype(np.int64)
cin = (cgx >= 0) & (cgx < ground.shape[1]) & (cgy >= 0) & (cgy < ground.shape[0])
hag = np.full(len(cxyz), np.inf)
hag[cin] = cxyz[cin, 2] - ground[cgy[cin], cgx[cin]]
np.savez(os.path.join(OUTD, "votes.npz"), free=free, seen=seen, hag=hag)
for lo, hi in ((0.0, 0.25), (0.25, 0.5), (0.5, 0.75), (0.75, 0.9), (0.9, 1.01)):
    m = (seen >= 4) & (ratio >= lo) & (ratio < hi)
    print(f"  ratio {lo:.2f}-{hi:.2f}: {m.sum():8d} voxels, median height {np.median(hag[m]) if m.any() else 0:.1f} m")
# moving traffic / pedestrians: almost always seen through, inside the 0.25-4 m band above ground;
# semi-transparent vegetation is seen through only part of the time and mostly sits higher
dyn_c = (free >= 3) & (ratio > 0.75) & (hag > 0.2) & (hag < 4.0)
print(f"dynamic coarse voxels: {dyn_c.sum()} / {len(uk)} ({dyn_c.mean()*100:.2f}%)")
dyn = np.zeros(len(xyz_all), bool)
dyn[cand] = dyn_c[cinv]
off = 0
for f, n in zip(files, n_of):
    np.save(os.path.join(OUTD, os.path.basename(f).replace(".npz", ".npy")), dyn[off:off + n])
    off += n
print(f"dynamic fine voxels removed: {dyn.sum()} ({dyn.mean()*100:.2f}%), {time.time()-t0:.0f}s")
