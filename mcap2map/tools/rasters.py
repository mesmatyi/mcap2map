#!/usr/bin/env python3
"""Road-surface rasters from the voxel map: ground height and ground reflectivity around the route.

The road surface is searched in a window around the height of the vehicle's wheels (nearest trajectory
point), so bridge decks and the ground beneath them are not confused.

usage: rasters.py TILES_DIR TRAJ_NPZ OUT_NPZ
"""
import glob
import sys

import os

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

TILES, TRAJ, OUT = sys.argv[1:4]
RES = 0.1          # raster resolution [m]
CORRIDOR = 30.0    # max distance from trajectory [m]
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import GROUND_HEIGHT
WHEEL_DROP = GROUND_HEIGHT  # base_link height above ground (base_link -> ground_link in /tf_static)

z = np.load(TRAJ)
pos = z["pos"][::100]  # 10 Hz is plenty
ground_ref = pos[:, 2] - WHEEL_DROP
kd = cKDTree(pos[:, :2])

xs, ys, zs, rs = [], [], [], []
for f in sorted(glob.glob(f"{TILES}/tile_*.npz")):
    t = np.load(f)
    p = t["xyz"].astype(np.float64)
    d, i = kd.query(p[:, :2], distance_upper_bound=CORRIDOR)
    m = np.isfinite(d)
    i = np.where(m, i, 0)
    dz = p[:, 2] - ground_ref[i]
    # road surface window: allow cross slope / camber growing with distance
    m &= (dz > -0.6 - 0.04 * d) & (dz < 0.4 + 0.04 * d)
    xs.append(p[m, 0]); ys.append(p[m, 1]); zs.append(p[m, 2]); rs.append(t["refl"][m])
x, y, zz, rf = map(np.concatenate, (xs, ys, zs, rs))
x0, y0 = np.floor(x.min()) - 1, np.floor(y.min()) - 1
W = int(np.ceil((x.max() - x0) / RES)) + 2
H = int(np.ceil((y.max() - y0) / RES)) + 2
ix = ((x - x0) / RES).astype(np.int64)
iy = ((y - y0) / RES).astype(np.int64)
lin = iy * W + ix
print(f"raster {W} x {H} @ {RES} m, {len(x)} candidate points")

# ground height: low percentile per 0.5 m cell, then fill/smooth
C = 5
cl = (iy // C) * (W // C + 1) + ix // C
order = np.lexsort((zz, cl))
cl_s, z_s = cl[order], zz[order]
starts = np.r_[0, np.flatnonzero(np.diff(cl_s)) + 1]
counts = np.diff(np.r_[starts, len(cl_s)])
p10 = z_s[starts + (counts * 0.1).astype(np.int64)]
gc = np.full((H // C + 1) * (W // C + 1), np.nan)
gc[cl_s[starts]] = np.where(counts >= 3, p10, np.nan)
gc = gc.reshape(H // C + 1, W // C + 1)
valid = np.isfinite(gc)
idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
gfill = gc[tuple(idx)]
gfill = ndimage.median_filter(gfill, size=5)
gfill = ndimage.uniform_filter(gfill, size=3)
ground = np.kron(gfill, np.ones((C, C)))[:H, :W]

# ground points: within 0.12 m of the ground model
gz = ground.reshape(-1)[lin]
gm = np.abs(zz - gz) < 0.12
cnt = np.bincount(lin[gm], minlength=W * H).reshape(H, W)
refl = (np.bincount(lin[gm], rf[gm].astype(np.float64), minlength=W * H).reshape(H, W) / np.maximum(cnt, 1))
refl[cnt == 0] = np.nan
# fine surface height (0.1 m): 30th percentile of near-ground points per cell, for curb detection
near = np.abs(zz - gz) < 0.35
order = np.lexsort((zz[near], lin[near]))
ln_s, zf_s = lin[near][order], zz[near][order]
st_f = np.r_[0, np.flatnonzero(np.diff(ln_s)) + 1]
cn_f = np.diff(np.r_[st_f, len(ln_s)])
zfine = np.full(W * H, np.nan, np.float32)
zfine[ln_s[st_f]] = zf_s[st_f + (cn_f * 0.3).astype(np.int64)]
zfine = zfine.reshape(H, W)
cover = ndimage.binary_dilation(valid, iterations=1)
cover = np.kron(cover, np.ones((C, C), bool))[:H, :W]
np.savez_compressed(OUT, x0=x0, y0=y0, res=RES, ground=ground.astype(np.float32), refl=refl.astype(np.float32), zfine=zfine,
                    count=cnt.astype(np.uint16), cover=cover)
print(f"ground cells with points: {(cnt > 0).sum()}, reflectivity p50 {np.nanpercentile(refl, 50):.1f} "
      f"p95 {np.nanpercentile(refl, 95):.1f} p99 {np.nanpercentile(refl, 99):.1f}")
