#!/usr/bin/env python3
"""Write the static point cloud map: full-route PCD (0.1 m), a 0.2 m copy and 100 m tiles + metadata.

Fields: x y z intensity (float32), intensity = Ouster calibrated reflectivity.
Frame: the run's local map frame (UTM - (OX, OY)), z = NovAtel orthometric height (same as the Lanelet2 map).

usage: export_pcd.py MAP_DIR OUT_DIR
"""
import glob
import os
import sys

import numpy as np

MAP, OUT = sys.argv[1:3]
TILE_DIR = os.path.join(OUT, "pointcloud_map_tiles")
os.makedirs(TILE_DIR, exist_ok=True)


def write_pcd(path, xyz, inten):
    n = len(xyz)
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z intensity\nSIZE 4 4 4 4\n"
              "TYPE F F F F\nCOUNT 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n" % (n, n))
    rec = np.empty(n, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("i", "<f4")])
    rec["x"], rec["y"], rec["z"], rec["i"] = xyz[:, 0], xyz[:, 1], xyz[:, 2], inten
    with open(path, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())


def voxel_down(xyz, inten, res):
    v = np.floor(xyz / res).astype(np.int64)
    v -= v.min(axis=0)
    span = v.max(axis=0) + 1
    key = (v[:, 0] * span[1] + v[:, 1]) * span[2] + v[:, 2]
    uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
    c = np.column_stack([np.bincount(inv, xyz[:, k].astype(np.float64), len(uk)) for k in range(3)]) / cnt[:, None]
    return c.astype(np.float32), (np.bincount(inv, inten.astype(np.float64), len(uk)) / cnt).astype(np.float32)


all_xyz, all_i, meta = [], [], []
n_dyn = n_tot = 0
for f in sorted(glob.glob(os.path.join(MAP, "tiles", "tile_*.npz"))):
    t = np.load(f)
    dyn = np.load(os.path.join(MAP, "dynamic", os.path.basename(f).replace(".npz", ".npy")))
    keep = ~dyn
    xyz, inten = t["xyz"][keep], t["refl"][keep]
    n_dyn += dyn.sum()
    n_tot += len(dyn)
    if not len(xyz):
        continue
    ix, iy = map(int, os.path.basename(f)[5:-4].split("_"))
    name = f"tile_{ix}_{iy}.pcd"
    write_pcd(os.path.join(TILE_DIR, name), xyz, inten)
    meta.append((name, ix * 100.0, iy * 100.0))
    all_xyz.append(xyz)
    all_i.append(inten)
xyz = np.concatenate(all_xyz)
inten = np.concatenate(all_i)
write_pcd(os.path.join(OUT, "pointcloud_map.pcd"), xyz, inten)
x2, i2 = voxel_down(xyz, inten, 0.2)
write_pcd(os.path.join(OUT, "pointcloud_map_0.2m.pcd"), x2, i2)
with open(os.path.join(TILE_DIR, "pointcloud_map_metadata.yaml"), "w") as f:
    f.write("x_resolution: 100.0\ny_resolution: 100.0\n")
    for name, x0, y0 in meta:
        f.write(f"{name}: [{x0:.1f}, {y0:.1f}]\n")
print(f"voxels {n_tot}, dynamic removed {n_dyn} ({n_dyn/n_tot*100:.1f}%), written {len(xyz)} points "
      f"(0.2 m copy: {len(x2)}), {len(meta)} tiles")
print(f"extent x {xyz[:,0].min():.1f}..{xyz[:,0].max():.1f}  y {xyz[:,1].min():.1f}..{xyz[:,1].max():.1f}  "
      f"z {xyz[:,2].min():.1f}..{xyz[:,2].max():.1f}")
