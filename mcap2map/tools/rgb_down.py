#!/usr/bin/env python3
"""Voxel-downsample an x y z rgb intensity PCD (mean position, mean colour, mean intensity per voxel).
usage: rgb_down.py IN_PCD OUT_PCD RES_M"""
import sys
import numpy as np

src, dst, res = sys.argv[1], sys.argv[2], float(sys.argv[3])
with open(src, "rb") as f:
    hdr = b""
    while not hdr.endswith(b"DATA binary\n"):
        hdr += f.read(1)
    rec = np.frombuffer(f.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4"), ("i", "<f4")])
xyz = np.column_stack([rec["x"], rec["y"], rec["z"]]).astype(np.float64)
pk = rec["rgb"].view(np.uint32)
col = np.column_stack([(pk >> 16) & 255, (pk >> 8) & 255, pk & 255]).astype(np.float64)
v = np.floor(xyz / res).astype(np.int64)
v -= v.min(axis=0)
span = v.max(axis=0) + 1
key = (v[:, 0] * span[1] + v[:, 1]) * span[2] + v[:, 2]
uk, inv, cnt = np.unique(key, return_inverse=True, return_counts=True)
mean = lambda a: np.bincount(inv, a, len(uk)) / cnt
out = np.empty(len(uk), dtype=rec.dtype)
out["x"], out["y"], out["z"] = mean(xyz[:, 0]), mean(xyz[:, 1]), mean(xyz[:, 2])
c = np.clip(np.round(np.column_stack([mean(col[:, k]) for k in range(3)])), 0, 255).astype(np.uint32)
out["rgb"] = ((c[:, 0] << 16) | (c[:, 1] << 8) | c[:, 2]).view(np.float32)
out["i"] = mean(rec["i"].astype(np.float64))
n = len(out)
with open(dst, "wb") as f:
    f.write(("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb intensity\nSIZE 4 4 4 4 4\n"
             "TYPE F F F F F\nCOUNT 1 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n" % (n, n)).encode())
    f.write(out.tobytes())
print(f"{len(rec)} -> {n} points at {res} m")
