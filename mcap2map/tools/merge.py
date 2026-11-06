#!/usr/bin/env python3
"""Combine the per-slice outputs of extract.py into one trajectory/topic file and global scan/image indexes.

usage: merge.py EXTRACT_DIR
writes EXTRACT_DIR/traj_raw.npz (odom, utm, fix, imu, tf, tfs, speed, steer, vs, info, pair_names, scan_meta)
       EXTRACT_DIR/scan_index.npy  (per global scan: slice, point offset, n points, header stamp, log time)
       EXTRACT_DIR/image_index.npz (cam, header stamp, log time, log time ns)
"""
import glob
import os
import sys

import numpy as np

D = sys.argv[1]
smalls = sorted(glob.glob(os.path.join(D, "small_*.npz")))
K = len(smalls)
out = {}
names_all = []
for f in smalls:
    names_all += [str(n) for n in np.load(f)["pair_names"]]
names = sorted(set(names_all))
for key in ("odom", "utm", "fix", "imu", "tf", "tfs", "speed", "steer", "vs", "info"):
    parts = []
    for f in smalls:
        z = np.load(f)
        a = z[key]
        if a.ndim != 2:
            continue
        if key in ("tf", "tfs") and len(a):  # remap pair ids to the global name table
            local = [str(n) for n in z["pair_names"]]
            a = a.copy()
            a[:, 2] = [names.index(local[int(i)]) for i in a[:, 2]]
        parts.append(a)
    a = np.concatenate(parts) if parts else np.zeros((0, 1))
    if key == "info":
        a = a[np.lexsort((a[:, 1], a[:, 0]))]
    else:
        a = a[np.argsort(a[:, 1], kind="stable")]  # by log time
    out[key] = a
    print(f"{key:6s} {a.shape}")

rows = []
for k in range(K):
    m = np.load(os.path.join(D, f"scans_{k:02d}_meta.npy"))
    off = np.r_[0, np.cumsum(m[:, 2])[:-1]].astype(np.int64)
    for j in range(len(m)):
        rows.append((k, off[j], m[j, 2], m[j, 0], m[j, 1]))
idx = np.array(rows, dtype=np.float64)
idx = idx[np.argsort(idx[:, 4], kind="stable")]
np.save(os.path.join(D, "scan_index.npy"), idx)
out["scan_meta"] = np.column_stack([np.arange(len(idx)), idx[:, 3], idx[:, 4], idx[:, 2]])
out["pair_names"] = np.array(names)
np.savez(os.path.join(D, "traj_raw.npz"), **out)
print(f"scans {len(idx)}, points {int(idx[:, 2].sum())}, header dt median {np.median(np.diff(idx[:, 3]))*1e3:.1f} ms, "
      f"max gap {np.diff(idx[:, 3]).max():.3f} s")

cam, st, lt, ln = [], [], [], []
for k in range(K):
    z = np.load(os.path.join(D, f"images_{k:02d}.npz"))
    cam.append(z["cam"]); st.append(z["stamp"]); lt.append(z["log"]); ln.append(z["log_ns"])
cam, st, lt, ln = map(np.concatenate, (cam, st, lt, ln))
o = np.lexsort((ln, cam))
np.savez(os.path.join(D, "image_index.npz"), cam=cam[o], stamp=st[o], log=lt[o], log_ns=ln[o])
for c in np.unique(cam):
    m = cam == c
    print(f"cam{c}: {m.sum()} images, stamp-log latency median {np.median(lt[m]-st[m])*1e3:.0f} ms "
          f"(p5 {np.percentile(lt[m]-st[m],5)*1e3:.0f}, p95 {np.percentile(lt[m]-st[m],95)*1e3:.0f})")
print("tf pairs:", names)
