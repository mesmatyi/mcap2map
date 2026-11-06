#!/usr/bin/env python3
"""Colour the static point cloud map from the two cameras.

For camera keyframes (every ~1.2 m of travel, every 5 s when stationary), map points within 60 m are projected with
the calibrated camera extrinsics and time offset; a depth buffer built from the map points *and* the synchronous
lidar scan rejects points hidden behind foreground objects (incl. moving vehicles that are no longer in the map).
Colours are averaged with weights favouring close range and the image centre. Points never seen by a camera get
the colour of their nearest coloured neighbour (within 5 m), else a reflectivity grey.

usage: colorize.py EXTRACT_DIR MAP_PCD TRAJ_NPZ LIDAR_CALIB_JSON CAM_CALIB_JSON OUT_PCD [N_WORKERS]
"""
import json
import multiprocessing as mp
import os
import sys
import time

import cv2
import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from camlib import NCAM, Images, model, project, read_pcd, size
from maplib import ScanCache, Trajectory, deskew_to_map

EXT, PCD, TRAJ, CALIB, CAMCAL, OUT = sys.argv[1:7]
NW = int(sys.argv[7]) if len(sys.argv) > 7 else 8
RMAX, KEY_DIST = 60.0, 1.2
IMG_REDUCE, ZB_SCALE = 2, 0.25
GRID = 10.0

PTS = read_pcd(PCD)
NP = len(PTS)
tr = Trajectory(TRAJ)
calib = json.load(open(CALIB))
EXTR, DT_L = np.array(calib["extrinsic_base_to_lidar"]), calib["dt_lidar_ms"] * 1e-3
camcal = {c["index"]: c for c in json.load(open(CAMCAL))["cameras"]}
imgs = Images(EXT)

gk = (np.floor(PTS[:, 0] / GRID).astype(np.int64) * 100000 + np.floor(PTS[:, 1] / GRID).astype(np.int64))
ORDER = np.argsort(gk, kind="stable")
gk_s = gk[ORDER]
GU, GS = np.unique(gk_s, return_index=True)
GE = np.r_[GS[1:], len(gk_s)]
GMAP = dict(zip(GU.tolist(), zip(GS.tolist(), GE.tolist())))
del gk, gk_s


def near(p, r):
    cx, cy, k = int(np.floor(p[0] / GRID)), int(np.floor(p[1] / GRID)), int(np.ceil(r / GRID))
    out = [ORDER[a:b] for i in range(cx - k, cx + k + 1) for j in range(cy - k, cy + k + 1)
           for a, b in [GMAP.get(i * 100000 + j, (0, 0))] if b > a]
    return np.concatenate(out) if out else np.zeros(0, np.int64)


def keyframes(cam):
    cc = camcal[cam]
    Tbc, dtc = np.array(cc["base_T_cam"]), cc["dt_cam_ms"] * 1e-3
    keys, last_p, last_t = [], None, -1e9
    for i in imgs.of(cam):
        ts = imgs.stamp[i] + dtc
        if not tr.valid(ts):
            continue
        p = (tr.pose(ts) @ Tbc)[:3, 3]
        if last_p is not None and np.linalg.norm(p - last_p) < KEY_DIST and ts - last_t < 5.0:
            continue
        keys.append(i)
        last_p, last_t = p, ts
    return keys


def worker(args):
    cam, frames, wid = args
    cc = camcal[cam]
    Tbc, dtc = np.array(cc["base_T_cam"]), cc["dt_cam_ms"] * 1e-3
    mdl = cc.get("model", model(cam))
    W, H = size(cam)
    sc = ScanCache(EXT)
    hs, ws = int(H * ZB_SCALE), int(W * ZB_SCALE)
    acc = [np.zeros(0, np.int64), np.zeros(0, np.float32), np.zeros((0, 3), np.float32)]
    buf, nbuf = [], 0

    def reduce_buf():
        nonlocal buf, nbuf
        if not buf:
            return
        ii = np.concatenate([acc[0]] + [b[0] for b in buf])
        ww = np.concatenate([acc[1]] + [b[2] for b in buf])
        cc_ = np.concatenate([acc[2]] + [b[1].astype(np.float32) * b[2][:, None] for b in buf])
        uk, inv = np.unique(ii, return_inverse=True)
        acc[0] = uk
        acc[1] = np.bincount(inv, ww, len(uk)).astype(np.float32)
        acc[2] = np.column_stack([np.bincount(inv, cc_[:, c], len(uk)) for c in range(3)]).astype(np.float32)
        buf, nbuf = [], 0

    for i in frames:
        ts = imgs.stamp[i] + dtc
        Tmc = tr.pose(ts) @ Tbc
        p = Tmc[:3, 3]
        img = imgs.color(i, IMG_REDUCE)
        idx = near(p, RMAX)
        if not len(idx):
            continue
        pc = (PTS[idx, :3].astype(np.float64) - p) @ Tmc[:3, :3]
        u, v, ok = project(cam, pc, 1.0, mdl)
        d = pc[:, 2]
        ok &= d < RMAX
        idx, u, v, d = idx[ok], u[ok], v[ok], d[ok]
        if not len(idx):
            continue
        zi = np.clip((v * ZB_SCALE).astype(np.int64), 0, hs - 1)
        zj = np.clip((u * ZB_SCALE).astype(np.int64), 0, ws - 1)
        D = np.full((hs, ws), np.inf)
        o = np.argsort(-d)
        D[zi[o], zj[o]] = d[o]
        # the lidar scan closest in time (true scan mid time vs true image time) adds moving objects to the z-buffer
        k = int(np.argmin(np.abs(sc.stamp + DT_L + 0.05 - ts)))
        s = sc.get(k)
        rr = np.linalg.norm(s["xyz"], axis=1)
        mm = (rr > 2) & (rr < RMAX + 5)
        pm = deskew_to_map(s["xyz"][mm], s["t"][mm], sc.stamp[k], tr, EXTR, DT_L)
        sc_c = (pm - p) @ Tmc[:3, :3]
        su, sv, sok = project(cam, sc_c, 1.0, mdl)
        if sok.any():
            si = np.clip((sv[sok] * ZB_SCALE).astype(np.int64), 0, hs - 1)
            sj = np.clip((su[sok] * ZB_SCALE).astype(np.int64), 0, ws - 1)
            np.minimum.at(D, (si, sj), sc_c[sok, 2])
        Dm = cv2.erode(np.where(np.isfinite(D), D, 1e4).astype(np.float32), np.ones((3, 3), np.uint8))
        fpx = float(cc.get("focal_px", 2650.0))
        vis = d <= Dm[zi, zj] + np.maximum(0.15, 0.03 * d) + d * d / (1.6 * fpx * ZB_SCALE)
        idx, u, v, d = idx[vis], u[vis], v[vis], d[vis]
        ui = np.clip(np.round(u / IMG_REDUCE).astype(np.int64), 0, img.shape[1] - 1)
        vi = np.clip(np.round(v / IMG_REDUCE).astype(np.int64), 0, img.shape[0] - 1)
        col = img[vi, ui]  # BGR
        rn2 = ((u - W / 2) / (W / 2)) ** 2 + ((v - H / 2) / (H / 2)) ** 2
        w = (1.0 / (1.0 + (d / 10.0) ** 2)) * (1.0 - 0.4 * np.clip(rn2, 0, 1))
        buf.append((idx.astype(np.int64), col.astype(np.uint8), w.astype(np.float32)))
        nbuf += len(idx)
        if nbuf > 4_000_000:
            reduce_buf()
    reduce_buf()
    return wid, len(frames), acc


if __name__ == "__main__":
    t0 = time.time()
    jobs = []
    for cam in sorted(camcal):
        keys = keyframes(cam)
        print(f"camera{cam}: {len(keys)} keyframes", flush=True)
        for part in np.array_split(np.array(keys), NW * 2):
            if len(part):
                jobs.append((cam, part.tolist(), len(jobs)))
    sum_c = np.zeros((NP, 3), np.float32)
    sum_w = np.zeros(NP, np.float32)
    nframes = 0
    with mp.get_context("fork").Pool(NW) as pool:
        for wid, nfr, (idx, w, cw) in pool.imap_unordered(worker, jobs):
            sum_w[idx] += w
            sum_c[idx] += cw
            nframes += nfr
            print(f"  job {wid} done ({nfr} frames), total {nframes}, {time.time()-t0:.0f}s", flush=True)
    seen = sum_w > 0
    rgb_f = np.zeros((NP, 3), np.float32)
    rgb_f[seen] = (sum_c[seen] / sum_w[seen, None])[:, ::-1]  # BGR -> RGB
    # gentle global level stretch: the camera exposure is low (dark images)
    lum = rgb_f[seen] @ np.array([0.299, 0.587, 0.114], np.float32)
    lo, hi = np.percentile(lum, 1), np.percentile(lum, 99)
    gain = 235.0 / max(hi - lo, 1.0)
    rgb_f[seen] = np.clip((rgb_f[seen] - lo) * gain + 10.0, 0, 255)
    rgb = rgb_f.astype(np.uint8)
    print(f"directly coloured: {seen.sum()} / {NP} ({seen.mean()*100:.1f}%); level stretch lum {lo:.0f}..{hi:.0f} -> "
          f"10..245", flush=True)
    kdt = cKDTree(PTS[seen, :3])
    un = np.flatnonzero(~seen)
    dist, nn = kdt.query(PTS[un, :3], k=1, distance_upper_bound=5.0, workers=16)
    okf = np.isfinite(dist)
    rgb[un[okf]] = rgb[np.flatnonzero(seen)[nn[okf]]]
    g = np.clip(PTS[un[~okf], 3] * 6, 30, 230).astype(np.uint8)
    rgb[un[~okf]] = np.stack([g, g, g], 1)
    print(f"filled from neighbours: {okf.sum()} ({okf.sum()/NP*100:.1f}%), reflectivity grey: {(~okf).sum()} "
          f"({(~okf).sum()/NP*100:.1f}%)", flush=True)
    packed = (rgb[:, 0].astype(np.uint32) << 16) | (rgb[:, 1].astype(np.uint32) << 8) | rgb[:, 2].astype(np.uint32)
    rec = np.empty(NP, dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4"), ("intensity", "<f4")])
    rec["x"], rec["y"], rec["z"] = PTS[:, 0], PTS[:, 1], PTS[:, 2]
    rec["rgb"] = packed.view(np.float32)
    rec["intensity"] = PTS[:, 3]
    header = ("# .PCD v0.7 - Point Cloud Data file format\nVERSION 0.7\nFIELDS x y z rgb intensity\nSIZE 4 4 4 4 4\n"
              "TYPE F F F F F\nCOUNT 1 1 1 1 1\nWIDTH %d\nHEIGHT 1\nVIEWPOINT 0 0 0 1 0 0 0\nPOINTS %d\nDATA binary\n" % (NP, NP))
    with open(OUT, "wb") as f:
        f.write(header.encode("ascii"))
        f.write(rec.tobytes())
    np.save(os.path.splitext(OUT)[0] + "_seen.npy", seen)
    print(f"wrote {OUT} ({NP} points) in {time.time()-t0:.0f}s")
