#!/usr/bin/env python3
"""Measure how well Lanelet2 boundaries match the point cloud map (paint centres / curb steps).

usage: boundary_check.py MAP_OSM PCD RASTER_NPZ [OUT_DIR for zoom renders]
"""
import os
import sys

import cv2
import numpy as np
import lanelet2
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcd_refine import STEP, SurfacePoints, curb_offset, frame, paint_offset, read_pcd_xyzi, resample, robust_spline
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import LAT0, LON0

MAP, PCD, RAS = sys.argv[1:4]
OUTD = sys.argv[4] if len(sys.argv) > 4 else None
lmap, _errs = lanelet2.io.loadRobust(MAP, UtmProjector(Origin(LAT0, LON0)))
z = np.load(RAS)
x0, y0, res, ground = float(z["x0"]), float(z["y0"]), float(z["res"]), z["ground"]

# boundaries of road lanelets, with the side the lanelet lies on
bounds = {}
for ll in lmap.laneletLayer:
    if ll.attributes["subtype"] != "road" or "turn_direction" in ll.attributes:
        continue
    for side, ls in (("left", ll.leftBound), ("right", ll.rightBound)):
        b = bounds.setdefault(ls.id, {"xy": np.array([[q.x, q.y] for q in ls]), "type": ls.attributes["type"],
                                      "sub": ls.attributes["subtype"] if "subtype" in ls.attributes else "",
                                      "inverted": ls.inverted(), "sides": []})
        b["sides"].append((side, ls.inverted()))

mask = np.zeros(ground.shape, np.uint8)
for b in bounds.values():
    pts = np.round((b["xy"] - [x0, y0]) / res).astype(np.int32)
    cv2.polylines(mask, [pts.reshape(-1, 1, 2)], False, 1, thickness=int(3.0 / res))
pcd = read_pcd_xyzi(PCD)
sp = SurfacePoints(pcd, ground, x0, y0, res, mask.astype(bool))
print(f"surface points in corridor: {len(sp.z)}")

rows = {"paint": [], "curb": []}
n_st = {"paint": 0, "curb": 0}
n_lid = {}
for lid, b in bounds.items():
    xy = b["xy"]
    if b["inverted"]:
        xy = xy[::-1]
    st_xy, st = resample(xy, STEP)
    _, t, nrm = frame(st_xy)
    if b["type"] == "line_thin":
        kind = "paint"
    elif b["type"] == "road_border" and len(b["sides"]) == 1:
        kind = "curb"
        side, inv = b["sides"][0]
        # lanelet on the left of a (stored-orientation) right bound, etc.
        road_sign = 1 if side == "right" else -1
        if inv != b["inverted"]:
            road_sign *= -1
    else:
        continue
    n_lid[lid] = len(st)
    for k in range(len(st)):
        n_st[kind] += 1
        r = paint_offset(sp, st_xy[k], t[k], nrm[k], 0.6, 0.5) if kind == "paint" else \
            curb_offset(sp, st_xy[k], t[k], nrm[k], road_sign)
        if r is not None:
            rows[kind].append((lid, st[k], r[0], st_xy[k][0], st_xy[k][1]))
for kind in ("paint", "curb"):
    a = np.array(rows[kind])
    if not len(a):
        print(f"{kind}: no measurements")
        continue
    off = np.abs(a[:, 2])
    print(f"{kind:5s}: {len(a)}/{n_st[kind]} stations with PCD evidence ({len(a)/max(n_st[kind],1)*100:.0f}%), "
          f"|offset| median {np.median(off)*100:.1f} cm, p90 {np.percentile(off,90)*100:.1f}, "
          f"p95 {np.percentile(off,95)*100:.1f}; >2 cm {np.mean(off>0.02)*100:.0f}%, >5 cm {np.mean(off>0.05)*100:.0f}%, "
          f">10 cm {np.mean(off>0.10)*100:.0f}%")
    per = []
    worst = []
    for lid in np.unique(a[:, 0]):
        sel_ = a[:, 0] == lid
        o_ = np.abs(a[sel_, 2])
        if len(o_) >= 10:
            per.append((np.median(o_), len(o_)))
            mid = a[sel_][len(o_) // 2]
            worst.append((np.median(o_), int(lid), len(o_), n_lid.get(int(lid), 0), mid[3], mid[4], bounds[int(lid)]["sub"]))
    if os.environ.get("CHECK_WORST"):
        for w_ in sorted(worst, reverse=True)[:14]:
            print(f"   worst {kind} ls {w_[1]}: median |off| {w_[0]*100:.1f} cm over {w_[2]}/{w_[3]} stations at ({w_[4]:.0f},{w_[5]:.0f}) {w_[6]}")
    # offset of each boundary to its own consistent line: robust spline through its station offsets,
    # stations explained by other markings (zig-zags, arrows, hatching) fall out as outliers
    own = []
    for lid in np.unique(a[:, 0]):
        sel_ = a[:, 0] == lid
        if sel_.sum() < 10:
            continue
        sp_ = robust_spline(a[sel_, 1], a[sel_, 2], np.ones(sel_.sum()))
        if sp_ is None:
            continue
        r_ = a[sel_, 2] - sp_(a[sel_, 1])
        inl_ = np.abs(r_) < 0.04
        if inl_.mean() < 0.5:
            continue
        own.append((np.median(np.abs(sp_(a[sel_, 1][inl_]))), inl_.sum()))
        if os.environ.get("CHECK_WORST") and own[-1][0] > 0.02:
            mid = a[sel_][sel_.sum() // 2]
            print(f"   own-line offender {kind} ls {int(lid)}: {own[-1][0]*100:.1f} cm over {inl_.sum()} inlier stations "
                  f"(signed median {np.median(sp_(a[sel_, 1][inl_]))*100:+.1f}) at ({mid[3]:.0f},{mid[4]:.0f}) {bounds[int(lid)]['sub']}")
    own = np.array(own)
    if len(own):
        print(f"       own-line fit: {len(own)} boundaries with a consistent paint line; |offset| to it <= 1 cm: "
              f"{np.sum(own[own[:,0]<=0.01,1])/np.sum(own[:,1])*100:.0f}%, <= 2 cm: {np.sum(own[own[:,0]<=0.02,1])/np.sum(own[:,1])*100:.0f}%, "
              f"<= 5 cm: {np.sum(own[own[:,0]<=0.05,1])/np.sum(own[:,1])*100:.0f}% (length-weighted)")
    per = np.array(per)
    if len(per):
        print(f"       per boundary (>=10 stations with evidence): {len(per)} boundaries, median |offset| <= 2 cm for "
              f"{np.mean(per[:,0] <= 0.02)*100:.0f}% of them (length-weighted {np.sum(per[per[:,0]<=0.02,1])/np.sum(per[:,1])*100:.0f}%)")

if OUTD:
    # zoom renders: PCD intensity at 2 cm/px with the boundaries
    a = np.array(rows["paint"])
    worst = a[np.argsort(-np.abs(a[:, 2]))]
    picks = [worst[0, 3:5], worst[len(worst) // 50, 3:5], a[len(a) // 3, 3:5], a[2 * len(a) // 3, 3:5]]
    for n, (cx, cy) in enumerate(picks):
        R, px = 6.0, 0.02
        idx = np.array(sp.kd.query_ball_point([cx, cy], R * 1.5))
        img = np.zeros((int(2 * R / px), int(2 * R / px), 3), np.uint8)
        q = sp.xy[idx]
        u = ((q[:, 0] - cx + R) / px).astype(int)
        v = (img.shape[0] - 1 - (q[:, 1] - cy + R) / px).astype(int)
        ok = (u >= 0) & (u < img.shape[1] - 4) & (v >= 0) & (v < img.shape[0] - 4)
        val = np.clip(sp.i[idx] * 6, 0, 255).astype(np.uint8)
        for du in range(4):
            for dv in range(4):
                img[v[ok] + dv, u[ok] + du] = val[ok][:, None]
        for b in bounds.values():
            pl = np.column_stack([(b["xy"][:, 0] - cx + R) / px, img.shape[0] - 1 - (b["xy"][:, 1] - cy + R) / px])
            col = (0, 0, 255) if b["type"] == "line_thin" else (255, 0, 255)
            cv2.polylines(img, [np.round(pl).astype(np.int32).reshape(-1, 1, 2)], False, col, 1, cv2.LINE_AA)
        cv2.putText(img, f"({cx:.1f},{cy:.1f}) 2 cm/px", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255), 2)
        cv2.imwrite(os.path.join(OUTD, f"zoom_{n}.png"), img)
    print("zoom renders written")
