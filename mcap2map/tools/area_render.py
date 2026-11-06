#!/usr/bin/env python3
"""Render a map area from the PCD road surface (intensity, height step in blue) with the Lanelet2 boundaries,
lanelet ids, the driven trajectory and the OSM reference lines.

usage: area_render.py MAP_OSM PCD RASTER_NPZ TRAJ_NPZ ROUTE_JSON OUT_PNG xa,xb,ya,yb [px_m] [ids]
"""
import json
import os
import sys

import cv2
import numpy as np
import lanelet2
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pcd_refine import SurfacePoints, read_pcd_xyzi
from lanegeom import fine_centerline
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import LAT0, LON0

MAP, PCD, RAS, TRAJ, ROUTE, OUT = sys.argv[1:7]
xa, xb, ya, yb = map(float, sys.argv[7].split(","))
PX = float(sys.argv[8]) if len(sys.argv) > 8 else 0.03
SHOW_IDS = len(sys.argv) > 9 and sys.argv[9] == "ids"
lmap, _errs = lanelet2.io.loadRobust(MAP, UtmProjector(Origin(LAT0, LON0)))
z = np.load(RAS)
x0, y0, res, ground = float(z["x0"]), float(z["y0"]), float(z["res"]), z["ground"]
mask = np.zeros(ground.shape, bool)
i0, i1 = int((xa - 1 - x0) / res), int((xb + 1 - x0) / res)
j0, j1 = int((ya - 1 - y0) / res), int((yb + 1 - y0) / res)
mask[max(j0, 0):j1, max(i0, 0):i1] = True
sp = SurfacePoints(read_pcd_xyzi(PCD), ground, x0, y0, res, mask, zlo=-0.25, zhi=0.30)
W, H = int((xb - xa) / PX), int((yb - ya) / PX)
img = np.zeros((H, W, 3), np.uint8)


def to_px(xy):
    xy = np.atleast_2d(xy)
    return np.column_stack([(xy[:, 0] - xa) / PX, H - 1 - (xy[:, 1] - ya) / PX])


q = sp.xy
u = ((q[:, 0] - xa) / PX).astype(int)
v = (H - 1 - (q[:, 1] - ya) / PX).astype(int)
ok = (u >= 0) & (u < W - 3) & (v >= 0) & (v < H - 3)
dz = sp.z - sp.g
val = np.clip(sp.i * 6, 0, 255).astype(np.uint8)
col = np.stack([val, val, val], 1)
hi = dz > float(os.environ.get("HI_T", "0.15"))  # kerb tops / objects (raster ground sits a few cm low)
col[hi] = np.stack([val[hi] // 3, val[hi] // 2, np.full(hi.sum(), 150)], 1)
for du in range(3):
    for dv in range(3):
        img[v[ok] + dv, u[ok] + du] = np.maximum(img[v[ok] + dv, u[ok] + du], col[ok])
# OSM reference lines (thin blue)
net = json.load(open(ROUTE))
for c in net["chains"]:
    pl = to_px(np.array(c["xy"]))
    cv2.polylines(img, [np.round(pl * 4).astype(np.int32).reshape(-1, 1, 2)], False, (255, 120, 0), 1, cv2.LINE_AA, shift=2)
# trajectory
traj = np.load(TRAJ)["pos"][::50]
for p in to_px(traj[:, :2]):
    if 0 <= p[0] < W and 0 <= p[1] < H:
        cv2.circle(img, (int(p[0]), int(p[1])), 2, (0, 255, 0), -1)
# lanelet boundaries
for ls in lmap.lineStringLayer:
    t = ls.attributes["type"] if "type" in ls.attributes else ""
    c = {"line_thin": (0, 0, 255), "road_border": (255, 0, 255), "virtual": (0, 165, 255),
         "stop_line": (0, 255, 0), "pedestrian_marking": (255, 255, 0)}.get(t)
    if c is None:
        continue
    xy = np.array([[p.x, p.y] for p in ls])
    pl = to_px(xy)
    sub = ls.attributes["subtype"] if "subtype" in ls.attributes else ""
    cv2.polylines(img, [np.round(pl * 4).astype(np.int32).reshape(-1, 1, 2)], False, c, 2 if sub == "solid_solid" else 1,
                  cv2.LINE_AA, shift=2)
    for p in pl:  # vertices
        if 0 <= p[0] < W and 0 <= p[1] < H:
            cv2.circle(img, (int(p[0]), int(p[1])), 2, c, 1)
for ll in lmap.laneletLayer:
    cl = fine_centerline(ll)
    if len(cl) < 2:
        print("degenerate lanelet", ll.id)
        continue
    conn = "turn_direction" in ll.attributes
    pl = to_px(cl)
    colr = (0, 255, 255) if conn else (255, 255, 0)
    cv2.polylines(img, [np.round(pl * 4).astype(np.int32).reshape(-1, 1, 2)], False, colr, 1, cv2.LINE_AA, shift=2)
    k = len(cl) // 2
    if len(cl) > 1:
        a, b = pl[max(k - 1, 0)], pl[min(k + 1, len(cl) - 1)]
        if 0 <= b[0] < W and 0 <= b[1] < H:
            cv2.arrowedLine(img, (int(a[0]), int(a[1])), (int(b[0]), int(b[1])), colr, 1, tipLength=0.6)
    if SHOW_IDS and 0 <= pl[k][0] < W and 0 <= pl[k][1] < H:
        cv2.putText(img, str(ll.id), (int(pl[k][0]) + 3, int(pl[k][1]) - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.4, colr, 1)
for k in range(5):  # 5 m scale bar
    cv2.line(img, (10, H - 20 + k // 3), (10 + int(5 / PX), H - 20 + k // 3), (0, 255, 255), 1)
cv2.putText(img, f"{os.path.basename(MAP)} x {xa:.0f}..{xb:.0f} y {ya:.0f}..{yb:.0f}  5 m bar", (10, 22),
            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
cv2.imwrite(OUT, img)
print("wrote", OUT, img.shape)
