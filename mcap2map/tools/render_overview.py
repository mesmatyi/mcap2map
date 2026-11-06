#!/usr/bin/env python3
"""Full-route pictures: top-down orthophoto of the coloured PCD with hillshade, with and without the
Lanelet2 map drawn on top, plus an oblique 3D view.

usage: render_overview.py RGB_PCD MAP_OSM OUT_DIR [res_m]
"""
import os
import sys

import cv2
import numpy as np
import lanelet2
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import FRAME, LAT0, LON0

PCD, MAP, OUTD = sys.argv[1:4]
RES = float(sys.argv[4]) if len(sys.argv) > 4 else 0.2
os.makedirs(OUTD, exist_ok=True)

with open(PCD, "rb") as f:
    hdr = b""
    while not hdr.endswith(b"DATA binary\n"):
        hdr += f.read(1)
    rec = np.frombuffer(f.read(), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("rgb", "<f4"), ("i", "<f4")])
x, y, z = rec["x"], rec["y"], rec["z"]
pk = rec["rgb"].view(np.uint32)
bgr = np.column_stack([pk & 255, (pk >> 8) & 255, (pk >> 16) & 255]).astype(np.uint8)
del rec
print(f"{len(x)} points, extent x {x.min():.0f}..{x.max():.0f} y {y.min():.0f}..{y.max():.0f}", flush=True)

# ---------------------------------------------------------------- top-down: highest point per cell + hillshade
PAD = 15.0
x0, y1 = float(np.floor(x.min() - PAD)), float(np.ceil(y.max() + PAD))
W = int(np.ceil((x.max() + PAD - x0) / RES))
H = int(np.ceil((y1 - (y.min() - PAD)) / RES))
ix = ((x - x0) / RES).astype(np.int64)
iy = ((y1 - y) / RES).astype(np.int64)
ok = (ix >= 0) & (ix < W) & (iy >= 0) & (iy < H)
lin = (iy * W + ix)[ok]
order = np.argsort(-z[ok], kind="stable")  # highest first
lin_s = lin[order]
uniq, first = np.unique(lin_s, return_index=True)
top = np.flatnonzero(ok)[order[first]]
img = np.zeros((H * W, 3), np.uint8)
img[uniq] = bgr[top]
zbuf = np.full(H * W, np.nan, np.float32)
zbuf[uniq] = z[top]
img = img.reshape(H, W, 3)
zbuf = zbuf.reshape(H, W)
filled = np.isfinite(zbuf)
# hillshade from the height buffer (light from the north-west), fill holes for the gradient only
zf = zbuf.copy()
zf[~filled] = np.nanmedian(zbuf)
zf = cv2.GaussianBlur(zf, (0, 0), 1.0)
gy, gx = np.gradient(zf, RES)
slope = np.arctan(np.hypot(gx, gy))
aspect = np.arctan2(-gx, gy)
alt, az = np.radians(50), np.radians(315)
shade = np.sin(alt) * np.cos(slope) + np.cos(alt) * np.sin(slope) * np.cos(az - aspect)
shade = np.clip(shade, 0, 1)
img = np.clip(img.astype(np.float32) * (0.55 + 0.75 * shade[..., None]), 0, 255).astype(np.uint8)
img[~filled] = (18, 18, 18)
# slight colour boost (the blended colours are a bit flat)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
hsv[..., 1] = np.clip(hsv[..., 1] * 1.35, 0, 255)
hsv[..., 2] = np.clip(hsv[..., 2] * 1.15, 0, 255)
img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
img[~filled] = (18, 18, 18)
print(f"orthophoto {W} x {H} px at {RES} m/px", flush=True)


def to_px(xy):
    xy = np.asarray(xy, float)
    return np.column_stack([(xy[:, 0] - x0) / RES, (y1 - xy[:, 1]) / RES])


def polyline(im, xy, color, thick, dashed=False, shift=3):
    p = np.round(to_px(xy) * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
    if not dashed:
        cv2.polylines(im, [p], False, color, thick, cv2.LINE_AA, shift=shift)
        return
    # dashes of 2 m with 1.5 m gaps along the line
    xy = np.asarray(xy, float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    t = 0.0
    while t < s[-1]:
        a, b = t, min(t + 2.0, s[-1])
        ss = np.linspace(a, b, max(2, int((b - a) / 0.5) + 1))
        d = np.column_stack([np.interp(ss, s, xy[:, 0]), np.interp(ss, s, xy[:, 1])])
        pp = np.round(to_px(d) * (1 << shift)).astype(np.int32).reshape(-1, 1, 2)
        cv2.polylines(im, [pp], False, color, thick, cv2.LINE_AA, shift=shift)
        t += 3.5


def deco(im, title):
    """Scale bar, north arrow, title."""
    Hh, Ww = im.shape[:2]
    bar = int(100 / RES)
    bx, by = 40, Hh - 50
    cv2.rectangle(im, (bx - 10, by - 44), (bx + bar + 10, by + 14), (0, 0, 0), -1)
    cv2.line(im, (bx, by), (bx + bar, by), (255, 255, 255), 4)
    for k in range(3):
        cv2.line(im, (bx + k * bar // 2, by - 8), (bx + k * bar // 2, by + 8), (255, 255, 255), 3)
    cv2.putText(im, "100 m", (bx + bar // 2 - 40, by - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    nx, ny = Ww - 70, 110
    cv2.rectangle(im, (nx - 40, ny - 80), (nx + 40, ny + 30), (0, 0, 0), -1)
    cv2.arrowedLine(im, (nx, ny + 20), (nx, ny - 55), (255, 255, 255), 4, cv2.LINE_AA, tipLength=0.3)
    cv2.putText(im, "N", (nx - 12, ny - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 255), 2, cv2.LINE_AA)
    (tw, th), _ = cv2.getTextSize(title, cv2.FONT_HERSHEY_SIMPLEX, 1.1, 2)
    cv2.rectangle(im, (30, 20), (50 + tw, 40 + th), (0, 0, 0), -1)
    cv2.putText(im, title, (40, 30 + th), cv2.FONT_HERSHEY_SIMPLEX, 1.1, (255, 255, 255), 2, cv2.LINE_AA)


pcd_img = img.copy()
deco(pcd_img, f"Point cloud map (RGB from cameras, {len(x)/1e6:.1f} M points, 0.1 m)  -  {FRAME} frame, {RES} m/px")
cv2.imwrite(os.path.join(OUTD, "map_overview_pcd.jpg"), pcd_img, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("wrote map_overview_pcd.jpg", flush=True)

# ---------------------------------------------------------------- Lanelet2 on top
lmap, _errs = lanelet2.io.loadRobust(MAP, UtmProjector(Origin(LAT0, LON0)))
ov = img.copy()
T = max(1, int(round(0.35 / RES)))
COL = {"solid": (60, 60, 255), "dashed": (60, 60, 255), "solid_solid": (0, 0, 255), "road_border": (255, 80, 255),
       "virtual": (0, 170, 255), "stop_line": (80, 255, 80), "pedestrian_marking": (255, 255, 0)}
# crosswalk fill
for ll in lmap.laneletLayer:
    if ll.attributes["subtype"] == "crosswalk":
        poly = np.round(to_px(np.array([[q.x, q.y] for q in ll.polygon2d()]))).astype(np.int32)
        layer = ov.copy()
        cv2.fillPoly(layer, [poly], (255, 220, 0))
        ov = cv2.addWeighted(ov, 0.55, layer, 0.45, 0)
# lane fill (very light) so lanelets read as areas
layer = ov.copy()
for ll in lmap.laneletLayer:
    if ll.attributes["subtype"] == "road":
        poly = np.round(to_px(np.array([[q.x, q.y] for q in ll.polygon2d()]))).astype(np.int32)
        cv2.fillPoly(layer, [poly], (90, 160, 90) if "turn_direction" not in ll.attributes else (60, 140, 170))
ov = cv2.addWeighted(ov, 0.8, layer, 0.2, 0)
drawn = set()
for ll in lmap.laneletLayer:
    if ll.attributes["subtype"] != "road":
        continue
    for b in (ll.leftBound, ll.rightBound):
        if b.id in drawn:
            continue
        drawn.add(b.id)
        t = b.attributes["type"] if "type" in b.attributes else ""
        sub = b.attributes["subtype"] if "subtype" in b.attributes else ""
        xy = np.array([[q.x, q.y] for q in b])
        if t == "line_thin":
            polyline(ov, xy, COL[sub if sub in COL else "solid"], T, dashed=(sub == "dashed"))
        elif t == "road_border":
            polyline(ov, xy, COL["road_border"], T)
        elif t == "virtual":
            polyline(ov, xy, COL["virtual"], max(1, T - 1))
    # direction arrow at the middle of the centreline
    cl = np.array([[q.x, q.y] for q in ll.centerline])
    seg = np.linalg.norm(np.diff(cl, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] > 6:
        for sm in np.arange(s[-1] / 2 % 25.0 + 3, s[-1] - 3, 25.0) if s[-1] > 40 else [s[-1] / 2]:
            a = np.array([np.interp(sm - 1.5, s, cl[:, 0]), np.interp(sm - 1.5, s, cl[:, 1])])
            b_ = np.array([np.interp(sm + 1.5, s, cl[:, 0]), np.interp(sm + 1.5, s, cl[:, 1])])
            pa, pb = to_px(a[None])[0], to_px(b_[None])[0]
            cv2.arrowedLine(ov, tuple(np.round(pa).astype(int)), tuple(np.round(pb).astype(int)),
                            (255, 255, 255), max(1, T - 1), cv2.LINE_AA, tipLength=0.5)
for ls in lmap.lineStringLayer:
    t = ls.attributes["type"] if "type" in ls.attributes else ""
    xy = np.array([[q.x, q.y] for q in ls])
    if t == "stop_line":
        polyline(ov, xy, COL["stop_line"], T + 1)
    elif t == "traffic_light":
        c = np.round(to_px(xy.mean(axis=0)[None])[0]).astype(int)
        cv2.circle(ov, tuple(c), max(3, int(1.2 / RES)), (0, 0, 0), -1)
        cv2.circle(ov, tuple(c), max(2, int(0.8 / RES)), (0, 60, 255), -1)
# street names
names = {}
for ll in lmap.laneletLayer:
    if "road_name" in ll.attributes and ll.attributes["road_name"]:
        cl = np.array([[q.x, q.y] for q in ll.centerline])
        L = np.sum(np.linalg.norm(np.diff(cl, axis=0), axis=1))
        key = (ll.attributes["road_name"], ll.attributes["osm_way"] if "osm_way" in ll.attributes else "")
        if key not in names or names[key][0] < L:
            names[key] = (L, cl[len(cl) // 2], cl[-1] - cl[0])
done_names = []
for (nm, _), (L, mid, d) in sorted(names.items(), key=lambda kv: -kv[1][0]):
    if L < 60 or any(n == nm and np.linalg.norm(m - mid) < 180 for n, m in done_names):
        continue
    done_names.append((nm, mid))
    ang = np.degrees(np.arctan2(d[1], d[0]))
    if ang > 90:
        ang -= 180
    if ang < -90:
        ang += 180
    (tw, th), _ = cv2.getTextSize(nm, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    canvas = np.zeros((th + 16, tw + 16, 3), np.uint8)
    cv2.putText(canvas, nm, (8, th + 6), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2, cv2.LINE_AA)
    M = cv2.getRotationMatrix2D((canvas.shape[1] / 2, canvas.shape[0] / 2), ang, 1.0)
    cos, sin = abs(M[0, 0]), abs(M[0, 1])
    nw, nh = int(canvas.shape[0] * sin + canvas.shape[1] * cos), int(canvas.shape[0] * cos + canvas.shape[1] * sin)
    M[0, 2] += nw / 2 - canvas.shape[1] / 2
    M[1, 2] += nh / 2 - canvas.shape[0] / 2
    rot = cv2.warpAffine(canvas, M, (nw, nh))
    c = np.round(to_px(mid[None])[0]).astype(int) + np.array([0, -int(9 / RES)])
    ya, xa = c[1] - nh // 2, c[0] - nw // 2
    if 0 <= ya and ya + nh < ov.shape[0] and 0 <= xa and xa + nw < ov.shape[1]:
        m = rot.max(axis=2) > 0
        # halo
        halo = cv2.dilate(m.astype(np.uint8), np.ones((5, 5), np.uint8)) > 0
        ov[ya:ya + nh, xa:xa + nw][halo] = (0, 0, 0)
        ov[ya:ya + nh, xa:xa + nw][m] = rot[m]
# legend
items = [("lane boundary (painted: solid / dashed)", COL["solid"]), ("double centre line", COL["solid_solid"]),
         ("road border (no paint)", COL["road_border"]), ("junction connector (virtual)", COL["virtual"]),
         ("stop line", COL["stop_line"]), ("traffic light", (0, 60, 255)), ("crosswalk", (255, 220, 0)),
         ("driving direction", (255, 255, 255))]
lx, ly = 40, 80
cv2.rectangle(ov, (lx - 10, ly - 10), (lx + 520, ly + 34 * len(items) + 4), (0, 0, 0), -1)
for k, (txt, col) in enumerate(items):
    yy = ly + 34 * k + 18
    if txt == "traffic light":
        cv2.circle(ov, (lx + 25, yy), 9, col, -1)
    elif txt == "driving direction":
        cv2.arrowedLine(ov, (lx + 5, yy), (lx + 45, yy), col, 3, cv2.LINE_AA, tipLength=0.4)
    elif txt == "crosswalk":
        cv2.rectangle(ov, (lx + 5, yy - 10), (lx + 45, yy + 10), col, -1)
    else:
        cv2.line(ov, (lx + 5, yy), (lx + 45, yy), col, 4, cv2.LINE_AA)
    cv2.putText(ov, txt, (lx + 60, yy + 7), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2, cv2.LINE_AA)
deco(ov, f"Lanelet2 map ({sum(1 for l in lmap.laneletLayer if l.attributes['subtype']=='road')} road lanelets) on the point cloud map  -  same frame, {RES} m/px")
cv2.imwrite(os.path.join(OUTD, "map_overview_lanelet2.jpg"), ov, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("wrote map_overview_lanelet2.jpg", flush=True)

# ---------------------------------------------------------------- oblique 3D view of the PCD (painter's algorithm)
P = np.column_stack([x, y, z]).astype(np.float64)
C = bgr
lo_, hi_ = P.min(axis=0), P.max(axis=0)
cen = (lo_ + hi_) / 2
cen[2] = float(np.median(z))
# camera south of the route (the route runs WNW-ESE), elevated, looking north and down
cam = cen + np.array([-40.0, -560.0, 420.0])
fwd = cen - cam
fwd /= np.linalg.norm(fwd)
right = np.cross(fwd, [0, 0, 1.0])
right /= np.linalg.norm(right)
up = np.cross(right, fwd)
Rm = np.stack([right, up, fwd])
Wv, Hv = 4000, 2400
pc = (P - cam) @ Rm.T
zc = pc[:, 2]
okv = zc > 1
# fit the focal length to the points themselves (0.2 % of the most peripheral points may fall outside)
rx = np.abs(pc[okv, 0] / zc[okv])
ry = np.abs(pc[okv, 1] / zc[okv])
f = 0.96 * min((Wv / 2) / np.percentile(rx, 99.8), (Hv / 2) / np.percentile(ry, 99.8))
u = f * pc[okv, 0] / zc[okv] + Wv / 2
v = Hv / 2 - f * pc[okv, 1] / zc[okv]
m = (u >= 0) & (u < Wv - 2) & (v >= 0) & (v < Hv - 2)
u, v, d = u[m].astype(np.int32), v[m].astype(np.int32), zc[okv][m]
col = C[okv][m]
hgt = P[okv][m, 2]
o = np.argsort(-d, kind="stable")  # far first
view = np.full((Hv, Wv, 3), 18, np.uint8)
sh = np.clip(0.8 + 0.3 * np.tanh((hgt[o] - cen[2]) / 6.0), 0.55, 1.15)
cc = np.clip(col[o].astype(np.float32) * sh[:, None] * 1.25, 0, 255).astype(np.uint8)
for du in (0, 1):
    for dv in (0, 1):
        view[v[o] + dv, u[o] + du] = cc
# crop to the content (the route is wide and flat in this view)
content = np.any(view != 18, axis=2)
rows, cols = np.flatnonzero(content.any(axis=1)), np.flatnonzero(content.any(axis=0))
r0, r1 = max(rows[0] - 90, 0), min(rows[-1] + 60, Hv)
c0, c1 = max(cols[0] - 60, 0), min(cols[-1] + 60, Wv)
view = np.ascontiguousarray(view[r0:r1, c0:c1])
cv2.putText(view, "Point cloud map, oblique view from the south (RGB from cameras)", (30, 55), cv2.FONT_HERSHEY_SIMPLEX,
            1.4, (255, 255, 255), 3, cv2.LINE_AA)
cv2.imwrite(os.path.join(OUTD, "map_overview_pcd_oblique.jpg"), view, [cv2.IMWRITE_JPEG_QUALITY, 90])
print("wrote map_overview_pcd_oblique.jpg", flush=True)
