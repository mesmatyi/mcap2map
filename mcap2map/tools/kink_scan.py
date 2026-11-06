#!/usr/bin/env python3
"""List kinked line strings and odd connectors of a Lanelet2 map: sharp turning angles between consecutive
segments, bounds that cross each other, connectors whose centre line turns more than 150 degrees.

usage: kink_scan.py MAP_OSM [max_turn_deg]
"""
import sys
import numpy as np
import lanelet2
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector
from shapely.geometry import LineString

sys.path.insert(0, __import__('os').path.dirname(__import__('os').path.abspath(__file__)))
from lanegeom import fine_centerline
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import LAT0, LON0

MAP = sys.argv[1]
MAXT = float(sys.argv[2]) if len(sys.argv) > 2 else 45.0
m, _e = lanelet2.io.loadRobust(MAP, UtmProjector(Origin(LAT0, LON0)))
used = {}
for ll in m.laneletLayer:
    for b in (ll.leftBound, ll.rightBound):
        used.setdefault(b.id, []).append(ll.id)
rows = []
for ls in m.lineStringLayer:
    t = ls.attributes["type"] if "type" in ls.attributes else ""
    if t not in ("line_thin", "road_border", "virtual"):
        continue
    xy = np.array([[q.x, q.y] for q in ls])
    if len(xy) < 3:
        continue
    v = np.diff(xy, axis=0)
    L = np.linalg.norm(v, axis=1)
    keep = L > 1e-6
    v, L = v[keep], L[keep]
    if len(v) < 2:
        continue
    h = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    turn = np.degrees(np.abs(np.diff(h)))
    k = int(np.argmax(turn))
    if turn[k] > MAXT:
        rows.append((turn[k], ls.id, t, xy[k + 1], len(xy), used.get(ls.id, [])))
rows.sort(reverse=True)
print(f"line strings with a turn > {MAXT} deg between consecutive segments: {len(rows)}")
for tr, i, t, p, n, u in rows:
    print(f"  ls {i} {t:11s} turn {tr:5.0f} deg at ({p[0]:.1f},{p[1]:.1f}) n={n} lanelets {u}")
print("connectors with crossing bounds / strong total turn:")
for ll in m.laneletLayer:
    if "turn_direction" not in ll.attributes:
        continue
    lb = LineString([[q.x, q.y] for q in ll.leftBound]); rb = LineString([[q.x, q.y] for q in ll.rightBound])
    cl = fine_centerline(ll)
    if len(cl) < 2:
        print(f"  lanelet {ll.id}: degenerate (centre line has {len(cl)} points)")
        continue
    v = np.diff(cl, axis=0); v = v[np.linalg.norm(v, axis=1) > 1e-6]
    tot = np.degrees(abs(np.unwrap(np.arctan2(v[:, 1], v[:, 0]))[-1] - np.arctan2(v[0, 1], v[0, 0]))) if len(v) > 1 else 0
    cross = lb.crosses(rb)
    if cross or tot > 150 or not lb.is_simple or not rb.is_simple:
        print(f"  lanelet {ll.id} {ll.attributes['turn_direction']} at ({cl[len(cl)//2][0]:.1f},{cl[len(cl)//2][1]:.1f}): "
              f"bounds cross={cross} simple={lb.is_simple and rb.is_simple} total turn {tot:.0f} deg len {lb.length:.1f}/{rb.length:.1f} m")
