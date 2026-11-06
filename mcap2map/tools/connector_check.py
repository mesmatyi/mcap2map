#!/usr/bin/env python3
"""Junction connector quality: lists connector lanelets (those with a turn_direction) whose centre line or bounds
hook (the heading leaves the range between the start and end headings) or turn tighter than a few metres.
Radii are measured as the turning over a 2 m window, so they do not depend on where the vertices fall.

usage: connector_check.py MAP_OSM [-v]     (-v lists every connector)
"""
import os
import sys

import numpy as np
import lanelet2
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lanegeom import fine_centerline
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import LAT0, LON0


def quality(p, step=0.5):
    seg = np.linalg.norm(np.diff(p, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < 2 * step:
        return 0.0, 0.0, np.inf, float(s[-1])
    si = np.linspace(0, s[-1], max(3, int(s[-1] / step) + 1))
    r = np.column_stack([np.interp(si, s, p[:, 0]), np.interp(si, s, p[:, 1])])
    v = np.diff(r, axis=0)
    h = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    w = min(4, len(h) - 1)
    dh = np.abs(h[w:] - h[:-w])
    r_min = w * (s[-1] / (len(si) - 1)) / max(float(np.max(dh)), 1e-9)
    lo, hi = min(h[0], h[-1]), max(h[0], h[-1])
    over = max(0.0, float(np.max(h) - hi), float(lo - np.min(h)))
    return float(np.degrees(h[-1] - h[0])), float(np.degrees(over)), float(r_min), float(s[-1])


def xy(ls):
    return np.array([[q.x, q.y] for q in ls])


m, _e = lanelet2.io.loadRobust(sys.argv[1], UtmProjector(Origin(LAT0, LON0)))
rows, n = [], 0
for ll in m.laneletLayer:
    a = ll.attributes
    if "turn_direction" not in a or a["subtype"] != "road":
        continue
    n += 1
    c = fine_centerline(ll)
    tot, over, rmin, L = quality(c)
    _, overL, rminL, _ = quality(xy(ll.leftBound))
    _, overR, rminR, _ = quality(xy(ll.rightBound))
    bad = over > 25 or rmin < 4.0 or min(rminL, rminR) < 2.0 or max(overL, overR) > 35
    if bad or "-v" in sys.argv:
        rows.append((ll.id, f"  ll {ll.id} {a['turn_direction']:8s} at ({c[0][0]:.1f},{c[0][1]:.1f}) L={L:.1f} m turn={tot:.0f} deg "
                            f"overshoot={over:.0f} deg min R={rmin:.1f} m | bounds: overshoot {overL:.0f}/{overR:.0f} deg, "
                            f"min R {rminL:.1f}/{rminR:.1f} m{'  <-- flagged' if bad and '-v' in sys.argv else ''}"))
flag = sum(1 for _, r_ in rows if "-v" not in sys.argv or "flagged" in r_)
print(f"{n} connectors; flagged (centre overshoot > 25 deg or R < 4 m, bound R < 2 m or overshoot > 35 deg): {flag}")
for _, r_ in sorted(rows):
    print(r_)
