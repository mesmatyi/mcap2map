#!/usr/bin/env python3
"""Can the driven route be planned on the Lanelet2 map? Samples the driven path every STEP m, picks the road lanelet
containing each sample (matching driving direction), and asks the lanelet2 routing graph (vehicle traffic rules,
lane changes allowed where the rules allow them) for a route between consecutive samples.

usage: route_test.py MAP_OSM TRAJ_NPZ [STEP_M]
"""
import os
import sys

import lanelet2
import numpy as np
from lanelet2.core import BasicPoint2d
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import LAT0, LON0

MAP, TRAJ = sys.argv[1:3]
STEP = float(sys.argv[3]) if len(sys.argv) > 3 else 50.0
lmap, _errs = lanelet2.io.loadRobust(MAP, UtmProjector(Origin(LAT0, LON0)))
rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany, lanelet2.traffic_rules.Participants.Vehicle)
graph = lanelet2.routing.RoutingGraph(lmap, rules)

p = np.load(TRAJ)["pos"][::50, :2]
d = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))]
idx = np.searchsorted(d, np.arange(0, d[-1], STEP))
hd = np.gradient(p, axis=0)


def candidates(i):
    """Road lanelets containing sample i with a matching driving direction, nearest centre line first."""
    q = BasicPoint2d(*p[i])
    out = []
    for ll in lmap.laneletLayer.search(lanelet2.core.BoundingBox2d(BasicPoint2d(p[i][0] - 1, p[i][1] - 1),
                                                                   BasicPoint2d(p[i][0] + 1, p[i][1] + 1))):
        if ll.attributes["subtype"] not in ("road",) or not lanelet2.geometry.inside(ll, q):
            continue
        c = np.array([[pt.x, pt.y] for pt in ll.centerline])
        j = int(np.argmin(np.linalg.norm(c - p[i], axis=1)))
        t = c[min(j + 1, len(c) - 1)] - c[max(j - 1, 0)]
        if float(t @ hd[i]) > 0:
            out.append((float(np.min(np.linalg.norm(c - p[i], axis=1))), ll))
    return [ll for _, ll in sorted(out, key=lambda z: z[0])]


def pick(i, prev):
    """Where lanelets overlap (junction connectors over a lanelet crossing the junction) a planner continues on the
    one it can reach: prefer a candidate routable from the previous sample's lanelet."""
    cands = candidates(i)
    if prev is not None:
        for ll in cands:
            if ll.id == prev.id or graph.getRoute(prev, ll) is not None:
                return ll
    return cands[0] if cands else None


samples, prev = [], None
for i in idx:
    ll = pick(int(i), prev)
    samples.append((int(i), ll))
    prev = ll if ll is not None else prev
ok = fail = miss = 0
for (i, a), (j, b) in zip(samples[:-1], samples[1:]):
    if a is None or b is None:
        miss += 1
        print(f"  no lanelet at d={d[i if a is None else j]:.0f} m ({p[i if a is None else j][0]:.0f},"
              f"{p[i if a is None else j][1]:.0f})")
        continue
    if a.id == b.id or graph.getRoute(a, b) is not None:
        ok += 1
    else:
        fail += 1
        print(f"  NO ROUTE d={d[i]:.0f}->{d[j]:.0f} m: lanelet {a.id} ({p[i][0]:.0f},{p[i][1]:.0f}) -> {b.id} "
              f"({p[j][0]:.0f},{p[j][1]:.0f})")
print(f"route segments of {STEP:.0f} m: {ok} routable, {fail} not routable, {miss} without a lanelet at an end "
      f"(of {len(samples) - 1})")
# whole drive in one route with intermediate points (if every segment is routable)
seq = [ll for _, ll in samples if ll is not None]
if len(seq) > 2 and fail == 0:
    r = graph.getRouteVia(seq[0], seq[1:-1], seq[-1])
    print("full route start -> via all samples -> end:", r is not None)
