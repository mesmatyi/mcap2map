#!/usr/bin/env python3
"""Validate the Lanelet2 map: reload through the projector, routing graph, trajectory coverage, overlay plots.

usage: validate_lanelet2.py MAP_OSM RASTER_NPZ TRAJ_NPZ OUT_DIR
"""
import os
import sys
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import lanelet2
from lanelet2.core import BasicPoint2d
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lanegeom import Raster, fine_centerline

MAP, RAS, TRAJ, OUTD = sys.argv[1:5]
from config import OX, OY, UTM_EPSG  # local map frame origin
lon0, lat0 = Transformer.from_crs(UTM_EPSG, "EPSG:4326", always_xy=True).transform(OX, OY)
proj = UtmProjector(Origin(lat0, lon0))
lmap, errs = lanelet2.io.loadRobust(MAP, proj)
print(f"loaded: {len(lmap.laneletLayer)} lanelets, {len(lmap.pointLayer)} points; load errors: {len(errs)}")
for e in errs[:5]:
    print("  ", e)

# 1) projector round trip vs. local_x/local_y tags written into the file
dev = []
for p in lmap.pointLayer:
    if "local_x" in p.attributes:
        dev.append(np.hypot(p.x - float(p.attributes["local_x"]), p.y - float(p.attributes["local_y"])))
dev = np.array(dev)
print(f"lat/lon -> local vs local_x/local_y tags: max deviation {dev.max()*1000:.2f} mm over {len(dev)} points")

# 2) routing graph
rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                      lanelet2.traffic_rules.Participants.Vehicle)
graph = lanelet2.routing.RoutingGraph(lmap, rules)
errors = graph.checkValidity() if hasattr(graph, "checkValidity") else []
road = [ll for ll in lmap.laneletLayer if ll.attributes["subtype"] == "road"]
no_succ = [ll for ll in road if not graph.following(ll)]
no_pred = [ll for ll in road if not graph.previous(ll)]
print(f"road lanelets {len(road)}: without successor {len(no_succ)}, without predecessor {len(no_pred)}")

# 3) trajectory coverage and a route along the driven path
z = np.load(TRAJ)
pos = z["pos"][::500]  # 2 Hz
mv = np.r_[True, np.linalg.norm(np.diff(pos[:, :2], axis=0), axis=1) > 0.5]
pos = pos[mv]
hd = np.gradient(pos[:, :2], axis=0)
hd /= np.maximum(np.linalg.norm(hd, axis=1), 1e-9)[:, None]


def ll_dir(ll, pt):
    cl = fine_centerline(ll)
    i = int(np.argmin(np.linalg.norm(cl - pt, axis=1)))
    v = cl[min(i + 1, len(cl) - 1)] - cl[max(i - 1, 0)]
    return v / max(np.linalg.norm(v), 1e-9)


inside, wrongdir, seq, seq_xy = [], [], [], []
for p, h in zip(pos, hd):
    pt = BasicPoint2d(p[0], p[1])
    hits = [ll for ll in road if lanelet2.geometry.inside(ll, pt)]
    good = [ll for ll in hits if ll_dir(ll, p[:2]) @ h > 0.5]
    inside.append(bool(good))
    wrongdir.append(bool(hits) and not good)
    if good:
        # junction connectors overlap: keep the current lanelet if still inside, else prefer one reachable from it
        if seq and any(g.id == seq[-1] for g in good):
            pick = seq[-1]
        else:
            pick = good[0].id
            if seq:
                prev = lmap.laneletLayer[seq[-1]]
                reach = [g for g in good if graph.getRoute(prev, g) is not None and
                         len(graph.getRoute(prev, g).shortestPath()) <= 4]
                if reach:
                    pick = reach[0].id
        seq.append(pick)
        seq_xy.append(p[:2])
inside = np.array(inside)
print(f"driven positions inside a road lanelet of matching direction: {inside.mean()*100:.1f}% of {len(inside)} "
      f"(inside only wrong-direction lanelets: {np.sum(wrongdir)})")
gaps = np.flatnonzero(~inside)
if len(gaps):
    print("  outside samples (x, y):", [tuple(np.round(pos[i, :2], 1)) for i in gaps[:: max(1, len(gaps) // 15)]])
# route from the first to the last covered lanelet, via the driven ones
ids, ids_xy = [seq[0]], [seq_xy[0]]
for i in range(1, len(seq)):
    if seq[i] != seq[i - 1]:
        ids.append(seq[i])
        ids_xy.append(seq_xy[i])
ok_steps = 0
for (a, b), xy in zip(zip(ids[:-1], ids[1:]), ids_xy[1:]):
    r = graph.getRoute(lmap.laneletLayer[a], lmap.laneletLayer[b])
    ok_steps += r is not None
    if r is None:
        la, lb = lmap.laneletLayer[a], lmap.laneletLayer[b]
        print(f"  no route {a} -> {b} near ({xy[0]:.0f},{xy[1]:.0f}): {a} turn={la.attributes['turn_direction'] if 'turn_direction' in la.attributes else '-'} "
              f"followers={[f.id for f in graph.following(la)]}, {b} preceded by {[f.id for f in graph.previous(lb)]}")
print(f"consecutive driven lanelet pairs routable: {ok_steps}/{len(ids)-1}")
full = graph.getRoute(lmap.laneletLayer[ids[0]], lmap.laneletLayer[ids[-1]])
print("route start -> end exists:", full is not None,
      f"(shortest path {len(full.shortestPath())} lanelets)" if full else "")

# 4) overlay plots
ras = Raster(RAS)


def draw(ax, xa, xb, ya, yb):
    i0, i1 = max(0, int((xa - ras.x0) / ras.res)), min(ras.W, int((xb - ras.x0) / ras.res))
    j0, j1 = max(0, int((ya - ras.y0) / ras.res)), min(ras.H, int((yb - ras.y0) / ras.res))
    ax.imshow(np.nan_to_num(ras.refl[j0:j1, i0:i1], nan=0), origin="lower", cmap="gray", vmin=0, vmax=35,
              extent=(ras.x0 + i0 * ras.res, ras.x0 + i1 * ras.res, ras.y0 + j0 * ras.res, ras.y0 + j1 * ras.res))
    for ll in lmap.laneletLayer:
        sub = ll.attributes["subtype"]
        if sub == "crosswalk":
            poly = np.array([[q.x, q.y] for q in ll.polygon2d()])
            ax.fill(poly[:, 0], poly[:, 1], color="deepskyblue", alpha=0.35, lw=0)
            continue
        conn = "turn_direction" in ll.attributes
        for bnd in (ll.leftBound, ll.rightBound):
            b = np.array([[q.x, q.y] for q in bnd])
            t = bnd.attributes["type"]
            col = {"line_thin": "red", "road_border": "magenta", "virtual": "orange"}.get(t, "y")
            ls = "--" if ("subtype" in bnd.attributes and bnd.attributes["subtype"] == "dashed") else "-"
            ax.plot(b[:, 0], b[:, 1], ls, color=col, lw=0.8 if not conn else 0.6)
        cl = fine_centerline(ll)
        ax.plot(cl[:, 0], cl[:, 1], ":", color="cyan" if not conn else "yellow", lw=0.6)
        mid = cl[len(cl) // 2]
        if len(cl) > 2:
            d = cl[min(len(cl) - 1, len(cl) // 2 + 1)] - cl[len(cl) // 2 - 1]
            ax.arrow(mid[0], mid[1], d[0] * 0.8, d[1] * 0.8, head_width=0.8, color="cyan" if not conn else "yellow")
    for re in lmap.regulatoryElementLayer:
        for sl in (re.parameters["ref_line"] if "ref_line" in re.parameters else []):
            b = np.array([[q.x, q.y] for q in sl])
            ax.plot(b[:, 0], b[:, 1], "-", color="lime", lw=2)
    ax.plot(pos[:, 0], pos[:, 1], ".", color="lime", ms=1.5)
    ax.set_xlim(xa, xb)
    ax.set_ylim(ya, yb)
    ax.set_aspect("equal")


_tp = np.load(TRAJ)["pos"][::200, :2]
_x0, _y0 = _tp.min(axis=0) - 40
_x1, _y1 = _tp.max(axis=0) + 40
_xm, _ym = 0.5 * (_x0 + _x1), 0.5 * (_y0 + _y1)
views = {"overview": (_x0, _x1, _y0, _y1), "south_west": (_x0, _xm + 20, _y0, _ym + 20),
         "south_east": (_xm - 20, _x1, _y0, _ym + 20), "north_west": (_x0, _xm + 20, _ym - 20, _y1),
         "north_east": (_xm - 20, _x1, _ym - 20, _y1)}
for name, v in views.items():
    w = v[1] - v[0]
    fig, ax = plt.subplots(figsize=(18, 18 * (v[3] - v[2]) / w))
    draw(ax, *v)
    ax.set_title(f"lanelet2 over PCD ground reflectivity: {name}")
    fig.savefig(os.path.join(OUTD, f"lanelet2_overlay_{name}.png"), dpi=110 if name != "overview" else 90,
                bbox_inches="tight")
    plt.close(fig)
print("overlays written")
