#!/usr/bin/env python3
"""Assemble the Lanelet2 map from the fitted lane boundaries and the OSM route topology.

* road lanelets per OSM way piece; neighbouring lanes share boundary line strings, opposite directions use the
  inverted centre line; consecutive pieces share points so lanelet2 derives successor relations
* junctions: connector lanelets for every movement between mapped legs that OSM allows (oneway, turn
  restrictions, turn:lanes; otherwise left from the leftmost / right from the rightmost lane); junction nodes
  joined by very short chains form one junction, a side road that OSM draws along a through road gets its junction
  where it really leaves that road, roundabout arms whose OSM flare lies inside the measured ring are rebuilt
* traffic lights + stop lines on approaches of signalised OSM junctions, crosswalk lanelets from detected
  zebra markings, speed_limit from OSM maxspeed
Coordinates: the run's local map frame (UTM - (OX, OY)), z = NovAtel orthometric height, identical to the PCD.

* every road boundary is finally fitted to the point cloud map: painted lines to the intensity-weighted
  paint centre, curbs to the height step, vertex elevations to the PCD road surface

usage: build_lanelet2.py LANES_JSON ROUTE_JSON RASTER_NPZ TRAJ_NPZ OUT_OSM [PCD]
"""
import json
import math
import os
import sys
from collections import defaultdict

import numpy as np
import lanelet2
from shapely.geometry import LineString as _LS
from shapely.geometry import Polygon as _Poly
from lanelet2.core import AttributeMap, Lanelet, LaneletMap, LineString3d, Point3d, TrafficLight, getId
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector
from pyproj import Transformer
from scipy.spatial import cKDTree

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lanegeom import resample_polyline, Raster, RefLine
from pcd_refine import SurfacePoints, read_pcd_xyzi, refine_polyline, surface_z

LANES, ROUTE, RAS, TRAJ, OUT = sys.argv[1:6]
PCD_PATH = sys.argv[6] if len(sys.argv) > 6 else None
RGB_PCD = sys.argv[7] if len(sys.argv) > 7 else None  # coloured map: camera luminance as a second paint channel
FIT_CACHE = os.environ.get("FIT_CACHE")  # directory caching the per-chain paint fit (the slow stage) between runs
from config import FRAME, GROUND_HEIGHT, OX, OY, UTM_EPSG  # local map frame origin
to_ll = Transformer.from_crs(UTM_EPSG, "EPSG:4326", always_xy=True)
LON0, LAT0 = to_ll.transform(OX, OY)

lanes = json.load(open(LANES))
net = json.load(open(ROUTE))
ras = Raster(RAS)
PCD_XYZI = read_pcd_xyzi(PCD_PATH) if PCD_PATH else None
traj = np.load(TRAJ)["pos"][::100]
tkd = cKDTree(traj[:, :2])
lmap = LaneletMap()


def ground_z(xy):
    xy = np.atleast_2d(xy)
    z = ras.sample(ras.ground, xy).astype(np.float64)
    bad = ~np.isfinite(z)
    if bad.any():
        _, i = tkd.query(xy[bad])
        z[bad] = traj[i, 2] - GROUND_HEIGHT
    return z


def mk_point(x, y, z=None):
    if z is None:
        z = float(ground_z([x, y])[0])
    return Point3d(getId(), float(x), float(y), float(z),
                   AttributeMap({"local_x": f"{x:.3f}", "local_y": f"{y:.3f}"}))


def mk_ls(points, attrs):
    return LineString3d(getId(), points, AttributeMap(attrs))


BOUND_ATTR = {"solid": {"type": "line_thin", "subtype": "solid"},
              "solid_solid": {"type": "line_thin", "subtype": "solid_solid"},
              "dashed": {"type": "line_thin", "subtype": "dashed"},
              "road_border": {"type": "road_border"},
              "virtual": {"type": "virtual"},
              "virtual_lc": {"type": "virtual", "lane_change": "yes"}}  # unpainted line between same-direction lanes


def speed(mx):
    try:
        return str(int(str(mx).split()[0]))
    except ValueError:
        return "50"


# ------------------------------------------------------------------ chain cut ranges at junctions
# junction clusters: junction nodes joined by a chain shorter than SHORT_CHAIN (e.g. a 5 m OSM stem between a
# junction and a roundabout-arm flare) form one junction. The short chain gets no lanelets and its end nodes are
# merged, so the connectors span the whole cluster.
SHORT_CHAIN = 12.0
_junc_raw = set(net["junctions"])
_ring_osm = {n for c_ in net["chains"] if c_.get("ring") for n in c_["nodes"]}
_alias = {}


def _root(n):
    while n in _alias:
        n = _alias[n]
    return n


_dropped = []
for c_ in lanes["chains"]:
    n0_, n1_ = c_["nodes"]
    if (not c_.get("ring") and c_["length"] < SHORT_CHAIN and n0_ != n1_ and n0_ in _junc_raw and n1_ in _junc_raw
            and n0_ not in _ring_osm and n1_ not in _ring_osm):
        _dropped.append(c_["index"])
        a_, b_ = _root(n0_), _root(n1_)
        if a_ != b_:
            _alias[b_] = a_
chains = [c_ for c_ in lanes["chains"] if c_["index"] not in _dropped]
for c_ in chains:
    c_["nodes"] = [_root(n) for n in c_["nodes"]]
_members = defaultdict(set)
for n in _junc_raw:
    _members[_root(n)].add(n)
if _dropped:
    print(f"junction clusters: {len(_dropped)} short chain(s) inside a junction dropped "
          f"({', '.join(str(i) for i in _dropped)}), {len(_alias)} node(s) merged")
junctions = {_root(n) for n in _junc_raw}
chain_by_idx = {c["index"]: c for c in chains}
ends_at = defaultdict(list)  # node -> [(chain idx, end 0/1)]
for c in chains:
    ends_at[c["nodes"][0]].append((c["index"], 0))
    ends_at[c["nodes"][1]].append((c["index"], 1))
refs = {c["index"]: RefLine(c["ref_xy"]) for c in chains}


def end_dir(ci, end):
    """Unit vector of chain ci at its end, pointing away from the junction into the chain."""
    r = refs[ci]
    i = 0 if end == 0 else -1
    t = r.t[min(10, len(r.t) - 1)] if end == 0 else r.t[max(-11, -len(r.t))]
    return t if end == 0 else -t


def end_halfwidth(ci, end):
    p = chain_by_idx[ci]["pieces"][0 if end == 0 else -1]
    bd = np.array(p["bounds_d"])[:, 0 if end == 0 else -1]
    return (bd.max() - bd.min()) / 2


_node_xy = {int(k): v for k, v in net["node_xy"].items()}
_next_chain = max(c_["index"] for c_ in lanes["chains"]) + 1
_next_node = -1  # ids of junction nodes this script creates (negative: never an OSM id)


def _slice_piece(p, s0, s1):
    """Copy of lane piece p restricted to chain stations [s0, s1], stations re-based to s0 (None if < 0.5 m)."""
    st = np.array(p["stations"])
    a_, b_ = max(s0, st[0]), min(s1, st[-1])
    if b_ - a_ < 0.5:
        return None
    m = (st > a_ + 1e-6) & (st < b_ - 1e-6)
    new_st = np.r_[a_, st[m], b_]
    q = dict(p)
    q["stations"] = (new_st - s0).tolist()
    q["s0"], q["s1"] = max(p["s0"], s0) - s0, min(p["s1"], s1) - s0
    q["bounds_d"] = [np.interp(new_st, st, np.asarray(row, float)).tolist() for row in p["bounds_d"]]
    q["bounds_xy"] = [np.column_stack([np.interp(new_st, st, np.asarray(row, float)[:, 0]),
                                       np.interp(new_st, st, np.asarray(row, float)[:, 1])]).tolist()
                      for row in p["bounds_xy"]]
    return q


def _slice_chain(c, s0, s1, index, nodes):
    """Copy of chain c restricted to stations [s0, s1] (re-based to 0) with a new index and end nodes."""
    xy = np.asarray(c["ref_xy"], float)
    sx = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    s1 = min(s1, float(sx[-1]))
    m = (sx > s0 + 1e-6) & (sx < s1 - 1e-6)
    ends = [[float(np.interp(s_, sx, xy[:, 0])), float(np.interp(s_, sx, xy[:, 1]))] for s_ in (s0, s1)]
    q = dict(c)
    q["index"], q["nodes"] = index, list(nodes)
    q["ref_xy"] = [ends[0]] + xy[m].tolist() + [ends[1]]
    q["length"] = s1 - s0
    q["pieces"] = [pp for pp in (_slice_piece(p, s0, s1) for p in c["pieces"]) if pp is not None]
    q["crosswalks"] = [dict(cw, s0=cw["s0"] - s0, s1=cw["s1"] - s0) for cw in c["crosswalks"]
                       if s0 <= 0.5 * (cw["s0"] + cw["s1"]) < s1]
    q["stoplines"] = [dict(sl, s=sl["s"] - s0) for sl in c["stoplines"] if s0 <= sl["s"] < s1]
    return q


def _rebuild_index():
    global chain_by_idx, ends_at, refs
    chain_by_idx = {c_["index"]: c_ for c_ in chains}
    ends_at = defaultdict(list)
    for c_ in chains:
        ends_at[c_["nodes"][0]].append((c_["index"], 0))
        ends_at[c_["nodes"][1]].append((c_["index"], 1))
    refs = {c_["index"]: refs[c_["index"]] if refs_src.get(c_["index"]) is c_["ref_xy"] else RefLine(c_["ref_xy"])
            for c_ in chains}  # sliced chains carry a new ref_xy list
    refs_src.clear()
    refs_src.update({c_["index"]: c_["ref_xy"] for c_ in chains})


refs_src = {c_["index"]: c_["ref_xy"] for c_ in chains}

# ------------------------------------------------------------------ junction relocation
# OSM sometimes draws a side road running along a through road for tens of metres before its junction node (in
# Kaposvár, Dózsa György utca meets Budai Nagy Antal utca ~35 m before the OSM node), so the side road's lanes lie on
# the through road's lanes and the driven path enters a through-road lanelet in its middle. Where a chain leaves a
# node inside the corridor of a through road (a chain end with a straight continuation at the node) and stays in it
# for more than RELOC_MIN m, the side road is truncated where it leaves the corridor and the through road is split
# there: the junction moves to where the roads really meet.
RELOC_MIN = 15.0
RELOC_ANG = 35.0  # side road and through road leave the OSM node within this angle of each other
CORR_LEN = 80.0


def _ang(u, v):
    return math.degrees(math.acos(float(np.clip(u @ v, -1.0, 1.0))))


def _corridor(ci, end, length=CORR_LEN):
    """Polygon covering all lanes of chain ci within `length` m of its end `end`."""
    c = chain_by_idx[ci]
    L = c["length"]
    lo_, hi_ = (0.0, min(L, length)) if end == 0 else (max(0.0, L - length), L)
    left, right = [], []
    for p in c["pieces"]:
        st = np.array(p["stations"])
        bd = np.array(p["bounds_d"])
        m = (st >= lo_ - 1e-6) & (st <= hi_ + 1e-6)
        if m.sum() < 2:
            continue
        left.append(refs[ci].at(st[m], bd.max(0)[m]))
        right.append(refs[ci].at(st[m], bd.min(0)[m]))
    if not left:
        return None
    return _Poly(np.vstack(left + [r_[::-1] for r_ in right[::-1]])).buffer(0)


def _cross_section(ci, s):
    """Segment across all lanes of chain ci at station s (clamped to the lane model), None in a gap of it."""
    pcs = chain_by_idx[ci]["pieces"]
    s = min(max(s, pcs[0]["stations"][0]), pcs[-1]["stations"][-1])
    for p in pcs:
        st = np.array(p["stations"])
        if st[0] - 1e-6 <= s <= st[-1] + 1e-6:
            bd = np.array(p["bounds_d"])
            return refs[ci].at([s, s], [float(np.interp(s, st, bd.min(0))), float(np.interp(s, st, bd.max(0)))])
    return None


def _ring_link(ci):
    c = chain_by_idx[ci]
    return c["length"] < 30 and (c["nodes"][0] in _ring_osm or c["nodes"][1] in _ring_osm)


_HW_RANK = {"motorway": 7, "trunk": 6, "primary": 5, "secondary": 4, "tertiary": 3, "unclassified": 2,
            "residential": 2, "living_street": 1, "service": 1}


def _end_piece(ci, end):
    return chain_by_idx[ci]["pieces"][0 if end == 0 else -1]


def _chord_dir(ci, end, dist=30.0):
    """Unit vector from the chain end to the reference point `dist` m into the chain."""
    r = refs[ci]
    d = min(dist, r.length)
    p0, p1 = r.at([0.0 if end == 0 else r.length, d if end == 0 else r.length - d])
    v = p1 - p0
    return v / max(np.linalg.norm(v), 1e-9)


def _through(X, x, lst, other):
    """How clearly chain end (X, x) is the through road at its node, compared lexicographically: (same street name
    as a straight continuation, straight over 30 m with it, highway rank). `other`: the chain it is compared with."""
    name = _end_piece(X, x)["name"]
    best_ = (0, 0)
    for C, c_ in lst:
        if C in (X, other) or _ang(end_dir(X, x), end_dir(C, c_)) <= 150:
            continue
        best_ = max(best_, (int(bool(name) and _end_piece(C, c_)["name"] == name),
                            int(_ang(_chord_dir(X, x), _chord_dir(C, c_)) > 165)))
    return best_ + (_HW_RANK.get(_end_piece(X, x)["highway"], 0),)


def _flow_in(ci, end):
    """+1: one-way traffic of chain end flows into the node, -1: out of it, 0: two-way."""
    c = chain_by_idx[ci]
    if c["two_way"]:
        return 0
    return 1 if end == 1 else -1


_relocs = []
for node, lst in list(ends_at.items()):
    if node in _ring_osm or len(lst) < 3:
        continue
    best = None
    for A, a in lst:
        cA = chain_by_idx[A]
        if cA.get("ring") or _ring_link(A):
            continue
        dA = end_dir(A, a)
        for B, b in lst:
            if B == A:
                continue
            cB = chain_by_idx[B]
            if cB.get("ring") or _ring_link(B) or _ang(dA, end_dir(B, b)) > RELOC_ANG:
                continue  # A must leave the node along B
            if not any(_ang(end_dir(B, b), end_dir(C, c_)) > 150 for C, c_ in lst if C not in (A, B)):
                continue  # B is no through road here
            if _through(B, b, lst, A) <= _through(A, a, lst, B):
                continue  # B is not more clearly the through road than A (two roads merging into one)
            fa, fb = _flow_in(A, a), _flow_in(B, b)
            if fa and fb and fa != fb:
                continue  # two one-way roads in opposite directions: a dual carriageway split, not a side road
            poly = _corridor(B, b)
            if poly is None:
                continue
            LA = cA["length"]
            s_sep, clear = None, 0.0
            for dist in np.arange(0.0, min(CORR_LEN, 0.8 * LA), 1.0):
                seg = _cross_section(A, dist if a == 0 else LA - dist)
                if seg is None:
                    break
                if _LS(seg).intersects(poly):
                    s_sep, clear = None, 0.0
                    continue
                s_sep = dist if s_sep is None else s_sep
                clear += 1.0
                if clear >= 3.0:
                    break
            if s_sep is None or clear < 3.0 or s_sep < RELOC_MIN:
                continue
            sA = s_sep if a == 0 else LA - s_sep
            sB = float(refs[B].project(refs[A].at([sA]))[0][0])
            LB = cB["length"]
            distB = sB if b == 0 else LB - sB
            if distB < 10.0 or LB - distB < 15.0:
                continue  # the through road would keep too little on one side of the new junction
            if best is None or s_sep > best[0]:
                best = (s_sep, A, a, B, b, sA, sB)
    if best:
        _relocs.append((node,) + best)

_touched = set()
for node, s_sep, A, a, B, b, sA, sB in _relocs:
    if A in _touched or B in _touched:
        continue
    _touched.update((A, B))
    N = _next_node
    _next_node -= 1
    cA, cB = chain_by_idx[A], chain_by_idx[B]
    _node_xy[N] = refs[B].at([sB])[0].tolist()
    if a == 1:
        newA = _slice_chain(cA, 0.0, sA, A, [cA["nodes"][0], N])
    else:
        newA = _slice_chain(cA, sA, cA["length"], A, [N, cA["nodes"][1]])
    B1 = _slice_chain(cB, 0.0, sB, B, [cB["nodes"][0], N])
    B2 = _slice_chain(cB, sB, cB["length"], _next_chain, [N, cB["nodes"][1]])
    _next_chain += 1
    chains = [c_ for c_ in chains if c_["index"] not in (A, B)] + [newA, B1, B2]
    junctions.add(N)
    _members[N] = {N}
    print(f"junction relocation: chain {A} ({cA['pieces'][-1 if a == 1 else 0]['name'] or 'unnamed'}, leaving the node "
          f"{_ang(end_dir(A, a), end_dir(B, b)):.0f} deg off it) runs {s_sep:.0f} m inside chain {B} ({cB['pieces'][-1 if b == 1 else 0]['name'] or 'unnamed'}) from OSM node "
          f"{node}; junction moved to ({_node_xy[N][0]:.1f}, {_node_xy[N][1]:.1f}), chain {B} split into {B} / {B2['index']}")
if _relocs:
    _rebuild_index()

# ------------------------------------------------------------------ roundabouts: geometry from the point cloud map
# A ring chain (OSM junction=roundabout) is not built from its OSM lane model: the circulating lane is the annulus
# between the central island's apron edge and the outer kerb measured in the PCD; arms join it through
# merge / diverge connectors that share the outer kerb line with the ring (see build_rings below).
rings = {}       # ring chain index -> geometry
ring_nodes = {}  # OSM node -> ring chain index
if PCD_PATH:
    from roundabout import ring_geometry, ring_r_out, ring_xy
    for c in chains:
        if not c.get("ring"):
            continue
        rxy = np.array(c["ref_xy"])
        A_ = np.column_stack([2 * rxy[:, 0], 2 * rxy[:, 1], np.ones(len(rxy))])
        sol_ = np.linalg.lstsq(A_, (rxy ** 2).sum(1), rcond=None)[0]
        c0 = sol_[:2]
        r0 = float(np.sqrt(sol_[2] + c0 @ c0))
        m_ = np.zeros(ras.ground.shape, bool)
        i0_, i1_ = int((c0[0] - r0 - 16 - ras.x0) / ras.res), int((c0[0] + r0 + 16 - ras.x0) / ras.res)
        j0_, j1_ = int((c0[1] - r0 - 16 - ras.y0) / ras.res), int((c0[1] + r0 + 16 - ras.y0) / ras.res)
        m_[max(j0_, 0):j1_, max(i0_, 0):i1_] = True
        spr = SurfacePoints(PCD_XYZI, ras.ground, ras.x0, ras.y0, ras.res, m_, zlo=-0.6, zhi=1.2)
        g = ring_geometry(spr, float(c0[0]), float(c0[1]), r_isl_guess=max(3.0, r0 - 4.0))
        # rotation sense: +1 if the azimuth increases along the chain (counter-clockwise ring)
        az_ = np.unwrap(np.arctan2(rxy[:, 1] - g["centre"][1], rxy[:, 0] - g["centre"][0]))
        g["sense"] = 1 if az_[-1] > az_[0] else -1
        rings[c["index"]] = g
        for node in dict.fromkeys(net["chains"][c["index"]]["nodes"]):  # every OSM node of the ring once (closed ring repeats its first)
            ring_nodes[node] = c["index"]
        print(f"roundabout chain {c['index']}: OSM circle r {r0:.1f} m; PCD centre ({g['centre'][0]:.2f}, {g['centre'][1]:.2f}), "
              f"island r {g['r_island']:.2f} m, apron edge r {g['r_in']:.2f} m, outer kerb r {np.min(g['r_out']):.1f}-{np.max(g['r_out']):.1f} m, "
              f"{'ccw' if g['sense'] > 0 else 'cw'}, {int(g['mouth'].sum())}/{len(g['mouth'])} sectors without kerb (arm mouths)")


def ring_u(g, xy):
    """Ring angle (rad, increasing along the driving direction) of points."""
    xy = np.atleast_2d(xy)
    return g["sense"] * np.arctan2(xy[:, 1] - g["centre"][1], xy[:, 0] - g["centre"][0])


def ring_pt(g, u, r):
    """Point(s) on the ring at angle u (rad, driving sense) and radius r (scalar / array)."""
    th = g["sense"] * np.asarray(u, float)
    r = np.asarray(r, float)
    return np.column_stack([g["centre"][0] + r * np.cos(th), g["centre"][1] + r * np.sin(th)])


def ring_outer_r(g, u):
    return ring_r_out(g, np.degrees(g["sense"] * np.asarray(u, float)))


# ------------------------------------------------------------------ roundabout arms: OSM flares inside the ring
# OSM draws some roundabouts smaller than they are. The short entry / exit links of a flared arm then lie (almost)
# entirely inside the measured outer kerb: after the cut at the kerb they degenerate into wedges that overlap the
# ring (validator: mapping.lane.border_sharing). Where the flare node lies less than FLARE_IN m outside the measured
# kerb, the arm is rebuilt: the approach road now splits FLARE_BACK m before the kerb (where a splitter island
# starts), and each link runs from there to 3 m inside the kerb along the path the vehicle drove in that direction,
# or as the straight continuation of the approach lane where it did not drive it.
FLARE_IN = 3.0
FLARE_BACK = 10.0
_tr_xy = traj[:, :2]
_tr_keep = np.r_[True, np.linalg.norm(np.diff(_tr_xy, axis=0), axis=1) > 0.05]  # moving samples only
_tr_xy = _tr_xy[_tr_keep]


def _kerb_excess(g, xy):
    """Distance of points outside the ring's measured outer kerb (negative inside)."""
    xy = np.atleast_2d(xy)
    return np.hypot(xy[:, 0] - g["centre"][0], xy[:, 1] - g["centre"][1]) - ring_outer_r(g, ring_u(g, xy))


def _driven_link(g, S, se, sS, entering):
    """Driven path of one direction between the cross-section of approach chain S at station sS (its end `se`
    touches the flare) and 3 m inside the ring's kerb, ordered in the travel direction; None if not driven."""
    ps, pd = refs[S].project(_tr_xy)
    f = (ps - sS) * (1 if se == 1 else -1)  # > 0: beyond the split point, towards the ring
    ex = _kerb_excess(g, _tr_xy)
    near = np.abs(pd[:-1]) < 12.0
    starts = np.where(near & (f[:-1] < 0) & (f[1:] >= 0))[0] if entering else \
        np.where(near & (f[:-1] >= 0) & (f[1:] < 0))[0]
    best = None
    for i in starts:
        step = 1 if entering else -1
        j = i + 1 if entering else i
        run = 0.0
        while 0 <= j + step < len(_tr_xy) and run < 40.0 and ex[j] > -3.0:
            run += float(np.linalg.norm(_tr_xy[j + step] - _tr_xy[j]))
            j += step
        if ex[j] > -3.0:
            continue  # did not reach 3 m inside the kerb within 40 m of the split point
        path = _tr_xy[i:j + 1] if entering else _tr_xy[j:i + 2]
        if best is None or run < best[0]:
            best = (run, path)
    return None if best is None else best[1]


def _straight_link(g, p0, t0, entering):
    """Straight line from p0 along t0 (towards the ring) to 3 m inside the kerb; reversed when exiting."""
    s_ = np.arange(0.0, 40.0, 0.5)
    line = p0 + s_[:, None] * t0
    k = int(np.argmax(_kerb_excess(g, line) < -3.0))
    line = line[:max(k, 2) + 1]
    return line if entering else line[::-1]


def _link_chain(orig, path, nodes, w):
    """Synthetic one-lane link chain on the reference `path` (driven-path modelled, `w` m wide)."""
    xy, s_ = resample_polyline(path, 0.5)
    L = float(s_[-1])
    st = np.r_[np.arange(0.0, L, 0.5), L]
    ref_ = RefLine(xy)
    q = dict(orig["pieces"][0])
    q.update({"s0": 0.0, "s1": L, "stations": st.tolist(), "bounds_d": [[-w / 2] * len(st), [w / 2] * len(st)],
              "bounds_xy": [ref_.at(st, -w / 2).tolist(), ref_.at(st, w / 2).tolist()],
              "bound_types": ["road_border", "road_border"], "nR": 1, "nL": 0, "traj_modeled": True,
              "driven_lane": 0, "lanes": [{"dir": 1, "side": "R", "k_from_centre": 0, "new_inner": False,
                                          "new_outer": False}]})
    c_ = dict(orig)
    c_.update({"nodes": list(nodes), "ref_xy": xy.tolist(), "length": L, "two_way": False, "pieces": [q],
               "crosswalks": [], "stoplines": [], "pass_dirs": [1], "ring": False})
    return c_


_rebuilt_arms = []
for node, lst in list(ends_at.items()):
    if node in ring_nodes or len(lst) != 3 or node not in _node_xy:
        continue
    links = [(cj, e) for cj, e in lst
             if chain_by_idx[cj]["length"] < 30 and chain_by_idx[cj]["nodes"][1 - e] in ring_nodes]
    rcis = {ring_nodes[chain_by_idx[cj]["nodes"][1 - e]] for cj, e in links}
    if len(links) != 2 or len(rcis) != 1:
        continue
    rci = rcis.pop()
    g = rings[rci]
    excess = float(_kerb_excess(g, np.asarray(_node_xy[node], float))[0])
    if excess >= FLARE_IN:
        continue
    S, se = next((cj, e) for cj, e in lst if (cj, e) not in links)
    cS = chain_by_idx[S]
    if cS["length"] < FLARE_BACK + 20.0 or cS.get("ring"):
        continue
    sS = cS["length"] - FLARE_BACK if se == 1 else FLARE_BACK  # split station on the approach road
    t_in = refs[S].t[min(len(refs[S].s) - 1, int(np.searchsorted(refs[S].s, sS)))] * (1 if se == 1 else -1)
    Fp = _next_node
    _next_node -= 1
    fp = refs[S].at([sS])[0]
    _node_xy[Fp] = fp.tolist()
    junctions.add(Fp)
    _members[Fp] = {Fp}
    newS = (_slice_chain(cS, 0.0, sS, S, [cS["nodes"][0], Fp]) if se == 1 else
            _slice_chain(cS, sS, cS["length"], S, [Fp, cS["nodes"][1]]))
    # lane centres of the approach road at the split point, per travel direction (entering = towards the ring)
    pS = newS["pieces"][-1 if se == 1 else 0]
    bdS = np.array(pS["bounds_d"])[:, -1 if se == 1 else 0]
    want_in = 1 if se == 1 else -1
    lane_c = {}
    for k_, ln in enumerate(pS["lanes"]):
        entering = ln["dir"] == want_in
        dk = 0.5 * (bdS[k_] + bdS[k_ + 1])
        # the outermost lane of each direction (right-hand side of its travel direction) leads to the ring
        key_ = "in" if entering else "out"
        rank = -dk * ln["dir"]
        if key_ not in lane_c or rank > lane_c[key_][0]:
            lane_c[key_] = (rank, refs[S].at([sS], dk)[0], abs(bdS[k_ + 1] - bdS[k_]))
    new_links, how = [], []
    for cj, e in links:
        entering = e == 0  # the link starts at the flare node: traffic flows towards the ring
        cl = chain_by_idx[cj]
        path = _driven_link(g, S, se, sS, entering)
        lc = lane_c.get("in" if entering else "out")
        w = float(np.clip(lc[2], 3.0, 4.5)) if lc else 3.5
        if path is None or len(path) < 3:
            if lc is None:
                path = None
            else:
                path = _straight_link(g, lc[1], t_in, entering)
            how.append("straight")
        else:
            how.append("driven")
        if path is None:
            continue
        ring_node = cl["nodes"][1 - e]
        new_links.append(_link_chain(cl, path, [Fp, ring_node] if entering else [ring_node, Fp], w))
    if len(new_links) != 2:
        continue
    chains = [c_ for c_ in chains if c_["index"] not in (S, links[0][0], links[1][0])] + [newS] + new_links
    _rebuilt_arms.append(node)
    print(f"roundabout chain {rci}: flare node {node} lies {excess:+.1f} m outside the measured kerb; arm rebuilt: "
          f"chain {S} splits {FLARE_BACK:.0f} m earlier at ({fp[0]:.1f}, {fp[1]:.1f}), links "
          f"{[cj for cj, _ in links]} follow the {' / '.join(how)} path to the kerb")
if _rebuilt_arms:
    _rebuild_index()

cut = {}
ring_bound_cut = {}  # (chain, end) -> station per boundary where it crosses the ring's outer kerb line
for c in chains:
    ci, L = c["index"], c["length"]
    r = refs[ci]
    if ci in rings:
        cut[ci] = (0.0, L)
        continue
    # data extent along the chain: where the trajectory passed
    ps, pd = r.project(traj[:, :2])
    m = (np.abs(pd) < 12) & (ps > -1) & (ps < L + 1)
    s_dat0, s_dat1 = (ps[m].min(), ps[m].max()) if m.any() else (0, L)
    lo, hi = 0.0, L
    for end in (0, 1):
        node = c["nodes"][end]
        others = [(cj, e) for cj, e in ends_at[node] if (cj, e) != (ci, end)]
        if node in ring_nodes:
            # arm of a roundabout: every boundary ends exactly where it crosses the ring's outer kerb line
            g = rings[ring_nodes[node]]
            p = c["pieces"][0 if end == 0 else -1]
            st = np.array(p["stations"])
            bd = np.array(p["bounds_d"])
            scs = []
            for k in range(len(bd)):
                bxy = r.at(st, bd[k])
                f = np.hypot(bxy[:, 0] - g["centre"][0], bxy[:, 1] - g["centre"][1]) - ring_outer_r(g, ring_u(g, bxy))
                inside = f < 0
                s_cross = None
                if end == 1:
                    for j in range(len(st) - 2, -1, -1):  # from the node backwards: last outside -> inside
                        if not inside[j] and inside[j + 1]:
                            s_cross = st[j] + (st[j + 1] - st[j]) * f[j] / (f[j] - f[j + 1])
                            break
                    scs.append(s_cross)
                else:
                    for j in range(1, len(st)):
                        if inside[j - 1] and not inside[j]:
                            s_cross = st[j - 1] + (st[j] - st[j - 1]) * f[j - 1] / (f[j - 1] - f[j])
                            break
                    scs.append(s_cross)
            # a boundary that never crosses the kerb line (the mouth flares out) starts / ends with the others
            found = [s_ for s_ in scs if s_ is not None]
            if end == 0:
                lo = float(min(found)) if found else lo
                scs = [s_ if s_ is not None else lo for s_ in scs]
            else:
                hi = float(max(found)) if found else hi
                scs = [s_ if s_ is not None else hi for s_ in scs]
            ring_bound_cut[(ci, end)] = np.array(scs)
            continue
        if not others:
            # route end: keep up to 10 m beyond the last driven point
            if end == 0:
                lo = max(0.0, s_dat0 - 10)
            else:
                hi = min(L, s_dat1 + 10)
            continue
        d0 = end_dir(ci, end)
        # an OSM 'flare': a two-way road splitting into a short one-way entry and exit link of a roundabout arm is
        # not a real junction -- keep the short links (cut only 1.5 m) and treat the split as straight
        ring_links = [(cj, e) for cj, e in ends_at[node] if chain_by_idx[cj]["length"] < 30 and
                      chain_by_idx[cj]["nodes"][1 - e] in ring_nodes]
        flare = len(ends_at[node]) == 3 and len(ring_links) == 2
        straightish = all(np.degrees(np.arccos(np.clip(-d0 @ end_dir(cj, e), -1, 1))) < (40 if flare else 20)
                          for cj, e in others)
        if flare:
            cd = 1.5
        elif straightish:
            cd = 3.0
        else:
            cd = float(np.clip(max(end_halfwidth(cj, e) for cj, e in others) + 2.0, 8.0, 25.0))
        cd = min(cd, L / 3)
        if end == 0:
            lo = cd
        else:
            hi = L - cd
    cut[ci] = (lo, hi)


# turn restrictions; those naming a way that is no longer mapped (a short chain merged into a junction cluster, a
# roundabout flare link joined to the ring) are dropped: their movement now runs through other legs
_gone_ways = ({p_["way"] for c_ in lanes["chains"] for p_ in c_["pieces"]} -
              {p_["way"] for c_ in chains for p_ in c_["pieces"]})
restr = defaultdict(list)
for r_ in net["restrictions"]:
    if r_["via_type"] == "node" and r_["from"] not in _gone_ways and r_["to"] not in _gone_ways:
        restr[(r_["from"], _root(r_["via"]))].append(r_)


def allowed(from_way, node, to_way, maneuver):
    for r_ in restr.get((from_way, node), []):
        kind = r_["restriction"] or ""
        if kind.startswith("no_") and r_["to"] == to_way:
            return False
        if kind.startswith("only_") and r_["to"] != to_way:
            return False
    return True


def hermite(p0, t0, p1, t1, scale, step=1.0):
    """Cubic Hermite curve; scale: tangent length as a fraction of the chord (one value, or one per end)."""
    chord = np.linalg.norm(p1 - p0)
    f0, f1 = scale if isinstance(scale, (tuple, list)) else (scale, scale)
    m0, m1 = t0 * chord * f0, t1 * chord * f1
    n = max(3, int(chord / step) + 1)
    t = np.linspace(0, 1, n)[:, None]
    h00, h10, h01, h11 = 2 * t**3 - 3 * t**2 + 1, t**3 - 2 * t**2 + t, -2 * t**3 + 3 * t**2, t**3 - t**2
    return h00 * p0 + h10 * m0 + h01 * p1 + h11 * m1


def curve_quality(curve, step=0.5):
    """(minimum radius in m, heading overshoot in deg) of a polyline: the overshoot is how far its heading leaves
    the interval spanned by the start and end headings (a hook), S-curves of lateral shifts stay small."""
    seg = np.linalg.norm(np.diff(curve, axis=0), axis=1)
    s_ = np.r_[0, np.cumsum(seg)]
    if s_[-1] < 2 * step:
        return np.inf, 0.0
    si = np.linspace(0, s_[-1], max(3, int(s_[-1] / step) + 1))
    r_ = np.column_stack([np.interp(si, s_, curve[:, 0]), np.interp(si, s_, curve[:, 1])])
    v = np.diff(r_, axis=0)
    h = np.unwrap(np.arctan2(v[:, 1], v[:, 0]))
    w_ = min(4, len(h) - 1)  # turning over a 2 m window: independent of where the vertices fall
    dh = np.abs(h[w_:] - h[:-w_])
    r_min = float(w_ * (s_[-1] / (len(si) - 1)) / max(float(np.max(dh)), 1e-9))
    lo_, hi_ = min(h[0], h[-1]), max(h[0], h[-1])
    over = max(0.0, float(np.max(h) - hi_), float(lo_ - np.min(h)))
    return r_min, math.degrees(over)


_HF = (0.4, 0.55, 0.7, 0.85, 1.0, 1.15, 1.3, 1.5, 1.75, 2.0)


def best_tangents(p0, t0, p1, t1):
    """Tangent lengths (fractions of the chord, one per end) of the Hermite connector p0,t0 -> p1,t1 that give the
    largest minimum radius without a heading overshoot (hook); lengths near the chord length are preferred."""
    best, best_f = -1e18, (1.0, 1.0)
    for f0 in _HF:
        for f1 in _HF:
            r_min, over = curve_quality(hermite(p0, t0, p1, t1, (f0, f1), step=0.25))
            score = min(r_min, 25.0) - 0.4 * over - 0.5 * (abs(f0 - 1.0) + abs(f1 - 1.0))
            if score > best:
                best, best_f = score, (f0, f1)
    return best_f


def lane_ends(ci, end, incoming):
    """(xy, unit direction) of the lane centres at the cut of chain end `end`, ordered right -> left in the travel
    direction; incoming=True: lanes driving towards the node, else lanes leaving it."""
    c = chain_by_idx[ci]
    p = c["pieces"][0 if end == 0 else -1]
    st = np.array(p["stations"])
    bd = np.array(p["bounds_d"])
    s_ = cut[ci][end]
    want = (1 if end == 1 else -1) if incoming else (-1 if end == 1 else 1)
    out = []
    for k, ln in enumerate(p["lanes"]):
        if ln["dir"] != want:
            continue
        d_ = 0.5 * (np.interp(s_, st, bd[k]) + np.interp(s_, st, bd[k + 1]))
        t_ = refs[ci].t[min(len(refs[ci].s) - 1, int(np.searchsorted(refs[ci].s, s_)))] * ln["dir"]
        out.append((d_ * ln["dir"], refs[ci].at(s_, d_)[0], t_ / max(np.linalg.norm(t_), 1e-9)))
    out.sort(key=lambda z: z[0])  # most negative signed d = rightmost in the travel direction
    return [(xy, t_) for _, xy, t_ in out]


gap_nodes = set()  # junctions whose gap was lengthened
_sig_xy = np.array([t["xy"] for t in net["tagged_nodes"] if t["tags"].get("highway") == "traffic_signals"])
def node_signalised(node):
    return len(_sig_xy) > 0 and any(
        float(np.min(np.linalg.norm(_sig_xy - np.array(_node_xy[n]), axis=1))) <= 35
        for n in (_members.get(node) or {node}) if n in _node_xy)


ext_acc = defaultdict(float)  # how far each chain end has been moved back from its junction cut
R_TURN = 6.0  # lane-centre radius a turning connector should be able to keep


def size_junction_gaps():
    """Lengthen the junction gaps so that the connectors fit (see the rules inside); uses the lane positions the
    pieces hold at the time of the call, so it runs once on the lane model and again on the fitted boundaries."""
    for node, lst in ends_at.items():
        if node in ring_nodes or len(lst) < 2:
            continue
        signalised = node_signalised(node)
        node_ext = False
        for it in range(4):
            need_end = defaultdict(float)
            for ci, e in lst:
                for cj, f in lst:
                    if (ci, e) == (cj, f):
                        continue
                    ins, outs = lane_ends(ci, e, True), lane_ends(cj, f, False)
                    if not ins or not outs:
                        continue
                    t0, t1 = ins[0][1], outs[0][1]
                    ang = math.degrees(math.atan2(t0[0] * t1[1] - t0[1] * t1[0], t0 @ t1))
                    if abs(ang) > 150:
                        continue  # U-turn: the junction stage builds no connector
                    maneuver = "straight" if abs(ang) < 35 else ("left" if ang > 0 else "right")
                    way_i = chain_by_idx[ci]["pieces"][0 if e == 0 else -1]["way"]
                    way_j = chain_by_idx[cj]["pieces"][0 if f == 0 else -1]["way"]
                    if not allowed(way_i, node, way_j, maneuver):
                        continue
                    if maneuver == "straight":
                        for (p0, t0), (p1, t1) in zip(ins, outs):  # paired from the right, like the connectors
                            d_ = p1 - p0
                            L_ = float(np.linalg.norm(d_))
                            delta = abs(t0[0] * d_[1] - t0[1] * d_[0])
                            L_req = math.sqrt(60.0 * delta)  # S-curve of a lateral shift `delta` with radius >= 10 m
                            for key_ in ((ci, e), (cj, f)):
                                need_end[key_] = max(need_end[key_], (L_req - L_) / 2)
                    else:
                        # turning pair (rightmost lanes for a right turn, leftmost for a left turn): both lane ends must
                        # be at least the tangent length R tan(angle / 2) away from the corner point where the two
                        # lane lines meet, otherwise the connector has to hook or turn tighter than R
                        (p0, t0), (p1, t1) = (ins[0], outs[0]) if maneuver == "right" else (ins[-1], outs[-1])
                        T_ = min(R_TURN * math.tan(math.radians(abs(ang)) / 2), 12.0)
                        try:
                            a_, b_ = np.linalg.solve(np.column_stack([t0, t1]), p1 - p0)  # p0 + a t0 = p1 - b t1
                        except np.linalg.LinAlgError:
                            continue
                        need_end[(ci, e)] = max(need_end[(ci, e)], min(T_ - a_, 15.0))
                        need_end[(cj, f)] = max(need_end[(cj, f)], min(T_ - b_, 15.0))
            if max(need_end.values(), default=0.0) < 0.3:
                break
            moved = False
            for ci, e in lst:  # lengthen the gap (capped: a chain keeps at least a third of its length; at signalised
                lo, hi = cut[ci]  # junctions an end that carries a stop line moves at most 2 m)
                L = chain_by_idx[ci]["length"]
                step_ = need_end[(ci, e)]
                if signalised and lane_ends(ci, e, True):
                    step_ = min(step_, max(0.0, 2.0 - ext_acc[(ci, e)]))
                if step_ <= 0.05:
                    continue
                if e == 0:
                    new = min(lo + step_, max(lo, min(L / 3, 20.0)))
                    moved |= new > lo + 1e-6
                    ext_acc[(ci, e)] += new - lo
                    lo = new
                else:
                    new = max(hi - step_, min(hi, max(L - L / 3, L - 20.0)))
                    moved |= new < hi - 1e-6
                    ext_acc[(ci, e)] += hi - new
                    hi = new
                cut[ci] = (lo, hi)
            if not moved:
                break
            node_ext = True
        if node_ext:
            gap_nodes.add(node)


size_junction_gaps()

# ------------------------------------------------------------------ lane boundaries from PCD paint tracks
# Every boundary is assigned to the precise paint line it represents in the point cloud map; where paint exists
# the boundary IS the paint centre line, dash gaps are interpolated, unpainted boundaries move with their neighbours.
if PCD_PATH:
    import cv2
    from pcd_refine import fit_chain_lines, simplify_indices
    _mask = np.zeros(ras.ground.shape, np.uint8)
    for c in chains:
        for p in c["pieces"]:
            for bxy in p["bounds_xy"]:
                pl = np.round((np.array(bxy) - [ras.x0, ras.y0]) / ras.res).astype(np.int32)
                cv2.polylines(_mask, [pl.reshape(-1, 1, 2)], False, 1, thickness=int(4.0 / ras.res))
    for g in rings.values():  # ring area (the arms' ends inside the ring are fitted like any road boundary)
        cxg, cyg = g["centre"]
        cv2.circle(_mask, (int((cxg - ras.x0) / ras.res), int((cyg - ras.y0) / ras.res)), int((np.max(g["r_out"]) + 3) / ras.res), 1, -1)
    if RGB_PCD:
        from pcd_refine import read_pcd_rgb
        _xyzi, _lum = read_pcd_rgb(RGB_PCD)
        SP = SurfacePoints(_xyzi, ras.ground, ras.x0, ras.y0, ras.res, _mask.astype(bool), lum=_lum)
    else:
        SP = SurfacePoints(PCD_XYZI, ras.ground, ras.x0, ras.y0, ras.res, _mask.astype(bool))
    tstats = defaultdict(int)
    kerb_pts, kerb_log = {}, []
    for c in chains:
        ci = c["index"]
        if ci in rings:
            continue
        ref = refs[ci]
        # knot spacing from the reference curvature: 3 m in tight curves (ring, corners), 8 m on straights
        kappa = np.abs(np.gradient(np.unwrap(np.arctan2(ref.t[:, 1], ref.t[:, 0])), ref.s))

        def knot_fn(x0, x1, kappa=kappa, ref=ref, ring=c.get("ring", False)):
            mm = (ref.s >= x0 - 2) & (ref.s <= x1 + 2)
            kmax = float(np.max(kappa[mm])) if mm.any() else 0.0
            return 3.0 if (ring or kmax > 1 / 40.0) else 5.0

        pcs = c["pieces"]
        fit_key = None
        if FIT_CACHE:
            import hashlib
            import pickle
            h_ = hashlib.sha1()
            h_.update(json.dumps([[p["stations"], p["bounds_d"], [sorted((a_, str(b_)) for a_, b_ in ln.items())
                                                                 for ln in p["lanes"]]] for p in pcs]).encode())
            h_.update(f"{os.path.getsize(PCD_PATH)}|{os.path.getmtime(PCD_PATH)}|{RGB_PCD}|{ras.res}|"
                      f"{sorted((k_, round(float(np.max(g_['r_out'])), 3)) for k_, g_ in rings.items())}|{os.path.getmtime(os.path.join(os.path.dirname(os.path.abspath(__file__)), "pcd_refine.py"))}|v2".encode())
            os.makedirs(FIT_CACHE, exist_ok=True)
            fit_key = os.path.join(FIT_CACHE, f"chain{ci}_{h_.hexdigest()[:16]}.pkl")
        if fit_key and os.path.exists(fit_key):
            new_bds, per_bound, lines = pickle.load(open(fit_key, "rb"))
            tstats["cache_hits"] += 1
        else:
            dmax = max(np.abs(np.array(p["bounds_d"])).max() for p in c["pieces"]) + 1.5
            idx = set()
            for q in ref.at(np.arange(0, ref.length + 4, 4.0)):
                idx.update(SP.kd.query_ball_point(q, dmax + 2.5))
            idx = np.fromiter(idx, np.int64)
            s_p, d_p = ref.project(SP.xy[idx])
            m = (s_p >= -1) & (s_p <= ref.length + 1) & (np.abs(d_p) <= dmax)
            o = np.argsort(s_p[m])
            s_c, d_c = s_p[m][o], d_p[m][o]
            i_c = SP.i[idx[m]][o]
            z_c = (SP.z - SP.g)[idx[m]][o]
            l_c = SP.lum[idx[m]][o] if SP.lum is not None else None
            new_bds, per_bound, lines = fit_chain_lines(s_c, d_c, i_c, z_c, pcs, knot_fn, lum=l_c)
            if fit_key:
                pickle.dump((new_bds, per_bound, lines), open(fit_key, "wb"))
        tstats["lines"] += len(lines)
        for pi, p in enumerate(pcs):
            st = np.array(p["stations"])
            bd = np.array(p["bounds_d"])
            nb = len(bd)
            new_bd = new_bds[pi]
            info = [per_bound[(pi, k)] for k in range(nb)]
            acc = np.array([x["accepted"] for x in info])
            delta = new_bd - bd
            if p.get("traj_modeled"):  # unpainted piece modelled from the driven path (traj_lanes.py): keep it
                acc[:] = False
                delta[:] = 0.0
            for k in range(nb):  # unpainted boundaries follow their painted neighbours (parallel, model width)
                if not acc[k]:
                    nbrs = [delta[j] for j in (k - 1, k + 1) if 0 <= j < nb and acc[j]]
                    delta[k] = np.mean(nbrs, axis=0) if nbrs else 0.0
            new_bd = bd + delta
            for k, ln in enumerate(p["lanes"]):  # full lanes keep 2.0 .. 5.5 m
                if ln.get("new_inner") or ln.get("new_outer"):
                    continue
                w_ = np.abs(new_bd[k + 1] - new_bd[k])
                if w_.min() < 2.0 or w_.max() > 6.5:
                    bad = k if info[k]["coverage"] < info[k + 1]["coverage"] else k + 1
                    nbrs = [delta[j] for j in (bad - 1, bad + 1) if 0 <= j < nb and acc[j] and j != bad]
                    new_bd[bad] = bd[bad] + (np.mean(nbrs, axis=0) if nbrs else 0.0)
                    acc[bad] = False
                    tstats["sanity_reverts"] += 1
            p["bounds_d"] = new_bd.tolist()
            p["_acc"] = acc.tolist()
            # per boundary and station: backed by paint evidence (an unpainted boundary inherits the flags of
            # the painted neighbours it runs parallel to; without any it counts as unsupported)
            sup_ = []
            for k in range(nb):
                own = info[k].get("supported")
                if acc[k] and own is not None:
                    sup_.append(np.array(own, bool))
                else:
                    nb_s = [np.array(info[j]["supported"], bool) for j in (k - 1, k + 1)
                            if 0 <= j < nb and acc[j] and info[j].get("supported") is not None]
                    # no painted neighbour either: the raw lane model, smoothed by the continuity pass below
                    sup_.append(np.all(nb_s, axis=0) if nb_s else np.zeros(len(st), bool))
            p["_sup"] = [x.tolist() for x in sup_]
            if os.environ.get("LANE_DEBUG"):
                for k in range(nb):
                    x = info[k]
                    print(f"  chain {ci} piece {pi} k {k} ({p['s0']:.0f}-{p['s1']:.0f} m): acc={acc[k]} cov={x['coverage']:.2f} "
                          f"resid={(x['resid']*100 if np.isfinite(x['resid']) else -1):.1f} n={x['n']} duty={x['duty']:.2f}", flush=True)
            for k in range(nb):
                if acc[k]:
                    tstats["painted_bounds"] += 1
                    tstats["resid_sum"] += info[k]["resid"]
                    p["bound_types"][k] = "solid_solid" if info[k]["double"] > 0.5 * info[k]["n"] else \
                        ("solid" if info[k]["duty"] > 0.75 else "dashed")
                else:
                    tstats["unpainted_bounds"] += 1
        # pieces without any paint: lanes from the kerbs in the point cloud (kerb_lanes.py)
        if not any(n_ in ring_nodes for n_ in c["nodes"]):
            for pi, p in enumerate(pcs):
                if any(p["_acc"]) or any(ln.get("new_inner") or ln.get("new_outer") for ln in p["lanes"]) or \
                        p.get("traj_modeled"):
                    continue  # painted, a lane tapers in / out here (the lane model carries the taper), or the
                    # piece is already modelled from the driven path
                st = np.array(p["stations"])
                lo_, hi_ = cut[ci]
                use = (st >= lo_ - 3.0) & (st <= hi_ + 3.0)
                if use.sum() < 12:
                    continue
                if ci not in kerb_pts:
                    from kerb_lanes import chain_points, kerb_bounds, seed_from_trajectory
                    m_k = np.zeros(ras.ground.shape, np.uint8)
                    pl = np.round((ref.xy - [ras.x0, ras.y0]) / ras.res).astype(np.int32)
                    cv2.polylines(m_k, [pl.reshape(-1, 1, 2)], False, 1, thickness=int(40.0 / ras.res))
                    kerb_pts[ci] = chain_points(SurfacePoints(PCD_XYZI, ras.ground, ras.x0, ras.y0, ras.res,
                                                              m_k.astype(bool), zlo=-0.5, zhi=0.6), ref)
                bd = np.array(p["bounds_d"])
                seed, ddir = seed_from_trajectory(ref, traj[:, :2], st, 0.5 * (bd[0] + bd[-1]))
                res, kinfo = kerb_bounds(*kerb_pts[ci], st, seed, p["lanes"], ddir, use)
                kerb_log.append((ci, p["s0"], p["s1"], p.get("name") or "", kinfo))
                if res is None:
                    continue
                new, keep = res["bounds"], res["keep"]
                rows = list(range(keep[0], keep[-1] + 2))  # the old boundaries of the lanes kept
                old = bd[rows]
                iu = np.flatnonzero(np.isfinite(new[0]))
                i0, i1 = iu[0], iu[-1]
                for k in range(len(new)):  # beyond the stations used: the model's shape, shifted to join
                    new[k, :i0] = new[k, i0] + (old[k, :i0] - old[k, i0])
                    new[k, i1 + 1:] = new[k, i1] + (old[k, i1 + 1:] - old[k, i1])
                if "trimmed_to" in kinfo:  # a kerb return: the junction area starts there
                    s_a, s_b = kinfo["trimmed_to"]
                    if pi == 0 and s_a > lo_:
                        lo_ = float(s_a)
                    if pi == len(pcs) - 1 and s_b < hi_:
                        hi_ = float(s_b)
                    cut[ci] = (lo_, hi_)
                lanes_new = [p["lanes"][i] for i in keep]
                types = []
                for k in range(len(new)):
                    if k in (0, len(new) - 1):
                        types.append("road_border")
                    elif lanes_new[k - 1]["dir"] != lanes_new[k]["dir"]:
                        types.append("virtual")  # an unpainted centre line between the two directions
                    else:
                        types.append(p["bound_types"][rows[k]])
                if len(keep) < len(p["lanes"]):  # separated carriageway: the other direction is not on this strip
                    if not any(ln["dir"] == 1 for ln in lanes_new):
                        p["turnR"] = []
                    if not any(ln["dir"] == -1 for ln in lanes_new):
                        p["turnL"] = []
                    p["nR"] = sum(1 for ln in lanes_new if ln["dir"] == 1)
                    p["nL"] = sum(1 for ln in lanes_new if ln["dir"] == -1)
                p["lanes"] = lanes_new
                p["bounds_d"] = new.tolist()
                p["bound_types"] = types
                p["_acc"] = [False] * len(new)
                p["_sup"] = [[True] * len(st) for _ in range(len(new))]
                p["_kerb"] = True
        # continuity at way splits: matching boundaries must meet; the paint-fitted side wins, the other blends in
        BL = 8.0
        for pi in range(1, len(c["pieces"])):
            A, B = c["pieces"][pi - 1], c["pieces"][pi]
            stA, stB = np.array(A["stations"]), np.array(B["stations"])
            bdA, bdB = np.array(A["bounds_d"]), np.array(B["bounds_d"])
            accA, accB = A.get("_acc", [False] * len(bdA)), B.get("_acc", [False] * len(bdB))
            endA = bdA[:, -1]
            for k in range(len(bdB)):
                j = int(np.argmin(np.abs(endA - bdB[k, 0])))
                gap = endA[j] - bdB[k, 0]
                if abs(gap) < 1e-4 or abs(gap) > 2.0:
                    continue
                fa, fb = bool(accA[j]), bool(accB[k])
                share_b = 1.0 if (fa and not fb) else (0.0 if (fb and not fa) else 0.5)
                wB = np.clip(1 - (stB - stB[0]) / min(BL, max(stB[-1] - stB[0], 1e-6)), 0, 1)
                wA = np.clip(1 - (stA[-1] - stA) / min(BL, max(stA[-1] - stA[0], 1e-6)), 0, 1)
                bdB[k] = bdB[k] + gap * share_b * wB
                bdA[j] = bdA[j] - gap * (1 - share_b) * wA
                endA = bdA[:, -1]
                tstats["split_joins"] += 1
            A["bounds_d"], B["bounds_d"] = bdA.tolist(), bdB.tolist()
        # final continuity pass: every linked line is refitted as one smooth spline across its pieces, and that
        # spline is blended in only (a) within a few metres of the piece joins that do not already meet smoothly
        # and (b) along stretches without paint evidence, where the line otherwise follows the lane-model shape;
        # the paint fit and the lines parallel to it stay exactly as fitted everywhere else
        from pcd_refine import link_lines, pspline
        lines2, line_of2 = link_lines(pcs)
        for L in lines2:
            if len(L) < 2 and len(pcs[L[0][0]]["stations"]) < 6:
                continue
            s_cat, d_cat, w_cat = [], [], []
            for pi, k in L:
                st = np.array(pcs[pi]["stations"])
                bdk = np.array(pcs[pi]["bounds_d"])[k]
                wk = np.where(np.array(pcs[pi]["_sup"][k], bool), 1.0, 0.15)  # unsupported stations steer it less
                if s_cat and abs(st[0] - s_cat[-1][-1]) < 1e-6:
                    st, bdk, wk = st[1:], bdk[1:], wk[1:]
                s_cat.append(st)
                d_cat.append(bdk)
                w_cat.append(wk)
            s_cat, d_cat, w_cat = np.concatenate(s_cat), np.concatenate(d_cat), np.concatenate(w_cat)
            if s_cat[-1] - s_cat[0] < 6:
                continue
            spl, _ = pspline(s_cat, d_cat, w_cat, s_cat[0], s_cat[-1], knot_fn(s_cat[0], s_cat[-1]), lam=2.0,
                             iters=1, inlier=10.0)
            if spl is None:
                continue
            # joins that need smoothing: one side is unpainted, or the two pieces do not meet smoothly (they were
            # fitted separately); a line fitted as one spline across the join is left exactly as fitted
            joins = []
            for (pa_, ka_), (pb_, kb_) in zip(L[:-1], L[1:]):
                sa_, sb_ = np.array(pcs[pa_]["stations"]), np.array(pcs[pb_]["stations"])
                da_, db_ = np.array(pcs[pa_]["bounds_d"])[ka_], np.array(pcs[pb_]["bounds_d"])[kb_]
                both_ = pcs[pa_].get("_acc", [False] * len(pcs[pa_]["bounds_d"]))[ka_] and \
                    pcs[pb_].get("_acc", [False] * len(pcs[pb_]["bounds_d"]))[kb_]
                sl_a = (da_[-1] - da_[-2]) / max(sa_[-1] - sa_[-2], 1e-6) if len(sa_) > 1 else 0.0
                sl_b = (db_[1] - db_[0]) / max(sb_[1] - sb_[0], 1e-6) if len(sb_) > 1 else 0.0
                if not both_ or abs(db_[0] - da_[-1]) > 0.005 or abs(sl_b - sl_a) > 0.01:
                    joins.append(float(sb_[0]))
            for pi, k in L:
                st = np.array(pcs[pi]["stations"])
                bd = np.array(pcs[pi]["bounds_d"])
                w_ = np.zeros(len(st))
                if joins:
                    dj = np.min(np.abs(st[:, None] - np.array(joins)[None, :]), axis=1)
                    w_ = np.clip(1.0 - (dj - 1.0) / 4.0, 0.0, 1.0)  # full within 1 m of such a join, none beyond 5 m
                uns = ~np.array(pcs[pi]["_sup"][k], bool)
                if uns.any():  # stretches without paint evidence: the smooth line, faded in over 3 m
                    du = np.min(np.abs(st[:, None] - st[uns][None, :]), axis=1)
                    w_ = np.maximum(w_, np.clip(1.0 - du / 3.0, 0.0, 1.0))
                if not w_.any():
                    continue
                bd[k] = w_ * spl(st) + (1.0 - w_) * bd[k]
                pcs[pi]["bounds_d"] = bd.tolist()
    for ci_, s0_, s1_, nm_, ki_ in kerb_log:
        print(f"unpainted piece chain {ci_} {s0_:.0f}-{s1_:.0f} m ({nm_}): " + (f"kept on the lane model ({ki_['rejected']})"
              if "rejected" in ki_ else f"lanes from the kerbs, {ki_['case']}, {ki_['lanes']} x {ki_['lane_width_median']} m"
              + (f", used {ki_['trimmed_to'][0]:.1f}-{ki_['trimmed_to'][1]:.1f} m" if "trimmed_to" in ki_ else "")))
    print(f"PCD paint fit: {tstats['lines']} lane lines; boundaries on paint {tstats['painted_bounds']} (mean robust residual "
          f"{tstats['resid_sum']/max(tstats['painted_bounds'],1)*100:.1f} cm), without paint (moved with neighbours) "
          f"{tstats['unpainted_bounds']}, sanity reverts {tstats['sanity_reverts']}, way-split joins {tstats['split_joins']}")

size_junction_gaps()  # again, with the fitted / kerb-based lane positions

# ------------------------------------------------------------------ driven-path fallback
# A fitted piece whose driven lane does not contain the driven path (paint fit locked onto parking / bus-stop
# hatching, or the lane model is off) is re-modelled from the driven path like an unpainted one (traj_lanes.py).
from traj_lanes import DrivenPath, driven_lane, remodel
_dp = DrivenPath(np.load(TRAJ)["pos"][::50, :2])
n_fb = 0
for c in chains:
    if c.get("ring") or c["index"] in rings:
        continue
    ref_ = refs[c["index"]]
    ps_, pd_, al_, nr_ = _dp.on_chain(ref_)
    for p in c["pieces"]:
        if p.get("traj_modeled"):
            continue
        inf_ = driven_lane(p, ps_, pd_, al_, nr_)
        if inf_ is None:
            continue
        if inf_[3] >= 0.6 and (inf_[4] > 0.8 or inf_[5] < 0.7):
            log_ = remodel(p, ref_, ps_, pd_, al_, nr_, inf_)
            if log_:
                n_fb += 1
                print(f"  driven-path fallback chain {c['index']} {log_}")
# propagate: a fitted piece next to a re-modelled one whose lanes cannot take over the driven lane at the seam
# (no lane centre within 1 m of the re-modelled driven lane) is re-modelled as well
n_prop, changed_ = 0, True
while changed_:
    changed_ = False
    for c in chains:
        if c.get("ring") or c["index"] in rings:
            continue
        pcs_ = c["pieces"]
        ref_ = refs[c["index"]]
        ps_, pd_, al_, nr_ = None, None, None, None
        for pi, p in enumerate(pcs_):
            if p.get("traj_modeled"):
                continue
            bad_ = False
            for side, ni in ((0, pi - 1), (1, pi + 1)):
                if not (0 <= ni < len(pcs_)) or not pcs_[ni].get("traj_modeled"):
                    continue
                q = pcs_[ni]
                qb_, kq = np.array(q["bounds_d"]), q.get("driven_lane", 0)
                qe = qb_[:, -1] if side == 0 else qb_[:, 0]   # neighbour at the seam
                c_q = 0.5 * (qe[kq] + qe[kq + 1])
                pb_ = np.array(p["bounds_d"])
                pe = pb_[:, 0] if side == 0 else pb_[:, -1]
                cen = [0.5 * (pe[j] + pe[j + 1]) for j in range(len(pe) - 1)
                       if p["lanes"][j]["dir"] == q["lanes"][kq]["dir"]]
                if not cen or min(abs(np.array(cen) - c_q)) > 1.0:
                    bad_ = True
            if not bad_:
                continue
            if ps_ is None:
                ps_, pd_, al_, nr_ = _dp.on_chain(ref_)
            log_ = remodel(p, ref_, ps_, pd_, al_, nr_)
            if log_:
                n_prop += 1
                changed_ = True
                print(f"  driven-path propagation chain {c['index']} {log_}")
print(f"fitted pieces whose lanes missed the driven path, re-modelled from it: {n_fb} (+{n_prop} neighbours)")

# ------------------------------------------------------------------ path-modelled pieces meet their neighbours
# A piece modelled from the driven path (traj_lanes.py) ramps into the final (paint / kerb fitted) end offsets of
# a neighbouring piece with the same lanes over up to 15 m, so the boundaries share their end points there and
# lanelet2 keeps the successor relation.
n_blend = 0
for c in chains:
    pcs_ = c["pieces"]
    for pi, p in enumerate(pcs_):
        if not p.get("traj_modeled"):
            continue
        st = np.array(p["stations"])
        bd = np.array(p["bounds_d"])
        L_ = min(15.0, 0.5 * (st[-1] - st[0]))
        for side, ni in ((0, pi - 1), (1, pi + 1)):
            if not (0 <= ni < len(pcs_)) or pcs_[ni].get("traj_modeled"):
                continue
            qb = np.array(pcs_[ni]["bounds_d"])
            if len(qb) != len(bd):
                # different lane count: the driven lane meets the neighbour lane whose centre is closest at the
                # seam (its two boundaries then share points); the piece's other boundaries move rigidly with it
                kd_ = p.get("driven_lane")
                if kd_ is None:
                    continue
                qe = qb[:, -1] if side == 0 else qb[:, 0]
                be = bd[:, 0] if side == 0 else bd[:, -1]
                c_drv = 0.5 * (be[kd_] + be[kd_ + 1])
                j_ = int(np.argmin([abs(0.5 * (qe[j] + qe[j + 1]) - c_drv) for j in range(len(qe) - 1)]))
                if abs(0.5 * (qe[j_] + qe[j_ + 1]) - c_drv) > 3.0:
                    continue
                sh_lo, sh_hi = qe[j_] - be[kd_], qe[j_ + 1] - be[kd_ + 1]
                target = be + 0.5 * (sh_lo + sh_hi)
                target[kd_], target[kd_ + 1] = qe[j_], qe[j_ + 1]
                for k_ in range(kd_ - 1, -1, -1):  # keep the other lanes' widths next to the driven lane
                    target[k_] = target[k_ + 1] - (be[k_ + 1] - be[k_])
                for k_ in range(kd_ + 2, len(be)):
                    target[k_] = target[k_ - 1] + (be[k_] - be[k_ - 1])
                cur = be
                w_ = np.clip(1 - ((st - st[0]) if side == 0 else (st[-1] - st)) / max(L_, 1e-6), 0, 1)
                bd = bd + np.outer(target - cur, w_)
                n_blend += 1
                continue
            target, cur = (qb[:, -1], bd[:, 0]) if side == 0 else (qb[:, 0], bd[:, -1])
            w_ = np.clip(1 - ((st - st[0]) if side == 0 else (st[-1] - st)) / max(L_, 1e-6), 0, 1)
            bd = bd + np.outer(target - cur, w_)
            n_blend += 1
        p["bounds_d"] = bd.tolist()
print(f"path-modelled piece ends blended into their neighbours: {n_blend}")

# ------------------------------------------------------------------ road lanelets
approach = defaultdict(list)  # (node, 'in'|'out') -> groups
all_lanelets = []
bound_recs = []  # every road boundary: points, kind (paint / curb / None), road side for curbs
wedge_excl = []  # lanelets next to a taper wedge (validator exclusion for mapping.lane.border_sharing)
lane_recs = []   # (lanelet, lane info) for the width guard
dbg_bounds = []  # where every road boundary linestring comes from (written to <map>_debug_bounds.json)
for c in chains:
    ci = c["index"]
    if ci in rings:
        continue
    lo, hi = cut[ci]
    prev_pts = None  # boundary end points of the previous piece (for point sharing)
    pieces = [p for p in c["pieces"] if p["s1"] > lo + 0.5 and p["s0"] < hi - 0.5]
    MAXLEN = 60.0  # road lanelets are split into pieces of at most this length (shared points keep them connected)
    for pi, p in enumerate(pieces):
        st = np.array(p["stations"])
        a, b = max(p["s0"], lo), min(p["s1"], hi)
        ss_all = np.unique(np.r_[a, st[(st > a) & (st < b)], b])
        bd = np.array(p["bounds_d"])
        nb = len(bd)
        # tapering lanes: the lanelet of a lane that appears (or disappears) within the piece exists only where the
        # lane is at least 0.5 m wide (+2 m); the road is cut there so the through lanes stay connected
        lane_w = [np.abs(bd[k + 1] - bd[k]) for k in range(nb - 1)]
        skip_before, skip_after, extra_cuts = {}, {}, []
        for k, ln in enumerate(p["lanes"]):
            if not (ln.get("new_inner") or ln.get("new_outer")):
                continue
            w = lane_w[k]
            wide = np.flatnonzero(w >= 0.5)
            if w[0] < 1.0:
                s_star = min(st[wide[0]] + 2.0, b) if len(wide) else b
                skip_before[k] = s_star
                extra_cuts.append(s_star)
            if w[-1] < 1.0:
                s_star = max(st[wide[-1]] - 2.0, a) if len(wide) else a
                skip_after[k] = s_star
                extra_cuts.append(s_star)
        nseg0 = max(1, int(np.ceil((b - a) / MAXLEN)))
        cuts = np.unique(np.r_[np.linspace(a, b, nseg0 + 1), [c_ for c_ in extra_cuts if a + 1.0 < c_ < b - 1.0]])
        nseg = len(cuts) - 1
        for sj in range(nseg):
            ss = ss_all[(ss_all >= cuts[sj] - 1e-9) & (ss_all <= cuts[sj + 1] + 1e-9)]
            ss = np.unique(np.r_[cuts[sj], ss, cuts[sj + 1]])
            if len(ss) < 2:
                continue
            bpts = []
            for k in range(nb):
                ss_k = ss
                if pi == len(pieces) - 1 and sj == nseg - 1 and (ci, 1) in ring_bound_cut:
                    s_k = max(ring_bound_cut[(ci, 1)][k], cuts[sj] + 0.2)
                    ss_k = np.unique(np.r_[ss[ss < s_k - 1e-6], s_k])
                if pi == 0 and sj == 0 and (ci, 0) in ring_bound_cut:
                    s_k = min(ring_bound_cut[(ci, 0)][k], cuts[sj + 1] - 0.2)
                    ss_k = np.unique(np.r_[s_k, ss[ss > s_k + 1e-6]])
                dk = np.interp(ss_k, st, bd[k])
                xy = refs[ci].at(ss_k, dk)
                if PCD_PATH:  # keep only the vertices needed to stay within 1 cm of the fitted line (max 5 m apart)
                    xy = xy[simplify_indices(xy, 0.01, 5.0)]
                z = ground_z(xy)
                pts = [mk_point(x, y, zz) for (x, y), zz in zip(xy, z)]
                bpts.append(pts)
            # a tapering lane's new boundary starts (or ends) exactly on the neighbour it grows out of
            # (only where a previous / next piece exists: at a junction cut nothing has to stay connected, and
            # snapping there would bend the boundary by the wedge width within one vertex)
            snapped = defaultdict(list)

            def snap_end(pts, target, at_start):
                """Put the end vertex of a boundary onto `target` and spread the shift over the vertices behind it
                (linear ramp over 12x the shift, at least 4 m, at most the whole line): a taper, not a kink."""
                seq = pts if at_start else pts[::-1]
                dx, dy = target.x - seq[0].x, target.y - seq[0].y
                shift = math.hypot(dx, dy)
                if shift > 0.02 and len(seq) > 2:
                    s_ = np.r_[0, np.cumsum([math.hypot(b_.x - a_.x, b_.y - a_.y) for a_, b_ in zip(seq[:-1], seq[1:])])]
                    Lr = min(float(s_[-1]), max(4.0, 12.0 * shift))
                    for q_, si_ in zip(seq[1:-1], s_[1:-1]):
                        f_ = max(0.0, 1.0 - si_ / Lr)
                        q_.x, q_.y = q_.x + f_ * dx, q_.y + f_ * dy
                pts[0 if at_start else -1] = target

            for k_ in skip_before:
                if sj == 0 and prev_pts is not None:
                    new_b, nbr_b = (k_, k_ + 1) if p["lanes"][k_].get("side") == "R" else (k_ + 1, k_)
                    snap_end(bpts[new_b], bpts[nbr_b][0], True)
                    snapped[new_b].append(0)
            for k_ in skip_after:
                if sj == nseg - 1 and pi < len(pieces) - 1:
                    new_b, nbr_b = (k_, k_ + 1) if p["lanes"][k_].get("side") == "R" else (k_ + 1, k_)
                    snap_end(bpts[new_b], bpts[nbr_b][-1], False)
                    snapped[new_b].append(1)
            # share transition points with the previous segment / piece (one-to-one, nearest pairs first)
            if prev_pts is not None:
                cand = sorted((math.hypot(bpts[k][0].x - o.x, bpts[k][0].y - o.y), k, j)
                              for k in range(nb) for j, o in enumerate(prev_pts))
                usedk, usedj = set(), set()
                for dd, k, j in cand:
                    if k in usedk:
                        continue
                    if dd < 0.05:  # coincident (zero-width taper start): several boundaries share the point
                        usedk.add(k)
                        usedj.add(j)
                        bpts[k][0] = prev_pts[j]
                        continue
                    if dd > 0.3 or j in usedj:
                        continue
                    usedk.add(k)
                    usedj.add(j)
                    bpts[k][0] = prev_pts[j]
            prev_pts = [bp[-1] for bp in bpts]
            lss = [mk_ls(bpts[k], BOUND_ATTR[p["bound_types"][k]]) for k in range(nb)]
            for k in range(nb):
                dbg_bounds.append({"ls": lss[k].id, "chain": ci, "piece": [float(p["s0"]), float(p["s1"])],
                                   "seg": [float(cuts[sj]), float(cuts[sj + 1])], "k": k, "nb": nb,
                                   "type": p["bound_types"][k], "acc": bool(p.get("_acc", [False] * nb)[k]),
                                   "skip_before": {str(a_): float(b_) for a_, b_ in skip_before.items()},
                                   "skip_after": {str(a_): float(b_) for a_, b_ in skip_after.items()},
                                   "snapped": [int(x_) for x_ in snapped.get(k, [])],
                                   "lanes": [{q_: ln[q_] for q_ in ("dir", "side", "new_inner", "new_outer") if q_ in ln}
                                             for ln in p["lanes"]]})
            for k in range(nb):
                ty = p["bound_types"][k]
                bound_recs.append({"pts": bpts[k], "ls": lss[k],  # kerb-based pieces are not refitted to the kerb again
                                   "kind": None if (p.get("_kerb") or p.get("traj_modeled")) else ("paint" if ty in ("solid", "dashed", "solid_solid")
                                                                         else ("curb" if ty == "road_border" else None)),
                                   "road_sign": 1 if k == 0 else (-1 if k == nb - 1 else 0)})
            lane_objs = []
            skipped_here = []
            for k, ln in enumerate(p["lanes"]):
                if (k in skip_before and cuts[sj] < skip_before[k] - 1e-6) or \
                        (k in skip_after and cuts[sj + 1] > skip_after[k] + 1e-6):
                    skipped_here.append(k)
                    continue  # the lane does not exist (yet / any more) in this segment
                if ln["dir"] == 1:
                    left, right = lss[k + 1], lss[k]
                else:
                    left, right = lss[k].invert(), lss[k + 1].invert()
                attrs = {"type": "lanelet", "subtype": "road", "location": "urban", "one_way": "yes",
                         "participant:vehicle": "yes", "speed_limit": speed(p["maxspeed"]),
                         "osm_way": str(p["way"]), "road_name": p["name"] or ""}
                ll = Lanelet(getId(), left, right, AttributeMap(attrs))
                lmap.add(ll)
                all_lanelets.append(ll)
                lane_objs.append((k, ln, ll))
                lane_recs.append((ll, ln))
            # lanelets either side of a taper wedge (< 0.5 m, no lanelet) look adjacent to the validator's heuristic
            for k_ in skipped_here:
                for lo_ in lane_objs:
                    if lo_[0] in (k_ - 1, k_ + 1):
                        wedge_excl.append(lo_[2].id)
            # approach groups at the chain ends (lanes ordered left -> right in their travel direction)
            fwd = sorted([lo_ for lo_ in lane_objs if lo_[1]["dir"] == 1], key=lambda z: -z[0])
            bwd = sorted([lo_ for lo_ in lane_objs if lo_[1]["dir"] == -1], key=lambda z: z[0])
            n0, n1 = c["nodes"]
            if pi == len(pieces) - 1 and sj == nseg - 1:
                if fwd:
                    approach[(n1, "in")].append({"chain": ci, "end": 1, "way": p["way"], "lanes": [z[2] for z in fwd],
                                                 "turns": p["turnR"], "dir": -end_dir(ci, 1)})
                if bwd:
                    approach[(n1, "out")].append({"chain": ci, "end": 1, "way": p["way"], "lanes": [z[2] for z in bwd],
                                                  "dir": end_dir(ci, 1)})
            if pi == 0 and sj == 0:
                if fwd:
                    approach[(n0, "out")].append({"chain": ci, "end": 0, "way": p["way"], "lanes": [z[2] for z in fwd],
                                                  "dir": end_dir(ci, 0)})
                if bwd:
                    approach[(n0, "in")].append({"chain": ci, "end": 0, "way": p["way"], "lanes": [z[2] for z in bwd],
                                                 "turns": p["turnL"], "dir": -end_dir(ci, 0)})
print(f"road lanelets: {len(all_lanelets)}")

# ------------------------------------------------------------------ fit road boundaries to the point cloud map
if PCD_PATH:
    import cv2
    mask = np.zeros(ras.ground.shape, np.uint8)
    for r_ in bound_recs:
        pl = np.round((np.array([[q.x, q.y] for q in r_["pts"]]) - [ras.x0, ras.y0]) / ras.res).astype(np.int32)
        cv2.polylines(mask, [pl.reshape(-1, 1, 2)], False, 1, thickness=int(3.0 / ras.res))
    sp = SP
    orig_xy = {q.id: (q.x, q.y) for r_ in bound_recs for q in r_["pts"]}

    def tame_offsets(orig, new, anchors, max_dev=0.8, max_turn=25.0):
        """Lateral shifts of a refined line relative to its model line, clamped to the median shift +- max_dev,
        smoothed along the line (2 m) and freed of kinks (turns above max_turn degrees are flattened); anchored
        vertices stay put."""
        v = np.diff(orig, axis=0)
        v = np.vstack([v, v[-1:]])
        nrm = np.column_stack([-v[:, 1], v[:, 0]])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-9)[:, None]
        off = np.sum((new - orig) * nrm, axis=1)
        along = np.sum((new - orig) * (v / np.maximum(np.linalg.norm(v, axis=1), 1e-9)[:, None]), axis=1)
        med = np.median(off)
        off2 = np.clip(off, med - max_dev, med + max_dev)
        n_changed = int(np.sum(np.abs(off2 - off) > 1e-3))
        # the kerb evidence scatters by a decimetre or two from station to station: smooth the shift along the line
        s_arc = np.r_[0, np.cumsum(np.linalg.norm(np.diff(orig, axis=0), axis=1))]
        w_s = np.exp(-0.5 * ((s_arc[:, None] - s_arc[None, :]) / 2.0) ** 2)
        off2 = (w_s @ off2) / w_s.sum(axis=1)
        for _ in range(10):  # flatten kinks
            cand = orig + off2[:, None] * nrm
            w = np.diff(cand, axis=0)
            Lw = np.linalg.norm(w, axis=1)
            ok = Lw > 1e-6
            if ok.sum() < 2:
                break
            h = np.unwrap(np.arctan2(w[:, 1], w[:, 0]))
            turn = np.degrees(np.abs(np.diff(h)))
            bad = np.flatnonzero(turn > max_turn) + 1
            if not len(bad):
                break
            for i in bad:
                off2[i] = 0.5 * (off2[i - 1] + off2[min(i + 1, len(off2) - 1)])
            n_changed += len(bad)
        out = orig + off2[:, None] * nrm
        if anchors is not None:
            out[anchors] = new[anchors]
        return out, n_changed

    # merge boundaries of consecutive pieces that share end points into continuous lines (same kind / side)
    by_first = defaultdict(list)
    for i, r_ in enumerate(bound_recs):
        by_first[r_["pts"][0].id].append(i)
    has_pred = set()
    nxt = {}
    for i, r_ in enumerate(bound_recs):
        cands = [j for j in by_first.get(r_["pts"][-1].id, []) if j != i and bound_recs[j]["kind"] == r_["kind"]
                 and bound_recs[j]["road_sign"] == r_["road_sign"]]
        if len(cands) == 1 and len(by_first[r_["pts"][-1].id]) == 1:
            nxt[i] = cands[0]
            has_pred.add(cands[0])
    lines = []
    for i in range(len(bound_recs)):
        if i in has_pred:
            continue
        seq = [i]
        while seq[-1] in nxt and nxt[seq[-1]] not in seq:
            seq.append(nxt[seq[-1]])
        lines.append(seq)
    done = set()
    stats = defaultdict(list)
    for kind in ("curb",):
        for seq in lines:
            if bound_recs[seq[0]]["kind"] != kind:
                continue
            pts = []
            for j in seq:
                for q in bound_recs[j]["pts"]:
                    if not pts or q.id != pts[-1].id:
                        pts.append(q)
            xy = np.array([[q.x, q.y] for q in pts])
            anchors = np.array([q.id in done for q in pts])
            new_xy, rep = refine_polyline(sp, xy, kind, bound_recs[seq[0]]["road_sign"], anchors=anchors)
            if rep["n_meas"] < 4 or not rep.get("accepted") or rep["moved_median"] > (1.0 if kind == "paint" else 3.6):
                stats[kind + "_skipped"].append(len(pts))
                continue
            if kind == "curb":  # a lane edge does not follow bays / kerb returns: keep the shift within 0.8 m of
                new_xy, n_tamed = tame_offsets(xy, new_xy, anchors)  # the line's median shift, no kinks
                stats["curb_tamed"].append(n_tamed)
            for q, (x, y) in zip(pts, new_xy):
                q.x, q.y = float(x), float(y)
            done.update(q.id for q in pts)
            stats[kind].append((rep["inlier_cov"], rep["robust_resid"], rep["moved_median"], rep["n_meas"], rep["double"]))
            line_of_point = locals().setdefault("line_of_point", {})
            for q in pts:
                line_of_point.setdefault(q.id, []).append(tuple(q.id for q in pts))
            if kind == "paint" and rep["double"] > rep["n_meas"] * 0.5:
                for j in seq:
                    bound_recs[j]["ls"].attributes["subtype"] = "solid_solid" \
                        if bound_recs[j]["ls"].attributes["subtype"] == "solid" else bound_recs[j]["ls"].attributes["subtype"]

    # width guard: a fit must not squeeze a full lane below 2 m (it would have locked onto a neighbouring line)
    def min_width(ll, skip_ends):
        lb = np.array([[q.x, q.y] for q in ll.leftBound])
        rb = np.array([[q.x, q.y] for q in ll.rightBound])
        from pcd_refine import resample as _rs
        a, sa = _rs(lb, 1.0)
        b, _ = _rs(rb, 1.0)
        n = min(len(a), len(b))
        d = np.min(np.linalg.norm(a[:n, None, :] - b[None, :n, :], axis=2), axis=1)
        k0 = int(skip_ends / 1.0)
        core = d[k0:n - k0] if n > 2 * k0 + 2 else d
        return float(core.min()) if len(core) else 99.0

    reverted = 0
    for ll, ln in lane_recs:
        taper = ln.get("new_inner") or ln.get("new_outer")
        if min_width(ll, 16.0 if taper else 1.0) < 2.0:
            moved = []
            for bnd in (ll.leftBound, ll.rightBound):
                mv_ = [np.hypot(q.x - orig_xy[q.id][0], q.y - orig_xy[q.id][1]) for q in bnd if q.id in orig_xy]
                moved.append(np.median(mv_) if mv_ else 0.0)
            bnd = (ll.leftBound, ll.rightBound)[int(np.argmax(moved))]
            for q in bnd:
                if q.id in orig_xy:
                    q.x, q.y = orig_xy[q.id]
            reverted += 1
    # elevations from the PCD road surface, local_x/local_y tags follow the fitted positions
    allp = {q.id: q for r_ in bound_recs for q in r_["pts"]}
    pl = list(allp.values())
    zz = surface_z(sp, np.array([[q.x, q.y] for q in pl]))
    for q, zv in zip(pl, zz):
        if np.isfinite(zv):
            q.z = float(zv)
        q.attributes["local_x"] = f"{q.x:.3f}"
        q.attributes["local_y"] = f"{q.y:.3f}"
    for kind in ("paint", "curb"):
        a_ = np.array(stats[kind]) if stats[kind] else np.zeros((0, 5))
        if len(a_):
            print(f"PCD fit {kind}: {len(a_)} lines fitted, {len(stats[kind + '_skipped'])} without evidence; "
                  f"robust station residual median {np.nanmedian(a_[:, 1])*100:.1f} cm; moved median {np.median(a_[:, 2])*100:.1f} cm; "
                  f"inlier coverage median {np.median(a_[:, 0])*100:.0f}%")
    print(f"width guard reverted {reverted} boundaries; curb lines with clamped bays / kinks: "
          f"{sum(1 for n_ in stats['curb_tamed'] if n_)}; elevations from PCD for {np.isfinite(zz).sum()}/{len(zz)} vertices")

# ------------------------------------------------------------------ junction connectors
def lane_dir(ll, at_end):
    cl = ll.centerline
    pts = np.array([[q.x, q.y] for q in cl])
    v = pts[-1] - pts[-2] if at_end else pts[1] - pts[0]
    return v / max(np.linalg.norm(v), 1e-9)


def turn_ok(tag, maneuver):
    tag = tag or "through"
    parts = tag.split(";")
    if maneuver == "left":
        return any("left" in q for q in parts)
    if maneuver == "right":
        return any("right" in q for q in parts)
    return any(q in ("through", "none", "", "merge_to_left", "merge_to_right") for q in parts)


connectors = []
conn_by_node = defaultdict(list)
vcache = {}


def build_rings():
    """Circulating lanelets of every roundabout and the merge / diverge connectors of its driven arms."""
    n_ring = 0
    for rci, g in rings.items():
        c = chain_by_idx[rci]
        p0 = c["pieces"][0]
        r_in = g["r_in"]
        r_mid = 0.5 * (r_in + float(np.median(g["r_out"])))
        du_merge = 10.0 / r_mid  # merge / diverge zones span 10 m of ring arc
        inner_left = g["sense"] > 0  # counter-clockwise ring: the island is on the left

        def ring_tangent(u):
            th = g["sense"] * u
            return g["sense"] * np.array([-math.sin(th), math.cos(th)])

        events = []  # (u, kind, inner point or None, outer point or None, payload)
        for node in [n_ for n_, r_ in ring_nodes.items() if r_ == rci]:  # OSM ring nodes + flares joined to the ring
            for A in approach.get((node, "in"), []):
                for ll in A["lanes"][-1:]:  # the ring is one lane wide: only the rightmost arm lane joins it
                    lb, rb_ = ll.leftBound, ll.rightBound
                    P_in, P_out = (lb[len(lb) - 1], rb_[len(rb_) - 1]) if inner_left else (rb_[len(rb_) - 1], lb[len(lb) - 1])
                    u_out = float(ring_u(g, [[P_out.x, P_out.y]])[0])
                    u_e = u_out + du_merge
                    events.append((u_out, "arm_in", None, P_out, (ll, P_in, P_out, u_e)))
                    events.append((u_e, "merge", None, None, None))
            for B in approach.get((node, "out"), []):
                for ll in B["lanes"][-1:]:  # (other arm lanes reach it by a lane change before the ring)
                    lb, rb_ = ll.leftBound, ll.rightBound
                    Q_in, Q_out = (lb[0], rb_[0]) if inner_left else (rb_[0], lb[0])
                    u_out = float(ring_u(g, [[Q_out.x, Q_out.y]])[0])
                    u_x = u_out - du_merge
                    events.append((u_x, "diverge", None, None, (ll, Q_in, Q_out)))
                    events.append((u_out, "arm_out", None, Q_out, (ll,)))
        if not events:
            continue
        # unwrap the event angles into one increasing cycle starting at the first merge
        base = min(u for u, k, *_ in events if k == "merge")
        ev = sorted(((u - base) % (2 * math.pi), k, i, o, pl) for u, k, i, o, pl in events)
        # extra splits so that no arc exceeds 30 m
        ev2 = []
        for j in range(len(ev)):
            u_a, u_b = ev[j][0], (ev[(j + 1) % len(ev)][0] if j + 1 < len(ev) else 2 * math.pi)
            ev2.append(ev[j])
            n_split = int(math.ceil((u_b - u_a) * r_mid / 30.0))
            for q in range(1, n_split):
                ev2.append((u_a + (u_b - u_a) * q / n_split, "split", None, None, None))
        ev = sorted(ev2, key=lambda z: z[0])
        if os.environ.get("RING_DEBUG"):
            for j_, e_ in enumerate(ev):
                nxt_u = ev[(j_ + 1) % len(ev)][0] + (2 * math.pi if j_ + 1 == len(ev) else 0)
                print(f"  ring event {j_:2d} u {math.degrees(e_[0]):7.2f} deg  {e_[1]:8s} arc to next "
                      f"{(nxt_u - e_[0]) * r_mid:6.2f} m")
        # ring points at every event (the outer point of an arm event is the arm's own boundary point)
        pts_in, pts_out = [], []
        for u_, k_, i_, o_, pl_ in ev:
            xy_i = ring_pt(g, base + u_, r_in)[0]
            pts_in.append(mk_point(xy_i[0], xy_i[1]))
            if o_ is not None:
                pts_out.append(o_)
            else:
                xy_o = ring_pt(g, base + u_, ring_outer_r(g, base + u_))[0]
                pts_out.append(mk_point(xy_o[0], xy_o[1]))

        def arc_ls(u_a, u_b, r_fn, p_a, p_b, step, attrs):
            n_ = max(2, int(math.ceil((u_b - u_a) * r_mid / step)) + 1)
            us = np.linspace(u_a, u_b, n_)[1:-1]
            mid = [mk_point(x, y) for x, y in ring_pt(g, base + us, r_fn(base + us))]
            return mk_ls([p_a] + mid + [p_b], attrs)

        da_o = float(g["az"][1] - g["az"][0])

        def ring_mouth(u):
            """True where the outer kerb was not measured (an arm mouth) at ring angle u."""
            th = math.degrees(g["sense"] * (base + u)) % 360.0
            return bool(g["mouth"][int(round((th - g["az"][0]) / da_o)) % len(g["az"])])

        def kerb_xy(u):
            return ring_pt(g, base + u, ring_outer_r(g, base + u))[0]

        def kerb_fillet(u_a, u_b, p_a, p_b, t_arm, at_end):
            """Outer boundary of a ring arc that touches an arm: the measured kerb up to the mouth, then a smooth
            fillet into the arm's own outer boundary (direction t_arm) instead of the kerb interpolated across
            the mouth. at_end: the arc ends at the arm point p_b, else it starts at the arm point p_a."""
            n_ = max(2, int(math.ceil((u_b - u_a) * r_mid / 1.5)) + 1)
            us = np.linspace(u_a, u_b, n_)
            meas = [not ring_mouth(u) for u in us]
            if at_end:
                m_ = 0
                while m_ + 1 < len(us) - 1 and meas[m_ + 1]:
                    m_ += 1
                kerb = [mk_point(*kerb_xy(u)) for u in us[1:m_ + 1]]
                q0 = np.array([kerb[-1].x, kerb[-1].y]) if kerb else np.array([p_a.x, p_a.y])
                u0 = us[m_] if kerb else u_a
                q1_ = np.array([p_b.x, p_b.y])
                curve = hermite(q0, ring_tangent(base + u0), q1_, t_arm, best_tangents(q0, ring_tangent(base + u0), q1_, t_arm))
                return mk_ls([p_a] + kerb + [mk_point(x, y) for x, y in curve[1:-1]] + [p_b], BOUND_ATTR["road_border"])
            m_ = len(us) - 1
            while m_ - 1 > 0 and meas[m_ - 1]:
                m_ -= 1
            kerb = [mk_point(*kerb_xy(u)) for u in us[m_:-1]] if m_ <= len(us) - 2 else []
            q1 = np.array([kerb[0].x, kerb[0].y]) if kerb else np.array([p_b.x, p_b.y])
            u1 = us[m_] if kerb else u_b
            q0_ = np.array([p_a.x, p_a.y])
            curve = hermite(q0_, t_arm, q1, ring_tangent(base + u1), best_tangents(q0_, t_arm, q1, ring_tangent(base + u1)))
            return mk_ls([p_a] + [mk_point(x, y) for x, y in curve[1:-1]] + kerb + [p_b], BOUND_ATTR["road_border"])

        attrs_road = {"type": "lanelet", "subtype": "road", "location": "urban", "one_way": "yes",
                      "participant:vehicle": "yes", "speed_limit": speed(p0["maxspeed"]), "osm_way": str(p0["way"]),
                      "road_name": p0["name"] or ""}
        n_ev = len(ev)
        outer_ls = {}

        def arc_range(j_from, j_to):
            """Arc indices from event j_from up to (not including) event j_to, cyclically."""
            out, q = [], j_from
            while q != j_to:
                out.append(q)
                q = (q + 1) % n_ev
                if len(out) > n_ev:
                    break
            return out

        def joined_outer(j_from, j_to):
            qs = arc_range(j_from, j_to)
            if len(qs) == 1:
                return outer_ls[qs[0]]  # shared with the ring arc
            pts_ = []
            for q in qs:
                seg = outer_ls[q]
                pts_ += [seg[i] for i in range(len(seg)) if not pts_ or seg[i].id != pts_[-1].id]
            return mk_ls(pts_, BOUND_ATTR["road_border"])
        for j in range(n_ev):
            u_a = ev[j][0]
            u_b = ev[(j + 1) % n_ev][0] if j + 1 < n_ev else 2 * math.pi
            inner = arc_ls(u_a, u_b, lambda u: np.full(np.shape(u), r_in), pts_in[j], pts_in[(j + 1) % n_ev], 1.0,
                           BOUND_ATTR["road_border"])
            k_a, k_b = ev[j][1], ev[(j + 1) % n_ev][1]
            if k_a == "arm_out" and k_b == "arm_in":  # across the mouth there is no kerb: a virtual edge
                outer = arc_ls(u_a, u_b, lambda u: ring_outer_r(g, u), pts_out[j], pts_out[(j + 1) % n_ev], 1.5,
                               BOUND_ATTR["virtual"])
            elif k_b == "arm_out":
                outer = kerb_fillet(u_a, u_b, pts_out[j], pts_out[(j + 1) % n_ev],
                                    lane_dir(ev[(j + 1) % n_ev][4][0], False), True)
            elif k_a == "arm_in":
                outer = kerb_fillet(u_a, u_b, pts_out[j], pts_out[(j + 1) % n_ev], lane_dir(ev[j][4][0], True), False)
            else:
                outer = arc_ls(u_a, u_b, lambda u: ring_outer_r(g, u), pts_out[j], pts_out[(j + 1) % n_ev], 1.5,
                               BOUND_ATTR["road_border"])
            outer_ls[j] = outer
            left, right = (inner, outer) if inner_left else (outer, inner)
            ll = Lanelet(getId(), left, right, AttributeMap(attrs_road))
            lmap.add(ll)
            all_lanelets.append(ll)
            n_ring += 1
        # connectors of the arms
        for j in range(n_ev):
            u_, k_, i_, o_, pl_ = ev[j]
            if k_ == "arm_in":
                ll_arm, P_in, P_out, u_e = pl_
                je = next(q for q in range(n_ev) if ev[q][1] == "merge" and abs(ev[q][0] - ((u_e - base) % (2 * math.pi))) < 1e-9)
                t0 = lane_dir(ll_arm, True)
                P1 = np.array([pts_in[je].x, pts_in[je].y])
                P0_ = np.array([P_in.x, P_in.y])
                t1_ = ring_tangent(base + ev[je][0])
                curve = hermite(P0_, t0, P1, t1_, best_tangents(P0_, t0, P1, t1_))
                inner_b = mk_ls([P_in] + [mk_point(x, y) for x, y in curve[1:-1]] + [pts_in[je]], BOUND_ATTR["virtual"])
                # outer bound: the ring's outer kerb from the arm's corner to the merge point (shared with the arcs)
                outer_b = joined_outer(j, je)
                left, right = (inner_b, outer_b) if inner_left else (outer_b, inner_b)
                attrs = {"type": "lanelet", "subtype": "road", "location": "urban", "one_way": "yes",
                         "participant:vehicle": "yes", "turn_direction": "straight",
                         "speed_limit": ll_arm.attributes["speed_limit"]}
                ll = Lanelet(getId(), left, right, AttributeMap(attrs))
                lmap.add(ll)
                connectors.append(ll)
                conn_by_node[c["nodes"][0]].append((None, None, ll))
            if k_ == "diverge":
                ll_arm, Q_in, Q_out = pl_
                jo = next(q for q in [(j + d_) % n_ev for d_ in range(1, n_ev)] if ev[q][1] == "arm_out")
                t1 = lane_dir(ll_arm, False)
                P0 = np.array([pts_in[j].x, pts_in[j].y])
                P1_ = np.array([Q_in.x, Q_in.y])
                curve = hermite(P0, ring_tangent(base + u_), P1_, t1, best_tangents(P0, ring_tangent(base + u_), P1_, t1))
                inner_b = mk_ls([pts_in[j]] + [mk_point(x, y) for x, y in curve[1:-1]] + [Q_in], BOUND_ATTR["virtual"])
                outer_b = joined_outer(j, jo)
                left, right = (inner_b, outer_b) if inner_left else (outer_b, inner_b)
                attrs = {"type": "lanelet", "subtype": "road", "location": "urban", "one_way": "yes",
                         "participant:vehicle": "yes", "turn_direction": "right" if inner_left else "left",
                         "speed_limit": ll_arm.attributes["speed_limit"]}
                ll = Lanelet(getId(), left, right, AttributeMap(attrs))
                lmap.add(ll)
                connectors.append(ll)
                conn_by_node[c["nodes"][-1]].append((None, None, ll))
    return n_ring


n_ring_ll = build_rings()
print(f"roundabout lanelets: {n_ring_ll}, arm connectors: {len(connectors)}")

for node in sorted({k[0] for k in approach}):
    if node in ring_nodes:
        continue
    ins, outs = approach.get((node, "in"), []), approach.get((node, "out"), [])
    for A in ins:
        for B in outs:
            if A["chain"] == B["chain"] and A["end"] == B["end"]:
                continue  # U-turn onto the same leg
            ang = math.degrees(math.atan2(A["dir"][0] * B["dir"][1] - A["dir"][1] * B["dir"][0], A["dir"] @ B["dir"]))
            if abs(ang) > 150:
                continue
            maneuver = "straight" if abs(ang) < 35 else ("left" if ang > 0 else "right")
            if not allowed(A["way"], node, B["way"], maneuver):
                continue
            la, lb = A["lanes"], B["lanes"]
            turns = A.get("turns")
            if turns and len(turns) == len(la):
                elig = [i for i, t_ in enumerate(turns) if turn_ok(t_, maneuver)]
            else:
                elig = []
            if not elig:
                elig = [0] if maneuver == "left" else ([len(la) - 1] if maneuver == "right" else list(range(len(la))))
            pairs = []
            if maneuver == "left":
                for j, i in enumerate(elig):
                    pairs.append((i, min(j, len(lb) - 1)))
            elif maneuver == "right":
                for j, i in enumerate(reversed(elig)):
                    pairs.append((i, max(len(lb) - 1 - j, 0)))
            else:
                for j, i in enumerate(reversed(elig)):  # align from the right
                    pairs.append((i, max(len(lb) - 1 - j, 0)))
            for i, j in pairs:
                lin, lout = la[i], lb[j]
                t0, t1 = lane_dir(lin, True), lane_dir(lout, False)
                c0_ = 0.5 * np.array([lin.leftBound[len(lin.leftBound) - 1].x + lin.rightBound[len(lin.rightBound) - 1].x,
                                      lin.leftBound[len(lin.leftBound) - 1].y + lin.rightBound[len(lin.rightBound) - 1].y])
                c1_ = 0.5 * np.array([lout.leftBound[0].x + lout.rightBound[0].x, lout.leftBound[0].y + lout.rightBound[0].y])
                frac = best_tangents(c0_, t0, c1_, t1)
                bounds = []
                for side in ("left", "right"):
                    bi = getattr(lin, side + "Bound")
                    bo = getattr(lout, side + "Bound")
                    p0, p1 = bi[len(bi) - 1], bo[0]
                    key = (p0.id, p1.id)
                    if key in vcache:  # same end points as the neighbouring connector: share the linestring
                        bounds.append(vcache[key])
                        continue
                    if (p1.id, p0.id) in vcache:  # the opposite-direction connector next door: share it inverted
                        bounds.append(vcache[(p1.id, p0.id)].invert())
                        continue
                    P0, P1 = np.array([p0.x, p0.y]), np.array([p1.x, p1.y])
                    curve = hermite(P0, t0, P1, t1, frac)
                    z = np.linspace(p0.z, p1.z, len(curve))
                    pts = [p0] + [mk_point(x, y, zz) for (x, y), zz in zip(curve[1:-1], z[1:-1])] + [p1]
                    vcache[key] = mk_ls(pts, BOUND_ATTR["virtual"])
                    bounds.append(vcache[key])
                attrs = {"type": "lanelet", "subtype": "road", "location": "urban", "one_way": "yes",
                         "participant:vehicle": "yes", "turn_direction": maneuver,
                         "speed_limit": lin.attributes["speed_limit"]}
                ll = Lanelet(getId(), bounds[0], bounds[1], AttributeMap(attrs))
                lmap.add(ll)
                connectors.append(ll)
                conn_by_node[node].append((A, B, ll))
print(f"junction connectors: {len(connectors)} at {len(conn_by_node)} junctions; "
      f"cut gaps lengthened at {len(gap_nodes)} junctions (laterally offset straight lanes, hooked or tight turns)")

# ------------------------------------------------------------------ traffic lights (signalised junctions)
sig_nodes = [t for t in net["tagged_nodes"] if t["tags"].get("highway") == "traffic_signals"]
sig_xy = np.array([t["xy"] for t in sig_nodes]) if sig_nodes else np.zeros((0, 2))
node_xy = {int(k): v for k, v in net["node_xy"].items()}
n_tl = 0
for node in sorted({k[0] for k in approach}):
    if not node_signalised(node):
        continue
    for A in approach.get((node, "in"), []):
        la = A["lanes"]

        def back_point(ls, dist=0.5):
            """A new point `dist` metres before the end of linestring ls (inside the lanelet)."""
            n_ = len(ls)
            a_, b_ = ls[n_ - 2], ls[n_ - 1]
            v_ = np.array([a_.x - b_.x, a_.y - b_.y, a_.z - b_.z])
            f_ = min(dist / max(np.hypot(v_[0], v_[1]), 1e-6), 1.0)
            return mk_point(b_.x + v_[0] * f_, b_.y + v_[1] * f_, b_.z + v_[2] * f_)

        pl = back_point(la[0].leftBound)
        pr = back_point(la[-1].rightBound)
        stop = mk_ls([pl] + [back_point(lane.rightBound) for lane in la[:-1]] + [pr], {"type": "stop_line"})
        mid = np.array([(pl.x + pr.x) / 2, (pl.y + pr.y) / 2])
        across = np.array([pr.x - pl.x, pr.y - pl.y])
        across /= max(np.linalg.norm(across), 1e-9)
        zg = float(ground_z(mid)[0])
        lp = [mk_point(*(mid - 0.3 * across), zg + 4.0), mk_point(*(mid + 0.3 * across), zg + 4.0)]
        light = mk_ls(lp, {"type": "traffic_light", "subtype": "red_yellow_green", "height": "1.0",
                           "approximate_position": "yes"})
        re = TrafficLight(getId(), AttributeMap({"type": "regulatory_element", "subtype": "traffic_light"}),
                          [light], stop)
        lmap.add(re)
        for lane in la:
            lane.addRegulatoryElement(re)
        n_tl += 1
print(f"traffic light regulatory elements: {n_tl}")

# ------------------------------------------------------------------ crosswalks from detected zebra markings
def is_zebra(poly):
    """True if the PCD shows zebra stripes across a detected crossing polygon (corners (s0,d0) (s1,d0) (s1,d1)
    (s0,d1)): at least three paint stripes, narrow (median <= 1 m) and dense (>= 0.7 per metre across). Lane
    arrows, kerb stones and bright paving give one or two wide blobs instead."""
    from scipy.ndimage import gaussian_filter1d
    e_al, e_ac = poly[1] - poly[0], poly[3] - poly[0]
    L_al, L_ac = float(np.linalg.norm(e_al)), float(np.linalg.norm(e_ac))
    if L_al < 2.0 or L_ac < 2.0:
        return False
    u_al, u_ac = e_al / L_al, e_ac / L_ac
    idx = np.array(SP.kd.query_ball_point(poly.mean(0), 0.5 * math.hypot(L_al, L_ac) + 1.0), np.int64)
    if len(idx) < 50:
        return False
    P = SP.xy[idx] - poly[0]
    a_, b_ = P @ u_al, P @ u_ac
    m_ = (a_ >= 0) & (a_ <= L_al) & (b_ >= -0.3) & (b_ <= L_ac + 0.3)
    if m_.sum() < 50:
        return False
    inten, b_ = SP.i[idx][m_], b_[m_]
    bg = np.percentile(inten, 30)
    paint = inten > max(bg + 4.5, 1.55 * bg)
    bins = np.arange(-0.3, L_ac + 0.35, 0.1)
    n_all, _ = np.histogram(b_, bins)
    n_p, _ = np.histogram(b_[paint], bins)
    on = gaussian_filter1d(n_p / np.maximum(n_all, 1), 1.0) > 0.3
    edges = np.flatnonzero(np.diff(np.r_[0, on.astype(int), 0]))
    widths = [0.1 * (edges[i + 1] - edges[i]) for i in range(0, len(edges), 2)]
    return len(widths) >= 3 and float(np.median(widths)) <= 1.0 and len(widths) / L_ac >= 0.7


n_cw = n_cw_rejected = 0
for c in chains:
    lo, hi = cut[c["index"]]
    for cw in c["crosswalks"]:
        poly = np.array(cw["poly"])  # (s0,d0) (s1,d0) (s1,d1) (s0,d1)
        if PCD_PATH and not is_zebra(poly):
            n_cw_rejected += 1
            continue
        z = ground_z(poly)
        p00, p10, p11, p01 = [mk_point(x, y, zz) for (x, y), zz in zip(poly, z)]
        left = mk_ls([p00, p01], {"type": "pedestrian_marking"})
        right = mk_ls([p10, p11], {"type": "pedestrian_marking"})
        ll = Lanelet(getId(), left, right, AttributeMap({"type": "lanelet", "subtype": "crosswalk",
                                                         "participant:pedestrian": "yes", "one_way": "no",
                                                         "location": "urban"}))
        # keep only crosswalks that cross mapped road lanelets (at least 30 % of their area: a zebra at a junction
        # mouth lies partly outside the mapped lanes)
        cwp = _Poly(poly)
        covered = 0.0
        for rl in all_lanelets + connectors:
            covered += cwp.buffer(0).intersection(_Poly([[q.x, q.y] for q in rl.polygon2d()]).buffer(0)).area
        if covered < 0.3 * cwp.area:
            continue
        lmap.add(ll)
        n_cw += 1
print(f"crosswalk lanelets: {n_cw} (detections without zebra stripes in the PCD dropped: {n_cw_rejected})")

# ------------------------------------------------------------------ elevations of all remaining vertices from the PCD
if PCD_PATH:
    light_pts = {q.id for ls in lmap.lineStringLayer if "type" in ls.attributes and
                 ls.attributes["type"] == "traffic_light" for q in ls}
    pts_all = [q for q in lmap.pointLayer if q.id not in light_pts and q.id not in allp]
    xy_all = np.array([[q.x, q.y] for q in pts_all])
    mask2 = np.zeros(ras.ground.shape, bool)
    ij = np.round((xy_all - [ras.x0, ras.y0]) / ras.res).astype(int)
    for di in range(-4, 5):
        for dj in range(-4, 5):
            ii, jj = np.clip(ij[:, 1] + di, 0, mask2.shape[0] - 1), np.clip(ij[:, 0] + dj, 0, mask2.shape[1] - 1)
            mask2[ii, jj] = True
    sp2 = SurfacePoints(PCD_XYZI, ras.ground, ras.x0, ras.y0, ras.res, mask2)
    zz2 = surface_z(sp2, xy_all)
    for q, zv in zip(pts_all, zz2):
        if np.isfinite(zv):
            q.z = float(zv)
        q.attributes["local_x"] = f"{q.x:.3f}"
        q.attributes["local_y"] = f"{q.y:.3f}"
    print(f"elevations from PCD for {np.isfinite(zz2).sum()}/{len(zz2)} connector / stop line / crosswalk vertices")

# ------------------------------------------------------------------ elevation smoothing along every linestring
for ls in lmap.lineStringLayer:
    if ("type" in ls.attributes and ls.attributes["type"] == "traffic_light") or len(ls) < 3:
        continue
    zs = np.array([q.z for q in ls])
    xy = np.array([[q.x, q.y] for q in ls])
    ds = np.maximum(np.linalg.norm(np.diff(xy, axis=0), axis=1), 0.05)
    zm = zs.copy()
    for i in range(1, len(zs) - 1):
        zm[i] = np.median(zs[i - 1:i + 2])
    for i in range(1, len(zm)):  # at most 8 % grade between consecutive vertices
        zm[i] = float(np.clip(zm[i], zm[i - 1] - 0.08 * ds[i - 1], zm[i - 1] + 0.08 * ds[i - 1]))
    for q, zv in zip(ls, zm):
        q.z = float(zv)

# crosswalk <-> road lanelet relations (for the crosswalk regulatory elements written below)
cw_roads = []
for cw in lmap.laneletLayer:
    if cw.attributes["subtype"] != "crosswalk":
        continue
    cwp_ = _Poly([[q.x, q.y] for q in cw.polygon2d()]).buffer(0)
    roads = []
    for ll in lmap.laneletLayer:
        if ll.attributes["subtype"] != "road":
            continue
        rp_ = _Poly([[q.x, q.y] for q in ll.polygon2d()]).buffer(0)
        if cwp_.intersection(rp_).area >= 0.5:  # touching corners do not make a lanelet cross the crosswalk
            roads.append(ll.id)
    cw_roads.append((cw.id, roads))

# ------------------------------------------------------------------ drop degenerate road lanelets
# A road lanelet whose bounds are both shorter than 1 m (e.g. a short OSM roundabout-arm link that lies almost
# entirely inside the measured ring, because OSM draws roundabouts smaller than they are) is dropped; the
# stitching below then joins its neighbours across the gap.


def _ls_len(ls):
    return sum(math.hypot(b.x - a.x, b.y - a.y) for a, b in zip(list(ls)[:-1], list(ls)[1:]))


_bad = {ll.id for ll in lmap.laneletLayer if ll.attributes["subtype"] == "road"
        and max(_ls_len(ll.leftBound), _ls_len(ll.rightBound)) < 1.0}
if _bad:
    _keep = LaneletMap()
    for ll in lmap.laneletLayer:
        if ll.id not in _bad:
            _keep.add(ll)
    for re_ in lmap.regulatoryElementLayer:
        _keep.add(re_)
    lmap = _keep
    cw_roads = [(cw_, [r for r in roads_ if r not in _bad]) for cw_, roads_ in cw_roads]
    wedge_excl = [i for i in wedge_excl if i not in _bad]
print(f"degenerate road lanelets dropped: {len(_bad)}")

# ------------------------------------------------------------------ stitch broken successor links
# Where a road lanelet ends without a successor and another road lanelet of the same direction starts within
# STITCH_MAX m ahead without a predecessor (lane-count changes, a path-modelled piece next to a paint-fitted one),
# the later lanelet's boundaries are made to start on the earlier one's end points (shared points = successor).
STITCH_MAX = 2.0  # m


def _end_state(ll, at_end):
    L, R_ = ll.leftBound, ll.rightBound
    i = -1 if at_end else 0
    pl, pr = np.array([L[i].x, L[i].y]), np.array([R_[i].x, R_[i].y])
    c = 0.5 * (pl + pr)
    j = -2 if at_end else 1
    q = 0.5 * (np.array([L[j].x, L[j].y]) + np.array([R_[j].x, R_[j].y]))
    h = (c - q) if at_end else (q - c)
    return pl, pr, c, h / max(np.linalg.norm(h), 1e-9)


road_lls = [ll for ll in lmap.laneletLayer if ll.attributes["subtype"] == "road"] if True else []
starts_by_pts = defaultdict(list)
for ll in road_lls:
    starts_by_pts[(ll.leftBound[0].id, ll.rightBound[0].id)].append(ll)
ends_by_pts = set((ll.leftBound[-1].id, ll.rightBound[-1].id) for ll in road_lls)
start_states = {ll.id: _end_state(ll, False) for ll in road_lls}
n_stitch, gaps_ = 0, []
for ll in road_lls:
    if (ll.leftBound[-1].id, ll.rightBound[-1].id) in starts_by_pts:
        continue  # has a successor
    el, er, ec, eh = _end_state(ll, True)
    best = None
    for nl in road_lls:
        if nl.id == ll.id or (nl.leftBound[0].id, nl.rightBound[0].id) in ends_by_pts:
            continue  # nl already has a predecessor
        sl, sr, sc_, sh = start_states[nl.id]
        if np.linalg.norm(sc_ - ec) > STITCH_MAX + 2.0 or float(eh @ sh) < np.cos(np.radians(35)):
            continue
        gap = max(np.linalg.norm(sl - el), np.linalg.norm(sr - er))
        ahead = float((sc_ - ec) @ eh)
        if gap <= STITCH_MAX and ahead > -1.0 and (best is None or gap < best[0]):
            best = (gap, nl)
    if best is None:
        continue
    gap, nl = best
    old_key = (nl.leftBound[0].id, nl.rightBound[0].id)
    nl.leftBound[0] = ll.leftBound[-1]
    nl.rightBound[0] = ll.rightBound[-1]
    starts_by_pts[(nl.leftBound[0].id, nl.rightBound[0].id)].append(nl)
    ends_by_pts.add((ll.leftBound[-1].id, ll.rightBound[-1].id))
    start_states[nl.id] = _end_state(nl, False)
    n_stitch += 1
    gaps_.append(gap)
print(f"stitched successor links: {n_stitch}" + (f" (gaps median {np.median(gaps_):.2f} m, max {max(gaps_):.2f} m)"
                                                 if gaps_ else ""))

# ------------------------------------------------------------------ connectors for driven manoeuvres
# The vehicle performed every manoeuvre of the drive, so the driven sequence of lanelets must be routable. Where a
# driven transition a -> b has no route (the OSM-based junction logic found no connector, or a lane-count change
# left a gap) and b starts ahead of a's end (< 50 m), a connector lanelet joins a's end edge to b's start edge.
from lanelet2.core import BasicPoint2d as _BP, BoundingBox2d as _BB
_rules = lanelet2.traffic_rules.create(lanelet2.traffic_rules.Locations.Germany,
                                       lanelet2.traffic_rules.Participants.Vehicle)
_tp = np.load(TRAJ)["pos"][::100, :2]
_sd = np.r_[0, np.cumsum(np.linalg.norm(np.diff(_tp, axis=0), axis=1))]
_sel = np.searchsorted(_sd, np.arange(0, _sd[-1], 3.0))
_P, _H = _tp[_sel], np.gradient(_tp[_sel], axis=0)


def _driven_ll(p, h):
    best = None
    for ll in lmap.laneletLayer.search(_BB(_BP(p[0] - 1, p[1] - 1), _BP(p[0] + 1, p[1] + 1))):
        if ll.attributes["subtype"] != "road" or not lanelet2.geometry.inside(ll, _BP(*p)):
            continue
        c = np.array([[q.x, q.y] for q in ll.centerline])
        j = int(np.argmin(np.linalg.norm(c - p, axis=1)))
        if float((c[min(j + 1, len(c) - 1)] - c[max(j - 1, 0)]) @ h) > 0:
            d_ = float(np.min(np.linalg.norm(c - p, axis=1)))
            if best is None or d_ < best[0]:
                best = (d_, ll)
    return best[1] if best else None


_seq = []
for p_, h_ in zip(_P, _H):
    ll_ = _driven_ll(p_, h_)
    if ll_ is not None and (not _seq or _seq[-1].id != ll_.id):
        _seq.append(ll_)
n_man = 0
_graph = lanelet2.routing.RoutingGraph(lmap, _rules)


def _edge(ll, at_end):
    L_, R_ = ll.leftBound, ll.rightBound
    i, j = (-1, -2) if at_end else (0, 1)
    e = np.array([[L_[i].x, L_[i].y], [R_[i].x, R_[i].y]])
    q = 0.5 * (np.array([L_[j].x, L_[j].y]) + np.array([R_[j].x, R_[j].y]))
    c = e.mean(axis=0)
    t = (c - q) if at_end else (q - c)
    return e, c, t / max(np.linalg.norm(t), 1e-9)


for ia in range(len(_seq) - 1):
    a_ = _seq[ia]
    if _graph.getRoute(a_, _seq[ia + 1]) is not None:
        continue
    ea, ca, ta = _edge(a_, True)
    for b_ in _seq[ia + 1:ia + 7]:  # the first later lanelet that lies ahead of a's end (lane change + turn)
        if _graph.getRoute(a_, b_) is not None:
            break  # reachable after all: the unroutable step was a brief detour into a neighbour lane
        sb, cb, tb = _edge(b_, False)
        gap = np.linalg.norm(cb - ca)
        if gap > 50.0 or float((cb - ca) @ ta) < 1.0:
            continue
        bounds_ = []
        for k_, (pa_, pb_) in enumerate(zip(ea, sb)):
            cur_ = hermite(pa_, ta, pb_, tb, best_tangents(pa_, ta, pb_, tb))
            start_pt = a_.leftBound[-1] if k_ == 0 else a_.rightBound[-1]
            end_pt = b_.leftBound[0] if k_ == 0 else b_.rightBound[0]
            bounds_.append(mk_ls([start_pt] + [mk_point(x, y) for x, y in cur_[1:-1]] + [end_pt], BOUND_ATTR["virtual"]))
        ang_ = math.degrees(math.atan2(ta[0] * tb[1] - ta[1] * tb[0], ta @ tb))
        attrs_ = {"type": "lanelet", "subtype": "road", "location": "urban", "one_way": "yes",
                  "participant:vehicle": "yes",
                  "speed_limit": a_.attributes["speed_limit"] if "speed_limit" in a_.attributes else "50",
                  "turn_direction": "straight" if abs(ang_) < 35 else ("left" if ang_ > 0 else "right"),
                  "mcap2map:driven_manoeuvre": "yes"}
        lmap.add(Lanelet(getId(), bounds_[0], bounds_[1], AttributeMap(attrs_)))
        n_man += 1
        _graph = lanelet2.routing.RoutingGraph(lmap, _rules)
        print(f"  driven-manoeuvre connector {a_.id} -> {b_.id} ({attrs_['turn_direction']}, {gap:.1f} m) at "
              f"({ca[0]:.0f},{ca[1]:.0f})")
        break
print(f"connectors added for driven manoeuvres without a route: {n_man} (driven lanelet sequence {len(_seq)})")

# ------------------------------------------------------------------ write
proj = UtmProjector(Origin(LAT0, LON0))
lanelet2.io.write(OUT, lmap, proj)

# ------------------------------------------------------------------ OSM post-processing (things the python API cannot express)
import xml.etree.ElementTree as ET
tree = ET.parse(OUT)
root = tree.getroot()
next_id = max(int(e.get("id")) for e in root) + 1
nodes = {int(e.get("id")): e for e in root.findall("node")}
ways = {int(e.get("id")): e for e in root.findall("way")}
rels = {int(e.get("id")): e for e in root.findall("relation")}


def tagval(e, k):
    t = e.find(f"tag[@k='{k}']")
    return t.get("v") if t is not None else None


def add_tag(e, k, v):
    ET.SubElement(e, "tag", k=k, v=str(v))


# crosswalk regulatory elements: subtype crosswalk, refers = the crosswalk lanelet, referenced by crossing road lanelets
for cw_id, roads in cw_roads:
    rel = ET.SubElement(root, "relation", id=str(next_id), visible="true", version="1")
    ET.SubElement(rel, "member", type="relation", ref=str(cw_id), role="refers")
    add_tag(rel, "type", "regulatory_element")
    add_tag(rel, "subtype", "crosswalk")
    for rid in roads:
        ET.SubElement(rels[rid], "member", type="relation", ref=str(next_id), role="regulatory_element")
    next_id += 1

# light bulbs for every traffic light (placeholder positions along the light bar, like the light itself)
for wid, w in list(ways.items()):
    if tagval(w, "type") != "traffic_light":
        continue
    nd = [int(n.get("ref")) for n in w.findall("nd")]
    a, b = nodes[nd[0]], nodes[nd[-1]]
    bulbs = ET.SubElement(root, "way", id=str(next_id), visible="true", version="1")
    bid = next_id
    next_id += 1
    for f, color in ((0.25, "red"), (0.5, "yellow"), (0.75, "green")):
        n = ET.SubElement(root, "node", id=str(next_id), visible="true", version="1",
                          lat=f"{float(a.get('lat')) * (1 - f) + float(b.get('lat')) * f:.11f}",
                          lon=f"{float(a.get('lon')) * (1 - f) + float(b.get('lon')) * f:.11f}")
        for k in ("ele", "local_x", "local_y"):
            add_tag(n, k, f"{float(tagval(a, k)) * (1 - f) + float(tagval(b, k)) * f:.3f}")
        add_tag(n, "color", color)
        ET.SubElement(bulbs, "nd", ref=str(next_id))
        next_id += 1
    add_tag(bulbs, "type", "light_bulbs")
    add_tag(bulbs, "traffic_light_id", str(wid))
    for rel in rels.values():
        if tagval(rel, "subtype") == "traffic_light" and any(
                m.get("role") == "refers" and int(m.get("ref")) == wid for m in rel.findall("member")):
            ET.SubElement(rel, "member", type="way", ref=str(bid), role="light_bulbs")
ET.indent(tree, space="  ")
tree.write(OUT, encoding="UTF-8", xml_declaration=True)

# editor copy with MGRS-square local coordinates: the frame origin taken modulo the 100 km grid square
# (UTM 33N E 714600 / N 5137000 -> square 33T YM, offsets 14600 / 37000)
for n in root.findall("node"):
    for k, off in (("local_x", OX % 100000.0), ("local_y", OY % 100000.0)):
        t = n.find(f"tag[@k='{k}']")
        if t is not None:
            t.set("v", f"{float(t.get('v')) + off:.3f}")
tree.write(os.path.splitext(OUT)[0] + "_mgrs_local.osm", encoding="UTF-8", xml_declaration=True)
# exclusion list for the Autoware validator: a crosswalk lying fully inside one road lanelet is reported as an
# unshared border by mapping.lane.border_sharing (a false positive of its adjacency heuristic)
excl = []
for cw_id, roads in cw_roads:
    cwp = _Poly([[q.x, q.y] for q in lmap.laneletLayer[cw_id].polygon2d()])
    for rid in roads:
        rp = _Poly([[q.x, q.y] for q in lmap.laneletLayer[rid].polygon2d()])
        if rp.is_valid and cwp.intersection(rp).area > 0.9 * cwp.area:
            excl.append({"primitive": "lanelet", "id": rid, "validators": [{"name": "mapping.lane.border_sharing"}]})
for lid in sorted(set(wedge_excl)):
    excl.append({"primitive": "lanelet", "id": lid, "validators": [{"name": "mapping.lane.border_sharing"}]})
json.dump({"exclusion": excl}, open(os.path.splitext(OUT)[0] + "_validator_exclusion.json", "w"), indent=2)
print(f"post-processed: {len(cw_roads)} crosswalk regulatory elements, light bulbs added, MGRS-local editor copy written, "
      f"{len(excl)} validator exclusions")
print(f"wrote {OUT}: {len(lmap.laneletLayer)} lanelets, {len(lmap.lineStringLayer)} line strings, "
      f"{len(lmap.pointLayer)} points, {len(lmap.regulatoryElementLayer)} regulatory elements")
json.dump({"origin_lat": LAT0, "origin_lon": LON0, "cut": {str(k): v for k, v in cut.items()}},
          open(os.path.splitext(OUT)[0] + "_build_info.json", "w"), indent=1)
json.dump(dbg_bounds, open(os.path.splitext(OUT)[0] + "_debug_bounds.json", "w"))
