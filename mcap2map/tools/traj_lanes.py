#!/usr/bin/env python3
"""Re-model road pieces from the driven path (post-processes lane_extract.py's lanes.json; also used by
build_lanelet2.py after the PCD paint fit).

On streets without lane paint (residential streets with parking on both sides, the bus terminal's service roads)
the raster lane model wanders between kerbs, parking strips and gutters, and the paint fit can lock onto parking or
bus-stop hatching. There the driven path is the best lane evidence: the vehicle kept to its lane.
A re-modelled piece:
  * the driven lane is centred on the smoothed driven path (robust penalised spline with +-15 m of context, so
    consecutive pieces stay continuous), width W_LANE
  * the piece's other lanes are stacked next to it with the same width (opposite direction to its left)
  * the boundary between lanes is virtual (no paint), outer edges are road_border
and carries traj_modeled=True (build_lanelet2.py then keeps it: no paint / kerb snapping).

As a script: re-models every piece whose boundaries are all unpainted (bound_types road_border).
usage: traj_lanes.py LANES_JSON TRAJ_NPZ OUT_JSON
"""
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lanegeom import RefLine
from pcd_refine import pspline

W_LANE = {"residential": 3.5, "service": 3.5, "living_street": 3.25, "unclassified": 3.5}
W_DEFAULT = 3.5


class DrivenPath:
    """Moving trajectory samples (20 Hz) with headings."""

    def __init__(self, pos_xy):
        step = np.r_[0, np.cumsum(np.linalg.norm(np.diff(pos_xy, axis=0), axis=1))]
        keep = np.r_[True, np.diff(step) > 0.02]
        self.xy = pos_xy[keep]
        hd = np.gradient(self.xy, axis=0)
        self.hd = hd / np.maximum(np.linalg.norm(hd, axis=1), 1e-9)[:, None]

    def on_chain(self, ref):
        """(s, d, along-cosine, near mask) of the samples in the chain's frame."""
        ps, pd = ref.project(self.xy)
        t_ref = np.column_stack([np.interp(ps, ref.s, ref.t[:, 0]), np.interp(ps, ref.s, ref.t[:, 1])])
        along = np.sum(self.hd * t_ref, axis=1)
        near = (np.abs(pd) < 9.0) & (ps > -2) & (ps < ref.length + 2) & (np.abs(along) > 0.9)
        return ps, pd, along, near


def driven_lane(p, ps, pd, along, near):
    """(lane index, direction, samples mask, coverage, median offset from that lane's centre, inside fraction) of
    the driven lane of a piece, or None if the vehicle did not drive along it."""
    st = np.array(p["stations"])
    bd = np.array(p["bounds_d"])
    m = near & (ps >= st[0] - 1) & (ps <= st[-1] + 1)
    if m.sum() < 10:
        return None
    ddir = 1 if np.median(along[m]) > 0 else -1
    mm = m & (np.sign(along) == ddir)
    cov = len(np.unique(np.floor(ps[mm]))) / max(st[-1] - st[0], 1.0)
    own = [k for k, ln in enumerate(p["lanes"]) if ln["dir"] == ddir]
    if not own or mm.sum() < 10:
        return None
    best = None
    for k in own:
        lo = np.interp(ps[mm], st, np.minimum(bd[k], bd[k + 1]))
        hi = np.interp(ps[mm], st, np.maximum(bd[k], bd[k + 1]))
        inside = float(np.mean((pd[mm] >= lo) & (pd[mm] <= hi)))
        off = float(np.median(np.abs(pd[mm] - 0.5 * (lo + hi))))
        if best is None or (inside, -off) > (best[5], -best[4]):
            best = (k, ddir, mm, cov, off, inside)
    return best


def remodel(p, ref, ps, pd, along, near, info=None):
    """Re-model piece p from the driven path in place; returns a log string or None if not possible."""
    info = info or driven_lane(p, ps, pd, along, near)
    if info is None:
        return None
    kd, ddir, mm, cov, off, inside = info
    if cov < 0.6:
        return None
    st = np.array(p["stations"])
    bd = np.array(p["bounds_d"])
    nb = len(bd)
    ctx = near & (np.sign(along) == ddir) & (ps >= st[0] - 15) & (ps <= st[-1] + 15)
    o = np.argsort(ps[ctx])
    xs, ys = ps[ctx][o], pd[ctx][o]
    if len(xs) < 10:
        return None
    spl, _ = pspline(xs, ys, np.ones(len(xs)), min(xs[0], st[0]), max(xs[-1], st[-1]), 8.0, lam=20.0, inlier=0.5)
    if spl is None:
        return None
    dc = spl(np.clip(st, xs[0], xs[-1]))
    w = W_LANE.get(p.get("highway"), W_DEFAULT)
    new = np.zeros_like(bd)
    new[kd] = dc - w / 2
    new[kd + 1] = dc + w / 2
    for k in range(kd - 1, -1, -1):
        new[k] = new[k + 1] - w
    for k in range(kd + 2, nb):
        new[k] = new[k - 1] + w
    old_c = np.median(0.5 * (bd[kd] + bd[kd + 1]) - dc)
    p["bounds_d"] = new.tolist()
    p["bounds_xy"] = [ref.at(st, new[k]).tolist() for k in range(nb)]
    types = ["road_border"] * nb
    for k in range(1, nb - 1):  # unpainted inner lines: same direction -> lane changes allowed, else virtual
        same = p["lanes"][k - 1]["dir"] == p["lanes"][k]["dir"]
        types[k] = "virtual_lc" if same else "virtual"
    p["bound_types"] = types
    p["driven_lane"] = int(kd)
    p["traj_modeled"] = True
    return (f"piece {p['s0']:6.1f}-{p['s1']:6.1f} m {p.get('name') or p.get('highway')}: lanes {nb - 1}, driven lane "
            f"{kd} (dir {ddir:+d}), coverage {cov:.2f}, path inside its lane {inside*100:.0f}%, off centre {off:.2f} m, "
            f"driven offset {dc.min():+.2f}..{dc.max():+.2f} m, old lane centre shift {old_c:+.2f} m, width {w} m")


if __name__ == "__main__":
    LANES, TRAJ, OUT = sys.argv[1:4]
    lanes = json.load(open(LANES))
    dp = DrivenPath(np.load(TRAJ)["pos"][::50, :2])
    n_mod = 0
    for c in lanes["chains"]:
        if c.get("ring"):
            continue
        ref = RefLine(c["ref_xy"])
        ps, pd, along, near = dp.on_chain(ref)
        for p in c["pieces"]:
            if any(b != "road_border" for b in p["bound_types"]):
                continue  # painted somewhere: the PCD paint fit handles it (build_lanelet2 falls back if needed)
            log = remodel(p, ref, ps, pd, along, near)
            if log:
                n_mod += 1
                print(f"chain {c['index']:2d} {log}")
    json.dump(lanes, open(OUT, "w"))
    print(f"{n_mod} pieces re-modelled from the driven path")
