#!/usr/bin/env python3
"""Map-match the trajectory to OSM and build the route road network (chains of ways between junctions).

usage: osm_route.py OSM_JSON TRAJ_NPZ OUT_JSON
"""
import json
import sys
from collections import defaultdict

import os

import numpy as np
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import FRAME, OX, OY, UTM_EPSG

OSM, TRAJ, OUT = sys.argv[1:4]
DRIVABLE = {"motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
            "motorway_link", "trunk_link", "primary_link", "secondary_link", "tertiary_link",
            "living_street", "service", "road"}

data = json.load(open(OSM))
to_utm = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
nodes = {}
for e in data["elements"]:
    if e["type"] == "node":
        if e["id"] in nodes and not e.get("tags"):
            continue  # keep the tagged copy (Overpass returns skeleton duplicates)
        nodes[e["id"]] = e
nid = np.array(list(nodes))
E, N = to_utm.transform(np.array([nodes[i]["lon"] for i in nid]), np.array([nodes[i]["lat"] for i in nid]))
nxy = {int(i): (float(x - OX), float(y - OY)) for i, x, y in zip(nid, E, N)}
ways = {e["id"]: e for e in data["elements"] if e["type"] == "way" and e.get("tags", {}).get("highway") in DRIVABLE
        and all(n in nxy for n in e["nodes"])}


def oneway(w):
    t = w["tags"]
    if t.get("oneway") in ("yes", "1", "true") or t.get("junction") in ("roundabout", "circular"):
        return 1
    if t.get("oneway") == "-1":
        return -1
    return 0


# --- segments
seg_way, seg_i, seg_a, seg_b = [], [], [], []
for wid, w in ways.items():
    for i in range(len(w["nodes"]) - 1):
        seg_way.append(wid); seg_i.append(i)
        seg_a.append(nxy[w["nodes"][i]]); seg_b.append(nxy[w["nodes"][i + 1]])
seg_way, seg_i = np.array(seg_way), np.array(seg_i)
seg_a, seg_b = np.array(seg_a), np.array(seg_b)
seg_d = seg_b - seg_a
seg_len = np.maximum(np.linalg.norm(seg_d, axis=1), 1e-6)
seg_u = seg_d / seg_len[:, None]

# --- trajectory samples every 1 m of travel
z = np.load(TRAJ)
t_all = z["t0"] + np.arange(len(z["pos"])) * z["dt"]
p_all = z["pos"][::20, :2]
t_s = t_all[::20]
step = np.r_[0, np.cumsum(np.linalg.norm(np.diff(p_all, axis=0), axis=1))]
sel = np.searchsorted(step, np.arange(0, step[-1], 1.0))
P, T = p_all[sel], t_s[sel]
H = np.gradient(P, axis=0)
H /= np.maximum(np.linalg.norm(H, axis=1), 1e-9)[:, None]
print(f"{len(P)} samples along {step[-1]:.0f} m")

# --- candidates per sample
cands = []
for k in range(len(P)):
    ap = P[k] - seg_a
    tt = np.clip(np.sum(ap * seg_u, axis=1), 0, seg_len)
    d = np.linalg.norm(P[k] - (seg_a + seg_u * tt[:, None]), axis=1)
    m = np.flatnonzero(d < 12)
    c = []
    for j in m:
        cosang = float(H[k] @ seg_u[j])
        ow = oneway(ways[seg_way[j]])
        for direc in (1, -1):
            if (ow == 1 and direc == -1) or (ow == -1 and direc == 1):
                continue
            ca = cosang * direc
            if ca < np.cos(np.radians(50)):
                continue
            t_ = ways[seg_way[j]]["tags"]
            try:
                nl = int(t_.get("lanes", "2" if not ow else "1"))
            except ValueError:
                nl = 2
            half = 1.75 * nl
            pen = 2.0 if t_.get("highway") == "service" else 0.0
            c.append((int(seg_way[j]), int(seg_i[j]), direc, max(0.0, d[j] - half) + 8 * (1 - ca) + pen, float(tt[j])))
    cands.append(c)


def connected(w1, w2):
    return w1 == w2 or bool(set(ways[w1]["nodes"]) & set(ways[w2]["nodes"]))


# --- Viterbi
INF = 1e18
cost = [np.array([c[3] for c in cands[0]])]
back = [None]
for k in range(1, len(P)):
    prev, cur = cands[k - 1], cands[k]
    if not cur:
        cands[k] = cur = prev
    if not prev:
        cost.append(np.array([c[3] for c in cur])); back.append(np.zeros(len(cur), int)); continue
    pc = cost[-1]
    nc, bk = np.empty(len(cur)), np.empty(len(cur), int)
    for j, (w, si, di, e, _) in enumerate(cur):
        tr_c = np.array([0.0 if (pw == w and pd == di) else (10.0 if connected(pw, w) else 1e6)
                         for (pw, psi, pd, pe, _) in prev])
        tot = pc + tr_c
        bk[j] = int(np.argmin(tot))
        nc[j] = tot[bk[j]] + e
    cost.append(nc); back.append(bk)
path = [int(np.argmin(cost[-1]))]
for k in range(len(P) - 1, 0, -1):
    path.append(int(back[k][path[-1]]))
path = path[::-1]
match = [cands[k][path[k]] for k in range(len(P))]

# --- runs of (way, dir)
runs = []
for k, (w, si, di, e, tt) in enumerate(match):
    if runs and runs[-1]["way"] == w and runs[-1]["dir"] == di:
        runs[-1]["k1"] = k
    else:
        runs.append({"way": w, "dir": di, "k0": k, "k1": k})
runs = [r for r in runs if r["k1"] - r["k0"] >= 2 or ways[r["way"]]["tags"].get("junction") == "roundabout"]
merged = []
for r in runs:
    if merged and merged[-1]["way"] == r["way"] and merged[-1]["dir"] == r["dir"]:
        merged[-1]["k1"] = r["k1"]
    else:
        merged.append(r)
for r in merged:
    r["t0"], r["t1"] = float(T[r["k0"]]), float(T[r["k1"]])
    w = ways[r["way"]]
    print(f"  t {r['t0']-T[0]:6.1f}-{r['t1']-T[0]:6.1f}  way {r['way']:>11} dir {r['dir']:+d}  "
          f"{w['tags'].get('highway')} {w['tags'].get('name', '')} lanes={w['tags'].get('lanes', '?')}")
route_ways = sorted({r["way"] for r in merged})
# a traversed roundabout is mapped completely: add every junction=roundabout way connected to a traversed one
ring_tag = lambda w: ways[w]["tags"].get("junction") in ("roundabout", "circular")
_rset = {w for w in route_ways if ring_tag(w)}
_grow = True
while _grow:
    _grow = False
    _rnodes = {n for w in _rset for n in ways[w]["nodes"]}
    for wid, w in ways.items():
        if wid not in _rset and ring_tag(wid) and set(w["nodes"]) & _rnodes:
            _rset.add(wid)
            _grow = True
route_ways = sorted(set(route_ways) | _rset)
print(f"roundabout ways mapped: {sorted(_rset)}")

# --- route graph: node -> incident route ways
inc = defaultdict(set)
for wid in route_ways:
    for n in ways[wid]["nodes"]:
        inc[n].add(wid)
full_deg = defaultdict(int)
for wid, w in ways.items():
    ns = w["nodes"]
    for i, n in enumerate(ns):
        full_deg[n] += 1 if i in (0, len(ns) - 1) else 2


def heading_at(wid, n, outward):
    """Unit direction of way wid leaving node n (outward=True) along the way."""
    ns = ways[wid]["nodes"]
    i = ns.index(n)
    j = i + 1 if i + 1 < len(ns) else i - 1
    if not outward:
        pass
    a, b = np.array(nxy[ns[i]]), np.array(nxy[ns[j]])
    v = b - a
    return v / max(np.linalg.norm(v), 1e-9)


# split route ways at nodes shared with other route ways (so every junction is a way end)
pieces = []  # (way_id, node list)
for wid in route_ways:
    ns = ways[wid]["nodes"]
    cut = [0] + [i for i in range(1, len(ns) - 1) if len(inc[ns[i]]) >= 2] + [len(ns) - 1]
    for a, b in zip(cut[:-1], cut[1:]):
        pieces.append((wid, ns[a:b + 1]))

# keep only pieces actually traversed (some part within 6 m of the matched samples)
trav = defaultdict(list)
for k, (w, si, di, e, tt) in enumerate(match):
    trav[w].append(P[k])
kept = []
for wid, ns in pieces:
    pts = np.array([nxy[n] for n in ns])
    tp = np.array(trav[wid]) if trav[wid] else np.zeros((0, 2))
    if ring_tag(wid):
        kept.append((wid, ns))  # roundabouts are mapped completely
        continue
    if not len(tp):
        continue
    dense = np.concatenate([a + (b - a) * np.linspace(0, 1, max(2, int(np.linalg.norm(b - a)) + 1))[:, None]
                            for a, b in zip(pts[:-1], pts[1:])])
    dd = np.min(np.linalg.norm(dense[:, None, :] - tp[None, :, :], axis=2), axis=1)
    if ring_tag(wid) or (dd < 10).sum() >= 15 or (dd < 10).mean() > 0.5:
        kept.append((wid, ns))
pieces = kept

# piece-level node degree
pdeg = defaultdict(list)
for pi, (wid, ns) in enumerate(pieces):
    pdeg[ns[0]].append((pi, 0))
    pdeg[ns[-1]].append((pi, 1))


def turn_angle(pa, pb, n):
    """Angle between piece pa arriving at n and piece pb leaving n (degrees, 0 = straight)."""
    wa, na = pieces[pa]
    wb, nb = pieces[pb]
    ia = na.index(n); ib = nb.index(n)
    a_prev = np.array(nxy[na[ia - 1 if ia > 0 else 1]])
    b_next = np.array(nxy[nb[ib + 1 if ib + 1 < len(nb) else ib - 1]])
    v1 = np.array(nxy[n]) - a_prev
    v2 = b_next - np.array(nxy[n])
    c = v1 @ v2 / max(np.linalg.norm(v1) * np.linalg.norm(v2), 1e-9)
    return float(np.degrees(np.arccos(np.clip(c, -1, 1))))


def is_ring(pi):
    return ways[pieces[pi][0]]["tags"].get("junction") in ("roundabout", "circular")


junction_nodes = set()
for n, lst in pdeg.items():
    if len(lst) >= 3 or len(lst) == 1:
        junction_nodes.add(n)
    elif len(lst) == 2:
        (p1, _), (p2, _) = lst
        if is_ring(p1) != is_ring(p2) or turn_angle(p1, p2, n) > 40 or \
                (full_deg[n] >= 3 and turn_angle(p1, p2, n) > 25) or oneway(ways[pieces[p1][0]]) != oneway(ways[pieces[p2][0]]):
            junction_nodes.add(n)

# --- chains: concatenate pieces through non-junction degree-2 nodes
used = set()
chains = []
for pi in range(len(pieces)):
    if pi in used:
        continue
    chain = [(pi, 1)]
    used.add(pi)
    for forward in (True, False):
        while True:
            last_pi, last_dir = chain[-1] if forward else chain[0]
            ns = pieces[last_pi][1] if last_dir == 1 else pieces[last_pi][1][::-1]
            n = ns[-1] if forward else ns[0]
            if n in junction_nodes:
                break
            nxt = [(q, e) for q, e in pdeg[n] if q != last_pi and q not in used]
            if len(nxt) != 1:
                break
            q, e = nxt[0]
            used.add(q)
            if forward:
                chain.append((q, 1 if e == 0 else -1))
            else:
                chain.insert(0, (q, 1 if e == 1 else -1))
    node_seq = []
    way_seq = []
    for q, d in chain:
        ns = pieces[q][1] if d == 1 else pieces[q][1][::-1]
        start = len(node_seq) - 1 if node_seq else 0
        node_seq += ns if not node_seq else ns[1:]
        way_seq.append({"way": pieces[q][0], "dir": d, "i0": start, "i1": len(node_seq) - 1})
    chains.append({"nodes": [int(n) for n in node_seq], "ways": way_seq,
                   "xy": [nxy[n] for n in node_seq], "ring": is_ring(chain[0][0])})

# one closed chain per roundabout: its arcs (split at the arm junctions) concatenated head to tail
_ring_ch = [c for c in chains if c["ring"]]
_other = [c for c in chains if not c["ring"]]
_merged_rings = []
while _ring_ch:
    grp = [_ring_ch.pop(0)]
    changed = True
    while changed:
        changed = False
        for c in list(_ring_ch):
            if any(c["nodes"][0] == g["nodes"][-1] or c["nodes"][-1] == g["nodes"][0] for g in grp):
                grp.append(c)
                _ring_ch.remove(c)
                changed = True
    # order head to tail
    seq = [grp.pop(0)]
    while grp:
        nxt_ = [c for c in grp if c["nodes"][0] == seq[-1]["nodes"][-1]]
        if not nxt_:
            break
        seq.append(nxt_[0])
        grp.remove(nxt_[0])
    nodes_, ways_, xy_ = [], [], []
    for c in seq:
        off_ = len(nodes_) - 1 if nodes_ else 0
        nodes_ += c["nodes"] if not nodes_ else c["nodes"][1:]
        xy_ += c["xy"] if not xy_ else c["xy"][1:]
        ways_ += [dict(w, i0=w["i0"] + off_, i1=w["i1"] + off_) for w in c["ways"]]
    _merged_rings.append({"nodes": nodes_, "ways": ways_, "xy": xy_, "ring": True})
    print(f"roundabout: {len(seq)} arcs merged into one ring chain ({len(nodes_)} nodes, closed: "
          f"{nodes_[0] == nodes_[-1]})" + (f", {len(grp)} arcs left over" if grp else ""))
    _other += [c for c in grp]  # arcs that did not join (should not happen) stay separate
chains = _other + _merged_rings

for ci, c in enumerate(chains):
    names = sorted({ways[w["way"]]["tags"].get("name", "?") for w in c["ways"]})
    L = np.sum(np.linalg.norm(np.diff(np.array(c["xy"]), axis=0), axis=1))
    print(f"chain {ci}: {len(c['ways'])} ways, {L:.0f} m, ring={c['ring']}, {names}, "
          f"ends {c['nodes'][0]}{'*' if c['nodes'][0] in junction_nodes else ''} -> "
          f"{c['nodes'][-1]}{'*' if c['nodes'][-1] in junction_nodes else ''}")

restr = []
for e in data["elements"]:
    if e["type"] == "relation" and e.get("tags", {}).get("type") == "restriction":
        m = {r["role"]: (r["type"], r["ref"]) for r in e["members"]}
        if "from" in m and "to" in m and "via" in m:
            restr.append({"id": e["id"], "restriction": e["tags"].get("restriction"), "from": m["from"][1],
                          "via": m["via"][1], "via_type": m["via"][0], "to": m["to"][1]})
tagged = [{"id": int(i), "xy": nxy[int(i)], "tags": nodes[int(i)]["tags"]} for i in nid
          if nodes[int(i)].get("tags") and nodes[int(i)]["tags"].get("highway") in
          ("traffic_signals", "crossing", "stop", "give_way")]
crossing_ways = [{"id": w["id"], "tags": w["tags"], "xy": [nxy[n] for n in w["nodes"] if n in nxy]}
                 for w in data["elements"] if w["type"] == "way" and w.get("tags", {}).get("footway") == "crossing"]

json.dump({"origin_utm33n": [OX, OY], "frame": FRAME, "ways": {str(w): {"tags": ways[w]["tags"], "nodes": ways[w]["nodes"]} for w in route_ways},
           "node_xy": {str(n): nxy[n] for c in chains for n in c["nodes"]},
           "runs": merged, "chains": chains, "junctions": sorted(int(n) for n in junction_nodes if n in pdeg),
           "restrictions": restr, "tagged_nodes": tagged, "crossing_ways": crossing_ways,
           "all_drivable": {str(w): {"tags": ways[w]["tags"], "xy": [nxy[n] for n in ways[w]["nodes"]]} for w in ways}},
          open(OUT, "w"))
print(f"route ways {len(route_ways)}, pieces {len(pieces)}, chains {len(chains)}, junctions {len(junction_nodes)}, "
      f"restrictions {len(restr)}, tagged nodes {len(tagged)}, crossing ways {len(crossing_ways)}")
