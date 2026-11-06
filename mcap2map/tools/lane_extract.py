#!/usr/bin/env python3
"""Fit OSM lane structure to lane markings and curbs detected in the road-surface rasters.

For every chain of OSM ways: straighten the reflectivity / surface-height rasters along the (smoothed) OSM
centre line, detect longitudinal marking tracks, transverse markings (stop lines, zebra crossings) and
curbs, then fit per OSM way the lane boundaries
    b(s) = c(s) +- cumulative lane widths
with the lane counts / turn lanes from OSM, the centre c(s) from the centre marking (or curbs / driven
passes) and widths from the marking spacing. Boundaries are finally snapped to nearby marking tracks.

usage: lane_extract.py ROUTE_JSON RASTER_NPZ TRAJ_NPZ OUT_JSON DEBUG_DIR
"""
import json
import os
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from scipy.ndimage import gaussian_filter1d

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from lanegeom import Raster, RefLine, smooth_polyline

ROUTE, RAS, TRAJ, OUT, DBG = sys.argv[1:6]
os.makedirs(DBG, exist_ok=True)
DS, DD, DMAX = 0.2, 0.1, 18.0
W_DEF, W_MIN, W_MAX = 3.25, 2.6, 4.6
TAPER = 15.0

net = json.load(open(ROUTE))
ras = Raster(RAS)
zr = np.load(RAS)
zfine = zr["zfine"]
tz = np.load(TRAJ)
tp = tz["pos"][::50]  # 20 Hz
tp_xy = tp[:, :2]
step = np.r_[0, np.cumsum(np.linalg.norm(np.diff(tp_xy, axis=0), axis=1))]
mv = np.r_[True, np.diff(step) > 0.05]
tp_xy = tp_xy[mv]
th = np.gradient(tp_xy, axis=0)
th /= np.maximum(np.linalg.norm(th, axis=1), 1e-9)[:, None]


def osm_lanes(tags):
    """(n_forward, n_backward, turn_fwd list, turn_bwd list) of an OSM way."""
    ow = tags.get("oneway") in ("yes", "1", "true") or tags.get("junction") in ("roundabout", "circular")
    rev = tags.get("oneway") == "-1"

    def num(k):
        try:
            return int(tags[k])
        except (KeyError, ValueError):
            return None
    n = num("lanes")
    nf, nb = num("lanes:forward"), num("lanes:backward")
    if ow or rev:
        nf, nb = (n or 1), 0
    else:
        if nf is None and nb is None:
            nf = nb = max(1, (n or 2) // 2)
            if n and n % 2:
                nf = n - nb
        elif nf is None:
            nf = max(1, (n or nb + 1) - nb)
        elif nb is None:
            nb = max(1, (n or nf + 1) - nf)
    tf = tags.get("turn:lanes:forward", tags.get("turn:lanes") if (ow or rev) else None)
    tb = tags.get("turn:lanes:backward")
    tf = tf.split("|") if tf else None
    tb = tb.split("|") if tb else None
    if rev:
        return 0, nf, None, tf
    return nf, nb, tf, tb


def peaks_1d(p, d, thr, min_sep=1.4):
    out = []
    for i in range(1, len(p) - 1):
        if p[i] < thr or p[i] < p[i - 1] or p[i] < p[i + 1]:
            continue
        lo, hi = max(0, i - int(min_sep / DD / 2)), min(len(p), i + int(min_sep / DD / 2) + 1)
        if p[i] < p[lo:hi].max():
            continue
        w = slice(max(0, i - 2), i + 3)
        out.append((float(np.sum(d[w] * p[w]) / np.sum(p[w])), float(p[i])))
    return out


def run_lengths(mask):
    """(start, end) index pairs of True runs."""
    m = np.r_[False, mask, False].astype(np.int8)
    st = np.flatnonzero(np.diff(m) == 1)
    en = np.flatnonzero(np.diff(m) == -1)
    return list(zip(st, en))


result = {"chains": []}
for ci, ch in enumerate(net["chains"]):
    ring = ch["ring"]
    ref = RefLine(smooth_polyline(ch["xy"], 4.0 if not ring else 2.0))
    s, d, I = ras.straighten(ref, DMAX, DS, DD)
    _, _, Z = ras.straighten(ref, DMAX, DS, DD, arr=zfine)
    L = ref.length
    nS, nD = I.shape

    # --- passes of the trajectory along this chain (d and direction per s)
    ps, pd = ref.project(tp_xy)
    near = (ps > 0) & (ps < L) & (np.abs(pd) < 12)
    tang = np.column_stack([np.interp(ps, ref.s, ref.t[:, 0]), np.interp(ps, ref.s, ref.t[:, 1])])
    cosang = np.sum(tang * th, axis=1)
    pass_s, pass_d, pass_dir = ps[near & (np.abs(cosang) > 0.8)], pd[near & (np.abs(cosang) > 0.8)], \
        np.sign(cosang[near & (np.abs(cosang) > 0.8)])

    # --- marking evidence
    I0 = np.nan_to_num(I, nan=0.0)
    B = ndimage.median_filter(I0, size=(1, 31))
    M = (I0 - B > 7) & (I0 > 13)
    road = np.isfinite(Z)
    frac = (M & road).sum(axis=1) / np.maximum(road.sum(axis=1), 1)
    trans_rows = ndimage.binary_opening(frac > 0.22, iterations=1)
    crosswalks, stoplines = [], []
    for a, b in run_lengths(trans_rows):
        ln = (b - a) * DS
        sm = (a + b) / 2 * DS
        cols = np.flatnonzero(M[a:b].mean(axis=0) > 0.3)
        if not len(cols):
            continue
        if 2.0 <= ln <= 8.0:
            crosswalks.append({"s0": a * DS, "s1": b * DS, "d0": float(d[cols.min()]), "d1": float(d[cols.max()])})
        elif ln < 1.2:
            stoplines.append({"s": sm, "d0": float(d[cols.min()]), "d1": float(d[cols.max()])})
    Ml = M & ~ndimage.binary_dilation(trans_rows, iterations=3)[:, None]

    # --- longitudinal marking peaks per window, then tracks
    Mp = ndimage.maximum_filter(Ml.astype(np.float32), size=(1, 3))
    win = int(4.0 / DS)
    obs = []  # (s, d, strength)
    for i0 in range(0, nS, int(1.0 / DS)):
        a, b = max(0, i0 - win), min(nS, i0 + win)
        valid = road[a:b].mean(axis=0) > 0.3
        p = Mp[a:b].mean(axis=0) * valid
        for dp, st in peaks_1d(p, d, 0.14):
            obs.append((s[i0], dp, st))
    obs = np.array(obs) if obs else np.zeros((0, 3))
    tracks = []
    for so in np.unique(obs[:, 0]) if len(obs) else []:
        cur = obs[obs[:, 0] == so]
        used = set()
        for tk in tracks:
            if so - tk[-1][0] > 6.0:
                continue
            j = np.argmin(np.abs(cur[:, 1] - tk[-1][1])) if len(cur) else None
            if j is not None and j not in used and abs(cur[j, 1] - tk[-1][1]) < 0.3:
                tk.append(tuple(cur[j]))
                used.add(j)
        for j in range(len(cur)):
            if j not in used:
                tracks.append([tuple(cur[j])])
    tracks = [np.array(t) for t in tracks if t[-1][0] - t[0][0] >= 8.0]

    # --- curbs per window (surface height step >= 6 cm), searched outward from the road interior
    curbL, curbR = np.full(nS, np.nan), np.full(nS, np.nan)
    interior = np.interp(s, pass_s[np.argsort(pass_s)], pass_d[np.argsort(pass_s)]) if len(pass_s) > 5 else np.zeros(nS)
    for i0 in range(0, nS, int(1.0 / DS)):
        a, b = max(0, i0 - win), min(nS, i0 + win)
        zp = np.nanmedian(Z[a:b], axis=0)
        if np.isnan(zp).all():
            continue
        j0 = int(np.clip(np.round((interior[i0] + DMAX) / DD), 0, nD - 1))
        for sgn, arr in ((1, curbL), (-1, curbR)):
            j = j0
            base = zp[j0] if np.isfinite(zp[j0]) else np.nanmedian(zp)
            while 0 <= j + sgn * 5 < nD:
                z1 = zp[j + sgn * 5]
                if np.isfinite(z1) and z1 - base > 0.06:
                    arr[i0] = d[j]
                    break
                if np.isfinite(zp[j]):
                    base = 0.7 * base + 0.3 * zp[j]
                j += sgn
    for arr in (curbL, curbR):
        ok = np.isfinite(arr)
        if ok.sum() > 3:
            arr[:] = ndimage.median_filter(np.interp(s, s[ok], arr[ok]), size=int(10 / DS) | 1)

    # --- centre line c(s): solid track near the middle (two-way), else curb middle, else passes
    ways = ch["ways"]
    s_nodes = ref.project(np.array(ch["xy"]))[0]
    pieces = []
    for w in ways:
        tags = net["ways"][str(w["way"])]["tags"]
        nf, nb, tf, tb = osm_lanes(tags)
        if w["dir"] == 1:
            nR, nL, tR, tL = nf, nb, tf, tb
        else:
            nR, nL, tR, tL = nb, nf, tb, tf
        s0 = 0.0 if w is ways[0] else float(s_nodes[w["i0"]])
        s1 = L if w is ways[-1] else float(s_nodes[w["i1"]])
        mx = tags.get("maxspeed", "50")
        pieces.append({"way": w["way"], "dir": w["dir"], "s0": s0, "s1": s1, "nR": nR, "nL": nL,
                       "turnR": tR, "turnL": tL, "maxspeed": mx, "name": tags.get("name", ""),
                       "highway": tags.get("highway")})
    two_way = any(p["nL"] > 0 and p["nR"] > 0 for p in pieces)

    mid = (curbL + curbR) / 2
    c = np.full(nS, np.nan)
    def pass_line(sign):
        m = pass_dir == sign
        if m.sum() < 5:
            return None
        o = np.argsort(pass_s[m])
        ps_, pd_ = pass_s[m][o], pass_d[m][o]
        line = np.interp(s, ps_, pd_)
        gap = np.min(np.abs(s[:, None] - ps_[None, ::max(1, len(ps_) // 400)]), axis=1)
        line[gap > 5] = np.nan
        return line
    fwd_line, bwd_line = pass_line(1), pass_line(-1)
    if two_way and tracks:
        # centre marking: a long, mostly solid track; prefer one separating the forward (right) and backward
        # (left) driven passes, otherwise the one closest to the middle between the curbs
        cands = []
        for t in tracks:
            if np.median(t[:, 2]) < 0.45 or t[-1, 0] - t[0, 0] < 15:
                continue
            seg = (s >= t[0, 0]) & (s <= t[-1, 0])
            td = np.interp(s[seg], t[:, 0], t[:, 1])
            score = 0.0
            if fwd_line is not None and bwd_line is not None:
                f, b = fwd_line[seg], bwd_line[seg]
                both = np.isfinite(f) & np.isfinite(b)
                if both.any():
                    sep = (td[both] > f[both] + 0.8) & (td[both] < b[both] - 0.8)
                    score -= 10.0 * sep.mean() * both.mean()
            elif fwd_line is not None or bwd_line is not None:
                ln = fwd_line if fwd_line is not None else bwd_line
                sg = 1 if fwd_line is not None else -1
                v = np.isfinite(ln[seg])
                if v.any():
                    dist = (td[v] - ln[seg][v]) * sg
                    nside = max(max(p["nR"], p["nL"]) for p in pieces)
                    ok_ = np.zeros(len(dist), bool)
                    for k in range(nside):
                        ok_ |= (dist > 1.2 + k * W_DEF) & (dist < 2.8 + k * W_DEF)
                    score -= 10.0 * ok_.mean() * v.mean()
                    score += 5.0 * (dist < 0.8).mean() * v.mean()
            ref_mid = np.nanmedian(np.interp(t[:, 0], s, np.nan_to_num(mid, nan=0.0)))
            score += 0.3 * abs(np.median(t[:, 1]) - ref_mid) - 0.02 * (t[-1, 0] - t[0, 0])
            cands.append((score, t))
        cands.sort(key=lambda z: z[0])
        if os.environ.get("LANE_DEBUG"):
            print(f"  chain {ci}: fwd pass {fwd_line is not None}, bwd pass {bwd_line is not None}; centre candidates:",
                  [(round(sc_, 2), round(float(t[0, 0]), 1), round(float(t[-1, 0]), 1), round(float(np.median(t[:, 1])), 2))
                   for sc_, t in cands[:6]])
        best_d = np.median(cands[0][1][:, 1]) if cands else 0.0
        for score, t in cands:
            if score > 1.0:
                break
            if abs(np.median(t[:, 1]) - best_d) > 1.0:
                continue
            seg = (s >= t[0, 0]) & (s <= t[-1, 0])
            fill = seg & np.isnan(c)
            td = np.interp(s[fill], t[:, 0], t[:, 1])
            # double centre line: average with a parallel solid track 0.5-1.6 m away
            for sc2, t2 in cands:
                if t2 is t:
                    continue
                ov = (s[fill] >= t2[0, 0]) & (s[fill] <= t2[-1, 0])
                if ov.sum() * DS < 10:
                    continue
                gap = np.interp(s[fill][ov], t2[:, 0], t2[:, 1]) - td[ov]
                if 0.5 < abs(np.median(gap)) < 1.6 and np.std(gap) < 0.25:
                    td[ov] += gap / 2
                    break
            c[fill] = td
    if np.isnan(c).all():
        if two_way and fwd_line is not None and bwd_line is not None:
            c = (fwd_line + bwd_line) / 2
        elif np.isfinite(mid).any():
            c = mid.copy()
        elif two_way and (fwd_line is not None or bwd_line is not None):
            ln = fwd_line if fwd_line is not None else bwd_line
            ok_ = np.isfinite(ln)
            c = np.interp(s, s[ok_], ln[ok_]) + (W_DEF / 2 if fwd_line is not None else -W_DEF / 2)
        elif len(pass_s):
            c = interior + (W_DEF / 2 if two_way else 0.0)
        else:
            c = np.zeros(nS)
    ok = np.isfinite(c)
    c = np.interp(s, s[ok], c[ok])
    c = gaussian_filter1d(ndimage.median_filter(c, size=int(6 / DS) | 1), 2.0 / DS)
    if not two_way:
        # oneway: c is the middle of the carriageway (curbs / passes), lanes placed symmetrically
        cm = mid if np.isfinite(mid).sum() > nS / 3 else interior
        okm = np.isfinite(cm)
        c = gaussian_filter1d(np.interp(s, s[okm], cm[okm]), 3.0 / DS) if okm.any() else c

    # --- per-piece lane widths from track offsets relative to c
    def side_offsets(p, sgn):
        """Observed |offsets| of marking tracks on one side within the piece."""
        vals = []
        for t in tracks:
            m = (t[:, 0] >= p["s0"]) & (t[:, 0] <= p["s1"])
            if not m.any():
                continue
            o = (t[m, 1] - np.interp(t[m, 0], s, c)) * sgn
            vals.append(o[o > 0.8])
        return np.concatenate(vals) if vals else np.zeros(0)

    for p in pieces:
        for side, sgn, n in (("R", -1, p["nR"]), ("L", 1, p["nL"])):
            if n == 0:
                p["w" + side] = []
                continue
            off = side_offsets(p, sgn)
            if not two_way:
                n_tot = p["nR"] + p["nL"]
            widths, prev = [], 0.0
            for k in range(1, n + 1):
                exp = prev + W_DEF
                m = off[(off > prev + W_MIN - 0.2) & (off < prev + W_MAX + 0.2)]
                if len(m) >= 5:
                    hist, edges = np.histogram(m, bins=np.arange(prev + W_MIN - 0.2, prev + W_MAX + 0.3, 0.1))
                    pk = edges[np.argmax(hist)] + 0.05
                    near_pk = m[np.abs(m - pk) < 0.25]
                    wk = float(np.median(near_pk)) - prev
                else:
                    # outermost lane: use the curb when no marking, keep within limits
                    cb = curbR if sgn < 0 else curbL
                    seg = (s >= p["s0"]) & (s <= p["s1"])
                    cv = (cb[seg] - c[seg]) * sgn
                    cv = cv[np.isfinite(cv)]
                    remaining = n - k + 1
                    wk = float(np.clip((np.median(cv) - prev - 0.2) / remaining, W_MIN, W_MAX)) if len(cv) else W_DEF
                wk = float(np.clip(wk, W_MIN, W_MAX))
                widths.append(wk)
                prev += wk
            p["w" + side] = widths
        if not two_way:
            # centred oneway carriageway: shift so that lanes are symmetric about c
            total = sum(p["wR"]) + sum(p["wL"])
            p["oneway_shift"] = total / 2 - sum(p["wL"])
        else:
            p["oneway_shift"] = 0.0

    # --- lane lineage across piece boundaries (added / dropped lanes from turn:lanes)
    def adds(prev_n, n, turns, inner_first):
        """Number of lanes added at the inner side when going from prev_n to n lanes."""
        if n <= prev_n:
            return 0
        k = n - prev_n
        if turns:
            lefts = sum(1 for t in turns if "left" in t and "through" not in t)
            return min(k, lefts) if inner_first else 0
        return 0

    # sample lane widths along s for each piece (stations every 0.5 m), with tapers for added/dropped lanes
    for pi, p in enumerate(pieces):
        ss = np.arange(p["s0"], p["s1"] + 1e-6, 0.5)
        if len(ss) < 2:
            ss = np.array([p["s0"], p["s1"]])
        p["_ss"] = ss
        cc = np.interp(ss, s, c)
        prev_p = pieces[pi - 1] if pi > 0 else None
        next_p = pieces[pi + 1] if pi + 1 < len(pieces) else None
        p["_W"], p["_new"] = {}, {}
        for side, sgn in (("R", -1), ("L", 1)):
            widths = np.array(p["w" + side])
            n = len(widths)
            Wk = np.tile(widths[:, None], (1, len(ss))) if n else np.zeros((0, len(ss)))
            new_inner = np.zeros(n, bool)
            new_outer = np.zeros(n, bool)
            for nb_p, at_start in ((prev_p, True), (next_p, False)):
                if nb_p is None or n == 0:
                    continue
                n_nb = len(nb_p["w" + side])
                if n > n_nb:
                    k_in = adds(n_nb, n, p["turn" + side], inner_first=True)
                    k_out = n - n_nb - k_in
                    # place the taper where the model boundaries agree best with the marking tracks
                    best = (-1, 0.0)
                    for shift in np.arange(0.0, max(0.0, min(40.0, p["s1"] - p["s0"] - TAPER)) + 1e-6, 2.0):
                        ramp = np.clip((ss - p["s0"] - shift) / TAPER, 0, 1) if at_start else \
                            np.clip((p["s1"] - shift - ss) / TAPER, 0, 1)
                        Wt = Wk.copy()
                        for k in list(range(k_in)) + list(range(n - k_out, n)):
                            Wt[k] *= ramp
                        bd_t = cc[None, :] + sgn * np.cumsum(Wt, axis=0)
                        score = 0
                        for t in tracks:
                            m = (ss >= t[0, 0]) & (ss <= t[-1, 0])
                            if m.any():
                                td = np.interp(ss[m], t[:, 0], t[:, 1])
                                score += int((np.min(np.abs(bd_t[:, m] - td[None, :]), axis=0) < 0.3).sum())
                        if score > best[0]:
                            best = (score, shift)
                    ramp = np.clip((ss - p["s0"] - best[1]) / TAPER, 0, 1) if at_start else \
                        np.clip((p["s1"] - best[1] - ss) / TAPER, 0, 1)
                    for k in range(k_in):
                        Wk[k] *= ramp
                        new_inner[k] = True
                    for k in range(n - k_out, n):
                        Wk[k] *= ramp
                        new_outer[k] = True
            p["_W"][side] = Wk
            p["_new"][side] = (new_inner, new_outer)

    # continuity: widths of lanes continuing from the previous piece blend from its end widths
    for pi in range(1, len(pieces)):
        p, q = pieces[pi], pieces[pi - 1]
        blend = np.clip((p["_ss"] - p["s0"]) / TAPER, 0, 1)
        for side in ("R", "L"):
            Wp, Wq = p["_W"][side], q["_W"][side]
            if not len(Wp) or not len(Wq):
                continue
            ni, no = p["_new"][side]
            cont = [k for k in range(len(Wp)) if not ni[k] and not no[k]]
            wq_end = [w for w in Wq[:, -1] if w > 0.05]
            if len(cont) == len(wq_end):
                for k, w0 in zip(cont, wq_end):
                    Wp[k] = Wp[k] + (w0 - Wp[k, 0]) * (1 - blend)

    out_pieces = []
    for pi, p in enumerate(pieces):
        ss = p["_ss"]
        sides = {}
        for side in ("R", "L"):
            Wk = p["_W"][side]
            cum = np.vstack([np.zeros(len(ss)), np.cumsum(Wk, axis=0)]) if len(Wk) else np.zeros((1, len(ss)))
            sides[side] = (cum, p["_new"][side][0].tolist(), p["_new"][side][1].tolist())
        cc = np.interp(ss, s, c)
        if not two_way:  # centred oneway carriageway
            cc = cc + sides["R"][0][-1] / 2 - sides["L"][0][-1] / 2
        # boundaries from rightmost to leftmost: d = cc - cumR[::-1] ... cc ... cc + cumL
        cumR, cumL = sides["R"][0], sides["L"][0]
        bnds = [cc - cumR[k] for k in range(cumR.shape[0] - 1, 0, -1)] + [cc] + [cc + cumL[k] for k in range(1, cumL.shape[0])]
        bnds = np.array(bnds)
        # snap to nearby marking tracks (outside taper zones), then smooth
        types = []
        for bi in range(len(bnds)):
            bd = bnds[bi].copy()
            snapped = np.zeros(len(ss), bool)
            duty = np.full(len(ss), np.nan)
            for t in tracks:
                m = (ss >= t[0, 0]) & (ss <= t[-1, 0])
                if not m.any():
                    continue
                td = np.interp(ss[m], t[:, 0], t[:, 1])
                close = np.abs(td - bd[m]) < 0.6
                idx = np.flatnonzero(m)[close]
                bd[idx] = td[close]
                snapped[idx] = True
                duty[idx] = np.interp(ss[idx], t[:, 0], t[:, 2])
            if snapped.any() and (~snapped).any():
                delta = np.where(snapped, bd - bnds[bi], np.nan)
                okd = np.isfinite(delta)
                delta_f = np.interp(ss, ss[okd], delta[okd])
                fade = np.exp(-ndimage.distance_transform_edt(~snapped) * 0.5 / 5.0)
                bd = bnds[bi] + delta_f * fade
            bnds[bi] = gaussian_filter1d(bd, 1.0 / 0.5) if len(bd) > 5 else bd
            frac_s = snapped.mean()
            dmed = np.nanmedian(duty) if np.isfinite(duty).any() else 0.0
            edge = bi == 0 or bi == len(bnds) - 1
            if frac_s > 0.3:
                types.append("solid" if dmed > 0.5 else "dashed")
            else:
                types.append("road_border" if edge else ("solid" if (two_way and bi == cumR.shape[0] - 1) else "dashed"))
        # lanes: index k between boundary k and k+1; direction +1 (along chain) for right-side lanes
        nRl = cumR.shape[0] - 1
        lanes = []
        for k in range(len(bnds) - 1):
            right_side = k < nRl
            side = "R" if right_side else "L"
            kk = (nRl - 1 - k) if right_side else (k - nRl)  # index from centre outward
            ni, no = sides[side][1], sides[side][2]
            lanes.append({"dir": 1 if right_side else -1, "side": side, "k_from_centre": kk,
                          "new_inner": bool(ni[kk]) if kk < len(ni) else False,
                          "new_outer": bool(no[kk]) if kk < len(no) else False})
        xy_b = [ref.at(ss, bd).tolist() for bd in bnds]
        out_pieces.append({"way": p["way"], "dir": p["dir"], "s0": p["s0"], "s1": p["s1"], "name": p["name"],
                           "maxspeed": p["maxspeed"], "highway": p["highway"], "turnR": p["turnR"], "turnL": p["turnL"],
                           "stations": ss.tolist(), "bounds_d": bnds.tolist(), "bounds_xy": xy_b,
                           "bound_types": types, "lanes": lanes, "nR": nRl, "nL": len(bnds) - 1 - nRl})

    # the driven path must lie inside a lane of its direction: otherwise shift the lane structure laterally
    # (per-piece shift, then a smooth profile along the chain so piece transitions stay continuous)
    piece_shift = []
    for op in out_pieces:
        st_ = np.array(op["stations"])
        bd = np.array(op["bounds_d"])
        m = (pass_s > op["s0"] + 3) & (pass_s < op["s1"] - 3)
        offs, inside_ = [], []
        for s_i, d_i, dir_i in zip(pass_s[m], pass_d[m], pass_dir[m]):
            b_i = np.array([np.interp(s_i, st_, bk) for bk in bd])
            cand_c = [(b_i[k] + b_i[k + 1]) / 2 for k, ln in enumerate(op["lanes"]) if ln["dir"] == dir_i]
            if not cand_c:
                continue
            inside_.append(any(min(b_i[k], b_i[k + 1]) <= d_i <= max(b_i[k], b_i[k + 1])
                               for k, ln in enumerate(op["lanes"]) if ln["dir"] == dir_i))
            offs.append(d_i - cand_c[int(np.argmin(np.abs(np.array(cand_c) - d_i)))])
        if len(offs) >= 5 and np.mean(inside_) < 0.7:
            piece_shift.append(float(np.median(offs)))
            print(f"  chain {ci} way {op['way']}: driven path inside its lanes only {np.mean(inside_)*100:.0f}% "
                  f"-> shift {piece_shift[-1]:+.2f} m", flush=True)
        elif len(offs) >= 5:
            piece_shift.append(0.0)
        else:
            piece_shift.append(np.nan)
    if np.any(np.nan_to_num(piece_shift) != 0):
        ps_ = np.array(piece_shift)
        known = np.flatnonzero(np.isfinite(ps_))
        for i in range(len(ps_)):
            if not np.isfinite(ps_[i]):
                ps_[i] = ps_[known[np.argmin(np.abs(known - i))]]
        prof = np.zeros(nS)
        for op, sh in zip(out_pieces, ps_):
            prof[(s >= op["s0"]) & (s <= op["s1"] + DS)] = sh
        prof = gaussian_filter1d(prof, 8.0 / DS)
        for op, sh in zip(out_pieces, ps_):
            st_ = np.array(op["stations"])
            bd = np.array(op["bounds_d"]) + np.interp(st_, s, prof)[None, :]
            op["bounds_d"] = bd.tolist()
            op["bounds_xy"] = [ref.at(st_, bk).tolist() for bk in bd]
            if sh != 0:
                op["bound_types"] = [("road_border" if k in (0, len(bd) - 1) else
                                      ("solid" if (two_way and k == op["nR"]) else "dashed")) for k in range(len(bd))]

    # transverse markings measured inside the fitted carriageway
    crosswalks, stoplines = [], []
    lo = np.full(nS, np.nan)
    hi = np.full(nS, np.nan)
    cen = np.full(nS, np.nan)
    for op in out_pieces:
        bd = np.array(op["bounds_d"])
        m = (s >= op["s0"]) & (s <= op["s1"])
        lo[m] = np.interp(s[m], op["stations"], bd.min(axis=0))
        hi[m] = np.interp(s[m], op["stations"], bd.max(axis=0))
        cen[m] = np.interp(s[m], op["stations"], bd[op["nR"]])
    inroad = (d[None, :] >= lo[:, None]) & (d[None, :] <= hi[:, None])
    road_med = float(np.nanmedian(np.where(inroad, I, np.nan))) if inroad.any() else 7.0
    Mabs = I0 > max(15.0, 2.2 * road_med)
    fr = (Mabs & inroad).sum(axis=1) / np.maximum(inroad.sum(axis=1), 1)
    for a, b in run_lengths(ndimage.binary_closing(fr > 0.3, iterations=2)):
        if 2.0 <= (b - a) * DS <= 12.0:
            crosswalks.append({"s0": a * DS, "s1": b * DS, "d0": float(np.nanmin(lo[a:b])), "d1": float(np.nanmax(hi[a:b]))})
    for sgn in (-1, 1):
        side = inroad & ((d[None, :] - cen[:, None]) * sgn > 0.3)
        fs_ = (Mabs & side).sum(axis=1) / np.maximum(side.sum(axis=1), 1)
        for a, b in run_lengths(fs_ > 0.7):
            sm_ = (a + b) / 2 * DS
            if (b - a) * DS <= 1.2 and side[(a + b) // 2].sum() > 15 and min(sm_, L - sm_) < 40:
                cols = np.flatnonzero(side[(a + b) // 2])
                stoplines.append({"s": (a + b) / 2 * DS, "d0": float(d[cols.min()]), "d1": float(d[cols.max()]),
                                  "side": "R" if sgn < 0 else "L"})
    cw_xy = []
    for cw in crosswalks:
        poly = [ref.at([cw["s0"]], cw["d0"])[0], ref.at([cw["s1"]], cw["d0"])[0],
                ref.at([cw["s1"]], cw["d1"])[0], ref.at([cw["s0"]], cw["d1"])[0]]
        cw_xy.append({"s0": cw["s0"], "s1": cw["s1"], "poly": [list(map(float, q)) for q in poly]})
    sl_xy = [{"s": sl["s"], "d0": sl["d0"], "d1": sl["d1"], "side": sl["side"],
              "line": [list(map(float, ref.at([sl["s"]], sl["d0"])[0])), list(map(float, ref.at([sl["s"]], sl["d1"])[0]))]}
             for sl in stoplines]
    result["chains"].append({"index": ci, "ring": ring, "nodes": [ch["nodes"][0], ch["nodes"][-1]],
                             "ref_xy": ref.at(np.arange(0, L, 0.5)).tolist(), "length": L, "two_way": two_way,
                             "pieces": out_pieces, "crosswalks": cw_xy, "stoplines": sl_xy,
                             "pass_dirs": sorted(set(pass_dir.astype(int).tolist()))})

    # --- debug image
    fig, ax = plt.subplots(figsize=(22, 6))
    ax.imshow(I0.T, origin="lower", cmap="gray", vmin=0, vmax=35, extent=(s[0], s[-1], d[0], d[-1]), aspect="auto")
    for t in tracks:
        ax.plot(t[:, 0], t[:, 1], ".", ms=2, color="yellow")
    ax.plot(s, curbL, "c-", lw=0.8)
    ax.plot(s, curbR, "c-", lw=0.8)
    if len(pass_s):
        ax.plot(pass_s, pass_d, ",", color="lime")
    for op in out_pieces:
        for bd, ty in zip(op["bounds_d"], op["bound_types"]):
            ax.plot(op["stations"], bd, {"solid": "r-", "dashed": "r--", "road_border": "m-"}[ty], lw=1.0)
        ax.axvline(op["s0"], color="orange", lw=0.6)
    for cw in crosswalks:
        ax.axvspan(cw["s0"], cw["s1"], color="blue", alpha=0.25)
    for sl in stoplines:
        ax.plot([sl["s"]] * 2, [sl["d0"], sl["d1"]], "b-", lw=2)
    names = sorted({p["name"] for p in pieces})
    ax.set_title(f"chain {ci} {names} two_way={two_way} lanes " +
                 " | ".join(f"{p['nR']}R/{p['nL']}L" for p in pieces))
    fig.savefig(os.path.join(DBG, f"lanes_chain_{ci:02d}.png"), dpi=60, bbox_inches="tight")
    plt.close(fig)
    print(f"chain {ci:2d} {str(names):40s} L={L:5.0f} m tracks={len(tracks):3d} crosswalks={len(crosswalks)} "
          f"stoplines={len(stoplines)} pieces: " +
          "; ".join(f"{op['nR']}R/{op['nL']}L w={[round(x,2) for x in p['wR']]}/{[round(x,2) for x in p['wL']]}"
                    for op, p in zip(out_pieces, pieces)), flush=True)

json.dump(result, open(OUT, "w"))
