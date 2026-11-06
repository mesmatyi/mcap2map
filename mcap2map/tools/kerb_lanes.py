#!/usr/bin/env python3
"""Lane boundaries of road sections without lane paint, from the kerbs in the point cloud map.

In the road's (s, d) frame the surface height is profiled across the road at every station. Starting from the
driven line (the trajectory, which is on the carriageway), the first upward step of a kerb's size is searched to
the right and to the left. Each kerb is fitted as a smooth distance from the driven line, which is steady where
the kerb runs parallel to the lane even if the road's OSM reference line does not.

usage (diagnostic plots): kerb_lanes.py LANES_JSON RASTER_NPZ TRAJ_NPZ PCD OUT_DIR chain:s0:s1 [chain:s0:s1 ...]
       (s0..s1: the stations used, i.e. the part of the chain between its junction cuts)
"""
import numpy as np

D_LO, D_HI, BIN_W = -18.0, 18.0, 0.1


def cross_profiles(s_p, d_p, z_p, stations, half_along=0.75, pct=15):
    """Low-percentile surface height per lateral bin (BIN_W) for every station: (n_stations, n_bins), NaN where a
    bin holds fewer than 3 points. The points must be sorted by s."""
    nb = int(round((D_HI - D_LO) / BIN_W))
    H = np.full((len(stations), nb), np.nan)
    for i, sc in enumerate(stations):
        a, b = np.searchsorted(s_p, sc - half_along), np.searchsorted(s_p, sc + half_along)
        if b - a < 20:
            continue
        j = np.floor((d_p[a:b] - D_LO) / BIN_W).astype(np.int64)
        ok = (j >= 0) & (j < nb)
        j, z = j[ok], z_p[a:b][ok]
        o = np.lexsort((z, j))
        j, z = j[o], z[o]
        starts = np.flatnonzero(np.r_[True, np.diff(j) > 0])
        cnt = np.r_[starts[1:], len(j)] - starts
        sel = cnt >= 3
        H[i, j[starts[sel]]] = z[starts[sel] + np.floor(cnt[sel] * pct / 100.0).astype(np.int64)]
    return H


def find_kerb(h, j0, direction, win=4, min_step=0.06, max_step=0.45, max_gap=8, max_reach=150):
    """Lateral position d of the first kerb face met when scanning the profile h from bin j0 in `direction`
    (+1 left, -1 right), or None. A kerb: the median height of the `win` bins beyond a bin is min_step..max_step
    above that of the `win` bins before it. The scan gives up at a run of more than max_gap empty bins
    (occluded, e.g. under a parked car) or at a higher step (a wall or a vehicle)."""
    n, gap, j = len(h), 0, j0
    for _ in range(max_reach):
        j += direction
        if j - win < 0 or j + win >= n:
            return None
        if not np.isfinite(h[j]):
            gap += 1
            if gap > max_gap:
                return None
            continue
        gap = 0
        inner = h[j - win:j] if direction > 0 else h[j + 1:j + 1 + win]
        outer = h[j + 1:j + 1 + win] if direction > 0 else h[j - win:j]
        if np.isfinite(inner).sum() < 2 or np.isfinite(outer).sum() < 2:
            continue
        step = float(np.nanmedian(outer) - np.nanmedian(inner))
        if step < min_step:
            continue
        if step > max_step:
            return None
        lvl = float(np.nanmedian(inner)) + 0.5 * step  # the face: where the profile passes half the step
        seq = range(j - win, j + win + 1) if direction > 0 else range(j + win, j - win - 1, -1)
        for q in seq:
            if np.isfinite(h[q]) and h[q] >= lvl:
                return D_LO + (q if direction > 0 else q + 1) * BIN_W
        return D_LO + (j + 0.5) * BIN_W
    return None


def raw_kerbs(H, seed_d):
    """Per station the kerb to the right and to the left of the driven line: d_right, d_left (NaN = none)."""
    dr, dl = np.full(len(H), np.nan), np.full(len(H), np.nan)
    for i in range(len(H)):
        if not np.isfinite(seed_d[i]):
            continue
        j0 = int(np.clip((seed_d[i] - D_LO) / BIN_W, 6, H.shape[1] - 7))
        r, l_ = find_kerb(H[i], j0, -1), find_kerb(H[i], j0, +1)
        if r is not None:
            dr[i] = r
        if l_ is not None:
            dl[i] = l_
    return dr, dl


def smooth_line(st, v, knot=8.0, lam=40.0, inlier=0.25, reach=6.0):
    """Robust smooth line through the values v (NaN = none) at stations st: penalised spline with outlier
    rejection (driveways, parked cars and kerb returns at junctions fall out). Returns (values at st, supported
    flags: an inlier within `reach` metres, number of inliers), or (None, None, 0)."""
    from pcd_refine import pspline
    ok = np.isfinite(v)
    if ok.sum() < 8:
        return None, None, 0
    # first a stiff line: stations where the kerb runs away from the lane (junction mouths, bays) fall out;
    # then the flexible line through what is left
    x, y = st[ok], v[ok]
    spl0, inl0 = pspline(x, y, np.ones(len(x)), st[0], st[-1], 25.0, lam=300.0, iters=6, inlier=0.6)
    if spl0 is None or inl0.sum() < 8:
        return None, None, 0
    x, y = x[inl0], y[inl0]
    spl, inl = pspline(x, y, np.ones(len(x)), st[0], st[-1], knot, lam=lam, iters=6, inlier=inlier)
    if spl is None or inl.sum() < 8:
        return None, None, 0
    si = x[inl]
    sup = np.min(np.abs(st[:, None] - si[None, :]), axis=1) <= reach
    return spl(st), sup, int(inl.sum())


def kerb_bounds(s_p, d_p, z_p, stations, seed_d, lanes, drive_dir, use, _trimmed=False):
    """Lane boundaries of an unpainted road piece from its kerbs.

    stations: the piece's stations; seed_d: the driven line (lateral position per station); lanes: the piece's
    lanes right to left (dicts with 'dir'); drive_dir: +1 / -1 direction the piece was driven in (0 unknown);
    use: flags of the stations between the junction cuts. Returns (result or None, info); result is a dict with
    'bounds' ((n_kept_lanes + 1, n_stations), NaN outside `use`) and 'keep' (indices of the lanes kept).

    * Normal case: the kerb the vehicle drove along (the nearer one) is the reference edge, openings in it (side
      streets, driveways) are bridged. The carriageway is as wide as its narrow sections (15th percentile of the
      kerb-to-kerb width, 2.75-3.75 m per lane), so parking bays and widenings on the far side are left out;
      where the far kerb comes closer than that, it is the edge. The lanes split the width equally.
    * Separated carriageway: on a two-way piece whose kerb-to-kerb strip only fits the lanes of the driven
      direction (a median or an island on the other side), those lanes take the strip from kerb to kerb and the
      lanes of the other direction are left out, since they are not on this strip."""
    from scipy.ndimage import gaussian_filter1d
    n_lanes = len(lanes)
    iu = np.flatnonzero(use)
    info = dict(n=int(len(iu)))
    if len(iu) < 12:
        info["rejected"] = "too short"
        return None, info
    st = stations[iu]
    seed = seed_d[iu]
    H = cross_profiles(s_p, d_p, z_p, st)
    j_seed = np.clip(((seed - D_LO) / BIN_W).astype(int), 0, H.shape[1] - 1)
    valid = np.array([np.isfinite(H[i, max(j - 25, 0):j + 26]).sum() >= 15 for i, j in enumerate(j_seed)])
    info["point_cloud_cover"] = float(valid.mean())
    if valid.sum() < 12 or valid.mean() < 0.6:
        info["rejected"] = "the point cloud does not cover most of the piece"
        return None, info
    dr, dl = raw_kerbs(H, seed)
    off = {"R": seed - dr, "L": dl - seed}  # distance of each kerb from the driven line
    fit = {}
    for side in ("R", "L"):
        v = off[side].copy()
        v[~valid] = np.nan
        v[v < 0.9] = np.nan  # closer than half a vehicle width to the driven line: not a kerb
        line, sup, n_inl = smooth_line(st, v)
        fit[side] = dict(line=line, sup=sup, cov=n_inl / max(int(valid.sum()), 1),
                         med=float(np.nanmedian(v)) if np.isfinite(v).any() else np.inf)
    info.update(cov_right=round(fit["R"]["cov"], 2), cov_left=round(fit["L"]["cov"], 2))

    def bridged(side):
        f = fit[side]
        return np.interp(st, st[f["sup"]], f["line"][f["sup"]])

    two_way = any(ln["dir"] == 1 for ln in lanes) and any(ln["dir"] == -1 for ln in lanes)
    driven = [i for i, ln in enumerate(lanes) if ln["dir"] == drive_dir] if drive_dir else list(range(n_lanes))
    have = {sd: fit[sd]["line"] is not None and fit[sd]["cov"] >= 0.35 for sd in ("R", "L")}
    # kerb returns: towards a junction the kerb the vehicle drove along curves away into the corner. The piece is
    # used only up to where that starts (the kerb more than 0.8 m further from the driven line than typical).
    if not _trimmed:
        cands0 = [sd for sd in ("R", "L") if have[sd]]
        if cands0:
            sd0 = min(cands0, key=lambda sd: fit[sd]["med"])
            dist = bridged(sd0)
            lim = float(np.median(dist[fit[sd0]["sup"]])) + 0.8
            a, b = 0, len(st) - 1
            while a < b - 12 and dist[a] > lim:
                a += 1
            while b > a + 12 and dist[b] > lim:
                b -= 1
            if a > 0 or b < len(st) - 1:
                use2 = use & (stations >= st[a]) & (stations <= st[b])
                res, info2 = kerb_bounds(s_p, d_p, z_p, stations, seed_d, lanes, drive_dir, use2, _trimmed=True)
                info2["trimmed_to"] = (float(st[a]), float(st[b]))
                return res, info2
    step = max(float(np.median(np.diff(st))), 0.1)
    keep = list(range(n_lanes))
    w_med = None
    if have["R"] and have["L"]:
        both = fit["R"]["sup"] & fit["L"]["sup"]
        if both.sum() >= 10:
            w_med = float(np.median((fit["R"]["line"] + fit["L"]["line"])[both]))
            info["kerb_to_kerb_median"] = round(w_med, 2)
    # a raised strip (island, planted median without a sharp kerb) where the other direction's lanes would be
    sep_width = None
    if two_way and drive_dir and driven:
        side_ref = "L" if drive_dir == -1 else "R"  # the kerb on the driven direction's own side
        if have[side_ref]:
            sgn = 1.0 if side_ref == "L" else -1.0
            kd = seed + sgn * bridged(side_ref)
            nd = len(driven)
            x = np.arange(2.0, 3.75 * n_lanes + 0.6, BIN_W)  # distance from that kerb towards the other side
            band = (x >= 3.75 * nd + 0.3) & (x <= 3.75 * nd + 2.8)
            edges, raised = [], []
            for i in np.flatnonzero(valid):
                b0 = H[i, max(j_seed[i] - 5, 0):j_seed[i] + 6]
                if not np.isfinite(b0).any():
                    continue
                jj = np.round((kd[i] - sgn * x - D_LO) / BIN_W).astype(int)
                okj = (jj >= 0) & (jj < H.shape[1])
                hh = np.full(len(x), np.nan)
                hh[okj] = H[i, jj[okj]] - np.nanmedian(b0)
                if np.isfinite(hh[band]).sum() >= 8:
                    raised.append(np.nanmedian(hh[band]) >= 0.08)
                e = np.nan
                for q in range(len(x) - 5):  # where the surface is, and stays, 5 cm above the driven lane
                    w5 = hh[q:q + 5]
                    if np.isfinite(w5).sum() >= 3 and np.nanmedian(w5) >= 0.05:
                        e = x[q]
                        break
                edges.append(e)
            if len(raised) >= 10 and np.mean(raised) >= 0.6 and np.isfinite(edges).sum() >= 10:
                sep_width = float(np.clip(np.nanmedian(edges), 2.75 * nd, 3.75 * nd))
                info["raised_far_strip"] = round(float(np.mean(raised)), 2)
    check_sides = []
    if sep_width is not None:
        if side_ref == "L":
            L = kd
            R = L - sep_width
        else:
            R = kd
            L = R + sep_width
        keep = driven
        check_sides = [side_ref]
        info.update(case="separated carriageway (raised strip beyond): lanes of the other direction left out",
                    width=round(sep_width, 2))
    elif two_way and drive_dir and driven and w_med is not None and 2.5 * len(driven) <= w_med < 2.5 * (len(driven) + 1):
        R, L = seed - bridged("R"), seed + bridged("L")
        keep = driven
        check_sides = ["R", "L"]
        info["case"] = "separated carriageway (kerb to kerb): lanes of the other direction left out"
    else:
        cands = [sd for sd in ("R", "L") if have[sd]]
        if not cands:
            info["rejected"] = "no kerb with evidence along at least 35 % of the piece"
            return None, info
        ref = min(cands, key=lambda sd: fit[sd]["med"])
        far = "L" if ref == "R" else "R"
        ref_off = bridged(ref)
        if fit[far]["line"] is not None and (fit[ref]["sup"] & fit[far]["sup"]).sum() >= 10:
            both = fit[ref]["sup"] & fit[far]["sup"]
            w_meas = ref_off + fit[far]["line"]
            W = float(np.clip(np.percentile(w_meas[both], 15), 2.75 * n_lanes, 3.75 * n_lanes))
            w_far = np.where(fit[far]["sup"], w_meas, np.inf)  # where the far kerb is nearer than W it is the edge
            width = np.minimum(W, np.maximum(w_far, 2.5 * n_lanes))
        else:
            W = 3.5 * n_lanes
            width = np.full(len(st), W)
        width = gaussian_filter1d(width, 3.0 / step, mode="nearest")
        if ref == "L":
            L = seed + ref_off
            R = L - width
        else:
            R = seed - ref_off
            L = R + width
        check_sides = [ref]
        info.update(case="reference kerb " + ("left" if ref == "L" else "right"), width=round(W, 2))
    nk = len(keep)
    for sd in check_sides:  # the edge must lie on the kerb steps found, not merely near the driven line
        raw, edge = (dr, R) if sd == "R" else (dl, L)
        m = valid & np.isfinite(raw) & (off[sd] >= 0.9)
        agree = float(np.mean(np.abs(raw[m] - edge[m]) <= 0.35)) if m.sum() >= 8 else 0.0
        info["edge_on_kerb_" + ("right" if sd == "R" else "left")] = round(agree, 2)
        if agree < 0.6:
            info["rejected"] = "the lane edge does not follow the kerb steps found (the driven line is not parallel to the kerb)"
            return None, info
    if np.min(L - R) < 2.4 * nk:
        info["rejected"] = "narrower than 2.4 m per lane"
        return None, info
    inside = (seed - R >= 0.3) & (L - seed >= 0.3)
    info["driven_line_inside"] = round(float(inside[valid].mean()), 2)
    if inside[valid].mean() < 0.95:
        info["rejected"] = "the driven line is not inside the kerb-based carriageway"
        return None, info
    bounds = np.full((nk + 1, len(stations)), np.nan)
    for k in range(nk + 1):
        bounds[k, iu] = R + k / nk * (L - R)
    info.update(lane_width_median=round(float(np.median(L - R)) / nk, 2), lanes=nk)
    return dict(bounds=bounds, keep=keep), info


def chain_points(sp, ref, half=17.0):
    """Surface points of a road in its (s, d) frame, sorted by s: s, d, z."""
    idx = set()
    for q in ref.at(np.arange(0, ref.length + 5, 5.0)):
        idx.update(sp.kd.query_ball_point(q, half + 4.0))
    idx = np.fromiter(idx, np.int64)
    s_p, d_p = ref.project(sp.xy[idx])
    m = (s_p >= -1) & (s_p <= ref.length + 1) & (np.abs(d_p) <= half)
    o = np.argsort(s_p[m])
    return s_p[m][o], d_p[m][o], sp.z[idx[m]][o]


def seed_from_trajectory(ref, traj_xy, stations, model_centre):
    """(driven line: lateral position per station, direction driven: +1 with the chain, -1 against it, 0 none).

    Only trajectory samples moving along the road within 10 m of the lane model's centre count, and only those of
    the direction driven most (a second pass on the opposite carriageway must not pull the line onto the
    median). Stations the vehicle did not pass get the model centre shifted like the nearest passed station."""
    v = np.diff(traj_xy, axis=0)
    v = np.vstack([v, v[-1:]])
    sp_ = np.linalg.norm(v, axis=1)
    s_t, d_t = ref.project(traj_xy)
    tx, ty = np.interp(s_t, ref.s, ref.t[:, 0]), np.interp(s_t, ref.s, ref.t[:, 1])
    cosang = (v[:, 0] * tx + v[:, 1] * ty) / np.maximum(sp_, 1e-9)
    near = (s_t > -0.5) & (s_t < ref.length + 0.5) & (np.abs(d_t - np.interp(s_t, stations, model_centre)) < 10) & \
        (sp_ > 0.02)
    fwd, bwd = near & (cosang > 0.7), near & (cosang < -0.7)
    if fwd.sum() < 5 and bwd.sum() < 5:
        return model_centre.copy(), 0
    drive_dir = 1 if fwd.sum() >= bwd.sum() else -1
    m = fwd if drive_dir == 1 else bwd
    seed = np.full(len(stations), np.nan)
    for i, sc in enumerate(stations):
        sel = m & (np.abs(s_t - sc) < 1.0)
        if sel.any():
            seed[i] = np.median(d_t[sel])
    ok = np.isfinite(seed)
    if ok.sum() < 2:
        return model_centre.copy(), drive_dir
    off = np.interp(stations, stations[ok], (seed - model_centre)[ok])
    return np.where(ok, seed, model_centre + off), drive_dir


if __name__ == "__main__":
    import json
    import os
    import sys

    import cv2
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from lanegeom import Raster, RefLine
    from pcd_refine import SurfacePoints, read_pcd_xyzi

    LANES, RAS, TRAJ, PCD, OUT_DIR = sys.argv[1:6]
    specs = [tuple(float(v) for v in a_.split(":")) for a_ in sys.argv[6:]]
    chains = json.load(open(LANES))["chains"]
    ras = Raster(RAS)
    mask = np.zeros(ras.ground.shape, np.uint8)
    for ci, _, _ in specs:
        c = [c for c in chains if c["index"] == int(ci)][0]
        pl = np.round((RefLine(c["ref_xy"]).xy - [ras.x0, ras.y0]) / ras.res).astype(np.int32)
        cv2.polylines(mask, [pl.reshape(-1, 1, 2)], False, 1, thickness=int(40.0 / ras.res))
    sp = SurfacePoints(read_pcd_xyzi(PCD), ras.ground, ras.x0, ras.y0, ras.res, mask.astype(bool), zlo=-0.5, zhi=0.6)
    traj = np.load(TRAJ)["pos"][::100, :2]
    for ci, S0, S1 in specs:
        ci = int(ci)
        c = [c for c in chains if c["index"] == ci][0]
        ref = RefLine(c["ref_xy"])
        s_p, d_p, z_p = chain_points(sp, ref)
        mid = 0.5 * (S0 + S1)
        p = [p for p in c["pieces"] if p["s0"] <= mid <= p["s1"]][0]
        st = np.array(p["stations"])
        bd = np.array(p["bounds_d"])
        seed, ddir = seed_from_trajectory(ref, traj, st, 0.5 * (bd[0] + bd[-1]))
        use = (st >= S0) & (st <= S1)
        res, info = kerb_bounds(s_p, d_p, z_p, st, seed, p["lanes"], ddir, use)
        print(f"chain {ci} piece {p['s0']:.0f}-{p['s1']:.0f} ({p.get('name')}), driven {ddir:+d}:", info)
        H = cross_profiles(s_p, d_p, z_p, st)
        dr, dl = raw_kerbs(H, seed)
        j_seed = np.clip(((seed - D_LO) / BIN_W).astype(int), 0, H.shape[1] - 1)
        with np.errstate(all="ignore"):
            base = np.array([np.nanmedian(H[i, max(j - 5, 0):j + 6]) if np.isfinite(H[i, max(j - 5, 0):j + 6]).any()
                             else np.nan for i, j in enumerate(j_seed)])
        fig, ax = plt.subplots(figsize=(max(10, len(st) * 0.06), 9))
        ax.imshow((H - base[:, None]).T, origin="lower", extent=[st[0], st[-1], D_LO, D_HI], aspect="auto",
                  cmap="RdBu_r", vmin=-0.2, vmax=0.2)
        ax.plot(st, seed, "g-", lw=1, label="driven line")
        ax.plot(st, dr, "k.", ms=3)
        ax.plot(st, dl, "k.", ms=3, label="kerb step found")
        for k in range(len(bd)):
            ax.plot(st, bd[k], "--", color="orange", lw=1, label="lane model" if k == 0 else None)
        if res is not None:
            nb_ = res["bounds"]
            for k in range(len(nb_)):
                ax.plot(st, nb_[k], "-", color="magenta" if k in (0, len(nb_) - 1) else "yellow", lw=1.6,
                        label="kerb-based bounds" if k == 0 else None)
        ax.axvline(S0, color="k", lw=0.5)
        ax.axvline(S1, color="k", lw=0.5)
        ax.set_xlabel("s along the road (m)")
        ax.set_ylabel("d across (m, left positive)")
        ax.set_title(f"chain {ci} piece {p['s0']:.0f}-{p['s1']:.0f} m ({p.get('name')}): height above the road at the "
                     "driven line (red = higher)")
        ax.legend(loc="upper right", fontsize=8)
        out = os.path.join(OUT_DIR, f"kerb_chain{ci}_{int(p['s0'])}.png")
        fig.savefig(out, dpi=80, bbox_inches="tight")
        plt.close(fig)
        print("   wrote", out)
