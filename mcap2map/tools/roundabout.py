"""Roundabout geometry from the point cloud map: central island curb (circle) and the outer curb radius as a
function of azimuth (mouths of the arms interpolated), for building the circulating lanelets.

usage (test): roundabout.py PCD RASTER_NPZ cx cy OUT_PNG
"""
import sys
import warnings

import numpy as np
from scipy.ndimage import median_filter, gaussian_filter1d

warnings.filterwarnings("ignore", category=RuntimeWarning)  # all-NaN sector profiles are expected


def _grid(sp, cx, cy, half=26.0, px=0.05):
    """Per-pixel 30th-percentile height and median intensity on a px grid around (cx, cy)."""
    n = int(2 * half / px)
    u = ((sp.xy[:, 0] - (cx - half)) / px).astype(int)
    v = ((sp.xy[:, 1] - (cy - half)) / px).astype(int)
    ok = (u >= 0) & (u < n) & (v >= 0) & (v < n)
    key = v[ok] * n + u[ok]
    o = np.argsort(key, kind="stable")
    key, zz, ii = key[o], sp.z[ok][o], sp.i[ok][o]
    st = np.r_[0, np.flatnonzero(np.diff(key)) + 1, len(key)]
    zimg = np.full(n * n, np.nan)
    iimg = np.full(n * n, np.nan)
    for a, b in zip(st[:-1], st[1:]):
        zimg[key[a]] = np.percentile(zz[a:b], 30)
        iimg[key[a]] = np.median(ii[a:b])
    return zimg.reshape(n, n), iimg.reshape(n, n), n


def ring_geometry(sp, cx0, cy0, r_isl_guess=7.0, da=2.0):
    """Returns dict(centre, r_island, az (deg), r_out (m, per az bin), mouth (bool per az bin), z_ring)."""
    px, half = 0.05, 26.0
    zimg, iimg, n = _grid(sp, cx0, cy0, half, px)
    yy, xx = np.mgrid[0:n, 0:n]
    X = (cx0 - half) + (xx + 0.5) * px
    Y = (cy0 - half) + (yy + 0.5) * px
    # --- island: the central island is a bright (grass / kerb) disc on dark asphalt and a mound in height; its
    # edge per azimuth = outermost radius where the intensity profile is still above the island/road midpoint
    # (height step as a fallback) -> robust circle fit, repeated with the refined centre
    cx, cy = cx0, cy0
    r_isl = r_isl_guess
    for _ in range(3):
        R = np.hypot(X - cx, Y - cy)
        AZ = np.degrees(np.arctan2(Y - cy, X - cx)) % 360
        lvl = np.nanmedian(zimg[(R > r_isl + 1.5) & (R < r_isl + 5)])
        rel = zimg - lvl
        edge = []
        rb = np.arange(2.0, r_isl + 3.0, 0.1)
        for a in np.arange(0, 360, da):
            m = (AZ >= a) & (AZ < a + da) & (R > 1.5) & (R < r_isl + 3.5)
            rr, ii, hh = R[m], iimg[m], rel[m]
            good = np.isfinite(ii)
            rr, ii, hh = rr[good], ii[good], hh[good]
            if len(rr) < 40:
                continue
            ip = np.array([np.median(ii[(rr >= b - 0.15) & (rr < b + 0.15)]) if np.sum((rr >= b - 0.15) & (rr < b + 0.15)) >= 4 else np.nan for b in rb])
            hp = np.array([np.median(hh[(rr >= b - 0.15) & (rr < b + 0.15)]) if np.sum((rr >= b - 0.15) & (rr < b + 0.15)) >= 4 else np.nan for b in rb])
            isl_i = np.nanmedian(ip[rb < r_isl - 1.5])
            road_i = np.nanmedian(ip[rb > r_isl + 1.0])
            if not (np.isfinite(isl_i) and np.isfinite(road_i)) or isl_i < road_i * 1.3 + 3:
                # low contrast: use the height (island >= 6 cm above the road, then road for 0.8 m)
                cand = [j for j in range(3, len(rb) - 8) if np.nan_to_num(hp[j], nan=-1) > 0.06 and
                        np.nanmax(hp[j + 1:j + 9]) < 0.05]
            else:
                thr = (isl_i + road_i) / 2
                cand = [j for j in range(3, len(rb) - 8) if np.nan_to_num(ip[j], nan=-1) > thr and
                        np.nanmedian(ip[j + 1:j + 9]) < thr and np.nan_to_num(hp[j], nan=-1) > 0.0]
            if cand:
                edge.append((a + da / 2, rb[cand[0]] + 0.05))
        edge = np.array(edge)
        pts = np.column_stack([cx + edge[:, 1] * np.cos(np.radians(edge[:, 0])), cy + edge[:, 1] * np.sin(np.radians(edge[:, 0]))])
        for it in range(8):
            A = np.column_stack([2 * pts[:, 0], 2 * pts[:, 1], np.ones(len(pts))])
            sol = np.linalg.lstsq(A, (pts ** 2).sum(1), rcond=None)[0]
            c = sol[:2]
            r_isl = float(np.sqrt(sol[2] + c @ c))
            d = np.abs(np.hypot(pts[:, 0] - c[0], pts[:, 1] - c[1]) - r_isl)
            keep = d < max(0.12, 2.0 * np.median(d))
            if keep.sum() < 20 or keep.all():
                break
            pts = pts[keep]
        cx, cy = float(c[0]), float(c[1])
    island_resid = float(np.median(d))
    n_edge = int(len(pts))
    R = np.hypot(X - cx, Y - cy)
    AZ = np.degrees(np.arctan2(Y - cy, X - cx)) % 360
    # --- mountable apron around the island (concrete: brighter than the asphalt): its outer edge from the
    # azimuth-averaged intensity profile; the circulating lane starts there
    rb = np.arange(r_isl - 0.5, r_isl + 4.0, 0.1)
    prof = np.array([np.nanmedian(iimg[(R >= b - 0.1) & (R < b + 0.1)]) for b in rb])
    road_i = np.nanmedian(iimg[(R > r_isl + 3.0) & (R < r_isl + 5.0)])
    apron_i = np.nanmedian(prof[(rb > r_isl + 0.2) & (rb < r_isl + 0.8)])
    r_in = r_isl
    if np.isfinite(apron_i) and apron_i > road_i * 1.3 + 2:
        thr = road_i + 0.35 * (apron_i - road_i)
        below = np.flatnonzero((rb >= r_isl + 0.3) & (np.nan_to_num(prof, nan=0) < thr))
        if len(below):
            r_in = float(rb[below[0]])
    # --- outer kerb per azimuth: 10th-percentile height profile against the local ring road level; kerb = first
    # upward step of >= 6 cm within 0.4 m beyond r_in + 3 (sectors without one are arm mouths)
    da_o = da
    azs = np.arange(0, 360, da_o) + da_o / 2
    r_out = np.full(len(azs), np.nan)
    rb = np.arange(r_in + 0.5, 24.0, 0.1)
    for k, a in enumerate(azs):
        m = (AZ >= a - da_o / 2) & (AZ < a + da_o / 2) & (R > r_in + 0.3) & (R < 24.5)
        rr, hh = R[m], zimg[m]
        good = np.isfinite(hh)
        rr, hh = rr[good], hh[good]
        if len(rr) < 50:
            continue
        prof = np.array([np.percentile(hh[(rr >= b - 0.2) & (rr < b + 0.2)], 15) if np.sum((rr >= b - 0.2) & (rr < b + 0.2)) >= 5 else np.nan for b in rb])
        for j in range(5, len(rb) - 5):
            if rb[j] < r_in + 3.0:
                continue
            inner = prof[j - 5:j - 1]
            outer = prof[j + 1:j + 5]
            if np.isfinite(inner).sum() >= 2 and np.isfinite(outer).sum() >= 2 and \
                    np.nanmedian(outer) - np.nanmedian(inner) > 0.06:
                r_out[k] = rb[j]
                break
    valid = np.isfinite(r_out) & (r_out > r_in + 4.0) & (r_out < r_in + 12.0)
    med = median_filter(np.where(valid, r_out, np.nan), size=9, mode="wrap")
    med = np.where(np.isfinite(med), med, np.nanmedian(r_out[valid]))
    valid &= np.abs(r_out - med) < 1.0
    mouth = ~valid
    idx = np.flatnonzero(valid)
    ext_az = np.r_[azs[idx] - 360, azs[idx], azs[idx] + 360]
    ext_r = np.r_[r_out[idx], r_out[idx], r_out[idx]]
    r_fill = np.interp(azs, ext_az, ext_r)
    r_sm = gaussian_filter1d(np.r_[r_fill, r_fill, r_fill], 3.0, mode="wrap")[len(azs):2 * len(azs)]
    return {"centre": (cx, cy), "r_island": r_isl, "r_in": r_in, "island_resid": island_resid, "n_edge": n_edge,
            "az": azs, "r_out": r_sm, "r_out_raw": r_out, "mouth": mouth, "z_ring": float(lvl),
            "apron_i": float(apron_i) if np.isfinite(apron_i) else None, "road_i": float(road_i)}


def ring_r_out(geom, theta_deg):
    """Outer boundary radius at azimuth (deg), periodic linear interpolation."""
    az, r = geom["az"], geom["r_out"]
    ext_az = np.r_[az - 360, az, az + 360]
    return np.interp(np.asarray(theta_deg) % 360, ext_az, np.r_[r, r, r])


def ring_xy(geom, theta_deg, r):
    t = np.radians(np.asarray(theta_deg, float))
    cx, cy = geom["centre"]
    return np.column_stack([cx + np.asarray(r) * np.cos(t), cy + np.asarray(r) * np.sin(t)])


if __name__ == "__main__":
    import os
    import cv2
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from pcd_refine import SurfacePoints, read_pcd_xyzi
    PCD, RAS = sys.argv[1:3]
    cx0, cy0 = float(sys.argv[3]), float(sys.argv[4])
    OUT = sys.argv[5]
    z = np.load(RAS)
    x0, y0, res, ground = float(z["x0"]), float(z["y0"]), float(z["res"]), z["ground"]
    mask = np.zeros(ground.shape, bool)
    i0, i1 = int((cx0 - 28 - x0) / res), int((cx0 + 28 - x0) / res)
    j0, j1 = int((cy0 - 28 - y0) / res), int((cy0 + 28 - y0) / res)
    mask[j0:j1, i0:i1] = True
    sp = SurfacePoints(read_pcd_xyzi(PCD), ground, x0, y0, res, mask, zlo=-0.6, zhi=1.2)
    g = ring_geometry(sp, cx0, cy0)
    print(f"centre {g['centre']}, island r {g['r_island']:.2f} (edge resid {g['island_resid']*100:.1f} cm, {g['n_edge']} edge pts), "
          f"apron edge r_in {g['r_in']:.2f} (apron intensity {g['apron_i']}, road {g['road_i']:.1f}), ring z {g['z_ring']:.2f}")
    for a, r, rr, mo in zip(g["az"], g["r_out"], g["r_out_raw"], g["mouth"]):
        if int(a) % 10 == 1:
            print(f"  az {a:5.1f}: r_out {r:5.2f} raw {rr:5.2f} {'MOUTH' if mo else ''}")
    # overlay on an intensity render
    PX, half = 0.05, 26.0
    cx, cy = g["centre"]
    zimg, iimg, n = _grid(sp, cx, cy, half, PX)
    img = cv2.cvtColor(np.clip(np.nan_to_num(iimg) * 6, 0, 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    img = cv2.flip(img, 0)
    def P(xy):
        return np.round(np.column_stack([(xy[:, 0] - (cx - half)) / PX, n - 1 - (xy[:, 1] - (cy - half)) / PX]) * 4).astype(np.int32).reshape(-1, 1, 2)
    th = np.arange(0, 361, 1.0)
    cv2.polylines(img, [P(ring_xy(g, th, g["r_island"]))], True, (255, 0, 255), 1, cv2.LINE_AA, shift=2)
    cv2.polylines(img, [P(ring_xy(g, th, g["r_in"]))], True, (0, 255, 255), 1, cv2.LINE_AA, shift=2)
    cv2.polylines(img, [P(ring_xy(g, th, ring_r_out(g, th)))], True, (255, 0, 255), 1, cv2.LINE_AA, shift=2)
    for a, rr, mo in zip(g["az"], g["r_out_raw"], g["mouth"]):
        if np.isfinite(rr):
            p = P(ring_xy(g, [a], [rr]))[0, 0] // 4
            cv2.circle(img, (int(p[0]), int(p[1])), 3, (0, 0, 255) if mo else (0, 255, 0), -1)
    cv2.imwrite(OUT, img)
    print("wrote", OUT)
