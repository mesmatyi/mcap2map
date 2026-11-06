"""Fit lane boundary polylines to the point cloud map itself.

* painted lines: intensity-weighted centre of the paint in the PCD voxels across the line (sub-voxel),
  double lines -> midpoint; robust smoothing spline along the line
* curbs: position of the height step between road surface and curb top
* elevation: median height of the PCD road surface around each vertex
"""
import numpy as np
from scipy.interpolate import make_smoothing_spline
from scipy.ndimage import gaussian_filter1d
from scipy.spatial import cKDTree

STEP = 0.5        # station spacing along a line [m]
HALF_ALONG = 0.25  # half length of the evidence window along the line [m]


def read_pcd_xyzi(path):
    """x y z intensity PCD -> (N, 4). An x y z rgb intensity PCD (the coloured map) is accepted too: the rgb
    column is replaced by nothing, and the camera luminance is returned as a second array (see read_pcd_rgb)."""
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"DATA binary\n"):
            hdr += f.read(1)
        fields = [l for l in hdr.decode().splitlines() if l.startswith("FIELDS")][0].split()[1:]
        data = np.frombuffer(f.read(), dtype="<f4").reshape(-1, len(fields))
    if "rgb" in fields:
        ix = [fields.index(k) for k in ("x", "y", "z", "intensity")]
        return np.ascontiguousarray(data[:, ix])
    return data


def read_pcd_rgb(path):
    """Coloured map PCD -> (xyzi (N, 4), luminance (N,) 0..255 from the camera colours)."""
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"DATA binary\n"):
            hdr += f.read(1)
        fields = [l for l in hdr.decode().splitlines() if l.startswith("FIELDS")][0].split()[1:]
        data = np.frombuffer(f.read(), dtype="<f4").reshape(-1, len(fields))
    ix = [fields.index(k) for k in ("x", "y", "z", "intensity")]
    pk = data[:, fields.index("rgb")].view(np.uint32)
    r, g, b = (pk >> 16) & 255, (pk >> 8) & 255, pk & 255
    lum = (0.299 * r + 0.587 * g + 0.114 * b).astype(np.float32)
    return np.ascontiguousarray(data[:, ix]), lum


class SurfacePoints:
    """PCD points near the road surface inside a corridor, with a 2-D KD-tree."""

    def __init__(self, pcd, ground, x0, y0, res, corridor_mask, zlo=-0.25, zhi=0.35, lum=None):
        ix = np.floor((pcd[:, 0] - x0) / res).astype(np.int64)
        iy = np.floor((pcd[:, 1] - y0) / res).astype(np.int64)
        ok = (ix >= 0) & (ix < ground.shape[1]) & (iy >= 0) & (iy < ground.shape[0])
        sel = np.flatnonzero(ok)
        sel = sel[corridor_mask[iy[sel], ix[sel]]]
        g = ground[iy[sel], ix[sel]]
        dz = pcd[sel, 2] - g
        keep = np.isfinite(g) & (dz > zlo) & (dz < zhi)
        sel = sel[keep]
        self.xy = pcd[sel, :2].astype(np.float64)
        self.z = pcd[sel, 2].astype(np.float64)
        self.i = pcd[sel, 3].astype(np.float64)
        self.g = g[keep].astype(np.float64)
        self.lum = lum[sel].astype(np.float64) if lum is not None else None  # camera luminance (coloured map)
        self.kd = cKDTree(self.xy)

    def box(self, p, t, n, half_along, half_across):
        idx = np.array(self.kd.query_ball_point(p, np.hypot(half_along, half_across)), dtype=np.int64)
        if not len(idx):
            return idx, np.zeros(0), np.zeros(0)
        v = self.xy[idx] - p
        a, c = v @ t, v @ n
        m = (np.abs(a) <= half_along) & (np.abs(c) <= half_across)
        return idx[m], a[m], c[m]


def frame(xy):
    """Arc length, unit tangents and left normals of a polyline (tangents from a lightly smoothed copy)."""
    xy = np.asarray(xy, float)
    s = np.r_[0, np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))]
    if len(xy) >= 5:
        sm = np.column_stack([gaussian_filter1d(xy[:, k], 1.5, mode="nearest") for k in range(2)])
    else:
        sm = xy
    t = np.gradient(sm, axis=0)
    t /= np.maximum(np.linalg.norm(t, axis=1), 1e-9)[:, None]
    return s, t, np.column_stack([-t[:, 1], t[:, 0]])


def resample(xy, step):
    s, _, _ = frame(xy)
    if s[-1] < 1e-6:
        return np.asarray(xy, float)[:1], np.zeros(1)
    st = np.linspace(0, s[-1], max(2, int(np.ceil(s[-1] / step)) + 1))
    return np.column_stack([np.interp(st, s, xy[:, 0]), np.interp(st, s, xy[:, 1])]), st


def gradient_corrected_centre(c, inten, mu0, half=0.12, bg_in=0.18, bg_out=0.35):
    """Paint centre near mu0: subtract a linear background fitted beside the line, centroid of the residual."""
    side = (np.abs(c - mu0) > bg_in) & (np.abs(c - mu0) < bg_out)
    core = np.abs(c - mu0) <= half
    if side.sum() < 6 or core.sum() < 3:
        return None
    A = np.column_stack([np.ones(side.sum()), c[side] - mu0])
    coef, *_ = np.linalg.lstsq(A, inten[side], rcond=None)
    w = inten[core] - (coef[0] + coef[1] * (c[core] - mu0))
    w = np.clip(w, 0, None)
    if w.sum() <= 0:
        return None
    return float(np.sum(c[core] * w) / w.sum())


def paint_offset(sp, p, t, n, half_across, half_along=HALF_ALONG):
    """Lateral offset of the paint centre near station p (None if no paint), plus evidence mass and kind."""
    idx, a, c = sp.box(p, t, n, half_along, half_across)
    if len(idx) < 8:
        return None
    dzb = sp.z[idx] - sp.g[idx]
    surf = np.abs(dzb - np.percentile(dzb, 30)) < 0.05  # local road-surface level (no curb tops / objects)
    idx, a, c = idx[surf], a[surf], c[surf]
    if len(idx) < 8:
        return None
    inten = sp.i[idx]
    bg = np.median(inten)
    thr = max(bg + 6.0, 1.8 * bg)
    paint = inten > thr
    if paint.sum() < 2:
        return None
    w = inten[paint] - bg
    cp = c[paint]
    bins = np.arange(-half_across, half_across + 0.02, 0.02)
    h, _ = np.histogram(cp, bins=bins, weights=w)
    h = gaussian_filter1d(h, 1.5)
    centers = (bins[:-1] + bins[1:]) / 2
    pk = [k for k in range(1, len(h) - 1) if h[k] >= h[k - 1] and h[k] > h[k + 1] and h[k] > 0.15 * h.max()]
    if not pk:
        return None
    peaks = []
    ap = a[paint]
    for k in pk:
        m = np.abs(cp - centers[k]) <= 0.12
        if m.sum() < 3:
            continue
        # line-shaped evidence: paint spread along the line, narrow across it
        if len(np.unique(np.floor((ap[m] + half_along) / 0.05))) < 3:
            continue
        mu = float(np.sum(cp[m] * w[m]) / np.sum(w[m]))
        if np.sqrt(np.sum(w[m] * (cp[m] - mu) ** 2) / np.sum(w[m])) > 0.08:
            continue
        mu2 = gradient_corrected_centre(c, inten, mu)
        if mu2 is not None and abs(mu2 - mu) < 0.06:
            mu = mu2
        peaks.append((mu, float(np.sum(w[m])), int(m.sum())))
    if not peaks:
        return None
    peaks.sort(key=lambda z: abs(z[0]))
    best = peaks[0]
    # double line: two comparable peaks 0.15-1.2 m apart with the current position between them
    for q in peaks[1:]:
        if 0.15 <= abs(q[0] - best[0]) <= 1.2 and min(q[1], best[1]) > 0.4 * max(q[1], best[1]) \
                and min(q[0], best[0]) < 0.05 and max(q[0], best[0]) > -0.05:
            return (best[0] * best[1] + q[0] * q[1]) / (best[1] + q[1]), best[1] + q[1], "double"
    return best[0], best[1], "single"


def curb_offset(sp, p, t, n, road_sign, half_across=1.0, outward=3.5):
    """Lateral offset of the curb (height step) near station p; road_sign=+1 if the road lies on +n.

    Scans the median height profile from the road side outward (up to `outward` m) for the first rise of
    6-35 cm that stays up for 0.3 m; returns the half-step crossing.
    """
    idx = np.array(sp.kd.query_ball_point(p, np.hypot(HALF_ALONG, outward + 0.1)), dtype=np.int64)
    if len(idx) < 15:
        return None
    v = sp.xy[idx] - p
    a, c = v @ t, (v @ n) * road_sign  # c > 0: road side
    m = (np.abs(a) <= HALF_ALONG) & (c > -outward) & (c < half_across)
    idx, c = idx[m], c[m]
    z = sp.z[idx]
    road = (c > 0.15) & (c < half_across)
    if road.sum() < 5:
        return None
    zr = np.percentile(z[road], 40)
    bins = np.arange(-outward, half_across + 1e-6, 0.05)
    centers = (bins[:-1] + bins[1:]) / 2
    which = np.digitize(c, bins) - 1
    zb = np.full(len(centers), np.nan)
    for k in range(len(centers)):
        mm = which == k
        if mm.sum() >= 2:
            zb[k] = np.median(z[mm]) - zr
    start = int(np.searchsorted(centers, 0.1))
    for k in range(start, 6, -1):  # outward = decreasing c
        if not (np.isfinite(zb[k]) and zb[k] < 0.04):
            continue
        after = zb[k - 6:k]
        after = after[np.isfinite(after)]
        if len(after) >= 4 and 0.06 <= np.median(after) <= 0.35 and np.all(after > 0.04):
            half = np.median(after) / 2
            j = k - 1
            while j > k - 6 and not (np.isfinite(zb[j]) and zb[j] >= half):
                j -= 1
            zj, zk = zb[j], zb[j + 1] if np.isfinite(zb[j + 1]) else zb[k]
            f = (half - zk) / max(zj - zk, 1e-6) if np.isfinite(zj) else 0.5
            off = centers[j + 1] - np.clip(f, 0, 1) * 0.05
            return off * road_sign, float(len(idx)), "curb"
    return None


def robust_spline(s, d, w, lam=None):
    """Smoothing spline with MAD outlier rejection; returns callable."""
    keep = np.ones(len(s), bool)
    spl = None
    for _ in range(3):
        if keep.sum() < 4:
            break
        try:
            spl = make_smoothing_spline(s[keep], d[keep], w=w[keep], lam=lam)
        except Exception:
            return None
        r = d - spl(s)
        mad = np.median(np.abs(r[keep])) * 1.4826 + 1e-3
        new = np.abs(r) < max(3 * mad, 0.02)
        if (new == keep).all():
            break
        keep = new
    return spl


def refine_polyline(sp, xy, kind, road_sign=0, n_iter=3, search=(1.0, 0.5, 0.35), anchors=None, lam_scale=1.0):
    """Move a polyline onto PCD evidence. kind: 'paint' or 'curb'. Returns (new_xy, report dict).

    anchors: boolean mask of vertices that must not move (shared with already refined lines).
    """
    xy = np.asarray(xy, float).copy()
    rep = {"coverage": 0.0, "n_meas": 0, "resid_rms": np.nan, "moved_median": 0.0, "double": 0}
    if len(xy) < 2:
        return xy, rep
    s_v, _, _ = frame(xy)
    L = s_v[-1]
    if L < 1.0:
        return xy, rep
    orig = xy.copy()
    for it in range(n_iter):
        st_xy, st = resample(xy, STEP)
        _, t, nrm = frame(st_xy)
        meas = []
        for k in range(len(st)):
            if kind == "paint":
                r = paint_offset(sp, st_xy[k], t[k], nrm[k], search[min(it, len(search) - 1)])
            else:
                r = curb_offset(sp, st_xy[k], t[k], nrm[k], road_sign)
            if r is not None:
                meas.append((st[k], r[0], r[1], r[2]))
        if len(meas) < 4:
            break
        m = np.array([(a, b, c) for a, b, c, _ in meas])
        rep["double"] = sum(1 for z in meas if z[3] == "double")
        spl = robust_spline(m[:, 0], m[:, 1], np.sqrt(m[:, 2]), lam=None)
        if spl is None:
            break
        off = spl(st)
        # far from any measurement the spline extrapolates: fade the correction to the neighbours' level
        gap = np.min(np.abs(st[:, None] - m[None, :, 0]), axis=1)
        off = np.where(gap > 6.0, np.interp(st, m[:, 0], spl(m[:, 0])), off)
        if it == n_iter - 1 or True:
            r_all = m[:, 1] - spl(m[:, 0])
            inl = np.abs(r_all) < 0.04
            rep["inlier_cov"] = float(inl.sum() / max(len(st), 1))
            rep["robust_resid"] = float(np.median(np.abs(r_all[inl])) * 1.4826) if inl.any() else np.nan
            if inl.sum() >= 4 and inl.sum() < len(m):
                spl2 = robust_spline(m[inl, 0], m[inl, 1], np.sqrt(m[inl, 2]), lam=None)
                if spl2 is not None:
                    spl = spl2
                    off = spl(st)
                    gap = np.min(np.abs(st[:, None] - m[inl, 0][None, :]), axis=1)
                    off = np.where(gap > 6.0, np.interp(st, m[inl, 0], spl(m[inl, 0])), off)
        new_st = st_xy + off[:, None] * nrm
        # map vertices onto the corrected station curve by arc length
        s_v, _, _ = frame(xy)
        sv = s_v / max(s_v[-1], 1e-9) * st[-1]
        cand = np.column_stack([np.interp(sv, st, new_st[:, 0]), np.interp(sv, st, new_st[:, 1])])
        if anchors is not None:
            cand[anchors] = xy[anchors]
        xy = cand
        rep["n_meas"] = len(m)
        rep["coverage"] = float(np.mean(gap <= STEP))
        rep["resid_rms"] = float(np.sqrt(np.mean((m[:, 1] - spl(m[:, 0])) ** 2)))
        rep["last_offsets"] = m
    # accept only a consistent line: enough stations agreeing within 4 cm, tight robust residual
    rep["accepted"] = bool(rep.get("inlier_cov", 0) >= (0.2 if kind == "paint" else 0.25) and
                           rep.get("robust_resid", 1) <= (0.025 if kind == "paint" else 0.035))
    if not rep["accepted"]:
        xy = orig
    rep["moved_median"] = float(np.median(np.linalg.norm(xy - orig, axis=1)))
    rep["moved_max"] = float(np.max(np.linalg.norm(xy - orig, axis=1)))
    return xy, rep


def surface_z(sp, xy, radius=0.25):
    """PCD road-surface height around each vertex: median of the points near the local surface level."""
    out = np.full(len(xy), np.nan)
    for k, p in enumerate(np.asarray(xy, float)):
        idx = sp.kd.query_ball_point(p, radius)
        if len(idx) >= 3:
            z = sp.z[np.array(idx)]
            lvl = np.percentile(z, 30)
            z = z[np.abs(z - lvl) < 0.05]
            if len(z) >= 3:
                out[k] = np.median(z)
    return out


def paint_tracks(s, d, inten, s_win=0.5, link_dd=0.07, max_gap=11.0):
    """Precise paint lines from PCD points given in a road's (s, d) frame.

    Returns a list of dicts with 's0', 's1', callable 'd' (smoothing spline), 'duty', 'resid', 'n'.
    """
    if len(s) < 50:
        return []
    order = np.argsort(s)
    s, d, inten = s[order], d[order], inten[order]
    edges = np.arange(s[0], s[-1] + s_win, s_win)
    starts = np.searchsorted(s, edges)
    obs = []
    for k in range(len(edges) - 1):
        a, b = starts[k], starts[k + 1]
        if b - a < 10:
            continue
        sw, dw, iw = s[a:b], d[a:b], inten[a:b]
        # local background per 1 m lateral band
        band = np.floor(dw).astype(np.int64)
        bg = np.empty(len(iw))
        for u in np.unique(band):
            m = band == u
            bg[m] = np.median(iw[m]) if m.sum() >= 5 else np.median(iw)
        paint = iw > np.maximum(bg + 6.0, 1.8 * bg)
        if paint.sum() < 3:
            continue
        w = iw[paint] - bg[paint]
        dp, sp_ = dw[paint], sw[paint]
        bins = np.arange(np.floor(dp.min() * 50) / 50 - 0.1, dp.max() + 0.12, 0.02)
        h, _ = np.histogram(dp, bins=bins, weights=w)
        h = gaussian_filter1d(h, 1.5)
        cen = (bins[:-1] + bins[1:]) / 2
        for j in range(1, len(h) - 1):
            if not (h[j] >= h[j - 1] and h[j] > h[j + 1] and h[j] > 0):
                continue
            m = np.abs(dp - cen[j]) <= 0.12
            if m.sum() < 3 or len(np.unique(np.floor((sp_[m] - edges[k]) / 0.05))) < 3:
                continue
            mu = float(np.sum(dp[m] * w[m]) / np.sum(w[m]))
            if np.sqrt(np.sum(w[m] * (dp[m] - mu) ** 2) / np.sum(w[m])) > 0.08:
                continue
            obs.append((edges[k] + s_win / 2, mu, float(np.sum(w[m]))))
    if not obs:
        return []
    obs = np.array(sorted(obs))
    # link observations into tracks (nearly parallel to the reference line)
    tracks = []
    for so, do, wo in obs:
        best, bd_ = None, link_dd
        for tk in tracks:
            ls_, ld_ = tk[-1][0], tk[-1][1]
            if so - ls_ > max_gap or so <= ls_:
                continue
            # predict with the recent slope
            if len(tk) >= 3:
                arr = np.array(tk[-6:])
                slope = np.polyfit(arr[:, 0], arr[:, 1], 1)[0] if arr[-1, 0] > arr[0, 0] else 0.0
            else:
                slope = 0.0
            err = abs(do - (ld_ + slope * (so - ls_)))
            if err < bd_ + 0.004 * (so - ls_):
                best, bd_ = tk, err
        if best is None:
            tracks.append([(so, do, wo)])
        else:
            best.append((so, do, wo))
    out = []
    for tk in tracks:
        a = np.array(tk)
        if len(a) < 6 or a[-1, 0] - a[0, 0] < 4.0:
            continue
        spl = robust_spline(a[:, 0], a[:, 1], np.sqrt(a[:, 2]))
        if spl is None:
            continue
        r = a[:, 1] - spl(a[:, 0])
        inl = np.abs(r) < 0.04
        if inl.sum() < 6:
            continue
        resid = float(np.median(np.abs(r[inl])) * 1.4826)
        if resid > 0.025:
            continue
        spl = robust_spline(a[inl, 0], a[inl, 1], np.sqrt(a[inl, 2])) or spl
        span = a[inl, 0].max() - a[inl, 0].min()
        out.append({"s0": float(a[inl, 0].min()), "s1": float(a[inl, 0].max()), "d": spl, "n": int(inl.sum()),
                    "duty": float(inl.sum() * s_win / max(span, s_win)), "resid": resid,
                    "obs_s": a[inl, 0], "obs_d": a[inl, 1]})
    return out


def window_peaks(sw, dw, iw, zw, lw=None):
    """Line-shaped paint peaks (centre d, mass) across a road cross-section window of PCD points.

    Paint is anything brighter than the local road in the lidar intensity OR (if lw is given) in the camera
    luminance of the coloured map; the weight of a point is its larger relative contrast."""
    # road-surface level from the local surface (30th height percentile per 1 m band), not the smoothed raster
    band = np.floor(dw).astype(np.int64)
    ref = np.full(len(zw), np.percentile(zw, 30))
    for u in np.unique(band):
        m = band == u
        if m.sum() >= 8:
            ref[m] = np.percentile(zw[m], 30)
    surf = np.abs(zw - ref) < 0.05  # no curb tops / objects
    sw, dw, iw = sw[surf], dw[surf], iw[surf]
    lw = lw[surf] if lw is not None else None
    if len(iw) < 10:
        return []
    band = np.floor(dw).astype(np.int64)
    bg = np.full(len(iw), np.median(iw))
    bgl = np.full(len(iw), np.median(lw)) if lw is not None else None
    for u in np.unique(band):
        m = band == u
        if m.sum() >= 5:
            bg[m] = np.median(iw[m])
            if lw is not None:
                bgl[m] = np.median(lw[m])
    contrast = (iw - bg) / np.maximum(bg, 3.0)
    paint = iw > np.maximum(bg + 4.5, 1.55 * bg)
    if lw is not None:
        cl = (lw - bgl) / np.maximum(bgl, 25.0)
        paint |= lw > np.maximum(bgl + 30.0, 1.35 * bgl)
        contrast = np.maximum(contrast, cl)
    if paint.sum() < 3:
        return []
    if lw is None:  # lidar only: weight = intensity excess over the local road (what the precision figures rest on)
        w, dp, sp_ = iw[paint] - bg[paint], dw[paint], sw[paint]
    else:  # with camera luminance: the larger relative contrast of the two channels
        w, dp, sp_ = np.maximum(contrast[paint], 0.1) * 10.0, dw[paint], sw[paint]
    bins = np.arange(np.floor(dp.min() * 100) / 100 - 0.1, dp.max() + 0.11, 0.01)
    h, _ = np.histogram(dp, bins=bins, weights=w)
    h = gaussian_filter1d(h, 2.0)
    cen = (bins[:-1] + bins[1:]) / 2
    out = []
    for j in range(1, len(h) - 1):
        if not (h[j] >= h[j - 1] and h[j] > h[j + 1] and h[j] > 0):
            continue
        m = np.abs(dp - cen[j]) <= 0.12
        if m.sum() < 3 or len(np.unique(np.floor(sp_[m] / 0.1))) < 3:
            continue
        mu = float(np.sum(dp[m] * w[m]) / np.sum(w[m]))
        if np.sqrt(np.sum(w[m] * (dp[m] - mu) ** 2) / np.sum(w[m])) > 0.08:
            continue
        mu2 = gradient_corrected_centre(dw, iw, mu)
        if mu2 is not None and abs(mu2 - mu) < 0.06:
            mu = mu2
        if out and abs(out[-1][0] - mu) < 0.08:  # same line seen as two bumps
            if np.sum(w[m]) > out[-1][1]:
                out[-1] = (mu, float(np.sum(w[m])))
            continue
        out.append((mu, float(np.sum(w[m]))))
    return out


def fit_boundaries_pcd(s, d, inten, dz, st, bd, centre_k=-1, win=1.0, step=0.5,
                       rounds=((0.6, False, 0.15), (1.3, True, 0.15), (0.2, False, 0.08))):
    """Fit all boundaries of one road piece to PCD paint.

    s, d, inten, dz: PCD surface points in the road's (s, d) frame, sorted by s (dz = height above ground model).
    st: piece stations; bd: (nb, len(st)) boundary offsets. Returns (new_bd, info list per boundary).
    """
    nb = len(bd)
    cur = np.array(bd, float).copy()
    info = [dict(accepted=False, coverage=0.0, resid=np.nan, double=0, duty=0.0, n=0) for _ in range(nb)]
    centers = np.arange(st[0] + win / 2, st[-1] - win / 2 + 1e-6, step) if st[-1] - st[0] > win else np.array([])
    win_peaks = []
    for sc in centers:
        a, b = np.searchsorted(s, sc - win / 2), np.searchsorted(s, sc + win / 2)
        win_peaks.append(window_peaks(s[a:b], d[a:b], inten[a:b], dz[a:b]) if b - a >= 10 else [])
    for search, only_unacc, min_cov in rounds:
        meas = [[] for _ in range(nb)]
        acc_now = np.array([x["accepted"] for x in info])
        for sc, peaks in zip(centers, win_peaks):
            if not peaks:
                continue
            P = np.array(peaks)
            bk = np.array([np.interp(sc, st, cur[k]) for k in range(nb)])
            if only_unacc:  # paint already explained by a fitted boundary is not up for grabs
                owned = np.zeros(len(P), bool)
                for k in np.flatnonzero(acc_now):
                    owned |= np.abs(P[:, 0] - bk[k]) < 0.15
                P = P[~owned]
                if not len(P):
                    continue
            near_k = np.argmin(np.abs(P[:, 0][:, None] - bk[None, :]), axis=1)
            for k in range(nb):
                if only_unacc and acc_now[k]:
                    continue
                sel = np.flatnonzero((near_k == k) & (np.abs(P[:, 0] - bk[k]) < search))
                if not len(sel):
                    continue
                if k == centre_k and len(sel) >= 2:
                    two = sel[np.argsort(-P[sel, 1])[:2]]
                    sep = abs(P[two[0], 0] - P[two[1], 0])
                    if 0.15 <= sep <= 1.3 and P[two[1], 1] > 0.4 * P[two[0], 1]:
                        meas[k].append((sc, float(P[two, 0].mean()), float(P[two, 1].sum()), 1))
                        continue
                j = sel[np.argmin(np.abs(P[sel, 0] - bk[k]))]
                meas[k].append((sc, float(P[j, 0]), float(P[j, 1]), 0))
        new = cur.copy()
        for k in range(nb):
            if only_unacc and acc_now[k]:
                continue
            if len(meas[k]) < 4:
                if not only_unacc:
                    info[k]["accepted"] = False
                continue
            M = np.array(meas[k])
            spl = robust_spline(M[:, 0], M[:, 1], np.sqrt(M[:, 2]))
            if spl is None:
                continue
            r = M[:, 1] - spl(M[:, 0])
            inl = np.abs(r) < 0.04
            if inl.sum() < 4:
                if not only_unacc:
                    info[k]["accepted"] = False
                continue
            spl = robust_spline(M[inl, 0], M[inl, 1], np.sqrt(M[inl, 2])) or spl
            resid = float(np.median(np.abs(M[inl, 1] - spl(M[inl, 0]))) * 1.4826)
            cov = inl.sum() * step / max(st[-1] - st[0], step)
            span = M[inl, 0].max() - M[inl, 0].min()
            ok_k = bool(cov >= min_cov and resid <= 0.025)
            if only_unacc and not ok_k:
                continue
            if not ok_k and not only_unacc and search < 0.3 and acc_now[k]:
                continue  # the tight final pass keeps an earlier accepted fit if it finds less paint
            info[k] = dict(accepted=ok_k, coverage=float(cov), resid=resid,
                           double=int(M[inl, 3].sum()), duty=float(inl.sum() * step / max(span, step)), n=int(inl.sum()))
            if info[k]["accepted"]:
                ms = M[inl, 0]
                delta = spl(st) - cur[k]
                lo_, hi_ = ms.min(), ms.max()
                # beyond the measured range keep the end corrections; across long gaps interpolate the corrections
                gap = np.min(np.abs(st[:, None] - ms[None, :]), axis=1)
                d_at = np.interp(st, ms, spl(ms) - np.interp(ms, st, cur[k]))
                delta = np.where((st < lo_) | (st > hi_) | (gap > 6.0), d_at, delta)
                new[k] = cur[k] + delta
        cur = new
    return cur, info


# ----------------------------------------------------------------------------------------------------------
# Chain-level line fitting: one smooth penalised B-spline per lane line over its whole length
# ----------------------------------------------------------------------------------------------------------
from scipy.interpolate import BSpline


def pspline(x, y, w, x0, x1, knot_spacing, lam=5.0, k=3, iters=5, inlier=0.04):
    """Penalised cubic B-spline d(x) on [x0, x1] (knots every knot_spacing m, second-difference penalty lam),
    fitted by iteratively re-weighted least squares with a hard inlier cut. Returns (BSpline, inlier mask)."""
    x0, x1 = float(x0), float(x1)
    if x1 - x0 < 1e-6:
        return None, np.zeros(len(x), bool)
    n_int = max(1, int(np.ceil((x1 - x0) / knot_spacing)))
    t = np.r_[[x0] * k, np.linspace(x0, x1, n_int + 1), [x1] * k]
    nc = len(t) - k - 1
    A = BSpline.design_matrix(np.clip(x, x0, x1), t, k).toarray()
    D = np.diff(np.eye(nc), n=2, axis=0)
    P = lam * (D.T @ D)
    wk = w / max(w.mean(), 1e-9)
    keep = np.ones(len(x), bool)
    c = None
    for it in range(iters):
        W = wk * keep
        M = A.T @ (A * W[:, None]) + P
        c = np.linalg.solve(M, A.T @ (W * y))
        r = y - A @ c
        mad = 1.4826 * np.median(np.abs(r[keep])) + 1e-4
        thr = max(inlier, 3 * mad) if it < 2 else inlier
        new = np.abs(r) < thr
        if new.sum() < 4:
            break
        if (new == keep).all():
            break
        keep = new
    return BSpline(t, c, k, extrapolate=True), keep


def link_lines(pieces, tol=0.6):
    """Link the boundaries of consecutive pieces into continuous lines (one-to-one, by end offset)."""
    lines, line_of = [], {}
    for pi, p in enumerate(pieces):
        nb = len(p["bounds_d"])
        if pi > 0:
            endA = np.asarray(pieces[pi - 1]["bounds_d"])[:, -1]
            startB = np.asarray(p["bounds_d"])[:, 0]
            pairs = sorted((abs(endA[j] - startB[k]), j, k) for j in range(len(endA)) for k in range(nb))
            usedA, usedB = set(), set()
            for dd, j, k in pairs:
                if dd > tol or j in usedA or k in usedB:
                    continue
                usedA.add(j)
                usedB.add(k)
                li = line_of[(pi - 1, j)]
                lines[li].append((pi, k))
                line_of[(pi, k)] = li
        for k in range(nb):
            if (pi, k) not in line_of:
                line_of[(pi, k)] = len(lines)
                lines.append([(pi, k)])
    return lines, line_of


def fit_chain_lines(s, d, inten, dz, pieces, knot_fn, win=1.0, step=0.5, lam=5.0,
                    rounds=((0.6, False, 0.15), (1.3, True, 0.15), (0.2, False, 0.08)), lum=None):
    """Fit every lane line of a chain to the PCD paint with one penalised spline over the line's whole length.

    s, d, inten, dz: chain-frame PCD surface points sorted by s. pieces: dicts with 'stations', 'bounds_d',
    'nR', 'nL'. knot_fn(x0, x1) -> knot spacing. Returns (new bounds per piece, info per (piece, k), lines).
    """
    st_all = [np.asarray(p["stations"], float) for p in pieces]
    bd_all = [np.asarray(p["bounds_d"], float).copy() for p in pieces]
    lines, line_of = link_lines(pieces)
    s_lo, s_hi = st_all[0][0], st_all[-1][-1]
    centers = np.arange(s_lo + win / 2, s_hi - win / 2 + 1e-6, step)
    piece_of = np.full(len(centers), -1)
    for pi, st in enumerate(st_all):
        piece_of[(centers >= st[0]) & (centers <= st[-1]) & (piece_of < 0)] = pi
    win_peaks = []
    for sc in centers:
        a, b = np.searchsorted(s, sc - win / 2), np.searchsorted(s, sc + win / 2)
        win_peaks.append(window_peaks(s[a:b], d[a:b], inten[a:b], dz[a:b], None if lum is None else lum[a:b])
                         if b - a >= 10 else [])

    def blank():
        return dict(accepted=False, coverage=0.0, resid=np.nan, double=0, duty=0.0, n=0)

    info = {(pi, k): blank() for pi in range(len(pieces)) for k in range(len(bd_all[pi]))}
    supported = {}  # (pi, k) -> per-station flag: the fitted spline is backed by paint evidence there
    line_range = {li: (min(st_all[pi][0] for pi, _ in L), max(st_all[pi][-1] for pi, _ in L)) for li, L in enumerate(lines)}

    def fit_and_apply(M, members, x0, x1):
        """Fit one spline to measurements M over [x0, x1] and, if acceptable, apply it to the member boundaries."""
        if len(M) < 4:
            return None
        spl, inl = pspline(M[:, 0], M[:, 1], np.sqrt(M[:, 2]), x0, x1, knot_fn(x0, x1), lam)
        if spl is None or inl.sum() < 4:
            return None
        r = M[inl, 1] - spl(M[inl, 0])
        resid = float(np.median(np.abs(r)) * 1.4826)
        cov = inl.sum() * step / max(x1 - x0, step)
        span = M[inl, 0].max() - M[inl, 0].min()
        # the smooth line does not follow the paint scatter (about 2 cm), so 3.5 cm robust residual is the limit
        ok = bool(resid <= 0.035 and inl.sum() >= 8 and span >= 6.0 and cov >= 0.08)
        res = dict(accepted=ok, coverage=float(cov), resid=resid, double=int(M[inl, 3].sum()),
                   duty=float(inl.sum() * step / max(span, step)), n=int(inl.sum()))
        if not ok:
            return res
        ms = np.sort(M[inl, 0])
        for pi, k in members:
            st = st_all[pi]
            delta = spl(st) - bd_all[pi][k]
            gap = np.min(np.abs(st[:, None] - ms[None, :]), axis=1)
            # stations with paint evidence within 6 m take the spline; elsewhere (long gaps, beyond the first /
            # last measurement) the line keeps its current shape, shifted by the correction interpolated between
            # the supported stations either side, so the hand-over is continuous
            sup = (st >= ms[0]) & (st <= ms[-1]) & (gap <= 6.0)
            if sup.any():
                delta = np.interp(st, st[sup], delta[sup])
            else:
                cur_at = np.interp(ms, st, bd_all[pi][k]) if len(st) > 1 else np.full(len(ms), bd_all[pi][k][0])
                delta = np.interp(st, ms, spl(ms) - cur_at)
            bd_all[pi][k] = bd_all[pi][k] + delta
            supported[(pi, k)] = sup
        return res

    for search, only_unacc, min_cov in rounds:
        meas = {li: [] for li in range(len(lines))}
        acc_now = {key: v["accepted"] for key, v in info.items()}
        for sc, peaks, pi in zip(centers, win_peaks, piece_of):
            if not peaks or pi < 0:
                continue
            P = np.array(peaks)
            st, bd = st_all[pi], bd_all[pi]
            nb = len(bd)
            bk = np.array([np.interp(sc, st, bd[k]) for k in range(nb)])
            if only_unacc:
                owned = np.zeros(len(P), bool)
                for k in range(nb):
                    if acc_now[(pi, k)]:
                        owned |= np.abs(P[:, 0] - bk[k]) < 0.15
                P = P[~owned]
                if not len(P):
                    continue
            near_k = np.argmin(np.abs(P[:, 0][:, None] - bk[None, :]), axis=1)
            centre_k = pieces[pi]["nR"] if (pieces[pi]["nR"] > 0 and pieces[pi]["nL"] > 0) else -1
            for k in range(nb):
                if only_unacc and acc_now[(pi, k)]:
                    continue
                sel = np.flatnonzero((near_k == k) & (np.abs(P[:, 0] - bk[k]) < search))
                if not len(sel):
                    continue
                li = line_of[(pi, k)]
                if k == centre_k and len(sel) >= 2:
                    two = sel[np.argsort(-P[sel, 1])[:2]]
                    sep = abs(P[two[0], 0] - P[two[1], 0])
                    if 0.15 <= sep <= 1.3 and P[two[1], 1] > 0.4 * P[two[0], 1]:
                        meas[li].append((sc, float(P[two, 0].mean()), float(P[two, 1].sum()), 1, pi))
                        continue
                j = sel[np.argmin(np.abs(P[sel, 0] - bk[k]))]
                meas[li].append((sc, float(P[j, 0]), float(P[j, 1]), 0, pi))
        for li, L in enumerate(lines):
            members = [(pi, k) for pi, k in L if not (only_unacc and acc_now[(pi, k)])]
            if not members:
                continue
            M = np.array(meas[li]) if meas[li] else np.zeros((0, 5))
            x0, x1 = line_range[li]
            res = fit_and_apply(M, members, x0, x1)
            if res is not None and res["accepted"]:
                for key in members:
                    info[key] = dict(res)
                continue
            # the whole line is not one consistent paint line: fit its pieces separately
            for pi, k in members:
                Mp = M[M[:, 4] == pi] if len(M) else M
                st = st_all[pi]
                resp = fit_and_apply(Mp, [(pi, k)], st[0], st[-1])
                if resp is not None and resp["accepted"]:
                    info[(pi, k)] = resp
                elif not only_unacc and not (search < 0.3 and acc_now[(pi, k)]):
                    info[(pi, k)] = resp if resp is not None else blank()
    per_bound = dict(info)
    for key, v in per_bound.items():
        v["supported"] = supported[key].tolist() if (v["accepted"] and key in supported) else None
    return bd_all, per_bound, lines


def simplify_indices(xy, tol=0.01, max_gap=5.0):
    """Douglas-Peucker vertex selection on a polyline, keeping at least one vertex every max_gap metres."""
    xy = np.asarray(xy, float)
    n = len(xy)
    if n <= 2:
        return np.arange(n)
    keep = np.zeros(n, bool)
    keep[0] = keep[-1] = True
    stack = [(0, n - 1)]
    while stack:
        a, b = stack.pop()
        if b - a < 2:
            continue
        pa, pb = xy[a], xy[b]
        v = pb - pa
        L = np.linalg.norm(v)
        if L < 1e-9:
            dist = np.linalg.norm(xy[a + 1:b] - pa, axis=1)
        else:
            u = v / L
            rel = xy[a + 1:b] - pa
            dist = np.abs(u[0] * rel[:, 1] - u[1] * rel[:, 0])
        j = int(np.argmax(dist)) + a + 1
        if dist[j - a - 1] > tol or L > max_gap:
            keep[j] = True
            stack.append((a, j))
            stack.append((j, b))
    return np.flatnonzero(keep)
