"""Geometry helpers for lane extraction: reference lines, straightened (s, d) raster sampling."""
import numpy as np
from scipy.ndimage import gaussian_filter1d


def resample_polyline(xy, step):
    xy = np.asarray(xy, float)
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    keep = np.r_[True, seg > 1e-6]
    xy, s = xy[keep], s[keep]
    n = max(2, int(np.ceil(s[-1] / step)) + 1)
    si = np.linspace(0, s[-1], n)
    return np.column_stack([np.interp(si, s, xy[:, 0]), np.interp(si, s, xy[:, 1])]), si


def smooth_polyline(xy, sigma_m, step=0.5, closed=False):
    """Gaussian-smooth a polyline (resampled to `step`); endpoints are kept fixed for open lines."""
    p, _ = resample_polyline(xy, step)
    if len(p) < 5:
        return p
    sg = sigma_m / step
    mode = "wrap" if closed else "nearest"
    q = np.column_stack([gaussian_filter1d(p[:, 0], sg, mode=mode), gaussian_filter1d(p[:, 1], sg, mode=mode)])
    if not closed:
        # blend back to the original near the ends so the end points stay on the OSM nodes
        n = len(p)
        w = np.clip(np.minimum(np.arange(n), np.arange(n)[::-1]) * step / (2 * sigma_m), 0, 1)[:, None]
        q = w * q + (1 - w) * p
    return q


class RefLine:
    """Arc-length parameterised reference polyline with normals (d > 0 is to the left)."""

    def __init__(self, xy, step=0.2):
        self.xy, self.s = resample_polyline(xy, step)
        t = np.gradient(self.xy, axis=0)
        self.t = t / np.maximum(np.linalg.norm(t, axis=1), 1e-9)[:, None]
        self.n = np.column_stack([-self.t[:, 1], self.t[:, 0]])
        self.length = self.s[-1]

    def at(self, s, d=0.0):
        s = np.asarray(s, float)
        x = np.interp(s, self.s, self.xy[:, 0])
        y = np.interp(s, self.s, self.xy[:, 1])
        nx = np.interp(s, self.s, self.n[:, 0])
        ny = np.interp(s, self.s, self.n[:, 1])
        nn = np.maximum(np.hypot(nx, ny), 1e-9)
        return np.column_stack([x + d * nx / nn, y + d * ny / nn])

    def project(self, pts):
        """(s, d) of points: nearest reference sample (fine enough at 0.2 m spacing)."""
        from scipy.spatial import cKDTree
        kd = cKDTree(self.xy)
        _, i = kd.query(pts)
        v = pts - self.xy[i]
        return self.s[i] + np.sum(v * self.t[i], axis=1), np.sum(v * self.n[i], axis=1)


class Raster:
    def __init__(self, path):
        z = np.load(path)
        self.x0, self.y0, self.res = float(z["x0"]), float(z["y0"]), float(z["res"])
        self.refl, self.ground, self.count = z["refl"], z["ground"], z["count"]
        self.H, self.W = self.refl.shape

    def sample(self, arr, xy):
        ix = np.floor((xy[..., 0] - self.x0) / self.res).astype(np.int64)
        iy = np.floor((xy[..., 1] - self.y0) / self.res).astype(np.int64)
        ok = (ix >= 0) & (ix < self.W) & (iy >= 0) & (iy < self.H)
        out = np.full(ix.shape, np.nan, np.float32)
        out[ok] = arr[iy[ok], ix[ok]]
        return out

    def straighten(self, ref, d_max=18.0, ds=0.2, dd=0.1, arr=None):
        """Image I[s_idx, d_idx] of `arr` (default reflectivity) along the reference line."""
        s = np.arange(0, ref.length, ds)
        d = np.arange(-d_max, d_max + 1e-9, dd)
        base = ref.at(s)
        nrm = np.column_stack([np.interp(s, ref.s, ref.n[:, 0]), np.interp(s, ref.s, ref.n[:, 1])])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1), 1e-9)[:, None]
        xy = base[:, None, :] + d[None, :, None] * nrm[:, None, :]
        return s, d, self.sample(self.refl if arr is None else arr, xy)


def fine_centerline(ll, step=0.5):
    """Centre line of a lanelet as the midpoints of its two bounds resampled to the same number of points (what
    Autoware's generateFineCenterline does); smooth also where the bounds have very different vertex spacing,
    unlike the lanelet2 default centerline."""
    L = np.array([[p.x, p.y] for p in ll.leftBound])
    R = np.array([[p.x, p.y] for p in ll.rightBound])

    def length(P):
        return float(np.sum(np.linalg.norm(np.diff(P, axis=0), axis=1))) if len(P) > 1 else 0.0

    n = max(2, int(max(length(L), length(R)) / step) + 1)

    def res(P):
        if len(P) < 2:
            return np.repeat(P[:1], n, axis=0)
        seg = np.linalg.norm(np.diff(P, axis=0), axis=1)
        s = np.r_[0, np.cumsum(seg)]
        if s[-1] < 1e-9:
            return np.repeat(P[:1], n, axis=0)
        si = np.linspace(0, s[-1], n)
        return np.column_stack([np.interp(si, s, P[:, 0]), np.interp(si, s, P[:, 1])])

    return 0.5 * (res(L) + res(R))
