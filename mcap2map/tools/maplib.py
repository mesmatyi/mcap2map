"""Shared helpers: scan cache access (memory-mapped slices from extract.py), trajectory lookup, deskew."""
import os

import numpy as np
from scipy.spatial.transform import Rotation as R

from config import LIDAR

# base_link -> lidar from /tf_static (detected by the inspect stage)
EXTRINSIC_TF = np.array(LIDAR["base_T_lidar_tf"], float)


def make_T(rotvec=(0, 0, 0), trans=(0, 0, 0)):
    T = np.eye(4)
    T[:3, :3] = R.from_rotvec(rotvec).as_matrix()
    T[:3, 3] = trans
    return T


class ScanCache:
    """Random access to lidar scans: EXTRACT_DIR/scans_<k>.<field>.bin memory maps + scan_index.npy."""
    FIELDS = {"xyz": (np.float32, 3), "inten": (np.uint16, 1), "t": (np.uint32, 1), "refl": (np.uint16, 1),
              "ring": (np.uint8, 1)}

    def __init__(self, extract_dir):
        idx = np.load(os.path.join(extract_dir, "scan_index.npy"))
        self.slice = idx[:, 0].astype(np.int64)
        self.off = idx[:, 1].astype(np.int64)
        self.n = idx[:, 2].astype(np.int64)
        self.stamp = idx[:, 3]
        self.log = idx[:, 4]
        self.maps = {}
        for k in np.unique(self.slice):
            for f, (dt, c) in self.FIELDS.items():
                m = np.memmap(os.path.join(extract_dir, f"scans_{k:02d}.{f}.bin"), dtype=dt, mode="r")
                self.maps[(int(k), f)] = m.reshape(-1, 3) if c == 3 else m

    def __len__(self):
        return len(self.stamp)

    def get(self, i):
        k, a = int(self.slice[i]), int(self.off[i])
        b = a + int(self.n[i])
        return {f: np.asarray(self.maps[(k, f)][a:b]) for f in self.FIELDS}


class Trajectory:
    """1 kHz base_link trajectory (map frame) from build_traj.py."""

    def __init__(self, path):
        z = np.load(path)
        self.t0, self.dt = float(z["t0"]), float(z["dt"])
        self.pos = z["pos"]
        self.quat = z["quat"]
        self.N = len(self.pos)

    def index(self, t):
        return np.clip(np.round((np.asarray(t) - self.t0) / self.dt).astype(np.int64), 0, self.N - 1)

    def valid(self, t):
        return (t >= self.t0) & (t <= self.t0 + (self.N - 1) * self.dt)

    def pose(self, t):
        i = int(self.index(t))
        T = np.eye(4)
        T[:3, :3] = R.from_quat(self.quat[i]).as_matrix()
        T[:3, 3] = self.pos[i]
        return T


def deskew_to_map(xyz, t_ns, stamp, traj, extrinsic, dt_lidar=0.0, ref_time=None):
    """Transform lidar points into the map frame using per-point timestamps.

    If ref_time is given, points are returned in the base_link frame at ref_time instead (deskewed scan).
    """
    tp = stamp + dt_lidar + t_ns.astype(np.float64) * 1e-9
    idx = traj.index(tp)
    uidx, inv = np.unique(idx, return_inverse=True)
    Rb = R.from_quat(traj.quat[uidx]).as_matrix()
    pb = traj.pos[uidx]
    p_base = xyz.astype(np.float64) @ extrinsic[:3, :3].T + extrinsic[:3, 3]
    p_map = np.einsum("nij,nj->ni", Rb[inv], p_base) + pb[inv]
    if ref_time is None:
        return p_map
    Tr = traj.pose(ref_time)
    return (p_map - Tr[:3, 3]) @ Tr[:3, :3]
