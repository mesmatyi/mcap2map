"""Camera helpers: intrinsics/extrinsics from the run config, image access (files written by extract.py),
map-point projection. Cameras are indexed by their position in config.CAMERAS."""
import os

import cv2
import numpy as np
from scipy.spatial.transform import Rotation as R

from config import CAMERAS, PRIOR

NCAM = len(CAMERAS)


def size(cam):
    c = CAMERAS[cam]
    return int(c["width"]), int(c["height"])


def model(cam):
    return CAMERAS[cam].get("model", "pinhole")


def base_T_cam(cam, rot_correction=None):
    """base_link -> optical frame from /tf_static (optionally a small rotation applied in the camera frame)."""
    T = np.array(CAMERAS[cam]["base_T_cam_tf"], float)
    if rot_correction is not None:
        T[:3, :3] = T[:3, :3] @ R.from_rotvec(rot_correction).as_matrix()
    return T


def prior_T_cam(cam):
    """Vehicle-profile prior for base_link -> optical (None if the profile has none for this camera)."""
    p = (PRIOR.get("cameras") or {}).get(CAMERAS[cam]["name"])
    return (np.array(p["base_T_cam"], float), float(p.get("dt_cam_ms", 0.0))) if p else None


def project(cam, p_cam, scale=1.0, mdl=None):
    """Pixel coordinates of camera-frame points: 'pinhole' (rectified image, P) or 'plumb_bob' (K, k1, k2, p1, p2,
    k3). Returns u, v, valid."""
    c = CAMERAS[cam]
    mdl = mdl or c.get("model", "pinhole")
    W, H = size(cam)
    z = p_cam[:, 2]
    ok = z > 0.5
    x = p_cam[:, 0] / np.where(ok, z, 1)
    y = p_cam[:, 1] / np.where(ok, z, 1)
    if mdl == "plumb_bob":
        fx, fy, cx, cy = c["K"][0], c["K"][4], c["K"][2], c["K"][5]
        d = list(c.get("D", [])) + [0.0] * 5
        k1, k2, p1, p2, k3 = d[:5]
        r2 = x * x + y * y
        f = 1 + k1 * r2 + k2 * r2 * r2 + k3 * r2 ** 3
        xd = x * f + 2 * p1 * x * y + p2 * (r2 + 2 * x * x)
        yd = y * f + p1 * (r2 + 2 * y * y) + 2 * p2 * x * y
    else:
        P = c["P"]
        fx, fy, cx, cy = P[0], P[5], P[2], P[6]
        r2 = x * x + y * y
        xd, yd = x, y
    u = (fx * xd + cx) * scale
    v = (fy * yd + cy) * scale
    ok &= (r2 < 0.8) & (u >= 0) & (u < W * scale - 1) & (v >= 0) & (v < H * scale - 1)
    return u, v, ok


class Images:
    """Camera frames on disk + their stamps (image_index.npz from merge.py)."""

    def __init__(self, extract_dir):
        z = np.load(os.path.join(extract_dir, "image_index.npz"))
        self.dir = os.path.join(extract_dir, "images")
        self.cam, self.stamp, self.log_ns = z["cam"], z["stamp"], z["log_ns"]

    def of(self, cam):
        return np.flatnonzero(self.cam == cam)

    def path(self, i):
        return os.path.join(self.dir, f"cam{int(self.cam[i])}", f"{int(self.log_ns[i])}.jpg")

    def gray(self, i, reduce=4):
        flag = {1: cv2.IMREAD_GRAYSCALE, 2: cv2.IMREAD_REDUCED_GRAYSCALE_2, 4: cv2.IMREAD_REDUCED_GRAYSCALE_4,
                8: cv2.IMREAD_REDUCED_GRAYSCALE_8}[reduce]
        return cv2.imread(self.path(i), flag)

    def color(self, i, reduce=2):
        flag = {1: cv2.IMREAD_COLOR, 2: cv2.IMREAD_REDUCED_COLOR_2, 4: cv2.IMREAD_REDUCED_COLOR_4}[reduce]
        return cv2.imread(self.path(i), flag)


def read_pcd(path):
    with open(path, "rb") as f:
        hdr = b""
        while not hdr.endswith(b"DATA binary\n"):
            hdr += f.read(1)
        fields = [l for l in hdr.decode().splitlines() if l.startswith("FIELDS")][0].split()[1:]
        return np.frombuffer(f.read(), dtype="<f4").reshape(-1, len(fields))
