"""Auto-detection of everything the pipeline needs from an MCAP bag (no prompts):

* GNSS/INS topics (NovAtel BESTUTM + NavSatFix + Odometry, both driver namespaces), IMU, TF, vehicle speed
* the main lidar (PointCloud2 with the most points) and its fields
* cameras = image topics (raw or compressed) that have a CameraInfo in their namespace (lidar range/intensity
  images have none), intrinsics, encoding, rectified or not
* /tf_static: base frame, base -> lidar / cameras / ground extrinsics, existing local map frames (/map -> /map_*)
* a matching vehicle profile (profiles/*.json) with calibration priors
"""
import glob
import json
import os

import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory
from scipy.spatial.transform import Rotation as R

HERE = os.path.dirname(os.path.abspath(__file__))
IMG_PREF = ["image_rect_color/compressed", "image_rect_color", "image_color/compressed", "image_color",
            "image_raw/compressed", "image_raw", "image_compressed", "image"]


def norm(frame):
    return frame.lstrip("/")


def first(reader, topic):
    for _s, _c, msg, ros in reader.iter_decoded_messages(topics=[topic]):
        return msg, ros
    return None, None


def tf_matrix(tr, q):
    T = np.eye(4)
    T[:3, :3] = R.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
    T[:3, 3] = [tr.x, tr.y, tr.z]
    return T


class TfTree:
    def __init__(self, edges):
        self.parent = {}  # child -> (parent, parent_T_child)
        for p, c, T in edges:
            self.parent[norm(c)] = (norm(p), T)

    def chain(self, frame):
        out, f = [frame], frame
        while f in self.parent and len(out) < 50:
            f = self.parent[f][0]
            out.append(f)
        return out

    def to_root(self, frame):
        T, f = np.eye(4), frame
        while f in self.parent:
            p, pT = self.parent[f]
            T = pT @ T
            f = p
        return f, T

    def lookup(self, base, target):
        """base_T_target, or None if the frames are not connected."""
        rb, Tb = self.to_root(norm(base))
        rt, Tt = self.to_root(norm(target))
        return np.linalg.inv(Tb) @ Tt if rb == rt else None


def inspect(bag):
    with open(bag, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        try:
            summ = reader.get_summary()
        except Exception as e:
            raise ValueError(f"the MCAP summary/index of {bag} is unreadable ({e}); repair it first, e.g. "
                             f"`mcap recover in.mcap -o fixed.mcap`") from e
        if summ is None or summ.statistics is None:
            raise ValueError(f"{bag} has no MCAP summary; repair it first (`mcap recover`)")
        st = summ.statistics
        topics = {}
        for cid, ch in summ.channels.items():
            sch = summ.schemas.get(ch.schema_id)
            topics[ch.topic] = (sch.name if sch else "", st.channel_message_counts.get(cid, 0))
        live = {t: s for t, (s, n) in topics.items() if n > 0}
        by_type = lambda suffix: sorted(t for t, s in live.items() if s.endswith(suffix))
        ns = lambda t: t.rsplit("/", 1)[0]
        cfg = {"bag": os.path.abspath(bag), "duration_s": (st.message_end_time - st.message_start_time) / 1e9,
               "topics": {}, "warnings": []}
        T = cfg["topics"]
        # --- GNSS / INS
        utm = by_type("/BESTUTM")
        T["bestutm"] = utm[0] if utm else None
        gns = ns(T["bestutm"]) if T["bestutm"] else ""
        pick = lambda cands: (sorted(cands, key=lambda t: (not t.startswith(gns), len(t)))[0] if cands else None)
        T["fix"] = pick(by_type("sensor_msgs/msg/NavSatFix"))
        T["odom"] = pick(by_type("nav_msgs/msg/Odometry"))
        imus = by_type("sensor_msgs/msg/Imu")
        T["imu"] = sorted(imus, key=lambda t: (t != "/imu/data", "os_" in t, len(t)))[0] if imus else None
        T["tf"] = "/tf" if "/tf" in live else None
        T["tf_static"] = "/tf_static" if "/tf_static" in topics else None
        spd = by_type("VehicleSpeedRpt")
        T["speed"] = spd[0] if spd else None
        if not (T["bestutm"] and T["fix"] and T["odom"]):
            cfg["warnings"].append(f"NovAtel BESTUTM/NavSatFix/Odometry not all found (bestutm={T['bestutm']}, "
                                   f"fix={T['fix']}, odom={T['odom']}); the trajectory stage needs them")
        # --- lidar: the PointCloud2 topic with the most points per message
        clouds = []
        for t in by_type("sensor_msgs/msg/PointCloud2"):
            _m, ros = first(reader, t)
            if ros is not None:
                clouds.append((ros.width * ros.height, t, ros))
        if not clouds:
            raise ValueError("no PointCloud2 topic with messages: nothing to map")
        clouds.sort(key=lambda z: (-("center" in z[1] or "top" in z[1]), -z[0]))
        npts, T["points"], cros = clouds[0]
        cfg["lidar"] = {"topic": T["points"], "frame": norm(cros.header.frame_id), "points_per_scan": int(npts),
                        "fields": [f.name for f in cros.fields], "point_step": cros.point_step,
                        "rate_hz": topics[T["points"]][1] / max(cfg["duration_s"], 1e-6)}
        # --- tf_static
        edges = []
        if T["tf_static"]:
            for _s, _c, _m, ros in reader.iter_decoded_messages(topics=[T["tf_static"]]):
                for tr in ros.transforms:
                    edges.append((tr.header.frame_id, tr.child_frame_id,
                                  tf_matrix(tr.transform.translation, tr.transform.rotation)))
        tree = TfTree(edges)
        lchain = tree.chain(cfg["lidar"]["frame"])
        base = next((f for f in lchain if f.endswith("base_link")), lchain[1] if len(lchain) > 1 else lchain[0])
        cfg["base_frame"] = base
        Tl = tree.lookup(base, cfg["lidar"]["frame"])
        if Tl is None:
            cfg["warnings"].append(f"no /tf_static path {base} -> {cfg['lidar']['frame']}; identity extrinsic used")
            Tl = np.eye(4)
        cfg["lidar"]["base_T_lidar_tf"] = Tl.tolist()
        ground = [f for f in tree.parent if f.endswith("ground_link")]
        Tg = tree.lookup(base, ground[0]) if ground else None
        cfg["ground_height"] = float(-Tg[2, 3]) if Tg is not None else 0.37
        if Tg is None:
            cfg["warnings"].append("no ground_link in /tf_static; base_link height above ground assumed 0.37 m")
        mf = {norm(c): (float(Tm[0, 3]), float(Tm[1, 3])) for p, c, Tm in edges
              if norm(p) == "map" and norm(c).startswith("map_")}
        cfg["map_frames"] = [{"name": k, "ox": v[0], "oy": v[1]} for k, v in sorted(mf.items())]
        # --- cameras: image topics with a CameraInfo in their (parent) namespace
        infos = set(by_type("sensor_msgs/msg/CameraInfo"))
        imgs = [t for t in live if live[t].endswith("sensor_msgs/msg/Image") or live[t].endswith("CompressedImage")]
        groups = {}
        for t in imgs:
            for base_ns in (ns(t), ns(ns(t))):
                if f"{base_ns}/camera_info" in infos:
                    groups.setdefault(f"{base_ns}/camera_info", []).append(t)
                    break
        cams = []
        for info_t, cands in sorted(groups.items()):
            rank = lambda t: next((k for k, suf in enumerate(IMG_PREF) if t.endswith(suf)), len(IMG_PREF))
            t = sorted(cands, key=rank)[0]
            _m, info = first(reader, info_t)
            _m, im = first(reader, t)
            if info is None or im is None:
                continue
            comp = live[t].endswith("CompressedImage")
            frame = norm(im.header.frame_id or info.header.frame_id)
            Tc = tree.lookup(base, frame)
            if Tc is None:
                cfg["warnings"].append(f"camera {t}: no /tf_static path {base} -> {frame}; camera skipped")
                continue
            rect = "rect" in t
            d = [float(x) for x in info.d]
            cams.append({"name": frame, "image_topic": t, "info_topic": info_t, "frame": frame,
                         "kind": "compressed" if comp else "raw",
                         "encoding": (im.format if comp else im.encoding),
                         "width": int(info.width), "height": int(info.height),
                         "K": [float(x) for x in info.k], "D": d, "P": [float(x) for x in info.p],
                         "model": "pinhole" if (rect or not any(abs(x) > 1e-9 for x in d)) else "plumb_bob",
                         "base_T_cam_tf": Tc.tolist(), "count": topics[t][1]})
        cfg["cameras"] = cams
        # --- vehicle profile (calibration priors)
        cfg["profile"], cfg["prior"] = None, {}
        for pf in sorted(glob.glob(os.path.join(HERE, "profiles", "*.json"))):
            prof = json.load(open(pf))
            m = prof.get("match", {})
            if m.get("base_frame") == base and m.get("lidar_frame") == cfg["lidar"]["frame"]:
                cfg["profile"], cfg["prior"] = os.path.basename(pf), prof.get("prior", {})
                break
        if T["bestutm"]:
            _m, u = first(reader, T["bestutm"])
            cfg["gnss"] = {"undulation": float(u.undulation), "utm_zone": int(u.lon_zone_number),
                           "utm_band": chr(int(u.lat_zone_letter)) if int(u.lat_zone_letter) > 64 else "?"}
    return cfg


def describe(cfg):
    L = [f"bag: {cfg['bag']} ({cfg['duration_s']:.0f} s)"]
    for k, v in cfg["topics"].items():
        L.append(f"  {k:10s} {v}")
    li = cfg["lidar"]
    L.append(f"lidar {li['topic']} frame {li['frame']} ~{li['points_per_scan']} pts, {li['rate_hz']:.1f} Hz, fields "
             f"{','.join(li['fields'])}")
    for c in cfg["cameras"]:
        L.append(f"camera {c['name']}: {c['image_topic']} ({c['kind']} {c['encoding']}, {c['width']}x{c['height']}, "
                 f"model {c['model']}, {c['count']} frames)")
    L.append(f"base frame {cfg['base_frame']}, ground height {cfg['ground_height']:.2f} m, map frames "
             f"{[m['name'] for m in cfg['map_frames']]}, profile {cfg['profile']}")
    for w in cfg["warnings"]:
        L.append(f"WARNING: {w}")
    return "\n".join(L)
