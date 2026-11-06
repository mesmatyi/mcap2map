#!/usr/bin/env python3
"""Parallel single pass over the bag (time slices, one process each). Topics come from the run config.

Writes into OUT_DIR:
  small_<k>.npz            GNSS/INS, IMU, TF, vehicle topics of slice k
  scans_<k>.{xyz,inten,t,refl,ring}.bin   raw concatenated lidar point fields of slice k (valid points only)
  scans_<k>_meta.npy       per scan: header stamp, log time, n points
  images/cam<i>/<log_time_ns>.jpg       camera frames (raw encodings converted, JPEG streams copied as-is)
  images_<k>.npz           per image: cam, header stamp, log time, log time [ns] (= jpg name)
then merge.py combines the slices. Without a usable MCAP summary (truncated file) a single streaming pass is made.

usage: extract.py OUT_DIR [N_WORKERS] [IMAGE_STRIDE]
"""
import os
import sys
import time
from multiprocessing import Process

import cv2
import numpy as np
from mcap.reader import make_reader
from mcap_ros2.decoder import DecoderFactory

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import BAG, CAMERAS, TOPICS

T = TOPICS
T_IMG = {c["image_topic"]: i for i, c in enumerate(CAMERAS)}
T_INFO = {c["info_topic"]: i for i, c in enumerate(CAMERAS) if c.get("info_topic")}
ENC = {i: (c.get("kind", "raw"), c.get("encoding", "")) for i, c in enumerate(CAMERAS)}
TOPIC_LIST = [t for t in (T.get("odom"), T.get("bestutm"), T.get("fix"), T.get("imu"), T.get("tf"), T.get("tf_static"),
                          T.get("points"), T.get("speed")) if t] + list(T_IMG) + list(T_INFO)
PF_DT = {1: "i1", 2: "u1", 3: "<i2", 4: "<u2", 5: "<i4", 6: "<u4", 7: "<f4", 8: "<f8"}


def stamp(h):
    return h.stamp.sec + h.stamp.nanosec * 1e-9


class CloudParser:
    """PointCloud2 -> valid points (xyz, intensity, per-point time offset [ns], reflectivity, ring), any layout."""

    def __init__(self, msg):
        names = {f.name: f for f in msg.fields}
        self.dtype = np.dtype({"names": [f.name for f in msg.fields],
                               "formats": [PF_DT[f.datatype] for f in msg.fields],
                               "offsets": [f.offset for f in msg.fields], "itemsize": msg.point_step})
        self.has = set(names)
        self.tkey = next((k for k in ("t", "time", "timestamp", "offset_time") if k in names), None)
        self.ikey = next((k for k in ("intensity", "signal", "i") if k in names), None)
        self.rkey = next((k for k in ("reflectivity", "intensity", "i") if k in names), None)
        self.ringkey = next((k for k in ("ring", "channel", "laser_id") if k in names), None)

    def parse(self, msg, hdr_stamp):
        a = np.frombuffer(msg.data, dtype=self.dtype, count=msg.width * msg.height)
        xyz = np.column_stack([a["x"], a["y"], a["z"]]).astype(np.float32)
        ok = np.isfinite(xyz).all(axis=1) & (np.abs(xyz).sum(axis=1) > 1e-3)
        if "range" in self.has:
            ok &= a["range"] > 0
        a, xyz = a[ok], xyz[ok]
        n = len(a)
        if self.tkey is None:
            t = np.zeros(n, np.uint32)
        else:
            tv = a[self.tkey].astype(np.float64)
            if self.tkey == "t" or self.dtype[self.tkey].kind in "iu":  # nanosecond offsets (Ouster 't')
                t = tv
            elif tv.size and np.nanmedian(tv) > 1e8:  # absolute seconds
                t = (tv - hdr_stamp) * 1e9
            else:  # relative seconds
                t = tv * 1e9
            t = np.clip(t, 0, 4.0e9).astype(np.uint32)
        inten = np.clip(a[self.ikey].astype(np.float64), 0, 65535).astype(np.uint16) if self.ikey else np.zeros(n, np.uint16)
        refl = np.clip(a[self.rkey].astype(np.float64), 0, 65535).astype(np.uint16) if self.rkey else inten
        ring = np.clip(a[self.ringkey].astype(np.int64), 0, 255).astype(np.uint8) if self.ringkey else np.zeros(n, np.uint8)
        return xyz, inten, t, refl, ring


def to_bgr(ros, cam):
    kind, enc = ENC[cam]
    if kind == "compressed":
        return None  # written as-is
    e = ros.encoding.lower()
    buf = np.frombuffer(ros.data, np.uint8)
    if e in ("yuv422", "uyvy"):
        return cv2.cvtColor(buf.reshape(ros.height, ros.width, 2), cv2.COLOR_YUV2BGR_UYVY)
    if e in ("yuv422_yuy2", "yuyv"):
        return cv2.cvtColor(buf.reshape(ros.height, ros.width, 2), cv2.COLOR_YUV2BGR_YUY2)
    if e == "bgr8":
        return buf.reshape(ros.height, ros.width, 3)
    if e == "rgb8":
        return cv2.cvtColor(buf.reshape(ros.height, ros.width, 3), cv2.COLOR_RGB2BGR)
    if e == "bgra8":
        return cv2.cvtColor(buf.reshape(ros.height, ros.width, 4), cv2.COLOR_BGRA2BGR)
    if e == "rgba8":
        return cv2.cvtColor(buf.reshape(ros.height, ros.width, 4), cv2.COLOR_RGBA2BGR)
    if e == "mono8":
        return cv2.cvtColor(buf.reshape(ros.height, ros.width), cv2.COLOR_GRAY2BGR)
    bay = {"bayer_rggb8": cv2.COLOR_BayerBG2BGR, "bayer_bggr8": cv2.COLOR_BayerRG2BGR,
           "bayer_gbrg8": cv2.COLOR_BayerGR2BGR, "bayer_grbg8": cv2.COLOR_BayerGB2BGR}
    if e in bay:
        return cv2.cvtColor(buf.reshape(ros.height, ros.width), bay[e])
    raise ValueError(f"unsupported image encoding {ros.encoding}")


def worker(out, k, t_a, t_b, stride, streaming=False):
    t0 = time.time()
    odom, utm, fix, imu, tf, tfs, speed, info, images, scan_meta = ([] for _ in range(10))
    pairs = {}
    files = {f: open(os.path.join(out, f"scans_{k:02d}.{f}.bin"), "wb") for f in ("xyz", "inten", "t", "refl", "ring")}
    n_img, img_count, parser = 0, {}, None
    with open(BAG, "rb") as fh:
        reader = make_reader(fh, decoder_factories=[DecoderFactory()])
        it = reader.iter_decoded_messages(topics=TOPIC_LIST) if streaming else \
            reader.iter_decoded_messages(topics=TOPIC_LIST, start_time=t_a, end_time=t_b)
        for _s, ch, msg, ros in it:
            topic, lt = ch.topic, msg.log_time * 1e-9
            try:
                if topic == T.get("points"):
                    if parser is None:
                        parser = CloudParser(ros)
                    xyz, inten, t, refl, ring = parser.parse(ros, stamp(ros.header))
                    for f_, a_ in (("xyz", xyz), ("inten", inten), ("t", t), ("refl", refl), ("ring", ring)):
                        files[f_].write(a_.tobytes())
                    scan_meta.append((stamp(ros.header), lt, len(xyz)))
                elif topic in T_IMG:
                    cam = T_IMG[topic]
                    img_count[cam] = img_count.get(cam, -1) + 1
                    if img_count[cam] % stride:
                        continue
                    path = os.path.join(out, "images", f"cam{cam}", f"{msg.log_time}.jpg")
                    bgr = to_bgr(ros, cam)
                    if bgr is None:
                        fmt = ros.format.lower()
                        if "jpeg" in fmt or "jpg" in fmt:
                            open(path, "wb").write(bytes(ros.data))
                        else:
                            cv2.imwrite(path, cv2.imdecode(np.frombuffer(ros.data, np.uint8), cv2.IMREAD_COLOR),
                                        [cv2.IMWRITE_JPEG_QUALITY, 92])
                    else:
                        cv2.imwrite(path, bgr, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    images.append((cam, stamp(ros.header), lt, msg.log_time))
                    n_img += 1
                elif topic == T.get("odom"):
                    p, q = ros.pose.pose.position, ros.pose.pose.orientation
                    tw = ros.twist.twist
                    c = ros.pose.covariance
                    odom.append((stamp(ros.header), lt, p.x, p.y, p.z, q.x, q.y, q.z, q.w,
                                 tw.linear.x, tw.linear.y, tw.linear.z, tw.angular.x, tw.angular.y, tw.angular.z,
                                 c[0], c[7], c[14], c[21], c[28], c[35]))
                elif topic == T.get("bestutm"):
                    h = ros.nov_header
                    gps_t = 315964800 + h.gps_week_number * 604800 + h.gps_week_milliseconds * 1e-3 - 18
                    utm.append((stamp(ros.header), lt, gps_t, ros.pos_type.type, ros.sol_status.status,
                                ros.easting, ros.northing, ros.height, ros.undulation,
                                ros.easting_stddev, ros.northing_stddev, ros.height_stddev,
                                ros.num_svs, ros.num_sol_svs, ros.diff_age, ros.lon_zone_number, h.time_status))
                elif topic == T.get("fix"):
                    c = ros.position_covariance
                    fix.append((stamp(ros.header), lt, ros.latitude, ros.longitude, ros.altitude,
                                ros.status.status, c[0], c[4], c[8]))
                elif topic == T.get("imu"):
                    q, g, a = ros.orientation, ros.angular_velocity, ros.linear_acceleration
                    imu.append((stamp(ros.header), lt, q.x, q.y, q.z, q.w, g.x, g.y, g.z, a.x, a.y, a.z))
                elif topic in (T.get("tf"), T.get("tf_static")):
                    for t_ in ros.transforms:
                        key = (t_.header.frame_id, t_.child_frame_id)
                        pid = pairs.setdefault(key, len(pairs))
                        tr, q = t_.transform.translation, t_.transform.rotation
                        (tf if topic == T.get("tf") else tfs).append(
                            (stamp(t_.header), lt, pid, tr.x, tr.y, tr.z, q.x, q.y, q.z, q.w))
                elif topic == T.get("speed"):
                    v = getattr(ros, "vehicle_speed", None)
                    if v is None and hasattr(ros, "twist"):
                        v = ros.twist.linear.x
                    speed.append((stamp(ros.header), lt, float(v or 0.0), 1.0))
                elif topic in T_INFO:
                    info.append((T_INFO[topic], stamp(ros.header), lt))
            except Exception as e:  # one bad message must not end the slice
                print(f"[{k}] skipped a message on {topic}: {e}", flush=True)
            if topic == T.get("points") and len(scan_meta) % 200 == 1:
                print(f"[{k}] {len(scan_meta)} scans {n_img} images {time.time() - t0:.0f}s", flush=True)
    for f in files.values():
        f.close()
    np.save(os.path.join(out, f"scans_{k:02d}_meta.npy"), np.array(scan_meta, dtype=np.float64).reshape(-1, 3))
    im = np.array(images, dtype=object).reshape(-1, 4)
    np.savez(os.path.join(out, f"images_{k:02d}.npz"), cam=im[:, 0].astype(np.int8), stamp=im[:, 1].astype(np.float64),
             log=im[:, 2].astype(np.float64), log_ns=im[:, 3].astype(np.int64))
    names = [None] * len(pairs)
    for (p, c), i in pairs.items():
        names[i] = f"{p}|{c}"
    arr = lambda x, n: np.array(x, dtype=np.float64).reshape(-1, n) if x else np.zeros((0, n))
    np.savez(os.path.join(out, f"small_{k:02d}.npz"), odom=arr(odom, 21), utm=arr(utm, 17), fix=arr(fix, 9),
             imu=arr(imu, 12), tf=arr(tf, 10), tfs=arr(tfs, 10), speed=arr(speed, 4), steer=np.zeros((0, 5)),
             vs=np.zeros((0, 4)), info=arr(info, 3), pair_names=np.array(names))
    print(f"[{k}] done: {len(scan_meta)} scans, {n_img} images, {time.time() - t0:.0f}s", flush=True)


if __name__ == "__main__":
    out = sys.argv[1]
    nw = int(sys.argv[2]) if len(sys.argv) > 2 else 12
    stride = int(sys.argv[3]) if len(sys.argv) > 3 else 1
    for i in range(len(CAMERAS)):
        os.makedirs(os.path.join(out, "images", f"cam{i}"), exist_ok=True)
    try:
        with open(BAG, "rb") as fh:
            st = make_reader(fh).get_summary().statistics
        streaming = False
    except Exception as e:
        print(f"no usable MCAP summary ({e}); falling back to one streaming pass", flush=True)
        streaming, nw = True, 1
    if streaming:
        worker(out, 0, 0, 0, stride, streaming=True)
        sys.exit(0)
    edges = np.linspace(st.message_start_time, st.message_end_time + 1, nw + 1).astype(np.int64)
    procs = [Process(target=worker, args=(out, k, int(edges[k]), int(edges[k + 1]), stride)) for k in range(nw)]
    for p in procs:
        p.start()
    for p in procs:
        p.join()
    bad = [k for k, p in enumerate(procs) if p.exitcode != 0]
    print("FAILED slices:" if bad else "all slices ok", bad, flush=True)
    sys.exit(1 if bad else 0)
