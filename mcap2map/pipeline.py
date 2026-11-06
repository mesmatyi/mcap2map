"""mcap2map pipeline: an MCAP bag in, point cloud map + RGB point cloud map + Lanelet2 map out, no prompts.

Every stage runs its tools (mcap2map/tools/*.py) as subprocesses with MCAP2MAP_RUN pointing at OUT/run.json,
logs to OUT/logs/<stage>.log and leaves OUT/stages/<stage>.json when it succeeds. A stage is skipped when its
marker is newer than every earlier stage's marker; re-running a stage therefore invalidates all later ones.
"""
import glob
import json
import math
import os
import shutil
import subprocess
import sys
import time

import numpy as np

from . import inspect_bag

HERE = os.path.dirname(os.path.abspath(__file__))
TOOLS = os.path.join(HERE, "tools")
STAGES = ["inspect", "extract", "frame", "trajectory", "lidar_calib", "reference", "refine", "map",
          "camera_calib", "colorize", "osm", "lanelet", "checks", "export"]
DEFAULT_PARAMS = {
    "image_stride": 1,            # keep every n-th camera frame at extraction
    "rtk_good_std_m": 0.10,       # GNSS epoch is 'good' (no lidar-odometry window) below this std, RTK fixed
    "lo_window_min_s": 5.0,       # shortest degraded-GNSS stretch that gets a lidar-odometry window
    "lo_window_merge_gap_s": 15.0,
    "refine_iterations": 4,
    "validator_ws": "/home/matyko/Downloads/baross_map/validator_ws",
    "lanelet2": True,             # set false to stop after the point cloud maps
}
FRAME_REUSE_KM = 10.0             # an existing /tf_static map frame this close to the route is reused


class StageError(RuntimeError):
    pass


def mem_avail_gb():
    try:
        for line in open("/proc/meminfo"):
            if line.startswith("MemAvailable:"):
                return int(line.split()[1]) / 1e6
    except OSError:
        pass
    return 8.0


def workers(per_worker_gb, cap):
    return max(1, min(os.cpu_count() or 4, cap, int(mem_avail_gb() / per_worker_gb)))


def deep_update(base, upd):
    for k, v in upd.items():
        if isinstance(v, dict) and isinstance(base.get(k), dict):
            deep_update(base[k], v)
        else:
            base[k] = v
    return base


class Run:
    def __init__(self, bag, out, user_cfg=None, python=None, log=print):
        self.bag = os.path.abspath(bag)
        self.out = os.path.abspath(out)
        self.user = user_cfg or {}
        self.python = python or sys.executable
        self.log = log
        self.ext = os.path.join(self.out, "extract")
        self.cache = os.path.join(self.out, "cache")
        self.work = os.path.join(self.out, "work")
        self.output = os.path.join(self.out, "output")
        self.val = os.path.join(self.output, "validation")
        self.logs = os.path.join(self.out, "logs")
        self.marks = os.path.join(self.out, "stages")
        self.run_json = os.path.join(self.out, "run.json")
        for d in (self.out, self.ext, self.cache, self.work, self.output, self.val, self.logs, self.marks):
            os.makedirs(d, exist_ok=True)
        self.stage = None

    # ---------------------------------------------------------------- config / markers / tools
    @property
    def cfg(self):
        return json.load(open(self.run_json))

    def save_cfg(self, cfg):
        tmp = self.run_json + ".tmp"
        json.dump(cfg, open(tmp, "w"), indent=1)
        os.replace(tmp, self.run_json)

    def marker(self, stage):
        return os.path.join(self.marks, f"{stage}.json")

    def done(self, stage):
        p = self.marker(stage)
        return json.load(open(p)) if os.path.exists(p) else None

    def stale(self, stage, forced):
        if stage in forced or not os.path.exists(self.marker(stage)):
            return True
        t = os.path.getmtime(self.marker(stage))
        return any(os.path.exists(self.marker(s)) and os.path.getmtime(self.marker(s)) > t
                   for s in STAGES[:STAGES.index(stage)])

    def tool(self, name, *args, env=None):
        logf = os.path.join(self.logs, f"{self.stage}.log")
        e = dict(os.environ, MCAP2MAP_RUN=self.run_json, PYTHONUNBUFFERED="1")
        e.update(env or {})
        cmd = [self.python, "-I", os.path.join(TOOLS, name)] + [str(a) for a in args]
        t0 = time.time()
        with open(logf, "a") as fh:
            fh.write(f"\n$ {' '.join(cmd)}\n")
            fh.flush()
            r = subprocess.run(cmd, stdout=fh, stderr=subprocess.STDOUT, env=e, cwd=self.out)
        if r.returncode != 0:
            tail = open(logf).read().splitlines()[-25:]
            raise StageError(f"{name} failed (exit {r.returncode}) in stage {self.stage}; log {logf}:\n" + "\n".join(tail))
        self.log(f"    {name} ok ({time.time() - t0:.0f} s)")

    def logtext(self, stage):
        p = os.path.join(self.logs, f"{stage}.log")
        return open(p).read() if os.path.exists(p) else ""

    # ---------------------------------------------------------------- driver
    def run(self, start=None, stop=None, force=(), only=None):
        names = [only] if only else STAGES[STAGES.index(start) if start else 0:(STAGES.index(stop) + 1) if stop else None]
        forced = set(force) | ({start} if start else set()) | ({only} if only else set())
        t_all = time.time()
        for st in names:
            if not self.stale(st, forced):
                self.log(f"[{st}] up to date")
                continue
            self.stage = st
            self.log(f"[{st}] running ...")
            open(os.path.join(self.logs, f"{st}.log"), "w").close()
            t0 = time.time()
            metrics = getattr(self, f"st_{st}")() or {}
            json.dump({"stage": st, "finished": time.strftime("%Y-%m-%d %H:%M:%S"), "seconds": round(time.time() - t0, 1),
                       "metrics": metrics}, open(self.marker(st), "w"), indent=1)
            self.log(f"[{st}] done in {time.time() - t0:.0f} s" + (f" {metrics}" if metrics and len(str(metrics)) < 300 else ""))
        self.log(f"pipeline finished in {(time.time() - t_all) / 60:.1f} min; outputs in {self.output}")

    # ---------------------------------------------------------------- stages
    def st_inspect(self):
        cfg = inspect_bag.inspect(self.bag)
        cfg["params"] = dict(DEFAULT_PARAMS)
        user = dict(self.user)
        if "params" in user:
            cfg["params"].update(user.pop("params"))
        if user.get("profile") is False:
            cfg["profile"], cfg["prior"] = None, {}
            user.pop("profile")
        excl = set(user.pop("cameras_exclude", []))
        cfg["cameras"] = [c for c in cfg["cameras"] if c["name"] not in excl and c["image_topic"] not in excl]
        cfg["frame_request"] = user.pop("frame", {})
        deep_update(cfg, user)
        cfg["frame"] = None
        cfg["ins_heading_offset_deg"] = (cfg.get("prior") or {}).get("ins_heading_offset_deg", 0.0)
        cfg["run_dir"] = self.out
        self.save_cfg(cfg)
        self.log(inspect_bag.describe(cfg))
        T = cfg["topics"]
        if not (T.get("bestutm") and T.get("fix") and T.get("odom")):
            raise StageError("this bag has no NovAtel BESTUTM + NavSatFix + Odometry (see warnings above); the "
                             "trajectory stage cannot run. Set the topics in the config if they were not found.")
        return {"cameras": len(cfg["cameras"]), "profile": cfg["profile"]}

    def st_extract(self):
        cfg = self.cfg
        for f in glob.glob(os.path.join(self.ext, "scans_*")) + glob.glob(os.path.join(self.ext, "small_*")):
            os.remove(f)
        stride = max(1, int(cfg["params"]["image_stride"]))
        need = 1.2 * (sum(c["count"] / stride * c["width"] * c["height"] * 0.22 for c in cfg["cameras"]) +
                      cfg["lidar"]["rate_hz"] * cfg["duration_s"] * cfg["lidar"]["points_per_scan"] * 21 * 3)
        free = shutil.disk_usage(self.out).free
        self.log(f"    disk: ~{need / 1e9:.0f} GB needed for extraction and maps, {free / 1e9:.0f} GB free")
        if need > free:
            raise StageError(f"not enough disk space in {self.out}: ~{need/1e9:.0f} GB needed, {free/1e9:.0f} GB free "
                             "(raise params.image_stride or free space)")
        nw = workers(1.0, 12)
        self.tool("extract.py", self.ext, nw, cfg["params"]["image_stride"])
        self.tool("merge.py", self.ext)
        idx = np.load(os.path.join(self.ext, "scan_index.npy"))
        return {"scans": int(len(idx)), "workers": nw}

    def st_frame(self):
        from pyproj import Transformer
        cfg = self.cfg
        d = np.load(os.path.join(self.ext, "traj_raw.npz"))
        utm = d["utm"]
        if not len(utm):
            raise StageError("no BESTUTM messages extracted")
        E, N = utm[:, 5], utm[:, 6]
        zone = int(utm[0, 15])
        band = (cfg.get("gnss") or {}).get("utm_band", "N")
        epsg = f"EPSG:{(32600 if band >= 'N' else 32700) + zone}"
        req = cfg.get("frame_request") or {}
        name, how = req.get("name"), "configured"
        if "ox" in req and "oy" in req:
            ox, oy = float(req["ox"]), float(req["oy"])
        else:
            near = [m for m in cfg.get("map_frames", [])
                    if math.hypot(m["ox"] - np.median(E), m["oy"] - np.median(N)) < FRAME_REUSE_KM * 1000]
            if near:
                m = min(near, key=lambda m: math.hypot(m["ox"] - np.median(E), m["oy"] - np.median(N)))
                ox, oy, how = m["ox"], m["oy"], f"vehicle map frame {m['name']} from /tf_static"
                name = name or m["name"]
            else:
                ox, oy = math.floor(E.min() / 100) * 100.0, math.floor(N.min() / 100) * 100.0
                how = "new: round UTM point south-west of the route"
        name = name or f"map_{zone}{band}_{int(ox)}_{int(oy)}"
        lon0, lat0 = Transformer.from_crs(epsg, "EPSG:4326", always_xy=True).transform(ox, oy)
        und = float(np.median(utm[:, 8]))
        cfg["frame"] = {"name": name, "ox": ox, "oy": oy, "utm_epsg": epsg, "undulation": und, "lat0": lat0,
                        "lon0": lon0, "origin_choice": how}
        self.save_cfg(cfg)
        self.log(f"    frame {name}: UTM {epsg} - ({ox:.0f}, {oy:.0f}), z = ellipsoidal - {und:.2f} m ({how})")
        return {"frame": name, "ox": ox, "oy": oy}

    def st_trajectory(self):
        self.tool("build_traj.py", self.ext, self.cache)
        info = json.load(open(os.path.join(self.cache, "traj_info.json")))
        cfg = self.cfg
        if info["ins_heading_offset_samples"] > 100:
            cfg["ins_heading_offset_deg"] = info["ins_heading_offset_deg"]
        self.save_cfg(cfg)
        self.tool("scan_summary.py", self.ext, os.path.join(self.cache, "traj_init_1khz.npz"),
                  os.path.join(self.cache, "gnss_epochs.npz"), os.path.join(self.cache, "scan_summary_init.npy"))
        return {"rtk_good": round(info["rtk_good_fraction"], 3), "route_m": round(info["route_length_m"]),
                "ins_heading_offset_deg": round(cfg["ins_heading_offset_deg"], 2), "lever_arm": info["lever_arm_source"]}

    def st_lidar_calib(self):
        c = self.cache
        self.tool("calib_opt.py", self.ext, f"{c}/traj_init_1khz.npz", f"{c}/scan_summary_init.npy", f"{c}/calib_lidar.json")
        cl = json.load(open(f"{c}/calib_lidar.json"))
        return {k: round(cl[k], 3) for k in ("yaw_deg", "pitch_deg", "roll_deg", "dt_lidar_ms")}

    def st_reference(self):
        c = self.cache
        self.tool("build_ref.py", c, "--detect", f"{c}/lo_windows.json")
        wins = json.load(open(f"{c}/lo_windows.json"))
        self.tool("build_ref.py", c)  # receiver-only reference (start poses / attitude for the odometry)
        shutil.copy(f"{c}/traj_ref_1khz.npz", f"{c}/traj_ref_rx_1khz.npz")
        used = []
        if wins:
            idx = np.load(os.path.join(self.ext, "scan_index.npy"))
            dtl = json.load(open(f"{c}/calib_lidar.json"))["dt_lidar_ms"] * 1e-3
            tm = idx[:, 3] + dtl + 0.05
            for k, w in enumerate(wins):
                i0, i1 = int(np.searchsorted(tm, w["ta"] + 0.03)), int(np.searchsorted(tm, w["tb"] - 0.02))
                if i1 - i0 < 20:
                    continue
                out = f"{c}/lo_window_{k}.npz"
                self.tool("lo.py", self.ext, f"{c}/traj_ref_rx_1khz.npz", f"{c}/calib_lidar.json", out, 1.0, i0, i1)
                used.append(dict(w, lo=out, scans=[i0, i1]))
        json.dump(used, open(f"{c}/lo_windows_used.json", "w"), indent=1)
        if used:
            self.tool("build_ref.py", c, "--windows", f"{c}/lo_windows_used.json")
        return {"lo_windows": [[round(w["t_rel"][0], 1), round(w["t_rel"][1], 1)] for w in used]}

    def st_refine(self):
        c = self.cache
        self.tool("refine_traj.py", self.ext, c, f"{c}/calib_lidar.json", f"{c}/traj_refined_1khz.npz",
                  f"{c}/calib_refined.json", self.cfg["params"]["refine_iterations"])
        m = [l for l in self.logtext("refine").splitlines() if l.startswith("iteration")][-2:]
        return {"last_iteration": m}

    def st_map(self):
        c, w = self.cache, os.path.join(self.work, "map")
        shutil.rmtree(w, ignore_errors=True)
        self.tool("build_map.py", self.ext, f"{c}/calib_refined.json", w, f"{c}/traj_refined_1khz.npz")
        self.tool("rasters.py", f"{w}/tiles", f"{c}/traj_refined_1khz.npz", f"{c}/rasters_v1.npz")
        self.tool("dynamic_filter.py", self.ext, f"{c}/calib_refined.json", f"{c}/traj_refined_1khz.npz", w,
                  f"{c}/rasters_v1.npz")
        shutil.rmtree(os.path.join(self.output, "pointcloud_map_tiles"), ignore_errors=True)
        self.tool("export_pcd.py", w, self.output)
        shutil.rmtree(f"{w}/raw", ignore_errors=True)  # per-scan tile shards: large, no longer needed
        return {}

    def st_camera_calib(self):
        if not self.cfg["cameras"]:
            self.log("    no cameras: skipped")
            return {"skipped": "no cameras"}
        c = self.cache
        self.tool("cam_calib.py", self.ext, f"{self.output}/pointcloud_map_0.2m.pcd", f"{c}/traj_refined_1khz.npz",
                  f"{c}/cam_calib.json")
        cams = json.load(open(f"{c}/cam_calib.json"))["cameras"]
        return {cc["name"]: {"dt_ms": round(cc["dt_cam_ms"]), "model": cc["model"]} for cc in cams}

    def st_colorize(self):
        if not self.cfg["cameras"]:
            return {"skipped": "no cameras"}
        c, o = self.cache, self.output
        nw = workers(1.6, 8)
        self.tool("colorize.py", self.ext, f"{o}/pointcloud_map.pcd", f"{c}/traj_refined_1khz.npz",
                  f"{c}/calib_refined.json", f"{c}/cam_calib.json", f"{o}/pointcloud_map_rgb.pcd", nw)
        if os.path.exists(f"{o}/pointcloud_map_rgb_seen.npy"):
            shutil.move(f"{o}/pointcloud_map_rgb_seen.npy", f"{self.work}/pointcloud_map_rgb_seen.npy")
        self.tool("rgb_down.py", f"{o}/pointcloud_map_rgb.pcd", f"{o}/pointcloud_map_rgb_0.2m.pcd", 0.2)
        return {"workers": nw}

    def st_osm(self):
        p = self.cfg["params"]
        if not p.get("lanelet2", True):
            return {"skipped": "lanelet2 disabled"}
        c = self.cache
        os.makedirs(f"{c}/osm", exist_ok=True)
        raw = f"{c}/osm/osm_raw.json"
        self.tool("fetch_osm.py", f"{c}/traj_refined_1khz.npz", raw)  # reuses a covering download if there is one
        self.tool("osm_route.py", raw, f"{c}/traj_refined_1khz.npz", f"{c}/route_network.json")
        net = json.load(open(f"{c}/route_network.json"))
        return {"chains": len(net["chains"]), "junctions": len(net["junctions"])}

    def st_lanelet(self):
        if not self.cfg["params"].get("lanelet2", True):
            return {"skipped": "lanelet2 disabled"}
        c, w, o = self.cache, self.work, self.output
        self.tool("lane_extract.py", f"{c}/route_network.json", f"{c}/rasters_v1.npz", f"{c}/traj_refined_1khz.npz",
                  f"{c}/lanes.json", f"{w}/lane_debug")
        self.tool("traj_lanes.py", f"{c}/lanes.json", f"{c}/traj_refined_1khz.npz", f"{c}/lanes_traj.json")
        os.makedirs(f"{w}/fitcache", exist_ok=True)
        self.tool("build_lanelet2.py", f"{c}/lanes_traj.json", f"{c}/route_network.json", f"{c}/rasters_v1.npz",
                  f"{c}/traj_refined_1khz.npz", f"{w}/lanelet2_map.osm", f"{o}/pointcloud_map.pcd",
                  env={"FIT_CACHE": f"{w}/fitcache"})
        shutil.copy(f"{w}/lanelet2_map.osm", f"{o}/lanelet2_map.osm")
        shutil.copy(f"{w}/lanelet2_map_build_info.json", f"{o}/lanelet2_map_build_info.json")
        shutil.copy(f"{w}/lanelet2_map_mgrs_local.osm", f"{self.val}/lanelet2_map_mgrs_local_for_editor.osm")
        shutil.copy(f"{w}/lanelet2_map_validator_exclusion.json", f"{self.val}/validator_exclusion_list.json")
        last = [l for l in self.logtext("lanelet").splitlines() if l.startswith("wrote ")]
        return {"result": last[-1] if last else None}

    def st_checks(self):
        cfg = self.cfg
        if not cfg["params"].get("lanelet2", True):
            return {"skipped": "lanelet2 disabled"}
        c, o, v = self.cache, self.output, self.val
        mp = f"{o}/lanelet2_map.osm"
        metrics = {}
        ws = cfg["params"].get("validator_ws")
        if ws and os.path.exists(os.path.join(ws, "install", "setup.bash")):
            vd = f"{v}/autoware_validator"
            shutil.rmtree(vd, ignore_errors=True)
            os.makedirs(vd)
            req = os.path.join(ws, "src/autoware_lanelet2_map_validator/autoware_lanelet2_map_validator/map_requirements/"
                                   "autoware_requirement_set.json")
            F = cfg["frame"]
            cmd = (f"cd {ws} && source /opt/ros/jazzy/setup.bash && source install/setup.bash && ros2 run "
                   f"autoware_lanelet2_map_validator autoware_lanelet2_map_validator -p utm --lat {F['lat0']} --lon "
                   f"{F['lon0']} -m {mp} -i {req} -x {v}/validator_exclusion_list.json -o {vd}")
            with open(f"{vd}/validator_stdout.txt", "w") as fh:
                subprocess.run(["bash", "-c", cmd], stdout=fh, stderr=subprocess.STDOUT)
            res = f"{vd}/lanelet2_validation_results.json"
            if os.path.exists(res):
                cnt = {}
                for rq in json.load(open(res)).get("requirements", []):
                    for vv in rq.get("validators", []):
                        for iss in vv.get("issues", []):
                            key = f"{iss.get('severity')}: {vv['name']}"
                            cnt[key] = cnt.get(key, 0) + 1
                metrics["validator"] = cnt
        else:
            metrics["validator"] = "not run (no validator workspace)"
        self.tool("validate_lanelet2.py", mp, f"{c}/rasters_v1.npz", f"{c}/traj_refined_1khz.npz", v)
        self.tool("route_test.py", mp, f"{c}/traj_refined_1khz.npz", 50)
        self.tool("boundary_check.py", mp, f"{o}/pointcloud_map.pcd", f"{c}/rasters_v1.npz", env={"CHECK_WORST": "1"})
        self.tool("wobble_check.py", mp)
        self.tool("kink_scan.py", mp, 40)
        self.tool("connector_check.py", mp)
        for name, box in closeup_boxes(f"{c}/route_network.json", f"{c}/lanes_traj.json"):
            self.tool("area_render.py", mp, f"{o}/pointcloud_map.pcd", f"{c}/rasters_v1.npz", f"{c}/traj_refined_1khz.npz",
                      f"{c}/route_network.json", f"{v}/closeup_{name}.png", ",".join(f"{x:.0f}" for x in box), 0.06)
        if os.path.exists(f"{o}/pointcloud_map_rgb.pcd"):
            self.tool("render_overview.py", f"{o}/pointcloud_map_rgb.pcd", mp, v)
        return metrics

    def st_export(self):
        from . import report
        self.tool("export_extras.py", self.cache, self.output)
        report.write(self)
        return {}


def closeup_boxes(route_json, lanes_json, half=35.0, max_n=12):
    """Render areas: every roundabout and every cluster of junctions of the route network."""
    net = json.load(open(route_json))
    nxy = {int(k): v for k, v in net["node_xy"].items()}
    boxes = []
    for c in json.load(open(lanes_json))["chains"]:
        if c.get("ring"):
            xy = np.array(c["ref_xy"])
            cx, cy = xy.mean(axis=0)
            r = np.max(np.linalg.norm(xy - [cx, cy], axis=1)) + 25
            boxes.append((f"roundabout_{len(boxes)}", (cx - r, cx + r, cy - r, cy + r)))
    pts = [nxy[n] for n in net["junctions"] if n in nxy]
    used = [False] * len(pts)
    for i, p in enumerate(pts):
        if used[i]:
            continue
        grp = [j for j in range(len(pts)) if not used[j] and math.dist(pts[j], p) < 40]
        for j in grp:
            used[j] = True
        cx, cy = np.mean([pts[j] for j in grp], axis=0)
        if any(b[0] <= cx <= b[1] and b[2] <= cy <= b[3] for _, b in boxes):
            continue
        boxes.append((f"junction_{len(boxes)}", (cx - half, cx + half, cy - half, cy + half)))
    return boxes[:max_n]
