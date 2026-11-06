#!/usr/bin/env python3
"""Side files of the delivery: refined base_link trajectory (10 Hz CSV), calibration summary, Autoware projector.

usage: export_extras.py CACHE_DIR OUT_DIR
"""
import json
import os
import sys

import numpy as np
from pyproj import Transformer
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import CAMERAS, FRAME, LAT0, LON0, OX, OY, UNDULATION, UTM_EPSG
from maplib import EXTRINSIC_TF, Trajectory

C, OUT = sys.argv[1:3]
tr = Trajectory(os.path.join(C, "traj_refined_1khz.npz"))
lidar = json.load(open(os.path.join(C, "calib_refined.json")))
_cc = os.path.join(C, "cam_calib.json")
cams = json.load(open(_cc))["cameras"] if os.path.exists(_cc) else []
lever = np.load(os.path.join(C, "gnss_epochs.npz"))["lever_arm"]

# heading offset of the INS body frame: course over ground minus INS yaw on straight driving above 5 m/s
p, q = tr.pos[::100], tr.quat[::100]
v = np.gradient(p[:, :2], 0.1, axis=0)
yaw = np.unwrap(R.from_quat(q).as_euler("ZYX")[:, 0])
m = (np.linalg.norm(v, axis=1) > 5) & (np.abs(np.degrees(np.gradient(yaw, 0.1))) < 0.5)
YAW_OFF = float(np.median(np.degrees(np.angle(np.exp(1j * (np.arctan2(v[:, 1], v[:, 0]) - yaw))))[m]))
Rv = R.from_euler("z", YAW_OFF, degrees=True)  # vehicle frame = INS frame rotated by YAW_OFF about z
V_T_B = np.eye(4)
V_T_B[:3, :3] = Rv.inv().as_matrix()  # coordinates in the INS frame -> vehicle frame

# trajectory CSV (10 Hz): t, map x y z, lat lon, ellipsoidal h, vehicle-frame quaternion, INS yaw, vehicle yaw
t = tr.t0 + np.arange(0, tr.N, 100) * tr.dt
qv = (R.from_quat(q) * Rv).as_quat()
to_ll = Transformer.from_crs(UTM_EPSG, "EPSG:4326", always_xy=True)
lon, lat = to_ll.transform(p[:, 0] + OX, p[:, 1] + OY)
with open(os.path.join(OUT, "trajectory_base_link_map_frame.csv"), "w") as f:
    f.write("# refined base_link trajectory (the NovAtel INS output point, 1.59 m behind and 1.44 m below the GNSS antenna), frame "
            f"{FRAME}; z = orthometric (ellipsoidal - {UNDULATION}); quaternion = vehicle heading (INS yaw + "
            f"{YAW_OFF:.2f} deg)\nt_unix,x,y,z,lat,lon,h_ellipsoidal,qx,qy,qz,qw,yaw_ins_deg,yaw_vehicle_deg\n")
    yv = np.degrees(R.from_quat(qv).as_euler("ZYX")[:, 0])
    yi = np.degrees(R.from_quat(q).as_euler("ZYX")[:, 0])
    for k in range(len(t)):
        f.write(f"{t[k]:.3f},{p[k,0]:.3f},{p[k,1]:.3f},{p[k,2]:.3f},{lat[k]:.9f},{lon[k]:.9f},{p[k,2]+UNDULATION:.3f},"
                f"{qv[k,0]:.6f},{qv[k,1]:.6f},{qv[k,2]:.6f},{qv[k,3]:.6f},{yi[k]:.3f},{yv[k]:.3f}\n")

E_B = np.array(lidar["extrinsic_base_to_lidar"])
corr = R.from_matrix(E_B[:3, :3] @ EXTRINSIC_TF[:3, :3].T).as_euler("ZYX", degrees=True)
E_V = V_T_B @ E_B
corr_v = R.from_matrix(E_V[:3, :3] @ EXTRINSIC_TF[:3, :3].T).as_euler("ZYX", degrees=True)
rnd = lambda M: np.round(np.asarray(M), 6).tolist()
out = {
    "frame": f"{FRAME}: UTM33N - ({OX:.0f}, {OY:.0f}), z = ellipsoidal - {UNDULATION} m",
    "ins_heading_offset_deg": round(YAW_OFF, 3),
    "ins_heading_offset_note": "course over ground minus NovAtel INS yaw on straight driving (> 5 m/s): the INS body "
                               "frame (what the INS odometry reports as base_link) is rotated by this much about z "
                               "against the vehicle's driving axis. Matrices below are given in both frames.",
    "gnss_antenna_in_base_link_ins_frame_m": rnd(lever),
    "lidar_os_center": {
        "base_link_ins_frame_T_os_center": rnd(E_B),
        "base_link_vehicle_frame_T_os_center": rnd(E_V),
        "correction_vs_tf_static_deg_ins_frame (yaw, pitch, roll)": rnd(corr),
        "correction_vs_tf_static_deg_vehicle_frame (yaw, pitch, roll)": rnd(corr_v),
        "stamp_offset_ms (true time = header stamp + offset)": round(lidar["dt_lidar_ms"], 2),
    },
}
for cc in cams:
    T_B = np.array(cc["base_T_cam"])
    cfg = CAMERAS[cc["index"]]
    hs = cc.get("heldout_spread")
    out[cc["name"]] = {
        "frame": f"{cfg.get('frame')} ({cfg['image_topic']})",
        "model": ("pinhole with the camera_info P matrix (rectified image)" if cc["model"] == "pinhole"
                  else "plumb_bob with camera_info K and D"),
        "base_link_ins_frame_T_camera_optical": rnd(T_B),
        "base_link_vehicle_frame_T_camera_optical": rnd(V_T_B @ T_B),
        "stamp_offset_ms (true capture time = header stamp + offset)": round(cc["dt_cam_ms"], 1),
        "note": (f"rotation and time offset by photo-consistency (held-out brightness spread {hs:.3f}, start "
                 f"{cc.get('heldout_spread_start', float('nan')):.3f}; start: {cc['start']})") if hs is not None
        else f"not calibrated (too little motion); start: {cc['start']}"}
json.dump(out, open(os.path.join(OUT, "calibration.json"), "w"), indent=2)
with open(os.path.join(OUT, "map_projector_info.yaml"), "w") as f:
    f.write(f"""# Autoware map projector for lanelet2_map.osm and pointcloud_map*.pcd (both in the same frame).
# Local frame "{FRAME}": x = UTM33N easting - {OX:.0f}, y = UTM33N northing - {OY:.0f} (WGS84). LocalCartesianUTM with
# this map_origin reproduces x/y exactly (lanelet2 UtmProjector round trip < 1 mm).
# z = NovAtel orthometric height = WGS84 ellipsoidal height - {UNDULATION:.2f} m (receiver geoid undulation, constant
# over this area); the lanelet2 "ele" tags and the PCD z use the same convention.
projector_type: LocalCartesianUTM
vertical_datum: EGM2008
map_origin:
  latitude: {LAT0}
  longitude: {LON0}
  altitude: 0.0
""")
print(f"INS heading offset {YAW_OFF:.2f} deg; lidar correction vs tf_static: INS frame {np.round(corr, 3)}, "
      f"vehicle frame {np.round(corr_v, 3)}; wrote trajectory ({len(t)} rows), calibration.json, map_projector_info.yaml")
