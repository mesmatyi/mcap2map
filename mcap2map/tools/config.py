"""Run configuration shared by every tool, loaded from the JSON file named by $MCAP2MAP_RUN.

The pipeline (`python -m mcap2map run ...`) writes OUT/run.json: detected topics, sensor frames and extrinsics
from /tf_static, camera intrinsics, the local map frame (chosen after extraction), vehicle-profile priors and
parameters. Tools only read it. Frame values are None until the 'frame' stage has run.
"""
import json
import os

_PATH = os.environ.get("MCAP2MAP_RUN")
if not _PATH or not os.path.exists(_PATH):
    raise RuntimeError("MCAP2MAP_RUN must name the run's run.json (tools are started by the mcap2map pipeline)")
RUN = json.load(open(_PATH))
_F = RUN.get("frame") or {}
OX, OY = _F.get("ox"), _F.get("oy")
UNDULATION = _F.get("undulation")
FRAME = _F.get("name")
UTM_EPSG = _F.get("utm_epsg")
LAT0, LON0 = _F.get("lat0"), _F.get("lon0")
BAG = RUN["bag"]
TOPICS = RUN["topics"]
LIDAR = RUN["lidar"]
CAMERAS = RUN.get("cameras", [])
GROUND_HEIGHT = RUN.get("ground_height", 0.37)
PRIOR = RUN.get("prior") or {}
PARAMS = RUN.get("params") or {}
INS_HEADING_OFFSET_DEG = RUN.get("ins_heading_offset_deg", 0.0)


def param(name, default):
    return PARAMS.get(name, default)
