#!/usr/bin/env python3
"""Download OpenStreetMap road data around the route (Overpass API).

An existing download that covers the route's bounding box is reused: the output file itself, else the shared cache
~/.cache/mcap2map/osm/. Otherwise several Overpass mirrors are tried with retries and backoff (the public servers
fail now and then with 502 / timeouts).

usage: fetch_osm.py TRAJ_NPZ OUT_JSON
"""
import glob
import json
import os
import shutil
import sys
import time

import numpy as np
import requests
from pyproj import Transformer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OX, OY, UTM_EPSG

TRAJ, OUT = sys.argv[1], sys.argv[2]
MARGIN = 200.0
CACHE = os.path.expanduser("~/.cache/mcap2map/osm")
MIRRORS = ["https://overpass-api.de/api/interpreter", "https://overpass.kumi.systems/api/interpreter",
           "https://overpass.private.coffee/api/interpreter", "https://maps.mail.ru/osm/tools/overpass/api/interpreter"]
p = np.load(TRAJ)["pos"][::100]
x, y = p[:, 0], p[:, 1]
to_ll = Transformer.from_crs(UTM_EPSG, "EPSG:4326", always_xy=True)
lon0, lat0 = to_ll.transform(x.min() - MARGIN + OX, y.min() - MARGIN + OY)
lon1, lat1 = to_ll.transform(x.max() + MARGIN + OX, y.max() + MARGIN + OY)
bb = (lat0, lon0, lat1, lon1)
bbox = f"{lat0:.6f},{lon0:.6f},{lat1:.6f},{lon1:.6f}"
print("bbox (s,w,n,e):", bbox)


def covers(path):
    try:
        s, w, n, e = map(float, json.load(open(path))["bbox"].split(","))
        return s <= bb[0] + 1e-5 and w <= bb[1] + 1e-5 and n >= bb[2] - 1e-5 and e >= bb[3] - 1e-5
    except Exception:
        return False


for cand in [OUT] + sorted(glob.glob(os.path.join(CACHE, "*.json"))):
    if os.path.exists(cand) and covers(cand):
        if cand != OUT:
            shutil.copy(cand, OUT)
        print(f"reusing OSM data covering the route: {cand}")
        sys.exit(0)

query = f"""
[out:json][timeout:120];
(
  way["highway"]({bbox});
  way["area:highway"]({bbox});
  node["highway"~"traffic_signals|crossing|stop|give_way|bus_stop"]({bbox});
  node["crossing"]({bbox});
  relation["type"="restriction"]({bbox});
  relation["route"="bus"]({bbox});
);
out body;
>;
out skel qt;
"""
data = None
for attempt in range(3):
    for url in MIRRORS:
        try:
            r = requests.post(url, data={"data": query}, timeout=120,
                              headers={"User-Agent": "mcap2map-lanelet2-builder/1.0 (research)"})
            r.raise_for_status()
            data = r.json()
            break
        except Exception as e:
            print(f"attempt {attempt + 1}: {url} failed: {str(e)[:160]}", flush=True)
    if data is not None:
        break
    time.sleep(30 * (attempt + 1))
if data is None:
    sys.exit("Overpass download failed on every mirror (3 rounds); re-run the osm stage later")

data["bbox"] = bbox
json.dump(data, open(OUT, "w"))
os.makedirs(CACHE, exist_ok=True)
shutil.copy(OUT, os.path.join(CACHE, f"osm_{bbox.replace(',', '_')}.json"))
els = data["elements"]
print(f"elements: {len(els)}  nodes {sum(e['type']=='node' for e in els)}  ways {sum(e['type']=='way' for e in els)}  "
      f"relations {sum(e['type']=='relation' for e in els)}")
