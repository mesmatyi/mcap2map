#!/usr/bin/env python3
"""Wobble metric of Lanelet2 boundaries: heading-change noise of painted boundaries at 1 m spacing after
removing the 10 m trend, plus the maximum lateral deviation from a 20 m moving straight-line fit.

usage: wobble_check.py MAP_OSM [MAP_OSM ...]
"""
import sys

import numpy as np
import lanelet2
from lanelet2.io import Origin
from lanelet2.projection import UtmProjector
from scipy.ndimage import uniform_filter1d
import os as _os, sys as _sys
_sys.path.insert(0, _os.path.dirname(_os.path.abspath(__file__)))
from config import LAT0, LON0


def resample(xy, step):
    seg = np.linalg.norm(np.diff(xy, axis=0), axis=1)
    s = np.r_[0, np.cumsum(seg)]
    if s[-1] < step * 3:
        return None
    si = np.arange(0, s[-1], step)
    return np.column_stack([np.interp(si, s, xy[:, 0]), np.interp(si, s, xy[:, 1])])


for path in sys.argv[1:]:
    m, _e = lanelet2.io.loadRobust(path, UtmProjector(Origin(LAT0, LON0)))
    noise, maxdev, nverts, lengths = [], [], [], []
    for ls in m.lineStringLayer:
        t = ls.attributes["type"] if "type" in ls.attributes else ""
        if t not in ("line_thin", "road_border"):
            continue
        xy = np.array([[q.x, q.y] for q in ls])
        r = resample(xy, 1.0)
        if r is None or len(r) < 30:
            continue
        h = np.unwrap(np.arctan2(np.diff(r[:, 1]), np.diff(r[:, 0])))
        trend = uniform_filter1d(h, 10, mode="nearest")
        noise.append(np.degrees(np.std(h - trend)))
        # lateral deviation from a straight line fitted over sliding 20 m windows
        dev = []
        for a in range(0, len(r) - 20, 5):
            w = r[a:a + 21]
            c = w.mean(axis=0)
            u, s_, vt = np.linalg.svd(w - c)
            n = vt[1]
            dev.append(np.max(np.abs((w - c) @ n)))
        maxdev.append(np.median(dev))
        nverts.append(len(xy))
        lengths.append(len(r))
    noise, maxdev = np.array(noise), np.array(maxdev)
    print(f"{path.split('/')[-1]}: {len(noise)} boundaries >= 30 m; heading noise median {np.median(noise):.3f} deg "
          f"(p90 {np.percentile(noise, 90):.3f}); lateral dev from 20 m straight fit median {np.median(maxdev)*100:.1f} cm "
          f"(p90 {np.percentile(maxdev, 90)*100:.1f} cm); vertices per 100 m {100*np.sum(nverts)/np.sum(lengths):.0f}")
