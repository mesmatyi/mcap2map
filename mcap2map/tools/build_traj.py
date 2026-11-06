#!/usr/bin/env python3
"""Initial base_link trajectory (1 kHz, map frame) from the NovAtel topics.

* attitude : every INSPVA attitude update carried by /novatel/oem7/odom (50 Hz, exactly one per 20 ms GPS epoch);
             message stamps are receipt times with burst jitter up to ~100 ms, so sample i is re-timed to the GPS
             epoch grid g0 + 0.02 i (min latency ~5 ms).
* position : GPS-timed BESTUTM (20 Hz): RTK antenna fixes (NARROW_INT) or the INS solution transferred to the
             antenna (INS_* types), moved to base_link with the receiver's INS lever arm, which is estimated here
             from INS_* epochs (BESTUTM vs the INS base_link positions in /novatel/oem7/fix).
Also writes the per-epoch GNSS records (base_link position, std, type) used as priors by the pose graph.

Frame: x = UTM33N E - OX, y = UTM33N N - OY, z = BESTUTM height (= ellipsoidal - 45.0 m undulation).

usage: build_traj.py EXTRACT_DIR OUT_DIR
"""
import os
import sys

import numpy as np
from pyproj import Transformer
from scipy.interpolate import CubicSpline
from scipy.spatial.transform import Rotation as R, RotationSpline

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import json

from config import OX, OY, PRIOR, UNDULATION, UTM_EPSG

D, OUT = sys.argv[1], sys.argv[2]
os.makedirs(OUT, exist_ok=True)
d = np.load(os.path.join(D, "traj_raw.npz"))
odom, utm, fix = d["odom"], d["utm"], d["fix"]
if len(utm) and not np.allclose(utm[:, 8], UNDULATION, atol=0.05):
    print(f"WARNING: BESTUTM undulation varies {utm[:,8].min():.2f}..{utm[:,8].max():.2f} m; using {UNDULATION}")


def regrid(stamps, period=0.02):
    """GPS epoch times of a stream published once per epoch with variable latency (message stamps = receipt
    times). Without dropped samples sample i is g0 + period * i; with gaps every sample is put on its epoch slot
    (latency floor removed, slots kept strictly increasing)."""
    n = len(stamps)
    i = np.arange(n)
    span = (stamps[-1] - stamps[0]) / period + 1
    if abs(span - n) < 0.5:
        rel = stamps - (stamps[0] + period * i)
        g0 = np.floor((stamps[0] + np.percentile(rel, 0.5)) / period + 1e-6) * period
        tg = g0 + period * i
    else:
        frac = np.mod(stamps, period)
        lat0 = np.percentile(frac, 1)  # phase of the earliest arrivals relative to the epoch grid
        slot = np.floor((stamps - lat0) / period + 1e-6).astype(np.int64)
        for k in range(n - 2, -1, -1):  # late (burst) arrivals: a sample is at least one slot before the next
            slot[k] = min(slot[k], slot[k + 1] - 1)
        for k in range(1, n):
            slot[k] = max(slot[k], slot[k - 1] + 1)
        tg = slot * period + 0.0
        print(f"  stream with gaps: {span:.0f} slots for {n} samples, re-gridded per sample")
    return tg, stamps - tg


def continuous_quat(q):
    q = q.copy()
    for k in range(1, len(q)):
        if np.dot(q[k], q[k - 1]) < 0:
            q[k] = -q[k]
    return q


# --- attitude
q_chg = np.r_[True, np.any(np.diff(odom[:, 5:9], axis=0) != 0, axis=1)]
t_att, lat_att = regrid(odom[q_chg, 0])
q_att = continuous_quat(odom[q_chg, 5:9])
print(f"attitude: {len(t_att)} samples on the 20 ms grid, latency p1 {np.percentile(lat_att,1)*1e3:.1f} "
      f"median {np.median(lat_att)*1e3:.1f} p99 {np.percentile(lat_att,99)*1e3:.1f} ms")
rs = RotationSpline(t_att, R.from_quat(q_att))

# --- INS base_link positions from /fix (lat/lon/alt, 50 Hz, one per epoch) during INS_* BESTUTM periods
t_fix, lat_fix = regrid(fix[:, 0])
print(f"fix: latency p1 {np.percentile(lat_fix,1)*1e3:.1f} median {np.median(lat_fix)*1e3:.1f} ms")
to_utm = Transformer.from_crs("EPSG:4326", UTM_EPSG, always_xy=True)
fe, fn = to_utm.transform(fix[:, 3], fix[:, 2])
p_fix = np.column_stack([fe - OX, fn - OY, fix[:, 4] - UNDULATION])

tg = utm[:, 2]
ptype = utm[:, 3].astype(int)
pa = np.column_stack([utm[:, 5] - OX, utm[:, 6] - OY, utm[:, 7]])
_, iu = np.unique(tg, return_index=True)
tg, ptype, pa, std = tg[iu], ptype[iu], pa[iu], utm[iu, 9:12]
ins_type = ptype >= 52
# fix samples at INS epochs: the fix position must not be one of the BESTUTM antenna positions
key = lambda e, n: np.round(e * 1e3).astype(np.int64) * 10**9 + np.round(n * 1e3).astype(np.int64)
ant_keys = set(key(pa[:, 0], pa[:, 1]).tolist())
fix_is_ins = np.array([k not in ant_keys for k in key(p_fix[:, 0], p_fix[:, 1])])


def fit(tau):
    """BESTUTM(t) = p_fix(t + tau) + R(t) l on INS_* epochs (both are the same INS solution)."""
    tt = tg[ins_type] + tau
    j = np.searchsorted(t_fix, tt)
    ok = (j > 0) & (j < len(t_fix))
    j = np.clip(j, 1, len(t_fix) - 1)
    ok &= fix_is_ins[j] & fix_is_ins[j - 1] & (tt > t_att[0]) & (tt < t_att[-1])
    w = ((tt - t_fix[j - 1]) / 0.02)[:, None]
    pf = p_fix[j - 1] * (1 - w) + p_fix[j] * w
    Rm = rs(tg[ins_type][ok]).as_matrix()
    rhs = pa[ins_type][ok] - pf[ok]
    l, *_ = np.linalg.lstsq(Rm.reshape(-1, 3), rhs.reshape(-1), rcond=None)
    return l, rhs - Rm @ l, ok


taus = np.arange(-0.03, 0.0301, 0.001)
cost = [np.sqrt(np.mean(np.sum(fit(t)[1] ** 2, axis=1))) for t in taus]
tau = taus[int(np.argmin(cost))]
l, res, ok = fit(tau)
rms = float(np.sqrt(np.mean(np.sum(res ** 2, 1)))) if ok.sum() else np.inf
lever_src = "INS epochs (BESTUTM INS solution vs /fix INS base_link positions)"
if ok.sum() < 50 or rms > 0.05:
    prior_l = PRIOR.get("gnss_antenna_in_base_link")
    print(f"WARNING: lever arm from INS epochs unreliable ({ok.sum()} pairs, RMS {rms*100:.1f} cm)", flush=True)
    if prior_l is not None:
        l, lever_src = np.array(prior_l, float), "vehicle profile prior"
    else:
        l, lever_src = np.zeros(3), "none (zero): no INS epochs and no profile prior"
print(f"lever arm (antenna in base_link): ({l[0]:.4f}, {l[1]:.4f}, {l[2]:.4f}) m from {lever_src}; "
      f"{ok.sum()} INS epochs, fix time offset {tau*1e3:.0f} ms, residual RMS {rms*1000:.1f} mm")

# --- base_link positions at every BESTUTM epoch
m = (tg > t_att[0]) & (tg < t_att[-1])
tg, ptype, pa, std = tg[m], ptype[m], pa[m], std[m]
p_base = pa - rs(tg).apply(l)
ps = CubicSpline(tg, p_base)
# grid as t0 + i*dt: np.arange(t0, t1, 0.001) with t0 ~ 1.78e9 uses the float-rounded step (t0 + 0.001) - t0
# = 0.00099993 s, which drifts the time base by ~47 ms over this bag
t_grid = tg[0] + np.arange(int((tg[-1] - tg[0]) / 0.001)) * 0.001
pos = ps(t_grid)
quat = rs(t_grid).as_quat()
np.savez(os.path.join(OUT, "traj_init_1khz.npz"), t0=t_grid[0], dt=0.001, pos=pos, quat=quat, lever_arm=l)
np.savez(os.path.join(OUT, "gnss_epochs.npz"), t=tg, p_base=p_base, p_ant=pa, std=std, type=ptype, lever_arm=l)
print(f"saved 1 kHz trajectory {len(t_grid)} samples ({t_grid[-1]-t_grid[0]:.1f} s), {len(tg)} GNSS epochs")

# jumps at solution-type changes (base_link position step vs the neighbouring motion)
v = np.diff(p_base, axis=0) / np.diff(tg)[:, None]
acc = np.linalg.norm(np.diff(v, axis=0)[:, :2], axis=1) / np.diff(tg)[1:]
chg = np.flatnonzero(np.diff(ptype) != 0)
print("type changes with horizontal acceleration spike (m/s2):")
for c in chg:
    a_ = acc[max(c - 2, 0):c + 2].max() if c + 2 <= len(acc) else np.nan
    print(f"  t {tg[c+1]-tg[0]:6.1f} s  {ptype[c]} -> {ptype[c+1]}  max |a| {a_:.1f}")

# --- summary for the pipeline: lever arm, INS heading offset (course over ground - INS yaw on RTK-good straight
#     driving), GNSS quality segments
tt = t_grid[::100]
pp, qq = pos[::100], quat[::100]
vv = np.gradient(pp[:, :2], 0.1, axis=0)
yaw = np.unwrap(R.from_quat(qq).as_euler("ZYX")[:, 0])
yr = np.degrees(np.gradient(yaw, 0.1))
jj = np.clip(np.searchsorted(tg, tt), 0, len(tg) - 1)
good = np.isin(ptype[jj], (50, 56)) & (np.hypot(std[jj, 0], std[jj, 1]) < 0.05)
msk = good & (np.linalg.norm(vv, axis=1) > 5) & (np.abs(yr) < 0.5)
off = np.degrees(np.angle(np.exp(1j * (np.arctan2(vv[:, 1], vv[:, 0]) - yaw))))
yaw_off = float(np.median(off[msk])) if msk.sum() > 100 else 0.0
segs = []
chg = np.r_[0, np.flatnonzero(np.diff(ptype)) + 1, len(ptype)]
for a_, b_ in zip(chg[:-1], chg[1:]):
    segs.append({"t0": float(tg[a_] - tg[0]), "t1": float(tg[b_ - 1] - tg[0]), "type": int(ptype[a_]),
                 "std_h_median": float(np.median(np.hypot(std[a_:b_, 0], std[a_:b_, 1])))})
dist = float(np.sum(np.linalg.norm(np.diff(pp[:, :2], axis=0), axis=1)))
json.dump({"lever_arm": l.tolist(), "lever_arm_source": lever_src, "lever_arm_rms_m": rms,
           "attitude_latency_ms_median": float(np.median(lat_att) * 1e3),
           "ins_heading_offset_deg": yaw_off, "ins_heading_offset_samples": int(msk.sum()),
           "rtk_good_fraction": float(np.mean(good)), "duration_s": float(tg[-1] - tg[0]), "route_length_m": dist,
           "gnss_segments": segs}, open(os.path.join(OUT, "traj_info.json"), "w"), indent=1)
print(f"INS heading offset (course over ground - INS yaw, {msk.sum()} straight RTK samples): {yaw_off:+.2f} deg; "
      f"RTK-good {np.mean(good)*100:.0f}% of the drive; route {dist:.0f} m")
