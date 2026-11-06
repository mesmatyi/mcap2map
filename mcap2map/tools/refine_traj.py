#!/usr/bin/env python3
"""Refine the GNSS/INS reference trajectory with lidar constraints (linearised pose graph on corrections).

Unknowns: corrections (dx, dy, dz, dyaw) at nodes every NODE_DT s, applied as p' = p + dp(t), R' = Rz(dyaw(t)) R,
linearly interpolated between nodes. At every 'break' of the reference (receiver solution jumps) a node pair
1 ms apart lets the correction jump too. Plus five global terms: lidar lever-arm correction (ex, ey) in base_link,
lidar time offset, and a residual lidar tilt (pitch, roll) relative to the INS level, which biases the vertical
offset of two submaps by tilt x their along / across displacement (and so makes lidar heights drift).

Factors
  * GNSS prior (correction ~ 0, the reference IS the receiver solution): sigma from the BESTUTM std and solution type
    (RTK fixed: cm; INS single-point / DGPS: metres, inflated for their time-correlated errors), robust (Huber)
  * weak INS yaw prior (the SPAN heading wanders by a few tenths of a degree, esp. after standstill)
  * random-walk smoothness between nodes; across a break the correction jumps by minus the reference's own step
    there (the vehicle moved continuously, the receiver solution / the odometry hand-over did not)
  The reference (build_ref.py) is the receiver solution, or lidar odometry where the GNSS was single-point / absent;
  GNSS / yaw prior targets are the receiver solution relative to the reference.
  * lidar: GICP between submaps of the same pass 10 m and 25 m apart (sequential) and between submaps of separate
    passes over the same place (revisits); information from the GICP Hessian with roll/pitch marginalised out

usage: refine_traj.py EXTRACT_DIR CACHE_DIR CALIB_JSON OUT_TRAJ_NPZ OUT_CALIB_JSON [N_ITER]
"""
import json
import multiprocessing as mp
import os
import sys
import time

import numpy as np
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import small_gicp
from scipy.spatial import cKDTree
from scipy.spatial.transform import Rotation as R

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from maplib import ScanCache, Trajectory, deskew_to_map

EXT, CACHE, CALIB, OUT, OUT_CALIB = sys.argv[1:6]
N_ITER = int(sys.argv[6]) if len(sys.argv) > 6 else 3
NODE_DT = 1.0
SIG_RW_P = {"rtk": 0.02, "psr": 0.05}       # m / sqrt(s)
SIG_RW_YAW = np.radians(0.02)                # rad / sqrt(s)
SIG_YAW_PRIOR = np.radians(1.0)
SIG_E, SIG_DT = 0.10, 0.02
SIG_TILT_P, SIG_TILT_R = np.radians(0.5), np.radians(0.1)  # roll is weakly observable (no sideways baselines)
NGL = 5
SEQ_STEP, SEQ_BASES, SEQ_HALF = 5.0, (10.0, 25.0), 4.0
NW = 12

calib = json.load(open(CALIB))
EXTR0 = np.array(calib["extrinsic_base_to_lidar"])
DT0 = calib["dt_lidar_ms"] * 1e-3
sc = ScanCache(EXT)
tr = Trajectory(os.path.join(CACHE, "traj_ref_1khz.npz"))
ref = np.load(os.path.join(CACHE, "traj_ref_1khz.npz"))
breaks = ref["breaks"]
# at a break the vehicle did not jump, the reference did: the correction must cancel the reference's own step
# (receiver solution jumps: cm; odometry -> RTK hand-over: metres), up to the uncertainty of that step estimate
brk_vec = ref["break_step_vec"]
brk_yaw = ref["break_yaw_step_signed"]
SIG_BRK_P, SIG_BRK_Y = 0.05, np.radians(0.02)
pos0, quat0 = tr.pos.copy(), tr.quat.copy()
ins = Trajectory(os.path.join(CACHE, "traj_init_1khz.npz"))  # receiver solution: GNSS targets and INS yaw
st = sc.stamp
g = np.load(os.path.join(CACHE, "gnss_epochs.npz"))
gt, gty, gstd = g["t"], g["type"], g["std"]

# ------------------------------------------------------------------ nodes
t_end = tr.t0 + (tr.N - 1) * tr.dt
reg = tr.t0 + np.arange(int((t_end - tr.t0) / NODE_DT) + 1) * NODE_DT
if reg[-1] < t_end - 1e-6:
    reg = np.r_[reg, t_end]
reg = reg[np.min(np.abs(reg[:, None] - breaks[None, :]), axis=1) > 0.3] if len(breaks) else reg
bpair = np.r_[breaks - 0.0005, breaks + 0.0005]
t_nodes = np.sort(np.r_[reg, bpair])
is_reg = np.isin(t_nodes, reg)
N = len(t_nodes)
NG = 4 * N
across_break = np.zeros(N - 1, bool)
jump_tgt = np.zeros((N - 1, 4))
for b, v_b, y_b in zip(breaks, brk_vec, brk_yaw):
    k = np.searchsorted(t_nodes, b) - 1
    across_break[k] = True
    jump_tgt[k] = (-v_b[0], -v_b[1], -v_b[2], -y_b)
# prior targets: receiver base_link position and INS yaw relative to the reference (zero where the reference is
# the receiver solution itself, non-zero inside the odometry window)
gpos = np.column_stack([np.interp(t_nodes, g["t"], g["p_base"][:, c]) for c in range(3)])
tgt_p = gpos - pos0[tr.index(t_nodes)]


def _yaw(q):
    return R.from_quat(q).as_euler("ZYX")[:, 0]


tgt_y = np.angle(np.exp(1j * (_yaw(ins.quat[ins.index(t_nodes)]) - _yaw(quat0[tr.index(t_nodes)]))))
print(f"{N} nodes ({is_reg.sum()} regular, {len(breaks)} break pairs)", flush=True)

# per-node GNSS sigma
j = np.clip(np.searchsorted(gt, t_nodes), 0, len(gt) - 1)
ntype = gty[j]
nstd = gstd[j]
psr = ntype >= 53
psr &= ntype != 56
node_v = np.linalg.norm(pos0[tr.index(t_nodes + 0.5)] - pos0[tr.index(t_nodes - 0.5)], axis=1)
inflate = np.where(node_v < 0.3, 3.0, 1.0) * np.where(psr, 3.0, 1.0)
floor_h = np.where(ntype == 50, 0.015, np.where(ntype == 56, 0.02, 0.5))
floor_z = np.where(ntype == 50, 0.03, np.where(ntype == 56, 0.04, 1.0))
sig_p = np.column_stack([np.maximum(floor_h, 1.5 * nstd[:, 0]), np.maximum(floor_h, 1.5 * nstd[:, 1]),
                         np.maximum(floor_z, 1.5 * nstd[:, 2])]) * inflate[:, None]
rw_p = np.where(psr[:-1] | psr[1:], SIG_RW_P["psr"], SIG_RW_P["rtk"])
J = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 0.0]])


def interp_w(t):
    k = int(np.clip(np.searchsorted(t_nodes, t) - 1, 0, N - 2))
    return k, float(np.clip((t - t_nodes[k]) / (t_nodes[k + 1] - t_nodes[k]), 0, 1))


def apply_corrections(xn):
    tt = tr.t0 + np.arange(tr.N) * tr.dt
    d = np.column_stack([np.interp(tt, t_nodes, xn[:, c]) for c in range(4)])
    pos = pos0 + d[:, :3]
    quat = (R.from_rotvec(np.column_stack([np.zeros((tr.N, 2)), d[:, 3]])) * R.from_quat(quat0)).as_quat()
    return pos, quat


def calib_from(gl):
    extr = EXTR0.copy()
    extr[:3, 3] += [gl[0], gl[1], 0.0]
    # remove the estimated excess tilt (pitch about base_link y, roll about x) of the lidar points
    extr[:3, :3] = R.from_euler("yx", [-gl[3], -gl[4]]).as_matrix() @ EXTR0[:3, :3]
    return extr, DT0 + gl[2]


# ------------------------------------------------------------------ keyframes and pairs
def keyframes():
    keys, last_p, last_y, last_t = [], None, None, -1e9
    for i, s in enumerate(st):
        t = s + DT0 + 0.05
        if not (tr.valid(t - 0.06) and tr.valid(t + 0.06)):
            continue
        T = tr.pose(t)
        y = np.arctan2(T[1, 0], T[0, 0])
        if last_p is None or np.linalg.norm(T[:3, 3] - last_p) >= 0.25 or \
                abs(np.angle(np.exp(1j * (y - last_y)))) >= np.radians(1.0) or s - last_t >= 2.0:
            keys.append(i)
            last_p, last_y, last_t = T[:3, 3], y, s
    return np.array(keys)


KEYS = keyframes()
print(f"keyframes {len(KEYS)}", flush=True)


def find_pairs(tk):
    P = tr.pos[tr.index(tk)]
    dist = np.r_[0, np.cumsum(np.linalg.norm(np.diff(P[:, :2], axis=0), axis=1))]
    pairs = []
    # sequential
    last = -1e9
    for a in range(len(KEYS)):
        if dist[a] - last < SEQ_STEP:
            continue
        last = dist[a]
        for D in SEQ_BASES:
            b = int(np.searchsorted(dist, dist[a] + D))
            if b < len(KEYS) and abs(dist[b] - dist[a] - D) < 1.0:
                pairs.append((a, b, "seq"))
    # revisits: anchor every 10 m, the closest keyframe of every later pass within 10 m
    kd = cKDTree(P[:, :2])
    last = -1e9
    for a in range(len(KEYS)):
        if dist[a] - last < 10:
            continue
        last = dist[a]
        cand = np.array(kd.query_ball_point(P[a, :2], 10.0), dtype=int)
        cand = cand[(dist[cand] - dist[a] > 40) & (np.abs(P[cand, 2] - P[a, 2]) < 2.0)]
        if not len(cand):
            continue
        cand = cand[np.argsort(dist[cand])]
        for grp in np.split(cand, np.flatnonzero(np.diff(dist[cand]) > 30) + 1):
            pairs.append((a, int(grp[np.argmin(np.linalg.norm(P[grp, :2] - P[a, :2], axis=1))]), "rev"))
    return pairs, dist


# globals for the worker processes (set before each pool is forked)
W_EXTR, W_DT, W_DIST = None, None, None


def submap(c, half, step):
    out, last = [], -1e9
    for jj in np.flatnonzero(np.abs(W_DIST - W_DIST[c]) < half):
        if W_DIST[jj] - last < step:
            continue
        last = W_DIST[jj]
        i = KEYS[jj]
        s = sc.get(i)
        r = np.linalg.norm(s["xyz"], axis=1)
        m = (r > 3) & (r < 40)
        out.append(deskew_to_map(s["xyz"][m][::2], s["t"][m][::2], st[i], tr, W_EXTR, W_DT))
    return np.concatenate(out)


def register_pair(args):
    a, b, kind = args
    half, step = (SEQ_HALF, 1.0) if kind == "seq" else (12.0, 1.5)
    A, B = submap(a, half, step), submap(b, half, step)
    off = A.mean(axis=0)
    ta, tta = small_gicp.preprocess_points(A - off, downsampling_resolution=0.25, num_threads=2)
    tb, _ = small_gicp.preprocess_points(B - off, downsampling_resolution=0.25, num_threads=2)
    r1 = small_gicp.align(ta, tb, tta, max_correspondence_distance=1.5, num_threads=2, max_iterations=30)
    r2 = small_gicp.align(ta, tb, tta, r1.T_target_source, max_correspondence_distance=0.5, num_threads=2,
                          max_iterations=30)
    return a, b, kind, r2.T_target_source, np.asarray(r2.H), off, r2.num_inliers / max(tb.size(), 1)


x = np.zeros(4 * N + NGL)
weights = {}
t_run = time.time()
for it in range(N_ITER):
    tr.pos, tr.quat = apply_corrections(x[:NG].reshape(N, 4))
    W_EXTR, W_DT = calib_from(x[NG:])
    tk = st[KEYS] + W_DT + 0.05
    pairs, W_DIST = find_pairs(tk)
    with mp.get_context("fork").Pool(NW) as pool:
        regs = pool.map(register_pair, pairs, chunksize=4)
    rows, cols, vals, rhs, kinds = [], [], [], [], []

    def add(entries, b, kind):
        n = len(rhs)
        for c, v in entries.items():
            rows.append(n)
            cols.append(c)
            vals.append(v)
        rhs.append(b)
        kinds.append(kind)

    for k in range(N):
        if is_reg[k]:
            for jj in range(3):
                add({4 * k + jj: 1.0 / sig_p[k, jj]}, tgt_p[k, jj] / sig_p[k, jj], ("gnss", k, jj))
            add({4 * k + 3: 1.0 / SIG_YAW_PRIOR}, tgt_y[k] / SIG_YAW_PRIOR, ("yawp", k))
        if k + 1 < N:
            dtn = t_nodes[k + 1] - t_nodes[k]
            sp_ = SIG_BRK_P if across_break[k] else rw_p[k] * np.sqrt(dtn)
            sy_ = SIG_BRK_Y if across_break[k] else SIG_RW_YAW * np.sqrt(dtn)
            for jj in range(3):
                add({4 * (k + 1) + jj: 1.0 / sp_, 4 * k + jj: -1.0 / sp_}, jump_tgt[k, jj] / sp_, ("rw",))
            add({4 * (k + 1) + 3: 1.0 / sy_, 4 * k + 3: -1.0 / sy_}, jump_tgt[k, 3] / sy_, ("rw",))
    add({NG: 1 / SIG_E}, 0.0, ("glob",))
    add({NG + 1: 1 / SIG_E}, 0.0, ("glob",))
    add({NG + 2: 1 / SIG_DT}, 0.0, ("glob",))
    add({NG + 3: 1 / SIG_TILT_P}, 0.0, ("glob",))
    add({NG + 4: 1 / SIG_TILT_R}, 0.0, ("glob",))

    stats = {"seq": [], "rev": []}
    rejected = {"seq": 0, "rev": 0}
    for a, b, kind, T, H, off, inl in regs:
        if inl < 0.3 or np.linalg.norm(T[:3, 3]) > 1.5:
            rejected[kind] += 1
            continue
        keep, marg = [2, 3, 4, 5], [0, 1]
        Hk = H[np.ix_(keep, keep)] - H[np.ix_(keep, marg)] @ np.linalg.solve(H[np.ix_(marg, marg)], H[np.ix_(marg, keep)])
        Cov = np.linalg.inv(Hk)
        Cov *= 0.02 ** 2 / min(Cov[1, 1], Cov[2, 2], Cov[3, 3])
        Cov[0, 0] = max(Cov[0, 0], np.radians(0.01) ** 2)
        Lw = np.linalg.cholesky(np.linalg.inv(Cov)).T
        z = np.r_[R.from_matrix(T[:3, :3]).as_rotvec()[2], T[:3, 3]]
        ta_, tb_ = tk[a], tk[b]
        Ta, Tb = tr.pose(ta_), tr.pose(tb_)
        pa, pb = Ta[:3, 3], Tb[:3, 3]
        va = (tr.pos[tr.index(ta_ + 0.05)] - tr.pos[tr.index(ta_ - 0.05)]) / 0.1
        vb = (tr.pos[tr.index(tb_ + 0.05)] - tr.pos[tr.index(tb_ - 0.05)]) / 0.1
        ka, wa = interp_w(ta_)
        kb, wb = interp_w(tb_)
        la, lb = J @ (off - pa), J @ (off - pb)
        dab = Ta[:3, :3].T @ (pb - pa)  # displacement a -> b in a's body frame (along, across)
        M = [{4 * kb + 3: 1 - wb, 4 * (kb + 1) + 3: wb}]
        for kk, ww in ((ka, 1 - wa), (ka + 1, wa)):
            M[0][4 * kk + 3] = M[0].get(4 * kk + 3, 0.0) - ww
        for jj in range(3):
            r_ = {}
            for kk, ww, sgn, lev in [(kb, 1 - wb, 1, lb), (kb + 1, wb, 1, lb), (ka, 1 - wa, -1, la), (ka + 1, wa, -1, la)]:
                r_[4 * kk + jj] = r_.get(4 * kk + jj, 0) + sgn * ww
                r_[4 * kk + 3] = r_.get(4 * kk + 3, 0) + sgn * ww * lev[jj]
            r_[NG] = Tb[jj, 0] - Ta[jj, 0]
            r_[NG + 1] = Tb[jj, 1] - Ta[jj, 1]
            r_[NG + 2] = vb[jj] - va[jj]
            if jj == 2:  # excess tilt: B's points sit -pitch*along + roll*across lower relative to A's
                r_[NG + 3] = -dab[0]
                r_[NG + 4] = dab[1]
            M.append(r_)
        for i in range(4):
            ent = {}
            for jj in range(4):
                for c, v in M[jj].items():
                    ent[c] = ent.get(c, 0.0) + Lw[i, jj] * v
            base = sum(v * x[c] for c, v in ent.items())
            add(ent, float(Lw[i] @ z) + base, ("lidar", kind))
        stats[kind].append((np.linalg.norm(T[:2, 3]), abs(T[2, 3]), np.degrees(abs(z[0]))))
    A_ = sp.csr_matrix((vals, (rows, cols)), shape=(len(rhs), 4 * N + NGL))
    b_ = np.array(rhs)
    # IRLS with Huber weights on GNSS priors and lidar rows (whitened residual threshold 3)
    robust = np.array([k_[0] in ("gnss", "lidar") for k_ in kinds])
    wts = np.ones(len(b_))
    for irls in range(4):
        Wd = sp.diags(wts)
        xs = spla.spsolve((A_.T @ Wd @ A_).tocsc(), A_.T @ (wts * b_))
        r = A_ @ xs - b_
        wts = np.where(robust & (np.abs(r) > 3.0), 3.0 / np.maximum(np.abs(r), 1e-9), 1.0)
    x = xs
    xn = x[:NG].reshape(N, 4)
    gres = np.abs(r[np.array([k_[0] == "gnss" for k_ in kinds])])
    for kind in ("seq", "rev"):
        s_ = np.array(stats[kind]) if stats[kind] else np.zeros((1, 3))
        print(f"iteration {it} {kind}: {len(stats[kind])} used / {rejected[kind]} rejected; pre-solve offsets xy median "
              f"{np.median(s_[:,0])*100:.1f} cm p90 {np.percentile(s_[:,0],90)*100:.1f} cm, z median {np.median(s_[:,1])*100:.1f} cm, "
              f"yaw median {np.median(s_[:,2]):.3f} deg p90 {np.percentile(s_[:,2],90):.3f}", flush=True)
    print(f"  corrections |dp_xy| max {np.hypot(xn[:,0],xn[:,1]).max()*100:.1f} cm (RTK nodes max "
          f"{np.hypot(xn[~psr,0],xn[~psr,1]).max()*100:.1f} cm), |dz| max {np.abs(xn[:,2]).max()*100:.1f} cm, "
          f"dyaw range {np.degrees(xn[:,3].min()):+.3f}..{np.degrees(xn[:,3].max()):+.3f} deg | ex {x[NG]*100:+.1f} cm "
          f"ey {x[NG+1]*100:+.1f} cm dt {x[NG+2]*1e3:+.1f} ms tilt p {np.degrees(x[NG+3]):+.3f} r {np.degrees(x[NG+4]):+.3f} deg | GNSS rows downweighted {np.mean(gres > 3)*100:.1f}% | "
          f"{time.time()-t_run:.0f}s", flush=True)

pos, quat = apply_corrections(x[:NG].reshape(N, 4))
np.savez(OUT, t0=tr.t0, dt=tr.dt, pos=pos, quat=quat, node_t=t_nodes, node_x=x[:NG].reshape(N, 4), keys=KEYS)
extr, dtl = calib_from(x[NG:])
calib.update({"extrinsic_base_to_lidar": extr.tolist(), "dt_lidar_ms": dtl * 1e3,
              "refine_ex_ey_dt": [x[NG], x[NG + 1], x[NG + 2]],
              "refine_tilt_pitch_roll_deg": [float(np.degrees(x[NG + 3])), float(np.degrees(x[NG + 4]))]})
json.dump(calib, open(OUT_CALIB, "w"), indent=2)
print("saved", OUT, OUT_CALIB)
