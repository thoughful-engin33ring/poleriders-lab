#!/usr/bin/env python3
"""Where does the worker fail pole-tip targets? Runs the worker (tip-only targets) natively and
reports success by target height above the ground and horizontal offset from the body.

    analysis/worker_tip.py [worker ckpt] [--matches 64] [--steps 1500]"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "bridge"))
import train_worker as tw  # noqa: E402
from hier_env import load_worker  # noqa: E402
from native_env import NativeHost  # noqa: E402

ap = argparse.ArgumentParser()
ap.add_argument("ckpt", nargs="?", default=os.path.join(HERE, "..", "runs", "worker2", "ckpt.pt"))
ap.add_argument("--matches", type=int, default=64)
ap.add_argument("--steps", type=int, default=1500)
args = ap.parse_args()
net, norm, mirror = load_worker(args.ckpt)
N = args.matches; B = 2 * N
env = NativeHost(1, N, 4, sd_frac=0.0, max_steps=10 ** 9, seed=9, threads=6)
O = env.reset_all().reshape(B, 62).copy()
G = tw.Goals(B, np.random.default_rng(3))


def sample(rows):
    G.sample(rows, O)
    G.body_on[rows], G.tip_on[rows] = False, True  # tip-only targets for this study
    G.prev[rows] = G.dist(O)[rows]
    s_dx[rows] = G.tx[rows] - O[rows, 0]
    s_h[rows] = G.ty[rows] - (O[rows, 1] - 0.69)
    s_best[rows] = G.dist(O)[rows]


s_dx, s_h, s_best = np.zeros(B), np.zeros(B), np.zeros(B)
sample(np.arange(B))
res = []  # dx, height, success, closest distance
torch.manual_seed(0)
for t in range(args.steps):
    x = np.concatenate([O, G.feats(O)], 1)
    flip = tw.target_behind(G, O) if mirror else np.zeros(B, bool)
    with torch.no_grad():
        a = np.clip(net.dist(torch.as_tensor(norm(tw.mirror(x, flip)), dtype=torch.float32)).sample().numpy(), -1, 1)
    a[flip] *= tw.ACT_MIRROR
    out = env.step(a.reshape(N, 2, 3))
    O2 = out["obs"].reshape(B, 62).copy()
    reset = np.repeat(out["reset"], 2)
    valid = ~reset & (O2[:, 9] < 0.5) & (np.abs(O2[:, 0] - O[:, 0]) < 3)
    ok = G.reached(O2) & valid
    s_best = np.where(valid, np.minimum(s_best, G.dist(O2)), s_best)
    G.left -= 1
    done = ok | (G.left <= 0) | reset | (O2[:, 9] > 0.5)
    for i in np.nonzero(done & ~reset & (O2[:, 9] < 0.5))[0]:
        res.append((s_dx[i], s_h[i], ok[i], s_best[i]))
    O = O2
    sample(np.nonzero(done)[0])
R = np.array(res, float)
print(f"{len(R)} tip-only targets (dead/reset excluded): reached {100 * R[:, 2].mean():.0f}%, "
      f"median closest miss {np.median(R[R[:, 2] == 0, 3]):.2f} m")
print("by height above ground:")
for lo, hi in ((1, 3), (3, 5), (5, 7), (7, 9)):
    m = (R[:, 1] >= lo) & (R[:, 1] < hi)
    print(f"  {lo}-{hi} m: n={m.sum():5d} reached {100 * R[m, 2].mean():3.0f}%  closest-miss median {np.median(R[m & (R[:, 2] == 0), 3]) if (m & (R[:, 2] == 0)).any() else 0:.2f} m")
print("by horizontal offset from the body (+ = ahead):")
for lo, hi in ((-9, -5), (-5, -2), (-2, 2), (2, 5), (5, 9)):
    m = (R[:, 0] >= lo) & (R[:, 0] < hi)
    print(f"  {lo:+d}..{hi:+d} m: n={m.sum():5d} reached {100 * R[m, 2].mean():3.0f}%")
