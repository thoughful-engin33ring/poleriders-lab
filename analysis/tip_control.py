#!/usr/bin/env python3
"""Pole-tip targets: learned worker vs a scripted controller, and how often the worker's mirror frame
flips mid-target. Same target sampler as train_worker (tip-only), 6 s limit, 1 m radius.

Scripted controller (rider's own frame, x toward the goal it attacks, y up): aim the pole (right stick
= pole angle setpoint) straight at the target from the body; run toward the target while it is
farther than the pole reaches, away from it while it is much closer.

    analysis/tip_control.py [worker ckpt] [--matches 64] [--steps 1500]"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "bridge"))
import train_worker as tw  # noqa: E402
from hier_env import load_worker  # noqa: E402
from native_env import NativeHost  # noqa: E402


def run(controller, N, steps, seed=9):
    B = 2 * N
    env = NativeHost(1, N, 4, sd_frac=0.0, max_steps=10 ** 9, seed=seed, threads=6)
    O = env.reset_all().reshape(B, 62).copy()
    G = tw.Goals(B, np.random.default_rng(3))
    h = np.zeros(B)

    def sample(rows):
        G.sample(rows, O)
        G.body_on[rows], G.tip_on[rows] = False, True
        h[rows] = G.ty[rows] - (O[rows, 1] - 0.69)
    sample(np.arange(B))
    res, flips, prev_flip, steps_n = [], 0, np.zeros(B, bool), 0
    for t in range(steps):
        a, flip = controller(O, G)
        flips += (flip != prev_flip).sum(); steps_n += B; prev_flip = flip
        out = env.step(a.reshape(N, 2, 3))
        O2 = out["obs"].reshape(B, 62).copy()
        reset = np.repeat(out["reset"], 2)
        valid = ~reset & (O2[:, 9] < 0.5) & (np.abs(O2[:, 0] - O[:, 0]) < 3)
        ok = G.reached(O2) & valid
        G.left -= 1
        done = ok | (G.left <= 0) | reset | (O2[:, 9] > 0.5)
        for i in np.nonzero(done & ~reset & (O2[:, 9] < 0.5))[0]:
            res.append((h[i], ok[i]))
        O = O2
        sample(np.nonzero(done)[0])
    R = np.array(res, float)
    return R, flips / steps_n


def learned(path):
    net, norm, mirror = load_worker(path)

    def ctl(O, G):
        x = np.concatenate([O, G.feats(O)], 1)
        flip = tw.target_behind(G, O, mirror == "goal") if mirror else np.zeros(len(O), bool)
        with torch.no_grad():
            xi = tw.mirror(x, flip)[:, :len(norm.mean)]
            a = np.clip(net.dist(torch.as_tensor(norm(xi), dtype=torch.float32)).sample().numpy(), -1, 1)
        a[flip] *= tw.ACT_MIRROR
        return a, flip
    return ctl


def scripted(O, G):
    me_x, me_y = O[:, 0], O[:, 1]
    reach = np.hypot(O[:, 13], O[:, 14])  # current body-to-tip distance
    dx, dy = G.tx - me_x, G.ty - me_y
    d = np.hypot(dx, dy)
    aim = np.stack([dx, dy], 1) / np.maximum(d, 1e-6)[:, None]
    run = np.where(d > reach + 0.3, np.sign(dx), np.where(d < reach - 1.0, -np.sign(dx), 0.0))
    return np.stack([run, aim[:, 0], aim[:, 1]], 1).astype(np.float32), np.zeros(len(O), bool)


def report(name, R, fr):
    print(f"{name:9s}: {len(R)} targets, reached {100 * R[:, 1].mean():.0f}%  "
          + "  ".join(f"{lo}-{hi} m high {100 * R[(R[:, 0] >= lo) & (R[:, 0] < hi), 1].mean():.0f}%"
                      for lo, hi in ((1, 3), (3, 5), (5, 7), (7, 9)))
          + (f"  | mirror flips per agent-step {fr:.3f}" if fr else ""))


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=os.path.join(HERE, "..", "runs", "worker2", "ckpt.pt"))
    ap.add_argument("--matches", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    torch.manual_seed(0)
    report("learned", *run(learned(a.ckpt), a.matches, a.steps))
    report("scripted", *run(scripted, a.matches, a.steps))
