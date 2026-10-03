#!/usr/bin/env python3
"""How a flat policy uses its pole's rotation, in each rider's own frame (0 deg = toward the goal it attacks,
90 = up, 180 = back toward its own goal). Backward rotation = counter-clockwise in that frame (the user: CCW for
the left rider, CW for the right one) = swinging the pole back over the top.

Self-play matches in the native sim (7 goals / 5 min, no sudden death). Reports: time per pole quadrant,
share of time rotating backward vs forward, backward vs forward passes through vertical, and the goal
difference over the next 10 s by quadrant/rotation, overall and near the ball (|ball dx| < 3 m).

    .venv/bin/python analysis/pole_rotation.py runs/farm2/ckpt.pt --matches 64
"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import eval_h2h as e  # noqa: E402
from native_env import NativeHost  # noqa: E402

QUAD = ["fwd-down (-90..0)", "fwd-up (0..90)", "back-up (90..180)", "back-down (180..270)"]


def run(ckpt, matches, steps, threads, seed):
    pol = e.load_policy(ckpt)[:2]
    env = NativeHost(1, matches, 4, sd_frac=0.0, seed=seed, threads=threads, end_on_sd=True,
                     match_format=e.FORMAT, max_steps=10 ** 9)
    obs = env.reset_all()
    A, W, DX, LIVE, G = [], [], [], [], []  # per step, per rider (matches x 2)
    for t in range(steps):
        a = np.stack([e.sample(pol, obs[:, 0]), e.sample(pol, obs[:, 1])], 1).clip(-1, 1).astype(np.float32)
        ang = np.degrees(np.arctan2(obs[:, :, 10], obs[:, :, 11]))  # agent frame
        A.append(np.where(ang < -90, ang + 360, ang)); W.append(obs[:, :, 12].copy())
        DX.append(obs[:, :, 60].copy()); LIVE.append(obs[:, :, 9] < 0.5)
        out = env.step(a)
        G.append(out["terms"][:, :, 0].copy())  # goal term per rider (+1 scored, -1 conceded)
        obs = out["obs"]
    env.close()
    return [np.array(x) for x in (A, W, DX, LIVE, G)]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("ckpt", nargs="+")
    p.add_argument("--matches", type=int, default=64)
    p.add_argument("--steps", type=int, default=4500)  # 5 min of game time
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(1)
    H = 150  # next 10 s = 150 steps at frame skip 4
    for ck in a.ckpt:
        A, Wv, DX, LIVE, G = run(ck, a.matches, a.steps, a.threads, 99)
        T = len(A)
        cg = np.concatenate([np.zeros((1,) + G.shape[1:]), np.cumsum(G, 0)])
        fut = (cg[np.minimum(np.arange(T) + H, T)] - cg[np.arange(T)])  # goal diff next 10 s, per rider
        valid = LIVE & (np.arange(T)[:, None, None] + H <= T)
        q = np.clip(((A + 90) // 90).astype(int), 0, 3)
        near = np.abs(DX) < 3.0
        back, fwd = Wv > 1.0, Wv < -1.0  # rad/s, mid segment, + = counter-clockwise in the agent frame = backward
        # passes through vertical (90 deg) going backward (CCW) vs forward (CW)
        up_b = LIVE[1:] & LIVE[:-1] & (A[:-1] < 90) & (A[1:] >= 90) & (A[1:] - A[:-1] < 60)
        up_f = LIVE[1:] & LIVE[:-1] & (A[:-1] >= 90) & (A[1:] < 90) & (A[:-1] - A[1:] < 60)
        mins = valid.sum() * 4 / 60 / 60
        print(f"\n== {ck}: {a.matches} matches x {T * 4 / 60 / 60:.1f} min of self-play")
        print(f"rotating backward {100 * (back & LIVE).sum() / LIVE.sum():.0f}% of live time, forward "
              f"{100 * (fwd & LIVE).sum() / LIVE.sum():.0f}%;  passes through vertical: backward "
              f"{up_b.sum() / mins:.1f}/min, forward {up_f.sum() / mins:.1f}/min")
        for lab, m0 in (("everywhere", valid), ("near the ball (|dx| < 3 m)", valid & near)):
            print(f"  {lab}:  (share of time, goal diff next 10 s)")
            for i, name in enumerate(QUAD):
                m = m0 & (q == i)
                print(f"    {name:22s} {100 * m.sum() / max(m0.sum(), 1):5.1f}%  {fut[m].mean() if m.any() else float('nan'):+.3f}")
            for name, m in (("rotating backward", m0 & back), ("rotating forward", m0 & fwd)):
                print(f"    {name:22s} {100 * m.sum() / max(m0.sum(), 1):5.1f}%  {fut[m].mean() if m.any() else float('nan'):+.3f}")


if __name__ == "__main__":
    main()
