#!/usr/bin/env python3
"""Why does policy A lose to policy B? Head-to-head in the native sim (7 goals / 5 min, no sudden death,
sides swapped), broken down per policy from its own observations (attack frame):
  - situations (docs/strategy.md, analysis/situations.py classes): share of live time and the goal difference
    over the next 10 s from that moment, per policy
  - goals and kills per minute; what the scorer and the conceder were doing 3 s before each goal
  - style: good side, airborne, in a pit, pole quadrant, backward / forward passes through vertical

    analysis/h2h_breakdown.py runs/farm3/ckpt.pt runs/farm2/ckpt.pt --matches 128
"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import eval_h2h as e  # noqa: E402
from native_env import NativeHost  # noqa: E402
from situations import NAMES, classify  # noqa: E402

H = 150  # 10 s at frame skip 4


def play(pa, pb, games, threads, seed):
    env = NativeHost(1, games, 4, sd_frac=0.0, seed=seed, threads=threads, end_on_sd=True,
                     match_format=e.FORMAT, max_steps=e.MAX_STEPS)
    a_rider = np.array([k % 2 for k in range(games)])  # 0/1: which rider A drives in match k
    obs = env.reset_all()
    done = np.zeros(games, bool)
    O, G, K, LIVE = [], [], [], []  # per step: [games, 2 (A, B), ...]
    idx = np.arange(games)
    while not done.all():
        oa, ob = obs[idx, a_rider], obs[idx, 1 - a_rider]
        act = np.zeros((games, 2, 3), np.float32)
        act[idx, a_rider] = np.clip(e.sample(pa, oa), -1, 1)
        act[idx, 1 - a_rider] = np.clip(e.sample(pb, ob), -1, 1)
        out = env.step(act)
        live = ~done
        t = out["terms"]
        O.append(np.stack([oa, ob], 1)); LIVE.append(live.copy())
        G.append(np.stack([t[idx, a_rider, 0], t[idx, 1 - a_rider, 0]], 1) * live[:, None])
        K.append(np.stack([t[idx, a_rider, 1], t[idx, 1 - a_rider, 1]], 1) * live[:, None])
        done |= out["reset"]
        obs = out["obs"]
    env.close()
    return np.array(O), np.array(G), np.array(K), np.array(LIVE)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("a"); p.add_argument("b")
    p.add_argument("--matches", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    a = p.parse_args()
    torch.set_num_threads(1)
    pa, pb = e.load_policy(a.a)[:2], e.load_policy(a.b)[:2]
    na, nb = (os.path.basename(os.path.dirname(x)) for x in (a.a, a.b))
    O, G, K, LIVE = play(pa, pb, a.matches, a.threads, 777)
    T = len(O)
    mins = LIVE.sum() * 4 / 60 / 60
    cg = np.concatenate([np.zeros((1,) + G.shape[1:]), np.cumsum(G, 0)])
    fut = cg[np.minimum(np.arange(T) + H, T)] - cg[np.arange(T)]  # [T, games, 2] each policy's own goal diff
    ok = LIVE[:, :, None] & (np.arange(T)[:, None, None] + H <= T) & (O[..., 9] < 0.5)  # alive, horizon inside
    cls = classify(O)
    gs = (G > 0).sum((0, 1)); kills = (K > 0).sum((0, 1))
    print(f"{na} vs {nb}: {a.matches} matches, {mins:.0f} match-minutes")
    print(f"  goals/min   {na} {gs[0] / mins:.2f}   {nb} {gs[1] / mins:.2f}")
    print(f"  kills/min   {na} {kills[0] / mins:.2f}   {nb} {kills[1] / mins:.2f}")
    print(f"\n  situation share of live time and goal diff next 10 s ({na} | {nb}):")
    for i, n in enumerate(NAMES):
        row = []
        for s in (0, 1):
            m = ok[..., s] & (cls[..., s] == i)
            row.append(f"{100 * m.sum() / max(ok[..., s].sum(), 1):5.1f}%  {fut[..., s][m].mean() if m.any() else float('nan'):+.3f}")
        print(f"    {n:16s} {row[0]:>16s}   | {row[1]:>16s}")
    # 3 s before each goal: the scorer's and the conceder's situation
    print("\n  3 s before a goal — situation of the scorer / the conceder:")
    for s, name in ((0, na), (1, nb)):
        tt, gg = np.nonzero(G[..., s] > 0)
        t3 = np.maximum(tt - 45, 0)
        sc = cls[t3, gg, s]; co = cls[t3, gg, 1 - s]
        dead_c = O[t3, gg, 1 - s, 9] > 0.5
        print(f"    {name} scored {len(tt)}: scorer " + ", ".join(f"{NAMES[i]} {100 * (sc == i).mean():.0f}%" for i in range(5))
              + f" | conceder dead/ghost {100 * dead_c.mean():.0f}%, conceder " +
              ", ".join(f"{NAMES[i]} {100 * (co == i).mean():.0f}%" for i in range(4)))
    # style
    ang = np.degrees(np.arctan2(O[..., 10], O[..., 11])); ang = np.where(ang < -90, ang + 360, ang)
    print("\n  style (share of live time):")
    for s, name in ((0, na), (1, nb)):
        L = LIVE[:, :, None].repeat(2, 2)[..., s] & (O[..., s, 9] < 0.5)
        good = (O[..., s, 0] < O[..., s, 52])[L].mean()
        air = (O[..., s, 38] < -1.0)[L].mean()  # floor under me (offset 0) more than 1 m below
        pit = ((np.abs(O[..., s, 0]) > 16.3) & (O[..., s, 1] < 0))[L].mean()
        q = np.clip(((ang[..., s] + 90) // 90).astype(int), 0, 3)[L]
        prev, cur = ang[:-1, :, s], ang[1:, :, s]
        both = L[1:] & L[:-1] & (np.abs(cur - prev) < 60)
        back = (both & (prev < 90) & (cur >= 90)).sum() / mins; fwd = (both & (prev >= 90) & (cur < 90)).sum() / mins
        dist = np.abs(O[..., s, 60])[L].mean()
        print(f"    {name:6s} good side {100 * good:4.1f}%  airborne {100 * air:4.1f}%  pit {100 * pit:4.1f}%  "
              f"|ball dx| {dist:4.1f} m  pole fwd-down/fwd-up/back-up/back-down "
              + "/".join(f"{100 * (q == i).mean():.0f}" for i in range(4)) + f"%  passes back {back:.1f} fwd {fwd:.1f} /min")


if __name__ == "__main__":
    main()
