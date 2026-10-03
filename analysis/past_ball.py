#!/usr/bin/env python3
"""What does the policy do once a rider has passed the ball? Native self-play with a checkpoint.

Agent frame (observe2): I attack +x; obs[0] my x, [2] my vx, [52] ball x, [60] ball x - my x,
[61] ball height above me (+ = up), [9] ghost, [8] on ground, [7] on pole.
"Past the ball" = the ball is more than 1 m behind me (obs[60] < -1).

    analysis/past_ball.py [ckpt] [--matches 64] [--steps 6000]"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "bridge"))
from native_env import NativeHost  # noqa: E402
from train_ppo import net_from_ckpt  # noqa: E402

STEP_S = 4 / 60


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=os.path.join(HERE, "..", "runs", "farm1", "ckpt.pt"))
    ap.add_argument("--matches", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, weights_only=False)
    net, norm, D = net_from_ckpt(ck)  # any architecture / action head
    torch.manual_seed(0)
    N = args.matches
    env = NativeHost(1, N, 4, sd_frac=0.0, max_steps=10 ** 9, seed=3)
    obs = env.reset_all().copy()
    O, A, MU, G, R = [], [], [], [], []  # obs, sampled action, mean action, goal term, reset
    for t in range(args.steps):
        x = torch.as_tensor(norm(obs.reshape(-1, D)), dtype=torch.float32)
        with torch.no_grad():
            d = net.dist(x)
            a = net.env_action(d.sample())
        O.append(obs.reshape(-1, D).copy()); A.append(a.numpy()); MU.append(d.mean.numpy())
        out = env.step(np.clip(a.numpy(), -1, 1).reshape(N, 2, 3).astype(np.float32))
        G.append(out["terms"][:, :, 0].reshape(-1).copy()); R.append(np.repeat(out["reset"], 2))
        obs = out["obs"].copy()
    O, A, MU, G, R = map(np.array, (O, A, MU, G, R))  # [T, 2N, ...]
    live = O[:, :, 9] < 0.5
    rel = O[:, :, 60]
    past = live & (rel < -1)
    good = live & (rel > 1)
    print(f"{args.ckpt}\n{N} matches x {args.steps} steps ({args.steps * STEP_S / 60:.0f} min each), self-play\n")
    print(f"live time: past the ball {100 * past.sum() / live.sum():.0f}%, good side {100 * good.sum() / live.sum():.0f}%")
    # conceded goals per minute by situation (G < 0 = the opponent scored)
    for name, m in (("past the ball", past), ("good side", good)):
        conc = ((G < 0) & m).sum() / (m.sum() * STEP_S / 60)
        scored = ((G > 0) & m).sum() / (m.sum() * STEP_S / 60)
        print(f"  while {name:13s}: conceded {conc:.2f}/min, scored {scored:.2f}/min")
    # what do riders do while past the ball?
    vx, run = O[:, :, 2], MU[:, :, 0]
    print("\nwhile past the ball (live):")
    for lo, hi in ((-3, -1), (-8, -3), (-60, -8)):
        m = live & (rel >= lo) & (rel < hi)
        if m.sum() == 0:
            continue
        print(f"  ball {-hi:>2}-{-lo:<2} m behind: n={m.sum():6d}  run cmd mean {run[m].mean():+.2f} "
              f"(share running back {100 * (run[m] < -0.2).mean():.0f}%, forward {100 * (run[m] > 0.2).mean():.0f}%)  "
              f"vx mean {vx[m].mean():+.2f} m/s  on ground {100 * O[:, :, 8][m].mean():.0f}%")
    m = good
    print(f"  (good side for reference: run cmd mean {run[m].mean():+.2f}, vx {vx[m].mean():+.2f} m/s)")
    # spells past the ball: how long, how they end
    T, B = past.shape
    spells = []  # (duration s, end: 'regained' | 'conceded' | 'died' | 'reset' | 'open')
    for b in range(B):
        t = 0
        while t < T:
            if past[t, b]:
                s = t
                end = "open"
                while t < T:
                    if G[t, b] < 0:
                        end = "conceded"; break
                    if R[t, b]:
                        end = "reset"; break
                    if not live[t, b]:
                        end = "died"; break
                    if rel[t, b] > 0:
                        end = "regained"; break
                    t += 1
                spells.append(((t - s) * STEP_S, end))
            t += 1
    dur = np.array([d for d, _ in spells])
    ends = [e for _, e in spells]
    print(f"\n{len(spells)} spells past the ball: duration p50 {np.percentile(dur, 50):.1f}s p75 {np.percentile(dur, 75):.1f}s "
          f"p90 {np.percentile(dur, 90):.1f}s p99 {np.percentile(dur, 99):.1f}s")
    for e in ("regained", "conceded", "died", "reset", "open"):
        sel = [d for d, x in spells if x == e]
        if sel:
            print(f"  ended {e:9s}: {100 * len(sel) / len(spells):4.0f}%  (median {np.median(sel):.1f}s)")
    long_ = dur > 3
    print(f"  spells > 3 s: {100 * long_.mean():.0f}% of spells, {100 * dur[long_].sum() / dur.sum():.0f}% of past-the-ball time; "
          f"of those, conceded {100 * np.mean([e == 'conceded' for (d, e) in spells if d > 3]):.0f}%")
    # counterfactual: move only the ball in real past-the-ball observations
    X = O[past].astype(np.float32)
    idx = np.random.default_rng(0).choice(len(X), min(20000, len(X)), replace=False)
    X = X[idx]
    ahead = X.copy()
    ahead[:, 60] = -X[:, 60]; ahead[:, 52] = X[:, 0] + ahead[:, 60]  # mirror the ball in front of me
    with torch.no_grad():
        f = lambda M: net.dist(torch.as_tensor(norm(M), dtype=torch.float32)).mean.numpy()
        v = lambda M: net.v(torch.as_tensor(norm(M), dtype=torch.float32)).squeeze(-1).numpy()
        mb, ma = f(X), f(ahead)
        vb, va = v(X), v(ahead)
    print(f"\ncounterfactual on {len(X)} past-the-ball states, ball mirrored to the same distance IN FRONT:")
    print(f"  run cmd: actual (ball behind) {mb[:, 0].mean():+.2f}  vs ball in front {ma[:, 0].mean():+.2f}  "
          f"(diff {np.mean(ma[:, 0] - mb[:, 0]):+.2f}; |diff| > 0.5 in {100 * np.mean(np.abs(ma[:, 0] - mb[:, 0]) > 0.5):.0f}%)")
    print(f"  aim: |change| x {np.abs(ma[:, 1] - mb[:, 1]).mean():.2f}, y {np.abs(ma[:, 2] - mb[:, 2]).mean():.2f}")
    print(f"  value: ball behind {vb.mean():+.3f} vs in front {va.mean():+.3f}")


if __name__ == "__main__":
    main()
