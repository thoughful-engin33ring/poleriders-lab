#!/usr/bin/env python3
"""Situation scoreboard (docs/strategy.md): classify every live moment of a flat policy's self-play by
relative positions along the attack axis and measure what follows.

Agent frame (observe2): I attack +x; my x = obs[0], opponent x = obs[17], ball x = obs[52]; the
opponent attacks -x (his good side is x > ball). "Half a pole" ~ 3 m (the pole is ~5.5-6 m).
    right-far    me < ball - 3                       (you-ball-goal, room)
    right-close  ball - 3 <= me < ball               (right side but within half a pole)
    A            ball < opp < me                     (you-opp-ball: opp on his good side, you beyond him)
    B            ball < me < opp                     (opp-you-ball: you between the ball and the opponent)
    both-wrong   me > ball and opp < ball            (both beyond the ball)
    opp-dead     opponent is out of play (obs ghost flag: killed and waiting to respawn, or spawn ghost)
Outcome per moment: my goals minus the opponent's in the next H seconds, and who scores next.
Also: what a kill is worth (goal difference for the killer in the next H seconds).

    analysis/situations.py [ckpt] [--matches 64] [--steps 6000] [--horizon 10]"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "bridge"))
from native_env import NativeHost  # noqa: E402
from train_ppo import net_from_ckpt  # noqa: E402

STEP_S = 4 / 60
HALF_POLE = 3.0
NAMES = ["right-far", "right-close", "A you-opp-ball", "B opp-you-ball", "both-wrong", "opp-dead"]


def classify(O):
    me, opp, ball = O[..., 0], O[..., 17], O[..., 52]
    opp_dead = O[..., 17 + 9] > 0.5
    c = np.full(me.shape, -1)
    c[me < ball - HALF_POLE] = 0
    c[(me >= ball - HALF_POLE) & (me < ball)] = 1
    past = me >= ball
    c[past & (opp > ball) & (opp < me)] = 2
    c[past & (opp >= me)] = 3
    c[past & (opp <= ball)] = 4
    c[opp_dead] = 5
    c[O[..., 9] > 0.5] = -1  # I'm dead: not classified
    return c


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=os.path.join(HERE, "..", "runs", "farm1", "ckpt.pt"))
    ap.add_argument("--matches", type=int, default=64)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--horizon", type=float, default=10.0)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, weights_only=False)
    net, norm, D = net_from_ckpt(ck)  # any architecture / action head
    torch.manual_seed(0)
    N = args.matches
    env = NativeHost(1, N, 4, sd_frac=0.0, max_steps=10 ** 9, seed=11, threads=6)
    obs = env.reset_all().copy()
    C, G, K, R, OBS = [], [], [], [], []  # class [T,N,2], goal/kill terms [T,N,2], reset [T,N], obs

    for t in range(args.steps):
        C.append(classify(obs))
        OBS.append(obs[..., [0, 9, 13, 52]].copy())
        with torch.no_grad():
            a = net.sample_env(torch.as_tensor(norm(obs.reshape(-1, D)), dtype=torch.float32)).numpy()
        out = env.step(np.clip(a, -1, 1).reshape(N, 2, 3).astype(np.float32))
        G.append(out["terms"][:, :, 0].copy()); K.append(out["terms"][:, :, 1].copy()); R.append(out["reset"].copy())
        obs = out["obs"].copy()
    C, G, K, R = map(np.array, (C, G, K, R))
    T = len(C)
    H = int(args.horizon / STEP_S)
    # goal difference in the next H steps (stop at a match reset), and sign of the next goal
    fut = np.zeros((T, N, 2)); nxt = np.zeros((T, N, 2))
    acc = np.zeros((N, 2)); last = np.zeros((N, 2))
    csum = np.cumsum(G, 0)
    seg = np.cumsum(R, 0)  # match segment id per step
    for t in range(T):
        t2 = min(T - 1, t + H)
        same = seg[t2] == (seg[t - 1] if t else 0)
        fut[t] = csum[t2] - (csum[t - 1] if t else 0)
        # if a reset happened in between, count only up to it (approximate: drop those rows)
        fut[t][~same] = np.nan
    for t in range(T - 1, -1, -1):  # who scores next (within the same match)
        g = G[t]
        last = np.where(g != 0, np.sign(g), last)
        last[R[t]] = 0
        nxt[t] = last
    print(f"{args.ckpt}: {N} matches x {T} steps ({T * STEP_S / 60:.0f} min each), self-play; horizon {args.horizon:.0f} s")
    live = C >= 0
    print(f"{'situation':18s} {'time':>6s} {'goal diff next %ds' % args.horizon:>20s} {'P(we score next)':>17s} {'P(they do)':>11s}")
    for i, name in enumerate(NAMES):
        m = C == i
        f = fut[m]; f = f[~np.isnan(f)]
        nx = nxt[m]
        print(f"{name:18s} {100 * m.sum() / live.sum():5.1f}% {np.mean(f):+20.3f} {100 * (nx > 0).mean():16.0f}% {100 * (nx < 0).mean():10.0f}%")
    # docs/strategy.md round 3: near the ball the pole tip decides the side. Split moments where my body is
    # within half a pole of the ball (either side) by where my tip is relative to the ball.
    Ot = np.array(OBS)  # columns: my x, my ghost/dead flag, tip dx, ball x
    me, ball, tip = Ot[..., 0], Ot[..., 3], Ot[..., 0] + Ot[..., 2]
    near = (np.abs(me - ball) < HALF_POLE) & (Ot[..., 1] < 0.5)
    print(f"\nnear the ball (body within {HALF_POLE:.0f} m of it, {100 * near.sum() / live.sum():.0f}% of live time), by pole tip:")
    for body_name, bm in (("body on the right side", me < ball), ("body slightly past it", me >= ball)):
        for tip_name, tm in (("tip ahead of the ball (hooked forward)", tip > ball), ("tip behind the ball", tip <= ball)):
            m = near & bm & tm
            f = fut[m]; f = f[~np.isnan(f)]; nx = nxt[m]
            print(f"  {body_name:22s} {tip_name:38s} {100 * m.sum() / live.sum():4.1f}%  goal diff next 10 s {np.mean(f):+.3f}"
                  f"  we score next {100 * (nx > 0).mean():3.0f}%  they do {100 * (nx < 0).mean():3.0f}%")
    # value of a kill: goal difference for the killer over the next H seconds after the kill
    kills = np.argwhere(K > 0)
    vals = [fut[min(T - 1, t + 1), n, a] for t, n, a in kills if not np.isnan(fut[min(T - 1, t + 1), n, a])]
    nk = [nxt[min(T - 1, t + 1), n, a] for t, n, a in kills]
    base = fut[live]; base = base[~np.isnan(base)]
    print(f"\nkills: {len(kills)}; killer's goal diff over the next {args.horizon:.0f} s: {np.mean(vals):+.3f} "
          f"(all moments: {np.mean(base):+.3f}); killer scores the next goal {100 * np.mean(np.array(nk) > 0):.0f}%, "
          f"victim does {100 * np.mean(np.array(nk) < 0):.0f}%")


if __name__ == "__main__":
    main()
