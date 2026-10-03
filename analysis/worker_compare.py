#!/usr/bin/env python3
"""Head-to-head evaluation of worker checkpoints on identical targets and matches: success by target
type (body / pole tip / both), mean actions (as the manager uses them), same seeds, the current
(all-reachable) target sampler. Opponent: farm1 playing its own game, or the worker itself (duo).

    analysis/worker_compare.py CKPT [CKPT ...] [--matches 64] [--steps 1500]"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..")); sys.path.insert(0, os.path.join(HERE, "..", "bridge"))
import train_worker as tw  # noqa: E402
from hier_env import load_worker  # noqa: E402
from native_env import NativeHost  # noqa: E402
from train_ppo import ActorCritic, RunningNorm  # noqa: E402


def evaluate(path, opp, N, steps, flat):
    net, norm, mode = load_worker(path)
    size = len(norm.mean)
    B = 2 * N
    env = NativeHost(1, N, 4, sd_frac=0.0, max_steps=10 ** 9, seed=31, threads=8)
    O = env.reset_all().reshape(B, 62).copy()
    G = tw.Goals(B, np.random.default_rng(7))
    G.sample(np.arange(B), O)
    work = np.ones(B, bool)
    if opp == "farm1":
        work[1::2] = False
    res = {0: [0, 0], 1: [0, 0], 2: [0, 0]}  # kind -> [reached, total]
    deaths = 0
    fnet, fnorm = flat
    for t in range(steps):
        x = np.concatenate([O, G.feats(O)], 1)
        flip = tw.target_behind(G, O, mode == "goal") if mode else np.zeros(B, bool)
        with torch.no_grad():
            a = np.clip(net.pi(torch.as_tensor(norm(tw.mirror(x, flip)[:, :size]), dtype=torch.float32)).numpy(), -1, 1)
        a[flip] *= tw.ACT_MIRROR
        if opp == "farm1":
            with torch.no_grad():
                a[1::2] = np.clip(fnet.pi(torch.as_tensor(fnorm(O[1::2]), dtype=torch.float32)).numpy(), -1, 1)
        out = env.step(a.reshape(N, 2, 3))
        O2 = out["obs"].reshape(B, 62).copy()
        reset = np.repeat(out["reset"], 2)
        valid = ~reset & (O2[:, 9] < 0.5) & (O[:, 9] < 0.5)
        deaths += ((O[:, 9] < 0.5) & (O2[:, 9] > 0.5) & work & ~reset).sum()
        ok = G.reached(O2) & valid
        G.left -= 1
        done = ok | (G.left <= 0) | reset | (O2[:, 9] > 0.5)
        kind = np.where(G.body_on & G.tip_on, 2, np.where(G.tip_on, 1, 0))
        for i in np.nonzero(done & ~reset & work & (O2[:, 9] < 0.5))[0]:
            res[kind[i]][0] += ok[i]; res[kind[i]][1] += 1
        O = O2
        G.sample(np.nonzero(done)[0], O)
    live_min = steps * work.sum() * 4 / 60 / 60
    return {k: v[0] / max(1, v[1]) for k, v in res.items()}, {k: v[1] for k, v in res.items()}, deaths / live_min


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--matches", type=int, default=64)
    ap.add_argument("--steps", type=int, default=1500)
    a = ap.parse_args()
    fck = torch.load(os.path.join(HERE, "..", "runs", "farm1", "ckpt.pt"), weights_only=False)
    fnet, fnorm = ActorCritic(62), RunningNorm(62)
    fnet.load_state_dict(fck["net"]); fnorm.load(fck["norm"])
    for opp in ("farm1", "duo"):
        print(f"opponent: {opp}")
        for c in a.ckpts:
            s, n, dpm = evaluate(c, opp, a.matches, a.steps, (fnet, fnorm))
            print(f"  {c:40s} body {100 * s[0]:4.0f}%  tip {100 * s[1]:4.0f}%  both {100 * s[2]:4.0f}%  "
                  f"(targets {n[0]}/{n[1]}/{n[2]})  deaths/min {dpm:.2f}")
