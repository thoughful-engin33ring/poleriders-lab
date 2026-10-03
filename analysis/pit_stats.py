#!/usr/bin/env python3
"""How does a policy cope with the enemy pit? Self-play in the native sim; for every visit of an
agent to the enemy pit (agent frame: x > 16.3 m past the plateau and y below it) record how long it
stays, whether it gets out, and what happens meanwhile (goals for/against, deaths).

    analysis/pit_stats.py [ckpt] [--matches 64] [--steps 12000]"""
import argparse, os, sys
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, os.path.join(HERE, "..", "bridge"))
from native_env import NativeHost  # noqa: E402
from train_ppo import net_from_ckpt  # noqa: E402

FS = 4  # frame skip: one step = 4 ticks = 1/15 s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=os.path.join(HERE, "..", "runs", "farm1", "ckpt.pt"))
    ap.add_argument("--matches", type=int, default=64)
    ap.add_argument("--steps", type=int, default=12000)
    ap.add_argument("--seed", type=int, default=5)
    args = ap.parse_args()
    ck = torch.load(args.ckpt, weights_only=False)
    net, norm, D = net_from_ckpt(ck)  # any architecture / action head
    torch.manual_seed(args.seed)
    N = args.matches
    env = NativeHost(1, N, FS, sd_frac=0.15, max_steps=5400, seed=args.seed)
    env.reset_all()
    obs = env.out["obs"].copy()
    inside = np.zeros((N, 2), bool)
    start = np.zeros((N, 2), int)
    ev = np.zeros((N, 2, 3))  # during the current visit: goals for, goals against, deaths
    visits = []  # (steps, escaped, goals_for, goals_against, deaths)
    in_pit_steps = live_steps = 0
    for t in range(args.steps):
        x = torch.as_tensor(norm(obs.reshape(-1, D)), dtype=torch.float32)
        with torch.no_grad():
            a = net.sample_env(x).numpy().reshape(N, 2, 3)
        out = env.step(np.clip(a, -1, 1).astype(np.float32))
        o = out["obs"]
        terms, reset = out["terms"], out["reset"]
        me_x, me_y, ghost = o[:, :, 0], o[:, :, 1], o[:, :, 9]
        pit = (me_x > 16.3) & (me_y < 0) & (ghost < 0.5)
        live = ghost < 0.5
        in_pit_steps += pit.sum(); live_steps += live.sum()
        ev[:, :, 0] += terms[:, :, 0] > 0
        ev[:, :, 1] += terms[:, :, 0] < 0
        ev[:, :, 2] += (ghost > 0.5) & inside  # died while in the pit
        for k in range(N):
            for g in range(2):
                ended = reset[k] or ghost[k, g] > 0.5
                if inside[k, g] and (not pit[k, g] or ended):
                    visits.append((t - start[k, g], not ended and not pit[k, g], *ev[k, g]))
                    inside[k, g] = False
                if not inside[k, g] and pit[k, g] and not reset[k]:
                    inside[k, g], start[k, g] = True, t
                    ev[k, g] = 0
        obs = o.copy()
    v = np.array(visits, float)
    if not len(v):
        print("no pit visits"); return
    dur = v[:, 0] / 15.0
    esc = v[:, 1] > 0
    print(f"{args.ckpt}: {N} matches x {args.steps} steps; share of live agent-time in enemy pit "
          f"{100 * in_pit_steps / max(1, live_steps):.1f}%")
    print(f"visits {len(v)}: escaped {100 * esc.mean():.0f}%, died in pit {100 * (v[:, 4] > 0).mean():.0f}%, "
          f"else match reset")
    for q in (25, 50, 75, 90, 99):
        print(f"  visit duration p{q}: {np.percentile(dur, q):.1f} s")
    long_ = dur > 5
    print(f"  visits > 5 s: {100 * long_.mean():.0f}% of visits, {100 * dur[long_].sum() / dur.sum():.0f}% of pit time")
    print(f"during visits: goals conceded {int(v[:, 3].sum())}, scored {int(v[:, 2].sum())} "
          f"(per minute in pit: conceded {v[:, 3].sum() / (dur.sum() / 60):.2f}, scored {v[:, 2].sum() / (dur.sum() / 60):.2f})")


if __name__ == "__main__":
    main()
