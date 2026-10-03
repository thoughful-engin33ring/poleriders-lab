#!/usr/bin/env python3
"""Does farm2 play better if it believes the match is at 6-6 with 15 s left than when it sees the real
score and clock? Head-to-head: a farm2 copy whose observation's match block (time left, my score, opponent
score, sudden death; obs 56-59) is replaced by a fixed fake vs an unmodified farm2 copy, sides swapped,
7 goals / 5 min, no sudden death (eval_h2h.py rules). Only the perception changes; the real match goes on.

    .venv/bin/python analysis/fake_score.py --games 256
"""
import argparse, os, sys, time
import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import eval_h2h as e  # noqa: E402
from native_env import NativeHost  # noqa: E402


def play(pol, games, fake, threads, seed):
    """fake: None or (time left s, my score, opp score) for the `me` rider; the opponent sees the truth."""
    env = NativeHost(1, games, 4, sd_frac=0.0, seed=seed, threads=threads, end_on_sd=True,
                     match_format=e.FORMAT, max_steps=e.MAX_STEPS, **e.COUNT_KW)
    my = np.array([1 if k % 2 == 0 else 2 for k in range(games)])
    rows_me, rows_op = (np.arange(games), my - 1), (np.arange(games), 2 - my)
    obs = env.reset_all()
    tot = np.zeros((games, 4)); done = np.zeros(games, bool); res = np.zeros((games, 4))
    act = np.zeros((games, 2, 3), np.float32)
    while not done.all():
        om = obs[rows_me].copy()
        if fake:
            om[:, 56], om[:, 57], om[:, 58], om[:, 59] = fake[0] / 180.0, fake[1], fake[2], 0.0
        act[rows_me] = np.clip(e.sample(pol, om), -1, 1)
        act[rows_op] = np.clip(e.sample(pol, obs[rows_op]), -1, 1)
        out = env.step(act)
        live = ~done
        tot[live, :2] += out["terms"][rows_me][live][:, :2]
        tot[live, 2] += 1
        tot[live, 3] += out["rew"][rows_me][live]
        ended = live & out["reset"]
        res[ended] = tot[ended]; done |= ended
        obs = out["obs"]
    env.close()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--ckpt", default=os.path.join(HERE, "..", "runs", "farm2", "ckpt.pt"))
    p.add_argument("--games", type=int, default=256)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--variants", default="main", choices=["main", "frozen"])
    a = p.parse_args()
    torch.set_num_threads(1)
    pol = e.load_policy(a.ckpt)[:2]
    p_variants = [("control: real vs real", None), ("thinks 6-6, 15 s left", (15.0, 6.0, 6.0))]
    if a.variants == "frozen":  # is it the end-game belief, or any frozen score/clock?
        p_variants = [("thinks 0-0, 300 s left", (300.0, 0.0, 0.0)), ("thinks 6-6, 300 s left", (300.0, 6.0, 6.0)),
                      ("thinks 0-0, 15 s left", (15.0, 0.0, 0.0))]
    for label, fake in p_variants:
        t0 = time.time()
        r = play(pol, a.games, fake, a.threads, 4242)
        g = r[:, 0]
        se = g.std() / np.sqrt(len(g))
        w, d, l = int((g > 0).sum()), int((g == 0).sum()), int((g < 0).sum())
        print(f"{label:24s} {a.games} matches: {w}-{d}-{l}  goal diff {g.mean():+.2f} ± {se:.2f}/match  "
              f"kills {r[:, 3].mean():.2f} deaths {(r[:, 3] - r[:, 1]).mean():.2f}  "
              f"{r[:, 2].mean() / 900:.1f} min  ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
