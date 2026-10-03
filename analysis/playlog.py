#!/usr/bin/env python3
"""Human play logs (runs/human/*.jsonl, written by the bridge while you play: play_human.py) -> training pairs.

For every 4th frame (the bots' decision rate) of every match: the human rider's observation (bridge/farm_env.observe2,
exactly what a bot sees) and the human's action in the bots' terms: run = left stick x, aim = right stick, mirrored
for the rider attacking -x (inverse of PoleRidersEnv._stick). The human rider is the one whose pad is a mirrored
human controller (log kind 3) or a real controller (kind 1).

    .venv/bin/python analysis/playlog.py                     # -> runs/human/demos.npz (obs [N, 62], act [N, 3], ...)
"""
import argparse, glob, json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "bridge"))
from farm_env import Terrain, observe2  # noqa: E402


def convert(path, terrain):
    obs, act, frame = [], [], []
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:  # a line cut off when the game was closed
            continue
        st = r.get("s")
        if not st or not st.get("in_match") or not st.get("players") or st.get("game_over"):
            continue
        humans = [i for i, k in enumerate(r["kind"]) if k in (1, 3)]
        for p in st["players"]:
            if p.get("pad") in humans and not p.get("ghost") and not p.get("dead"):
                agent, sign = p["id"], (1.0 if p["id"] == 1 else -1.0)
                lx, ly, rx, ry = r["ax"][p["pad"]]
                obs.append(observe2(st, agent, terrain))
                act.append([sign * lx, sign * rx, -ry])
                frame.append(r["f"])
    return np.array(obs, np.float32).reshape(-1, 62), np.array(act, np.float32).reshape(-1, 3), np.array(frame)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default=os.path.join(HERE, "..", "runs", "human"))
    p.add_argument("--out", default=None)
    p.add_argument("--arena", type=int, default=2)
    a = p.parse_args()
    terrain = Terrain(a.arena)
    O, A, M = [], [], []
    files = sorted(glob.glob(os.path.join(a.dir, "*.jsonl")))
    for i, f in enumerate(files):
        o, ac, _ = convert(f, terrain)
        O.append(o); A.append(ac); M.append(np.full(len(o), i))
        print(f"{os.path.basename(f)}: {len(o)} decisions ({len(o) * 4 / 60 / 60:.1f} min of live play)")
    if not files or not sum(len(o) for o in O):
        sys.exit("no human play recorded yet")
    out = a.out or os.path.join(a.dir, "demos.npz")
    np.savez(out, obs=np.concatenate(O), act=np.concatenate(A), match=np.concatenate(M), files=np.array(files))
    print(f"-> {out}: {sum(len(o) for o in O)} (observation, action) pairs from {len(files)} matches")


if __name__ == "__main__":
    main()
