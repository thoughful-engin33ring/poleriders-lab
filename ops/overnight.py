#!/usr/bin/env python3
"""Overnight batch on search1 t20's recipe: 4 runs side by side, each rated in the gauntlet every 30M env steps.
    .venv/bin/python -u ops/overnight.py >> runs/overnight.out 2>&1 &
Runs: search1_t20 (continued to 300M), n1_t20_seed1 (same recipe, seed 1, 90M), n1_t20_gated (--action-head gated, 90M),
n1_t20_res (--arch resmlp --hidden 256 --layers 2, 90M). Ratings: candidates "<run>@<N>M" in runs/gauntlet."""
import json, os, shutil, subprocess, sys, time
HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)
import search  # noqa: E402  (train_cmd, PY)
from types import SimpleNamespace
P = json.load(open(os.path.join(HERE, "runs", "search1", "state.json")))["trials"]["20"]["params"]
A = SimpleNamespace(hosts=4, matches=24, threads=4, seed=0)
RUNS = [("search1_t20", 300e6, [], 0), ("n1_t20_seed1", 90e6, [], 1),
        ("n1_t20_gated", 90e6, ["--action-head", "gated"], 0),
        ("n1_t20_res", 90e6, ["--arch", "resmlp", "--hidden", "256", "--layers", "2"], 0)]
EVERY = 30e6


def steps(run):
    try:
        return int(json.load(open(os.path.join(HERE, "runs", run, "status.json")))["steps"])
    except (OSError, ValueError, KeyError):
        return 0


def main():
    procs, rated = {}, {}
    for run, total, extra, seed in RUNS:
        A.seed = seed
        resume = os.path.exists(os.path.join(HERE, "runs", run, "ckpt.pt"))
        cmd = search.train_cmd(run, P, total, resume, A) + extra
        os.makedirs(os.path.join(HERE, "runs", run), exist_ok=True)
        procs[run] = subprocess.Popen(cmd, cwd=HERE, stdout=open(os.path.join(HERE, "runs", run, "train.out"), "a"),
                                      stderr=subprocess.STDOUT, start_new_session=True)
        rated[run] = (steps(run) // EVERY) * EVERY if resume else 0
        print(f"{time.ctime()} {run}: {'resumed' if resume else 'new'} -> {total / 1e6:.0f}M {' '.join(extra)}", flush=True)
    while True:
        for run, total, _, _ in RUNS:
            s = steps(run)
            due = rated[run] + EVERY
            if s >= due or (procs[run].poll() is not None and s > rated[run]):
                mark = min(due, s) if s >= due else s
                snap = os.path.join(HERE, "runs", run, f"rate_{int(mark / 1e6)}M.pt")
                shutil.copy(os.path.join(HERE, "runs", run, "ckpt.pt"), snap)
                name = f"{run}@{int(mark / 1e6)}M"
                rated[run] = mark
                print(f"{time.ctime()} rating {name}", flush=True)
                subprocess.call([search.PY, "-u", "gauntlet.py", "rate", snap, "--name", name, "--games", "48",
                                 "--procs", "3"], cwd=HERE, stdout=open(os.path.join(HERE, "runs", run, "rate.out"), "a"),
                                stderr=subprocess.STDOUT)
                c = json.load(open(os.path.join(HERE, "runs", "gauntlet", "candidates.json"))).get(name, {})
                print(f"{time.ctime()} {name}: Elo {c.get('elo')}", flush=True)
        if all(p.poll() is not None for p in procs.values()) and \
                all(rated[r] >= steps(r) for r, *_ in RUNS):
            print(f"{time.ctime()} all done", flush=True)
            return
        time.sleep(30)


if __name__ == "__main__":
    main()
