#!/usr/bin/env python3
"""Wait until killer2 saturates, then pause it and start worker3 (user, 2026-09-30).

Saturation: killer2's own objective against the fixed farm2 opponent (eval_h2h.py rounds in
runs/killer2/h2h.jsonl: 2 x kills - deaths per minute), averaged over the last 8 rounds, sets no new high
(by more than 0.05) for 60 minutes, after at least 90 minutes of training — or 10 hours at the most.

Then: stop killer2's trainer + evaluator, freeze its checkpoint and farm2's as worker3's opponents,
start worker3 (from farm2; opponents: self-play 1/3, farm2 1/3, killer2 1/3; adaptive jump-height
curriculum; 7 goals / 5 min, no sudden death) and point the dashboard's Worker tab at it.

    nohup .venv/bin/python -u ops/killer_then_worker.py >> runs/killer2/watch.out 2>&1 &
"""
import json, os, shutil, subprocess, sys, time

import numpy as np

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS = os.path.join(HERE, "runs")
PY = os.path.join(HERE, ".venv", "bin", "python")
MIN_H, QUIET_MIN, MAX_H, WIN, MARGIN = 1.5, 60, 10.0, 8, 0.05


def rounds():
    out = []
    try:
        for line in open(os.path.join(RUNS, "killer2", "h2h.jsonl")):
            r = json.loads(line)
            if r.get("kills_per_min") is not None:
                out.append((r["t"], 2 * r["kills_per_min"] - r["deaths_per_min"]))
    except OSError:
        pass
    return sorted(out)


def saturated(start):
    rs = rounds()
    now = time.time()
    if now - start > MAX_H * 3600:
        return f"{MAX_H} h cap reached"
    if now - start < MIN_H * 3600 or len(rs) < 2 * WIN:
        return None
    t = np.array([x[0] for x in rs]); v = np.array([x[1] for x in rs])
    roll = np.array([v[max(0, i - WIN + 1):i + 1].mean() for i in range(len(v))])
    old, new = roll[t <= now - QUIET_MIN * 60], roll[t > now - QUIET_MIN * 60]
    if len(old) < WIN or len(new) < 3:
        return None
    if new.max() <= old.max() + MARGIN:
        return (f"no new high in {QUIET_MIN} min: best 8-round mean {old.max():.2f} before, {new.max():.2f} since "
                f"(2 x kills - deaths per min vs farm2; {len(rs)} rounds)")
    return None


def main():
    status = json.load(open(os.path.join(RUNS, "killer2", "status.json")))
    first = rounds()
    start = first[0][0] if first else time.time()
    print(f"{time.ctime()} watching killer2 (update {status['update']})", flush=True)
    while True:
        why = saturated(start)
        if why:
            break
        time.sleep(300)
    print(f"{time.ctime()} killer2 saturated: {why}", flush=True)
    subprocess.run(["pkill", "-f", "train_farm.py --run killer2"])
    subprocess.run(["pkill", "-f", "eval_h2h.py --run killer2"])
    time.sleep(10)
    w = os.path.join(RUNS, "worker3")
    os.makedirs(w, exist_ok=True)
    shutil.copy(os.path.join(RUNS, "killer2", "ckpt.pt"), os.path.join(w, "opp_killer2.pt"))
    shutil.copy(os.path.join(RUNS, "farm2", "ckpt.pt"), os.path.join(w, "opp_farm2.pt"))
    with open(os.path.join(RUNS, "killer2", "SATURATED.txt"), "w") as f:
        f.write(f"{time.ctime()}: {why}\n")
    cmd = [PY, "-u", "train_worker.py", "--run", "worker3", "--init", os.path.join(RUNS, "farm2", "ckpt.pt"),
           "--init-std", "0.3", "--min-std", "0.15", "--hosts", "12", "--matches", "16", "--threads", "16",
           "--torch-threads", "4", "--solo-frac", "0", "--dead-frac", "0",
           "--opps", f"farm2:{w}/opp_farm2.pt:0.34,killer2:{w}/opp_killer2.pt:0.33",
           "--no-sudden-death", "--format", "7,300", "--air-curriculum"]
    with open(os.path.join(w, "train.out"), "a") as out:
        subprocess.Popen(cmd, cwd=HERE, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"{time.ctime()} started worker3: {' '.join(cmd)}", flush=True)
    time.sleep(120)  # the worker's first checkpoint (every 25 updates) before the dashboard loads it
    subprocess.run(["pkill", "-f", "dashboard.py --run"])
    time.sleep(3)
    dash = [PY, "-u", "dashboard.py", "--run", "farm2", "--manager-run", "manager2", "--worker-run", "worker3",
            "--killer-run", "killer2", "--port", "8765"]
    with open(os.path.join(RUNS, "dashboard.out"), "w") as out:
        subprocess.Popen(dash, cwd=HERE, stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
    print(f"{time.ctime()} dashboard restarted with the Worker tab on worker3", flush=True)


if __name__ == "__main__":
    main()
