#!/usr/bin/env python3
"""Population-based training (PBT) of flat policies from scratch, on the native simulator.

Members are ordinary train_farm.py runs (runs/<name>_m<i>) sharing the machine; each has a genome: PPO settings,
reward weights, curricula and whether it sees the score / clock. Every generation (--gen-steps env steps per member
since the last tournament — user: 25M; or --gen-minutes if --gen-steps 0):
  1. each member's current checkpoint is frozen (runs/<name>/gen<g>/m<i>.pt);
  2. tournament on the real objective: every pair of members and every member vs the anchors (farm2, farm3),
     --games matches each (eval_h2h.play_flat: 7 goals / 5 min, sides swapped). Fitness = mean goal difference
     per match over all its opponents (each opponent weighted equally) — never the shaped reward;
  3. exploit: the worst member takes over the best member's weights, optimizer, normalizer and league pool
     (if the best is ahead by more than --min-gap goals/match), explore: the copied genome is perturbed
     (continuous genes x0.8 / x1.25, switches flipped now and then; the score/clock mask is inherited, never
     mutated, since the network was trained with it).
Members also play each other: the other members' frozen checkpoints make up half of each member's league games.
All members learn goal to goal (--goal-episodes), 7 goals / 5 min, no sudden death (user, 2026-10-02).
State, results and lineage: runs/<name>/state.json (dashboard tab "PBT"). Safe to stop and restart (--resume
relaunches the members from their checkpoints).

    .venv/bin/python -u pbt.py --name pbt1 >> runs/pbt1/pbt.out 2>&1 &        # new population
    .venv/bin/python -u pbt.py --name pbt1 --resume >> runs/pbt1/pbt.out 2>&1 &
"""
import argparse, json, math, os, random, shutil, signal, subprocess, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
import eval_h2h as e  # noqa: E402

RUNS = os.path.join(HERE, "runs")
PY = os.path.join(HERE, ".venv", "bin", "python")

# continuous genes: (low, high, log scale, initial range)
CONT = {
    "lr":            (5e-5, 2e-3, True, (1e-4, 6e-4)),
    "ent_coef":      (1e-4, 3e-2, True, (3e-4, 1e-2)),
    "min_std":       (0.03, 0.5, True, (0.08, 0.3)),
    "gamma":         (0.98, 0.9995, False, (0.99, 0.999)),
    "kill":          (0.0, 1.0, False, (0.1, 0.5)),
    "position":      (0.0, 0.4, False, (0.0, 0.2)),
    "ball_progress": (0.0, 0.02, False, (0.0, 0.008)),
    "rotation_bonus": (0.0, 0.02, False, (0.0, 0.006)),
    "rotation_anneal": (10e6, 400e6, True, (30e6, 150e6)),
    "goal_spawn":    (0.0, 0.9, False, (0.0, 0.5)),
}
BOOL = {"random_pole": 0.2}  # switch: flip probability when mutated
FIXED = ["mask_match_info"]  # inherited with the weights, never mutated


def sample_genome(rng, i):
    g = {}
    for k, (lo, hi, log, (a, b)) in CONT.items():
        g[k] = float(math.exp(rng.uniform(math.log(a), math.log(b))) if log else rng.uniform(a, b))
    g["random_pole"] = bool(rng.random() < 0.5)
    g["mask_match_info"] = bool(i % 2)  # half the population hides the score and clock
    return g


def mutate(g, rng):
    g = dict(g)
    for k, (lo, hi, log, _) in CONT.items():
        if rng.random() < 0.5:
            if k == "gamma":  # perturb the horizon 1 / (1 - gamma), not gamma itself
                g[k] = float(min(hi, max(lo, 1 - (1 - g[k]) / rng.choice((0.8, 1.25)))))
                continue
            v = g[k] * rng.choice((0.8, 1.25))
            if not log and g[k] == 0.0 and rng.random() < 0.3:  # let a switched-off weight come back
                v = lo + 0.1 * (hi - lo)
            g[k] = float(min(hi, max(lo, v)))
    for k, p in BOOL.items():
        if rng.random() < p:
            g[k] = not g[k]
    return g


def member_cmd(run, g, others, resume, a):
    reward = {"goal": 1.0, "kill": g["kill"], "position": g["position"], "ball_progress": g["ball_progress"]}
    cmd = [PY, "-u", "train_farm.py", "--run", run, "--native", "--hosts", str(a.hosts), "--matches", str(a.matches),
           "--procs", str(a.procs), "--threads", str(a.threads), "--torch-threads", "1", "--arena", "2",
           "--no-sudden-death", "--format", "7,300", "--goal-episodes", "--pit-frac", "0",
           "--lr", f"{g['lr']:.6g}", "--ent-coef", f"{g['ent_coef']:.6g}", "--min-std", f"{g['min_std']:.4g}",
           "--gamma", f"{g['gamma']:.6g}", "--reward", json.dumps(reward),
           "--rotation-bonus", f"{g['rotation_bonus']:.6g}", "--rotation-anneal", f"{g['rotation_anneal']:.6g}",
           "--goal-spawn", f"{g['goal_spawn']:.4g}"]
    if g["random_pole"]:
        cmd.append("--random-pole")
    if g["mask_match_info"]:
        cmd.append("--mask-match-info")
    if others:  # the other members' frozen checkpoints: half of the league games between them
        share = 0.5 / len(others)
        cmd += ["--league-fixed", ",".join(f"{n}:{p}:{share:.4f}" for n, p in others)]
    if resume:
        cmd.append("--resume")
    return cmd


class PBT:
    def __init__(self, a):
        self.a = a
        self.dir = os.path.join(RUNS, a.name)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "state.json")
        self.rng = random.Random(a.seed)
        self.procs = {}
        if a.resume and os.path.exists(self.path):
            self.st = json.load(open(self.path))
            print(f"{time.ctime()} resumed {a.name} at generation {self.st['generation']}", flush=True)
        else:
            self.st = {"name": a.name, "generation": 0, "started": time.time(), "anchors": a.anchors.split(","),
                       "members": [{"id": f"m{i}", "run": f"{a.name}_m{i}", "genome": sample_genome(self.rng, i),
                                    "parent": None, "born_gen": 0} for i in range(a.members)],
                       "generations": []}
            self.save()

    def save(self):
        tmp = self.path + ".tmp"
        json.dump(self.st, open(tmp, "w"), indent=1)
        os.replace(tmp, self.path)

    def ckpt(self, m):
        return os.path.join(RUNS, m["run"], "ckpt.pt")

    def others(self, m):
        g = self.st["generation"] - 1  # the last tournament's frozen checkpoints
        out = []
        for o in self.st["members"]:
            p = os.path.join(self.dir, f"gen{g}", f"{o['id']}.pt")
            if o is not m and os.path.exists(p):
                out.append((o["id"], p))
        return out

    def start(self, m):
        resume = os.path.exists(self.ckpt(m))
        os.makedirs(os.path.join(RUNS, m["run"]), exist_ok=True)
        cmd = member_cmd(m["run"], m["genome"], self.others(m), resume, self.a)
        log = open(os.path.join(RUNS, m["run"], "train.out"), "a")
        log.write(f"===== {time.ctime()} pbt {self.a.name} gen {self.st['generation']}: {' '.join(cmd[2:])} =====\n")
        log.flush()
        self.procs[m["id"]] = subprocess.Popen(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT,
                                               start_new_session=True)

    def stop(self, m):
        p = self.procs.pop(m["id"], None)
        if p and p.poll() is None:
            os.killpg(p.pid, signal.SIGTERM)
            try:
                p.wait(30)
            except subprocess.TimeoutExpired:
                os.killpg(p.pid, signal.SIGKILL)
        subprocess.run(["pkill", "-f", f"train_farm.py --run {m['run']} "])

    def restart_all(self):
        for m in self.st["members"]:
            self.stop(m)
            self.start(m)
        # each member's step count at this (re)start: a generation ends once every member trained --gen-steps
        # more (a member that took over another's weights also took over its step count)
        self.base = {m["id"]: self.steps(m) for m in self.st["members"]}

    def gen_due(self, t_gen):
        if self.a.gen_steps > 0:
            done = [self.steps(m) - self.base.get(m["id"], 0) for m in self.st["members"]]
            return min(done) >= self.a.gen_steps
        return time.time() - t_gen >= self.a.gen_minutes * 60

    def steps(self, m):
        try:
            return int(json.load(open(os.path.join(RUNS, m["run"], "status.json")))["steps"])
        except (OSError, ValueError, KeyError):
            return 0

    def tournament(self, g):
        """Freeze every member, play all pairs + every member vs the anchors; returns fitness + raw results."""
        gdir = os.path.join(self.dir, f"gen{g}")
        os.makedirs(gdir, exist_ok=True)
        pols = {}
        for m in self.st["members"]:
            dst = os.path.join(gdir, f"{m['id']}.pt")
            shutil.copy(self.ckpt(m), dst)
            pols[m["id"]] = e.load_policy(dst)[:2]
        for an in self.st["anchors"]:
            pols[an] = e.load_policy(os.path.join(RUNS, an, "ckpt.pt"))[:2]
        ids = [m["id"] for m in self.st["members"]]
        pairs = [(x, y) for i, x in enumerate(ids) for y in ids[i + 1:]] + [(x, an) for x in ids for an in self.st["anchors"]]
        res, per = [], {i: [] for i in ids}
        for x, y in pairs:
            r = e.play_flat(pols[x], pols[y], self.a.games, self.a.eval_threads, 50_000 + 101 * g + len(res))
            gd, kd = float(r[:, 0].mean()), float(r[:, 1].mean())
            res.append({"a": x, "b": y, "goal_diff": gd, "kill_diff": kd, "w": int((r[:, 0] > 0).sum()),
                        "d": int((r[:, 0] == 0).sum()), "l": int((r[:, 0] < 0).sum())})
            per[x].append(gd)
            if y in per:
                per[y].append(-gd)
            print(f"   {x} vs {y}: {gd:+.2f} goals/match ({res[-1]['w']}-{res[-1]['d']}-{res[-1]['l']})", flush=True)
        fit = {i: float(np.mean(v)) for i, v in per.items()}
        vs_anchor = {i: {an: next(r["goal_diff"] for r in res if r["a"] == i and r["b"] == an) for an in self.st["anchors"]}
                     for i in ids}
        return fit, vs_anchor, res

    def exploit_explore(self, fit, g):
        ms = sorted(self.st["members"], key=lambda m: fit[m["id"]])
        worst, best = ms[0], ms[-1]
        if fit[best["id"]] - fit[worst["id"]] < self.a.min_gap:
            return None
        self.stop(worst)
        src_dir, dst_dir = os.path.join(RUNS, best["run"]), os.path.join(RUNS, worst["run"])
        shutil.copy(os.path.join(self.dir, f"gen{g}", f"{best['id']}.pt"), os.path.join(dst_dir, "ckpt.pt"))
        if os.path.isdir(os.path.join(dst_dir, "pool")):
            shutil.rmtree(os.path.join(dst_dir, "pool"))
        if os.path.isdir(os.path.join(src_dir, "pool")):
            shutil.copytree(os.path.join(src_dir, "pool"), os.path.join(dst_dir, "pool"))
        old = worst["genome"]
        worst["genome"] = mutate(best["genome"], self.rng)
        worst["parent"], worst["born_gen"] = best["id"], g
        return {"replaced": worst["id"], "by": best["id"], "old_genome": old, "new_genome": worst["genome"]}

    def run(self):
        self.restart_all()
        t_gen = time.time()
        while True:
            time.sleep(30)
            for m in self.st["members"]:  # a crashed member is restarted from its checkpoint
                p = self.procs.get(m["id"])
                if p is None or p.poll() is not None:
                    print(f"{time.ctime()} {m['id']} not running: restarting", flush=True)
                    self.start(m)
            if not self.gen_due(t_gen):
                continue
            if any(not os.path.exists(self.ckpt(m)) for m in self.st["members"]):
                continue
            g = self.st["generation"]
            print(f"{time.ctime()} generation {g}: tournament", flush=True)
            fit, vs_anchor, res = self.tournament(g)
            change = self.exploit_explore(fit, g) if g >= self.a.first_exploit else None
            rec = {"gen": g, "t": time.time(), "fitness": fit, "vs_anchor": vs_anchor, "results": res,
                   "steps": {m["id"]: self.steps(m) for m in self.st["members"]},
                   "genomes": {m["id"]: dict(m["genome"]) for m in self.st["members"]}, "change": change}
            self.st["generations"].append(rec)
            self.st["generation"] = g + 1
            self.save()
            print(f"{time.ctime()} fitness " + " ".join(f"{k} {v:+.2f}" for k, v in sorted(fit.items()))
                  + (f" | {change['replaced']} <- {change['by']}" if change else " | no replacement"), flush=True)
            # everyone continues against the new frozen set of other members (and the replaced member as its copy)
            self.restart_all()
            t_gen = time.time()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="pbt1")
    p.add_argument("--members", type=int, default=4)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--gen-steps", type=float, default=25e6, help="env steps per member between tournaments")
    p.add_argument("--gen-minutes", type=float, default=45, help="(with --gen-steps 0) minutes between tournaments")
    p.add_argument("--first-exploit", type=int, default=1, help="generations before the first replacement")
    p.add_argument("--min-gap", type=float, default=0.25, help="goals/match the best must lead the worst by")
    p.add_argument("--games", type=int, default=32, help="matches per tournament pairing")
    p.add_argument("--eval-threads", type=int, default=2)
    p.add_argument("--anchors", default="farm2,farm3")
    p.add_argument("--hosts", type=int, default=4)
    p.add_argument("--matches", type=int, default=24)
    p.add_argument("--procs", type=int, default=1)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=1)
    a = p.parse_args()
    torch.set_num_threads(1)
    pbt = PBT(a)

    def bye(*_):
        for m in pbt.st["members"]:
            pbt.stop(m)
        sys.exit(0)
    signal.signal(signal.SIGTERM, bye)
    signal.signal(signal.SIGINT, bye)
    pbt.run()


if __name__ == "__main__":
    main()
