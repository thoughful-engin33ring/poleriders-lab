#!/usr/bin/env python3
"""Hyperparameter / reward search for the flat pretrain (farmX recipe): Optuna samples, successive halving
spends the steps, the gauntlet scores.

Each batch:
  1. --batch trials sampled by Optuna (TPE, informed by every earlier screen) train from scratch to rung 1
     (--rungs, default 10M env steps), --slots at a time. Each is rated in the gauntlet (Elo vs the frozen
     members, farm2 latest = 1500; gauntlet.py) and that Elo is the trial's Optuna value.
  2. The best third continue (--resume) to rung 2 and are rated again; the best third of those to rung 3.
Promotion waits until the whole rung is done, so early trials don't get promoted by default.

Every trial trains under the house rules: goal-to-goal episodes, 7 goals / 5 min, no sudden death, arena 2,
its own snapshot league. Baselines enqueued first: farm2's recipe and PBT m1's genome.

    .venv/bin/python -u search.py --name search1 >> runs/search1/search.out 2>&1 &
    .venv/bin/python search.py --name search1 --report

State: runs/<name>/state.json (+ study.db, Optuna's own record); trial runs are runs/<name>_t<N>.
"""
import argparse, json, math, os, signal, subprocess, sys, time

import optuna

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
PY = os.path.join(HERE, ".venv", "bin", "python")
sys.path.insert(0, HERE)
import gauntlet  # noqa: E402


def space(trial):
    """The search space (Optuna define-by-run). Ranges: around farm2/farm3/PBT values, wide."""
    p = {
        "lr": trial.suggest_float("lr", 5e-5, 1e-3, log=True),
        "ent_coef": trial.suggest_float("ent_coef", 1e-4, 1e-2, log=True),
        "min_std": trial.suggest_float("min_std", 0.03, 0.3, log=True),
        "init_std": trial.suggest_float("init_std", 0.3, 1.2, log=True),
        "horizon": trial.suggest_float("horizon", 50, 2000, log=True),  # gamma = 1 - 1/horizon (steps of 1/15 s)
        "lam": trial.suggest_float("lam", 0.85, 0.99),
        "kill": trial.suggest_float("kill", 0.0, 0.6),
        "position": trial.suggest_float("position", 0.0, 0.2),
        "ball_progress": trial.suggest_float("ball_progress", 1e-4, 0.02, log=True),
        # capped: pbt1_m3 farmed a 0.006 bonus annealed over 120M (52 backward passes/min, never scored), and so did
        # search1 t8 / t11 with 0.0051 / 0.0043 (Elo -824 / -858 at 10M); every trial at <= 0.0023 played normally
        "rotation_bonus": trial.suggest_float("rotation_bonus", 0.0, 0.0025),
        "rotation_anneal": trial.suggest_float("rotation_anneal", 10e6, 60e6, log=True),
        "goal_spawn": trial.suggest_float("goal_spawn", 0.0, 0.3),
        "random_pole": trial.suggest_categorical("random_pole", [False, True]),
        "mask_match_info": trial.suggest_categorical("mask_match_info", [False, True]),
        "league_frac": trial.suggest_float("league_frac", 0.2, 0.8),
    }
    return p


BASELINES = {
    # farm2: default reward (goal 1, kill 0.3, position 0.1, ball progress 0.003), gamma 0.998, std from 0.61
    "farm2": {"lr": 3e-4, "ent_coef": 1e-3, "min_std": 0.1, "init_std": 0.6065, "horizon": 500, "lam": 0.95,
              "kill": 0.3, "position": 0.1, "ball_progress": 0.003, "rotation_bonus": 0.0, "rotation_anneal": 30e6,
              "goal_spawn": 0.0, "random_pole": False, "mask_match_info": False, "league_frac": 0.5},
    # PBT m1 (the best member, generation 0-1)
    "pbt1_m1": {"lr": 2.1715e-4, "ent_coef": 4.345e-3, "min_std": 0.0802, "init_std": 0.6065, "horizon": 166.9,
                "lam": 0.95, "kill": 0.3886, "position": 0.04575, "ball_progress": 0.007562,
                "rotation_bonus": 0.005409, "rotation_anneal": 31.51e6, "goal_spawn": 0.01272,
                "random_pole": False, "mask_match_info": True, "league_frac": 0.5},
}


def train_cmd(run, p, steps, resume, a):
    reward = {"goal": 1.0, "kill": p["kill"], "position": p["position"], "ball_progress": p["ball_progress"]}
    cmd = [PY, "-u", "train_farm.py", "--run", run, "--native", "--hosts", str(a.hosts), "--matches", str(a.matches),
           "--procs", "1", "--threads", str(a.threads), "--torch-threads", "1", "--arena", "2",
           "--no-sudden-death", "--format", "7,300", "--goal-episodes", "--pit-frac", "0",
           "--lr", f"{p['lr']:.6g}", "--ent-coef", f"{p['ent_coef']:.6g}", "--min-std", f"{p['min_std']:.4g}",
           "--gamma", f"{1 - 1 / p['horizon']:.6g}", "--lam", f"{p['lam']:.4g}", "--reward", json.dumps(reward),
           "--rotation-bonus", f"{p['rotation_bonus']:.6g}", "--rotation-anneal", f"{p['rotation_anneal']:.6g}",
           "--goal-spawn", f"{p['goal_spawn']:.4g}", "--league-frac", f"{p['league_frac']:.4g}",
           "--total-steps", str(int(steps)), "--seed", str(a.seed)]
    if p["random_pole"]:
        cmd.append("--random-pole")
    if p["mask_match_info"]:
        cmd.append("--mask-match-info")
    cmd += ["--resume"] if resume else ["--init-std", f"{p['init_std']:.4g}"]
    return cmd


class Search:
    def __init__(self, a):
        self.a = a
        self.dir = os.path.join(RUNS, a.name)
        os.makedirs(self.dir, exist_ok=True)
        self.path = os.path.join(self.dir, "state.json")
        self.st = gauntlet.jload(self.path, {"name": a.name, "trials": {}, "batches": 0})
        self.study = optuna.create_study(
            study_name=a.name, storage=f"sqlite:///{os.path.join(self.dir, 'study.db')}", load_if_exists=True,
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=a.seed, multivariate=True, constant_liar=True, n_startup_trials=8))
        self.procs = {}   # trial key -> training Popen
        self.evals = {}   # trial key -> rating Popen
        self.rungs = [float(r) * 1e6 for r in a.rungs.split(",")]

    def save(self):
        gauntlet.jsave(self.path, self.st)

    def log(self, msg):
        print(f"{time.strftime('%H:%M:%S')} {msg}", flush=True)

    # -------------------------------------------------------------- one rung for a set of trials
    def run_rung(self, keys, r):
        """Train trials `keys` to rung r (index), --slots at a time, rate each as it finishes; returns when all
        are rated."""
        steps = self.rungs[r]
        todo = [k for k in keys if str(r) not in self.st["trials"][k]["elo"]]
        while todo or self.procs or self.evals:
            while todo and len(self.procs) < self.a.slots:
                k = todo.pop(0)
                t = self.st["trials"][k]
                ck = os.path.join(RUNS, t["run"], "ckpt.pt")
                resume = os.path.exists(ck)
                if resume and gauntlet.ckpt_steps(ck) >= steps:
                    self.start_eval(k, r)
                    continue
                os.makedirs(os.path.join(RUNS, t["run"]), exist_ok=True)
                out = open(os.path.join(RUNS, t["run"], "train.out"), "a")
                self.procs[k] = subprocess.Popen(train_cmd(t["run"], t["params"], steps, resume, self.a), cwd=HERE,
                                                 stdout=out, stderr=subprocess.STDOUT, start_new_session=True)
                self.log(f"t{k}: training to {steps / 1e6:.0f}M ({'resumed' if resume else 'new'})")
            for k, pr in list(self.procs.items()):
                rc = pr.poll()
                if rc is None:
                    continue
                del self.procs[k]
                if rc != 0:
                    self.log(f"t{k}: training failed (rc {rc}); see runs/{self.st['trials'][k]['run']}/train.out")
                    self.st["trials"][k]["elo"][str(r)] = None
                    self.save()
                    continue
                self.start_eval(k, r)
            for k, (pr, name) in list(self.evals.items()):
                if pr.poll() is None:
                    continue
                del self.evals[k]
                elo = gauntlet.jload(os.path.join(gauntlet.DIR, "candidates.json"), {}).get(name, {}).get("elo")
                if elo is None:  # fall back to the rating's own printout
                    import re
                    m = re.findall(rf"{re.escape(name)}: Elo (-?[\d.]+)",
                                   open(os.path.join(RUNS, self.st["trials"][k]["run"], "rate.out")).read())
                    elo = float(m[-1]) if m else None
                t = self.st["trials"][k]
                t["elo"][str(r)] = elo
                self.save()
                self.log(f"t{k}: Elo {elo if elo is None else round(elo)} at {steps / 1e6:.0f}M")
                if r == 0:
                    if elo is None:
                        self.study.tell(t["number"], state=optuna.trial.TrialState.FAIL)
                    else:
                        self.study.tell(t["number"], elo)
            time.sleep(10)

    def start_eval(self, k, r):
        t = self.st["trials"][k]
        name = f"{self.a.name}_t{k}@{self.rungs[r] / 1e6:.0f}M"
        out = open(os.path.join(RUNS, t["run"], "rate.out"), "a")
        self.evals[k] = (subprocess.Popen([PY, "-u", "gauntlet.py", "rate", os.path.join(RUNS, t["run"], "ckpt.pt"),
                                           "--name", name, "--games", str(self.a.games), "--procs", str(self.a.eval_procs)],
                                          cwd=HERE, stdout=out, stderr=subprocess.STDOUT, start_new_session=True), name)
        self.log(f"t{k}: rating {name}")

    # -------------------------------------------------------------- batches
    def new_trials(self, n):
        keys = []
        for _ in range(n):
            tr = self.study.ask()
            p = space(tr)
            k = str(tr.number)
            self.st["trials"][k] = {"number": tr.number, "run": f"{self.a.name}_t{k}", "params": p, "elo": {},
                                    "batch": self.st["batches"], "baseline": tr.user_attrs.get("baseline")}
            keys.append(k)
        self.save()
        return keys

    def run(self):
        if not self.st["trials"]:
            for name, p in BASELINES.items():
                self.study.enqueue_trial(p, user_attrs={"baseline": name})
        while self.st["batches"] < self.a.max_batches:
            b = self.st["batches"]
            keys = [k for k, t in self.st["trials"].items() if t["batch"] == b]
            keys += self.new_trials(self.a.batch - len(keys))
            self.log(f"batch {b}: {len(keys)} trials")
            for r in range(len(self.rungs)):
                self.run_rung(keys, r)
                done = [k for k in keys if self.st["trials"][k]["elo"].get(str(r)) is not None]
                done.sort(key=lambda k: -self.st["trials"][k]["elo"][str(r)])
                self.log(f"batch {b} rung {self.rungs[r] / 1e6:.0f}M: " +
                         ", ".join(f"t{k} {self.st['trials'][k]['elo'][str(r)]:.0f}" for k in done))
                keys = done[:max(1, math.ceil(len(done) / self.a.eta))]
            self.st["batches"] = b + 1
            self.save()
        self.report()

    def stop(self, *_):
        for pr in list(self.procs.values()) + [p for p, _ in self.evals.values()]:
            try:
                os.killpg(pr.pid, signal.SIGTERM)
            except OSError:
                pass
        sys.exit(0)

    def report(self):
        ts = self.st["trials"]
        print(f"\n{'trial':<7}{'base':<9}" + "".join(f"{r / 1e6:>7.0f}M" for r in self.rungs) + "   params")
        best = lambda t: max([(int(r), e) for r, e in t["elo"].items() if e is not None] or [(-1, -1e9)])
        for k, t in sorted(ts.items(), key=lambda kv: tuple(-x for x in best(kv[1]))):
            p = t["params"]
            print(f"t{k:<6}{(t.get('baseline') or ''):<9}" +
                  "".join(f"{t['elo'][str(r)]:>8.0f}" if t["elo"].get(str(r)) is not None else f"{'':>8}"
                          for r in range(len(self.rungs))) +
                  f"   lr {p['lr']:.1e} ent {p['ent_coef']:.1e} std {p['min_std']:.2f}/{p['init_std']:.2f} "
                  f"h {p['horizon']:.0f} lam {p['lam']:.2f} kill {p['kill']:.2f} pos {p['position']:.2f} "
                  f"bp {p['ball_progress']:.4f} rot {p['rotation_bonus']:.4f}/{p['rotation_anneal'] / 1e6:.0f}M "
                  f"gs {p['goal_spawn']:.2f} league {p['league_frac']:.2f}"
                  f"{' pole' if p['random_pole'] else ''}{' masked' if p['mask_match_info'] else ''}")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--name", default="search1")
    p.add_argument("--batch", type=int, default=24, help="trials per batch (screened at rung 1)")
    p.add_argument("--max-batches", type=int, default=1)
    p.add_argument("--rungs", default="10,30,90", help="env steps (millions) per rung")
    p.add_argument("--eta", type=float, default=3, help="keep the best 1/eta at each rung")
    p.add_argument("--slots", type=int, default=4, help="trials training at once")
    p.add_argument("--hosts", type=int, default=4)
    p.add_argument("--matches", type=int, default=24)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--games", type=int, default=48, help="gauntlet matches per member per rating")
    p.add_argument("--eval-procs", type=int, default=3)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--report", action="store_true")
    a = p.parse_args()
    s = Search(a)
    if a.report:
        return s.report()
    signal.signal(signal.SIGTERM, s.stop)
    signal.signal(signal.SIGINT, s.stop)
    s.run()


if __name__ == "__main__":
    main()
