#!/usr/bin/env python3
"""The gauntlet: a fixed set of frozen opponents and one rating scale for every policy we train.

Members are frozen copies of reference policies: farm2 and farm3 at several training stages (the "rungs", so
weak young policies still get a graded score) and latest, killer2, manager2 (hierarchical), the best PBT
member. Every pairing of members is played once (--games matches, sides alternating, native simulator under
the training rules: 7 goals / 5 min, no sudden death, a tie at the end of time is a draw), and the members'
Elo ratings are fitted to those results (Bradley-Terry on win/draw/loss, draws = half, a weak prior so 64-0
pairings stay finite; farm2 latest = 1500). A candidate then plays every member and gets its own Elo with the
members' ratings held fixed, so ratings of candidates rated at different times stay comparable.

    .venv/bin/python gauntlet.py init                     # freeze the default members
    .venv/bin/python gauntlet.py fill                     # play the missing member pairings, fit
    .venv/bin/python gauntlet.py rate CKPT --name X       # X vs every member -> X's Elo (--add: X joins)
    .venv/bin/python gauntlet.py table                    # members + rated candidates
    .venv/bin/python gauntlet.py exploit X --steps 30e6   # exploitability: train a fresh exploiter against X

Exploitability: an exploiter starts as a copy of X and trains only against frozen X (--league-frac 1) for
--steps env steps; its goal difference against X afterwards is X's exploitability (0 = nothing found).

Files: runs/gauntlet/members.json, ckpt/<name>.pt (frozen copies), games.jsonl (one line per pairing played),
candidates.json, exploit.jsonl.
"""
import argparse, json, math, os, shutil, subprocess, sys, time
from concurrent.futures import ProcessPoolExecutor

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
RUNS = os.path.join(HERE, "runs")
DIR = os.path.join(RUNS, "gauntlet")
PY = os.path.join(HERE, ".venv", "bin", "python")
ANCHOR, ANCHOR_ELO, PRIOR_SD = "farm2", 1500.0, 600.0

# name: (source run, env steps of the pool snapshot; None = latest checkpoint) — flat policies
DEFAULT_FLAT = {
    "farm2@5M": ("farm2", 5e6), "farm2@10M": ("farm2", 10e6), "farm2@25M": ("farm2", 25e6),
    "farm2@50M": ("farm2", 50e6), "farm2@100M": ("farm2", 100e6), "farm2@200M": ("farm2", 200e6),
    "farm2": ("farm2", None),
    "farm3@10M": ("farm3", 10e6), "farm3@50M": ("farm3", 50e6), "farm3@200M": ("farm3", 200e6),
    "farm3@400M": ("farm3", 400e6), "farm3": ("farm3", None),
    "killer2": ("killer2", None),
}
DEFAULT_HIER = {"manager2": "manager2"}  # name: run (its own frozen worker; trained before the obs fix)
PBT_BEST = ("pbt1_m1", os.path.join(RUNS, "pbt1"))  # added from the PBT's last tournament snapshot


def jload(path, default):
    try:
        return json.load(open(path))
    except (OSError, ValueError):
        return default


def jupdate(path, fn):
    """Read-modify-write a JSON file under an exclusive lock (several ratings may finish at once)."""
    import fcntl
    with open(path + ".lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        obj = jload(path, {})
        fn(obj)
        jsave(path, obj)
        return obj


def jsave(path, obj):
    tmp = path + ".tmp"
    json.dump(obj, open(tmp, "w"), indent=1)
    os.replace(tmp, path)


def ckpt_steps(path):
    import torch
    return int(torch.load(path, weights_only=False).get("steps", 0))


# ------------------------------------------------------------------ playing
def _policy(m):
    from eval_h2h import load_policy
    return load_policy(m["path"])[:2]


def play_pair(a, b, games, seed):
    """a, b: member/candidate dicts. Returns eval_h2h.result for `a` (plus a, b names)."""
    import torch
    torch.set_num_threads(1)
    from eval_h2h import play_flat, play_hier, result
    flip = a["kind"] == "hier"  # the hierarchical side is always the opponent in play_hier
    me, opp = (b, a) if flip else (a, b)
    if me["kind"] == "hier":
        raise ValueError("hier vs hier is not supported")
    if me.get("legacy"):
        raise ValueError(f"{me['name']}: legacy observations only supported on the opponent side")
    if opp["kind"] == "hier":
        res = play_hier(_policy(me), opp["run"], games, 1, seed, opp.get("legacy", False))
    else:
        res = play_flat(_policy(me), _policy(opp), games, 1, seed, opp.get("legacy", False))
    if flip:  # back to a's view: goal and kill difference change sign; a's kills = b's deaths
        kills_b, kd_b = res[:, 3].copy(), res[:, 1].copy()
        res = res.copy()
        res[:, 0] *= -1
        res[:, 1] = -kd_b
        res[:, 3] = kills_b - kd_b
    r = result(res[:, 0], res[:, 1], games, res[:, 2], res[:, 3])
    return {"a": a["name"], "b": b["name"], "seed": seed, "t": time.time(), **r}


def play_many(pairs, games, procs):
    """pairs: [(a, b)]. Plays them in parallel worker processes; appends each to games.jsonl as it ends."""
    out = []
    path = os.path.join(DIR, "games.jsonl")
    with ProcessPoolExecutor(max_workers=procs) as ex:
        futs = {ex.submit(play_pair, a, b, games, 1000 + 7919 * i): (a, b) for i, (a, b) in enumerate(pairs)}
        from concurrent.futures import as_completed
        for f in as_completed(futs):
            a, b = futs[f]
            try:
                r = f.result()
            except Exception as e:  # noqa: BLE001
                print(f"  {a['name']} vs {b['name']}: {e!r}", flush=True)
                continue
            with open(path, "a") as fh:
                fh.write(json.dumps(r) + "\n")
            out.append(r)
            print(f"  {r['a']:>12} vs {r['b']:<12} {r['w']:>3}-{r['d']}-{r['l']:<3} goals {r['goal_diff']:+.2f}",
                  flush=True)
    return out


def games_log():
    rows = []
    try:
        for line in open(os.path.join(DIR, "games.jsonl")):
            rows.append(json.loads(line))
    except OSError:
        pass
    return rows


# ------------------------------------------------------------------ ratings
def fit(rows, names, fixed=None):
    """Bradley-Terry Elo from pairing rows (w/d/l of a vs b). fixed: {name: elo} held constant (candidates are
    fitted against the members this way). Returns {name: elo}."""
    import torch
    fixed = fixed or {}
    free = [n for n in names if n not in fixed]
    idx = {n: i for i, n in enumerate(free)}
    data = [(r["a"], r["b"], r["w"] + 0.5 * r["d"], r["games"]) for r in rows
            if (r["a"] in idx or r["a"] in fixed) and (r["b"] in idx or r["b"] in fixed)
            and (r["a"] in idx or r["b"] in idx)]
    if not free:
        return dict(fixed)
    base = np.mean(list(fixed.values())) if fixed else ANCHOR_ELO
    x = torch.zeros(len(free), dtype=torch.float64, requires_grad=True)
    opt = torch.optim.LBFGS([x], max_iter=500, tolerance_grad=1e-10, line_search_fn="strong_wolfe")
    k = math.log(10) / 400

    def elo(n):
        return x[idx[n]] if n in idx else torch.tensor(fixed[n] - base, dtype=torch.float64)

    def closure():
        opt.zero_grad()
        loss = (x ** 2).sum() / (2 * PRIOR_SD ** 2)
        for a, b, s, n in data:
            d = k * (elo(a) - elo(b))
            loss = loss - (s * torch.nn.functional.logsigmoid(d) + (n - s) * torch.nn.functional.logsigmoid(-d))
        loss.backward()
        return loss

    opt.step(closure)
    out = {n: float(x[idx[n]].detach()) + base for n in free}
    out.update(fixed)
    if not fixed and ANCHOR in out:  # members: farm2 latest = 1500
        shift = ANCHOR_ELO - out[ANCHOR]
        out = {n: v + shift for n, v in out.items()}
    return out


def member_ratings():
    mem = jload(os.path.join(DIR, "members.json"), {})
    rows = [r for r in games_log() if r["a"] in mem and r["b"] in mem]
    return fit(rows, list(mem))


def latest_rows(rows):
    """The most recent row per (a, b) pairing (a re-rated candidate replaces its old results)."""
    last = {}
    for r in rows:
        last[(r["a"], r["b"])] = r
    return list(last.values())


def rate_candidate(name, ratings=None):
    ratings = ratings or member_ratings()
    rows = [r for r in latest_rows(games_log()) if r["a"] == name and r["b"] in ratings]
    if not rows:
        return None, rows
    return fit(rows, [name] + [r["b"] for r in rows], fixed={r["b"]: ratings[r["b"]] for r in rows})[name], rows


# ------------------------------------------------------------------ commands
def cmd_init(a):
    import torch
    from eval_h2h import snapshot_at
    os.makedirs(os.path.join(DIR, "ckpt"), exist_ok=True)
    mem = jload(os.path.join(DIR, "members.json"), {})
    for name, (run, steps) in DEFAULT_FLAT.items():
        if name in mem:
            continue
        src = os.path.join(RUNS, run, "ckpt.pt") if steps is None else snapshot_at(run, steps)[0]
        dst = os.path.join(DIR, "ckpt", name.replace("@", "_") + ".pt")
        shutil.copy(src, dst)
        st = ckpt_steps(dst) if steps is None else int(snapshot_at(run, steps)[1])
        mem[name] = {"name": name, "kind": "flat", "path": dst, "src": src, "run": run, "steps": st}
        print(f"{name}: {src} ({st / 1e6:.0f}M steps)")
    for name, run in DEFAULT_HIER.items():
        if name not in mem:
            mem[name] = {"name": name, "kind": "hier", "run": run, "legacy": True, "path": None,
                         "src": os.path.join(RUNS, run, "ckpt.pt"), "steps": ckpt_steps(os.path.join(RUNS, run, "ckpt.pt"))}
            print(f"{name}: hierarchical ({run} + its frozen worker)")
    run, pdir = PBT_BEST
    st = jload(os.path.join(pdir, "state.json"), {})
    if run not in mem and st.get("generations"):
        g = len(st["generations"]) - 1
        src = os.path.join(pdir, f"gen{g}", run.split("_")[-1] + ".pt")
        if os.path.exists(src):
            dst = os.path.join(DIR, "ckpt", run + ".pt")
            shutil.copy(src, dst)
            mem[run] = {"name": run, "kind": "flat", "path": dst, "src": src, "run": run, "steps": ckpt_steps(dst)}
            print(f"{run}: PBT generation {g} snapshot ({mem[run]['steps'] / 1e6:.0f}M steps)")
    jsave(os.path.join(DIR, "members.json"), mem)


def cmd_fill(a):
    mem = jload(os.path.join(DIR, "members.json"), {})
    done = {(r["a"], r["b"]) for r in games_log()} | {(r["b"], r["a"]) for r in games_log()}
    names = list(mem)
    pairs = [(mem[x], mem[y]) for i, x in enumerate(names) for y in names[i + 1:] if (x, y) not in done]
    print(f"{len(pairs)} member pairings to play ({a.games} matches each, {a.procs} processes)", flush=True)
    if pairs:
        play_many(pairs, a.games, a.procs)
    cmd_table(a)


def cmd_rate(a):
    mem = jload(os.path.join(DIR, "members.json"), {})
    path = os.path.abspath(a.ckpt if a.ckpt.endswith(".pt") else os.path.join(RUNS, a.ckpt, "ckpt.pt"))
    name = a.name or os.path.basename(os.path.dirname(path))
    frozen = os.path.join(DIR, "ckpt", "cand_" + name.replace("@", "_").replace("/", "_") + ".pt")
    os.makedirs(os.path.dirname(frozen), exist_ok=True)
    shutil.copy(path, frozen)  # rate (and keep) exactly this version
    me = {"name": name, "kind": "flat", "path": frozen, "src": path, "steps": ckpt_steps(frozen)}
    opps = [m for n, m in mem.items() if n != name and (not a.only or n in a.only.split(","))]
    print(f"{name} ({me['steps'] / 1e6:.0f}M steps) vs {len(opps)} members, {a.games} matches each", flush=True)
    play_many([(me, o) for o in opps], a.games, a.procs)
    ratings = member_ratings()
    elo, rows = rate_candidate(name, ratings)
    me.update({"elo": elo, "t": time.time()})
    jupdate(os.path.join(DIR, "candidates.json"), lambda c: c.__setitem__(name, me))
    print(f"\n{name}: Elo {elo:.0f}  (farm2 latest = {ANCHOR_ELO:.0f})")
    if a.add:
        mem[name] = {k: v for k, v in me.items() if k not in ("elo", "t")}
        jsave(os.path.join(DIR, "members.json"), mem)
        print(f"{name} added to the gauntlet")
    return elo


def cmd_table(a):
    mem = jload(os.path.join(DIR, "members.json"), {})
    ratings = member_ratings()
    rows = latest_rows(games_log())
    gd = {}
    for r in rows:
        gd[(r["a"], r["b"])] = r["goal_diff"]
        gd[(r["b"], r["a"])] = -r["goal_diff"]
    print(f"\n{'member':<14}{'steps':>7}{'Elo':>7}   goal diff vs farm2 / farm3 latest")
    for n in sorted(mem, key=lambda n: -ratings.get(n, 0)):
        f2, f3 = gd.get((n, "farm2")), gd.get((n, "farm3"))
        print(f"{n:<14}{mem[n].get('steps', 0) / 1e6:>6.0f}M{ratings.get(n, float('nan')):>7.0f}   "
              f"{'' if f2 is None else f'{f2:+.2f}':>6} / {'' if f3 is None else f'{f3:+.2f}'}")
    cands = jload(os.path.join(DIR, "candidates.json"), {})
    cands = {n: c for n, c in cands.items() if n not in mem}
    if cands:
        print(f"\n{'candidate':<20}{'steps':>7}{'Elo':>7}   goal diff vs farm2 / farm3 latest")
        for n, c in sorted(cands.items(), key=lambda kv: -(rate_candidate(kv[0], ratings)[0] or 0)):
            elo = rate_candidate(n, ratings)[0]
            f2, f3 = gd.get((n, "farm2")), gd.get((n, "farm3"))
            print(f"{n:<20}{c.get('steps', 0) / 1e6:>6.0f}M{elo:>7.0f}   "
                  f"{'' if f2 is None else f'{f2:+.2f}':>6} / {'' if f3 is None else f'{f3:+.2f}'}")


def cmd_exploit(a):
    """Exploiter run exploit_<name>: a copy of the target trained only against the frozen target."""
    mem = jload(os.path.join(DIR, "members.json"), {})
    cands = jload(os.path.join(DIR, "candidates.json"), {})
    tgt = mem.get(a.name) or cands.get(a.name)
    if not tgt or tgt["kind"] != "flat":
        sys.exit(f"{a.name}: not a flat member/candidate of the gauntlet (rate it first)")
    run = f"exploit_{a.name.replace('@', '_')}"
    rdir = os.path.join(RUNS, run)
    if os.path.isdir(rdir) and not a.resume:
        shutil.rmtree(rdir)
    cmd = [PY, "-u", "train_farm.py", "--run", run, "--native", "--hosts", str(a.hosts), "--matches", str(a.matches),
           "--procs", str(a.train_procs), "--threads", str(os.cpu_count()), "--torch-threads", "1", "--arena", "2",
           "--no-sudden-death", "--format", "7,300", "--goal-episodes", "--pit-frac", "0",
           "--league-frac", "1", "--league-fixed", f"target:{tgt['path']}:1.0", "--min-std", "0.1",
           "--total-steps", str(int(a.steps))]
    cmd += ["--resume"] if a.resume else ["--init-from", tgt["path"]]
    os.makedirs(rdir, exist_ok=True)
    print(" ".join(cmd), flush=True)
    with open(os.path.join(rdir, "train.out"), "a") as log:
        rc = subprocess.call(cmd, cwd=HERE, stdout=log, stderr=subprocess.STDOUT)
    if rc:
        sys.exit(f"exploiter training failed (rc {rc}), see {rdir}/train.out")
    ex = {"name": run, "kind": "flat", "path": os.path.join(rdir, "ckpt.pt")}
    r = play_many([(ex, tgt)], a.games, 1)[0]
    rec = {"target": a.name, "exploiter": run, "steps": a.steps, "t": time.time(), **r}
    with open(os.path.join(DIR, "exploit.jsonl"), "a") as fh:
        fh.write(json.dumps(rec) + "\n")
    print(f"\n{a.name} exploitability after {a.steps / 1e6:.0f}M exploiter steps: "
          f"{r['goal_diff']:+.2f} goals/match ({r['w']}-{r['d']}-{r['l']})")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)
    sub.add_parser("init")
    for n in ("fill", "rate", "table", "exploit"):
        s = sub.add_parser(n)
        s.add_argument("--games", type=int, default=64)
        s.add_argument("--procs", type=int, default=8)
        if n == "rate":
            s.add_argument("ckpt", help="checkpoint path or run name")
            s.add_argument("--name", default=None)
            s.add_argument("--add", action="store_true", help="make it a gauntlet member")
            s.add_argument("--only", default=None, help="comma-separated members to play (default: all)")
        if n == "exploit":
            s.add_argument("name")
            s.add_argument("--steps", type=float, default=30e6)
            s.add_argument("--hosts", type=int, default=12)
            s.add_argument("--matches", type=int, default=38)
            s.add_argument("--train-procs", type=int, default=6)
            s.add_argument("--resume", action="store_true")
    a = p.parse_args()
    os.makedirs(DIR, exist_ok=True)
    sys.path.insert(0, HERE)
    {"init": cmd_init, "fill": cmd_fill, "rate": cmd_rate, "table": cmd_table, "exploit": cmd_exploit}[a.cmd](a)


if __name__ == "__main__":
    main()
