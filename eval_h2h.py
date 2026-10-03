"""Head-to-head evaluation daemon: the run's latest checkpoint against reference opponents, in the
native simulator under the training rules (7 goals / 5 min, no sudden death: a tie at the end of time is a draw).

Every time the run saves a new checkpoint (or --every seconds, whichever is later) it plays --games
matches against each opponent, sides swapped half the time, and appends one line per opponent to
runs/<run>/h2h.jsonl, which the dashboard plots:

    same    <ref> at the same number of env steps (the ref pool snapshot closest to it, by the ref's log)
    latest  <ref>'s latest checkpoint
    hier    a manager run with the frozen worker it was trained with (e.g. manager2)

    .venv/bin/python -u eval_h2h.py --run farm2 --ref farm1 --hier manager2 >> runs/farm2/h2h.out 2>&1
"""
import argparse, bisect, csv, json, os, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bridge"))
from farm_env import OBS2_SIZE  # noqa: E402
from native_env import NativeHost  # noqa: E402
from train_ppo import net_from_ckpt  # noqa: E402

RUNS = os.path.join(HERE, "runs")
FORMAT = (7, 300.0)                       # 7 goals to win, 5 min (the training format since 2026-09-30)
MAX_STEPS = int(2 * FORMAT[1] * 60 / 4)   # safety cap: twice the round time at frame skip 4
# the evaluation env's reward only counts kills made (opponent deaths, incl. its suicides); with the net kill
# term (kills - own deaths) that gives kills and deaths separately
COUNT_KILLS = {"goal": 0.0, "kill": 1.0, "death": 0.0, "ball_progress": 0.0, "position": 0.0}
# ... counted on every step: the training reward rule goal_refund (masks kills between a goal and the respawn,
# takes kills back) must stay off here — these are statistics, not rewards
COUNT_KW = {"reward": COUNT_KILLS, "goal_refund": False}


def load_policy(path):
    ck = torch.load(path, weights_only=False)
    net, norm, _ = net_from_ckpt(ck)  # any architecture / action head (train_ppo.spec_from_state)
    return net, norm, ck


def sample(pol, O):
    net, norm = pol
    x = torch.as_tensor(norm(O[:, :len(norm.mean)]), dtype=torch.float32)
    with torch.no_grad():
        return net.sample_env(x).numpy()  # env actions (a gated head's gates applied)


def log_steps(run):
    """update -> env steps from a run's log.csv (sorted lists)."""
    ups, steps = [], []
    try:
        for r in csv.DictReader(open(os.path.join(RUNS, run, "log.csv"))):
            try:
                ups.append(int(r["update"])); steps.append(int(float(r["steps"])))
            except (TypeError, ValueError):
                pass
    except OSError:
        pass
    return ups, steps


def snapshot_at(ref, steps):
    """The ref pool snapshot trained for the number of env steps closest to `steps` (and its steps)."""
    ups, st = log_steps(ref)
    pool = os.path.join(RUNS, ref, "pool")
    best = None
    for f in sorted(os.listdir(pool)):
        if not (f.startswith("u") and f.endswith(".pt")):
            continue
        u = int(f[1:-3])
        i = bisect.bisect_left(ups, u)
        if i >= len(ups):
            continue
        d = abs(st[i] - steps)
        if best is None or d < best[0]:
            best = (d, os.path.join(pool, f), st[i], f[:-3])
    return best[1:] if best else None


def result(me_goals, me_kills, games, steps=None, kills_made=None):
    g = np.asarray(me_goals, float)
    r = {"games": games, "w": int((g > 0).sum()), "d": int((g == 0).sum()), "l": int((g < 0).sum()),
         "goal_diff": float(g.mean()), "kill_diff": float(np.mean(me_kills))}
    if steps is not None:  # match length (env steps at frame skip 4 = 15 per s): kill difference per minute
        r["minutes"] = float(np.mean(steps)) / 900.0
        r["kill_diff_per_min"] = float(np.sum(me_kills) / (np.sum(steps) / 900.0))
    if kills_made is not None:
        km = np.asarray(kills_made, float)
        r["kills"], r["deaths"] = float(km.mean()), float((km - np.asarray(me_kills, float)).mean())
        if steps is not None:
            mins = np.sum(steps) / 900.0
            r["kills_per_min"], r["deaths_per_min"] = float(km.sum() / mins), float((km - me_kills).sum() / mins)
    return r


def play_flat(me, opp, games, threads, seed, opp_legacy=False):
    """me, opp = (net, norm). Returns per-match (goal diff, kill diff) for `me`. opp_legacy: the opponent
    was trained before the dead-rider observation fix and sees dead riders the old way."""
    env = NativeHost(1, games, 4, sd_frac=0.0, seed=seed, threads=threads, end_on_sd=True,
                     match_format=FORMAT, max_steps=MAX_STEPS, **COUNT_KW)
    my = np.array([1 if k % 2 == 0 else 2 for k in range(games)])  # rider `me` drives in match k
    if opp_legacy:
        for k in range(games):
            env.set_legacy_obs(k, 1 << (2 - my[k]))  # the opponent's rider bit
    obs = env.reset_all()
    rows_me, rows_op = (np.arange(games), my - 1), (np.arange(games), 2 - my)
    tot = np.zeros((games, 4)); done = np.zeros(games, bool); res = np.zeros((games, 4))
    act = np.zeros((games, 2, 3), np.float32)
    while not done.all():
        act[rows_me] = np.clip(sample(me, obs[rows_me]), -1, 1)
        act[rows_op] = np.clip(sample(opp, obs[rows_op]), -1, 1)
        out = env.step(act)
        live = ~done
        tot[live, :2] += out["terms"][rows_me][live][:, :2]
        tot[live, 2] += 1
        tot[live, 3] += out["rew"][rows_me][live]  # COUNT_KILLS reward: the opponent's deaths = my kills
        ended = live & out["reset"]
        res[ended] = tot[ended]; done |= ended
        obs = out["obs"]
    env.close()
    return res


def play_hier(me, hier_run, games, threads, seed, legacy=False):
    """`me` (flat) against a manager run driving its frozen worker (legacy: both trained before the
    dead-rider observation fix)."""
    import hier_env
    mnet, mnorm, mck = load_policy(os.path.join(RUNS, hier_run, "ckpt.pt"))
    ha = mck["args"]
    wpath = ha["hier"] if os.path.isabs(ha["hier"]) else os.path.join(HERE, ha["hier"])
    env = hier_env.HierHost(wpath, macro=ha.get("macro", 8), n_hosts=1, matches=games, sd_frac=0.0,
                            seed=seed, threads=threads, end_on_sd=True, match_format=FORMAT, max_steps=MAX_STEPS,
                            **COUNT_KW)
    my = np.array([1 if k % 2 == 0 else 2 for k in range(games)])
    for k in range(games):
        env.set_flat(k, int(my[k]), (me[0], me[1], "me"))
        if legacy:
            env.env.set_legacy_obs(k, 1 << (2 - my[k]))
    obs = env.reset_all()
    rows_me = (np.arange(games), my - 1)
    tot = np.zeros((games, 4)); done = np.zeros(games, bool); res = np.zeros((games, 4))
    while not done.all():
        ma = np.clip(sample((mnet, mnorm), obs.reshape(2 * games, OBS2_SIZE)), -1, 1).reshape(games, 2, -1)
        out = env.step(ma)  # the flat rows' manager actions are ignored
        live = ~done
        tot[live, :2] += out["terms"][rows_me][live][:, :2]
        tot[live, 2] += 1
        tot[live, 3] += out["rew"][rows_me][live]  # COUNT_KILLS reward: the opponent's deaths = my kills
        ended = live & out["reset"]
        res[ended] = tot[ended]; done |= ended
        obs = out["obs"]
    env.close()
    return res


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--run", default="farm2")
    p.add_argument("--ref", default="farm1", help="flat reference run (same-steps snapshot + latest)")
    p.add_argument("--hier", default="manager2", help="manager run (with its own frozen worker); '' = off")
    p.add_argument("--legacy", default="farm1,manager2",
                   help="runs trained before the dead-rider observation fix: they see dead riders the old way")
    p.add_argument("--kinds", default="same,latest,hier", help="which opponents: same, latest, hier")
    p.add_argument("--games", type=int, default=32, help="matches per opponent per round")
    p.add_argument("--threads", type=int, default=2)
    p.add_argument("--every", type=float, default=300, help="min seconds between rounds")
    p.add_argument("--once", action="store_true")
    a = p.parse_args()
    torch.set_num_threads(1)
    legacy = set(filter(None, a.legacy.split(",")))
    out_path = os.path.join(RUNS, a.run, "h2h.jsonl")
    last_mt, rnd = None, 0
    while True:
        ck_path = os.path.join(RUNS, a.run, "ckpt.pt")
        try:
            mt = os.path.getmtime(ck_path)
            if mt == last_mt:
                raise FileNotFoundError
            net, norm, ck = load_policy(ck_path)
        except Exception:  # noqa: BLE001 — not there yet / mid-write / unchanged
            time.sleep(20)
            continue
        last_mt, t0 = mt, time.time()
        me, steps, upd = (net, norm), int(ck.get("steps", 0)), int(ck.get("update", 0))
        opps = []
        kinds = set(a.kinds.split(","))
        snap = snapshot_at(a.ref, steps) if "same" in kinds else None
        if snap:
            path, ref_steps, name = snap
            opps.append(("same", f"{a.ref} {name}", ref_steps, path))
        if "latest" in kinds:
            opps.append(("latest", f"{a.ref} latest", None, os.path.join(RUNS, a.ref, "ckpt.pt")))
        if a.hier and "hier" in kinds:
            opps.append(("hier", a.hier, None, None))
        for kind, name, ref_steps, path in opps:
            seed = 10_000 + 97 * rnd + len(kind)
            try:
                if kind == "hier":
                    res = play_hier(me, a.hier, a.games, a.threads, seed, a.hier in legacy)
                else:
                    onet, onorm, _ = load_policy(path)
                    res = play_flat(me, (onet, onorm), a.games, a.threads, seed, a.ref in legacy)
            except Exception as e:  # noqa: BLE001
                print(f"{kind} {name}: {e!r}", flush=True)
                continue
            r = {"t": time.time(), "update": upd, "steps": steps, "kind": kind, "opp": name,
                 "opp_steps": ref_steps, **result(res[:, 0], res[:, 1], a.games, res[:, 2], res[:, 3])}
            with open(out_path, "a") as f:
                f.write(json.dumps(r) + "\n")
            print(f"u{upd} {steps / 1e6:.1f}M vs {name}: {r['w']}-{r['d']}-{r['l']} goal diff {r['goal_diff']:+.2f} "
                  f"kills {r['kills']:.2f} deaths {r['deaths']:.2f} per match ({r['kills_per_min']:.2f} / "
                  f"{r['deaths_per_min']:.2f} per min, {r['minutes']:.1f} min)",
                  flush=True)
        rnd += 1
        if a.once:
            break
        time.sleep(max(0.0, a.every - (time.time() - t0)))


if __name__ == "__main__":
    main()
