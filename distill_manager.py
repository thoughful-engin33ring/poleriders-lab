#!/usr/bin/env python3
"""Behaviour-cloning warm start for a --hier manager: distil the flat policy (farm1) into the
manager's target space (bridge/hier_env.py), so manager PPO starts from "describe what farm1 does".

    .venv/bin/python distill_manager.py all                 # collect -> train -> eval (defaults below)
    .venv/bin/python distill_manager.py collect --steps 6000  # 6000 x 128 matches x 2 = 1.54M agent-steps
    .venv/bin/python distill_manager.py train                 # --a3-rule body_soft (default), still_tip, both, ...
    .venv/bin/python distill_manager.py eval  --eval-steps 300 [--eval-pit-frac 0] [--no-baselines]

collect  Runs farm1 (both riders, sampled actions as in training) on NativeHost (sd_frac 0.15,
         pit_frac 0.3 like training) and labels every agent-step t with the manager action that
         describes farm1's next H=8 steps (hier_env.set_goals inverted, agent frame):
             a0 = clip((x[t+8] - x[t]) / 16)                       body target dx
             a1 = clip((tipx[t+8] - x[t]) / 8)                     tip target dx
             a2 = clip((tipy[t+8] - (y[t] - 0.69) - 5) / 4)        tip target height
         tip = body + (obs[13], obs[14]). Windows are dropped if the rider is a ghost (obs[9] > .5)
         at t or t+8, the match resets inside the window, or x jumps by > 3 m in one step (teleport).
         -> runs/distill/data.npz  (obs, lab, match/t/agent row index), runs/distill/farm1_rollout.npz
         (the raw time-major rollout incl. farm1's actions and the masks), runs/distill/flat_src.pt
         (snapshot of the flat checkpoint used; train/eval read it instead of the live ckpt.pt)
train    Manager ActorCritic(62, 4) initialised exactly like `train_farm.py --init-from farm1`
         (critic + pi.0/pi.2 + norm copied, fresh output layer), then net.pi (all layers) fitted to
         the labels by MSE; 10% of MATCHES held out (overlapping windows of one match are strongly
         correlated, a random 10% of rows would leak). a3 comes from --a3-rule (see A3_RULES).
         log_std = log(0.3). -> runs/distill/manager_init.pt, a train_farm checkpoint
         (net, opt = fresh Adam state, norm, steps 0, update 0, args with hier/macro).
eval     HierHost + the frozen worker: (a) label reproduction: the distilled manager's mean actions
         (a3 per rule / forced modes) through the worker; achieved 8-step body/tip displacement vs
         commanded; (b) self-play quality: goals/kills per minute of game time, good side of the ball,
         enemy/own pit share, for the distilled manager, a random-init (--init-from style) manager and
         the flat farm1 policy as reference.

Use in training: copy manager_init.pt to runs/<new>/ckpt.pt and run
    train_farm.py --run <new> --resume --native --hier runs/manager2/worker.pt --macro 8 ...
(resume loads net/opt/norm/steps/update; a fresh Adam state_dict loads fine).
"""
import argparse, json, math, os, shutil, sys, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bridge"))
from farm_env import OBS2_SIZE  # noqa: E402
from native_env import NativeHost  # noqa: E402
from train_ppo import ActorCritic, RunningNorm  # noqa: E402

D, A = OBS2_SIZE, 4
MACRO = 8
TICK_S = 4 / 60  # one env step (frame skip 4)
X_MAX = 19.8
BODY_R, TIP_R = 0.75, 1.0  # the worker's "reached" radii (train_worker.py)


# ---------------------------------------------------------------- labelling

def hindsight(O0, O1):
    """Manager action (a0, a1, a2) whose targets are exactly where body/tip ended up (O1) from O0."""
    x0, y0 = O0[..., 0], O0[..., 1]
    x1, y1 = O1[..., 0], O1[..., 1]
    tx1, ty1 = x1 + O1[..., 13], y1 + O1[..., 14]
    a0 = (x1 - x0) / 16.0
    a1 = (tx1 - x0) / 8.0
    a2 = (ty1 - (y0 - 0.69) - 5.0) / 4.0
    return np.clip(np.stack([a0, a1, a2], -1), -1, 1).astype(np.float32)


# a3 rules: functions of (a0, a1, a2) [..., 3] (labels at train time, the net's own outputs at eval)
def _still(a):
    return np.abs(16.0 * a[..., 0]) < BODY_R  # body target already within the worker's reach radius


A3_RULES = {
    "both": lambda a: np.zeros(a.shape[:-1], np.float32),
    "body": lambda a: -np.ones(a.shape[:-1], np.float32),
    "tip": lambda a: np.ones(a.shape[:-1], np.float32),
    # body barely moves -> tip only (don't pin the body to a spot it's already on), else both
    "still_tip": lambda a: np.where(_still(a), 1.0, 0.0).astype(np.float32),
    # same modes as still_tip, but "both" is labelled -0.3 (still inside the both band): an MSE head
    # predicts p(still) * 1 + (1 - p) * -0.3, which crosses the tip threshold 1/3 at p ~= 0.49 instead
    # of p = 1/3 with 0/1 labels (the net then picks tip-only only when "still" is the likelier case)
    "still_tip_bal": lambda a: np.where(_still(a), 1.0, -0.3).astype(np.float32),
    # body-only targets (the worker executes them best in play: no stalls in pits, see the report in
    # runs/distill), labelled -0.6 rather than -1: still well inside the body band (< -1/3) for the mean
    # action, but PPO's std-0.3 exploration then tries "both" ~18% of the time instead of ~1%
    "body_soft": lambda a: np.full(a.shape[:-1], -0.6, np.float32),
    # body barely moves -> both (hold it there) ; body moves -> body only
    "move_body": lambda a: np.where(_still(a), 0.0, -1.0).astype(np.float32),
}


def load_flat(path):
    ck = torch.load(path, weights_only=False)
    net, norm = ActorCritic(D, 3), RunningNorm(D)
    net.load_state_dict(ck["net"]); norm.load(ck["norm"])
    return net, norm, ck


def window_any(flag, H):
    """flag [T, ...] bool -> [T-H+1, ...]: any(flag[t:t+H])."""
    c = np.concatenate([np.zeros((1,) + flag.shape[1:], np.int32), np.cumsum(flag, 0, dtype=np.int32)])
    return (c[H:] - c[:-H]) > 0


def flat_src(args):
    """The flat checkpoint the data was collected with (collect snapshots it: farm1's ckpt.pt keeps
    changing while it trains); falls back to --flat."""
    snap = os.path.join(os.path.dirname(args.data), "flat_src.pt")
    return snap if os.path.exists(snap) else args.flat


def collect(args):
    snap = os.path.join(os.path.dirname(args.data), "flat_src.pt")
    os.makedirs(os.path.dirname(snap), exist_ok=True)
    shutil.copyfile(args.flat, snap)  # ckpt.pt is replaced atomically by its trainer: the copy is whole
    net, norm, ck = load_flat(snap)
    print(f"flat policy {args.flat} (snapshot {snap}): {ck['steps']} steps, std {net.log_std.exp().tolist()}", flush=True)
    env = NativeHost(args.hosts, args.matches, 4, sd_frac=0.15, max_steps=5400, seed=args.seed,
                     threads=args.threads, pit_frac=0.3)
    N, T, H = env.n, args.steps, MACRO
    torch.manual_seed(args.seed)
    obs = np.zeros((T + 1, N, 2, D), np.float32)
    act = np.zeros((T, N, 2, 3), np.float32)
    reset = np.zeros((T, N), bool)
    obs[0] = env.reset_all()
    t0 = time.time()
    for t in range(T):
        with torch.no_grad():
            a = net.dist(torch.as_tensor(norm(obs[t].reshape(-1, D)), dtype=torch.float32)).sample()
        act[t] = np.clip(a.numpy(), -1, 1).reshape(N, 2, 3)  # agent frame, exactly what the env gets
        out = env.step(act[t])
        obs[t + 1] = out["obs"]
        reset[t] = out["reset"]
        if (t + 1) % 500 == 0:
            print(f"  step {t + 1}/{T}  {(t + 1) * N / (time.time() - t0):.0f} match-steps/s", flush=True)
    env.close()
    x, ghost = obs[..., 0], obs[..., 9] > 0.5
    jump = np.abs(x[1:] - x[:-1]) > 3.0                              # [T, N, 2]
    L = T - H + 1                                                    # windows t = 0..T-H
    bad = window_any(np.repeat(reset[..., None], 2, -1), H)[:L] | window_any(jump, H)[:L]
    bad |= ghost[:L] | ghost[H:H + L]
    ok = ~bad                                                        # [L, N, 2]
    O0, O1 = obs[:L][ok], obs[H:H + L][ok]
    lab = hindsight(O0, O1)
    idx = np.nonzero(ok)  # (t, match, agent) of every kept window: rows of data.npz -> farm1_rollout.npz
    match = idx[1].astype(np.int32)
    raw = L * N * 2
    stats = {"agent_steps": T * N * 2, "windows": raw, "kept": int(ok.sum()),
             "dropped_reset": int(window_any(np.repeat(reset[..., None], 2, -1), H)[:L].sum()),
             "dropped_teleport": int(window_any(jump, H)[:L].sum()),
             "dropped_ghost": int((ghost[:L] | ghost[H:H + L]).sum()),
             "flat_ckpt_steps": int(ck["steps"])}
    print(json.dumps(stats), flush=True)
    os.makedirs(os.path.dirname(args.data), exist_ok=True)
    np.savez(args.data, obs=O0, lab=lab, match=match, t=idx[0].astype(np.int32), agent=idx[2].astype(np.int8),
             stats=json.dumps(stats))
    # the raw rollout (e.g. for behaviour-cloning the worker): time-major, window [t0, t0+8] = obs[t0:t0+9]
    np.savez(os.path.join(os.path.dirname(args.data), "farm1_rollout.npz"),
             obs=obs,                    # [T+1, N, 2, 62] observe2 (agent frame); obs[t] seen before action t
             act=act,                    # [T, N, 2, 3] farm1's sampled action at t, clipped, agent frame
             reset=reset,                # [T, N] step t ended the match: obs[t+1] is a new match's first obs
             ghost=ghost,                # [T+1, N, 2] obs[..., 9] > 0.5 (dead rider)
             teleport=jump,              # [T, N, 2] |x[t+1] - x[t]| > 3 m
             window_ok=ok,               # [T-7, N, 2] the manager-label window mask used for data.npz
             step=np.arange(T + 1, dtype=np.int32),
             meta=json.dumps(dict(stats, hosts=args.hosts, matches=args.matches, seed=args.seed, sd_frac=0.15,
                                  pit_frac=0.3, frame_skip=4, macro=H, flat=args.flat, flat_src=snap)))
    label_stats(lab)
    return stats


def label_stats(lab):
    names = ["a0 body dx/16", "a1 tip dx/8", "a2 tip h"]
    for i, n in enumerate(names):
        v = lab[:, i]
        q = np.percentile(v, [1, 10, 50, 90, 99])
        print(f"  {n:14s} mean {v.mean():+.3f} std {v.std():.3f}  p1/10/50/90/99 "
              + " ".join(f"{e:+.2f}" for e in q) + f"  |.|=1 (clipped) {np.mean(np.abs(v) >= 1):.1%}")
    still = _still(lab)
    print(f"  body moves < {BODY_R} m in 8 steps: {still.mean():.1%} of windows; "
          f"|body dx| mean {np.abs(16 * lab[:, 0]).mean():.2f} m", flush=True)


# ---------------------------------------------------------------- training

def fresh_manager(flat_ck, seed):
    """Exactly train_farm.py --init-from: critic + pi.0/pi.2 + norm from the flat policy."""
    torch.manual_seed(seed)
    net, norm = ActorCritic(D, A), RunningNorm(D)
    sd, own = flat_ck["net"], net.state_dict()
    for k in sd:
        if k.startswith("v.") or k.startswith("pi.0.") or k.startswith("pi.2."):
            own[k] = sd[k]
    net.load_state_dict(own)
    norm.load(flat_ck["norm"])
    return net, norm


def train_args(args):
    """A train_farm.py-shaped args dict (the dashboard reads hier/macro from it)."""
    return {"run": "distill", "hosts": 14, "matches": 16, "arena": 2, "frame_skip": 4, "episode_steps": 5400,
            "sd_frac": 0.15, "hier": args.worker, "macro": MACRO, "flat_league": None, "flat_frac": 0.5,
            "init_from": args.flat, "min_std": 0.0, "pit_frac": 0.3, "total_steps": 2_000_000_000,
            "rollout": 64, "epochs": 4, "minibatch": 4096, "lr": 3e-4, "gamma": 0.998 ** MACRO, "lam": 0.95,
            "clip": 0.2, "ent_coef": 0.001, "vf_coef": 0.5, "league_frac": 0.5, "pool_every": 25,
            "save_every": 5, "seed": 0, "resume": True, "watch": False, "deterministic": False,
            "native": True, "threads": 8, "pipeline": True}


def train(args):
    z = np.load(args.data)
    obs, lab, match = z["obs"], z["lab"], z["match"]
    y = np.concatenate([lab, A3_RULES[args.a3_rule](lab)[:, None]], 1).astype(np.float32)
    val = match % 10 == 3  # hold out 10% of the matches
    flat_ck = torch.load(flat_src(args), weights_only=False)
    net, norm = fresh_manager(flat_ck, args.seed)
    X = torch.as_tensor(norm(obs).astype(np.float32))
    Y = torch.as_tensor(y)
    tr, va = torch.as_tensor(~val), torch.as_tensor(val)
    Xt, Yt, Xv, Yv = X[tr], Y[tr], X[va], Y[va]
    print(f"train {len(Xt)} / val {len(Xv)} rows (a3 rule '{args.a3_rule}': "
          f"body {np.mean(y[:, 3] < -1/3):.1%} both {np.mean(np.abs(y[:, 3]) <= 1/3):.1%} "
          f"tip {np.mean(y[:, 3] > 1/3):.1%})", flush=True)

    def val_loss():
        with torch.no_grad():
            p = torch.cat([net.pi(Xv[i:i + 65536]) for i in range(0, len(Xv), 65536)])
            e = (p - Yv).pow(2).mean(0)
            r2 = 1 - e / Yv.var(0).clamp(min=1e-8)
        return e.mean().item(), e.tolist(), r2.tolist()

    base = (Yv - Yt.mean(0)).pow(2).mean(0)
    print(f"  predict-the-mean baseline val MSE {base.mean().item():.4f} per dim "
          + " ".join(f"{v:.4f}" for v in base.tolist()), flush=True)
    m0 = val_loss()
    print(f"  init (fresh output layer) val MSE {m0[0]:.4f}", flush=True)
    opt = torch.optim.Adam(net.pi.parameters(), lr=args.lr)
    steps = args.epochs * math.ceil(len(Xt) / args.batch)
    sched = torch.optim.lr_scheduler.OneCycleLR(opt, max_lr=args.lr, total_steps=steps, pct_start=0.1)
    hist = []
    for ep in range(args.epochs):
        perm = torch.randperm(len(Xt))
        tl, nb = 0.0, 0
        for s in range(0, len(Xt), args.batch):
            idx = perm[s:s + args.batch]
            loss = (net.pi(Xt[idx]) - Yt[idx]).pow(2).mean()
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(net.pi.parameters(), 1.0)
            opt.step(); sched.step()
            tl += loss.item(); nb += 1
        vl, per, r2 = val_loss()
        hist.append({"epoch": ep + 1, "train": tl / nb, "val": vl, "val_per_dim": per, "val_r2": r2})
        print(f"  epoch {ep + 1}: train MSE {tl / nb:.4f}  val MSE {vl:.4f}  per dim "
              + " ".join(f"{v:.4f}" for v in per) + "  R2 " + " ".join(f"{v:.2f}" for v in r2), flush=True)
    with torch.no_grad():
        net.log_std.fill_(math.log(0.3))
    ck = {"net": net.state_dict(),
          "opt": torch.optim.Adam(net.parameters(), lr=3e-4, eps=1e-5).state_dict(),  # train_farm's optimizer
          "norm": norm.state(), "steps": 0, "update": 0, "args": train_args(args),
          "distill": {"flat": args.flat, "flat_src": flat_src(args), "flat_steps": int(flat_ck["steps"]), "a3_rule": args.a3_rule,
                      "data": args.data, "rows": int(len(obs)), "history": hist}}
    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    torch.save(ck, args.out + ".tmp")
    os.replace(args.out + ".tmp", args.out)
    print(f"saved {args.out}", flush=True)
    return hist


# ---------------------------------------------------------------- evaluation

class Play:
    """Self-play statistics: events per minute of game time, positioning of live riders."""

    def __init__(self):
        self.goals = self.kills = self.seconds = 0.0
        self.pos = np.zeros(4)

    def book(self, out, O, seconds):
        self.goals += np.abs(out["terms"][:, 0, 0]).sum()
        self.kills += np.abs(out["terms"][:, 0, 1]).sum()
        self.seconds += seconds * len(out["reset"])
        r = O[O[:, 9] < 0.5]
        self.pos += [len(r), (r[:, 0] < r[:, 52]).sum(), ((r[:, 0] > 16.3) & (r[:, 1] < 0)).sum(),
                     ((r[:, 0] < -16.3) & (r[:, 1] < 0)).sum()]

    def summary(self):
        n = max(self.pos[0], 1)
        return {k: round(float(v), 4) for k, v in (
            ("goals_per_min", 60 * self.goals / self.seconds), ("kills_per_min", 60 * self.kills / self.seconds),
            ("good_side", self.pos[1] / n), ("enemy_pit", self.pos[2] / n), ("own_pit", self.pos[3] / n),
            ("match_minutes", self.seconds / 60))}


def run_hier(args, policy, n_steps, seed, a3_rule=None, sample_std=0.0):
    """Manager self-play through the frozen worker. policy: normalised obs tensor -> mean [B, 4].
    Returns (play summary, tracking records)."""
    from hier_env import HierHost
    env = HierHost(args.worker, MACRO, args.hosts, args.matches, 4, sd_frac=0.15, max_steps=5400, seed=seed,
                   threads=args.threads, pit_frac=args.eval_pit_frac)
    gen = torch.Generator().manual_seed(seed)
    O = env.reset_all().reshape(-1, D).copy()
    play, rec = Play(), {"O0": [], "O1": [], "a": [], "ok": []}
    for _ in range(n_steps):
        with torch.no_grad():
            mu = policy(O)
            a = mu + sample_std * torch.randn(mu.shape, generator=gen) if sample_std else mu
        a = np.clip(a.numpy(), -1, 1)
        if a3_rule is not None:
            a[:, 3] = A3_RULES[a3_rule](a[:, :3])
        out = env.step(a.reshape(-1, 2, A))
        O1 = out["obs"].reshape(-1, D).copy()
        play.book(out, O, MACRO * TICK_S)
        ok = ~np.repeat(out["reset"], 2) & (O[:, 9] < 0.5) & (O1[:, 9] < 0.5)
        rec["O0"].append(O); rec["O1"].append(O1); rec["a"].append(a); rec["ok"].append(ok)
        O = O1
    env.close()
    return play.summary(), {k: np.concatenate(v) for k, v in rec.items()}


def run_flat(args, net, norm, n_env_steps, seed):
    env = NativeHost(args.hosts, args.matches, 4, sd_frac=0.15, max_steps=5400, seed=seed, threads=args.threads,
                     pit_frac=args.eval_pit_frac)
    torch.manual_seed(seed)
    O = env.reset_all().reshape(-1, D).copy()
    play = Play()
    for _ in range(n_env_steps):
        with torch.no_grad():
            a = net.dist(torch.as_tensor(norm(O), dtype=torch.float32)).sample()
        out = env.step(np.clip(a.numpy(), -1, 1).reshape(-1, 2, 3))
        play.book(out, O, TICK_S)
        O = out["obs"].reshape(-1, D).copy()
    env.close()
    return play.summary()


def tracking(rec):
    """How well the worker realised the commanded targets over one macro step (valid rows only)."""
    ok = rec["ok"]
    O0, O1, a = rec["O0"][ok], rec["O1"][ok], rec["a"][ok]
    x0, y0 = O0[:, 0], O0[:, 1]
    bx = np.clip(x0 + 16 * a[:, 0], -X_MAX, X_MAX)
    tx, ty = x0 + 8 * a[:, 1], (y0 - 0.69) + 5 + 4 * a[:, 2]
    x1, tx1, ty1 = O1[:, 0], O1[:, 0] + O1[:, 13], O1[:, 1] + O1[:, 14]
    eb, et = np.abs(x1 - bx), np.hypot(tx1 - tx, ty1 - ty)
    ach = hindsight(O0, O1)  # what the hierarchy actually did, in label space
    cmd = np.clip(a[:, :3], -1, 1)

    def r2(p, q):  # achieved vs commanded
        return float(1 - np.mean((p - q) ** 2) / max(np.var(q), 1e-8))

    mode = np.where(a[:, 3] < -1 / 3, "body", np.where(a[:, 3] > 1 / 3, "tip", "both"))
    return {"n": int(ok.sum()),
            "body_err_m": float(eb.mean()), "body_err_med": float(np.median(eb)),
            "body_within_0.75": float(np.mean(eb < BODY_R)),
            "tip_err_m": float(et.mean()), "tip_err_med": float(np.median(et)),
            "tip_within_1": float(np.mean(et < TIP_R)),
            "sum_err_m": float(eb.mean() + et.mean()),
            "cmd_body_dx_abs": float(np.abs(bx - x0).mean()), "ach_body_dx_abs": float(np.abs(x1 - x0).mean()),
            "label_mae": np.abs(ach - cmd).mean(0).round(4).tolist(),
            "label_r2": [round(r2(ach[:, i], cmd[:, i]), 3) for i in range(3)],
            "mode_share": {m: float(np.mean(mode == m)) for m in ("body", "both", "tip")}}


def evaluate(args):
    torch.set_num_threads(args.torch_threads)
    ck = torch.load(args.out, weights_only=False)
    net, norm = ActorCritic(D, A), RunningNorm(D)
    net.load_state_dict(ck["net"]); norm.load(ck["norm"])
    flat_net, flat_norm, flat_ck = load_flat(flat_src(args))
    rnd, rnorm = fresh_manager(flat_ck, 12345)  # a PPO start from --init-from (untrained output layer)

    def pol(n, nm):
        return lambda O: n.pi(torch.as_tensor(nm(O), dtype=torch.float32))

    res = {"checkpoint": args.out, "pit_frac": args.eval_pit_frac, "a3_rule_trained": ck.get("distill", {}).get("a3_rule"), "tracking": {}, "play": {}}
    S = args.eval_steps
    rules = [None] + [r for r in args.eval_rules.split(",") if r]
    for rule in rules:
        t0 = time.time()
        play, rec = run_hier(args, pol(net, norm), S, seed=args.seed + 100, a3_rule=rule)
        name = "net" if rule is None else rule
        res["tracking"][name] = tracking(rec)
        res["play"][f"distilled mean, a3={name}"] = play
        print(f"[a3={name}] {time.time() - t0:.0f}s tracking {json.dumps(res['tracking'][name])}\n"
              f"          play {json.dumps(play)}", flush=True)
    extra = [("distilled sampled std0.3 a3=net", pol(net, norm), 0.3)]
    if args.baselines:
        extra += [("random-init sampled std0.61", pol(rnd, rnorm), math.exp(-0.5)),
                  ("random-init mean", pol(rnd, rnorm), 0.0)]
    for name, p, std in extra:
        play, rec = run_hier(args, p, S, seed=args.seed + 100, sample_std=std)
        res["play"][name] = play
        res["tracking"][name] = tracking(rec)
        print(f"[{name}] play {json.dumps(play)}\n          tracking {json.dumps(res['tracking'][name])}",
              flush=True)
    if args.baselines:
        res["play"]["farm1 flat (sampled)"] = run_flat(args, flat_net, flat_norm, S * MACRO, seed=args.seed + 100)
        print(f"[farm1 flat] play {json.dumps(res['play']['farm1 flat (sampled)'])}", flush=True)
    path = os.path.splitext(args.out)[0] + (f"_eval_pit{args.eval_pit_frac:g}.json" if args.eval_pit_frac != 0.3
                                            else "_eval.json")
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(f"wrote {path}", flush=True)
    return res


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("cmd", choices=["collect", "train", "eval", "all"])
    p.add_argument("--flat", default="runs/farm1/ckpt.pt")
    p.add_argument("--worker", default="runs/manager2/worker.pt")
    p.add_argument("--data", default="runs/distill/data.npz")
    p.add_argument("--out", default="runs/distill/manager_init.pt")
    p.add_argument("--hosts", type=int, default=8)
    p.add_argument("--matches", type=int, default=16)
    p.add_argument("--threads", type=int, default=4, help="native env threads")
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--steps", type=int, default=4000, help="collect: env steps (x 2 x hosts x matches agent-steps)")
    p.add_argument("--a3-rule", default="body_soft", choices=list(A3_RULES))
    p.add_argument("--epochs", type=int, default=8)
    p.add_argument("--batch", type=int, default=2048)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--eval-steps", type=int, default=300, help="eval: manager steps per run (x8 env steps)")
    p.add_argument("--eval-rules", default="both,body,tip,still_tip,move_body",
                   help="eval: a3 overrides to compare (plus the net's own a3)")
    p.add_argument("--eval-pit-frac", type=float, default=0.3,
                   help="eval: pit-start curriculum fraction (0.3 = training conditions; 0 = normal kick-offs)")
    p.add_argument("--no-baselines", dest="baselines", action="store_false",
                   help="eval: skip the random-init manager and flat farm1 reference runs")
    p.add_argument("--seed", type=int, default=0)
    args = p.parse_args()
    os.chdir(HERE)
    torch.set_num_threads(args.torch_threads)
    if args.cmd in ("collect", "all"):
        collect(args)
    if args.cmd in ("train", "all"):
        train(args)
    if args.cmd in ("eval", "all"):
        evaluate(args)


if __name__ == "__main__":
    main()
