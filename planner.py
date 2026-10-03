#!/usr/bin/env python3
"""Lookahead planner prototype on the native sim's planning branches (NativeHost.plan_*; exact clones of a match).

At every decision of a match, for the planning rider:
  * K candidate first actions: the policy mean, "do nothing" (sticks centred = action 0; --no-zero: none), and K-2
    samples from the policy (N(mean, cand_std), cand_std 0 = the policy's own std), clipped to [-1, 1];
  * M opponent reply models (--models): opp = the opponent's own policy (sampled), oppmean = its mean action,
    farm2 = farm2's policy (sampled; any NAME from --model-ckpt NAME=PATH likewise), self = the planning policy
    itself (sampled), zero = do nothing, random = uniform [-1, 1] each decision;
  * one branch per (candidate, model): the candidate is held for --hold decisions, then the planning policy
    continues (--cont sample|mean), the model plays the opponent; H decisions (--H 15 = 1 s);
  * branch score = sum_t gamma^t r_t (the planning rider's reward, with the weights / goal refund / gamma its critic
    was trained with) + gamma^H V(leaf) from the planning policy's own critic (no bootstrap after a real terminal);
  * candidate score = aggregate over the models (--agg): mean | min | meanstd:L (mean - L * std) |
    softmin:T (-T log mean exp(-S / T): between mean (T large) and min (T -> 0));
  * the best candidate is played for one decision, then everything is re-planned.
All branches of all matches being played are stepped together (C++ thread pool), and every step's policy /
model / value inference runs batched over all branches in one torch call each.

Evaluation (eval_h2h's rules: 7 goals / 5 min, no sudden death, tie = draw; sides swapped every match):
    .venv/bin/python -u planner.py --policy runs/farm2/ckpt.pt --opp runs/farm2/ckpt.pt --matches 32 \
        --K 8 --models opp,oppmean,zero,random --H 15 --threads 8 --name k8m4h15
-> runs/planner/<name>.json (per-match results, goal diff + 95% CI, decisions/s, time per decision).
--timing N --matches 1: one match, N decisions, time per decision (real time = 66.7 ms per decision).
"""
import argparse, json, math, os, sys, time

os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")  # numpy (Accelerate) inference: small batches, 1 thread
import numpy as np  # noqa: E402
import torch  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bridge"))
from native_env import DEFAULT_REWARD, NativeHost  # noqa: E402
from eval_h2h import FORMAT, MAX_STEPS, load_policy  # noqa: E402

RUNS = os.path.join(HERE, "runs")


class Pol:
    """A checkpoint's policy + critic for batched inference in numpy (float32; per call ~20x less overhead than
    torch for the planner's small batches; the same function as the torch net up to float rounding)."""

    def __init__(self, path):
        net, self.norm, ck = load_policy(path)
        self.args = ck.get("args", {}) or {}
        self.size = len(self.norm.mean)
        self.std = net.log_std.exp().detach().numpy().astype(np.float32)
        self.path = path
        sd = {k: v.detach().numpy().astype(np.float32) for k, v in net.state_dict().items()}
        self.pi = [(sd[f"pi.{i}.weight"].T.copy(), sd[f"pi.{i}.bias"]) for i in (0, 2, 4)]
        self.v = [(sd[f"v.{i}.weight"].T.copy(), sd[f"v.{i}.bias"]) for i in (0, 2, 4)]
        self.mu_ = self.norm.mean.astype(np.float32)
        self.isd_ = (1.0 / np.sqrt(self.norm.var + 1e-8)).astype(np.float32)

    def _x(self, O):
        x = O.reshape(-1, O.shape[-1])[:, :self.size].astype(np.float32)
        return np.clip((x - self.mu_) * self.isd_, -10, 10)

    @staticmethod
    def _mlp(L, x):
        (w0, b0), (w1, b1), (w2, b2) = L
        return np.tanh(np.tanh(x @ w0 + b0) @ w1 + b1) @ w2 + b2

    def mean(self, O):
        return self._mlp(self.pi, self._x(O))

    def value(self, O):
        return self._mlp(self.v, self._x(O))[:, 0].astype(np.float64)

    def sample(self, O, rng, mean=None):
        mu = self.mean(O) if mean is None else mean
        return np.clip(mu + self.std * rng.standard_normal(mu.shape).astype(np.float32), -1, 1)

    def reward_cfg(self):
        """The reward the critic learned: weights, goal refund (default off for checkpoints from before the rule
        was saved in args), gamma."""
        w = dict(DEFAULT_REWARD)
        if self.args.get("reward"):
            w.update(self.args["reward"] if isinstance(self.args["reward"], dict) else json.loads(self.args["reward"]))
        return w, bool(self.args.get("goal_refund", False)), float(self.args.get("gamma", 0.995))


def aggregate(S, how):
    """S [..., M] branch scores per opponent model -> candidate score."""
    if how == "mean":
        return S.mean(-1)
    if how == "min":
        return S.min(-1)
    kind, _, p = how.partition(":")
    p = float(p)
    if kind == "meanstd":
        return S.mean(-1) - p * S.std(-1)
    if kind == "softmin":
        m = S.min(-1, keepdims=True)
        return (m - p * np.log(np.mean(np.exp(-(S - m) / p), -1, keepdims=True)))[..., 0]
    raise ValueError(how)


class Planner:
    def __init__(self, env, pol, opp, models, K, H, hold, agg, cont, cand_std, gamma, seed, named=None, no_zero=False):
        self.env, self.pol, self.opp = env, pol, opp
        self.named = dict(named or {}, self=pol)  # policy opponent models by name
        self.models, self.K, self.M, self.H = models, K, len(models), H
        self.hold, self.agg, self.cont, self.cand_std, self.gamma = hold, agg, cont, cand_std, gamma
        self.no_zero = no_zero
        self.rng = np.random.default_rng(seed)
        G = env.n
        self.G, self.KM = G, K * self.M
        self.owner = np.repeat(np.arange(G), self.KM)
        env.plan_init(self.owner)
        B = len(self.owner)
        self.acts = np.zeros((B, 2, 3), np.float32)
        self.model_of = np.tile(np.arange(self.M), G * K)  # branch b = (g, i, j): b = g*K*M + i*M + j
        self.t_env = self.t_torch = 0.0
        self.picks = np.zeros(K, np.int64)  # which candidate slot won (0 = mean, 1 = zero, 2.. = samples)

    def candidates(self, O_me):
        """[G, K, 3]: mean, zero, K-2 samples."""
        G, K = len(O_me), self.K
        mu = self.pol.mean(O_me)
        c = np.zeros((G, K, 3), np.float32)
        c[:, 0] = np.clip(mu, -1, 1)
        k0 = 1 if self.no_zero else 2  # slot 1 = do nothing (sticks centred) unless --no-zero
        if K > k0:
            s = self.cand_std if self.cand_std > 0 else self.pol.std
            c[:, k0:] = np.clip(mu[:, None] + s * self.rng.standard_normal((G, K - k0, 3)).astype(np.float32), -1, 1)
        return c

    def opp_actions(self, Oo):
        """Opponent-model actions for every branch: Oo [B, 62] -> [B, 3]."""
        B = len(Oo)
        a = np.zeros((B, 3), np.float32)
        need = {m for m in self.models if m in ("opp", "oppmean")}
        mu = self.opp.mean(Oo) if need else None
        for j, m in enumerate(self.models):
            rows = self.model_of == j
            if m == "opp":
                a[rows] = self.opp.sample(Oo[rows], self.rng, mu[rows])
            elif m == "oppmean":
                a[rows] = np.clip(mu[rows], -1, 1)
            elif m in self.named:
                a[rows] = self.named[m].sample(Oo[rows], self.rng)
            elif m == "random":
                a[rows] = self.rng.uniform(-1, 1, (rows.sum(), 3))
            elif m != "zero":
                raise ValueError(m)
        return a

    def plan(self, obs, my, live):
        """obs [G, 2, 62] root observations, my [G] planning rider (1/2), live [G] matches to plan for.
        Returns [G, 3] actions for the planning riders (rows of matches not live: the policy mean)."""
        env, G, K, M, H = self.env, self.G, self.K, self.M, self.H
        rows_g = np.arange(G)
        O_me = obs[rows_g, my - 1]
        t0 = time.perf_counter()
        cand = self.candidates(O_me)
        active = live[self.owner]
        t1 = time.perf_counter()
        P = env.plan_begin(active)
        t2 = time.perf_counter()
        self.t_torch += t1 - t0
        self.t_env += t2 - t1
        B = len(self.owner)
        mb = my[self.owner]
        own, oppr = (np.arange(B), mb - 1), (np.arange(B), 2 - mb)
        first = np.repeat(cand.reshape(G * K, 3), M, axis=0)  # [B, 3]
        R = np.zeros(B)
        disc = np.ones(B)
        stopped = np.zeros(B, bool)
        term = np.zeros(B, bool)
        for h in range(H):
            t0 = time.perf_counter()
            Ob = P["obs"]
            if h < self.hold:
                a_own = first
            elif self.cont == "mean":
                a_own = np.clip(self.pol.mean(Ob[own]), -1, 1)
            else:
                a_own = self.pol.sample(Ob[own], self.rng)
            self.acts[own] = a_own
            self.acts[oppr] = self.opp_actions(Ob[oppr])
            t1 = time.perf_counter()
            P = env.plan_step(self.acts)
            t2 = time.perf_counter()
            self.t_torch += t1 - t0
            self.t_env += t2 - t1
            go = ~stopped
            R[go] += disc[go] * P["rew"][own][go]
            disc[go] *= self.gamma
            d = go & P["done"]
            term |= d & ~P["trunc"]
            stopped |= d
        t0 = time.perf_counter()
        V = self.pol.value(P["obs"][own])
        R += np.where(term, 0.0, disc * V)
        t1 = time.perf_counter()
        env.plan_end()
        t2 = time.perf_counter()
        self.t_torch += t1 - t0
        self.t_env += t2 - t1
        S = aggregate(R.reshape(G, K, M), self.agg)
        best = S.argmax(1)
        for g in np.flatnonzero(live):
            self.picks[best[g]] += 1
        return cand[rows_g, best], S


def play(a, pol, opp, named, n_matches, seed, timing=0):
    w, refund, gamma = pol.reward_cfg()
    env = NativeHost(1, n_matches, 4, sd_frac=0.0, seed=seed, threads=a.threads, end_on_sd=True,
                     match_format=FORMAT, max_steps=MAX_STEPS, reward=w, goal_refund=refund)
    models = a.models.split(",")
    pl = None
    if a.K > 0:
        pl = Planner(env, pol, opp, models, a.K, a.H, a.hold, a.agg, a.cont, a.cand_std, gamma, seed + 1, named,
                     a.no_zero)
    rng = np.random.default_rng(seed + 2)
    G = n_matches
    my = np.array([1 if (k + seed) % 2 == 0 else 2 for k in range(G)])
    rows_me, rows_op = (np.arange(G), my - 1), (np.arange(G), 2 - my)
    obs = env.reset_all().copy()
    act = np.zeros((G, 2, 3), np.float32)
    done = np.zeros(G, bool)
    tot = np.zeros((G, 3))  # goal diff, kill diff, steps (for the planning rider)
    res = np.zeros((G, 3))
    n_dec, t_plan = 0, 0.0
    t_start = time.time()
    last_print = t_start
    step = 0
    while not done.all():
        live = ~done
        act[rows_op] = opp.sample(obs[rows_op], rng)
        t0 = time.perf_counter()
        if pl is not None:
            act[rows_me], _ = pl.plan(obs, my, live)
        else:
            act[rows_me] = pol.sample(obs[rows_me], rng)
        t_plan += time.perf_counter() - t0
        n_dec += int(live.sum())
        out = env.step(act)
        tot[live, :2] += out["terms"][rows_me][live]
        tot[live, 2] += 1
        ended = live & out["reset"]
        res[ended] = tot[ended]
        done |= ended
        obs = out["obs"].copy()
        step += 1
        if timing and step >= timing:
            break
        if time.time() - last_print > 60:
            last_print = time.time()
            print(f"  step {step}: {int(done.sum())}/{G} done, {n_dec / (time.time() - t_start):.1f} decisions/s, "
                  f"score so far {tot[:, 0].sum():+.0f}", flush=True)
    wall = time.time() - t_start
    env.close()
    info = {"wall_s": wall, "decisions": n_dec, "decisions_per_s": n_dec / wall, "plan_s": t_plan,
            "ms_per_cycle": 1e3 * t_plan / max(step, 1), "cycles": step}
    if pl is not None:
        info.update(env_s=pl.t_env, torch_s=pl.t_torch, picks=pl.picks.tolist())
    return res, info


def summarize(res):
    g = res[:, 0]
    n = len(g)
    se = g.std(ddof=1) / math.sqrt(n) if n > 1 else float("nan")
    return {"matches": n, "w": int((g > 0).sum()), "d": int((g == 0).sum()), "l": int((g < 0).sum()),
            "goal_diff": float(g.mean()), "ci95": [float(g.mean() - 1.96 * se), float(g.mean() + 1.96 * se)],
            "kill_diff": float(res[:, 1].mean()), "minutes": float(res[:, 2].mean() / 900.0)}


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", default=os.path.join(RUNS, "farm2", "ckpt.pt"), help="planning rider's policy + critic")
    p.add_argument("--opp", default=os.path.join(RUNS, "farm2", "ckpt.pt"), help="the real opponent (plain policy)")
    p.add_argument("--farm2", default=os.path.join(RUNS, "farm2", "ckpt.pt"), help="the 'farm2' opponent model")
    p.add_argument("--model-ckpt", default="", help="more policy opponent models: NAME=PATH,NAME=PATH (use NAME in --models)")
    p.add_argument("--K", type=int, default=8, help="candidate first actions (0 = no planning: plain policy)")
    p.add_argument("--models", default="opp,oppmean,zero,random")
    p.add_argument("--H", type=int, default=15, help="horizon in decisions (15 = 1 s)")
    p.add_argument("--hold", type=int, default=1, help="decisions the candidate is held in a branch")
    p.add_argument("--agg", default="meanstd:0.5")
    p.add_argument("--cont", default="sample", choices=["sample", "mean"])
    p.add_argument("--cand-std", type=float, default=0.0, help="candidate sample std (0 = the policy's own)")
    p.add_argument("--no-zero", action="store_true", help="no 'do nothing' candidate (another policy sample)")
    p.add_argument("--matches", type=int, default=32)
    p.add_argument("--parallel", type=int, default=0, help="matches played at once (0 = all)")
    p.add_argument("--threads", type=int, default=8, help="native env threads")
    p.add_argument("--torch-threads", type=int, default=2)
    p.add_argument("--seed", type=int, default=50_000)
    p.add_argument("--timing", type=int, default=0, help="play only this many decisions (of --matches at once): timing")
    p.add_argument("--name", default=None)
    a = p.parse_args()
    torch.set_num_threads(a.torch_threads)
    pol, opp = Pol(a.policy), Pol(a.opp)
    named = {}
    for item in filter(None, (f"farm2={a.farm2}," + a.model_ckpt).split(",")):
        n, _, path = item.partition("=")
        named[n] = opp if os.path.abspath(path) == os.path.abspath(a.opp) else Pol(path)
    w, refund, gamma = pol.reward_cfg()
    cfg = {k: v for k, v in vars(a).items()}
    print(f"planner {cfg}\n  critic reward {w} goal_refund={refund} gamma={gamma}", flush=True)
    if a.timing:  # --matches matches at once (e.g. 1), a.timing decisions each
        res, info = play(a, pol, opp, named, a.parallel or a.matches, a.seed, timing=a.timing)
        info["ms_per_decision"] = 1e3 * info["plan_s"] / max(info["decisions"], 1)
        print(json.dumps(info), flush=True)
        return
    par = a.parallel or a.matches
    allres, infos = [], []
    for b0 in range(0, a.matches, par):
        n = min(par, a.matches - b0)
        res, info = play(a, pol, opp, named, n, a.seed + b0)
        allres.append(res); infos.append(info)
        s = summarize(np.concatenate(allres))
        print(f"  batch {b0 // par}: {info}\n  so far: {s}", flush=True)
    res = np.concatenate(allres)
    s = summarize(res)
    wall = sum(i["wall_s"] for i in infos)
    dec = sum(i["decisions"] for i in infos)
    s.update(decisions_per_s=dec / wall, wall_s=wall, ms_per_decision_amortized=1e3 * wall / dec,
             ms_per_cycle=[i["ms_per_cycle"] for i in infos], parallel=par)
    if a.K > 0:
        picks = np.sum([i["picks"] for i in infos], 0)
        s["pick_share"] = (picks / picks.sum()).round(3).tolist()
        s["env_s"] = sum(i["env_s"] for i in infos)
        s["torch_s"] = sum(i["torch_s"] for i in infos)
    print(json.dumps(s), flush=True)
    name = a.name or f"k{a.K}_{a.models.replace(',', '-')}_h{a.H}"
    os.makedirs(os.path.join(RUNS, "planner"), exist_ok=True)
    with open(os.path.join(RUNS, "planner", name + ".json"), "w") as f:
        json.dump({"config": cfg, "critic": {"reward": w, "goal_refund": refund, "gamma": gamma}, "summary": s,
                   "per_match": res.tolist()}, f, indent=1)


if __name__ == "__main__":
    main()
