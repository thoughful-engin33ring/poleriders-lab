"""Rollouts on the native simulator for train_farm.py (--native): the matches of hosts h0..h1-1 played by
the current policy against itself and league snapshots, with GAE, returned as PPO training data + stats.

`Roller` does one rollout of T steps at a time. With --procs P > 0 the hosts are split over P worker
processes (`Workers`), each with its own native env threads, policy copy and league bank, so the Python
work of acting (policy / league forward passes, bookkeeping) runs in parallel instead of sharing one GIL;
the learner process only does PPO. With --procs 0 one Roller runs in a thread of the learner process.

Per rollout a worker gets the acting weights and the master observation normalizer, updates its copy
every step while it plays (as the single-process trainer did), and returns the raw observation
statistics it saw; the learner merges all workers' statistics into the master normalizer. (A frozen
normalizer per rollout learned about half as fast from scratch: the first rollout ran unnormalized.)
"""
import os, random, sys, threading, time
from types import SimpleNamespace

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "bridge"))
from farm_env import OBS2_SIZE  # noqa: E402
from spr_env import SOCK_DIR  # noqa: E402
from train_ppo import ActorCritic, LeagueBank, RunningNorm, net_from_ckpt, spec_from_args  # noqa: E402


class Pool:
    def __init__(self, path):
        self.path, self.cache = path, {}
        os.makedirs(path, exist_ok=True)

    def files(self):
        return sorted(f for f in os.listdir(self.path) if f.endswith(".pt"))

    def add(self, net, norm, update):
        torch.save({"net": net.state_dict(), "norm": norm.state()},
                   os.path.join(self.path, f"u{update:06d}.pt"))

    def sample(self, rng):
        files = self.files()
        if not files:
            return None
        name = rng.choice(files[-5:]) if rng.random() < 0.5 else rng.choice(files)
        if name not in self.cache:
            ck = torch.load(os.path.join(self.path, name), weights_only=False)
            n, nm, _ = net_from_ckpt(ck)  # any architecture; older snapshots saw a shorter observation
            self.cache[name] = (n, nm, name)
        return self.cache[name]


class Match:
    """Bookkeeping for one shadow match slot."""

    def __init__(self, h, k, league, rng):
        self.h, self.k, self.league, self.rng = h, k, league, rng
        self.new_episode(None)

    def new_episode(self, pool, flat_pool=None, flat_frac=0.0, fixed=None):
        """fixed: [(share, opp)] frozen league opponents (e.g. another run's policy) drawn in that share of
        this slot's league games; the rest are pool snapshots as before."""
        self.opp, self.opp_agent, self.opp_flat, self.opp_fixed = None, 0, False, False
        if self.league and fixed:
            u = self.rng.random()
            for share, o in fixed:
                if u < share:
                    self.opp, self.opp_agent, self.opp_fixed = o, self.rng.choice((1, 2)), True
                    break
                u -= share
            if self.opp:
                self.sd_start = False
                return
        if self.league and flat_pool is not None and self.rng.random() < flat_frac:
            o = flat_pool.sample(self.rng)  # a flat stick policy (e.g. farm1) vs this manager
            if o:
                self.opp, self.opp_agent, self.opp_flat = o, self.rng.choice((1, 2)), True
        elif self.league and pool is not None:
            o = pool.sample(self.rng)
            if o:
                self.opp, self.opp_agent = o, self.rng.choice((1, 2))
        self.sd_start = False


def merge_norm(norm, count, mean, var):
    """Fold a batch's raw observation statistics into a RunningNorm (RunningNorm.update's formula)."""
    if count <= 0:
        return
    d, tot = mean - norm.mean, norm.count + count
    norm.mean = norm.mean + d * count / tot
    norm.var = (norm.var * norm.count + var * count + d ** 2 * norm.count * count / tot) / tot
    norm.count = tot
    norm.pin()  # hidden inputs stay hidden


class Roller:
    """Hosts h0..h1-1 (K matches each) of a native training farm; see the module docstring."""

    def __init__(self, a, h0, h1, league_flags, A, threads, cfg):
        from native_env import ARENA_LEVEL, NativeFarm
        self.a, self.h0, self.K, self.A = a, h0, a.matches, A
        Hw, K = h1 - h0, a.matches
        socks = [os.path.join(SOCK_DIR, f"spr-native-{a.run}-{h}.sock") for h in range(h0, h1)]
        groups = 2 if a.pipeline and Hw % 2 == 0 else 1
        # seed = h0: host h's reset RNG is random.Random(h) wherever it runs (NativeFarm: seed + local h)
        self.farm = NativeFarm(Hw, K, dict(cfg, level=ARENA_LEVEL[a.arena]), threads=threads, seed=h0,
                               socks=socks, groups=groups, hier=(a.hier, a.macro) if a.hier else None)
        self.obs = np.zeros((Hw, K, 2, OBS2_SIZE), np.float32)
        sd_start = np.zeros((Hw, K), bool)
        for h, c in enumerate(self.farm.conns):
            _, msg = c.recv()
            self.obs[h], sd_start[h] = msg["obs"], msg["sd_start"]
        self.obs_rows = self.obs.reshape(-1, 2, OBS2_SIZE)
        run_dir = os.path.join(HERE, "runs", a.run)
        self.pool = Pool(os.path.join(run_dir, "pool"))
        self.flat_pool = Pool(a.flat_league) if a.flat_league else None
        # --league-fixed NAME:CKPT:SHARE,...: frozen opponents in that share of the league games
        self.fixed = []
        for spec in filter(None, (getattr(a, "league_fixed", None) or "").split(",")):
            name, path, share = spec.rsplit(":", 2)
            ck = torch.load(path, weights_only=False)
            with torch.random.fork_rng():
                n, nm, _ = net_from_ckpt(ck)
            self.fixed.append((float(share), (n, nm, name)))
        with torch.random.fork_rng():  # its weights come from the learner: don't consume the sampling RNG
            self.net = ActorCritic(OBS2_SIZE, A, **spec_from_args(a))  # --arch / --action-head ...
        self.norm = RunningNorm(OBS2_SIZE)
        self.flat = [Match(h, k, league_flags[h][k], random.Random(1000 * h + k))
                     for h in range(h0, h1) for k in range(K)]
        NM = self.NM = len(self.flat)
        self.bank = LeagueBank(NM, OBS2_SIZE, act_size=A)  # league opponents, one slot per match
        for j, m in enumerate(self.flat):
            m.sd_start = bool(sd_start.reshape(-1)[j])
        # per-match episode state as arrays (row j = local match index), so a step's bookkeeping is a few
        # numpy ops; only episode ends and goals/kills loop in Python
        self.opp_mask = np.ones(2 * NM, np.float32)  # 0 = that rider is a league opponent (not trained on)
        self.league_on = np.zeros(NM, bool)          # a league snapshot (LeagueBank) drives a rider of match j
        self.opp_row = np.zeros(NM, int)             # its rider (0/1)
        self.ang = np.full(2 * NM, np.nan)  # last pole angle per rider (deg, attack frame), for pass counting
        self.acc = {"t": np.zeros(NM, int), "ret": np.zeros((NM, 2)), "goal": np.zeros((NM, 2)),
                    "kill": np.zeros((NM, 2)), "eg": np.zeros(NM), "ek": np.zeros(NM), "sd": np.zeros(NM, bool)}

    def new_episode(self, j):
        """Next episode's opponent for match j; flat opponents play inside the env (HierHost)."""
        a, m = self.a, self.flat[j]
        m.new_episode(self.pool, self.flat_pool, a.flat_frac, self.fixed)
        if a.hier:
            self.farm.set_flat(j, m.opp_agent if m.opp_flat else 0, m.opp)
        self.opp_mask[2 * j:2 * j + 2] = 1.0
        if m.opp and m.opp_agent:
            self.opp_mask[2 * j + m.opp_agent - 1] = 0.0
        self.league_on[j] = bool(m.opp) and not m.opp_flat  # flat opponents act inside the env
        self.opp_row[j] = max(m.opp_agent - 1, 0)
        if self.league_on[j]:
            self.bank.load(j, m.opp)
        for v in self.acc.values():
            v[j] = 0

    def match_status(self):
        return [{"host": m.h, "k": m.k, "league": m.league, "opp": m.opp[2] if m.opp else None,
                 "opp_agent": m.opp_agent, "opp_flat": bool(m.opp_flat), "sd_start": m.sd_start}
                for m in self.flat]

    def rollout(self, weights, norm_state, T, rotation=None):
        """rotation: this rollout's reward weight of backward pole rotation (None = unchanged)."""
        a, K, A, D, NM = self.a, self.K, self.A, OBS2_SIZE, self.NM
        net, norm, farm, acc = self.net, self.norm, self.farm, self.acc
        if rotation is not None:
            for h in farm.hosts:
                h.set_rotation_weight(rotation)
        net.load_state_dict(weights)
        norm.load(norm_state)
        B = 2 * NM
        buf_o = np.zeros((T, B, D), np.float32)
        buf_a = np.zeros((T, B, net.stored_size), np.float32)  # gated head: + the gate bits
        buf_lp, buf_v, buf_r, buf_d, buf_m = (np.zeros((T, B), np.float32) for _ in range(5))
        ep_stats, vs_pool, vs_flat, sd_stats, events = [], [], [], [], []
        prof = {"policy": 0.0, "league": 0.0, "wait": 0.0, "book": 0.0}
        pos = np.zeros(4)  # the current policy's live riders: n, good side, enemy pit, own pit
        passes = np.zeros(2)  # the current policy's pole passing through vertical: backward, forward
        ost = [0, np.zeros(D), np.zeros(D)]  # raw observation stats of the current policy's riders: n, sum, sum sq

        def act(t, h0, h1):
            """Actions of local hosts h0..h1-1 at step t (current policy + league opponents)."""
            _t0 = time.time()
            j0, j1 = h0 * K, h1 * K
            rows = slice(2 * j0, 2 * j1)
            raw = self.obs[h0:h1].reshape(-1, D)
            mask = self.opp_mask[2 * j0:2 * j1]
            mine = raw[mask > 0]
            norm.update(mine)  # this worker's copy keeps adapting within the rollout, as in one process
            mine = mine.astype(np.float64)
            ost[0] += len(mine); ost[1] += mine.sum(0); ost[2] += (mine * mine).sum(0)
            # agent frame (attacking +x, y up): me.x = 0, me.y = 1, ghost = 9, ball.x = 52.
            # Pits are below the middle plateau (y up < 0) beyond |x| = 16.3 m.
            live = (mask > 0) & (raw[:, 9] < 0.5)
            # pole angle in the attack frame (0 = toward the goal attacked, 90 = up, 180 = back), -90..270
            ang = np.degrees(np.arctan2(raw[:, 10], raw[:, 11]))
            ang = np.where(ang < -90, ang + 360, ang)
            prev = self.ang[rows]
            ok = live & np.isfinite(prev) & (np.abs(ang - prev) < 60)
            passes[0] += (ok & (prev < 90) & (ang >= 90)).sum()
            passes[1] += (ok & (prev >= 90) & (ang < 90)).sum()
            self.ang[rows] = np.where(live, ang, np.nan)
            if live.any():
                r = raw[live]
                pos[:] += [len(r), (r[:, 0] < r[:, 52]).sum(), ((r[:, 0] > 16.3) & (r[:, 1] < 0)).sum(),
                           ((r[:, 0] < -16.3) & (r[:, 1] < 0)).sum()]
            o = torch.as_tensor(norm(raw), dtype=torch.float32)
            with torch.no_grad():
                act_, lp, v = net.act(o)
            a_np = act_.numpy()
            a_env = net.env_action(a_np)  # gauss: a_np itself; gated: sticks with gate 0 centred
            _t1 = time.time()
            js = np.nonzero(self.league_on[j0:j1])[0] + j0  # league opponents act with their own frozen nets
            if len(js):
                idx = 2 * (js - j0) + self.opp_row[js]
                a_env[idx] = self.bank.act(js, raw[idx])
            buf_o[t, rows], buf_a[t, rows], buf_lp[t, rows], buf_v[t, rows], buf_m[t, rows] = \
                o.numpy(), a_np, lp.numpy(), v.numpy(), mask
            prof["policy"] += _t1 - _t0
            prof["league"] += time.time() - _t1
            return np.clip(a_env, -1, 1).reshape(h1 - h0, K, 2, A)

        def book(t, j0, out):
            """Step-t results of local matches j0, j0 + 1, ...: rewards, episode ends, events, next obs."""
            _t0 = time.time()
            rew, tm = out["rew"], out["terms"]
            n = len(rew)
            js = slice(j0, j0 + n)
            acc["t"][js] += 1
            acc["ret"][js] += rew
            acc["goal"][js] += tm[:, :, 0]
            acc["kill"][js] += tm[:, :, 1]
            acc["eg"][js] += np.abs(tm[:, 0, 0])
            acc["ek"][js] += np.abs(tm[:, 0, 1])
            acc["sd"][js] |= out["sd"].astype(bool)
            rr = rew.astype(np.float64)
            tr = np.nonzero(out["trunc"])[0]
            if len(tr):  # bootstrap the time limit from the final state's value
                with torch.no_grad():
                    vb = net.v(torch.as_tensor(norm(out["final"][tr].reshape(-1, D)), dtype=torch.float32))
                rr[tr] += a.gamma * vb.squeeze(-1).numpy().reshape(-1, 2)
            buf_r[t, 2 * j0:2 * (j0 + n)] = rr.reshape(-1)
            if getattr(a, "goal_episodes", False):  # goal-to-goal: every goal ends a learning episode (no
                gk = np.nonzero(tm[:, 0, 0] != 0)[0] + j0  # value across it); the match itself goes on
                buf_d[t, 2 * gk] = buf_d[t, 2 * gk + 1] = 1.0
            for k in np.nonzero((tm[:, 0, 0] != 0) | (tm[:, 0, 1] != 0))[0]:
                j, g1, k1 = j0 + k, float(tm[k, 0, 0]), float(tm[k, 0, 1])
                m = self.flat[j]
                events.append({"t": time.time(), "instance": f"{m.h}.{m.k}",
                               "type": "goal" if g1 else "kill",
                               "winner": 1 if (g1 or k1) > 0 else 2,
                               "score": out["score"][k].tolist(), "step": int(acc["t"][j]),
                               "sudden_death": bool(out["sd"][k]),
                               "opp": m.opp[2] if m.opp else None, "opp_agent": m.opp_agent})
            for k in np.nonzero(out["reset"])[0]:
                j = j0 + k
                m = self.flat[j]
                buf_d[t, 2 * j] = buf_d[t, 2 * j + 1] = 1.0
                ep_stats.append((int(acc["t"][j]), float(acc["eg"][j]), float(acc["ek"][j])))
                if m.opp:
                    me = 2 - m.opp_agent  # the current policy's rider (0/1)
                    (vs_flat if m.opp_flat or m.opp_fixed else vs_pool).append(
                        (float(acc["ret"][j, me]), float(acc["goal"][j, me]), float(acc["kill"][j, me])))
                if acc["sd"][j]:
                    sd_stats.append(1.0 if acc["goal"][j, 0] != 0 else 0.0)
                self.new_episode(j)
                m.sd_start = bool(out["sd_start"][k])
            self.obs_rows[js] = out["obs"]
            prof["book"] += time.time() - _t0

        def wait(g):
            _w0 = time.time()
            out = farm.wait(g)
            prof["wait"] += time.time() - _w0
            return out

        Hw = len(self.flat) // K
        if farm.G > 1:
            # two host groups, staggered by half a step: while one group's physics runs (C++ threads, GIL
            # released) this process books the other group's last step and computes its next actions
            Hg = farm.Hg
            farm.step_async(0, act(0, 0, Hg))
            acts1 = act(0, Hg, Hw)
            for t in range(T):
                out0 = wait(0)
                farm.step_async(1, acts1)
                book(t, 0, out0)
                if t + 1 < T:
                    acts0 = act(t + 1, 0, Hg)
                out1 = wait(1)
                if t + 1 < T:
                    farm.step_async(0, acts0)
                book(t, Hg * K, out1)
                if t + 1 < T:
                    acts1 = act(t + 1, Hg, Hw)
        else:
            for t in range(T):
                farm.step_async(0, act(t, 0, Hw))
                book(t, 0, wait(0))

        # GAE
        with torch.no_grad():
            last_v = net.v(torch.as_tensor(norm(self.obs.reshape(-1, D)), dtype=torch.float32)).squeeze(-1).numpy()
        adv = np.zeros((T, B), np.float32)
        gae = np.zeros(B, np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf_v[t + 1]
            nonterm = 1.0 - buf_d[t]
            delta = buf_r[t] + a.gamma * nv * nonterm - buf_v[t]
            gae = delta + a.gamma * a.lam * nonterm * gae
            adv[t] = gae
        ret = adv + buf_v
        keep = buf_m.reshape(-1) > 0
        n = ost[0]
        mean = ost[1] / max(n, 1)
        return {"data": (buf_o.reshape(-1, D)[keep], buf_a.reshape(-1, buf_a.shape[-1])[keep], buf_lp.reshape(-1)[keep],
                         adv.reshape(-1)[keep], ret.reshape(-1)[keep]),
                "obs_stats": (n, mean, np.maximum(ost[2] / max(n, 1) - mean ** 2, 0.0)), "norm": norm.state(),
                "ep_stats": ep_stats, "vs_pool": vs_pool, "vs_flat": vs_flat, "sd_stats": sd_stats,
                "events": events, "pos": pos, "passes": passes, "prof": prof, "matches": self.match_status(),
                "env_steps": T * NM * (a.macro if a.hier else 1)}


def _worker(conn, a, h0, h1, league_flags, A, threads, cfg):
    a = SimpleNamespace(**a)
    torch.set_num_threads(a.torch_threads)
    torch.manual_seed(a.seed + 7919 * (h0 + 1))
    roller = Roller(a, h0, h1, league_flags, A, threads, cfg)
    conn.send("ready")
    while True:
        try:
            msg = conn.recv()
        except EOFError:  # the learner is gone
            break
        if msg[0] == "stop":
            break
        conn.send(roller.rollout(*msg[1:]))


def _merge(parts):
    """Rollout results of several host ranges -> one (data concatenated, stats summed / joined)."""
    if len(parts) == 1:
        return dict(parts[0], obs_stats=[parts[0]["obs_stats"]])
    out = {"data": tuple(np.concatenate(x) for x in zip(*(p["data"] for p in parts))),
           "pos": sum(p["pos"] for p in parts), "passes": sum(p["passes"] for p in parts),
           "env_steps": sum(p["env_steps"] for p in parts),
           "prof": {k: max(p["prof"][k] for p in parts) for k in parts[0]["prof"]},  # parallel: the slowest
           "obs_stats": [p["obs_stats"] for p in parts]}
    for k in ("ep_stats", "vs_pool", "vs_flat", "sd_stats", "events", "matches"):
        out[k] = [x for p in parts for x in p[k]]
    return out


class Workers:
    """The farm's hosts split over P rollout worker processes (P = 0: one Roller in a thread here).
    start(weights, norm_state, T) launches a rollout on all of them; collect() waits and merges."""

    def __init__(self, a, league_flags, A, cfg, procs, threads):
        H = a.hosts
        self.procs = procs
        if procs == 0:
            torch.set_num_threads(a.torch_threads)
            self.roller = Roller(a, 0, H, league_flags, A, threads, cfg)
            self.thread, self.result = None, None
            return
        if H % procs:
            raise SystemExit(f"--hosts {H} doesn't split over --procs {procs}")
        import multiprocessing as mp
        ctx = mp.get_context("spawn")
        per = H // procs
        self.conns, self.ps = [], []
        for i in range(procs):
            c, w = ctx.Pipe()
            p = ctx.Process(target=_worker, args=(w, vars(a), i * per, (i + 1) * per, league_flags, A,
                                                  max(1, threads // procs), cfg), daemon=True)
            p.start()
            self.conns.append(c)
            self.ps.append(p)
        for c in self.conns:
            assert c.recv() == "ready"

    def start(self, weights, norm_state, T, rotation=None):
        if self.procs == 0:
            def run():
                self.result = self.roller.rollout(weights, norm_state, T, rotation)
            self.thread = threading.Thread(target=run)
            self.thread.start()
            return
        for c in self.conns:
            c.send(("run", weights, norm_state, T, rotation))

    def collect(self):
        if self.procs == 0:
            self.thread.join()
            return _merge([self.result])
        return _merge([c.recv() for c in self.conns])

    def close(self):
        if self.procs:
            for c in self.conns:
                try:
                    c.send(("stop",))
                except OSError:
                    pass
