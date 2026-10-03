#!/usr/bin/env python3
"""Self-play PPO for Super Pole Riders (one shared policy plays both riders).

    .venv/bin/python train_ppo.py --instances 4 --run first          # train
    .venv/bin/python train_ppo.py --run first --resume               # continue
    .venv/bin/python train_ppo.py --run first --watch                # watch the latest checkpoint

Both agents see mirrored observations (each attacks +x), so one network serves both sides.
Rewards come from bridge/spr_env.py (goal ±1, kill ±0.3, small ball-progress shaping).

League: every --pool-every updates the policy is frozen into runs/<run>/pool/. In a fraction
(--league-frac) of the game instances one rider is driven by a random past snapshot instead
(side picked at random each episode); those opponent transitions are not trained on.

Measuring skill: goals/kills per episode are NOT a skill metric in self-play (two strong
players may well cancel out to 0-0). The number to watch is `vs_pool`: the current policy's
zero-sum return per episode against past snapshots (goal diff + 0.3 * kill diff + shaping).
It should stay > 0 while the pool keeps filling with ever-stronger snapshots.

Logs: runs/<run>/log.csv, latest checkpoint runs/<run>/ckpt.pt.
"""
import argparse, csv, json, math, os, random, subprocess, sys, threading, time
from concurrent.futures import ThreadPoolExecutor

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bridge"))
from spr_env import OBS_SIZE, SOCK_DIR, PoleRidersEnv, launch_game, launch_instances  # noqa: E402

ACT_SIZE = 3  # run, aim_x, aim_y


# ------------------------------------------------------------------ model

class RunningNorm:
    """Running mean/std observation normalizer (saved with the checkpoint).

    `masked` inputs are hidden from the policy: their mean is pinned to 0 and their variance to MASK_VAR, so they
    normalize to ~0 whatever their value (e.g. the score and clock, --mask-match-info). Every user of a
    checkpoint's normalizer (league opponents, evaluation, the dashboard, the real game) hides them too."""
    MASK_VAR = 1e12

    def __init__(self, n):
        self.mean, self.var, self.count = np.zeros(n), np.ones(n), 1e-4
        self.masked = np.zeros(n, bool)

    def pin(self):
        if self.masked.any():
            self.mean = np.where(self.masked, 0.0, self.mean)
            self.var = np.where(self.masked, self.MASK_VAR, self.var)

    def update(self, x):
        bm, bv, bc = x.mean(0), x.var(0), x.shape[0]
        d, tot = bm - self.mean, self.count + bc
        self.mean = self.mean + d * bc / tot
        self.var = (self.var * self.count + bv * bc + d ** 2 * self.count * bc / tot) / tot
        self.count = tot
        self.pin()

    def __call__(self, x):
        return np.clip((x - self.mean) / np.sqrt(self.var + 1e-8), -10, 10)

    def state(self):
        s = {"mean": self.mean, "var": self.var, "count": self.count}
        if self.masked.any():
            s["masked"] = self.masked
        return s

    def load(self, s):
        self.mean, self.var, self.count = s["mean"], s["var"], s["count"]
        self.masked = np.asarray(s.get("masked", np.zeros(len(self.mean), bool)), bool)
        self.pin()


# Network options (train_farm.py --arch / --hidden / --layers / --res-expand / --action-head; saved in the
# checkpoint's args, and inferred from the state_dict alone by spec_from_state / net_from_ckpt, so pool snapshots
# without args load too). The default (mlp, 256, 2, gauss) is the original network, parameter for parameter and
# random draw for random draw: keys pi.0 / pi.2 / pi.4 (and v.*), log_std.
ARCHS, ACTION_HEADS = ("mlp", "resmlp"), ("gauss", "gated")
DEFAULT_SPEC = {"arch": "mlp", "hidden": 256, "layers": 2, "expand": 4, "action_head": "gauss"}
# --action-head gated: one Bernoulli "touch this stick" gate per stick. Gate 0 = that stick exactly centred.
# Stick groups over the action (run, aim_x, aim_y): left stick x = run; right stick (aim_x, aim_y) = pole.
STICK_GROUPS = ((0,), (1, 2))
GATE_OF_DIM = [0, 1, 1]  # action dim -> its gate
# initial gate logit: sigmoid(2) = 0.88 "touching". A fresh gated policy starting at 0.5 let go of both sticks 87% of the
# time after 1.5M steps (smoke test 2026-10-02: the early "do nothing" attractor); start close to the Gaussian policy,
# which practically always has its sticks out of the dead zones, and let it learn to release
GATE_INIT_LOGIT = 2.0


def mlp(i, o, h=256, layers=2):
    """tanh MLP with `layers` hidden layers of width h (layers=2: the original Linear/Tanh/Linear/Tanh/Linear)."""
    mods = [nn.Linear(i, h), nn.Tanh()]
    for _ in range(layers - 1):
        mods += [nn.Linear(h, h), nn.Tanh()]
    return nn.Sequential(*mods, nn.Linear(h, o))


class ResBlock(nn.Module):
    """SimBa block: x + Linear(ReLU(Linear(LayerNorm(x)))) with an `expand`x wider inner layer."""

    def __init__(self, h, expand):
        super().__init__()
        self.ln, self.fc1, self.fc2 = nn.LayerNorm(h), nn.Linear(h, expand * h), nn.Linear(expand * h, h)

    def forward(self, x):
        return x + self.fc2(torch.relu(self.fc1(self.ln(x))))


class ResMLP(nn.Module):
    """SimBa-style residual MLP (Lee et al. 2024): Linear in -> h, `layers` pre-LN residual blocks, final
    LayerNorm, Linear head. (The observation is already running-normalised by RunningNorm, as in SimBa.)
    Init: orthogonal (gain sqrt 2 into ReLUs / the residual stream, 1 for the block outputs, out_gain for the head:
    its input is LayerNorm'd, so the output's initial std ~ out_gain), zero biases."""

    def __init__(self, i, o, h=256, layers=2, expand=4, out_gain=1.0):
        super().__init__()
        self.inp = nn.Linear(i, h)
        self.blocks = nn.Sequential(*[ResBlock(h, expand) for _ in range(layers)])
        self.ln = nn.LayerNorm(h)
        self.out = nn.Linear(h, o)
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.orthogonal_(m.weight, out_gain if m is self.out else 1.0 if any(m is b.fc2 for b in self.blocks)
                                    else math.sqrt(2))
                nn.init.zeros_(m.bias)

    def forward(self, x):
        return self.out(self.ln(self.blocks(self.inp(x))))


def out_layer(tower):
    return tower.out if isinstance(tower, ResMLP) else tower[-1]


class GatedNormal:
    """Two-part stick distribution (--action-head gated). Per stick group k (STICK_GROUPS): a gate
    g_k ~ Bernoulli(sigmoid(l_k)) ("touch this stick") and the Gaussian N(mu, std) of the stick's value dims.
    Stored action = [x_0 .. x_{A-1}, g_0 .. g_{G-1}] (x always sampled; it only counts where its gate is 1);
    env action = x * g (gate 0 -> that stick exactly 0 = centred).
      log p(a) = sum_k [ log Bern(g_k) + g_k * sum_{i in k} log N(x_i) ]
      H        = sum_k [ H_Bern(p_k) + p_k * sum_{i in k} H_N(std_i) ]   (exact: the entropy of the mixed
                 discrete/continuous variable (g, x restricted to the touched sticks))
    log_prob / entropy return one column per group (callers .sum(-1), as with Normal's per-dim columns)."""

    def __init__(self, mu, std, logits):
        self.normal, self.logits = torch.distributions.Normal(mu, std), logits

    def _expand(self, g):
        return g[..., GATE_OF_DIM]

    @property
    def mean(self):
        """Deterministic env action: the Gaussian mean on sticks whose gate is more likely on than off."""
        return self.normal.mean * self._expand((self.logits > 0).to(self.normal.mean.dtype))

    def sample(self):
        with torch.no_grad():
            x = self.normal.sample()
            g = torch.bernoulli(torch.sigmoid(self.logits))
            return torch.cat([x, g], -1)

    def log_prob(self, a):
        A = self.normal.loc.shape[-1]
        x, g = a[..., :A], a[..., A:]
        lpn = self.normal.log_prob(x)
        l = self.logits
        lpg = g * F.logsigmoid(l) + (1 - g) * F.logsigmoid(-l)
        return torch.stack([lpg[..., k] + g[..., k] * lpn[..., list(dims)].sum(-1)
                            for k, dims in enumerate(STICK_GROUPS)], -1)

    def entropy(self):
        l = self.logits
        p = torch.sigmoid(l)
        hb = -(p * F.logsigmoid(l) + (1 - p) * F.logsigmoid(-l))
        hn = self.normal.entropy()
        return torch.stack([hb[..., k] + p[..., k] * hn[..., list(dims)].sum(-1)
                            for k, dims in enumerate(STICK_GROUPS)], -1)

    def gate_prob(self):
        return torch.sigmoid(self.logits)


class ActorCritic(nn.Module):
    """Separate policy (pi) and value (v) towers + a state-independent log_std.
    arch mlp (tanh MLP, `layers` x `hidden`) or resmlp (ResMLP, `layers` blocks of width `hidden`, inner width
    expand * hidden); action_head gauss (diagonal Gaussian, the original) or gated (GatedNormal: pi also outputs
    one gate logit per stick group). Callers that send actions to the env use sample_env / mean_env (the gated
    head's stored action has the gate bits appended); act() returns the stored action for PPO."""

    def __init__(self, obs_size=OBS_SIZE, act_size=ACT_SIZE, arch="mlp", hidden=256, layers=2,
                 action_head="gauss", expand=4):
        super().__init__()
        if arch not in ARCHS or action_head not in ACTION_HEADS:
            raise ValueError(f"arch {arch!r} / action head {action_head!r}")
        if action_head == "gated" and act_size != len(GATE_OF_DIM):
            raise ValueError(f"the gated head is for stick policies (3 actions), not {act_size}")
        self.arch, self.hidden, self.layers, self.expand, self.action_head = arch, hidden, layers, expand, action_head
        self.act_size, self.n_gates = act_size, len(STICK_GROUPS) if action_head == "gated" else 0
        out = act_size + self.n_gates
        if arch == "mlp":
            self.pi, self.v = mlp(obs_size, out, hidden, layers), mlp(obs_size, 1, hidden, layers)
        else:
            # value head gain 0.3: initial values ~ +-0.3 (about the default MLP's); policy head 1 x 0.01 (below)
            self.pi = ResMLP(obs_size, out, hidden, layers, expand)
            self.v = ResMLP(obs_size, 1, hidden, layers, expand, out_gain=0.3)
        self.log_std = nn.Parameter(torch.full((act_size,), -0.5))
        with torch.no_grad():
            out_layer(self.pi).weight.mul_(0.01)
            if self.n_gates:
                out_layer(self.pi).bias[act_size:] = GATE_INIT_LOGIT

    @property
    def stored_size(self):
        """Columns of an action as act() returns it / PPO stores it (env actions + gate bits)."""
        return self.act_size + self.n_gates

    def spec(self):
        return {"arch": self.arch, "hidden": self.hidden, "layers": self.layers, "expand": self.expand,
                "action_head": self.action_head}

    def dist(self, obs):
        if self.n_gates:
            o = self.pi(obs)
            return GatedNormal(o[..., :self.act_size], self.log_std.exp(), o[..., self.act_size:])
        return torch.distributions.Normal(self.pi(obs), self.log_std.exp())

    def act(self, obs):
        d = self.dist(obs)
        a = d.sample()
        return a, d.log_prob(a).sum(-1), self.v(obs).squeeze(-1)

    def env_action(self, a):
        """Stored action (tensor or array) -> env action (run, aim_x, aim_y); the same object for gauss."""
        if not self.n_gates:
            return a
        A = self.act_size
        return a[..., :A] * a[..., A:][..., GATE_OF_DIM]

    def sample_env(self, obs):
        """A sampled env action (unclipped). gauss: exactly dist(obs).sample()."""
        return self.env_action(self.dist(obs).sample())

    def mean_env(self, obs):
        """The deterministic env action (gauss: the mean = pi(obs))."""
        return self.dist(obs).mean if self.n_gates else self.pi(obs)


def spec_from_state(sd):
    """Network spec (DEFAULT_SPEC keys) + obs size + act size of an ActorCritic state_dict."""
    act = sd["log_std"].shape[0]
    if "pi.inp.weight" in sd:
        obs, hidden = sd["pi.inp.weight"].shape[1], sd["pi.inp.weight"].shape[0]
        layers = len({k.split(".")[2] for k in sd if k.startswith("pi.blocks.") and k.endswith(".fc1.weight")})
        expand = sd["pi.blocks.0.fc1.weight"].shape[0] // hidden if layers else DEFAULT_SPEC["expand"]
        out_dim, arch = sd["pi.out.weight"].shape[0], "resmlp"
    else:
        idx = sorted(int(k.split(".")[1]) for k in sd if k.startswith("pi.") and k.endswith(".weight"))
        obs, hidden = sd["pi.0.weight"].shape[1], sd["pi.0.weight"].shape[0]
        layers, expand, arch = len(idx) - 1, DEFAULT_SPEC["expand"], "mlp"
        out_dim = sd[f"pi.{idx[-1]}.weight"].shape[0]
    if out_dim == act:
        head = "gauss"
    elif out_dim == act + len(STICK_GROUPS):
        head = "gated"
    else:
        raise ValueError(f"policy output {out_dim} vs {act} actions: unknown action head")
    return {"arch": arch, "hidden": hidden, "layers": layers, "expand": expand, "action_head": head}, obs, act


def spec_from_args(a):
    """Network spec from train_farm args (a namespace or its dict, e.g. a checkpoint's ck['args']); missing or
    None entries = the default network."""
    get = (lambda k: a.get(k)) if isinstance(a, dict) else (lambda k: getattr(a, k, None))
    s = dict(DEFAULT_SPEC)
    for k, ak in (("arch", "arch"), ("hidden", "hidden"), ("layers", "layers"), ("expand", "res_expand"),
                  ("action_head", "action_head")):
        if get(ak) is not None:
            s[k] = get(ak)
    return s


def net_from_ckpt(ck):
    """(net, norm, obs_size) rebuilt from a checkpoint or pool snapshot (dict, or a path): the architecture comes
    from the state_dict itself (spec_from_state), so every checkpoint — old 2x256 MLPs, pool snapshots without
    args, resmlp / gated ones — loads. Constructs exactly like ActorCritic(obs, act) did for old checkpoints
    (same random draws), then loads the weights and the normalizer."""
    if isinstance(ck, (str, os.PathLike)):
        ck = torch.load(ck, weights_only=False)
    spec, obs, act = spec_from_state(ck["net"])
    net, norm = ActorCritic(obs, act, **spec), RunningNorm(obs)
    net.load_state_dict(ck["net"]); norm.load(ck["norm"])
    return net, norm, obs


class LeagueBank:
    """Frozen league opponents evaluated in one batched pass: slot s holds (a zero-padded copy of)
    the snapshot currently driving league match s. Opponents only change between episodes, so the
    per-step cost is one bmm over all slots instead of a forward pass per distinct snapshot.

    Only plain opponents take the bmm path: tanh MLP, 2 hidden layers of width <= the bank width, Gaussian head.
    Narrower ones are zero-padded (exact: padded units get zero weights and bias, tanh(0) = 0, and their outgoing
    weights are zero). Any other opponent (resmlp, deeper MLP, wider, gated head) keeps its own module and is
    evaluated per distinct opponent, batched over the slots it drives (sample_env)."""

    def __init__(self, n_slots, obs_size, hidden=256, act_size=ACT_SIZE):
        S, D, Hd = n_slots, obs_size, hidden
        self.names = [None] * S
        self.mods = [None] * S  # slot -> (net, norm) of a non-plain opponent (None: the bmm path)
        self.mean, self.inv_sd = torch.zeros(S, D), torch.ones(S, D)
        self.w1, self.b1 = torch.zeros(S, Hd, D), torch.zeros(S, Hd)
        self.w2, self.b2 = torch.zeros(S, Hd, Hd), torch.zeros(S, Hd)
        self.w3, self.b3 = torch.zeros(S, act_size, Hd), torch.zeros(S, act_size)
        self.std = torch.ones(S, act_size)

    def plain(self, net):
        h = self.w1.shape[1]
        return (getattr(net, "arch", "mlp") == "mlp" and getattr(net, "layers", 2) == 2
                and getattr(net, "action_head", "gauss") == "gauss" and net.pi[0].out_features <= h
                and net.pi[0].in_features <= self.w1.shape[2] and net.pi[4].out_features == self.w3.shape[1])

    def load(self, s, opp):
        net, norm, name = opp
        if self.names[s] == name:
            return
        if not self.plain(net):
            self.mods[s] = (net, norm)
            self.names[s] = name
            return
        self.mods[s] = None
        n = net.pi[0].in_features  # older snapshots saw a shorter observation: pad with zero weights
        h = net.pi[0].out_features  # narrower nets: pad the hidden units with zeros
        with torch.no_grad():
            self.w1[s].zero_(); self.w1[s, :h, :n] = net.pi[0].weight
            self.b1[s].zero_(); self.b1[s, :h] = net.pi[0].bias
            self.w2[s].zero_(); self.w2[s, :h, :h] = net.pi[2].weight
            self.b2[s].zero_(); self.b2[s, :h] = net.pi[2].bias
            self.w3[s].zero_(); self.w3[s, :, :h] = net.pi[4].weight
            self.b3[s] = net.pi[4].bias
            self.std[s] = net.log_std.exp()
            self.mean[s].zero_(); self.mean[s, :n] = torch.as_tensor(norm.mean, dtype=torch.float32)
            self.inv_sd[s].fill_(1.0)
            self.inv_sd[s, :n] = torch.as_tensor(1.0 / np.sqrt(norm.var + 1e-8), dtype=torch.float32)
        self.names[s] = name

    def mu(self, slots, raw):
        """Mean actions of bmm-path slots (for tests)."""
        s = torch.as_tensor(slots)
        x = torch.clamp((torch.as_tensor(raw, dtype=torch.float32) - self.mean[s]) * self.inv_sd[s], -10, 10)
        with torch.no_grad():
            h = torch.tanh(torch.bmm(self.w1[s], x.unsqueeze(-1)).squeeze(-1) + self.b1[s])
            h = torch.tanh(torch.bmm(self.w2[s], h.unsqueeze(-1)).squeeze(-1) + self.b2[s])
            return torch.bmm(self.w3[s], h.unsqueeze(-1)).squeeze(-1) + self.b3[s]

    def act(self, slots, raw):
        """Sampled actions for league slots `slots` given their raw observations [len(slots), D]."""
        mods = [self.mods[int(j)] for j in slots]
        if all(m is None for m in mods):  # all plain: one bmm pass (the original code path)
            s = torch.as_tensor(slots)
            mu = self.mu(slots, raw)
            return (mu + self.std[s] * torch.randn_like(mu)).numpy()
        raw = np.asarray(raw, np.float32)
        out = np.zeros((len(mods), self.w3.shape[1]), np.float32)
        rows = [i for i, m in enumerate(mods) if m is None]
        if rows:
            s = torch.as_tensor(np.asarray(slots)[rows])
            mu = self.mu(s, raw[rows])
            out[rows] = (mu + self.std[s] * torch.randn_like(mu)).numpy()
        groups = {}
        for i, m in enumerate(mods):
            if m is not None:
                groups.setdefault(id(m[0]), (m, []))[1].append(i)
        for (net, norm), rows in groups.values():
            x = torch.as_tensor(norm(raw[rows][:, :len(norm.mean)]), dtype=torch.float32)
            with torch.no_grad():
                out[rows] = net.sample_env(x).numpy()
        return out


# ------------------------------------------------------------------ rollout workers

class Worker:
    """One game instance: two agent slots, optional league opponent, time-limit truncation."""

    def __init__(self, sock, args, league):
        self.env = PoleRidersEnv(frame_skip=args.frame_skip, fast=args.fast, sock=sock)
        self.args, self.league = args, league
        self.rng = random.Random(hash(sock))
        self.opp, self.opp_agent = None, 0  # opp: (net, norm, name) driving agent opp_agent
        self.idx, self.new_events, self.sock = 0, [], sock

    def start(self):
        self.env.start_match(arena=self.args.arena)
        return self.reset()

    def recover(self, why):
        """The game instance died or hung: log why, relaunch it and drive it back into a match."""
        proc = getattr(self, "proc", None)
        rc = None
        if proc:
            try:
                rc = proc.wait(timeout=5)  # a dying game may take a moment to exit
            except subprocess.TimeoutExpired:
                rc = None
        self.new_events.append({"t": time.time(), "instance": self.idx, "type": "crash",
                                "why": why[:300], "returncode": rc})
        print(f"[instance {self.idx}] {why[:120]} (returncode {rc}); relaunching", flush=True)
        if proc and rc is None:
            proc.kill()
        for attempt in range(3):
            try:
                self.proc = launch_game(self.sock)
                time.sleep(16)
                self.env.b, self.env.synced = None, False
                self.env.start_match(arena=self.args.arena)
                return
            except Exception as e:  # noqa: BLE001
                print(f"[instance {self.idx}] relaunch attempt {attempt + 1} failed: {e}", flush=True)
                self.proc.kill()
        raise RuntimeError(f"instance {self.idx} could not be recovered")

    def reset(self, pool=None):
        try:
            return self._reset(pool)
        except Exception as e:  # noqa: BLE001
            self.recover(f"reset failed: {e!r}")
            return self._reset(pool)

    def _reset(self, pool=None):
        self.opp, self.opp_agent = None, 0
        if self.league and pool:
            self.opp = pool.sample(self.rng)
            self.opp_agent = self.rng.choice((1, 2))
        obs = self.env.reset()
        if self.args.random_ball:
            by = self.env.state["ball"]["y"]
            x = self.rng.uniform(-self.args.random_ball, self.args.random_ball)
            self.env.b.cmd(f"ball {x:.3f} {by:.3f} 0 0")
            obs, _, _, _ = self.env.step({1: (0, 0, 0), 2: (0, 0, 0)})
        self.t = 0
        self.ret = {1: 0.0, 2: 0.0}
        self.terms = {a: {"goal": 0.0, "kill": 0.0} for a in (1, 2)}
        self.events = {"goals": 0, "kills": 0}
        return obs

    def step(self, a1, a2):
        try:
            obs, rew, done, info = self.env.step({1: tuple(a1), 2: tuple(a2)})
        except Exception as e:  # noqa: BLE001  (dead socket, empty reply, ...)
            self.recover(f"step failed: {e!r}")
            return None, (0.0, 0.0), True, False
        self.t += 1
        tr = info["terms"]
        if tr[1]["goal"] or tr[1]["kill"]:
            self.new_events.append({"t": time.time(), "instance": self.idx,
                                    "type": "goal" if tr[1]["goal"] else "kill",
                                    "winner": 1 if (tr[1]["goal"] or tr[1]["kill"]) > 0 else 2,
                                    "score": info["state"].get("score"), "step": self.t,
                                    "opp": self.opp[2] if self.opp else None,
                                    "opp_agent": self.opp_agent})
        for a in (1, 2):
            self.ret[a] += rew[a]
            for k in ("goal", "kill"):
                self.terms[a][k] += info["terms"][a][k]
        self.events["goals"] += abs(info["terms"][1]["goal"])
        self.events["kills"] += abs(info["terms"][1]["kill"])
        truncated = not done and self.t >= self.args.episode_steps
        return obs, (rew[1], rew[2]), done, truncated


class Pool:
    """Frozen past policies on disk; loaded lazily and cached."""

    def __init__(self, path):
        self.path, self.cache = path, {}
        os.makedirs(path, exist_ok=True)

    def files(self):
        return sorted(f for f in os.listdir(self.path) if f.endswith(".pt"))

    def __bool__(self):
        return bool(self.files())

    def add(self, net, norm, update):
        torch.save({"net": net.state_dict(), "norm": norm.state()},
                   os.path.join(self.path, f"u{update:06d}.pt"))

    def sample(self, rng):
        files = self.files()
        # half the time one of the 5 newest (hardest), otherwise uniform over history
        name = rng.choice(files[-5:]) if rng.random() < 0.5 else rng.choice(files)
        if name not in self.cache:
            ck = torch.load(os.path.join(self.path, name), weights_only=False)
            n, nm = ActorCritic(), RunningNorm(OBS_SIZE)
            n.load_state_dict(ck["net"]); nm.load(ck["norm"])
            self.cache[name] = (n, nm, name)
        return self.cache[name]


def parallel(pool, fn, items):
    return list(pool.map(fn, items))


# ------------------------------------------------------------------ training

def train(args):
    run_dir = os.path.join(HERE, "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    torch.manual_seed(args.seed)
    torch.set_num_threads(4)
    net, norm = ActorCritic(), RunningNorm(OBS_SIZE)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    total_steps, update = 0, 0
    if args.resume and os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); norm.load(ck["norm"])
        total_steps, update = ck["steps"], ck["update"]
        print(f"resumed {ckpt_path} at {total_steps} steps, update {update}")

    pool = Pool(os.path.join(run_dir, "pool"))
    socks = launch_instances(args.instances)
    n_league = int(round(args.instances * args.league_frac))
    workers = [Worker(s, args, league=i < n_league) for i, s in enumerate(socks)]
    for i, w in enumerate(workers):
        w.idx = i
        w.proc = launch_instances.procs.get(w.sock)
    ev_f = open(os.path.join(run_dir, "events.jsonl"), "a")
    tp = ThreadPoolExecutor(len(workers))
    obs = parallel(tp, lambda w: w.start(), workers)  # list of {1: vec, 2: vec}
    print(f"{len(workers)} instances in match ({n_league} league)")

    log_f = open(os.path.join(run_dir, "log.csv"), "a", newline="")
    log = csv.writer(log_f)
    if os.path.getsize(log_f.name) == 0:
        log.writerow(["update", "steps", "sps", "episodes", "ep_len", "goals_per_ep",
                      "kills_per_ep", "pool_size", "vs_pool", "vs_pool_goal_diff",
                      "vs_pool_kill_diff", "vs_pool_n", "pi_loss", "v_loss", "entropy", "std"])
    B, T = 2 * len(workers), args.rollout
    while total_steps < args.total_steps:
        buf_o = np.zeros((T, B, OBS_SIZE), np.float32)
        buf_a = np.zeros((T, B, ACT_SIZE), np.float32)
        buf_lp, buf_v, buf_r, buf_d, buf_m = (np.zeros((T, B), np.float32) for _ in range(5))
        ep_stats, vs_pool = [], []
        t0 = time.time()
        for t in range(T):
            raw = np.array([o[a] for o in obs for a in (1, 2)], np.float32)
            mask = np.array([0.0 if w.opp and w.opp_agent == a else 1.0
                             for w in workers for a in (1, 2)], np.float32)
            norm.update(raw[mask > 0])
            o = torch.as_tensor(norm(raw), dtype=torch.float32)
            with torch.no_grad():
                a, lp, v = net.act(o)
            a_np = a.numpy()
            for i, w in enumerate(workers):  # league opponents act with their own frozen net
                if w.opp:
                    k = 2 * i + w.opp_agent - 1
                    onet, onorm, _ = w.opp
                    with torch.no_grad():
                        a_np[k] = onet.dist(torch.as_tensor(onorm(raw[k:k + 1]), dtype=torch.float32)
                                            ).sample().numpy()[0]
            buf_o[t], buf_a[t], buf_lp[t], buf_v[t], buf_m[t] = o.numpy(), a_np, lp.numpy(), v.numpy(), mask
            clipped = np.clip(a_np, -1, 1)
            res = parallel(tp, lambda iw: iw[1].step(clipped[2 * iw[0]], clipped[2 * iw[0] + 1]),
                           list(enumerate(workers)))
            new_obs = []
            for i, (w, (o2, r, done, trunc)) in enumerate(zip(workers, res)):
                if trunc and o2 is not None:  # time limit: bootstrap from the value of the last state
                    with torch.no_grad():
                        vb = net.v(torch.as_tensor(norm(np.array([o2[1], o2[2]], np.float32)),
                                                   dtype=torch.float32)).squeeze(-1).numpy()
                    r = (r[0] + args.gamma * vb[0], r[1] + args.gamma * vb[1])
                buf_r[t, 2 * i], buf_r[t, 2 * i + 1] = r
                end = done or trunc or o2 is None
                buf_d[t, 2 * i] = buf_d[t, 2 * i + 1] = float(end)
                if end:
                    ep_stats.append((w.t, w.events["goals"], w.events["kills"]))
                    if w.opp:
                        me = 3 - w.opp_agent
                        vs_pool.append((w.ret[me], w.terms[me]["goal"], w.terms[me]["kill"]))
                    o2 = w.reset(pool)
                new_obs.append(o2)
            obs = new_obs
        total_steps += T * len(workers)
        sps = T * len(workers) / (time.time() - t0)

        # GAE
        with torch.no_grad():
            raw = np.array([o[a] for o in obs for a in (1, 2)], np.float32)
            last_v = net.v(torch.as_tensor(norm(raw), dtype=torch.float32)).squeeze(-1).numpy()
        adv = np.zeros((T, B), np.float32)
        gae = np.zeros(B, np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf_v[t + 1]
            nonterm = 1.0 - buf_d[t]
            delta = buf_r[t] + args.gamma * nv * nonterm - buf_v[t]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv[t] = gae
        ret = adv + buf_v

        # PPO update
        keep = torch.as_tensor(buf_m.reshape(-1) > 0)  # drop league-opponent transitions
        fo = torch.as_tensor(buf_o.reshape(-1, OBS_SIZE))[keep]
        fa = torch.as_tensor(buf_a.reshape(-1, ACT_SIZE))[keep]
        flp = torch.as_tensor(buf_lp.reshape(-1))[keep]
        fadv = torch.as_tensor(adv.reshape(-1))[keep]
        fret = torch.as_tensor(ret.reshape(-1))[keep]
        n = fo.shape[0]
        for _ in range(args.epochs):
            perm = torch.randperm(n)
            for s in range(0, n, args.minibatch):
                idx = perm[s:s + args.minibatch]
                d = net.dist(fo[idx])
                lp = d.log_prob(fa[idx]).sum(-1)
                ratio = (lp - flp[idx]).exp()
                ad = fadv[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pi_loss = -torch.min(ratio * ad, ratio.clamp(1 - args.clip, 1 + args.clip) * ad).mean()
                v_loss = 0.5 * (net.v(fo[idx]).squeeze(-1) - fret[idx]).pow(2).mean()
                ent = d.entropy().sum(-1).mean()
                loss = pi_loss + args.vf_coef * v_loss - args.ent_coef * ent
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(net.parameters(), 0.5)
                opt.step()
        update += 1

        for w in workers:
            for e in w.new_events:
                ev_f.write(json.dumps(e) + "\n")
            w.new_events.clear()
        ev_f.flush()
        status = {"run": args.run, "update": update, "steps": total_steps, "sps": sps, "t": time.time(),
                  "instances": [{"idx": w.idx, "sock": w.sock, "league": w.league,
                                 "opp": w.opp[2] if w.opp else None, "opp_agent": w.opp_agent}
                                for w in workers]}
        with open(os.path.join(run_dir, "status.json.tmp"), "w") as f:
            json.dump(status, f)
        os.replace(os.path.join(run_dir, "status.json.tmp"), os.path.join(run_dir, "status.json"))
        mean = lambda xs: float(np.mean(xs)) if xs else float("nan")
        el, eg, ek = (mean([e[j] for e in ep_stats]) for j in range(3))
        vp, vg, vk = (mean([e[j] for e in vs_pool]) for j in range(3))
        std = net.log_std.exp().mean().item()
        log.writerow([update, total_steps, round(sps, 1), len(ep_stats), el, eg, ek,
                      len(pool.files()), vp, vg, vk, len(vs_pool),
                      pi_loss.item(), v_loss.item(), ent.item(), std])
        log_f.flush()
        print(f"upd {update:4d} steps {total_steps:8d} sps {sps:5.0f} eps {len(ep_stats):2d} "
              f"goals/ep {eg:5.2f} kills/ep {ek:5.2f} | pool {len(pool.files()):3d} "
              f"vs_pool {vp:+.3f} (n={len(vs_pool)}) | pi {pi_loss.item():+.3f} "
              f"v {v_loss.item():.4f} std {std:.2f}", flush=True)
        if update % args.pool_every == 0:
            pool.add(net, norm, update)
        if update % args.save_every == 0:
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(), "norm": norm.state(),
                        "steps": total_steps, "update": update, "args": vars(args)}, ckpt_path)
    for w in workers:
        w.env.close()


def _alive(sock):
    import socket as _s
    try:
        c = _s.socket(_s.AF_UNIX, _s.SOCK_STREAM); c.connect(sock); c.close()
        return True
    except OSError:
        return False


def watch(args):
    ck = torch.load(os.path.join(HERE, "runs", args.run, "ckpt.pt"), weights_only=False)
    net, norm = ActorCritic(), RunningNorm(OBS_SIZE)
    net.load_state_dict(ck["net"]); norm.load(ck["norm"])
    print(f"watching {args.run} at {ck['steps']} steps (real time; ctrl-c to stop)")
    sock = os.path.join(SOCK_DIR, "spr-0.sock")
    if args.launch or not _alive(sock):
        sock = launch_instances(1, mute=False, size=(1280, 720))[0]
    env = PoleRidersEnv(frame_skip=args.frame_skip, fast=0, sock=sock)
    env.start_match(arena=args.arena)
    obs = env.reset()
    try:
        while True:
            raw = np.array([obs[1], obs[2]], np.float32)
            with torch.no_grad():
                d = net.dist(torch.as_tensor(norm(raw), dtype=torch.float32))
                a = (d.mean if args.deterministic else d.sample()).clamp(-1, 1).numpy()
            obs, rew, done, info = env.step({1: tuple(a[0]), 2: tuple(a[1])})
            tr = info["terms"][1]
            if tr["goal"] or tr["kill"]:
                print(f"score {info['state']['score']}  terms {tr}")
            if done or obs is None:
                obs = env.reset()
            time.sleep(args.frame_skip / 60)  # lockstep, paced to real time for viewing
    finally:
        env.close()


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="default")
    p.add_argument("--instances", type=int, default=4)
    p.add_argument("--total-steps", type=int, default=20_000_000, help="env steps (x2 transitions)")
    p.add_argument("--arena", type=int, default=2, help="1 Kalgoorlie, 2 Flatirons, 3 Dakhla, 4 Princeton")
    p.add_argument("--frame-skip", type=int, default=4)
    p.add_argument("--fast", type=int, default=20, help="draw every N-th frame while training")
    p.add_argument("--episode-steps", type=int, default=5400,
                   help="safety cap only (5400 x 4 ticks = 6 min); real matches end on game over")
    p.add_argument("--random-ball", type=float, default=0.0,
                   help="curriculum: start the ball at uniform x in [-v, v] meters")
    p.add_argument("--rollout", type=int, default=256)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=512)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--gamma", type=float, default=0.995)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--ent-coef", type=float, default=0.0)
    p.add_argument("--vf-coef", type=float, default=0.5)
    p.add_argument("--save-every", type=int, default=5)
    p.add_argument("--pool-every", type=int, default=20, help="freeze a snapshot every N updates")
    p.add_argument("--league-frac", type=float, default=0.5,
                   help="fraction of instances where one rider is a past snapshot")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--resume", action="store_true")
    p.add_argument("--watch", action="store_true")
    p.add_argument("--launch", action="store_true", help="(watch) relaunch the game")
    p.add_argument("--deterministic", action="store_true", help="(watch) use the policy mean")
    args = p.parse_args()
    watch(args) if args.watch else train(args)


if __name__ == "__main__":
    main()
