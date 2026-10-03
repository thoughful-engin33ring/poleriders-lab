#!/usr/bin/env python3
"""Equivalence tests for the batched native env (bridge/native_env.py, sim/src/env/).

    test_env_equiv.py unit                       # stick mapping, terrain, level geometry vs Python
    test_env_equiv.py python [--steps 3000]      # (a) native env vs the Python farm code on the same sim
                     [--goal-refund off|on|both] [--policy CKPT]   (default both: the reward rule off and on)
    test_env_equiv.py oracle [--steps 2000]      # (b) native env vs a real-game oracle (FarmHost)
    test_env_equiv.py dead [--steps 3000]        # killed riders are hidden in obs for their 3 s death
    test_env_equiv.py all

(a) The reference is the production Python path, literally: farm_env.FarmHost + farm_worker.step_out
    (PoleRidersEnv._stick, "%.4f" on the wire, snapshot JSON, observe2, compute_reward, resets,
    truncation, random.Random(h) coins), talking to a fake bridge whose `farm act` / `farm reset`
    run native matches (strtof + clamp like bridge/farm.c) and reply with the native snapshot_ps JSON.
    The env under test does all of that natively. Every output array must be bit-identical.
(b) Same comparison with the real game behind FarmHost: an oracle instance (socket
    spr-oracle-E.sock, level 1) running K shadows. Killed at the end.

Actions: per match and agent, random (run, aim_x, aim_y) held for 1-12 steps — uniform, extremes,
zeros, and a ball-chasing run command half the time (so goals, kills and sudden deaths happen) —
clipped to [-1, 1] like train_farm.py.
"""
import argparse, ctypes, json, os, random, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..", "..")
sys.path.insert(0, os.path.join(ROOT, "bridge"))
sys.path.insert(0, HERE)
from farm_env import FarmHost, Terrain  # noqa: E402
from farm_worker import step_out  # noqa: E402
from native_env import NativeHost, stick  # noqa: E402
from spr_env import PoleRidersEnv  # noqa: E402

KEYS = ["obs", "final", "rew", "terms", "reset", "trunc", "sd", "score", "sd_start"]
_libc = ctypes.CDLL(None)
_libc.strtof.restype = ctypes.c_float
_libc.strtof.argtypes = [ctypes.c_char_p, ctypes.c_void_p]


def strtof(s):
    return _libc.strtof(s.encode(), None)


def clampf(v, lo=-1.0, hi=1.0):
    return lo if v < lo else hi if v > hi else v


class FakeBridge:
    """The bridge's `farm act` / `farm reset` on native matches (a 1-host NativeHost in raw mode)."""

    def __init__(self, raw):
        self.raw = raw

    def cmd(self, line):
        p = line.split()
        if p[:2] == ["farm", "act"]:
            n = int(p[2])
            vals = np.array([clampf(strtof(v)) for v in p[3:]], np.float32)  # farm.c: strtof + clampf
            self.raw.raw_act(vals.reshape(self.raw.n, 6), n)
            return '{"matches":[' + ",".join(self.raw.json(k) for k in range(self.raw.n)) + "]}"
        if p[:2] == ["farm", "reset"]:
            clock = strtof(p[3]) if len(p) > 3 else 0.0
            return self.raw.raw_reset(int(p[2]), clock)
        raise ValueError(line)


class Actions:
    """Sticky random actions for N matches x 2 agents (see module doc)."""

    def __init__(self, n, seed):
        self.rng = np.random.default_rng(seed)
        self.n = n
        self.a = np.zeros((n, 2, 3), np.float32)
        self.hold = np.zeros((n, 2), int)
        self.chase = np.zeros((n, 2), bool)

    def __call__(self, obs):
        r = self.rng
        for k in range(self.n):
            for g in range(2):
                if self.hold[k, g] <= 0:
                    self.hold[k, g] = r.integers(1, 13)
                    kind = r.integers(0, 4)
                    if kind == 0:
                        v = r.uniform(-1, 1, 3)
                    elif kind == 1:
                        v = r.choice([-1.0, 0.0, 1.0], 3)
                    elif kind == 2:
                        v = r.normal(0, 1.2, 3)
                    else:
                        v = np.array([r.uniform(-1, 1), np.cos(r.uniform(0, 6.3)), np.sin(r.uniform(0, 6.3))])
                    self.a[k, g] = v
                    self.chase[k, g] = r.random() < 0.5
                self.hold[k, g] -= 1
                if self.chase[k, g]:  # run at the ball (obs[60] = ball x relative to me, my frame)
                    self.a[k, g, 0] = np.clip(obs[k, g, 60] + 0.3, -1, 1)
        return np.clip(self.a, -1, 1).astype(np.float32)


class PolicyActions:
    """Actions from a trained checkpoint (both agents, stochastic, seeded): real play with goals and
    kills in normal time, which random sticks rarely produce."""

    def __init__(self, ckpt, seed):
        import torch
        sys.path.insert(0, os.path.join(HERE, "..", ".."))
        from train_ppo import ActorCritic, RunningNorm
        ck = torch.load(ckpt, weights_only=False)
        size = ck["net"]["pi.0.weight"].shape[1]
        self.net, self.norm = ActorCritic(size), RunningNorm(size)
        self.net.load_state_dict(ck["net"]); self.norm.load(ck["norm"])
        self.size, self.torch = size, torch
        torch.manual_seed(seed)

    def __call__(self, obs):
        torch = self.torch
        n = obs.shape[0]
        x = torch.as_tensor(self.norm(obs.reshape(-1, obs.shape[-1])[:, :self.size]), dtype=torch.float32)
        with torch.no_grad():
            a = self.net.dist(x).sample().numpy()
        return np.clip(a, -1, 1).reshape(n, 2, 3).astype(np.float32)


def bits(x):
    x = np.asarray(x)
    return x.view(np.uint32) if x.dtype == np.float32 else x


class Compare:
    def __init__(self):
        self.bad = {k: 0 for k in KEYS}
        self.first = []
        self.n = 0

    def __call__(self, t, ref, nat, keys=KEYS):
        self.n += 1
        for key in keys:
            a, b = bits(ref[key]), bits(nat[key])
            if a.shape != b.shape or not np.array_equal(a, b):
                self.bad[key] += 1
                if len(self.first) < 8:
                    if a.shape != b.shape:
                        self.first.append(f"step {t} {key}: shapes {a.shape} vs {b.shape}")
                    else:
                        i = tuple(int(v) for v in np.argwhere(a != b)[0])
                        self.first.append(f"step {t} {key}{list(i)}: python/oracle {np.asarray(ref[key])[i]!r} "
                                          f"native {np.asarray(nat[key])[i]!r}")

    def ok(self):
        return not any(self.bad.values())

    def report(self, label):
        print(f"{label}: {self.n} comparisons, mismatching steps per key: {self.bad}")
        for f in self.first:
            print("   ", f)


class Events:
    def __init__(self):
        self.goals = self.kills = self.resets = self.truncs = self.sd_steps = self.sd_starts = 0

    def add(self, out):
        self.goals += int(np.abs(out["terms"][:, 0, 0]).sum())
        self.kills += int(np.abs(out["terms"][:, 0, 1]).sum())
        self.resets += int(out["reset"].sum())
        self.truncs += int(out["trunc"].sum())
        self.sd_steps += int(out["sd"].sum())
        self.sd_starts += int((out["reset"] & out["sd_start"]).sum())

    def __str__(self):
        return (f"goals {self.goals}, kills {self.kills}, resets {self.resets} (truncated {self.truncs}, "
                f"new sudden-death starts {self.sd_starts}), steps in sudden death {self.sd_steps}")


# ---------------------------------------------------------------- unit checks

def test_unit():
    ok = True
    # 1. stick mapping: PoleRidersEnv._stick (numpy float32 actions) -> "%.4f" -> strtof + clampf
    rng = np.random.default_rng(1)
    acts = np.concatenate([rng.uniform(-1, 1, (60000, 3)), rng.normal(0, 2, (30000, 3)),
                           rng.choice([-1, 0, 1, 1e-5, -1e-5, 0.5, -0.5, 0.03125, -0.03125], (10000, 3)),
                           np.clip(rng.normal(0, 1.2, (20000, 3)), -1, 1)]).astype(np.float32)
    bad = 0
    for a in acts:
        for agent in (1, 2):
            ref = [clampf(strtof(f"{v:.4f}")) for v in PoleRidersEnv._stick(agent, tuple(a))]
            nat = stick(agent, a)
            if np.array(ref, np.float32).view(np.uint32).tolist() != np.array(nat, np.float32).view(np.uint32).tolist():
                bad += 1
                if bad < 4:
                    print(f"  stick mismatch agent {agent} act {a.tolist()}: python {ref} native {nat}")
    print(f"stick mapping: {2 * len(acts)} actions, {bad} mismatches")
    ok &= bad == 0
    # 2. level geometry: the native world's static polygons vs arenas/arena2.json (bridge `level`)
    h = NativeHost(1, 1)
    h.reset_all()
    lv = json.loads(h.level_json())
    ref = json.load(open(os.path.join(ROOT, "arenas", "arena2.json")))
    same = lv["bodies"] == ref["bodies"]  # (the file also has spawn/ball positions of the real match)
    print(f"level id 1 static geometry == arenas/arena2.json: {same} "
          f"({len(lv['bodies'])} static bodies, {sum(len(b['polys']) for b in lv['bodies'])} polygons)")
    ok &= same
    # 3. terrain table and queries vs farm_env.Terrain(2)
    t = Terrain(2)
    cols = sum(h.terrain_column(i) != t.surf[i] for i in range(len(t.surf)))
    extra = h.terrain_column(len(t.surf)) is not None
    qbad = 0
    for x in np.concatenate([rng.uniform(-50, 50, 20000), np.arange(-46, 46, 0.025)]):
        for y in (-12.0, -8.0, -3.0, -0.5, 0.0, 0.3, 1.5, 3.0, float(rng.uniform(-12, 5))):
            x = float(x)
            if t.floor(x, y) != h.terrain_floor(x, y) or t.above(x, y) != h.terrain_above(x, y):
                qbad += 1
    print(f"terrain: {len(t.surf)} columns, {cols} differ{' (native has extra columns)' if extra else ''}; "
          f"{qbad} differing floor/above queries")
    ok &= cols == 0 and not extra and qbad == 0
    h.close()
    return ok


# ---------------------------------------------------------------- (a) native vs Python farm code

def test_python(steps, hosts, K, seed, sd_frac, max_steps, frame_skip=4, threads=0, goal_refund=False, policy=None):
    print(f"(a) native env vs FarmHost + step_out on native matches: {hosts} hosts x {K} matches, {steps} steps, "
          f"frame_skip {frame_skip}, sd_frac {sd_frac}, max_steps {max_steps}, goal_refund {goal_refund}"
          + (f", policy {policy}" if policy else ""))
    nat = NativeHost(hosts, K, frame_skip, sd_frac, max_steps, seed=seed, threads=threads, goal_refund=goal_refund)
    refs = []
    for h in range(hosts):
        fh = FarmHost("fake", K, 2, frame_skip, sudden_death_frac=sd_frac, max_steps=max_steps, seed=seed + h,
                      goal_refund=goal_refund)
        fh.b, fh.level = FakeBridge(NativeHost(1, K, threads=1)), 1
        refs.append(fh)
    cmp_, ev = Compare(), Events()
    ref_obs = np.concatenate([np.array([[o[1], o[2]] for o in fh.reset_all()], np.float32) for fh in refs])
    ref_sd = np.concatenate([np.array(fh.sd_start) for fh in refs])
    nat.reset_all()
    cmp_(0, {"obs": ref_obs, "sd_start": ref_sd}, nat.out, ["obs", "sd_start"])
    act = PolicyActions(policy, seed + 99) if policy else Actions(hosts * K, seed + 99)
    t0 = time.time()
    for t in range(1, steps + 1):
        a = act(nat.out["obs"])
        ref = [step_out(fh, a[h * K:(h + 1) * K]) for h, fh in enumerate(refs)]
        ref = {k: np.concatenate([r[k] for r in ref]) for k in KEYS}
        out = nat.step(a)
        cmp_(t, ref, out)
        ev.add(ref)
        if t % 500 == 0:
            print(f"  step {t}: {time.time() - t0:.0f}s, mismatching steps so far {sum(cmp_.bad.values())}; {ev}",
                  flush=True)
    cmp_.report("(a)")
    print(f"    events (reference): {ev}")
    return cmp_.ok()


# ---------------------------------------------------------------- dead riders are hidden

def test_dead(steps, K, seed, frame_skip=4):
    """Kills in a native rollout: for the victim's whole death window (PlayState::respawnTimer, 3 s) the
    killer's observation must hide it (opp block all zeros except ghost = 1 and a respawn_in >= 0 that
    decreases, holding still only during the post-kill hitstop), and the victim's own observation must have ghost flag 1 and the same respawn_in.
    Kills are found from the snapshot stats (victim deaths +1; sudden-death KOs end the match and are
    skipped); the window ends at the victim's respawn
    (snapshot "dead" false), a goal (everyone respawns), or a match reset."""
    print(f"dead-rider check: native, {K} matches, {steps} steps, frame_skip {frame_skip}")
    nat = NativeHost(1, K, frame_skip, 0.0, 10 ** 9, seed=seed)
    nat.record(True)
    nat.reset_all()
    act = Actions(K, seed + 3)
    prev = [json.loads(nat.json(k, 2)) for k in range(K)]
    watch = {}  # k -> dict(victim, killer, steps, last respawn_in)
    kills = sd_kills = windows = checked = bad = early = 0
    first = []

    def fail(msg):
        nonlocal bad
        bad += 1
        if len(first) < 10:
            first.append(msg)

    for t in range(1, steps + 1):
        out = nat.step(act(nat.out["obs"]))
        obs, reset = out["obs"], out["reset"]
        for k in range(K):
            st = json.loads(nat.json(k, 1))
            pl = {p["id"]: p for p in st.get("players", [])}
            pp = {p["id"]: p for p in prev[k].get("players", [])}
            w = watch.get(k)
            if w and (reset[k] or st.get("score") != prev[k].get("score") or not pl
                      or not pl[w["victim"]].get("dead", w["steps"] < 44)):  # no "dead" key: 44 steps
                if w["steps"] < 40 and not reset[k] and st.get("score") == prev[k].get("score"):
                    kd = pl[w["killer"]]["stats"] if pl else {}
                    bx0, bx1 = (prev[k].get("ball") or {}).get("x"), (st.get("ball") or {}).get("x")
                    # legit early respawns: a ball reset (rebuilt at x = 0; respawnTimer 0.5*dt for everyone), or the killer also died
                    # while the victim waited (all dead -> everyone respawns next tick)
                    if not (pl and (abs((bx1 or 0) - (bx0 or 0)) > 5 or (bx1 == 0.0 and bx0 != 0.0)
                                    or kd["deaths"] + kd["suicides"] > w["kd"])):
                        fail(f"match {k} step {t}: victim {w['victim']} back after only {w['steps']} steps "
                             f"(ball x {bx0} -> {bx1})")
                    else:
                        early += 1
                if w["steps"] > 1 and not w["last"] < w["first"]:
                    fail(f"match {k} step {t}: respawn_in never decreased ({w['first']} -> {w['last']})")
                windows += w["steps"] > 0
                del watch[k]
                w = None
            if not reset[k] and pl and pp and k not in watch:
                for v in (1, 2):
                    if pl[v]["stats"]["deaths"] > pp[v]["stats"]["deaths"]:
                        if st.get("sudden_death"):  # the deciding KO: no respawn timer, the match ends
                            sd_kills += 1
                            break
                        kills += 1
                        ks = pl[3 - v]["stats"]
                        watch[k] = w = {"victim": v, "killer": 3 - v, "steps": 0, "last": None,
                                        "kd": ks["deaths"] + ks["suicides"]}
                        break
            if w and not reset[k]:
                v, kl = w["victim"], w["killer"]
                ko, vo = obs[k, kl - 1], obs[k, v - 1]
                blk = ko[17:34]
                rest = np.delete(blk, [9, 16])
                ri = float(blk[16])
                checked += 1
                if blk[9] != 1.0 or np.any(rest != 0):
                    fail(f"match {k} step {t} (+{w['steps']}): killer {kl} sees victim {v} as live "
                         f"(ghost {blk[9]}, x {blk[0]:.2f}, respawn_in {ri:.4f})")
                elif not ri >= 0 or (w["last"] is not None and not ri <= w["last"]):
                    # (it holds still during the post-kill hitstop, when PlayState::Update freezes)
                    fail(f"match {k} step {t} (+{w['steps']}): respawn_in {ri} increased (was {w['last']})")
                if vo[9] != 1.0 or vo[16] != blk[16]:
                    fail(f"match {k} step {t} (+{w['steps']}): victim's own ghost flag {vo[9]}, respawn_in "
                         f"{vo[16]:.4f} (killer sees {ri:.4f})")
                w["last"] = ri
                w.setdefault("first", ri)
                w["steps"] += 1
            prev[k] = json.loads(nat.json(k, 2)) if reset[k] else st
    nat.close()
    print(f"  kills {kills} (+{sd_kills} sudden-death KOs, not checked), finished death windows {windows} ({early} ended early by a ball reset / the killer's death), victim-steps checked {checked}, failures {bad}")
    for f in first:
        print("   ", f)
    ok = bad == 0 and kills > 0
    print(f"  dead-rider check: {'OK' if ok else 'FAILED'}")
    return ok


# ---------------------------------------------------------------- (b) native vs the real game

def test_oracle(steps, K, seed, sd_frac, max_steps, frame_skip=4, keep=False, policy=None, goal_refund=False):
    from oracle import Oracle, kill_oracles, SOCK
    sock = os.path.join(os.path.dirname(SOCK), "spr-oracle-E.sock")
    print(f"(b) native env vs real-game oracle ({os.path.basename(sock)}, level 1): {K} matches, {steps} steps, "
          f"frame_skip {frame_skip}, sd_frac {sd_frac}, max_steps {max_steps}")
    o = Oracle(sock=sock, n=K, arena=2, level=1)
    try:
        try:
            if not os.path.exists(sock):
                raise FileNotFoundError(sock)
            o.boot(launch=False)
        except Exception:  # noqa: BLE001 — no oracle on this socket yet
            o.boot(launch=True)
        d0 = o.dump(0)
        sess = [bytes.fromhex(d0["objects"][f"player{i}"]["raw"])[0x220:0x23c] for i in (0, 1)]
        import struct
        recs = [struct.unpack("<7i", s) for s in sess]
        heads = (recs[0][3], recs[1][3])
        expect = [(0, -1, 0, heads[0], 0, 7, 0), (0, 1, 1, heads[1], 0, 7, 0)]
        print(f"  oracle session records {recs} (heads {heads}); native defaults match: {list(recs) == expect}")
        fh = FarmHost(sock, K, 2, frame_skip, sudden_death_frac=sd_frac, max_steps=max_steps, seed=seed,
                      goal_refund=goal_refund)
        fh.b, fh.level = o.host.b, o.host.level
        nat = NativeHost(1, K, frame_skip, sd_frac, max_steps, seed=seed, heads=heads, goal_refund=goal_refund)
        cmp_, ev = Compare(), Events()
        ref_obs = np.array([[x[1], x[2]] for x in fh.reset_all()], np.float32)
        nat.reset_all()
        cmp_(0, {"obs": ref_obs, "sd_start": np.array(fh.sd_start)}, nat.out, ["obs", "sd_start"])
        act = PolicyActions(policy, seed + 7) if policy else Actions(K, seed + 7)
        t0 = time.time()
        for t in range(1, steps + 1):
            a = act(nat.out["obs"])
            ref = step_out(fh, a)
            out = nat.step(a)
            cmp_(t, ref, out)
            ev.add(ref)
            if t % 250 == 0:
                print(f"  step {t}: {time.time() - t0:.0f}s, mismatching steps so far {sum(cmp_.bad.values())}; {ev}",
                      flush=True)
        cmp_.report("(b)")
        print(f"    events (oracle): {ev}")
        return cmp_.ok()
    finally:
        if not keep:
            kill_oracles(sock)
            print(f"  oracle on {os.path.basename(sock)} killed")


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("what", choices=["unit", "python", "oracle", "dead", "all"])
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--hosts", type=int, default=2)
    ap.add_argument("--matches", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--sd-frac", type=float, default=0.4)
    ap.add_argument("--max-steps", type=int, default=600)
    ap.add_argument("--frame-skip", type=int, default=4)
    ap.add_argument("--keep", action="store_true", help="leave the oracle game running")
    ap.add_argument("--policy", default=None, help="checkpoint whose policy plays both sides (python / oracle test)")
    ap.add_argument("--goal-refund", choices=["off", "on", "both"], default="both",
                    help="the goal-refund reward rule on both sides (python: both = run off, then on; oracle: on "
                         "unless off)")
    args = ap.parse_args()
    ok = True
    if args.what in ("unit", "all"):
        ok &= test_unit()
    if args.what in ("python", "all"):
        for gr in {"off": [False], "on": [True], "both": [False, True]}[args.goal_refund]:
            ok &= test_python(args.steps or 3000, args.hosts, args.matches, args.seed, args.sd_frac, args.max_steps,
                              args.frame_skip, goal_refund=gr, policy=args.policy)
    if args.what in ("dead", "all"):
        ok &= test_dead(args.steps or 3000, args.matches, args.seed, args.frame_skip)
    if args.what in ("oracle", "all"):
        ok &= test_oracle(args.steps or 2000, args.matches, args.seed, args.sd_frac, args.max_steps, args.frame_skip,
                          args.keep, args.policy, goal_refund=args.goal_refund != "off")
    print("ALL IDENTICAL" if ok else "DIFFERENCES FOUND")
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
