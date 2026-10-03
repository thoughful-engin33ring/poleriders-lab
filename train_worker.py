#!/usr/bin/env python3
"""Goal-conditioned "worker" pretraining (the lower level of a manager/worker hierarchy).

The worker gets the normal observation plus a target and learns to reach it; a manager trained on
the game reward will later issue these targets. Targets, in each rider's own frame (x toward the goal
it attacks, y up), are drawn at random so the worker must learn movement in every direction —
including getting far back behind itself, which the flat game policy never learned:
    body   (50%): body x at target_x = my x + U(-16, 16) m (height free)
    tip    (25%): pole tip at (my x + U(-6, 6), U(0, 6) m above the ground) — the pole is ~6 m (10
                  segments 0.61 m apart; tip ~5 m from the body), so low / planted tips are included and
                  nothing above what the tip can reach (farm1's tip: p99 6.3 m high)
    both   (25%): both at once (the tip target then lies within pole reach of the BODY target)
Mirror symmetry (--mirror, default): the policy always sees a frame in which its target is ahead;
rows with the target behind are mirrored in and their actions mirrored back ("reverse the controls").
Reward: 0.1 per metre of progress toward each active target, +1 on reaching it (body within 0.75 m in
x, tip within 1 m), -0.5 for dying. A target not reached within 6 s (90 steps) is a failure and is
replaced. Both riders of every match are workers with independent targets (real ball, collisions,
kills). Native simulator only.

    .venv/bin/python train_worker.py --run worker1 --init runs/farm1/ckpt.pt
"""
import argparse, copy, csv, json, math, os, sys, time

import numpy as np
import torch
import torch.nn as nn

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bridge"))
from farm_env import OBS2_SIZE  # noqa: E402
from native_env import NativeHost  # noqa: E402
from train_ppo import ACT_SIZE, ActorCritic, RunningNorm, net_from_ckpt  # noqa: E402

GOAL_SIZE = 10  # (+ body target dy /10 and an air flag, appended 2026-09-30 for jump targets)
# body dx/10, body on, tip dx/10, tip dy/10 (target - tip), tip on, steps left / limit,
#               tip target - body dx/10, dy/10 (the direction to point the pole in; added 2026-09-30,
#               older 6-feature workers just use the first 68 inputs)
OBS_W = OBS2_SIZE + GOAL_SIZE
LIMIT = 90     # steps per target (6 s)
BODY_R, TIP_R = 0.75, 1.0
WALL_X = 20.3  # goal-wall face (the `wall` bodies are solid blocks from +-20.3 m outward)
X_MAX = WALL_X - 0.5  # body targets: the rider (half-width 0.375 m) must fit before the wall
TIP_CLEAR, TIP_REACH = 0.3, 5.3  # tip targets: clearance from any solid; max distance from where the
#                                   body stands (the tip sits ~5 m from the body; was 10.5 m, assuming a 12 m pole)
TIP_DX, TIP_H = 6.0, (0.0, 6.0)   # tip target: horizontal offset from the body, height above the ground
BOTH_RING = (4.3, 5.6)            # "both" targets: tip distance from the body target (the rigid pole's reach)
AIR_FRAC, AIR_H, AIR_DX, AIR_R = 0.4, (1.5, 7.0), 12.0, 1.0  # air body targets: share, height above the
#                                  surface, horizontal range, radius (farm1 airborne height p50 4 m, p90 6 m)
PART_BONUS = 0.02                 # "both": per step, per correctly satisfied part (the right assignment pays
#                                  before full success; the swapped body/tip arrangement doesn't)
AIR_BUCKETS = [(1.5, 3.0), (3.0, 4.5), (4.5, 6.0), (6.0, 7.5)]  # air target height above the surface (m)
BUCKETS = [(-40, -10), (-10, -5), (-5, -1), (-1, 1), (1, 5), (5, 10), (10, 40)]  # body target dx (m)


# ---------------------------------------------------------------- mirror symmetry
# The game is left-right symmetric (the teams' riders are mirror images), so moving toward a target
# behind me is moving forward with the controls reversed. With --mirror the worker always acts in a
# frame where its current target is AHEAD: rows whose target is behind are mirrored before the policy
# and its action mirrored back. Feature layout: farm_env.observe2 + the 6 goal features above.
def _mirror_spec():
    neg = []
    for base in (0, 17):  # me, opponent: x, vx, sin(body a), w, pole cos, pole w, tip dx
        neg += [base + i for i in (0, 2, 4, 6, 11, 12, 13)]
    neg += [52, 54, 60, 62, 64, 68]  # ball x, ball vx, ball x rel, body/tip target dx, tip target - body dx
    perm = list(range(OBS_W))
    for start in (34, 43):  # terrain samples at -6 -4 -2 -1 0 1 2 4 6 m: reverse the order
        for i in range(9):
            perm[start + i] = start + 8 - i
    sign = np.ones(OBS_W, np.float32)
    sign[neg] = -1
    return np.array(perm), sign


MIRROR_PERM, MIRROR_SIGN = _mirror_spec()
ACT_MIRROR = np.array([-1.0, -1.0, 1.0], np.float32)  # run, aim x, aim y


def mirror(x, rows):
    """Mirror the observation+goal rows `rows` (bool mask) of x in place-safe fashion."""
    y = x.copy()
    y[rows] = x[rows][:, MIRROR_PERM] * MIRROR_SIGN
    return y


def target_behind(goals, O, stable=False):
    """Rows to mirror. stable=True (workers trained with --flip goal): decided once per target, when it
    is set (target behind the BODY then) — the frame never flips mid-target. stable=False (older
    workers): re-decided every step from the body target, or for tip targets from the target vs the
    TIP, which flipped the frame every ~0.7 s near a tip target and cost ~20 points of tip success."""
    if stable:
        return goals.flip0.copy()
    me_x = O[:, 0]
    dx = np.where(goals.body_on, goals.bx - me_x, goals.tx - (me_x + O[:, 13]))
    return dx < 0


class Solid:
    """Point-in-solid test against the arena's ground and wall polygons (arenas/arena2.json = level 1,
    the training map; Box2D y down). The arena is mirror-symmetric, so agent-frame points work too."""

    def __init__(self, path=os.path.join(HERE, "arenas", "arena2.json")):
        g = json.load(open(path))
        self.polys = [np.array(p["v"], float) for b in g["bodies"] if b["tag"] in ("ground", "wall")
                      for p in b["polys"] if not p["sensor"]]

    def inside(self, x, y_up, margin=0.0):
        """Bool array: point (x, y up) within `margin` of a solid (tested at the point and 4 offsets)."""
        x, yd = np.asarray(x, float), -np.asarray(y_up, float)
        hit = np.zeros(x.shape, bool)
        offs = [(0, 0)] + ([(margin, 0), (-margin, 0), (0, margin), (0, -margin)] if margin else [])
        for P in self.polys:
            xs, ys = P[:, 0], P[:, 1]
            x2, y2 = np.roll(xs, -1), np.roll(ys, -1)
            for ox, oy in offs:
                px, py = x[..., None] + ox, yd[..., None] + oy
                dy = np.where(y2 != ys, y2 - ys, 1e-12)
                cross = ((ys > py) != (y2 > py)) & (px < (x2 - xs) * (py - ys) / dy + xs)
                hit |= cross.sum(-1) % 2 == 1
        return hit


SOLID = None


def _solid():
    global SOLID
    if SOLID is None:
        SOLID = Solid()
    return SOLID


class Surface:
    """Height (y up, m) of the top ground surface at x on the training map (arenas/arena2.json)."""

    def __init__(self):
        from farm_env import Terrain
        t = Terrain(2)
        self.x0, self.dx = t.x0, t.dx
        self.top = np.array([-min(c) if c else 0.0 for c in t.surf])

    def __call__(self, x):
        i = np.clip(np.round((np.asarray(x) - self.x0) / self.dx).astype(int), 0, len(self.top) - 1)
        return self.top[i]


SURFACE = None


def _surface():
    global SURFACE
    if SURFACE is None:
        SURFACE = Surface()
    return SURFACE


class Goals:
    """Per-rider targets (agent frame) and bookkeeping; B = 2 x matches rows.

    Body targets are either on the ground (x only: be there, standing or not) or in the AIR (pass
    through the point (bx, by) within AIR_R — a jump/vault to that height; heights 1.5-7 m above the
    surface under it, from farm1's airborne heights: median 4 m, p90 6 m). Tip targets are points."""

    def __init__(self, B, rng):
        self.rng, self.B = rng, B
        self.body_on, self.tip_on = np.zeros(B, bool), np.zeros(B, bool)
        self.air = np.zeros(B, bool)
        self.bx, self.by, self.tx, self.ty = np.zeros(B), np.zeros(B), np.zeros(B), np.zeros(B)
        self.left, self.age = np.zeros(B, int), np.zeros(B, int)
        self.d0 = np.zeros(B)  # initial body dx (for success-by-distance stats)
        self.flip0 = np.zeros(B, bool)  # mirror decision for the current target (stable frame)
        self.prev = np.zeros(B)  # previous total distance to the active targets
        # jump-height curriculum (--air-curriculum): probability of each AIR_BUCKETS band; None = uniform heights
        self.air_p = None

    def air_heights(self, n):
        """Heights above the surface for n air targets: uniform over AIR_H, or per band with air_p."""
        r = self.rng
        if self.air_p is None:
            return r.uniform(*AIR_H, n)
        b = r.choice(len(AIR_BUCKETS), n, p=self.air_p)
        lo = np.array([x[0] for x in AIR_BUCKETS])[b]
        hi = np.minimum(np.array([x[1] for x in AIR_BUCKETS])[b], AIR_H[1])
        return r.uniform(lo, hi)

    def sample(self, rows, O):
        r = self.rng
        n = len(rows)
        if not n:
            return
        rows = np.asarray(rows)
        surf = _surface()
        solid = _solid()
        kind = r.choice(3, n, p=[0.5, 0.25, 0.25])  # body, tip, both
        me_x, me_y = O[rows, 0], O[rows, 1]
        self.body_on[rows] = kind != 1
        self.tip_on[rows] = kind != 0
        air = (r.random(n) < AIR_FRAC) & (kind != 1)
        self.air[rows] = air
        # body target: re-draw until reachable (not clip, which would pile targets up at the wall)
        span = np.where(air, AIR_DX, 16.0)
        bx = me_x + r.uniform(-1, 1, n) * span
        by = np.zeros(n)
        for _ in range(12):
            by = surf(bx) + np.where(air, self.air_heights(n), 0.69)
            bad = (np.abs(bx) > X_MAX) | (air & solid.inside(bx, by, 0.5))
            if not bad.any():
                break
            k = bad.sum()
            bx[bad] = me_x[bad] + r.uniform(-1, 1, k) * span[bad]
        bx = np.clip(bx, -X_MAX, X_MAX)
        by = surf(bx) + np.where(air, np.clip(by - surf(bx), *AIR_H), 0.69)
        self.bx[rows], self.by[rows], self.d0[rows] = bx, by, bx - me_x
        # the tip must be reachable from where the body will be: for "both" targets that is the body
        # target point (standing or in the air); tip-only targets anywhere the tip can go
        both = self.body_on[rows] & self.tip_on[rows]
        cx = np.where(both, bx, me_x)

        def draw(idx):
            """Tip targets for rows idx (local): tip-only within reach of the rider, 0-6 m above the
            surface under the target; "both" on the ring the tip actually lies on when the body is at
            its target (the rigid pole keeps the tip 4.6-5.5 m from the body in farm1's play)."""
            k = len(idx)
            txs = cx[idx] + r.uniform(-TIP_DX, TIP_DX, k)
            tys = surf(txs) + r.uniform(*TIP_H, k)
            bm = both[idx]
            if bm.any():
                th = r.uniform(np.radians(-15), np.radians(195), bm.sum())
                rho = r.uniform(*BOTH_RING, bm.sum())
                bi = idx[bm]
                txs[bm] = bx[bi] + rho * np.cos(th)
                tys[bm] = by[bi] + rho * np.sin(th)
            return txs, tys

        tx, ty = draw(np.arange(n))
        for _ in range(12):
            bad = ((np.abs(tx) > X_MAX) | (~both & (np.hypot(tx - cx, ty - me_y) > TIP_REACH))
                   | solid.inside(tx, ty, TIP_CLEAR))
            if not bad.any():
                break
            bi = np.nonzero(bad)[0]
            tx[bi], ty[bi] = draw(bi)
        else:  # rare: fall back to a point above where the body will be
            tx[bad], ty[bad] = cx[bad], np.where(both, by, me_y)[bad] + 5.0
        self.tx[rows], self.ty[rows] = tx, ty
        self.left[rows], self.age[rows] = LIMIT, 0
        self.prev[rows] = self.dist(O)[rows]
        self.set_flip(rows, O)

    def set_flip(self, rows, O):
        """Fix each row's mirror frame for its current target: mirrored if the target (body target, or
        the tip target when only the tip is active) lies behind the body right now."""
        me_x = O[rows, 0]
        dx = np.where(self.body_on[rows], self.bx[rows] - me_x, self.tx[rows] - me_x)
        self.flip0[rows] = dx < 0

    def _body_d(self, O):
        me_x, me_y = O[:, 0], O[:, 1]
        return np.where(self.air, np.hypot(self.bx - me_x, self.by - me_y), np.abs(self.bx - me_x))

    def dist(self, O):
        """Total distance (m) to each row's active targets."""
        me_x, me_y = O[:, 0], O[:, 1]
        tip_x, tip_y = me_x + O[:, 13], me_y + O[:, 14]
        dt = np.hypot(self.tx - tip_x, self.ty - tip_y)
        return np.where(self.body_on, self._body_d(O), 0.0) + np.where(self.tip_on, dt, 0.0)

    def parts_ok(self, O):
        """(body on target, tip on target) per row, regardless of whether the part is active."""
        me_x, me_y = O[:, 0], O[:, 1]
        tip_x, tip_y = me_x + O[:, 13], me_y + O[:, 14]
        return (self._body_d(O) < np.where(self.air, AIR_R, BODY_R),
                np.hypot(self.tx - tip_x, self.ty - tip_y) < TIP_R)

    def reached(self, O):
        bok, tok = self.parts_ok(O)
        return (~self.body_on | bok) & (~self.tip_on | tok) & (O[:, 9] < 0.5)

    def feats(self, O):
        me_x, me_y = O[:, 0], O[:, 1]
        tip_x, tip_y = me_x + O[:, 13], me_y + O[:, 14]
        b, t = self.body_on.astype(np.float32), self.tip_on.astype(np.float32)
        return np.stack([b * (self.bx - me_x) / 10, b, t * (self.tx - tip_x) / 10, t * (self.ty - tip_y) / 10,
                         t, self.left / LIMIT, t * (self.tx - me_x) / 10, t * (self.ty - me_y) / 10,
                         b * self.air * (self.by - me_y) / 10, b * self.air],
                        1).astype(np.float32)


def widen(sd, old_in, new_in):
    """Checkpoint surgery: zero weights for the new (goal) inputs, so the network starts as the game
    policy and learns to use the target."""
    for head in ("pi", "v"):
        w = sd[f"{head}.0.weight"]
        if w.shape[1] < new_in:
            sd[f"{head}.0.weight"] = torch.cat([w, torch.zeros(w.shape[0], new_in - w.shape[1])], 1)
    return sd


def train(args):
    run_dir = os.path.join(HERE, "runs", args.run)
    os.makedirs(run_dir, exist_ok=True)
    ckpt_path = os.path.join(run_dir, "ckpt.pt")
    torch.manual_seed(args.seed)
    torch.set_num_threads(args.torch_threads)
    rng = np.random.default_rng(args.seed)
    net, norm = ActorCritic(OBS_W), RunningNorm(OBS_W)
    opt = torch.optim.Adam(net.parameters(), lr=args.lr, eps=1e-5)
    steps, update = 0, 0
    if os.path.exists(ckpt_path):
        ck = torch.load(ckpt_path, weights_only=False)
        old_in = ck["net"]["pi.0.weight"].shape[1]
        if old_in < OBS_W:  # the goal features grew: widen with zero weights, restart Adam
            net.load_state_dict(widen(ck["net"], old_in, OBS_W))
            ns = ck["norm"]
            norm.load({"mean": np.concatenate([ns["mean"], np.zeros(OBS_W - old_in)]),
                       "var": np.concatenate([ns["var"], np.ones(OBS_W - old_in)]), "count": ns["count"]})
            print(f"inputs grew {old_in} -> {OBS_W}: network widened, optimizer restarted", flush=True)
        else:
            net.load_state_dict(ck["net"]); opt.load_state_dict(ck["opt"]); norm.load(ck["norm"])
        steps, update = ck["steps"], ck["update"]
        print(f"resumed at {steps} steps, update {update}", flush=True)
        if args.reset_std:  # re-open exploration (e.g. for a new skill) without losing the learned mean
            with torch.no_grad():
                net.log_std.fill_(math.log(args.reset_std))
            print(f"action std reset to {args.reset_std}", flush=True)
    elif args.init:
        ck = torch.load(args.init, weights_only=False)
        old_in = ck["net"]["pi.0.weight"].shape[1]
        net.load_state_dict(widen(ck["net"], old_in, OBS_W))
        ns = ck["norm"]
        norm.load({"mean": np.concatenate([ns["mean"], np.zeros(OBS_W - len(ns["mean"]))]),
                   "var": np.concatenate([ns["var"], np.ones(OBS_W - len(ns["var"]))]), "count": ns["count"]})
        with torch.no_grad():
            net.log_std.fill_(math.log(args.init_std))
        print(f"initialised from {args.init} ({old_in} -> {OBS_W} inputs), std {args.init_std}", flush=True)

    # PPO runs on a learner copy on --ppo-device (the Apple GPU is ~2.5x the CPU); the rollout acts on the CPU
    dev = torch.device(args.ppo_device)
    lnet = copy.deepcopy(net).to(dev)
    opt_state = opt.state_dict()
    opt = torch.optim.Adam(lnet.parameters(), lr=args.lr, eps=1e-5)
    opt.load_state_dict(opt_state)

    def to_cpu(x):
        if torch.is_tensor(x):
            return x.cpu()
        if isinstance(x, dict):
            return {k: to_cpu(v) for k, v in x.items()}
        if isinstance(x, list):
            return [to_cpu(v) for v in x]
        return x

    N = args.hosts * args.matches
    B = 2 * N
    g_, sec_ = args.format.split(",")
    env = NativeHost(args.hosts, args.matches, 4, sd_frac=0.0, max_steps=10 ** 9, seed=args.seed,
                     threads=args.threads, pit_frac=args.pit_frac, end_on_sd=args.no_sudden_death,
                     match_format=(int(g_), float(sec_)))
    O = env.reset_all().reshape(B, OBS2_SIZE).copy()
    goals = Goals(B, rng)
    # Jump-height curriculum: a running success rate per height band; bands are drawn in proportion to
    # s(1 - s) + 0.02 — mostly where success is neither ~0 (no signal) nor ~1 (nothing left to learn); the
    # highest bands take over as the lower ones are mastered. Starts uniform.
    air_s = np.full(len(AIR_BUCKETS), 0.5)
    if args.air_curriculum:
        goals.air_p = np.full(len(AIR_BUCKETS), 1.0 / len(AIR_BUCKETS))
    goals.sample(np.arange(B), O)

    # Opponent setting per match, redrawn at every match reset:
    #   solo (--solo-frac): the other rider is locked in the game's ghost state (terrain collisions only:
    #     no contact with the worker or the ball, no kills) and idle — clean movement practice;
    #   flat (--opps, or --flat-opp-frac of --flat-opp): the other rider is a flat game policy (e.g. farm2,
    #     or the killer2 specialist) playing its normal game — a realistic opponent that attacks, blocks, kills;
    #   duo (the rest): both riders are workers with their own targets (self-play, the original setup).
    # Only worker-controlled riders are trained on and counted.
    opps = []  # (name, frac, net, norm, obs size)
    specs = [x.split(":") for x in args.opps.split(",")] if args.opps else (
        [["flat", args.flat_opp, str(args.flat_opp_frac)]] if args.flat_opp_frac > 0 else [])
    for name, path, frac in specs:
        fnet, fnorm, fsize = net_from_ckpt(path)  # a flat policy of any architecture / action head
        opps.append((name, float(frac), fnet, fnorm, fsize))
        print(f"opponent {name}: {path} in {float(frac):.0%} of the matches", flush=True)
    opp_frac = sum(o[1] for o in opps)
    mode = np.zeros(N, int)       # 0 duo, 1 solo, 2 flat, 3 dead (solo with the opponent shown dead)
    opp_of = np.zeros(N, int)     # flat: which of `opps`
    wagent = np.zeros(N, int)     # solo/flat: which rider (0/1) the worker drives
    trainable = np.ones(B, bool)
    dead_clock = np.zeros(N, int)  # steps into the (simulated) death window, "dead" matches

    def draw_mode(ks):
        for k in ks:
            u = rng.random()
            mode[k] = (1 if u < args.solo_frac else 2 if u < args.solo_frac + opp_frac
                       else 3 if u < args.solo_frac + opp_frac + args.dead_frac else 0)
            if mode[k] == 2:  # which flat opponent, in proportion to their shares
                c = np.cumsum([o[1] for o in opps])
                opp_of[k] = int(np.searchsorted(c, u - args.solo_frac, side="right"))
            wagent[k] = rng.integers(2)
            other = 1 - wagent[k]
            env.set_ghost_lock(int(k), (1 << other) if mode[k] in (1, 3) else 0)
            trainable[2 * k:2 * k + 2] = True
            if mode[k]:
                trainable[2 * k + other] = False
            dead_clock[k] = rng.integers(45)
    draw_mode(range(N))

    # Solo matches: the observation would show the locked ghost as a dead opponent (hidden block), and a
    # worker initialised from the game policy freezes while its opponent is dead (farm1's habit: 3%
    # success). Show it instead as the alive, idle rider it physically is — its own features converted
    # to the worker's frame (the opponent block of observe2 is the other rider's own block, mirrored).
    OPP_NEG = np.array([0, 2, 4, 6, 11, 12, 13])

    def show_idle_opponent(O):
        # "dead" matches: the opponent is shown dead — hidden block, ghost flag, a respawn countdown
        # cycling 1 -> 0 every 3 s like a real death window — so the worker learns to keep working while
        # its opponent is dead (it slowed down ~16% there: dead opponents are rare in normal practice)
        dd = np.nonzero(mode == 3)[0]
        if len(dd):
            w = 2 * dd + wagent[dd]
            O[w, 17:34] = 0.0
            O[w, 17 + 9] = 1.0
            O[w, 17 + 16] = 1.0 - (dead_clock[dd] % 45) / 45.0
        solo = np.nonzero(mode == 1)[0]
        if not len(solo):
            return O
        w = 2 * solo + wagent[solo]
        o = 2 * solo + 1 - wagent[solo]
        blk = O[o, 0:17].copy()
        blk[:, OPP_NEG] *= -1
        blk[:, 9] = 0.0   # alive
        blk[:, 16] = 0.0  # no respawn countdown
        O[w, 17:34] = blk
        return O
    O = show_idle_opponent(O)

    log_path = os.path.join(run_dir, "log.csv")
    header = ["update", "steps", "sps", "success", "success_body", "success_tip", "success_both",
              "deaths_per_min", "reward"] + [f"body_{lo}_{hi}" for lo, hi in BUCKETS] + \
             [f"time_{lo}_{hi}" for lo, hi in BUCKETS] + ["pi_loss", "v_loss", "std", "success_air", "success_ground"] + \
             [f"air_h_{lo}_{hi}" for lo, hi in AIR_BUCKETS] + ["deaths_per_min_noopp"] + \
             [f"deaths_per_min_{o[0]}" for o in opps] + ["real_sps"] + [f"air_p_{lo}_{hi}" for lo, hi in AIR_BUCKETS]
    new_log = not os.path.exists(log_path)
    if not new_log:  # resumed with new columns: upgrade the header in place
        lines = open(log_path).read().splitlines()
        if lines and lines[0] != ",".join(header):
            with open(log_path, "w") as f:
                f.write("\n".join([",".join(header)] + lines[1:]) + "\n")
    log_f = open(log_path, "a", newline="")
    log = csv.writer(log_f)
    if new_log:
        log.writerow(header)

    T = args.rollout
    while steps < args.total_steps:
        buf_o = np.zeros((T, B, OBS_W), np.float32)
        buf_a = np.zeros((T, B, ACT_SIZE), np.float32)
        buf_lp, buf_v, buf_r, buf_d = (np.zeros((T, B), np.float32) for _ in range(4))
        buf_m = np.zeros((T, B), bool)
        ends = []  # (kind 0/1/2, success, d0, steps)
        deaths, live_steps, rew_sum = 0, 0, 0.0
        opp_row = np.repeat(np.where(mode == 2, opp_of + 1, 0), 2)  # 0 = no flat opponent, i + 1 = opps[i]
        d_by = np.zeros(len(opps) + 1); l_by = np.zeros(len(opps) + 1)
        t0 = time.time()
        for t in range(T):
            x = np.concatenate([O, goals.feats(O)], 1)
            flip = target_behind(goals, O, args.flip == "goal") if args.mirror else np.zeros(B, bool)
            x = mirror(x, flip)  # the policy's frame: target ahead
            norm.update(x)
            xo = torch.as_tensor(norm(x), dtype=torch.float32)
            with torch.no_grad():
                a, lp, v = net.act(xo)
            buf_o[t], buf_a[t], buf_lp[t], buf_v[t] = xo.numpy(), a.numpy(), lp.numpy(), v.numpy()
            act = np.clip(a.numpy(), -1, 1)
            act[flip] *= ACT_MIRROR  # back to the rider's own frame
            idle = ~trainable & np.repeat((mode == 1) | (mode == 3), 2)  # solo/dead: the ghosted rider stands still
            dead_clock[:] += 1
            act[idle] = 0.0
            for i, (_, _, fnet, fnorm, fsize) in enumerate(opps):  # flat opponents play their own game
                fl = np.nonzero(~trainable & np.repeat((mode == 2) & (opp_of == i), 2))[0]
                if len(fl):
                    with torch.no_grad():
                        fx = torch.as_tensor(fnorm(O[fl][:, :fsize]), dtype=torch.float32)
                        act[fl] = np.clip(fnet.sample_env(fx).numpy(), -1, 1)
            buf_m[t] = trainable
            out = env.step(act.reshape(N, 2, 3).astype(np.float32))
            O2 = out["obs"].reshape(B, OBS2_SIZE).copy()
            reset = np.repeat(out["reset"], 2)
            if out["reset"].any():  # new matches: new opponent settings before their first observation
                draw_mode(np.nonzero(out["reset"])[0])
            O2 = show_idle_opponent(O2)
            ghost0, ghost1 = O[:, 9] > 0.5, O2[:, 9] > 0.5
            teleport = np.abs(O2[:, 0] - O[:, 0]) > 3.0
            d = goals.dist(O2)
            valid = ~reset & ~teleport & ~ghost0 & ~ghost1
            r = np.where(valid, 0.1 * (goals.prev - d), 0.0)
            died = ~ghost0 & ghost1 & ~reset
            r -= 0.5 * died
            deaths += (died & trainable).sum()
            live_steps += (~ghost1 & trainable).sum()
            np.add.at(d_by, opp_row[died & trainable], 1)
            np.add.at(l_by, opp_row[~ghost1 & trainable], 1)
            opp_row = np.repeat(np.where(mode == 2, opp_of + 1, 0), 2)  # after resets: the new matches' settings
            ok = goals.reached(O2) & valid
            r += ok * 1.0
            bok, tok = goals.parts_ok(O2)  # "both": partial credit per correctly satisfied part
            r += PART_BONUS * (goals.body_on & goals.tip_on & valid) * (bok.astype(np.float32) + tok)
            goals.prev = d
            goals.left -= 1
            goals.age += 1
            timeout = (goals.left <= 0) & ~ok
            kind = np.where(goals.body_on & goals.tip_on, 2, np.where(goals.tip_on, 1, 0))
            for i in np.nonzero((ok | timeout) & trainable)[0]:
                ends.append((kind[i], bool(ok[i]), goals.d0[i], goals.age[i], goals.air[i],
                             goals.by[i] - _surface()(goals.bx[i])))
            buf_r[t], buf_d[t] = r, reset.astype(np.float32)
            rew_sum += r[trainable].sum()
            O = O2
            resample = np.nonzero(ok | timeout | reset)[0]
            goals.sample(resample, O)
        sps = T * N / (time.time() - t0)
        steps += T * N

        with torch.no_grad():
            xl = np.concatenate([O, goals.feats(O)], 1)
            xl = mirror(xl, target_behind(goals, O, args.flip == "goal")) if args.mirror else xl
            last_v = net.v(torch.as_tensor(norm(xl), dtype=torch.float32)).squeeze(-1).numpy()
        adv = np.zeros((T, B), np.float32)
        gae = np.zeros(B, np.float32)
        for t in reversed(range(T)):
            nv = last_v if t == T - 1 else buf_v[t + 1]
            nonterm = 1.0 - buf_d[t]
            delta = buf_r[t] + args.gamma * nv * nonterm - buf_v[t]
            gae = delta + args.gamma * args.lam * nonterm * gae
            adv[t] = gae
        ret = adv + buf_v
        keep = torch.as_tensor(buf_m.reshape(-1))  # worker-controlled riders only
        fo, fa = torch.as_tensor(buf_o.reshape(-1, OBS_W))[keep], torch.as_tensor(buf_a.reshape(-1, ACT_SIZE))[keep]
        flp, fadv, fret = (torch.as_tensor(z.reshape(-1))[keep] for z in (buf_lp, adv, ret))
        n = fo.shape[0]
        fo, fa, flp, fadv, fret = (z.to(dev) for z in (fo, fa, flp, fadv, fret))
        for _ in range(args.epochs):
            perm = torch.randperm(n).to(dev)
            for s in range(0, n, args.minibatch):
                idx = perm[s:s + args.minibatch]
                dist = lnet.dist(fo[idx])
                lp = dist.log_prob(fa[idx]).sum(-1)
                ratio = (lp - flp[idx]).clamp(-20.0, 20.0).exp()
                ad = fadv[idx]
                ad = (ad - ad.mean()) / (ad.std() + 1e-8)
                pi_loss = -torch.min(ratio * ad, ratio.clamp(1 - args.clip, 1 + args.clip) * ad).mean()
                v_loss = 0.5 * (lnet.v(fo[idx]).squeeze(-1) - fret[idx]).pow(2).mean()
                ent = dist.entropy().sum(-1).mean()
                loss = pi_loss + 0.5 * v_loss - args.ent_coef * ent
                if not torch.isfinite(loss):
                    continue
                opt.zero_grad()
                loss.backward()
                nn.utils.clip_grad_norm_(lnet.parameters(), 0.5)
                opt.step()
                if args.min_std > 0:
                    lnet.log_std.data.clamp_(min=math.log(args.min_std))
        with torch.no_grad():  # the acting copy takes the new weights
            for pa, pl in zip(net.parameters(), lnet.parameters()):
                pa.copy_(pl.detach().cpu())
        update += 1

        E = np.array(ends, float) if ends else np.zeros((0, 6))
        rate = lambda m: float(E[m, 1].mean()) if m.any() else float("nan")
        succ = rate(np.ones(len(E), bool)) if len(E) else float("nan")
        # target types: body = body-only on the GROUND, air = body-only in the air (a jump point), tip, both
        by_kind = [rate((E[:, 0] == 0) & (E[:, 4] == 0)), rate(E[:, 0] == 1), rate(E[:, 0] == 2)]
        body = E[(E[:, 0] == 0) & (E[:, 4] == 0)]  # the by-distance buckets are ground body targets
        airE = E[(E[:, 0] == 0) & (E[:, 4] > 0)]
        air_h = [float(airE[(airE[:, 5] >= lo) & (airE[:, 5] < hi), 1].mean())
                 if ((airE[:, 5] >= lo) & (airE[:, 5] < hi)).any() else float("nan") for lo, hi in AIR_BUCKETS]
        bsucc, btime = [], []
        for lo, hi in BUCKETS:
            m = (body[:, 2] >= lo) & (body[:, 2] < hi)
            bsucc.append(float(body[m, 1].mean()) if m.any() else float("nan"))
            won = body[m & (body[:, 1] > 0)]
            btime.append(float(np.median(won[:, 3]) * 4 / 60) if len(won) else float("nan"))
        dpm = deaths / max(1, live_steps * 4 / 60 / 60)
        for i, (lo, hi) in enumerate(AIR_BUCKETS):  # jump-height curriculum update
            m = (airE[:, 5] >= lo) & (airE[:, 5] < hi)
            if m.sum() >= 5:
                air_s[i] = 0.95 * air_s[i] + 0.05 * airE[m, 1].mean()
        if args.air_curriculum:
            w = air_s * (1 - air_s) + 0.02
            goals.air_p = w / w.sum()
        air_rate = rate((E[:, 0] == 0) & (E[:, 4] > 0))    # body-only targets in the air (jump points)
        ground_rate = by_kind[0]
        std = net.log_std.exp().mean().item()
        dpm_by = d_by / np.maximum(1, l_by * 4 / 60 / 60)
        real_sps = T * N / (time.time() - t0)  # the whole iteration, PPO included
        log.writerow([update, steps, round(sps), succ, *by_kind, dpm, rew_sum / max(1, buf_m.sum()), *bsucc, *btime,
                      pi_loss.item(), v_loss.item(), std, air_rate, ground_rate, *air_h, *dpm_by, round(real_sps),
                      *(goals.air_p if goals.air_p is not None else [float("nan")] * len(AIR_BUCKETS))])
        log_f.flush()
        fmt = lambda v: "  -" if v != v else f"{100 * v:3.0f}"
        print(f"upd {update:5d} steps {steps:>11d} sps {sps:6.0f} | success {fmt(succ)}% "
              f"(body {fmt(by_kind[0])} air {fmt(air_rate)} tip {fmt(by_kind[1])} both {fmt(by_kind[2])}) | body by dx "
              + " ".join(f"{lo}..{hi}:{fmt(s)}" for (lo, hi), s in zip(BUCKETS, bsucc))
              + " | air by height " + " ".join(f"{lo:g}-{hi:g}m:{fmt(s)}" for (lo, hi), s in zip(AIR_BUCKETS, air_h))
              + f" | deaths/min {dpm:.2f} (" + " ".join(f"{n} {v:.2f}" for n, v in
                                                     zip(["no opp"] + [o[0] for o in opps], dpm_by))
              + f") std {std:.2f} | real sps {real_sps:.0f}"
              + (" | jump heights drawn " + " ".join(f"{100 * v:.0f}%" for v in goals.air_p) if goals.air_p is not None else ""),
              flush=True)
        if update % 25 == 0:
            torch.save({"net": net.state_dict(), "opt": to_cpu(opt.state_dict()), "norm": norm.state(),
                        "steps": steps, "update": update, "args": vars(args)}, ckpt_path + ".tmp")
            os.replace(ckpt_path + ".tmp", ckpt_path)
            json.dump({"run": args.run, "mode": "worker", "update": update, "steps": steps, "sps": sps,
                       "mirror": args.mirror, "flip": args.flip, "t": time.time()},
                      open(os.path.join(run_dir, "status.json"), "w"))


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--run", default="worker1")
    p.add_argument("--init", default=None, help="game-policy checkpoint to start from (goal inputs widened)")
    p.add_argument("--init-std", type=float, default=0.3)
    p.add_argument("--min-std", type=float, default=0.1)
    p.add_argument("--reset-std", type=float, default=0.0, help="on resume: set the action std to this value")
    p.add_argument("--hosts", type=int, default=8)
    p.add_argument("--matches", type=int, default=16)
    p.add_argument("--threads", type=int, default=6)
    p.add_argument("--torch-threads", type=int, default=3)
    p.add_argument("--pit-frac", type=float, default=0.2)
    p.add_argument("--solo-frac", type=float, default=0.5,
                   help="matches where the other rider is a locked, idle ghost (no contact, no kills)")
    p.add_argument("--flat-opp-frac", type=float, default=0.3,
                   help="matches where the other rider is a flat game policy playing normally (--flat-opp)")
    p.add_argument("--flat-opp", default=os.path.join(HERE, "runs", "farm1", "ckpt.pt"))
    p.add_argument("--opps", default=None, metavar="NAME:CKPT:FRAC,...",
                   help="flat opponents and their shares of the matches (replaces --flat-opp/--flat-opp-frac), "
                        "e.g. farm2:runs/farm2/ckpt.pt:0.33,killer2:runs/killer2/ckpt.pt:0.33; the rest is duo")
    p.add_argument("--format", default="7,300", metavar="GOALS,SECONDS", help="match format (goals to win, s)")
    p.add_argument("--no-sudden-death", action="store_true", help="a match tied at the end of time ends as a draw")
    p.add_argument("--ppo-device", default="mps" if torch.backends.mps.is_available() else "cpu")
    p.add_argument("--air-curriculum", action="store_true",
                   help="draw jump-target heights where success is neither ~0 nor ~1 (per height band, adaptive)")
    p.add_argument("--dead-frac", type=float, default=0.1,
                   help="matches where the (idle, ghosted) opponent is shown DEAD with a running respawn countdown")
    p.add_argument("--flip", choices=["goal", "dynamic"], default="goal",
                   help="mirror frame decided once per target (goal) or re-decided every step (dynamic, old)")
    p.add_argument("--no-mirror", dest="mirror", action="store_false",
                   help="don't mirror rows whose target is behind (the worker then has to learn backward moves)")
    p.add_argument("--rollout", type=int, default=64)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--lam", type=float, default=0.95)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=4)
    p.add_argument("--minibatch", type=int, default=4096)
    p.add_argument("--ent-coef", type=float, default=0.003)
    p.add_argument("--total-steps", type=float, default=2e9)
    p.add_argument("--seed", type=int, default=7)
    train(p.parse_args())


if __name__ == "__main__":
    main()
