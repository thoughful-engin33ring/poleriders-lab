#!/usr/bin/env python3
"""Headless farm environment: one game process ("host") runs K shadow matches of the real game.

    host = FarmHost(sock, n_matches=16, arena=2)
    host.start()                                  # menus -> real match -> lockstep -> K shadows
    obs = host.reset_all()                        # [{1: vec, 2: vec}, ...] one per match
    obs, rew, done, info = host.step([(a1, a2), ...])   # a = (run, aim_x, aim_y) per agent

Shadows are the game's own PlayState objects ticked directly (bit-identical to rendered matches,
see test_farm_equivalence.py); one socket round trip steps every match of a host.

Observation v2 (OBS2_SIZE floats, agent-centric: mirrored so each agent attacks +x, y up):
    me[0:17]   x y vx vy sin(a) cos(a) w on_pole on_ground ghost pole_sin pole_cos pole_w
               tip_dx tip_dy hammer respawn_in
               ghost = 1 while the rider is out of play: dead (killed / fell into a deleter, waiting
               to respawn: snapshot "dead", PlayState::respawnTimer > 0) or in the short post-spawn
               ghost state; respawn_in = seconds until it is back in play / 3 (respawn_t while dead,
               else ghost_t)
    opp[17:34] same for the opponent; while it is out of play (dead or ghost) its corpse is hidden:
               all zeros except ghost = 1 and respawn_in
    floor[34:43]    floor height relative to my body at -6 -4 -2 -1 0 1 2 4 6 m ahead
    overhead[43:52] height of the nearest overhead ledge at the same offsets (10 = open sky)
    ball[52:56]     x y vx vy
    match[56:60]    time_left/180, my score, opp score, sudden_death
    ball_rel[60:62] ball x and y relative to me (+x = the ball is ahead of me toward the goal I
                    attack): "which side of the ball am I on" as a single input
"""
import json, math, os, random, sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spr import Bridge  # noqa: E402
from spr_env import (DEFAULT_REWARD, PoleRidersEnv, _body_feats, _player_feats,  # noqa: E402
                     GoalRefund, absent, compute_reward, respawn_in)

HERE = os.path.dirname(os.path.abspath(__file__))
TERRAIN_OFFSETS = (-6, -4, -2, -1, 0, 1, 2, 4, 6)
OBS2_SIZE = 17 + 17 + 2 * len(TERRAIN_OFFSETS) + 4 + 4 + 2


class Terrain:
    """Top surfaces of an arena's static ground polygons, sampled every 5 cm (Box2D, y down).

    Arenas have layers (Flatirons: a raised middle, pits in front of the goals, thin ledges high
    above the pits), so each x keeps every surface: `floor` = nearest surface at or under a point,
    `above` = nearest surface overhead."""

    NONE = 50.0

    def __init__(self, arena, x0=-45.0, x1=45.0, dx=0.05):
        path = os.path.join(HERE, "..", "arenas", f"arena{arena}.json")
        if not os.path.exists(path):
            raise FileNotFoundError(f"{path} missing: run `python3 bridge/dump_level.py {arena}`")
        geo = json.load(open(path))
        self.x0, self.dx = x0, dx
        n = int((x1 - x0) / dx) + 1
        self.surf = [[] for _ in range(n)]  # top y of every ground polygon over each x
        for body in geo["bodies"]:
            if body["tag"] != "ground":
                continue
            for poly in body["polys"]:
                if poly["sensor"]:
                    continue
                v = poly["v"]
                for i in range(n):
                    x = x0 + i * dx
                    ys = [ay + (by - ay) * (x - ax) / (bx - ax)
                          for (ax, ay), (bx, by) in zip(v, v[1:] + v[:1])
                          if ax != bx and min(ax, bx) <= x <= max(ax, bx)]
                    if ys:
                        self.surf[i].append(min(ys))
        for lst in self.surf:
            lst.sort()

    def _col(self, x):
        return self.surf[max(0, min(len(self.surf) - 1, int(round((x - self.x0) / self.dx))))]

    def floor(self, x, y):
        """Nearest surface at or below height y (y down); NONE if there is none."""
        below = [t for t in self._col(x) if t >= y - 0.2]
        return min(below) if below else self.NONE

    def above(self, x, y):
        """Nearest surface overhead (more than 1 m above y); None if there is none."""
        over = [t for t in self._col(x) if t < y - 1.0]
        return max(over) if over else None


def observe2(state, agent, terrain):
    sign = 1 if agent == 1 else -1
    players = {p["id"]: p for p in state["players"]}
    me, opp = players[agent], players[3 - agent]
    ball = state.get("ball") or {"x": 0, "y": 0, "a": 0, "vx": 0, "vy": 0, "w": 0}
    # around me, in my own frame: floor height relative to my body (negative = below me) and the
    # height of the nearest overhead ledge (+10 = open sky)
    ter = []
    for d in TERRAIN_OFFSETS:
        x = me["x"] + sign * d
        ter.append(max(-10.0, min(10.0, me["y"] - terrain.floor(x, me["y"]))))
    for d in TERRAIN_OFFSETS:
        x = me["x"] + sign * d
        a = terrain.above(x, me["y"])
        ter.append(10.0 if a is None else max(0.0, min(10.0, me["y"] - a)))
    sc = state["score"]
    extra = lambda p: [float(p.get("hammer", False)), respawn_in(p)]
    if absent(opp):
        # A dead opponent's corpse stays where it fell but no longer plays. Hide it: the agent
        # sees "absent" (all zeros) plus the ghost flag and how long until it respawns.
        opp_block = [0.0] * 17
        opp_block[9] = 1.0                 # ghost (out of play)
        opp_block[16] = respawn_in(opp)    # respawn_in
    else:
        opp_block = _player_feats(opp, sign) + extra(opp)
    return (_player_feats(me, sign) + extra(me)
            + opp_block
            + ter + _body_feats(ball, sign)[:4]
            + [state["time_left"] / 180.0, float(sc[agent - 1]), float(sc[2 - agent]),
               float(state.get("sudden_death", False))]
            + [sign * (ball["x"] - me["x"]), -(ball["y"] - me["y"])])


class FarmHost:
    """K shadow matches in one game process. Pad 0 = player 1 (Vaulters), pad 1 = player 2."""

    def __init__(self, sock, n_matches=16, arena=2, frame_skip=4, reward=None,
                 sudden_death_frac=0.0, max_steps=5400, seed=0, goal_refund=False):
        self.sock, self.n, self.arena, self.frame_skip = sock, n_matches, arena, frame_skip
        self.w = dict(DEFAULT_REWARD, **(reward or {}))
        self.sd_frac, self.max_steps = sudden_death_frac, max_steps
        self.rng = random.Random(seed)
        self.terrain = Terrain(arena)
        self.env = PoleRidersEnv(frame_skip=frame_skip, sock=sock)  # drives the real match/menus
        self.prev = [None] * n_matches
        self.steps = [0] * n_matches
        self.level = None
        self.sd_start = [False] * n_matches
        # the goal-refund reward rule (spr_env.GoalRefund), one state per match; train_farm turns it on
        self.gr = [GoalRefund() for _ in range(n_matches)] if goal_refund else None

    def start(self):
        self.env.start_match(arena=self.arena)
        self.b = self.env.b
        self.level = self.env.level
        self.b.cmd("sync on")  # freeze the real match; the game thread now just runs farm jobs
        self.b.cmd("farm destroy")
        r = json.loads(self.b.cmd(f"farm create {self.n}"))
        if r.get("n") != self.n:
            raise RuntimeError(f"farm create failed: {r}")

    def reset(self, k):
        sd = self.rng.random() < self.sd_frac
        st = json.loads(self.b.cmd(f"farm reset {k} 0.05" if sd else f"farm reset {k}"))
        if "error" in st:
            raise RuntimeError(st["error"])
        self.prev[k], self.steps[k], self.sd_start[k] = st, 0, sd
        if self.gr:
            self.gr[k].reset()
        return {1: observe2(st, 1, self.terrain), 2: observe2(st, 2, self.terrain)}

    def reset_all(self):
        return [self.reset(k) for k in range(self.n)]

    def step(self, actions):
        """actions[k] = (action_agent1, action_agent2), each (run, aim_x, aim_y) in its own frame.
        Returns per-match lists: obs, rewards {1,2}, done, info {"state", "terms", "truncated"}."""
        vals = []
        for a1, a2 in actions:
            vals += PoleRidersEnv._stick(1, a1) + PoleRidersEnv._stick(2, a2)
        reply = json.loads(self.b.cmd(f"farm act {self.frame_skip} " + " ".join(f"{v:.4f}" for v in vals)))
        if "error" in reply:
            raise RuntimeError(reply["error"])
        out_obs, out_rew, out_done, out_info = [], [], [], []
        for k, st in enumerate(reply["matches"]):
            rew, terms = compute_reward(self.prev[k], st, self.w, self.frame_skip)
            if self.gr:
                self.gr[k].apply(self.prev[k], st, self.w, self.frame_skip, terms, rew)
            self.steps[k] += 1
            wrong_map = self.level is not None and st.get("level") not in (None, self.level)
            done = bool(st.get("game_over")) or not st.get("players") or wrong_map
            trunc = not done and self.steps[k] >= self.max_steps
            if st.get("players"):
                self.prev[k] = st
            out_obs.append({1: observe2(st, 1, self.terrain), 2: observe2(st, 2, self.terrain)}
                           if st.get("players") else None)
            out_rew.append(rew)
            out_done.append(done)
            out_info.append({"state": st, "terms": terms, "truncated": trunc,
                             "sudden_death_start": self.sd_start[k]})
        return out_obs, out_rew, out_done, out_info
