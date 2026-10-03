#!/usr/bin/env python3
"""Tests of per-match snapshot / restore and the planning branch pool (native per-match arenas).

    .venv/bin/python bridge/test_snapshot.py            # SPRSIM_LIB=... to test another build
    SNAP_STEPS=3000 .venv/bin/python bridge/test_snapshot.py   # longer

Twin tests: env A and env B are the same configuration and seed, driven by the same main-line actions (a frozen
policy's stochastic play if runs/farm2/ckpt.pt exists, else sticky random sticks) and the same scripted test
events (NativeHost.debug_event kills / goals). B never snapshots. A, every few steps, snapshots every match,
wanders off for 1-40 steps with junk actions and its own scripted events (crossing goals, kills, the post-goal
reset, match ends / resets, scrambles), sometimes snapshots that junk state in a second slot, replays the junk
from it twice (must be identical both times), then restores the main-line snapshot. After every main-line
step every output array (obs, final, rew, terms, reset, trunc, sd, score, sd_start, touch, touch_final) and
every match's snapshot JSON (minus match_id: ids are handed out by a shared counter) must be bit-identical
to B's. Configurations: training rules with the goal refund, short format with sudden death and match ends,
random pole + goal spawns + rotation bonus + death weight, scramble + pit drops, ghost lock + legacy obs.
(sd_frac = 0 throughout: the host's sudden-death coin stream is shared by its matches and not part of a
match snapshot.)

Planning test: B is run first and its per-step outputs + JSON recorded; A replays the same actions with a
planning pool (3 branches per match). At random decisions A begins a search: branch 0 of each match follows the
recorded future actions and must reproduce B's outputs (obs = final, rew, terms, score, touch = touch_final,
done = reset, trunc) and JSON step for step until its match ends; the other branches take junk actions.
Sometimes the search is left without plan_end, sometimes A's root is restored from an older snapshot and
re-stepped (branches must rebuild by replay), sometimes a scripted event happens on the main line.
Also: no game object outside its arena (check_arena) and no game allocation outside an arena.
"""
import hashlib, os, re, sys, unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
sys.path.insert(0, os.path.join(HERE, "..", "sim", "tools"))
from native_env import NativeHost, load_lib  # noqa: E402
from test_env_equiv import Actions, PolicyActions  # noqa: E402

KEYS = ["obs", "final", "rew", "terms", "reset", "trunc", "sd", "score", "sd_start", "touch", "touch_final"]
POLICY = os.path.join(HERE, "..", "runs", "farm2", "ckpt.pt")
STEPS = int(os.environ.get("SNAP_STEPS", 1200))
M = 16
CONFIGS = [
    ("train", dict(end_on_sd=True, match_format=(7, 300), goal_refund=True), {}),
    ("short_sd", dict(match_format=(2, 25), goal_refund=True), {}),
    ("pole_spawn", dict(end_on_sd=True, match_format=(7, 300), random_pole=True, goal_spawn=0.5,
                        reward={"rotation": 0.005, "kill": 2.0, "death": 1.0}), {}),
    ("scramble_pit", dict(scramble=True, pit_frac=0.3, match_format=(3, 60)), {}),
    ("ghost_legacy", dict(end_on_sd=True, match_format=(7, 300)), {"legacy": True}),
]
MATCH_ID = re.compile(r'"match_id":\d+')


def make(kw, ext, seed):
    env = NativeHost(1, M, 4, sd_frac=0.0, max_steps=900, seed=seed, threads=4, **kw)
    if ext.get("legacy"):
        for k in range(0, M, 3):
            env.set_legacy_obs(k, 1 + k % 3)
            if k + 1 < M:
                env.set_ghost_lock(k + 1, 1 + k % 2)
    return env


def policy(seed):
    return PolicyActions(POLICY, seed) if os.path.exists(POLICY) else Actions(M, seed)


def js(env, k, which=0):
    return MATCH_ID.sub("", env.json(k, which))


def events(env, rng, p):
    """Random scripted kills / goals (both envs get the same list)."""
    ev = []
    for k in range(M):
        if rng.random() < p:
            kind, who = ("kill", "goal")[rng.integers(0, 2)], int(rng.integers(0, 2))
            ev.append((k, kind, who))
    for k, kind, who in ev:
        try:
            env.debug_event(k, kind, who)
        except RuntimeError:  # e.g. the victim is already dead
            pass
    return ev


def apply_events(env, ev):
    for k, kind, who in ev:
        try:
            env.debug_event(k, kind, who)
        except RuntimeError:
            pass


def hashes(out):
    return [hashlib.sha1(np.ascontiguousarray(out[k]).tobytes()).hexdigest() for k in KEYS]


class TestSnapshot(unittest.TestCase):
    def test_twin(self):
        L = load_lib()
        cnt = (__import__("ctypes").c_ulonglong * 2)()
        for name, kw, ext in CONFIGS:
            with self.subTest(config=name):
                A, B = make(kw, ext, 11), make(kw, ext, 11)
                oa, ob = A.reset_all().copy(), B.reset_all()
                np.testing.assert_array_equal(oa, ob)
                act = policy(5)
                rng = np.random.default_rng(3)
                junk = Actions(M, 99)
                L.sprenv_arena_counters(cnt)
                heap0 = cnt[1]
                stats = {"goals": 0, "kills": 0, "resets": 0, "restores": 0, "junk_steps": 0, "junk_goals": 0,
                         "junk_kills": 0, "junk_resets": 0, "replays": 0}
                next_snap = 0
                obs = oa
                for t in range(STEPS):
                    if t == next_snap:
                        for k in range(M):
                            A.snapshot(k, 0)
                        n = int(rng.integers(1, 41))
                        mid = int(rng.integers(0, n)) if rng.random() < 0.4 else -1
                        o2 = obs.copy()
                        rec = []
                        for j in range(n):
                            if j == mid:  # snapshot the junk state too, record what follows
                                for k in range(M):
                                    A.snapshot(k, 1)
                                rec = []
                                o_mid = o2.copy()
                            a2 = junk(o2)
                            evj = events(A, rng, 0.03)
                            out = A.step(a2)
                            if j >= mid >= 0:
                                rec.append((a2, evj, hashes(out)))
                            stats["junk_steps"] += 1
                            stats["junk_goals"] += int((out["terms"][:, 0, 0] != 0).sum())
                            stats["junk_kills"] += int((out["terms"][:, :, 1] > 0).sum())
                            stats["junk_resets"] += int(out["reset"].sum())
                            o2 = out["obs"].copy()
                        if mid >= 0:  # replay the recorded junk twice from slot 1: identical both times
                            for _ in range(2):
                                for k in range(M):
                                    A.restore(k, 1)
                                np.testing.assert_array_equal(A.out["obs"], o_mid)
                                for a2, evj, h in rec:
                                    apply_events(A, evj)
                                    self.assertEqual(hashes(A.step(a2)), h, f"{name}: junk replay differs at t={t}")
                                stats["replays"] += 1
                        for k in range(M):
                            A.restore(k, 0)
                        stats["restores"] += 1
                        np.testing.assert_array_equal(A.out["obs"], obs, f"{name}: obs after restore, t={t}")
                        next_snap = t + int(rng.integers(1, 9))
                    a = act(obs)
                    ev = events(A, rng, 0.004)
                    apply_events(B, ev)
                    outa = A.step(a)
                    ha = hashes(outa)
                    outb = B.step(a)
                    hb = hashes(outb)
                    if ha != hb:
                        bad = [KEYS[i] for i in range(len(KEYS)) if ha[i] != hb[i]]
                        self.fail(f"{name}: step {t} differs in {bad}")
                    for k in range(M):
                        self.assertEqual(js(A, k), js(B, k), f"{name}: json of match {k} differs at step {t}")
                    stats["goals"] += int((outa["terms"][:, 0, 0] != 0).sum())
                    stats["kills"] += int((outa["terms"][:, :, 1] > 0).sum())
                    stats["resets"] += int(outa["reset"].sum())
                    obs = outa["obs"].copy()
                    if t % 200 == 0:
                        self.assertEqual(sum(A.check_arena(k) for k in range(M)), 0)
                L.sprenv_arena_counters(cnt)
                self.assertEqual(cnt[1], heap0, "game code allocated outside an arena")
                print(f"  twin {name:13s} {STEPS} steps x {M}: {stats}", flush=True)
                A.close(), B.close()

    def test_plan(self):
        for name, kw, ext in CONFIGS:
            with self.subTest(config=name):
                self._plan(name, kw, ext)

    def _plan(self, name, kw, ext):
        T = STEPS
        B = make(kw, ext, 21)
        B.record(True)
        obs = B.reset_all().copy()
        act = policy(8)
        rng = np.random.default_rng(4)
        acts, evs, rows, jsons = [], [], [], []
        for t in range(T):
            a = act(obs)
            ev = events(B, rng, 0.004)
            out = B.step(a)
            acts.append(a.copy()); evs.append(ev)
            rows.append({"obs": out["final"].copy(), "rew": out["rew"].copy(), "terms": out["terms"].copy(),
                         "score": out["score"].copy(), "touch": out["touch_final"].copy(),
                         "done": out["reset"].copy(), "trunc": out["trunc"].copy()})
            # the branch's state after the step: the current one (incl. a scramble re-spawn), or for a match that
            # ended (and was reset) the post-step one
            jsons.append([MATCH_ID.sub("", B.json(k, 1 if out["reset"][k] else 0)) for k in range(M)])
            obs = out["obs"].copy()
        B.close()

        A = make(kw, ext, 21)
        A.reset_all()
        NB = 3
        owner = np.repeat(np.arange(M), NB)
        A.plan_init(owner)
        junk = Actions(M * NB, 77)
        stats = {"searches": 0, "branch_steps_checked": 0, "branch_ends": 0, "root_restores": 0, "no_end": 0}
        t = 0
        snap_t = None
        while t < T:
            if rng.random() < 0.15 and t + 1 < T:
                active = rng.random(M * NB) < 0.9
                P = A.plan_begin(active)
                np.testing.assert_array_equal(P["obs"][active], A.out["obs"][owner[active]])
                for b in np.flatnonzero(active):
                    self.assertEqual(MATCH_ID.sub("", A.plan_json(b)), MATCH_ID.sub("", A.json(owner[b])))
                H = int(rng.integers(1, 30))
                alive = active & (np.arange(M * NB) % NB == 0)
                for h in range(min(H, T - t)):
                    # scripted main-line events are not part of a branch's actions: stop following those matches
                    hit = {k for k, _, _ in evs[t + h]}
                    for b in np.flatnonzero(alive):
                        if owner[b] in hit:
                            alive[b] = False
                    pa = junk(P["obs"])
                    follow = np.flatnonzero(np.arange(M * NB) % NB == 0)
                    pa[follow] = acts[t + h][owner[follow]]
                    P = A.plan_step(pa)
                    r = rows[t + h]
                    for b in np.flatnonzero(alive):
                        k = owner[b]
                        for key in ("obs", "rew", "terms", "score", "touch", "done", "trunc"):
                            np.testing.assert_array_equal(P[key][b], r[key][k],
                                                          f"{name}: branch {b} {key} at t={t}+{h}")
                        self.assertEqual(MATCH_ID.sub("", A.plan_json(b)), jsons[t + h][k])
                        stats["branch_steps_checked"] += 1
                        if r["done"][k]:
                            alive[b] = False
                            stats["branch_ends"] += 1
                self.assertEqual(sum(A.check_arena(0, b) for b in range(M * NB)), 0)
                if rng.random() < 0.8:
                    A.plan_end()
                else:
                    stats["no_end"] += 1
                stats["searches"] += 1
            if snap_t is None and rng.random() < 0.02:
                for k in range(M):
                    A.snapshot(k, 2)
                snap_t = t
            if snap_t is not None and t - snap_t > 5 and rng.random() < 0.05:  # back in time; re-step
                for k in range(M):
                    A.restore(k, 2)
                t = snap_t
                snap_t = None
                stats["root_restores"] += 1
            apply_events(A, evs[t])
            out = A.step(acts[t])
            np.testing.assert_array_equal(out["final"], rows[t]["obs"], f"{name}: main line at t={t}")
            t += 1
        print(f"  plan {name:13s} {T} steps x {M} matches x {NB} branches: {stats}", flush=True)
        A.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)
