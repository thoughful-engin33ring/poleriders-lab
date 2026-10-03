#!/usr/bin/env python3
"""Tests of the goal-refund reward rule (native GoalRefundState, Python mirror spr_env.GoalRefund).

    .venv/bin/python bridge/test_goal_refund.py            # SPRSIM_LIB=... to test another build

Scripted scenarios in the native env: fixed actions (both riders run forward, poles waving, so the position
potential and the rotation bonus are non-zero) and the test-only hook NativeHost.debug_event (a kill = the
game's kill bookkeeping; a goal = the ball put against a goal wall, moving into it, scored by the game itself).
An env with the rule (ON) and one without (OFF) are stepped in lockstep with the same actions and events (the
reward never feeds back into the physics), so their observations/terms are identical and only rew differs.
Every transition is also replayed through compute_reward + spr_env.GoalRefund on the recorded snapshots.
The last test plays real matches (a frozen gauntlet policy, if present) and checks every step where ON and OFF
rewards differ is a goal -> respawn window step or a goal step with a refund.
"""
import json, math, os, sys, unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from native_env import NativeHost  # noqa: E402
from spr_env import GoalRefund, compute_reward  # noqa: E402

W = {"goal": 1.0, "kill": 0.3, "ball_progress": 0.003, "position": 0.1, "rotation": 0.01}
FS = 4


def actions(t):
    """Both riders run forward (own frame) with the pole aim rotating: shaping and rotation terms move."""
    a = np.zeros((1, 2, 3), np.float32)
    for g in range(2):
        ang = 0.4 * t + 1.5 * g
        a[0, g] = (1.0, math.cos(ang), math.sin(ang))
    return a


class Pair:
    """ON and OFF envs (1 match each) in lockstep + the Python mirror on the ON env's snapshots."""

    def __init__(self, reward=None, seed=5, match_format=(7, 300)):
        w = dict(W, **(reward or {}))
        kw = dict(sd_frac=0.0, max_steps=10 ** 6, seed=seed, threads=1, end_on_sd=True, match_format=match_format,
                  reward=w)
        self.on, self.off = NativeHost(1, 1, FS, goal_refund=True, **kw), NativeHost(1, 1, FS, goal_refund=False, **kw)
        assert self.on.goal_refund, "this libsprsim has no goal refund"
        self.on.record(True)
        self.on.reset_all(); self.off.reset_all()
        self.w = dict(w)
        if self.w.get("death") is None:
            self.w.pop("death", None)
        self.prev, self.gr, self.t = json.loads(self.on.json(0)), GoalRefund(), 0
        self.rows = []  # per step: dict(on, off, py, terms, state)

    def event(self, kind, who):
        self.on.debug_event(0, kind, who); self.off.debug_event(0, kind, who)

    def step(self, n=1):
        for _ in range(n):
            a = actions(self.t)
            o1 = self.on.step(a); r_on, terms, obs_on, reset = o1["rew"][0].copy(), o1["terms"][0].copy(), o1["obs"].copy(), bool(o1["reset"][0])
            o2 = self.off.step(a)
            assert np.array_equal(obs_on, o2["obs"]) and np.array_equal(terms, o2["terms"][0]), "ON/OFF diverged"
            cur = json.loads(self.on.json(0, 1))
            rew, tm = compute_reward(self.prev, cur, self.w, FS)
            plain = np.array([rew[1], rew[2]], np.float32)
            self.gr.apply(self.prev, cur, self.w, FS, tm, rew)
            self.rows.append({"on": r_on, "off": o2["rew"][0].copy(), "py": np.array([rew[1], rew[2]], np.float32),
                              "py_off": plain, "terms": terms, "state": cur, "reset": reset})
            if cur.get("players"):
                self.prev = cur
            if reset:
                self.prev, self.gr = json.loads(self.on.json(0, 2)), GoalRefund()
            self.t += 1
        return self.rows[-1]

    def until_goal(self, limit=10):
        for _ in range(limit):
            r = self.step()
            if r["terms"][0, 0] != 0:
                return len(self.rows) - 1
        raise AssertionError("no goal")

    def close(self):
        self.on.close(); self.off.close()


def f32(x):
    return np.float32(x)


class GoalRefundTest(unittest.TestCase):
    def check_mirror(self, p):
        """ON == compute_reward + GoalRefund, OFF == plain compute_reward (the old reward), every step."""
        for i, r in enumerate(p.rows):
            np.testing.assert_array_equal(r["on"], r["py"], err_msg=f"native vs Python mirror at step {i}")
            np.testing.assert_array_equal(r["off"], r["py_off"], err_msg=f"OFF vs plain compute_reward at step {i}")

    def test_mask_between_goal_and_respawn(self):
        p = Pair()
        p.step(12)
        g = (p.event("goal", 0), p.until_goal())[1]
        r = p.rows[g]
        np.testing.assert_array_equal(r["on"], [1.0, -1.0])        # the goal only
        p.step(3)
        p.event("kill", 1)                                          # a kill inside the window: not paid
        k = len(p.rows)
        p.step(60)
        ball = [s["state"]["ball"]["x"] for s in p.rows]
        reset = next(i for i in range(g + 1, len(p.rows)) if abs(ball[i] - ball[i - 1]) > 6.0)
        # every step from the goal to the reset step: no kill / death / shaping / rotation reward
        for i in range(g + 1, reset + 1):
            np.testing.assert_array_equal(p.rows[i]["on"], [0.0, 0.0], err_msg=f"window step {i}")
        self.assertGreater(reset - g, 15)                           # ~1.4 s + the goal's hit-stop
        self.assertTrue(p.rows[k]["terms"][0, 1] > 0)               # the kill happened (raw term) ...
        self.assertNotEqual(float(p.rows[k]["off"][0]), 0.0)        # ... and is paid without the rule
        self.assertTrue(any(np.any(p.rows[i]["off"] != 0) for i in range(g + 1, k)))  # shaping/rotation moved
        # both riders respawned at the reset step (the game rebuilds the ball and respawns in one tick)
        st = p.rows[reset]["state"]
        self.assertTrue(all(pl["ghost"] and not pl["dead"] for pl in st["players"]), st["players"])
        # after the respawn the two rewards agree again: no potential jump carried over
        for i in range(reset + 1, len(p.rows)):
            np.testing.assert_array_equal(p.rows[i]["on"], p.rows[i]["off"], err_msg=f"after window, step {i}")
        self.check_mirror(p)
        p.close()

    def test_match_deciding_goal(self):
        """The winning goal has no ball reset: the match plays on (~1 s clock, slow motion, fade) until game_over,
        and nothing but that goal is paid in between; the next match starts normally."""
        p = Pair(match_format=(1, 300))
        p.step(12)
        p.event("goal", 1)
        g = p.until_goal()
        np.testing.assert_array_equal(p.rows[g]["on"], [-1.0, 1.0])
        while not p.rows[-1]["reset"]:
            p.step()
        end = len(p.rows) - 1
        self.assertGreater(end - g, 60)                                   # > 4 s of end phase
        for i in range(g + 1, end + 1):
            np.testing.assert_array_equal(p.rows[i]["on"], [0.0, 0.0], err_msg=f"end phase step {i}")
        self.assertTrue(any(np.any(p.rows[i]["off"] != 0) for i in range(g + 1, end + 1)))
        p.step(30)                                                        # the new match: back to normal
        for i in range(end + 1, len(p.rows)):
            np.testing.assert_array_equal(p.rows[i]["on"], p.rows[i]["off"])
        self.check_mirror(p)
        p.close()

    def test_refund_while_dead(self):
        p = Pair()
        p.step(12)
        p.event("kill", 0)                                          # player 1 killed by player 2
        r = p.step()
        np.testing.assert_array_equal(r["on"], r["off"])            # the death is paid as usual ...
        self.assertAlmostEqual(float(r["terms"][0, 1]), -1.0)
        p.step(8)                                                   # 0.6 s later, still dead (3 s timer)
        self.assertTrue(p.rows[-1]["state"]["players"][0]["dead"])
        p.event("goal", 0)                                          # ... and player 1's team scores
        g = p.until_goal()
        np.testing.assert_array_equal(p.rows[g]["on"], [f32(1.0 + 0.3), f32(-1.0 - 0.3)])  # full refund / take-back
        self.check_mirror(p)
        p.close()

    def test_kill_take_back_symmetric_and_death_weight(self):
        p = Pair(reward={"kill": 2.0, "death": 1.0})                 # killer specialist weights
        p.step(12)
        p.event("kill", 1)                                          # player 2 killed by player 1
        r = p.step()
        self.assertAlmostEqual(float(r["on"][0] - r["off"][0]), 0.0)
        p.step(20)
        p.event("goal", 1)                                          # player 2's team scores while he is dead
        g = p.until_goal()
        # victim (player 2): goal + death refund 1.0; killer (player 1): -goal - kill take-back 2.0
        np.testing.assert_array_equal(p.rows[g]["on"], [f32(-1.0 - 2.0), f32(1.0 + 1.0)])
        self.check_mirror(p)
        p.close()

    def test_no_refund_after_respawn(self):
        p = Pair()
        p.step(12)
        p.event("kill", 0)
        p.step(55)                                                  # 3.7 s: respawned
        self.assertFalse(p.rows[-1]["state"]["players"][0]["dead"])
        p.event("goal", 0)
        g = p.until_goal()
        np.testing.assert_array_equal(p.rows[g]["on"], [1.0, -1.0])
        self.check_mirror(p)
        p.close()

    def test_no_refund_when_the_other_team_scores(self):
        p = Pair()
        p.step(12)
        p.event("kill", 0)
        p.step(5)
        p.event("goal", 1)                                          # the killer's team scores: death stands
        g = p.until_goal()
        np.testing.assert_array_equal(p.rows[g]["on"], [-1.0, 1.0])
        self.check_mirror(p)
        p.close()

    def test_real_play_differences_are_window_or_refund(self):
        """Policy play (frozen farm2 from the gauntlet if present, else random sticks): ON vs OFF rewards differ
        only at steps from a goal to the post-goal reset, and goal steps differ only by the refunds."""
        ckpt = os.path.join(HERE, "..", "runs", "gauntlet", "ckpt", "farm2.pt")
        N, T = 16, int(os.environ.get("GR_STEPS", 2500))
        kw = dict(sd_frac=0.0, max_steps=4500, seed=11, threads=4, end_on_sd=True, match_format=(7, 300))
        on, off = NativeHost(1, N, FS, goal_refund=True, **kw), NativeHost(1, N, FS, goal_refund=False, **kw)
        obs = on.reset_all().copy(); off.reset_all()
        if os.path.exists(ckpt):
            sys.path.insert(0, os.path.join(HERE, "..", "sim", "tools"))
            from test_env_equiv import PolicyActions
            act = PolicyActions(ckpt, 3)
        else:
            rng = np.random.default_rng(0)
            act = lambda o: rng.uniform(-1, 1, (N, 2, 3)).astype(np.float32)  # noqa: E731
        window = np.zeros(N, bool)
        out = np.zeros((N, 2), bool)  # rider out of play (dead / ghost) after the previous step
        stats = {"goal_steps": 0, "refund_goals": 0, "window_steps": 0, "diff_steps": 0, "diff_outside": 0,
                 "masked_kill_terms": 0, "kill_terms": 0}
        ballx = obs[:, 0, 52].copy()
        for t in range(T):
            a = act(obs)
            o1 = on.step(a); o2 = off.step(a)
            np.testing.assert_array_equal(o1["obs"], o2["obs"])
            fin = o1["final"]
            bx = fin[:, 0, 52]                                       # rider 1's frame: ball x
            goal = o1["terms"][:, 0, 0] != 0
            reset_step = ~goal & (np.abs(bx - ballx) > 6.0)
            diff = np.any(o1["rew"] != o2["rew"], 1)
            for k in range(N):
                stats["kill_terms"] += int(np.any(o1["terms"][k, :, 1] != 0))
                if goal[k]:
                    stats["goal_steps"] += 1
                    scorer = 0 if o1["terms"][k, 0, 0] > 0 else 1
                    expect = np.array([o1["terms"][k, 0, 0], o1["terms"][k, 1, 0]], np.float32)
                    if not np.array_equal(o1["rew"][k], expect):   # goal step: the goal only, or + a refund
                        stats["refund_goals"] += 1
                        self.assertTrue(out[k, scorer] and fin[k, scorer, 9] > 0.5, "refund for a rider in play")
                        d = o1["rew"][k] - expect
                        self.assertAlmostEqual(float(d[scorer]), 0.3, 5)
                        self.assertAlmostEqual(float(d[1 - scorer]), -0.3, 5)
                    window[k] = True
                elif window[k]:
                    stats["window_steps"] += 1
                    stats["masked_kill_terms"] += int(np.any(o1["terms"][k, :, 1] != 0))
                    np.testing.assert_array_equal(o1["rew"][k], [0.0, 0.0])
                    if reset_step[k] or o1["reset"][k]:
                        window[k] = False
                elif diff[k]:
                    stats["diff_outside"] += 1
                stats["diff_steps"] += int(diff[k])
                if o1["reset"][k]:
                    window[k] = False
            out = fin[:, :, 9] > 0.5
            obs = o1["obs"].copy()
            ballx = obs[:, 0, 52].copy()
            out[o1["reset"]] = False
        on.close(); off.close()
        print(f"\n  real play {N} x {T} steps: {stats}")
        self.assertEqual(stats["diff_outside"], 0)
        self.assertGreater(stats["goal_steps"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
