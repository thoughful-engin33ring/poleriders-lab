#!/usr/bin/env python3
"""Tests of the native env's ball-touch bookkeeping (NativeHost out["touch"] / out["touch_final"]).

    .venv/bin/python bridge/test_touch.py                  # SPRSIM_LIB=... to test another build

Real play at frame skip 1 (a frozen gauntlet policy acting every tick if present, else random sticks) with
snapshots recorded; checks against
the game's own bookkeeping in the snapshots:
  - a kick (TOUCH_KICK_AGE < one step) <=> the rider's `kicks` stat went up in that step without a kill
    (the game's AddKick: ball kicks, and kills, which also count as kicks);
  - a kick is always a body or handle-segment touch in the same step (BallPlayerCollision needs a solid
    contact of a body tagged as the rider: the hull/foot or a pole handle segment);
  - ages grow by exactly one step unless touched / kicked, never-touched = TOUCH_NEVER;
  - TOUCH_GAME_LAST is the game's PlayState::lastTouch: at every goal it names the rider the game credited
    (goals / own_goals stats);
  - touch == touch_final except on reset steps, where touch is a fresh match (never touched).
Equivalence of everything else (the sim is only read) is checked by sim/tools/compare_libs.py.
"""
import json, os, sys, unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE)
from native_env import (NativeHost, TOUCH_AGE, TOUCH_BODY, TOUCH_GAME_LAST, TOUCH_HANDLE,  # noqa: E402
                        TOUCH_KICK_AGE, TOUCH_NEVER, TOUCH_NOW, TOUCH_POLE, TOUCH_WHAT)

FS = 1           # one tick per step: per-tick checks are exact (the policy just acts every tick)
STEP = FS / 60


class TouchTest(unittest.TestCase):
    def test_real_play(self):
        ckpt = os.path.join(HERE, "..", "runs", "gauntlet", "ckpt", "farm2.pt")
        N, T = 8, int(os.environ.get("TOUCH_STEPS", 6000))
        env = NativeHost(1, N, FS, sd_frac=0.0, max_steps=4500, seed=21, threads=4, end_on_sd=True,
                         match_format=(7, 300), goal_refund=False)
        self.assertIn("touch", env.out, "this libsprsim has no touch buffers")
        env.record(True)
        obs = env.reset_all().copy()
        prev = [json.loads(env.json(k)) for k in range(N)]
        tprev = env.out["touch"].copy()
        np.testing.assert_array_equal(tprev[:, :, [TOUCH_AGE, TOUCH_KICK_AGE]], TOUCH_NEVER)
        if os.path.exists(ckpt):
            sys.path.insert(0, os.path.join(HERE, "..", "sim", "tools"))
            from test_env_equiv import PolicyActions
            act = PolicyActions(ckpt, 5)
        else:
            rng = np.random.default_rng(0)
            act = lambda o: rng.uniform(-1, 1, (N, 2, 3)).astype(np.float32)  # noqa: E731
        n = {"kicks": 0, "touches_body": 0, "touches_pole": 0, "touches_handle": 0, "goals": 0, "kills": 0, "resets": 0}
        for t in range(T):
            out = env.step(act(obs))
            tf, tn = out["touch_final"].copy(), out["touch"].copy()
            for k in range(N):
                cur = json.loads(env.json(k, 1))
                for i in range(2):
                    a, b = prev[k]["players"][i], cur["players"][i]
                    kicked = tf[k, i, TOUCH_KICK_AGE] < STEP - 1e-6
                    dk = b["stats"]["kicks"] - a["stats"]["kicks"]
                    dkill = b["stats"]["kills"] - a["stats"]["kills"]
                    self.assertEqual(kicked, dk > dkill, f"step {t} match {k} rider {i}: kick {kicked} stats {dk}/{dkill}")
                    n["kills"] += dkill
                    if kicked:
                        n["kicks"] += 1
                        self.assertLess(tf[k, i, TOUCH_AGE], STEP - 1e-6)
                        self.assertTrue(int(tf[k, i, TOUCH_WHAT]) & (TOUCH_BODY | TOUCH_HANDLE))
                    for f in (TOUCH_AGE, TOUCH_KICK_AGE):  # ages: + one step, or reset within this step
                        old, new = tprev[k, i, f], tf[k, i, f]
                        if new >= STEP - 1e-6:
                            self.assertAlmostEqual(float(new), min(float(old) + STEP, TOUCH_NEVER) if old < TOUCH_NEVER
                                                   else TOUCH_NEVER, 4)
                    if tf[k, i, TOUCH_AGE] < STEP - 1e-6 and tprev[k, i, TOUCH_AGE] >= STEP - 1e-6:
                        w = int(tf[k, i, TOUCH_WHAT])
                        n["touches_body"] += bool(w & TOUCH_BODY); n["touches_pole"] += bool(w & TOUCH_POLE)
                        n["touches_handle"] += bool(w & TOUCH_HANDLE)
                    if tf[k, i, TOUCH_NOW]:
                        self.assertEqual(float(tf[k, i, TOUCH_AGE]), 0.0)
                # goal credit: the game's lastTouch at the goal = the credited rider
                gs = [cur["players"][i]["stats"]["goals"] + cur["players"][i]["stats"]["own_goals"]
                      - prev[k]["players"][i]["stats"]["goals"] - prev[k]["players"][i]["stats"]["own_goals"] for i in range(2)]
                if any(gs):
                    n["goals"] += 1
                    self.assertEqual([bool(g) for g in gs], [bool(x) for x in tf[k, :, TOUCH_GAME_LAST]])
                if out["reset"][k]:
                    n["resets"] += 1
                    np.testing.assert_array_equal(tn[k, :, [TOUCH_AGE, TOUCH_KICK_AGE]], TOUCH_NEVER)
                    prev[k] = json.loads(env.json(k, 2))
                else:
                    np.testing.assert_array_equal(tn[k], tf[k])
                    prev[k] = cur
            tprev = tn
            obs = out["obs"].copy()
        env.close()
        print(f"\n  {N} x {T} steps: {n}")
        self.assertGreater(n["kicks"], 0)
        self.assertGreater(n["touches_pole"], 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)
