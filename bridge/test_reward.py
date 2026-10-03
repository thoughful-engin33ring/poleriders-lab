#!/usr/bin/env python3
"""Offline tests for PoleRidersEnv._reward and observe() — no game needed.

    python3 bridge/test_reward.py
"""
import copy, math, os, sys, unittest

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spr_env import OBS_SIZE, PoleRidersEnv, observe  # noqa: E402


def body(x=0.0, y=0.0):
    return {"x": x, "y": y, "a": 0.0, "vx": 0.0, "vy": 0.0, "w": 0.0}


def player(pid, x):
    p = dict(body(x, -0.7), id=pid, on_pole=True, on_ground=True, ghost=False, ghost_t=0.0,
             dead=False, respawn_t=0.0, hammer=False,
             stats={"kicks": 0, "jumps": 0, "suicides": 0, "kills": 0, "deaths": 0,
                    "goals": 0, "own_goals": 0})
    p["pole"] = [body(x + k * 0.3, -1.0) for k in range(10)]
    return p


def state(ball_x=0.0, score=(0, 0)):
    return {"in_match": True, "game_over": False, "score": list(score), "time_left": 180.0,
            "players": [player(1, -15.7), player(2, 15.7)], "ball": body(ball_x, -5.7)}


class RewardTest(unittest.TestCase):
    def setUp(self):
        self.env = PoleRidersEnv(frame_skip=4)

    def test_goal_vaulters(self):
        a, b = state(), state(score=(1, 0))
        rew, terms = self.env._reward(a, b)
        self.assertEqual(rew, {1: 1.0, 2: -1.0})
        self.assertEqual(terms[1]["ball_progress"], 0.0)

    def test_goal_benders(self):
        rew, _ = self.env._reward(state(), state(score=(0, 1)))
        self.assertEqual(rew, {1: -1.0, 2: 1.0})

    def test_kill_by_player1(self):
        a, b = state(), state()
        b["players"][1]["stats"]["deaths"] = 1
        b["players"][0]["stats"]["kills"] = 1
        rew, terms = self.env._reward(a, b)
        self.assertAlmostEqual(rew[1], 0.3)
        self.assertAlmostEqual(rew[2], -0.3)
        self.assertEqual(terms[1]["kill"], 1.0)

    def test_suicide_counts_as_death(self):
        a, b = state(), state()
        b["players"][0]["stats"]["suicides"] = 1
        rew, _ = self.env._reward(a, b)
        self.assertAlmostEqual(rew[1], -0.3)
        self.assertAlmostEqual(rew[2], 0.3)

    def test_kill_and_death_weighted_separately(self):
        # killer specialist: kills pay 2, own deaths cost 1 (not zero-sum); default = zero-sum kill term
        env = PoleRidersEnv(frame_skip=4, reward={"goal": 0.0, "kill": 2.0, "death": 1.0, "ball_progress": 0.0,
                                                  "position": 0.0})
        a, b = state(ball_x=0.0), state(ball_x=0.0)
        b["players"][1]["stats"]["deaths"] = 1   # player 1 killed player 2
        rew, terms = env._reward(a, b)
        self.assertAlmostEqual(rew[1], 2.0)
        self.assertAlmostEqual(rew[2], -1.0)
        self.assertAlmostEqual(terms[1]["kill"], 1.0)   # the stats term stays the net kill difference

    def test_ball_progress_zero_sum_and_mirrored(self):
        env = PoleRidersEnv(frame_skip=4, reward={"position": 0.0})
        rew, terms = env._reward(state(ball_x=0.0), state(ball_x=2.0))
        self.assertAlmostEqual(terms[1]["ball_progress"], 2.0)
        self.assertAlmostEqual(terms[2]["ball_progress"], -2.0)
        self.assertAlmostEqual(rew[1], 0.006)
        self.assertAlmostEqual(rew[1] + rew[2], 0.0)

    def test_position_shaping(self):
        # player 1 (attacks +x) runs from 6 m past the ball back to 4 m behind it (range 10 m)
        a, b = state(ball_x=0.0), state(ball_x=0.0)
        a["players"][0]["x"], b["players"][0]["x"] = 6.0, -4.0
        rew, terms = self.env._reward(a, b)
        self.assertAlmostEqual(terms[1]["position"], 0.0 - (-0.6))   # behind the ball = 0
        self.assertAlmostEqual(rew[1], 0.06)
        # over-running the ball costs the same amount
        rew2, terms2 = self.env._reward(b, a)
        self.assertAlmostEqual(terms2[1]["position"], -0.6)
        # approaching the ball from behind costs nothing (spawn is 6.25 m behind it)
        a["players"][0]["x"], b["players"][0]["x"] = -6.25, -3.5
        _, terms4 = self.env._reward(a, b)
        self.assertEqual(terms4[1]["position"], 0.0)
        # far past the ball still pulls back: 20 m -> 12 m past is +0.0, 12 m -> 8 m past is +0.2
        a["players"][0]["x"], b["players"][0]["x"] = 12.0, 8.0
        _, terms3 = self.env._reward(a, b)
        self.assertAlmostEqual(terms3[1]["position"], -0.8 - (-1.0))

    def test_position_shaping_mirrored_for_benders(self):
        # player 2 attacks -x: being at larger x than the ball is the good side
        a, b = state(ball_x=0.0), state(ball_x=0.0)
        a["players"][1]["x"], b["players"][1]["x"] = -6.0, 4.0
        _, terms = self.env._reward(a, b)
        self.assertAlmostEqual(terms[2]["position"], 0.6)

    def test_tip_decides_side_near_ball(self):
        # player 1 (attacks +x) 1 m PAST the ball: body-based it is on the wrong side, but with the pole
        # tip 2 m behind the ball it is hooked to push -> no penalty; tip 2 m beyond -> negative
        from spr_env import _position_potential
        s = state(ball_x=0.0)
        s["players"][0]["x"] = 1.0
        s["players"][0]["pole"] = [{"x": 1.0, "y": 0.0}, {"x": -2.0, "y": -3.0}]
        self.assertAlmostEqual(_position_potential(s, 1), 0.0)          # blend weight 1 (|ahead| = 1 < 2)
        s["players"][0]["pole"][-1]["x"] = 2.0
        self.assertAlmostEqual(_position_potential(s, 1), -0.2)
        # far from the ball the body decides (blend weight 0 beyond 3 m)
        s["players"][0]["x"] = 5.0
        self.assertAlmostEqual(_position_potential(s, 1), -0.5)

    def test_ball_respawn_ignored(self):
        _, terms = self.env._reward(state(ball_x=25.0), state(ball_x=0.0))
        self.assertEqual(terms[1]["ball_progress"], 0.0)

    def test_weights_override(self):
        env = PoleRidersEnv(reward={"kill": 0.25})
        a, b = state(), state()
        b["players"][1]["stats"]["deaths"] = 1
        rew, _ = env._reward(a, b)
        self.assertAlmostEqual(rew[1], 0.25)

    def test_observation_mirroring(self):
        s = state(ball_x=5.0)
        o1, o2 = observe(s, 1), observe(s, 2)
        self.assertEqual(len(o1), OBS_SIZE)
        self.assertAlmostEqual(o1[30], 5.0)    # ball ahead of agent 1 (attacks +x)
        self.assertAlmostEqual(o2[30], -5.0)   # same ball is behind agent 2 in its own frame
        self.assertAlmostEqual(o1[0], -15.7)
        self.assertAlmostEqual(o2[0], -15.7)   # both start on "their" left in their own frame
        self.assertAlmostEqual(o1[1], 0.7)     # y up


class DeadRiderTest(unittest.TestCase):
    """A killed rider (snapshot dead / respawn_t, PlayState::respawnTimer) is out of play like a ghost."""

    def kill(self, s, pid, t=2.4):
        p = s["players"][pid - 1]
        p.update(dead=True, respawn_t=t)
        return p

    def test_observe2_hides_dead_opponent(self):
        from farm_env import OBS2_SIZE, observe2

        class Flat:  # terrain stub
            def floor(self, x, y):
                return 0.0

            def above(self, x, y):
                return None
        s = state(ball_x=3.0)
        self.kill(s, 2, 2.4)
        o1, o2 = observe2(s, 1, Flat()), observe2(s, 2, Flat())
        self.assertEqual(len(o1), OBS2_SIZE)
        opp = o1[17:34]
        self.assertEqual(opp[9], 1.0)
        self.assertAlmostEqual(opp[16], 0.8)
        self.assertEqual([v for i, v in enumerate(opp) if i not in (9, 16)], [0.0] * 15)
        self.assertEqual(o2[9], 1.0)          # the victim's own ghost flag
        self.assertAlmostEqual(o2[16], 0.8)   # and its own countdown
        self.assertEqual(o2[17 + 9], 0.0)     # the live killer stays visible to it
        self.assertNotEqual(o2[17], 0.0)
        self.assertEqual(observe(s, 1)[15 + 9], 1.0)  # old observe(): ghost flag of the opponent

    def test_ghost_countdown_when_not_dead(self):
        from spr_env import respawn_in
        s = state()
        p = s["players"][0]
        p.update(ghost=True, ghost_t=0.3)
        self.assertAlmostEqual(respawn_in(p), 0.1)
        p.update(dead=True, respawn_t=1.5)   # dead wins over the ghost timer
        self.assertAlmostEqual(respawn_in(p), 0.5)

    def test_no_position_potential_while_dead(self):
        env = PoleRidersEnv(frame_skip=4)
        a, b = state(ball_x=0.0), state(ball_x=0.0)
        a["players"][0]["x"], b["players"][0]["x"] = 6.0, -4.0   # would be +0.6 if alive
        self.kill(b, 1)
        _, terms = env._reward(a, b)
        self.assertAlmostEqual(terms[1]["position"], 0.0 - (-0.6))  # only leaving the live state counts
        self.kill(a, 1)
        _, terms = env._reward(a, b)
        self.assertEqual(terms[1]["position"], 0.0)   # dead -> dead: nothing, wherever the corpse moves


class SymmetryTest(unittest.TestCase):
    def test_mirrored_situation_gives_identical_observations(self):
        """A situation and its left-right mirror must look the same to the two agents."""
        s = state(ball_x=1.5)
        p1, p2 = s["players"]
        p1.update(x=-4.0, y=-1.2, vx=2.0, vy=-0.5, a=0.3, w=1.1)
        p2.update(x=4.0, y=-1.2, vx=-2.0, vy=-0.5, a=-0.3, w=-1.1)   # exact mirror of p1
        for k in range(10):
            p1["pole"][k].update(x=-4.0 + 0.5 * k, y=-1.0 - 0.2 * k, w=0.7)
            p2["pole"][k].update(x=4.0 - 0.5 * k, y=-1.0 - 0.2 * k, w=-0.7)
        s["ball"].update(x=0.0, vx=0.0)
        o1, o2 = observe(s, 1), observe(s, 2)
        for i, (u, v) in enumerate(zip(o1, o2)):
            self.assertAlmostEqual(u, v, places=6, msg=f"feature {i}")

    def test_upright_rider_reads_upright_for_both(self):
        s = state()
        o1, o2 = observe(s, 1), observe(s, 2)
        self.assertAlmostEqual(o1[5], 1.0)  # cos(angle) of an upright rider
        self.assertAlmostEqual(o2[5], 1.0)


if __name__ == "__main__":
    unittest.main(verbosity=1)
