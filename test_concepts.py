#!/usr/bin/env python3
"""Sanity tests for concepts.py on hand-made states (no game needed; the last test uses the native sim).

    .venv/bin/python test_concepts.py
"""
import json, math, os, sys, unittest

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, "bridge"))
import concepts as C  # noqa: E402

POLE = 5.3  # tip distance from the body used below


def row(x=-5.0, y=0.69, vx=0.0, vy=0.0, ground=True, ghost=False, respawn=0.0, ang=-20.0, w=0.0,
        ox=5.0, oy=0.69, ovx=0.0, ovy=0.0, oground=True, oghost=False, orespawn=0.0, oang=-20.0,
        bx=0.0, by=5.0, score=(0, 0), tip=None):
    """One observe2 row in my attack frame (angles in deg, 0 = toward the goal I attack, 90 = up)."""
    o = np.zeros(62, np.float32)
    for base, (px, py, pvx, pvy, gr, gh, rs, a) in ((0, (x, y, vx, vy, ground, ghost, respawn, ang)),
                                                    (17, (ox, oy, ovx, ovy, oground, oghost, orespawn, oang))):
        if base == 17 and oghost:  # observe2 hides an out-of-play opponent
            o[17 + 9], o[17 + 16] = 1.0, orespawn
            continue
        r = math.radians(a)
        o[base:base + 4] = px, py, pvx, pvy
        o[base + 5], o[base + 7], o[base + 8], o[base + 9] = 1.0, 1.0, float(gr), float(gh)
        o[base + 10], o[base + 11] = math.sin(r), math.cos(r)
        o[base + 13], o[base + 14] = POLE * math.cos(r), POLE * math.sin(r)
        o[base + 16] = rs
    o[12] = w
    if tip is not None:  # explicit tip offset (dx, dy) from my body
        o[13], o[14] = tip
    o[52], o[53] = bx, by
    o[57], o[58] = score
    o[60], o[61] = bx - x, by - y
    return o


def frames(*rows):
    """Rows (one rider, consecutive steps) -> Frames with time first and a batch dim of 1."""
    return C.Frames(np.stack(rows)[:, None])


def at(F, name):
    return C.CONCEPTS[name][1](F)[:, 0]


# ---- snapshots (checks the mirroring of rider 2 through obs_from_states)

def body(x, y_up):
    return {"x": x, "y": -y_up, "a": 0.0, "vx": 0.0, "vy": 0.0, "w": 0.0}


def player(pid, x, y_up=0.69, ang=-20.0, ground=True, dead=False):
    """Rider `pid` with its pole at `ang` deg in ITS OWN attack frame (rider 2 attacks -x)."""
    sign = 1 if pid == 1 else -1
    r = math.radians(ang)
    p = dict(body(x, y_up), id=pid, on_pole=True, on_ground=ground, ghost=False, ghost_t=0.0, dead=dead,
             respawn_t=3.0 if dead else 0.0, hammer=False,
             stats={"kicks": 0, "jumps": 0, "suicides": 0, "kills": 0, "deaths": 0, "goals": 0, "own_goals": 0})
    p["pole"] = [body(x + sign * k / 9 * POLE * math.cos(r), y_up + k / 9 * POLE * math.sin(r)) for k in range(10)]
    return p


def snap(p1, p2, ball_x=0.0, score=(0, 0)):
    return {"in_match": True, "game_over": False, "match_id": 1, "score": list(score), "time_left": 300.0,
            "players": [p1, p2], "ball": dict(body(ball_x, 5.0))}


def from_snap(s, name):
    """Concept value for rider 1 and rider 2 in snapshot s."""
    O, live, same = C.obs_from_states([s])
    return C.CONCEPTS[name][1](C.Frames(O, live, same))[0]


class StateTest(unittest.TestCase):
    def test_floor(self):
        self.assertAlmostEqual(float(C.floor_height(0.0)), 0.0)
        self.assertAlmostEqual(float(C.floor_height(18.0)), -1.5)
        self.assertAlmostEqual(float(C.floor_height(-18.0)), -1.5)

    def test_good_side_and_situations_both_riders(self):
        s = snap(player(1, -5.0), player(2, 5.0))            # both behind the ball in their own frames
        self.assertEqual(list(from_snap(s, "good_side")), [True, True])
        self.assertEqual(list(from_snap(s, "sit_right_far")), [True, True])
        s = snap(player(1, 6.0), player(2, 3.0))             # rider 1: you-opp-ball = A; rider 2 right-close
        self.assertEqual(list(from_snap(s, "sit_A")), [True, False])
        self.assertEqual(list(from_snap(s, "sit_right_close")), [False, True])
        s = snap(player(1, 3.0), player(2, 6.0))             # rider 1: opp-you-ball = B; rider 2 is fine
        self.assertEqual(list(from_snap(s, "sit_B")), [True, False])
        self.assertEqual(list(from_snap(s, "opp_wrong_side")), [False, True])  # rider 1 is past the ball
        s = snap(player(1, -5.0), player(2, 5.0, dead=True))
        self.assertEqual(list(from_snap(s, "opp_out")), [True, False])  # the dead rider is not "in play"
        self.assertEqual(list(from_snap(s, "good_side")), [True, False])

    def test_situation_matches_situations_py(self):
        sys.path.insert(0, os.path.join(HERE, "analysis"))
        from situations import classify
        rng = np.random.default_rng(0)
        O = np.zeros((500, 62), np.float32)
        O[:, [0, 17, 52]] = rng.uniform(-15, 15, (500, 3))
        O[:, 26] = rng.random(500) < 0.1
        O[:, 9] = rng.random(500) < 0.1
        np.testing.assert_array_equal(C.situation(C.Frames(O)), classify(O))

    def test_tip_decides_near_ball(self):
        # rider 2 (attacks -x) 1 m past the ball: pole back-up (135) = tip behind the ball = hooked to push
        s = snap(player(1, -10.0), player(2, -1.0, ang=135.0))
        self.assertEqual(list(from_snap(s, "tip_behind_ball")), [False, True])
        s = snap(player(1, -10.0), player(2, -1.0, ang=45.0))
        self.assertEqual(list(from_snap(s, "tip_past_ball")), [False, True])
        self.assertEqual(list(from_snap(s, "good_side")), [True, False])

    def test_pole_quadrants(self):
        for a, name in ((-45, "pole_fwd_down"), (45, "pole_fwd_up"), (135, "pole_back_up"), (225, "pole_back_down")):
            F = frames(row(ang=a))
            self.assertEqual([n for n in ("pole_fwd_down", "pole_fwd_up", "pole_back_up", "pole_back_down")
                              if at(F, n)[0]], [name])

    def test_pole_over_head(self):
        # my pole 15 deg up: at the opponent 4 m out it is ~1.1 m above his body centre
        self.assertTrue(at(frames(row(x=0, ang=15, ox=4.0)), "pole_over_opp_head")[0])
        self.assertFalse(at(frames(row(x=0, ang=15, ox=1.0)), "pole_over_opp_head")[0])   # too close to me
        self.assertFalse(at(frames(row(x=0, ang=40, ox=4.0)), "pole_over_opp_head")[0])   # 3.4 m above him
        self.assertFalse(at(frames(row(x=0, ang=15, ox=4.0, oghost=True)), "pole_over_opp_head")[0])

    def test_opp_on_my_pole_and_mirror(self):
        # my pole 10 deg up; the opponent's feet on it 3 m out (body 0.69 above the pole)
        oy = 0.69 + 3 * math.tan(math.radians(10)) + 0.69
        s = snap(player(1, 0.0, ang=10.0), player(2, 3.0, y_up=oy, ground=True))
        self.assertEqual(list(from_snap(s, "opp_on_my_pole")), [True, False])
        self.assertEqual(list(from_snap(s, "on_opp_pole")), [False, True])
        s = snap(player(1, 0.0, ang=10.0), player(2, 3.0, y_up=oy, ground=False))  # flying past: not standing
        self.assertEqual(list(from_snap(s, "opp_on_my_pole")), [False, False])

    def test_tip_peek(self):
        # me behind the ball, the opponent on his good side, my tip 2.6 m past the ball toward him
        self.assertTrue(at(frames(row(x=-2, ang=30, ox=6)), "tip_peek")[0])
        self.assertFalse(at(frames(row(x=-2, ang=150, ox=6)), "tip_peek")[0])     # tip behind the ball
        self.assertFalse(at(frames(row(x=-2, ang=30, ox=-6)), "tip_peek")[0])     # opponent on my side

    def test_planted_and_airborne(self):
        F = frames(row(ang=-8.0, y=0.69, ground=False))   # tip 0.69 - 5.3 sin 8 = -0.05 -> on the floor
        self.assertTrue(at(F, "pole_planted")[0]); self.assertTrue(at(F, "airborne")[0])
        F = frames(row(ang=10.0))
        self.assertFalse(at(F, "pole_planted")[0]); self.assertFalse(at(F, "airborne")[0])

    def test_dead_rider_has_no_states(self):  # (incl. the 2026-10-02 states)
        F = frames(row(ghost=True, respawn=1.0))
        for n in C.STATE_NAMES:
            self.assertFalse(np.any(at(F, n)), n)


class EventTest(unittest.TestCase):
    def test_goal_and_kill(self):
        F = frames(row(), row(score=(1, 0)), row(score=(1, 0), oghost=True, orespawn=1.0),
                   row(score=(1, 1), oghost=True, orespawn=0.9))
        self.assertEqual(list(at(F, "goal_for") > 0), [False, True, False, False])
        self.assertEqual(list(at(F, "goal_against") > 0), [False, False, False, True])
        self.assertEqual(list(at(F, "kill") > 0), [False, False, True, False])
        # a post-goal respawn ghost (respawn_in 0.1) is not a kill
        F = frames(row(), row(oghost=True, orespawn=0.1))
        self.assertFalse(np.any(at(F, "kill")))
        F = frames(row(), row(ghost=True, respawn=1.0))
        self.assertEqual(list(at(F, "death") > 0), [False, True])

    def test_events_do_not_cross_matches(self):
        O = np.stack([row(), row(score=(1, 0))])[:, None]
        F = C.Frames(O, same=np.array([[False], [False]]))
        self.assertFalse(np.any(C.goal_for(F)))

    def test_passes(self):
        F = frames(row(ang=80), row(ang=100), row(ang=80), row(ang=170))
        self.assertEqual(list(at(F, "back_pass") > 0), [False, True, False, False])  # 80 -> 170 jumps > 60
        self.assertEqual(list(at(F, "fwd_pass") > 0), [False, False, True, False])

    def test_vault_and_pogo(self):
        plant = -8.0  # tip on the floor with the body at 0.69 m
        fly = dict(ground=False)
        # falls onto the planted pole without touching down, rebounds at 9 m/s, releases airborne
        seq = [row(ang=30, vy=-6, y=3, **fly), row(ang=plant, vy=-5, **fly), row(ang=plant, vy=2, **fly),
               row(ang=plant, vy=9, **fly), row(ang=40, vy=8, y=3, **fly)]
        F = frames(*seq)
        self.assertEqual(list(at(F, "vault")), [0, 0, 0, 0, 4])
        self.assertEqual(list(at(F, "pogo_vault")), [0, 0, 0, 0, 4])
        # same launch from standing on the ground: a vault but no pogo
        seq[1] = row(ang=plant, vy=0, ground=True)
        F = frames(*seq)
        self.assertEqual(int(at(F, "vault")[-1]), 4)
        self.assertEqual(int(at(F, "pogo_vault")[-1]), 0)
        # a slow release (max 3 m/s) is no vault
        F = frames(row(ang=30, y=3, **fly), row(ang=plant, vy=1, **fly), row(ang=plant, vy=3, **fly),
                   row(ang=40, vy=3, y=3, **fly))
        self.assertFalse(np.any(at(F, "vault")))

    def test_jump_over_ball(self):
        A = row(x=4, ox=2, bx=0)                                  # you-opp-ball
        cross = row(x=-1, ox=2, bx=0, vx=-6, ground=False)
        good = row(x=-3, ox=2, bx=0, ground=False)
        F = frames(A, A, row(x=1, ox=2, bx=0, vx=-6, ground=False), cross, *[good] * 6)
        ev = at(F, "jump_over_ball")
        self.assertEqual(list(np.nonzero(ev)[0]), [3 + C.HZ // 2 - 1])
        # walking back on the ground is not the jump
        F = frames(A, A, row(x=1, ox=2, bx=0, vx=-6), row(x=-1, ox=2, bx=0, vx=-6), *[row(x=-3, ox=2, bx=0)] * 6)
        self.assertFalse(np.any(at(F, "jump_over_ball")))

    def test_pushed_opp_wrong_side(self):
        # the opponent 2 m past the ball on his good side, my pole on him (pole 0 deg, he is 4 m out at my height)
        good = row(x=-2, ang=0, ox=2, bx=0)
        F = frames(*[good] * 8, row(x=-1, ang=0, ox=-0.5, bx=0), *[row(x=-3, ang=60, ox=-1, bx=0)] * C.HZ)
        ev = at(F, "pushed_opp_wrong_side")
        self.assertEqual(list(np.nonzero(ev)[0]), [8 + C.HZ - 1])
        # no contact (my pole straight up): he crossed on his own
        F = frames(*[row(x=-2, ang=90, ox=2, bx=0)] * 8, *[row(x=-3, ang=90, ox=-1, bx=0)] * (C.HZ + 1))
        self.assertFalse(np.any(at(F, "pushed_opp_wrong_side")))

    def test_catapult(self):
        oy = 0.69 + 3 * math.tan(math.radians(10)) + 0.69
        on = row(x=0, ang=10, ox=3, oy=oy, oground=True, ovx=1.0)
        thrown = row(x=0, ang=50, ox=5, oy=4, oground=False, ovx=8, ovy=9, w=6)
        F = frames(on, on, on, row(x=0, ang=30, ox=4, oy=3, oground=False, ovx=2, ovy=3), thrown, thrown)
        self.assertEqual(list(at(F, "catapult") > 0), [False, False, False, False, True, False])
        # he only touched it (one step): no throw
        F = frames(row(x=0, ang=10, ox=8), on, row(x=0, ang=30, ox=4, oy=3, oground=False), thrown)
        self.assertFalse(np.any(at(F, "catapult")))

    def test_tip_got_behind_ball(self):
        past, behind = row(x=1, ang=60, bx=0, ox=8), row(x=1, ang=150, bx=0, ox=8)
        F = frames(past, past, *[behind] * 5)
        self.assertEqual(list(np.nonzero(at(F, "tip_got_behind_ball"))[0]), [6])
        F = frames(past, past, *[behind] * 3, past)
        self.assertFalse(np.any(at(F, "tip_got_behind_ball")))

    def test_rhythmic_pressing(self):
        # situation B, my pole forward-down on the opponent 4 m ahead, rotation reversing every 3 steps (2.5 Hz)
        rows = [row(x=2, bx=0, ox=6, oy=0.69, ang=-5, w=(3.0 if (t // 3) % 2 else -3.0)) for t in range(2 * C.HZ)]
        self.assertTrue(at(frames(*rows), "rhythmic_pressing")[-1])
        still = [row(x=2, bx=0, ox=6, oy=0.69, ang=-5, w=0.0) for _ in range(2 * C.HZ)]
        self.assertFalse(np.any(at(frames(*still), "rhythmic_pressing")))

    def test_window_helpers(self):
        b = np.array([1, 1, 0, 1, 1, 1], bool)[:, None]
        same = np.array([0, 1, 1, 0, 1, 1], bool)[:, None]
        self.assertEqual(list(C._run_length(b, same)[:, 0]), [1, 2, 0, 1, 2, 3])
        self.assertEqual(list(C._window_count(b, 2, same)[:, 0]), [1, 2, 1, 1, 2, 2])
        self.assertEqual(list(C._age(b, same, cap=99)[:, 0]), [0, 0, 1, 0, 0, 0])


def tc(age=999.0, what=0, kick=999.0, oage=999.0, okick=999.0, last=False, now=False):
    """Touch rows (me, opp) as NativeHost out["touch"] gives them (seconds since touch / kick)."""
    t = np.zeros((2, 5), np.float32)
    t[0] = age, what, kick, float(now), float(last)
    t[1] = oage, 0, okick, 0, float(not last and oage < 999)
    return t


def tframes(rows, touches):
    return C.Frames(np.stack(rows)[:, None], touch=np.stack(touches)[:, None])


class NewConceptsTest(unittest.TestCase):
    """2026-10-02 concepts: commitment / agency / energy / hitting the ball second."""

    def test_energy(self):
        F = frames(row(y=0.69), row(y=5.69, vx=10.0), row(y=6.69, vx=10.0))
        np.testing.assert_allclose(F.energy[:, 0], [0.0, 150.0, 170.0], atol=1e-3)
        self.assertEqual(list(at(F, "high_energy")), [False, False, True])

    def test_takeoff_speed(self):
        F = frames(row(), row(ground=False, vx=10.0, vy=5.0), row(ground=False, vx=15.0), row(ground=False))
        self.assertEqual(list(at(F, "takeoff") > 0), [False, True, False, False])
        self.assertAlmostEqual(float(C.takeoff_speed(F)[1, 0]), 15.0, 4)   # the next step's speed if higher
        self.assertEqual(list(at(F, "fast_takeoff") > 0), [False, True, False, False])

    def test_lunge_and_overcommit(self):
        fly = dict(ground=False, ang=30)
        # [ball, me, opp]: ball 0, me 2, the opponent 6 m on; I jump at him, he stays put, I land past him
        seq = [row(x=2, bx=0, ox=6), row(x=3, bx=0, ox=6, vx=8, **fly), row(x=5, bx=0, ox=6, vx=8, **fly),
               row(x=7, bx=0, ox=6, vx=8, **fly), row(x=8, bx=0, ox=6, vx=4, **fly)]
        F = frames(*seq)
        self.assertEqual(list(np.nonzero(at(F, "lunge_at_opp"))[0]), [1])
        self.assertEqual(list(np.nonzero(at(F, "overcommit"))[0]), [3])            # the flip to [ball, opp, me]
        self.assertEqual(int(at(F, "overcommit")[3]), 3)
        self.assertEqual(list(np.nonzero(at(F, "overcommit_opp_still"))[0]), [3])
        # he moved (5 m/s): still an overcommit, but not "he did nothing"
        F = frames(*[row(**{**dict(x=r[0], bx=0, ox=6, ovx=5.0), **({} if i == 0 else dict(vx=8, **fly))})
                     for i, r in enumerate(([2], [3], [5], [7], [8]))])
        self.assertTrue(np.any(at(F, "overcommit")))
        self.assertFalse(np.any(at(F, "overcommit_opp_still")))
        # the ball ahead of me (I'm on the good side): no lunge
        F = frames(row(x=2, bx=5, ox=6), row(x=3, bx=5, ox=6, vx=8, **fly))
        self.assertFalse(np.any(at(F, "lunge_at_opp")))
        # deep defence: in my pit with the ball over me, he waits in front of the hole
        F = frames(row(x=-17.5, bx=-17.2, ox=-14), row(x=-17, bx=-17.2, ox=-14, vx=7, **fly))
        self.assertEqual(list(np.nonzero(at(F, "lunge_from_pit"))[0]), [1])

    def test_committed_and_at_mercy(self):
        free = dict(y=5.0, ground=False, ang=30)
        F = frames(row(**free, ox=15, oground=False, bx=-10), row(**free, ox=2, bx=3), row(y=0.69, ang=30))
        self.assertEqual(list(at(F, "committed")), [True, True, False])            # grounded: not committed
        self.assertEqual(list(at(F, "at_mercy")), [False, True, False])            # he stands within reach
        # touching the ball (touch rows): not committed
        F = tframes([row(**free, ox=15, bx=-10)] * 2, [tc(age=0.0, what=1, now=True), tc(age=0.2)])
        self.assertEqual(list(at(F, "committed")), [False, True])

    def test_second_touch_and_kick(self):
        r = row()
        F = tframes([r] * 4, [tc(oage=0.1), tc(age=0.0, what=2, oage=0.15), tc(age=0.0167, oage=0.2),
                              tc(age=0.1, oage=0.27)])
        self.assertEqual(list(at(F, "second_touch") > 0), [False, True, False, False])  # the onset only
        F = tframes([r] * 3, [tc(okick=0.9), tc(kick=0.0, age=0.0, okick=0.95), tc(kick=0.0, age=0.0, okick=0.3)])
        self.assertEqual(list(at(F, "second_kick") > 0), [False, False, True])   # 0.95 s: too late
        F = frames(r, r)                                                         # no touch rows: never
        self.assertFalse(np.any(at(F, "second_touch")))

    def test_got_flung_and_own_goal(self):
        # I stand on his pole (his pole 10 deg up from x = 3, attack frame of mine: he is ahead of me)
        oy = 0.69 + 3 * math.tan(math.radians(10)) + 0.69
        on = row(x=0, y=oy, ground=True, ox=3, oang=170, vx=1.0)  # his pole points back toward me
        flung = row(x=-4, y=4, ground=False, vx=-8, vy=9, ox=3, oang=120)
        F = frames(on, on, on, row(x=-1, y=3, ground=False, vx=-2, vy=3, ox=3, oang=150), flung, flung)
        self.assertTrue(at(F, "on_opp_pole")[0])
        self.assertEqual(list(at(F, "got_flung") > 0), [False, False, False, False, True, False])
        # ... flung into the ball, then the ball goes into my own goal credited to me
        rows = [on, on, on, flung, flung, row(x=-6, ground=False), row(x=-6, score=(0, 1))]
        tcs = [tc()] * 4 + [tc(age=0.0, what=1, now=True), tc(age=0.1, last=True), tc(age=0.2, last=True)]
        F = tframes(rows, tcs)
        self.assertEqual(list(np.nonzero(at(F, "flung_own_goal"))[0]), [6])
        tcs[-1] = tc(age=0.2, oage=0.05, last=False)                              # he touched it last: his goal
        self.assertFalse(np.any(at(tframes(rows, tcs), "flung_own_goal")))

    def test_impale_brake(self):
        y = 1.9  # pole at 200 deg: tip 1.8 m below the body -> on the floor
        F = frames(row(y=y, ang=150, vx=-9, ground=False), row(y=y, ang=200, vx=-9, ground=False),
                   row(y=y, ang=200, vx=-7, ground=False), row(y=y, ang=200, vx=-4, ground=False))
        self.assertEqual(list(at(F, "impale_plant") > 0), [False, True, False, False])
        self.assertEqual(list(np.nonzero(at(F, "impale_brake"))[0]), [3])           # 9 -> 4 m/s
        F = frames(row(y=y, ang=150, vx=9, ground=False), row(y=y, ang=200, vx=9, ground=False))
        self.assertFalse(np.any(at(F, "impale_plant")))                            # flying forward: no


def aframes(rows, acts):
    """Frames with action rows (me, opp) per step: acts = [(my aim x, my aim y), ...] (opp holds his stick up)."""
    A = np.zeros((len(rows), 1, 2, 3), np.float32)
    for t, (ax, ay) in enumerate(acts):
        A[t, 0, 0] = 0.0, ax, ay
        A[t, 0, 1] = 0.0, 0.0, 1.0
    return C.Frames(np.stack(rows)[:, None], act=A)


LOCK = dict(x=-1.0, ang=0.0, ox=1.0, oang=180.0)   # my pole on him, his on me (mutual contact)
APART = dict(x=-1.0, ang=0.0, ox=12.0, oang=180.0)


class Concepts1003Test(unittest.TestCase):
    """2026-10-03 concepts: interlocks, letting go, open goals, the pole servo."""

    def test_servo_off(self):
        self.assertTrue(C.servo_off(0.0, 0.0) and C.servo_off(0.25, 0.0) and C.servo_off(0.0, -0.279))
        self.assertFalse(C.servo_off(0.35, 0.0) or C.servo_off(0.25, 0.25) or C.servo_off(1.0, 1.0))

    def test_interlock(self):
        G = C.INTERLOCK_GAP
        F = frames(*([row(**APART)] * 5 + [row(**LOCK)] * 20 + [row(**APART)] * 10))
        mc, il, ev = at(F, "mutual_pole_contact"), at(F, "interlocked"), at(F, "pole_interlock")
        self.assertEqual(list(np.nonzero(mc)[0]), list(range(5, 25)))
        self.assertEqual(int(np.argmax(il)), 5 + C.INTERLOCK_MIN - 1)               # after 1 s of contact
        self.assertTrue(il[24] and il[24 + G] and not il[25 + G])                     # bridged up to the gap
        self.assertEqual(list(np.nonzero(ev)[0]), [24 + G + 1])                       # marked when the gap is exceeded
        self.assertEqual(int(ev[24 + G + 1]), 20 + G + 1)
        # a short gap is bridged (one interlock of 23 steps), a long one splits it into two too-short runs
        F = frames(*([row(**LOCK)] * 10 + [row(**APART)] * 3 + [row(**LOCK)] * 10 + [row(**APART)] * 10))
        ep = C.interlock_episodes(F)
        self.assertEqual(len(ep), 1)
        self.assertAlmostEqual(ep[0]["dur_s"], 23 / C.HZ)
        F = frames(*([row(**LOCK)] * 10 + [row(**APART)] * (G + 2) + [row(**LOCK)] * 10 + [row(**APART)] * 10))
        self.assertFalse(np.any(at(F, "pole_interlock")))
        # still interlocked when the rows end: no end seen, no mark (a match end inside the rows marks its last row)
        self.assertFalse(np.any(at(frames(*([row(**LOCK)] * 20)), "pole_interlock")))
        F = C.Frames(np.stack([row(**LOCK)] * 20 + [row(**APART)] * 3)[:, None],
                     same=np.r_[False, [True] * 19, False, True, True][:, None])
        self.assertEqual(list(np.nonzero(at(F, "pole_interlock"))[0]), [19])
        # episode fields: me on the good side (ball ahead), him on his wrong side -> I'm the attacker
        F = frames(*([row(**LOCK, bx=5.0)] * 20 + [row(**APART, bx=5.0)] * 10))
        ep, = C.interlock_episodes(F)
        self.assertEqual((ep["attacker"], ep["start"], ep["last"]), (1, 0, 19))

    def test_let_go_escape(self):
        rows = [row(**LOCK)] * 20 + [row(**APART)] * 10
        hold, off = (0.0, 1.0), (0.0, 0.0)
        F = aframes(rows, [hold] * 18 + [off] * 12)                                  # let go at the end
        self.assertEqual(list(np.nonzero(at(F, "let_go_escape"))[0]), list(np.nonzero(at(F, "pole_interlock"))[0]))
        self.assertTrue(at(F, "interlock_let_go")[18])
        self.assertTrue(np.all(F.oreleased == False))  # noqa: E712 (he holds his stick up)
        F = aframes(rows, [hold] * 8 + [off] * 3 + [hold] * 19)                       # let go and took it back
        self.assertFalse(np.any(at(F, "let_go_escape")))
        F = aframes(rows, [off] * 30)                                                 # limp all along
        self.assertFalse(np.any(at(F, "let_go_escape")))
        F = aframes(rows, [hold] * 10 + [off] * 20)                                   # released 0.67 s before the end
        self.assertFalse(np.any(at(F, "let_go_escape")))
        F = frames(*rows)                                                             # no action rows
        self.assertFalse(np.any(at(F, "let_go_escape")) or np.any(at(F, "interlock_let_go")))

    def test_open_goal(self):
        self.assertEqual(list(at(frames(row(x=-5, bx=0, oghost=True, orespawn=1.0), row(x=-5, bx=0, ox=-6),
                                        row(x=-5, bx=0, ox=-3), row(x=-5, bx=0, ox=5), row(x=2, bx=0, ox=-6)), "open_goal")),
                         [True, True, False, False, False])
        # 10 steps open, then my goal: converted, length = from the window start
        seq = [row(x=-5, bx=0, ox=5)] * 3 + [row(x=-5, bx=0, ox=-8)] * 10 + [row(x=-5, bx=0, ox=-8, score=(1, 0))] + \
              [row(x=-5, bx=0, ox=-8, score=(1, 0))] * 5
        F = frames(*seq)
        ev = at(F, "open_goal_conversion")
        self.assertEqual(list(np.nonzero(ev)[0]), [13])
        self.assertEqual(int(ev[13]), 11)
        self.assertFalse(np.any(at(F, "open_goal")[14:]))                               # not in the 2 s after a goal
        # the goal comes 2 s after the window closed: not a conversion
        F = frames(*([row(x=-5, bx=0, ox=-8)] * 5 + [row(x=-5, bx=0, ox=5)] * 30 + [row(x=-5, bx=0, ox=5, score=(1, 0))]))
        self.assertFalse(np.any(at(F, "open_goal_conversion")))

    def test_aim_back_pass(self):
        a = lambda deg: (math.cos(math.radians(deg)), math.sin(math.radians(deg)))  # noqa: E731
        rows = [row()] * 5
        F = aframes(rows, [a(30), a(80), a(100), a(150), a(100)])                    # over the top backward once
        self.assertEqual(list(np.nonzero(at(F, "aim_back_pass"))[0]), [2])
        self.assertAlmostEqual(float(F.aim[3, 0]), 150.0, 3)
        F = aframes(rows, [a(30), a(80), (0.0, 0.0), a(100), a(150)])                # through the centre: servo off
        self.assertFalse(np.any(at(F, "aim_back_pass")))

    def test_playlog_sudden_death_and_sticks(self):
        """A tie at the end of time: play ends with regulation (house rules), not at the sudden-death goal; the logged
        sticks map to actions in each rider's attack frame."""
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x_m1.jsonl")
            with open(path, "w") as f:
                for t in range(80):
                    st = snap(player(1, -5.0), player(2, 5.0), score=(1, 1) if t < 70 else (2, 1))
                    st["time_left"] = max(0.0, 2.0 - t * 4 / 60) if t < 31 else -0.004
                    st["sudden_death"] = t >= 31
                    for p in st["players"]:
                        p["pad"] = p["id"] - 1
                    f.write(json.dumps({"f": 4 * t, "kind": [1, 2, 0, 0],
                                        "ax": [[0.5, 0.0, 0.3, -0.8], [0.4, 0.0, 0.6, 0.2], [0] * 4, [0] * 4], "s": st}) + "\n")
            (O, live, same, humans, (idx, info)), = list(C.playlog_matches(path, with_info=True))
            self.assertEqual(int(O[-1, 0, C.MY_SCORE]), 1)                           # the sudden-death goal is cut
            self.assertLessEqual(info["end"], 31)
            frames_, states, _ = C.playlog_read(path)[1]
            ticks, _, _ = C.playlog_ticks(frames_, states)
            self.assertEqual(states[C.playlog_end(states, ticks, sudden_death=True)]["score"], [2, 1])
            A = C.playlog_actions(path, frames_[idx], {1: 0, 2: 1})
        np.testing.assert_allclose(A[0, 0, 0], [0.5, 0.3, 0.8], atol=1e-6)            # rider 1: run lx, aim (rx, -ry)
        np.testing.assert_allclose(A[0, 1, 0], [-0.4, -0.6, -0.2], atol=1e-6)         # rider 2 mirrored
        np.testing.assert_allclose(A[0, 0, 1], A[0, 1, 0])                            # (me, opp) views


class SnapshotPathTest(unittest.TestCase):
    def test_native_snapshots_give_native_obs(self):
        """obs_from_states(NativeHost.json) == the native observations (the play-log path)."""
        from native_env import NativeHost
        env = NativeHost(1, 1, 4, seed=3, threads=1)
        obs = env.reset_all()
        rng = np.random.default_rng(1)
        S, O = [], []
        for _ in range(60):
            S.append(env.json(0)); O.append(obs[0].copy())
            obs = env.step(rng.uniform(-1, 1, (1, 2, 3)).astype(np.float32))["obs"]
        env.close()
        O2, live, same = C.obs_from_states(S)
        self.assertTrue(live.all() and same[1:].all() and not same[0].any())
        np.testing.assert_allclose(O2, np.array(O), atol=2e-3)
        # the same snapshots as a play log (bridge playlog_frame format: every frame a line, "s" every 4th)
        import tempfile
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x_m1.jsonl")
            with open(path, "w") as f:
                for t, st in enumerate(S):
                    st = json.loads(st)
                    for p in st["players"]:
                        p["pad"] = p["id"] - 1
                    for k in range(4):
                        f.write(json.dumps({"f": 4 * t + k, "kind": [3, 2, 0, 0], "ax": [[0, 0, 0, 0]] * 4,
                                            **({"s": st} if k == 0 else {})}) + "\n")
                f.write('{"f": 999, "kind"')  # cut off when the game closed
            (O3, live3, same3, humans), = list(C.playlog_matches(path))
        np.testing.assert_allclose(O3, O2)
        self.assertEqual(list(humans), [True, False])
        self.assertTrue(same3[1:].all())
        # a 120 Hz display: a snapshot every 4 rendered frames = every 2 ticks (here: an in-between copy with the
        # clock half a step on); resampling by the clock gives the 4-tick rows back, the old path every snapshot
        with tempfile.TemporaryDirectory() as d:
            path = os.path.join(d, "x_m1.jsonl")
            with open(path, "w") as f:
                for t, st in enumerate(S):
                    for h in range(2):
                        s2 = json.loads(st)
                        for p in s2["players"]:
                            p["pad"] = p["id"] - 1
                        s2["time_left"] -= h * 2 / 60
                        f.write(json.dumps({"f": 8 * t + 4 * h, "kind": [1, 2, 0, 0], "ax": [[0, 0, 0, 0]] * 4,
                                            "s": s2}) + "\n")
            (O4, live4, same4, humans4, (idx, info)), = list(C.playlog_matches(path, with_info=True))
            (O5, _, _, _), = list(C.playlog_matches(path, resample=False))
        self.assertEqual(info["ticks_per_snapshot"], 2.0)
        self.assertEqual(list(idx), list(range(0, 2 * len(S), 2)))
        np.testing.assert_allclose(O4, O2)
        self.assertEqual(len(O5), 2 * len(S))
        self.assertTrue(same4[1:].all() and list(humans4) == [True, False])


if __name__ == "__main__":
    unittest.main(verbosity=1)
