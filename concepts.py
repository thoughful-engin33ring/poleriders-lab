#!/usr/bin/env python3
"""Concept library: the game ideas of docs/strategy.md as measurable predicates on game states.

Works on any sequence of observe2 rows (bridge/farm_env.py, 62 floats in one rider's attack frame: I attack +x,
y up), so the native sim's observations, NativeHost.json(k) snapshots and the human play logs (runs/human/*.jsonl,
a snapshot every 4th frame) all go through the same code:

    F = Frames(O, live)            # O float[T, ..., 62] (time first, any batch dims), live bool[T, ...]
    F = Frames(O, live, touch=TC)  # + ball touches float[T, ..., 2, 5] (me, opp): NativeHost out["touch"] rows
                                   #   (rows without it: the touch concepts are all false)
    F = Frames(O, live, act=A)     # + actions float[T, ..., 2, 3] (me, opp), each in its own attack frame (run,
                                   #   aim x, aim y): the sim's actions / a play log's sticks (playlog_actions);
                                   #   needed by the pole-servo concepts (let_go_escape, aim_back_pass)
    S = state_concepts(F)          # name -> bool[T, ...] (or float): true at that moment
    E = event_concepts(F)          # name -> int16[T, ...]: > 0 on the step the event is complete (no lookahead),
                                   #   value = its length in steps (it started at t - value + 1)
    O, live, same = obs_from_states(states)          # snapshots (one match or several) -> rows for both riders
    for O, live, same in playlog_matches(path): ...  # a human play log, one match at a time

One step = 4 frames = 1/15 s (frame skip 4, also the play logs' snapshot rate). Geometry (arena 2 = level 1, the
training map): goal-wall faces at x = +-20.3 m, pits 16.3-20.3 m (floor 1.5 m below the plateau at y = 0); the pole
is ~6 m (10 segments 0.61 m apart, tip 4.3-5.6 m from the body); the body's centre is 0.69 m above the floor when
standing. Rows of a dead / ghost rider (obs[9]) are false for every state concept except opp_out. Thresholds
were set from farm2 self-play distributions (see each docstring); tests: test_concepts.py.

The sim's own facts used here (sim/src/game/contacts.cpp, sim/re_notes): `on_ground` (obs[8]) is the foot
sensor touching the ground OR a pole; kills are rider-on-rider contacts and set a 3 s respawn timer (obs respawn_in
= 1.0), deleter falls 1/60 s, and ~1.4 s after a goal both riders respawn with a 0.3 s spawn ghost (respawn_in 0.1).
"""
import json, os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "bridge"))

STEP_S = 4 / 60          # one decision step
HZ = 15                  # steps per second
HALF_POLE = 3.0          # m, "within half a pole of the ball" (situations.py, the position potential)
G = 20.0                 # m/s^2, the game's gravity (PlayState ctor: b2World gravity (0, 20), y down)
REACH = 5.5              # m, how far a rider reaches with the pole tip (tip 4.3-5.6 m from the body)
PIT_X = 16.3             # m, pits 16.3..20.3 m from the centre
GOAL_X = 20.3            # m, goal-wall face
PLANTED_H = 0.35         # m, tip this close to the floor = planted (a resting tip sits 0.10-0.16 m up)
STAND_H = 0.69           # m, body centre above the floor when standing

# observe2 layout (bridge/farm_env.py): me[0:17], opp[17:34], floor[34:43], overhead[43:52], ball[52:56],
# match[56:60], ball_rel[60:62]
X, Y, VX, VY, ON_POLE, ON_GROUND, GHOST, PSIN, PCOS, PW, TIPDX, TIPDY, RESPAWN = 0, 1, 2, 3, 7, 8, 9, 10, 11, 12, 13, 14, 16
OPP, BALL_X, BALL_Y, MY_SCORE, OPP_SCORE, BALL_DX = 17, 52, 53, 57, 58, 60

_FLOOR = None


def floor_height(x):
    """Height (y up, m) of the main floor of arena 2 under x: 0 on the plateau, sloping to -1.5 m in the pits.
    The arena is mirror-symmetric, so it also works on attack-frame (mirrored) x."""
    global _FLOOR
    if _FLOOR is None:
        from farm_env import Terrain
        t = Terrain(2)
        xs = np.arange(0.0, 30.0001, 0.05)
        _FLOOR = (xs, np.array([-t.floor(v, -0.5) for v in xs]))  # first surface at/below plateau level
    return np.interp(np.abs(x), *_FLOOR)


def _seg_dist(px, py, ax, ay, bx, by, u0=0.0):
    """Distance from point p to the segment a->b restricted to the part u >= u0, and the parameter u."""
    vx, vy = bx - ax, by - ay
    u = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy + 1e-9), u0, 1.0)
    return np.hypot(px - ax - u * vx, py - ay - u * vy), u


class Frames:
    """Derived quantities of observe2 rows O[T, ..., 62] (time first). live[T, ...]: the row is a moment of a
    match in play (default: all). same[T, ...]: row t continues row t-1 (default: live[t] & live[t-1]; a play
    log also breaks it at a missing snapshot). touch[T, ..., 2, 5]: optional ball-touch rows (me, opp) from
    the native env (bridge/native_env.py TOUCH_*: seconds since the rider's last touch, what touched, seconds
    since its last kick, touching now, the game's lastTouch is this rider); not in observations / play logs."""

    def __init__(self, O, live=None, same=None, touch=None, act=None):
        O = np.asarray(O, np.float32)
        self.O = O
        self.live = np.ones(O.shape[:-1], bool) if live is None else np.asarray(live, bool)
        if same is None:
            same = np.zeros_like(self.live)
            same[1:] = self.live[1:] & self.live[:-1]
        self.same = np.asarray(same, bool) & self.live
        o = lambda i: O[..., i]
        self.alive = self.live & (o(GHOST) < 0.5)
        self.x, self.y, self.vx, self.vy = o(X), o(Y), o(VX), o(VY)
        self.ground = o(ON_GROUND) > 0.5
        a = np.degrees(np.arctan2(o(PSIN), o(PCOS)))
        self.ang = np.where(a < -90, a + 360, a)           # pole base->tip, deg in [-90, 270): 0 fwd, 90 up
        self.w = o(PW)                                     # rad/s, + = counter-clockwise = backward
        self.tipx, self.tipy = self.x + o(TIPDX), self.y + o(TIPDY)
        self.h = self.y - floor_height(self.x)             # body centre above the floor
        self.tip_h = self.tipy - floor_height(self.tipx)
        self.opp_alive = self.alive & (o(OPP + GHOST) < 0.5)
        self.ox, self.oy, self.ovx, self.ovy = o(OPP + X), o(OPP + Y), o(OPP + VX), o(OPP + VY)
        self.oground = o(OPP + ON_GROUND) > 0.5
        self.oh = self.oy - floor_height(self.ox)
        self.otipx, self.otipy = self.ox + o(OPP + TIPDX), self.oy + o(OPP + TIPDY)
        self.bx, self.by = o(BALL_X), o(BALL_Y)
        self.planted = self.tip_h < PLANTED_H
        self.otip_h = self.otipy - floor_height(self.otipx)
        self.speed, self.ospeed = np.hypot(self.vx, self.vy), np.hypot(self.ovx, self.ovy)
        # total mechanical energy per unit mass (J/kg = m^2/s^2): kinetic 0.5 |v|^2 + potential G * height of the
        # body centre above a rider standing on the plateau (y = STAND_H; standing in a pit = -30). The pole's
        # elastic energy (its bend) is not observed and not included.
        self.energy = 0.5 * self.speed ** 2 + G * (self.y - STAND_H)
        self.oenergy = 0.5 * self.ospeed ** 2 + G * (self.oy - STAND_H)
        self.touch = None if touch is None else np.asarray(touch, np.float32)
        if self.touch is not None:
            tc = self.touch
            self.t_age, self.ot_age = tc[..., 0, 0], tc[..., 1, 0]      # s since my / his last ball touch
            self.k_age, self.ok_age = tc[..., 0, 2], tc[..., 1, 2]      # s since my / his last kick
            self.touched = self.alive & (self.t_age < STEP_S - 1e-4)   # I touched the ball during this step
            self.otouched = self.live & (self.ot_age < STEP_S - 1e-4)
            self.kicked = self.alive & (self.k_age < STEP_S - 1e-4)
            self.game_last = tc[..., 0, 4] > 0.5                        # the game credits the ball to me
        self.act = None if act is None else np.asarray(act, np.float32)
        if self.act is not None:
            ac = self.act
            self.released = self.alive & servo_off(ac[..., 0, 1], ac[..., 0, 2])        # my pole hangs free
            self.oreleased = self.opp_alive & servo_off(ac[..., 1, 1], ac[..., 1, 2])
            a = np.degrees(np.arctan2(ac[..., 0, 2], ac[..., 0, 1]))
            self.aim = np.where(a < -90, a + 360, a)       # where my stick aims the pole, deg in [-90, 270) like ang

    def pole_to_opp(self, u0=0.2):
        """Distance from the opponent's body to my pole (straight body->tip line, from u0 of its length out)."""
        return _seg_dist(self.ox, self.oy, self.x, self.y, self.tipx, self.tipy, u0)

    def opp_pole_to_me(self, u0=0.2):
        return _seg_dist(self.x, self.y, self.ox, self.oy, self.otipx, self.otipy, u0)


CONCEPTS = {}  # name -> ("state" | "event", function(Frames))


def concept(kind):
    def deco(fn):
        CONCEPTS[fn.__name__] = (kind, fn)
        return fn
    return deco


# ---------------------------------------------------------------- state concepts

@concept("state")
def good_side(F):
    """you-ball-goal: my body is behind the ball (ball x > my x in my attack frame). strategy.md "Frame"."""
    return F.alive & (F.bx > F.x)


@concept("state")
def near_ball(F):
    """Body within half a pole (3 m) of the ball along x: where farm1 "gets bad" and the tip starts to decide
    the side (strategy.md round 2 + round 3)."""
    return F.alive & (np.abs(F.bx - F.x) < HALF_POLE)


@concept("state")
def tip_behind_ball(F):
    """Near the ball with the pole tip behind it (tip x < ball x): hooked to PUSH the ball on, good even with
    the body slightly past (strategy.md round 3; the user's "0-89 deg" was stated attacking left, i.e. back-up
    in the attack frame — see the data note there)."""
    return near_ball(F) & (F.tipx < F.bx)


@concept("state")
def tip_past_ball(F):
    """Near the ball with the tip beyond it (tip x >= ball x): can only pull the ball back = effectively the
    dreaded situation (strategy.md round 3)."""
    return near_ball(F) & (F.tipx >= F.bx)


def situation(F):
    """analysis/situations.py classes from relative x (strategy.md "Recognising situations"): 0 right-far,
    1 right-close, 2 A (you-opp-ball), 3 B (opp-you-ball), 4 both-wrong, 5 opp-out; -1 = I'm out / not live."""
    me, opp, ball = F.x, F.ox, F.bx
    c = np.full(me.shape, -1)
    c[me < ball - HALF_POLE] = 0
    c[(me >= ball - HALF_POLE) & (me < ball)] = 1
    past = me >= ball
    c[past & (opp > ball) & (opp < me)] = 2
    c[past & (opp >= me)] = 3
    c[past & (opp <= ball)] = 4
    c[~F.opp_alive] = 5
    c[~F.alive] = -1
    return c


@concept("state")
def sit_right_far(F):
    """Good side with room: more than half a pole behind the ball — can vault straight into it."""
    return situation(F) == 0


@concept("state")
def sit_right_close(F):
    """Good side but within half a pole of the ball: time to back up (strategy.md round 2)."""
    return situation(F) == 1


@concept("state")
def sit_A(F):
    """Situation A, you-opp-ball: past the ball with the opponent between me and it, on his good side."""
    return situation(F) == 2


@concept("state")
def sit_B(F):
    """Situation B, opp-you-ball ("you're fucked"): past the ball, between it and the opponent."""
    return situation(F) == 3


@concept("state")
def sit_both_wrong(F):
    """Both riders past the ball (from my side's view)."""
    return situation(F) == 4


@concept("state")
def opp_out(F):
    """Kill window: I'm in play, the opponent is not (killed and waiting 3 s to respawn, or his 0.3 s spawn
    ghost). "Kill: very often guarantees a goal" (strategy.md round 2)."""
    return F.alive & ~F.opp_alive


@concept("state")
def opp_wrong_side(F):
    """The opponent is past the ball from his point of view (he attacks -x: ball x > his x) — what tip
    peeking and the throw aim for ("GG")."""
    return F.opp_alive & (F.ox < F.bx)


@concept("state")
def airborne(F):
    """Feet off everything (the game's foot sensor: not on the ground or on a pole). Riders pogo on the pole,
    so this is true most of the time (farm2 ~72%)."""
    return F.alive & ~F.ground


@concept("state")
def pole_planted(F):
    """Pole tip on the floor (tip less than PLANTED_H = 0.35 m above it): the slingshot is loading."""
    return F.alive & F.planted


@concept("state")
def in_pit(F):
    """Body in a pit (|x| > 16.3 m and below plateau level, as h2h_breakdown.py)."""
    return F.alive & (np.abs(F.x) > PIT_X) & (F.y < 0)


@concept("state")
def pole_fwd_down(F):
    """Pole quadrant -90..0 deg (attack frame): forward and down — the normal low carry / planting."""
    return F.alive & (F.ang < 0)


@concept("state")
def pole_fwd_up(F):
    """Pole quadrant 0..90 deg: forward and up."""
    return F.alive & (F.ang >= 0) & (F.ang < 90)


@concept("state")
def pole_back_up(F):
    """Pole quadrant 90..180 deg: back and up (near the ball: the hooked-to-push pole)."""
    return F.alive & (F.ang >= 90) & (F.ang < 180)


@concept("state")
def pole_back_down(F):
    """Pole quadrant 180..270 deg: back and down."""
    return F.alive & (F.ang >= 180)


@concept("state")
def backswing_in_flight(F):
    """Airborne, tip off the floor, pole rotating backward (counter-clockwise in the attack frame, > 1 rad/s):
    the user's trick that prolongs a jump and fine-tunes its height (strategy.md "Normal play")."""
    return airborne(F) & ~F.planted & (F.w > 1.0)


@concept("state")
def pole_on_opp(F):
    """My pole (outer 80%, straight body->tip line) within 1 m of the opponent's body centre: contact /
    pressing proxy (strategy.md situation B "keep your pole on him")."""
    d, _ = F.pole_to_opp()
    return F.opp_alive & (d < 1.0)


@concept("state")
def opp_pole_on_me(F):
    """The opponent's pole within 1 m of my body: he presses me (situation B "the opponent presses you down")."""
    d, _ = F.opp_pole_to_me()
    return F.opp_alive & (d < 1.0)


@concept("state")
def pole_over_opp_head(F):
    """My pole passes over the opponent: at his x (between 25% and 105% of my pole's length out) the pole line
    is 0.6-2.5 m above his body centre. Situation A option 2: "put your pole over his head: threatens a kill
    and limits his movement" (kills are stomps: the upper rider wins)."""
    dx = F.tipx - F.x
    ok = np.abs(dx) > 0.5                                    # not a vertical pole (no height at his x)
    u = np.where(ok, (F.ox - F.x) / np.where(ok, dx, 1.0), -1.0)
    clear = F.y + u * (F.tipy - F.y) - F.oy
    return F.opp_alive & ok & (u >= 0.25) & (u <= 1.05) & (clear > 0.6) & (clear < 2.5)


@concept("state")
def opp_on_my_pole(F):
    """The opponent stands on my pole: his foot sensor is down (on_ground), he is more than 1 m above the
    floor, his body is within 1.5 m of my pole line and above it. The setup of the throw (strategy.md "From
    above — the throw"). (On_ground at height is ~90% someone's pole: checked against full snapshots.)"""
    d, u = F.pole_to_opp()
    line_y = F.y + u * (F.tipy - F.y)
    return F.opp_alive & F.oground & (F.oh > 1.0) & (d < 1.5) & (F.oy > line_y)


@concept("state")
def on_opp_pole(F):
    """I stand on the opponent's pole (mirror of opp_on_my_pole): he can throw me."""
    d, u = F.opp_pole_to_me()
    line_y = F.oy + u * (F.otipy - F.oy)
    return F.opp_alive & F.ground & (F.h > 1.0) & (d < 1.5) & (F.y > line_y)


@concept("state")
def tip_peek(F):
    """Tip peeking: the opponent and I are on opposite sides of the ball and my tip is past the ball on his
    side, between the ball and him and at most half a pole past the ball — the order "opp-(our tip)-ball-us"
    (strategy.md "Ball handling")."""
    sb = F.ox - F.bx
    return (F.opp_alive & ((F.x - F.bx) * sb < 0) & ((F.tipx - F.bx) * sb > 0)
            & (np.abs(F.tipx - F.bx) <= np.minimum(np.abs(sb), HALF_POLE)))


@concept("state")
def rhythmic_pressing(F):
    """Situation B defence: in B now, my pole on the opponent in >= half of the last 2 s, and my pole's
    rotation reversed >= 4 times in them (|w| > 0.5 rad/s; >= 1 Hz push/pull; the user eyeballed 2-3 Hz)
    — "alternate pressing and releasing" (strategy.md situation B + round 2 timing)."""
    W = 2 * HZ
    on = _window_count(pole_on_opp(F), W, F.same)
    sgn = np.where(F.w > 0.5, 1, np.where(F.w < -0.5, -1, 0))
    flips = np.zeros(F.w.shape, np.int32)
    last = np.zeros(F.w.shape[1:], np.int32)
    for t in range(len(sgn)):
        last = np.where(F.same[t], last, 0)
        flips[t] = (sgn[t] != 0) & (last != 0) & (sgn[t] != last)
        last = np.where(sgn[t] != 0, sgn[t], last)
    return sit_B(F) & (on >= W // 2) & (_window_count(flips > 0, W, F.same) >= 4)


# ---------------------------------------------------------------- event concepts (int16 arrays: length at the end step)

def _run_length(b, same):
    """r[t] = how many steps b has held up to t (within a contiguous segment)."""
    r = np.zeros(b.shape, np.int32)
    prev = np.zeros(b.shape[1:], np.int32)
    for t in range(len(b)):
        prev = np.where(b[t], np.where(same[t], prev, 0) + 1, 0)
        r[t] = prev
    return r


def _age(b, same, cap=10 ** 6):
    """a[t] = steps since b was last true (0 = now; cap if never in this segment)."""
    a = np.zeros(b.shape, np.int32)
    prev = np.full(b.shape[1:], cap, np.int32)
    for t in range(len(b)):
        prev = np.where(b[t], 0, np.where(same[t], np.minimum(prev + 1, cap), cap))
        a[t] = prev
    return a


def _window_count(b, W, same):
    """How many of the last W steps (incl. t, within the segment) b was true."""
    T = len(b)
    start = np.zeros(b.shape, np.int64)  # first row of the segment each row belongs to
    for t in range(1, T):
        start[t] = np.where(same[t], start[t - 1], t)
    cs = np.concatenate([np.zeros((1,) + b.shape[1:], np.int64), np.cumsum(b, 0)])  # cs[i] = sum of b[:i]
    lo = np.maximum(np.arange(T).reshape((T,) + (1,) * (b.ndim - 1)) - W + 1, start)
    return (cs[1:] - np.take_along_axis(cs, lo, 0)).astype(np.int32)


def _onset(b, same):
    """b turns true at t (was false at t-1 in the same segment)."""
    o = np.zeros(b.shape, bool)
    o[1:] = b[1:] & ~b[:-1] & same[1:]
    return o


def _ev(mask, length=1):
    return np.where(mask, length, 0).astype(np.int16)


@concept("event")
def goal_for(F):
    """My side scored (my score went up)."""
    m = np.zeros(F.x.shape, bool)
    m[1:] = F.same[1:] & (F.O[1:, ..., MY_SCORE] > F.O[:-1, ..., MY_SCORE])
    return _ev(m)


@concept("event")
def goal_against(F):
    """The opponent scored."""
    m = np.zeros(F.x.shape, bool)
    m[1:] = F.same[1:] & (F.O[1:, ..., OPP_SCORE] > F.O[:-1, ..., OPP_SCORE])
    return _ev(m)


@concept("event")
def kill(F):
    """I killed the opponent: he drops out of play with the 3 s death timer (respawn_in > 0.5). In 1v1 only
    rider contacts set it (deleter falls and post-goal respawns give <= 0.1); matches the game's kill stats
    186/186 in farm2 self-play. "Kill: very often guarantees a goal"."""
    o = _onset(F.O[..., OPP + GHOST] > 0.5, F.same) & (F.O[..., OPP + RESPAWN] > 0.5)
    o[1:] &= F.alive[:-1]  # I was alive just before (not a double kill)
    return _ev(o)


@concept("event")
def death(F):
    """I was killed (my 3 s death timer started)."""
    return _ev(_onset(F.O[..., GHOST] > 0.5, F.same) & (F.O[..., RESPAWN] > 0.5))


@concept("event")
def back_pass(F):
    """Backward swing over the top: the pole passes vertical going backward (angle crosses 90 deg upward,
    counter-clockwise in the attack frame), as analysis/pole_rotation.py. The rotation all agents struggle
    with (HANDOFF farm3)."""
    a0, a1 = F.ang[:-1], F.ang[1:]
    m = np.zeros(F.x.shape, bool)
    m[1:] = F.same[1:] & F.alive[1:] & F.alive[:-1] & (a0 < 90) & (a1 >= 90) & (a1 - a0 < 60)
    return _ev(m)


@concept("event")
def fwd_pass(F):
    """Forward swing over the top (angle crosses 90 deg downward): every normal vault stride does one."""
    a0, a1 = F.ang[:-1], F.ang[1:]
    m = np.zeros(F.x.shape, bool)
    m[1:] = F.same[1:] & F.alive[1:] & F.alive[:-1] & (a0 >= 90) & (a1 < 90) & (a0 - a1 < 60)
    return _ev(m)


def _plant_phases(F):
    """Per planted phase ending at t (the tip lifts at t after >= 2 planted steps): its length, whether it
    began while airborne and falling (vy <= -3 m/s), whether the body touched down (on_ground) during it, the
    max body vy in it, and the body vy when it began. Arrays [T, ...]; valid where `end`."""
    P = pole_planted(F)
    shp = F.x.shape[1:]
    n = np.zeros(shp, np.int32); fall = np.zeros(shp, bool); touched = np.zeros(shp, bool)
    vmax = np.zeros(shp, np.float32); v0 = np.zeros(shp, np.float32)
    end = np.zeros(F.x.shape, bool); L = np.zeros(F.x.shape, np.int32)
    FALL = np.zeros(F.x.shape, bool); TOUCH = np.zeros(F.x.shape, bool)
    VMAX = np.zeros(F.x.shape, np.float32); V0 = np.zeros(F.x.shape, np.float32)
    for t in range(len(P)):
        cont = F.same[t] & (n > 0)
        e = cont & ~P[t] & F.alive[t] & (n >= 2)
        end[t], L[t], FALL[t], TOUCH[t], VMAX[t], V0[t] = e, n, fall, touched, vmax, v0
        start = P[t] & ~cont
        n = np.where(P[t], np.where(cont, n + 1, 1), 0)
        fall = np.where(start, ~F.ground[t] & (F.vy[t] <= -3.0), fall)
        touched = np.where(start, F.ground[t], touched | (P[t] & F.ground[t]))
        v0 = np.where(start, F.vy[t], v0)
        vmax = np.where(start, F.vy[t], np.maximum(vmax, np.where(P[t], F.vy[t], -1e9)))
    return end, L, FALL, TOUCH, VMAX, V0


@concept("event")
def vault(F):
    """A slingshot launch: the tip leaves the floor after a planted phase (>= 2 steps) in which the body rose
    at >= 6 m/s, and the body is airborne at release (strategy.md "Vaulting = a slingshot"). Marked at the
    release; length = the planted phase. (farm2 vaults reach 10-15 m/s.)"""
    end, L, FALL, TOUCH, VMAX, V0 = _plant_phases(F)
    return _ev(end & (VMAX >= 6.0) & ~F.ground, L + 1)


@concept("event")
def pogo_vault(F):
    """Landing on the pole to store energy: a vault (see `vault`) whose planted phase began while airborne and
    falling at >= 3 m/s and in which the body never touched down — the fall was caught by the pole and turned
    into the next jump (strategy.md "Landing: store kinetic + gravitational energy ... -> bigger next jump").
    The robust proxy we have: the pole's bend is not in the observation."""
    end, L, FALL, TOUCH, VMAX, V0 = _plant_phases(F)
    return _ev(end & (VMAX >= 6.0) & ~F.ground & FALL & ~TOUCH, L + 1)


@concept("event")
def jump_over_ball(F):
    """Situation A option 1, the long jump over the ball: I was in A within the last 2 s, and now, airborne
    and moving backward (vx < -1 m/s, so I moved, not the ball), I cross to the good side and stay there for
    0.5 s. Marked when confirmed (crossing + 0.5 s); length from the last A moment."""
    K = HZ // 2
    gs = good_side(F)
    ageA = _age(sit_A(F), F.same)
    cross = _onset(gs, F.same) & airborne(F) & (F.vx < -1.0) & (ageA <= 2 * HZ)
    r = _run_length(gs, F.same)
    m = np.zeros(F.x.shape, bool); L = np.zeros(F.x.shape, np.int32)
    m[K - 1:] = cross[: len(cross) - K + 1] & (r[K - 1:] == K)
    L[K - 1:] = ageA[: len(cross) - K + 1] + K
    return _ev(m, L)


@concept("event")
def pushed_opp_wrong_side(F):
    """I pushed the opponent to the wrong side: he had been on his good side for >= 0.5 s, crosses past the
    ball (from his view) while my pole touched him or he stood on it in the last 1 s, and stays wrong (or
    dies) for 1 s. "makes him miss and land on the wrong side ... GG" (strategy.md "Ball handling"). Marked
    when confirmed; length from the first contact step considered."""
    K = HZ
    wrong = F.opp_alive & (F.ox < F.bx)
    good = F.opp_alive & (F.ox >= F.bx)
    contact = _age(pole_on_opp(F) | opp_on_my_pole(F), F.same)
    rg = _run_length(good, F.same)
    cross = np.zeros(F.x.shape, bool)
    cross[1:] = F.same[1:] & wrong[1:] & (rg[:-1] >= HZ // 2) & (contact[1:] <= HZ)
    hold = _run_length(wrong | (F.alive & ~F.opp_alive), F.same)
    m = np.zeros(F.x.shape, bool)
    m[K - 1:] = cross[: len(cross) - K + 1] & (hold[K - 1:] >= K)
    return _ev(m, 2 * K)


@concept("event")
def tip_got_behind_ball(F):
    """Near the ball my tip went from past the ball to behind it and stays 0.33 s: re-hooked to push
    (strategy.md round 3; also the goal of the B defence for the opponent's tip). Marked when confirmed."""
    K = 5
    beh, past = tip_behind_ball(F), tip_past_ball(F)
    start = np.zeros(F.x.shape, bool)
    start[1:] = F.same[1:] & beh[1:] & past[:-1]
    r = _run_length(beh, F.same)
    m = np.zeros(F.x.shape, bool)
    m[K - 1:] = start[: len(start) - K + 1] & (r[K - 1:] == K)
    return _ev(m, K + 1)


@concept("event")
def catapult(F):
    """The throw: the opponent stood on my pole for >= 0.2 s, and within 0.5 s of leaving it he flies, airborne,
    at >= 8 m/s and >= 4 m/s faster than while on it (I flung him; riders also just bounce off a pole) —
    strategy.md "From above — the throw: when the opponent steps on our pole we fling it and he flies".
    Marked at the first step the launch speed is reached; length from when he left the pole."""
    on = opp_on_my_pole(F)
    sp = np.hypot(F.ovx, F.ovy)
    shp = F.x.shape[1:]
    run = np.zeros(shp, np.int32); left = np.full(shp, 99, np.int32); v_on = np.zeros(shp, np.float32)
    m = np.zeros(F.x.shape, bool); L = np.zeros(F.x.shape, np.int32)
    for t in range(len(on)):
        cont = F.same[t]
        run = np.where(cont, run, 0); left = np.where(cont, left, 99)
        leave = ~on[t] & (run >= 3)                       # just stepped off after >= 3 steps on it
        left = np.where(on[t], 99, np.where(leave, 1, left + 1))
        v_on = np.where(on[t], sp[t], v_on)
        hit = (left <= HZ // 2) & F.opp_alive[t] & ~F.oground[t] & (sp[t] >= 8.0) & (sp[t] >= v_on + 4.0)
        m[t], L[t] = hit, left
        left = np.where(hit, 99, left)                    # one per throw
        run = np.where(on[t], run + 1, 0)
    return _ev(m, L)


def _recent(b, k, same):
    """b was true at some step in (t-k, t]."""
    return _age(b, same) < k


@concept("event")
def peek_touch(F):
    """Tip peeking from below pays off: while peeking (my tip between the ball and the opponent), my tip
    touches him (tip within 1 m of his body) as he is airborne going at the ball — "a slight touch on the
    opponent as he vaults at the ball makes him miss" (strategy.md "Ball handling"). Marked at the touch."""
    near = np.hypot(F.tipx - F.ox, F.tipy - F.oy) < 1.0
    m = tip_peek(F) & near & ~F.oground
    return _ev(_onset(m, F.same))


# ---------------------------------------------------------------- user notes 2026-10-02 (strategy.md): commitment,
# agency, energy, hitting the ball second. Touch-based ones need Frames(touch=...) (native env only).

def _no_touch(F):
    return np.zeros(F.x.shape, bool)


def takeoff_mask(F):
    """A jump starts at t: airborne now, on the ground / a pole the step before (same rider in play)."""
    m = np.zeros(F.x.shape, bool)
    m[1:] = F.same[1:] & F.alive[1:] & F.alive[:-1] & ~F.ground[1:] & F.ground[:-1]
    return m


@concept("state")
def committed(F):
    """Committed (no agency): airborne, pole tip off the floor, and touching nothing that matters — not the
    ball this step, my pole not on the opponent, his pole not on me, not standing on each other's poles. "In the
    air you can barely affect the outcome until you touch something (a pole swing only bends the trajectory a
    little)" (strategy.md 2026-10-02 "Agency"). Without touch rows the ball part is the ball > 1 m from my
    body and tip."""
    if F.touch is not None:
        ball = F.touched
    else:
        ball = (np.hypot(F.bx - F.x, F.by - F.y) < 1.0) | (np.hypot(F.bx - F.tipx, F.by - F.tipy) < 1.0)
    return (airborne(F) & ~F.planted & ~ball & ~pole_on_opp(F) & ~opp_pole_on_me(F)
            & ~opp_on_my_pole(F) & ~on_opp_pole(F))


@concept("state")
def at_mercy(F):
    """At his mercy: committed (above) while the opponent can act — on the ground / a pole or with his tip planted
    — and is within reach (REACH = 5.5 m, a pole) of the ball or of me: he can move the ball so I miss, kill
    me, or hit the ball right after me (strategy.md 2026-10-02 "Agency")."""
    can_act = F.oground | (F.otip_h < PLANTED_H)
    near = (np.hypot(F.ox - F.bx, F.oy - F.by) < REACH) | (np.hypot(F.ox - F.x, F.oy - F.y) < REACH)
    return committed(F) & F.opp_alive & can_act & near


E_HIGH = 165.0  # J/kg: farm2 self-play quartiles 35 / 100 / 165 (16 matches, 2026-10-02) -> its top quarter


@concept("state")
def high_energy(F):
    """Total energy (Frames.energy = 0.5 v^2 + G * height above standing, J/kg; the pole's bend not included)
    at least E_HIGH: "high total energy is generally good (rough guide)" (strategy.md 2026-10-02 "Energy")."""
    return F.alive & (F.energy >= E_HIGH)


@concept("event")
def takeoff(F):
    """Every jump: the feet leave the ground / a pole (the game's foot sensor). Its speed is the body speed one
    step later (Frames.speed); analysis/concept_report.py bins the value by it ("if you jump, faster is better")."""
    return _ev(takeoff_mask(F))


V_FAST = 14.0  # m/s: farm2 self-play take-off speed quartiles 9.0 / 11.9 / 14.1 (16 matches) -> its top quarter


def takeoff_speed(F):
    """Body speed one step after the take-off step (the launch is often finished by the pole then); nan elsewhere."""
    sp = np.full(F.x.shape, np.nan, np.float32)
    m = takeoff_mask(F)
    nxt = np.zeros_like(F.speed); nxt[:-1] = F.speed[1:]
    ok = np.zeros(F.x.shape, bool); ok[:-1] = F.same[1:]
    sp[m] = np.where(ok, np.maximum(F.speed, nxt), F.speed)[m]
    return sp


@concept("event")
def fast_takeoff(F):
    """A jump launched at >= V_FAST m/s (takeoff_speed): "if you jump, faster is better" (2026-10-02 "Energy")."""
    return _ev(takeoff_mask(F) & (np.nan_to_num(takeoff_speed(F)) >= V_FAST))


def _lunge(F, pit_only=False):
    m = takeoff_mask(F)
    ahead = F.ox - F.x
    pre = (F.opp_alive & (ahead > 0.5) & (ahead < 8.0)          # he is in front of me, within a jump
           & (F.bx < F.x + 1.0)                                   # the ball behind me or about over me
           & (F.vx >= 4.0))                                       # and I fly at him
    if pit_only:
        pre &= F.x < -PIT_X
    return m & pre


@concept("event")
def lunge_at_opp(F):
    """Jumping at the opponent from in front of the ball: a take-off toward him (vx >= 4 m/s in my attack direction)
    while he is 0.5-8 m ahead of me and the ball is behind me or about over me (ball x < my x + 1 m) — the
    user's [ball, you, opp] (attacking ->) and the deep-defence version (in my hole with the ball above me, he
    waits in front of it). "Hoping they catch you with their pole" (strategy.md 2026-10-02 "No action is an
    action"). The commitment; whether it backfires is `overcommit`."""
    return _ev(_lunge(F))


@concept("event")
def lunge_from_pit(F):
    """lunge_at_opp from my own hole (body in the pit in front of the goal I defend, x < -16.3 m)."""
    return _ev(_lunge(F, pit_only=True))


def _overcommit(F):
    """(event mask, opponent-still mask, length): within 1.5 s of a lunge I end up past the opponent with him
    between me and the ball (situation A: ball < opp < me) — the flip to [ball, opp, you far]."""
    lunge = _lunge(F)
    A = situation(F) == 2
    still = F.opp_alive & (F.ospeed < 2.0) & (np.abs(F.O[..., OPP + PW]) < 1.5)
    shp = F.x.shape[1:]
    age = np.full(shp, 999, np.int32); n_still = np.zeros(shp, np.int32)
    m = np.zeros(F.x.shape, bool); M_STILL = np.zeros(F.x.shape, bool); L = np.zeros(F.x.shape, np.int32)
    for t in range(len(A)):
        age = np.where(F.same[t], age + 1, 999)
        n_still = np.where(F.same[t], n_still + still[t], 0)
        start = lunge[t]
        age = np.where(start, 0, age); n_still = np.where(start, still[t].astype(np.int32), n_still)
        hit = (age > 0) & (age <= 3 * HZ // 2) & A[t] & F.alive[t]
        m[t], L[t] = hit, age + 1
        M_STILL[t] = hit & (n_still >= 0.8 * (age + 1))   # he was nearly still for >= 80% of my flight
        age = np.where(hit, 999, age)                     # one per lunge
    return m, M_STILL, L


@concept("event")
def overcommit(F):
    """The lunge backfired: within 1.5 s of `lunge_at_opp` I am past the opponent with him between me and the
    ball (situation A) — he didn't need to catch me: "the situation becomes [ball, opp, you far] — so much tempo
    it's basically worth a goal" (strategy.md 2026-10-02 "No action is an action"). Marked at the flip."""
    m, _, L = _overcommit(F)
    return _ev(m, L)


@concept("event")
def overcommit_opp_still(F):
    """`overcommit` while the opponent "did nothing": for >= 80% of my flight his body speed was < 2 m/s and his
    pole turned at < 1.5 rad/s. For a human, not touching the controller."""
    _, ms, L = _overcommit(F)
    return _ev(ms, L)


def _fling(on, sp, alive, ground, same):
    """Generic throw detector (catapult): a rider stood on a pole (`on`) for >= 3 steps and, within 0.5 s of
    leaving it, flies airborne at >= 8 m/s and >= 4 m/s faster than while on it. (event mask, length)."""
    shp = on.shape[1:]
    run = np.zeros(shp, np.int32); left = np.full(shp, 99, np.int32); v_on = np.zeros(shp, np.float32)
    m = np.zeros(on.shape, bool); L = np.zeros(on.shape, np.int32)
    for t in range(len(on)):
        cont = same[t]
        run = np.where(cont, run, 0); left = np.where(cont, left, 99)
        leave = ~on[t] & (run >= 3)
        left = np.where(on[t], 99, np.where(leave, 1, left + 1))
        v_on = np.where(on[t], sp[t], v_on)
        hit = (left <= HZ // 2) & alive[t] & ~ground[t] & (sp[t] >= 8.0) & (sp[t] >= v_on + 4.0)
        m[t], L[t] = hit, left
        left = np.where(hit, 99, left)
        run = np.where(on[t], run + 1, 0)
    return m, L


@concept("event")
def got_flung(F):
    """I was thrown off the opponent's pole: the mirror of `catapult` (I stood on his pole >= 0.2 s and within
    0.5 s of leaving it I fly at >= 8 m/s, >= 4 m/s faster than on it). "When you land on the opponent's pole
    they can fling you" (strategy.md 2026-10-02 "Agency")."""
    m, L = _fling(on_opp_pole(F), F.speed, F.alive, F.ground, F.same)
    return _ev(m, L)


@concept("event")
def flung_own_goal(F):
    """The worst case of being flung: `got_flung`, then I touch the ball within 1.5 s, then within 3 s of that touch
    the opponent's team scores with the game crediting the ball to me (an own goal: PlayState::lastTouch = me) —
    "being flung into the ball and auto-scoring". Marked at the goal; needs touch rows."""
    if F.touch is None:
        return _ev(_no_touch(F))
    fl = got_flung(F) > 0
    age_fl = _age(fl, F.same)
    touch_after = F.touched & (age_fl <= 3 * HZ // 2)
    age_t = _age(touch_after, F.same)
    ga = goal_against(F) > 0
    return _ev(ga & (age_t <= 3 * HZ) & F.game_last, np.minimum(age_t + 1, 32000))


SECOND_W = 0.5  # s: a touch within this long after the other rider's counts as "second"


def _touch_onset(F):
    """I start touching the ball this step (no touch at any tick of the previous step)."""
    prev = np.zeros(F.x.shape, bool)
    prev[1:] = F.touched[:-1] & F.same[1:]
    return F.touched & ~prev


@concept("event")
def second_touch(F):
    """I hit the ball second: my touch begins while the opponent touched it within the last SECOND_W = 0.5 s,
    before me (his touch older than mine). "Hitting the ball second very often guarantees the best outcome"
    (strategy.md 2026-10-02). Needs touch rows."""
    if F.touch is None:
        return _ev(_no_touch(F))
    return _ev(_touch_onset(F) & F.opp_alive & (F.ot_age <= SECOND_W) & (F.ot_age > F.t_age))


@concept("event")
def second_kick(F):
    """I KICK the ball second (the game's kick impulse, from my body / pole handle) within 0.6 s of the
    opponent's kick: his kick turned off his ball collisions for 0.6 s (the ball glitches through him and his
    pole), mine sends it my way; I am pushed back by it (strategy.md 2026-10-02). Needs touch rows."""
    if F.touch is None:
        return _ev(_no_touch(F))
    return _ev(F.kicked & F.opp_alive & (F.ok_age <= 0.6) & (F.ok_age > F.k_age))


def _impale_plant(F):
    return _onset(pole_planted(F), F.same) & (F.ang >= 180) & (F.vx <= -6.0)


@concept("event")
def impale_plant(F):
    """Impaling the pole to brake (the attempt): flying backward (toward my own goal, vx <= -6 m/s in my attack
    frame — the way the second kicker is pushed) when the tip goes into the floor in the lower-back quadrant
    (pole angle 180-270 deg). "When flying the wrong way you can impale the pole into the ground (lower back
    quadrant) to shed energy quickly" (strategy.md 2026-10-02). `impale_brake` = it worked."""
    return _ev(_impale_plant(F))


@concept("event")
def impale_brake(F):
    """`impale_plant` that sheds speed: within 0.33 s of the plant my speed drops by >= 3 m/s and >= 30%. Marked
    when the drop is reached; length from the plant."""
    K = 5
    start = _impale_plant(F)
    shp = F.x.shape[1:]
    age = np.full(shp, 99, np.int32); v0 = np.zeros(shp, np.float32)
    m = np.zeros(F.x.shape, bool); L = np.zeros(F.x.shape, np.int32)
    for t in range(len(start)):
        age = np.where(F.same[t], age + 1, 99)
        age = np.where(start[t], 0, age); v0 = np.where(start[t], F.speed[t], v0)
        drop = v0 - F.speed[t]
        hit = (age <= K) & F.alive[t] & (drop >= 3.0) & (drop >= 0.3 * v0)
        m[t], L[t] = hit, age + 1
        age = np.where(hit, 99, age)
    return _ev(m, L)


# ---------------------------------------------------------------- user notes 2026-10-03 (t20 series): interlocks, letting
# go of the pole, finishing, the pole servo. Pole contact is rider-on-pole: the game's two poles never collide with
# each other (pole_ctor.cpp collision filter: team 0 pole category 0x8, mask 0xfe25; team 1 0x10 / 0xfe23 — neither
# mask has the other pole's bit), a pole only hits the other RIDER (body / foot), the ball and the terrain. So "our
# poles are interlocked" = each rider's pole rests on the other rider.

SERVO_DZ = 0.1     # per-axis dead zone of the game's sticks 0/1 (input.cpp: GamepadStick.deadZone)
SERVO_MIN = 0.2    # the pole servo works only above this stick length (player.cpp: `if (mag > 0.2f)`)


def servo_off(aim_x, aim_y):
    """The game's pole servo is OFF for this right-stick input: the hand joints get zero motor torque and the pole
    hangs free ("letting go of the pole stick"). The stick goes through Gamepad_setJoystickPosition (per-axis dead
    zone 0.1, rescaled to 0..1) and Player::Update drives the pole only if the result is longer than 0.2
    (sim/src/game/player.cpp pole servo): with the stick on one axis that is |v| <= 0.28. A policy's (aim_x, aim_y)
    is first scaled to length 1 when longer (PoleRidersEnv._stick): that never turns the servo off."""
    ax, ay = np.asarray(aim_x, np.float32), np.asarray(aim_y, np.float32)
    m = np.hypot(ax, ay)
    k = np.where(m > 1.0, 1.0 / np.maximum(m, 1e-9), 1.0)
    dz = lambda v: np.clip((np.abs(v) - SERVO_DZ) / (1.0 - SERVO_DZ), 0.0, 1.0)  # noqa: E731
    return np.hypot(dz(ax * k), dz(ay * k)) <= SERVO_MIN


@concept("state")
def mutual_pole_contact(F):
    """Each rider's pole touches the other rider: (my pole on him | he stands on my pole) & (his pole on me | I stand
    on his pole) — the moment-to-moment part of `pole_interlock` (pole_on_opp / opp_pole_on_me: straight body->tip
    line within 1 m of the body centre; checked against the real polylines in analysis/t20_vs_user_report)."""
    return ((pole_on_opp(F) | opp_on_my_pole(F)) & (opp_pole_on_me(F) | on_opp_pole(F)))


INTERLOCK_MIN = HZ        # steps: mutual contact held >= 1 s (gaps bridged) is an interlock
INTERLOCK_GAP = 5         # steps: contact gaps up to 1/3 s are bridged
INTERLOCK_LONG = 3 * HZ   # steps: "long" interlocks (the user's 5-10 s stand-offs)


def _interlock(F, c=None):
    """(interlocked bool[T, ...], end mark bool, mark length L, contact span D): a run of mutual_pole_contact (or the
    contact mask c) with gaps <= INTERLOCK_GAP bridged; interlocked once it has lasted INTERLOCK_MIN; it ends
    INTERLOCK_GAP + 1 steps after the last contact (the mark: no lookahead; first contact at t - L + 1, last contact at
    t - L + D)."""
    c = mutual_pole_contact(F) if c is None else np.asarray(c, bool)
    T = len(c)
    shp = c.shape[1:]
    start = np.full(shp, -1, np.int64); last = np.full(shp, -1, np.int64)
    inl = np.zeros(c.shape, bool); mark = np.zeros(c.shape, bool)
    L = np.zeros(c.shape, np.int32); D = np.zeros(c.shape, np.int32)
    for t in range(T):
        if t > 0:  # a segment break (match end) closes a run at the previous row
            brk = ~F.same[t] & (start >= 0)
            ok = brk & (last - start + 1 >= INTERLOCK_MIN)
            mark[t - 1] |= ok
            L[t - 1] = np.where(ok, t - start, L[t - 1]); D[t - 1] = np.where(ok, last - start + 1, D[t - 1])
            start = np.where(~F.same[t], -1, start); last = np.where(~F.same[t], -1, last)
        ct = c[t]
        start = np.where(ct & (start < 0), t, start)
        last = np.where(ct, t, last)
        ended = (start >= 0) & ~ct & (t - last > INTERLOCK_GAP)
        ok = ended & (last - start + 1 >= INTERLOCK_MIN)
        mark[t] = ok
        L[t] = np.where(ok, t - start + 1, 0); D[t] = np.where(ok, last - start + 1, 0)
        start = np.where(ended, -1, start); last = np.where(ended, -1, last)
        inl[t] = (start >= 0) & (t - start + 1 >= INTERLOCK_MIN) & F.alive[t]
    return inl, mark, L, D


@concept("state")
def interlocked(F):
    """Interlocked: mutual_pole_contact has held for >= 1 s (INTERLOCK_MIN; gaps <= 1/3 s bridged), up to 1/3 s after
    the last contact. User 2026-10-03: "we're interlocked with poles and stay like that for 5-10 seconds ... troll:
    you don't know what will happen, so the winning player is gambling their position; for the losing player it can be
    a hail mary". In the real logs the long ones are mostly one rider standing on the other's upright pole with his own
    pole pointing down onto him (`stacked` in analysis/t20_vs_user.py) or both pressing each other on the ground."""
    return _interlock(F)[0]


@concept("event")
def pole_interlock(F):
    """An interlock (see `interlocked`) ended: marked INTERLOCK_GAP + 1 steps after the last mutual contact (no
    lookahead); length = from the first contact to the mark (the contact span is length - INTERLOCK_GAP - 1 steps).
    Who was winning at its start and what followed: interlock_episodes() / analysis/t20_vs_user.py."""
    _, mark, L, _ = _interlock(F)
    return _ev(mark, L)


@concept("state")
def interlock_let_go(F):
    """Interlocked with my pole servo off (servo_off: the right stick near the centre, the pole hangs free). Needs act
    rows (Frames(act=...)); all false without them."""
    if F.act is None:
        return _no_touch(F)
    return interlocked(F) & F.released


LETGO_W = HZ // 2  # steps: the release began at most 0.5 s before the interlock's last contact


def _let_go(F, released):
    """Mask at pole_interlock marks: the pole servo (`released`) was off at the interlock's last contact or the step
    after, in a release of >= 2 steps that began during the interlock and at most LETGO_W steps before its last contact
    — letting go ended it. (A rider who was limp all along, released before the interlock began, is not counted; nor is
    a release that was taken back before the end.)"""
    _, mark, L, D = _interlock(F)
    out = np.zeros(mark.shape, bool)
    if released is None:
        return out
    rl = _run_length(released, F.same)
    T = len(rl)
    for idx in zip(*np.nonzero(mark)):
        t, rest = idx[0], idx[1:]
        s = t - L[idx] + 1
        e = s + D[idx] - 1
        for te in (e, e + 1):
            if te >= T or te > t or not released[(te,) + rest]:
                continue
            n = int(rl[(te,) + rest])
            r0 = te - n + 1
            longer = n >= 2 or (te + 1 <= t and te + 1 < T and released[(te + 1,) + rest])
            if max(s, e - LETGO_W) <= r0 <= e and longer:
                out[idx] = True
                break
    return out


@concept("event")
def let_go_escape(F):
    """Letting go to escape an interlock: the interlock ended (pole_interlock) with my pole servo off — released at its
    last contact or the step after, in a release of >= 2 steps that began during the interlock and at most 0.5 s
    (LETGO_W) before the last contact (a rider limp all along does not count). User 2026-10-03: "just letting go can
    give the attacking player (usually the one in the air in this configuration) a clean tempo to kick the ball before
    the enemy can recover, because he gets flung kind of randomly while I gracefully fall into an immediate jump".
    Marked with pole_interlock (same length). Needs act rows."""
    if F.act is None:
        return _ev(_no_touch(F))
    _, mark, L, _ = _interlock(F)
    return _ev(_let_go(F, F.released), L)


OPEN_DX = REACH   # m: the opponent this far on his wrong side of the ball can't reach it with his tip
OPEN_AFTER_GOAL = 2 * HZ  # steps: no open goal in the 2 s after a goal (both respawn ~1.4 s after it)


@concept("state")
def open_goal(F):
    """Open goal: I'm in play on the good side of the ball and the opponent can't defend — he is out of play (dead /
    respawning, opp_out) or on his wrong side of the ball by more than OPEN_DX = 5.5 m (a pole's reach: his tip can't
    get to it); not in the 2 s after a goal. User 2026-10-03: "his finishing (when I'm dead or terribly out of
    position and he just needs to score) is pretty bad". Conversion: `open_goal_conversion`."""
    g = (goal_for(F) > 0) | (goal_against(F) > 0)
    recent = _age(g, F.same) < OPEN_AFTER_GOAL
    far = F.opp_alive & (F.ox < F.bx - OPEN_DX)
    return F.alive & (F.bx > F.x) & ~recent & (~F.opp_alive | far)


OPEN_GRACE = HZ  # steps: a goal up to 1 s after the open goal closed still counts as its conversion


@concept("event")
def open_goal_conversion(F):
    """I scored from an open goal: a goal for me while `open_goal` held at the previous step or at most 1 s earlier
    (OPEN_GRACE); length = steps from the start of that open-goal window to the goal (time to convert). Conversion rate
    = these / open_goal entries (runs_of(open_goal(F), F.same))."""
    og = open_goal(F)
    age = _age(og, F.same)
    rl = _run_length(og, F.same)
    gf = goal_for(F) > 0
    m = np.zeros(F.x.shape, bool); L = np.zeros(F.x.shape, np.int32)
    for idx in zip(*np.nonzero(gf)):
        t, rest = idx[0], idx[1:]
        if t == 0:
            continue
        a = age[(t - 1,) + rest]
        if a > OPEN_GRACE:
            continue
        te = t - 1 - a
        on = te - rl[(te,) + rest] + 1
        m[idx], L[idx] = True, t - on + 1
    return _ev(m, np.minimum(L, 32000))


@concept("event")
def aim_back_pass(F):
    """My stick sweeps the pole aim backward over the top: the aimed angle (Frames.aim, attack frame) crosses 90 deg
    going up (counter-clockwise in the attack frame = backward), the servo on at both steps. The servo always turns the
    pole the SHORT way to the aimed angle (player.cpp: error wrapped to +-180 deg), so a full backward rotation needs
    the stick swept round backward ahead of the pole. User 2026-10-03 on t20 in [bot, user, ball]: "he doesn't want to
    move his pole clockwise" (screen clockwise for the rider attacking left = backward). Needs act rows."""
    if F.act is None:
        return _ev(_no_touch(F))
    a0, a1 = F.aim[:-1], F.aim[1:]
    on = ~F.released & F.alive
    m = np.zeros(F.x.shape, bool)
    m[1:] = F.same[1:] & on[1:] & on[:-1] & (a0 < 90) & (a1 >= 90) & (a1 - a0 < 90)
    return _ev(m)


# ---------------------------------------------------------------- episode helpers (analysis; not concepts)

def interlock_episodes(F):
    """One dict per interlock in F (any batch dims; `col` = the batch index): first / last contact step, the mark,
    duration (s, the contact span), from that view: attacker at the start (+1 me, -1 him, 0 neither: me on the good
    side and he on his wrong side or vice versa; both on their good sides -> the one whose attack half holds the ball),
    stacked (one rider on the other's pole for >= half of it: `top` +1 me / -1 him), my / his release (let-go) and the
    rows needed for outcomes (the caller adds goals / kicks after `last`)."""
    inl, mark, L, D = _interlock(F)
    mine_lg = _let_go(F, F.released if F.act is not None else None)
    his_lg = _let_go(F, F.oreleased if F.act is not None else None)
    onp, oon = on_opp_pole(F), opp_on_my_pole(F)
    out = []
    for idx in zip(*np.nonzero(mark)):
        t, rest = int(idx[0]), tuple(int(i) for i in idx[1:])
        s = t - int(L[idx]) + 1
        e = s + int(D[idx]) - 1
        at = lambda a, k=s: a[(k,) + rest]  # noqa: E731
        me_good, opp_good = at(F.bx) > at(F.x), at(F.ox) >= at(F.bx)
        if me_good and not opp_good:
            att = 1
        elif opp_good and not me_good:
            att = -1
        elif me_good and opp_good:
            att = 1 if at(F.bx) > 0 else -1
        else:
            att = 0
        sl = (slice(s, e + 1),) + rest
        top = 1 if onp[sl].mean() >= 0.5 else (-1 if oon[sl].mean() >= 0.5 else 0)
        out.append({"col": rest, "start": s, "last": e, "mark": t, "dur_s": (e - s + 1) / HZ, "attacker": att,
                    "top": top, "stacked": top != 0, "my_let_go": bool(mine_lg[idx]), "his_let_go": bool(his_lg[idx]),
                    "ball_x": float(at(F.bx)), "sit": int(situation(F)[(s,) + rest])})
    return out


def runs_of(b, same):
    """(start, end inclusive) of every run of b (bool[T, ...]) along time, per batch index: [(col, s, e)]."""
    rl = _run_length(b, same)
    endm = b.copy()
    endm[:-1] &= ~(b[1:] & same[1:])
    out = []
    for idx in zip(*np.nonzero(endm)):
        e = int(idx[0])
        out.append((tuple(int(i) for i in idx[1:]), e - int(rl[idx]) + 1, e))
    return out


# ---------------------------------------------------------------- whole-vocabulary helpers

STATE_NAMES = [n for n, (k, _) in CONCEPTS.items() if k == "state"]
EVENT_NAMES = [n for n, (k, _) in CONCEPTS.items() if k == "event"]


def state_concepts(F, names=None):
    return {n: CONCEPTS[n][1](F) for n in (names or STATE_NAMES)}


def event_concepts(F, names=None):
    return {n: CONCEPTS[n][1](F) for n in (names or EVENT_NAMES)}


def describe():
    """name, kind, first docstring sentence — the vocabulary at a glance."""
    return [(n, k, " ".join(fn.__doc__.split())) for n, (k, fn) in CONCEPTS.items()]


# ---------------------------------------------------------------- snapshots / play logs -> rows

_TERRAIN = None


def obs_from_states(states, frames=None):
    """Snapshots (NativeHost.json(k) / play-log "s" dicts, in time order, one per step) -> (O float[T, 2, 62]
    observe2 rows of rider 1 and rider 2, live bool[T, 2], same bool[T, 2]). Rows that are not a match in
    play are not live (the first game_over snapshot still is: it carries the deciding goal); contiguity breaks at a new match_id and, if `frames` (the snapshots' frame numbers) is
    given, at gaps other than 4 frames."""
    global _TERRAIN
    from farm_env import Terrain, observe2
    if _TERRAIN is None:
        _TERRAIN = Terrain(2)
    T = len(states)
    O = np.zeros((T, 2, 62), np.float32); live = np.zeros((T, 2), bool); same = np.zeros((T, 2), bool)
    prev_id, last_id, over = None, None, False
    for t, st in enumerate(states):
        if isinstance(st, str):
            st = json.loads(st)
        if (st or {}).get("match_id") != last_id:
            last_id, over = (st or {}).get("match_id"), False
        # the first game-over snapshot is kept (it carries the deciding goal), the celebration after it is not
        ok = bool(st and st.get("in_match") and len(st.get("players") or []) == 2 and not over)
        over = over or bool(st and st.get("game_over"))
        if ok:
            O[t, 0], O[t, 1] = observe2(st, 1, _TERRAIN), observe2(st, 2, _TERRAIN)
            live[t] = True
            cont = t > 0 and live[t - 1, 0] and st.get("match_id") == prev_id
            if frames is not None and t > 0:
                cont = cont and frames[t] - frames[t - 1] == 4
            same[t] = cont
            prev_id = st.get("match_id")
    return O, live, same


def playlog_read(path):
    """A play log -> {match_id: (frames int[N], states [N dicts], kinds of the last row)}: every in-match snapshot."""
    by_match = {}
    for line in open(path):
        try:
            r = json.loads(line)
        except ValueError:  # a line cut off when the game was closed
            continue
        st = r.get("s")
        if not st or not st.get("in_match"):
            continue
        by_match.setdefault(st.get("match_id"), []).append((r["f"], st, r.get("kind", [])))
    return {mid: (np.array([f for f, _, _ in rows]), [s for _, s, _ in rows], rows[-1][2])
            for mid, rows in by_match.items()}


def playlog_ticks(frames, states):
    """Game ticks (1/60 s) elapsed at each snapshot of one match, from the match clock (time_left).

    The bridge logs a snapshot every 4th RENDERED frame, and the real game renders at the display's rate: on a
    120 Hz display that is every ~2 game ticks (1/30 s), not 4 (the game ticks at a fixed 60 Hz; its clock
    drops 1/60 s per tick). While the clock stands (the hit-stop after kicks / kills, the spawn freeze, after the
    deciding goal) a snapshot counts as the median ticks per snapshot (real time). Returns (ticks float[N],
    brk bool[N]: a snapshot is missing before i, ticks per snapshot = the median)."""
    tl = np.array([s.get("time_left", 0.0) for s in states], np.float64)
    N = len(tl)
    fg = np.diff(frames).astype(np.float64)
    fmed = float(np.median(fg)) if N > 1 else 4.0
    d = np.round((tl[:-1] - tl[1:]) * 60)
    run = (d > 0) & (d <= 8 * fg / fmed)
    rate = float(np.median(d[run])) if run.sum() >= 10 else fmed  # fallback: one tick per frame
    inc = np.where(run, d, rate * fg / fmed)
    ticks = np.concatenate([[0.0], np.cumsum(inc)])
    brk = np.zeros(N, bool)
    brk[1:] = fg != fmed
    return ticks, brk, rate


def playlog_end(states, ticks, sudden_death=False):
    """Index of the last snapshot of play: the deciding goal (the clock jumps to 0 about 1 s after it; the log has
    no game_over flag: the slow-motion end phase is still `in_match`) or the end of time; None if the log ends first
    (an aborted match). A match tied at the end of time is a draw under the house rules (no sudden death, as in
    training): play ends with regulation time even when the game went on into sudden death; sudden_death=True =
    the old behaviour (the first goal in sudden death ends it)."""
    tl = np.array([s.get("time_left", 0.0) for s in states])
    sc = np.array([s.get("score", [0, 0]) for s in states])
    sd = np.array([bool(s.get("sudden_death")) for s in states])
    over = [i for i, s in enumerate(states) if s.get("game_over")]
    goal = np.zeros(len(states), bool)
    goal[1:] = np.any(sc[1:] != sc[:-1], 1)
    if sd.any() and sudden_death:
        g = np.nonzero(goal & (np.arange(len(states)) >= np.argmax(sd)))[0]
        return int(g[0]) if len(g) else (over[0] if over else None)
    z = np.nonzero(tl <= 0)[0]
    if not len(z):
        return over[0] if over else None
    e = int(z[0])
    g = np.nonzero(goal[:e + 1])[0]
    if e > 0 and tl[e - 1] - tl[e] > 1.0 and len(g) and ticks[e] - ticks[g[-1]] <= 3 * 60:
        return int(g[-1])  # the clock jumped: the match was decided by that goal
    return e


def playlog_resample(frames, states, step_ticks=4):
    """Pick one snapshot per decision step (4 game ticks = 1/15 s, as the sim / the bots) by the match clock, up to
    the end of play (playlog_end). Returns (idx int[T] into the snapshots, same bool[T]: row t continues row t-1,
    info dict)."""
    ticks, brk, rate = playlog_ticks(frames, states)
    end = playlog_end(states, ticks)
    last = len(states) - 1 if end is None else end
    tk = ticks[:last + 1]
    targets = np.arange(0.0, tk[-1] + 1e-9, step_ticks)
    j = np.clip(np.searchsorted(tk, targets), 1, len(tk) - 1) if len(tk) > 1 else np.zeros(len(targets), int)
    if len(tk) > 1:
        j = np.where(np.abs(tk[j - 1] - targets) <= np.abs(tk[j] - targets), j - 1, j)
    idx = np.unique(j)
    same = np.zeros(len(idx), bool)
    cb = np.cumsum(brk)
    same[1:] = (np.diff(tk[idx]) <= 2 * step_ticks) & (cb[idx[1:]] == cb[idx[:-1]])
    gaps = np.diff(tk[idx])
    info = {"ticks_per_snapshot": rate, "end": end, "aborted": end is None, "seconds": float(tk[-1] / 60),
            "missing_snapshots": int(brk[:last + 1].sum()),
            "step_ticks": {int(k): int(v) for k, v in zip(*np.unique(np.round(gaps), return_counts=True))}}
    return idx, same, info


def playlog_sticks(path, frames):
    """{frame: ax} for the wanted frames of a play log: ax = the four pads' sticks [lx, ly, rx, ry] (SDL: +y down) of
    the logged line that holds that snapshot."""
    want, out = set(int(f) for f in frames), {}
    with open(path) as fh:
        for line in fh:
            if '"ax"' not in line:
                continue
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("f") in want and "s" in r:
                out[r["f"]] = r["ax"]
    return out


def playlog_actions(path, frames, pads):
    """Action rows [T, 2 views (rider 1, 2), 2 (me, opp), 3] in the bots' terms (run, aim x, aim y in each rider's
    attack frame: rider 1 attacks +x, rider 2 is mirrored; analysis/playlog.py's mapping) from the sticks logged with
    the snapshots of `frames` (T frame numbers); pads = {rider id: pad index}. NaN where unknown."""
    ax = playlog_sticks(path, frames)
    A = np.full((len(frames), 2, 3), np.nan, np.float32)  # [T, rider, 3]
    for t, f in enumerate(frames):
        a = ax.get(int(f))
        if a is None:
            continue
        for r, sign in ((0, 1.0), (1, -1.0)):
            p = pads.get(r + 1)
            if p is not None and p < len(a):
                lx, ly, rx, ry = a[p]
                A[t, r] = sign * lx, sign * rx, -ry
    return np.stack([np.stack([A[:, 0], A[:, 1]], 1), np.stack([A[:, 1], A[:, 0]], 1)], 1)


def playlog_matches(path, resample=True, with_info=False):
    """A human play log (runs/human/*.jsonl, analysis/playlog.py) -> one (O [T, 2, 62], live, same, humans)
    per match, humans = bool[2]: which rider (1, 2) a person drove (pad kind 1 = real controller, 3 = a virtual pad
    mirroring one, 4 = keyboard; bridge/sprbridge.c playlog_frame). resample (default): one row per 4 game ticks
    by the match clock and cut at the end of play (playlog_resample; a 120 Hz display logs every 2 ticks);
    False = every logged snapshot (the old behaviour: rows are 1/30 s apart on a 120 Hz display). with_info:
    also yield (idx, info) of playlog_resample (idx into playlog_read's states)."""
    for mid, (frames, states, kinds) in playlog_read(path).items():
        if resample:
            idx, same_r, info = playlog_resample(frames, states)
            O, live, same = obs_from_states([states[i] for i in idx])
            same &= same_r[:, None]
        else:
            idx, info = np.arange(len(states)), {}
            O, live, same = obs_from_states(states, list(frames))
        pads = {p["id"]: p.get("pad") for p in states[-1]["players"]}
        humans = np.array([pads.get(i) is not None and pads[i] < len(kinds) and kinds[pads[i]] in (1, 3, 4)
                           for i in (1, 2)])
        yield (O, live, same, humans) + (((idx, info),) if with_info else ())


if __name__ == "__main__":
    for n, k, d in describe():
        print(f"{k:5s}  {n:22s} {d}")
