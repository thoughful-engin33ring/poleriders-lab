#!/usr/bin/env python3
"""Self-play RL environment for Super Pole Riders over sprbridge (no dependencies).

    env = PoleRidersEnv(frame_skip=4)
    env.start_match()                 # launch/drive the menus into a 1v1 match if needed
    obs = env.reset()                 # {1: [...], 2: [...]}  one vector per agent
    obs, rew, done, info = env.step({1: (run, aim_x, aim_y), 2: (run, aim_x, aim_y)})
    env.close()

Frames are agent-centric: each agent's view is mirrored so it always attacks toward +x,
and y points up. Units are Box2D meters (64 px = 1 m).

Action per agent, all in the agent's own frame:
    run    -1..1   left stick x (run backward / forward)
    aim_x, aim_y   right stick = the pole-angle setpoint the rider's arms servo toward
                   (vector; length is clamped to 1, direction is the target pole angle)
Reward (zero-sum between the agents; weights in PoleRidersEnv(reward=...)):
    goal          +1 per goal your team scores, -1 per goal conceded
    kill          +0.3 per opponent death (killed or suicide), -0.3 per own death —
                  a kill is worth about a third of a goal because it opens the field
    ball_progress potential-based shaping: w * change in ball x toward the opponent's end
                  wall (Vaulters score on the +x wall, Benders on the -x wall); skipped on
                  steps with a goal or a ball respawn. Default 0.003 per meter, so pushing
                  the ball across the whole ~40 m field is worth ~0.12 of a goal.
info["terms"][agent] has the unweighted goal / kill / ball_progress terms of each step.

The game runs in lockstep while the env is active: one `step` = frame_skip physics ticks of
exactly 1/60 s, as fast as the machine renders, independent of wall-clock time.
"""
import json, math, os, subprocess, sys, time

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spr import Bridge, SOCK  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
OBS_SIZE = 2 * 15 + 4 + 3  # me[0:15] opp[15:30] ball[30:34] time,score_me,score_opp[34:37]


def _body_feats(b, sign, ox=0.0, oy=0.0):
    """x, y (relative to origin), vx, vy, sin a, cos a, w — mirrored by sign, y up.

    Orientation is a rotation, not a direction: flipping y (Box2D is y-down) negates it, and
    mirroring x negates it again, so the agent-frame angle is -sign * a (NOT pi - a, which
    would show a mirrored upright rider upside down)."""
    phi = -sign * b["a"]
    return [sign * (b["x"] - ox), -(b["y"] - oy), sign * b["vx"], -b["vy"],
            math.sin(phi), math.cos(phi), -sign * b["w"]]


def absent(p):
    """The rider is not in play: the short post-spawn ghost, or dead (killed / fell into a deleter and
    waiting to respawn: PlayState::respawnTimer > 0; its corpse body stays in the world meanwhile)."""
    return bool(p.get("ghost") or p.get("dead"))


def respawn_in(p):
    """Seconds until the rider is back in play, / 3: the respawn countdown while dead, else the ghost
    timer (0 for a live rider)."""
    return (p.get("respawn_t", 0.0) if p.get("dead") else p.get("ghost_t", 0.0)) / 3.0


def _player_feats(p, sign):
    base, tip = p["pole"][0], p["pole"][-1]
    dx, dy = sign * (tip["x"] - base["x"]), -(tip["y"] - base["y"])
    ang = math.atan2(dy, dx)
    mid = p["pole"][len(p["pole"]) // 2]
    return (_body_feats(p, sign)
            + [float(p["on_pole"]), float(p["on_ground"]), float(absent(p))]
            + [math.sin(ang), math.cos(ang), -sign * mid["w"],
               sign * (tip["x"] - p["x"]), -(tip["y"] - p["y"])])


def observe(state, agent):
    """Flat observation vector (length OBS_SIZE) for agent 1 (Vaulters) or 2 (Benders)."""
    sign = 1 if agent == 1 else -1
    players = {p["id"]: p for p in state["players"]}
    me, opp = players[agent], players[3 - agent]
    ball = state.get("ball") or {"x": 0, "y": 0, "a": 0, "vx": 0, "vy": 0, "w": 0}
    b = _body_feats(ball, sign)[:4]
    my_score, opp_score = state["score"][agent - 1], state["score"][2 - agent]
    return (_player_feats(me, sign) + _player_feats(opp, sign) + b
            + [state["time_left"] / 180.0, float(my_score), float(opp_score)])


SOCK_DIR = os.path.expanduser("~/Library/Application Support/sportsfriends")


GAME_DIR = os.path.expanduser(
    "~/Library/Application Support/Steam/steamapps/common/Sportsfriends/Sportsfriends.app/Contents/MacOS")
BRIDGE = os.path.join(HERE, "sprbridge.dylib")


def launch_game(sock, mute=True, size=(640, 360), fullscreen=False, canon=True, extra_env=None):
    """Start one game with the bridge on `sock`. Returns the Popen; stdout/stderr go to
    <sock dir>/logs/<sock name>.log (the game's own log.txt is shared by all instances)."""
    os.makedirs(os.path.join(SOCK_DIR, "logs"), exist_ok=True)
    try:
        os.unlink(sock)
    except FileNotFoundError:
        pass
    log = open(os.path.join(SOCK_DIR, "logs", os.path.basename(sock) + ".log"), "a")
    log.write(f"\n===== launch {time.strftime('%Y-%m-%d %H:%M:%S')} =====\n")
    log.flush()
    env = dict(os.environ, SPRBRIDGE_SOCK=sock, SPRBRIDGE_MUTE="1" if mute else "0",
               SPRBRIDGE_NOMOVE="1", SPRBRIDGE_NOUNLOCK="1", SPRBRIDGE_CANON="1" if canon else "0",
               DYLD_INSERT_LIBRARIES=BRIDGE, **(extra_env or {}))
    from pad_mappings import PAD_MAPPINGS  # real controllers the game's old SDL has no mapping for (DualSense)
    env["SDL_GAMECONTROLLERCONFIG"] = "\n".join(filter(None, [env.get("SDL_GAMECONTROLLERCONFIG"), PAD_MAPPINGS]))
    return subprocess.Popen([os.path.join(GAME_DIR, "Sportsfriends"), "-windowed", "0" if fullscreen else "1",
                             "-width", str(size[0]), "-height", str(size[1])],
                            cwd=GAME_DIR, env=env, stdout=log, stderr=subprocess.STDOUT,
                            start_new_session=True)


def launch_instances(n, boot_wait=14, mute=True, size=(640, 360)):
    """Kill running games, turn on exhibition mode and start n game instances with the bridge,
    each on its own socket (muted, small windows by default). Returns the socket paths (pass
    them as PoleRidersEnv(sock=...)); the Popen objects are in launch_instances.procs."""
    prmod = os.path.join(HERE, "..", "prmod.py")
    subprocess.run(["pkill", "-9", "-f", "MacOS/Sportsfriends"])
    subprocess.run(["pkill", "-9", "-f", "MacOS/moved"])
    time.sleep(1)
    subprocess.run([sys.executable, prmod, "exhibition", "on"], check=True, capture_output=True)
    socks, procs = [], []
    for k in range(n):
        sock = os.path.join(SOCK_DIR, f"spr-{k}.sock")
        procs.append(launch_game(sock, mute, size))
        socks.append(sock)
        time.sleep(1.5)
    launch_instances.procs = dict(zip(socks, procs))
    time.sleep(boot_wait)
    return socks


DEFAULT_REWARD = {"goal": 1.0, "kill": 0.3, "ball_progress": 0.003, "position": 0.1}


POSITION_RANGE = 10.0  # m; keep equal to kPositionRange in sim/src/env/features.cpp
HALF_POLE, TIP_BLEND = 3.0, 1.0  # m; keep equal to kHalfPole / kTipBlend in sim/src/env/features.cpp


def _position_potential(state, agent):
    """The "wrong side of the ball" potential in [-1, 0]: 0 anywhere behind the ball (you-ball-goal),
    falling to -1 when I'm POSITION_RANGE m or more past it. 0 while dead (killed and waiting to respawn)
    or ghost.
    (Was 3 m: flat beyond 3 m, so a rider 8+ m past the ball got no signal to come back. Was [-1, 1]
    with +1 at 10 m behind: riders spawn 6.25 m behind the ball, so every step toward it cost reward
    and fresh policies (farm2) learned never to touch the ball. Behind the ball is now flat.)"""
    ball = state.get("ball")
    me = next((p for p in state.get("players", []) if p["id"] == agent), None)
    if not ball or not me or absent(me):
        return 0.0
    ahead = (ball["x"] - me["x"]) if agent == 1 else (me["x"] - ball["x"])
    body = max(-1.0, min(0.0, ahead / POSITION_RANGE))
    if not me.get("pole"):
        return body
    # Near the ball the pole tip decides the side (docs/strategy.md): tip behind the ball = hooked to push
    # it on (good) even with the body slightly past; tip beyond it = can only pull it back (bad). Within
    # HALF_POLE of the ball the tip's position replaces the body's, blended over TIP_BLEND m.
    tip = me["pole"][-1]
    tip_ahead = (ball["x"] - tip["x"]) if agent == 1 else (tip["x"] - ball["x"])
    tipv = max(-1.0, min(0.0, tip_ahead / POSITION_RANGE))
    wgt = max(0.0, min(1.0, (HALF_POLE - abs(ahead)) / TIP_BLEND))
    return body + wgt * (tipv - body)


def backward_rotation(a, b, agent):
    """Backward rotation of the rider's pole from snapshot a to b: the base->tip angle in its attack frame
    (0 = toward the goal it attacks, pi/2 = up), change wrapped to (-pi, pi], positive part only (radians);
    0 unless the rider is alive (not dead or ghost) in both. Keep equal to BackwardRotation in features.cpp."""
    p = next((x for x in a.get("players", []) if x["id"] == agent), None)
    q = next((x for x in b.get("players", []) if x["id"] == agent), None)
    if not p or not q or absent(p) or absent(q) or not p.get("pole") or not q.get("pole"):
        return 0.0
    sign = 1.0 if agent == 1 else -1.0
    ang = lambda r: math.atan2(-(r["pole"][-1]["y"] - r["pole"][0]["y"]), sign * (r["pole"][-1]["x"] - r["pole"][0]["x"]))
    d = ang(q) - ang(p)
    while d > math.pi:
        d -= 2.0 * math.pi
    while d <= -math.pi:
        d += 2.0 * math.pi
    return d if d > 0.0 else 0.0


def compute_reward(a, b, w, frame_skip):
    """Zero-sum rewards for the transition a -> b (full snapshots), weights w (DEFAULT_REWARD
    keys). Returns ({1: r1, 2: r2}, {agent: unweighted terms})."""
    if not (a.get("players") and b.get("players")):
        zero = {"goal": 0.0, "kill": 0.0, "ball_progress": 0.0, "position": 0.0}
        return {1: 0.0, 2: 0.0}, {1: dict(zero), 2: dict(zero)}
    dscore = [b["score"][0] - a["score"][0], b["score"][1] - a["score"][1]]
    # entering sudden death zeroes both scores (e.g. 2-2 -> 0-0): not a goal for anyone
    if b.get("sudden_death") and not a.get("sudden_death"):
        dscore = [0, 0]
    da, db = _died(a), _died(b)
    died = {i: db.get(i, 0) - da.get(i, 0) for i in (1, 2)}
    ba, bb = a.get("ball"), b.get("ball")
    dx = (bb["x"] - ba["x"]) if ba and bb else 0.0
    # a goal or an out-of-bounds respawn teleports the ball; don't count that as movement
    teleport = bool(dscore[0] or dscore[1] or abs(dx) > max(5.0, 1.5 * frame_skip))
    if teleport:
        dx = 0.0
    rew, terms = {}, {}
    for agent in (1, 2):
        me, opp = agent - 1, 2 - agent
        # potential-based shaping (change of the good-side potential): pays for getting back
        # behind the ball, costs for over-running it; telescopes to ~0 over an episode, so it
        # speeds learning without changing which play is optimal. Skipped when the ball teleports.
        dpos = 0.0 if teleport else _position_potential(b, agent) - _position_potential(a, agent)
        t = {"goal": float(dscore[me] - dscore[opp]),
             "kill": float(died[3 - agent] - died[agent]),
             "ball_progress": dx if agent == 1 else -dx,
             "position": dpos}
        terms[agent] = t
        if w.get("death") is None:  # zero-sum kill term (default)
            rew[agent] = sum(w[k] * v for k, v in t.items())
        else:  # kills paid w["kill"] each, own deaths (incl. suicides) cost w["death"] each
            rew[agent] = (w["goal"] * t["goal"] + (w["kill"] * died[3 - agent] - w["death"] * died[agent])
                          + w["ball_progress"] * t["ball_progress"] + w["position"] * t["position"])
        if w.get("rotation"):  # training incentive: backward pole rotation (see backward_rotation)
            rew[agent] += w["rotation"] * backward_rotation(a, b, agent)
    return rew, terms


def _died(state):
    """Deaths + suicides per player id (a suicide doesn't count as a death in the game's stats)."""
    return {p["id"]: p["stats"]["deaths"] + p["stats"]["suicides"] for p in state["players"]}


class GoalRefund:
    """The goal-refund reward rule (user, 2026-10-02), per match, applied after compute_reward to the rewards
    only (terms stay the raw events). Keep equal to GoalRefundState in sim/src/env/features.cpp (the native
    env's default for training; sim/tools/test_env_equiv.py python --goal-refund checks the two agree).
    (a) From the step a goal is scored until the step in which the game's post-goal reset happens (ball
        rebuilt at the centre + both riders respawned in the same tick, ~1.4 s later; seen as the ball
        jumping more than max(5, 1.5 * frame_skip) m in one step without a goal) every term except the goal
        is 0 for both riders: kill/death, ball progress, position potential, rotation bonus. The goal step and
        the reset step have no position/ball-progress term anyway (compute_reward's teleport rule) and the next
        step starts from the respawned state: no spurious potential jump (as if the goal were terminal and the
        potential restarted at the respawn). A match-deciding goal has no reset: the match plays on ~5 s (clock,
        slow motion, fade) until game_over, all inside the window; reset() starts every match without one.
    (b) A death whose penalty was paid (outside a window) is refunded in full at the goal step if the victim's
        team scores while the victim is still dead (before its respawn); the kill reward the other rider got
        for it is taken back at the same step. Records clear when the victim is seen alive and at every goal."""

    def __init__(self):
        self.reset()

    def reset(self):
        self.window, self.steps = False, 0
        self.rec, self.pen, self.cred = [False, False], [0.0, 0.0], [0.0, 0.0]

    def apply(self, a, b, w, frame_skip, terms, rew):
        """rew {1: r1, 2: r2} / terms = compute_reward(a, b, w, frame_skip); rewrites rew in place."""
        if not (a.get("players") and b.get("players")):
            return rew
        dscore = [b["score"][0] - a["score"][0], b["score"][1] - a["score"][1]]
        if b.get("sudden_death") and not a.get("sudden_death"):
            dscore = [0, 0]
        goal = dscore[0] != 0 or dscore[1] != 0
        da, db = _died(a), _died(b)
        died = [db.get(i, 0) - da.get(i, 0) for i in (1, 2)]
        ba, bb = a.get("ball"), b.get("ball")
        dx = (bb["x"] - ba["x"]) if ba and bb else 0.0
        reset = not goal and abs(dx) > max(5.0, 1.5 * frame_skip)  # post-goal ball reset + respawn
        players = {p["id"]: p for p in b["players"]}
        if goal or self.window:
            for i in (1, 2):
                r = 0.0
                r += w["goal"] * terms[i]["goal"]
                rew[i] = r
            if goal:
                for team in (0, 1):  # 1v1: rider index == team
                    p = players.get(team + 1)
                    if dscore[team] > 0 and self.rec[team] and p is not None and p.get("dead"):
                        rew[team + 1] += self.pen[team]
                        rew[2 - team] -= self.cred[team]
                self.rec = [False, False]
                self.window, self.steps = True, 0
            else:
                self.steps += 1
                if reset:  # (a match-deciding goal has no reset: the window lasts to the match's end)
                    self.window = False
            return rew
        for x in (0, 1):
            p = players.get(x + 1)
            if died[x] > 0:
                self.rec[x] = True
                self.pen[x] = (w["kill"] if w.get("death") is None else w["death"]) * died[x]
                self.cred[x] = w["kill"] * died[x]
            elif self.rec[x] and p is not None and not p.get("dead"):
                self.rec[x] = False  # respawned
        return rew


class PoleRidersEnv:
    def __init__(self, frame_skip=4, fast=0, reward=None, sock=SOCK, goal_refund=False):
        """fast=N (>1): vsync off and only draw every N-th frame while the env is active.
        reward: dict overriding DEFAULT_REWARD weights. goal_refund: apply GoalRefund (the native env's
        training default; off here, the single lockstep env is not used for training any more)."""
        self.frame_skip = frame_skip
        self.gr = GoalRefund() if goal_refund else None
        self.w = dict(DEFAULT_REWARD, **(reward or {}))
        self.fast = fast
        self.sock_path = sock
        self.b = None
        self.state = None
        self.synced = False
        self.level = None  # level id of the arena this env trains on (set by start_match)

    # ------------------------------------------------------------------ plumbing
    def _connect(self, timeout=60):
        t0 = time.time()
        while True:
            try:
                self.b = Bridge(self.sock_path)
                return
            except OSError:
                if time.time() - t0 > timeout:
                    raise RuntimeError("sprbridge socket not available — is the game running with --bridge?")
                time.sleep(0.5)

    def _sync(self, on):
        if on != self.synced:
            self.b.cmd("sync on" if on else "sync off")
            self.synced = on

    def _raw_step(self, n):
        self.state = json.loads(self.b.cmd(f"step {n}"))
        return self.state

    @staticmethod
    def _stick(agent, action):
        """Agent-frame (run, aim_x, aim_y) -> pad sticks (lx, rx, ry); SDL +y is down."""
        run, ax, ay = (list(action) + [0, 0, 0])[:3]
        sign = 1 if agent == 1 else -1
        m = math.hypot(ax, ay)
        if m > 1:
            ax, ay = ax / m, ay / m
        clamp = lambda v: max(-1.0, min(1.0, float(v)))
        return clamp(sign * run), clamp(sign * ax), clamp(-ay)

    # ------------------------------------------------------------------ menus
    def start_match(self, launch=False, arena=2):
        """Get into a running 1v1 match with two virtual pads (pad 0 = Vaulters, pad 1 = Benders).

        launch=True kills any running game and starts one on this env's socket via
        launch_instances (exhibition mode: config.ini [game] exhibition = PoleRiders, undo with
        `prmod.py exhibition off`), so it boots straight into Pole Riders team select. From team
        select the rest is scripted: pick teams, confirm players, pick the arena (1 Kalgoorlie,
        2 Flatirons, 3 Dakhla, 4 Princeton — the select grid is 2 wide). Does nothing if a match is
        already running."""
        if launch:
            self.sock_path = launch_instances(1)[0]
        self._connect()
        self._sync(False)
        for i in (0, 1):
            self.b.cmd(f"pad connect {i}")
        if self.b.state().get("in_match"):
            self.level = self.b.state().get("level")
            return
        for attempt in range(3):
            self.b.cmd("pad set 0 lx=-1"); self.b.cmd("pad set 1 lx=1"); time.sleep(0.5)
            self.b.cmd("pad clear 0"); self.b.cmd("pad clear 1"); time.sleep(0.8)
            for pads, wait in [((0,), 2.5), ((0, 1), 2.5)]:  # teams, players
                for p in pads:
                    self.b.cmd(f"pad tap {p} a")
                time.sleep(wait)
            col, row = (arena - 1) % 2, (arena - 1) // 2
            for btn in ["right"] * col + ["down"] * row:
                self.b.cmd(f"pad tap 0 {btn}")
                time.sleep(0.5)
            self.b.cmd("pad tap 0 a")
            time.sleep(6)
            if self.b.state().get("in_match"):
                self.level = self.b.state().get("level")
                return
        raise RuntimeError("menu script did not reach a match; take a screenshot and adjust")

    # ------------------------------------------------------------------ RL API
    def reset(self, seed=None):
        if self.b is None:
            self._connect()
        for i in (0, 1):
            self.b.cmd(f"pad clear {i}")
        self._sync(True)
        if self.fast > 1:
            self.b.cmd(f"fast on {self.fast}")
        st = self._raw_step(1)
        old = st.get("match_id", 0)
        self.b.cmd("reset" if seed is None else f"reset {int(seed)}")
        for _ in range(2000):  # fade-out + rebuild takes ~1-2 s of game time
            st = self._raw_step(4)
            if st.get("in_match") and st.get("match_id", 0) != old and st.get("players"):
                break
        else:
            raise RuntimeError("match did not restart")
        if self.level is not None and st.get("level") != self.level:
            raise RuntimeError(f"match is on level {st.get('level')}, expected {self.level}")
        seats = {p["id"]: p["x"] for p in st["players"]}
        if not seats.get(1, 0) < 0 < seats.get(2, 0):
            raise RuntimeError(f"unexpected seats {seats}: player 1 must start left (Vaulters)")
        self.prev = st
        if self.gr:
            self.gr.reset()
        return {1: observe(st, 1), 2: observe(st, 2)}

    def step(self, actions):
        s1 = self._stick(1, actions.get(1, (0, 0, 0)))
        s2 = self._stick(2, actions.get(2, (0, 0, 0)))
        self.state = json.loads(self.b.cmd(
            f"act {self.frame_skip} " + " ".join(f"{v:.4f}" for v in s1 + s2)))
        st = self.state
        if not st.get("in_match") or not st.get("players"):
            # scored/ended: the game may be between PlayStates for a frame or two
            for _ in range(30):
                st = self._raw_step(1)
                if st.get("in_match") and st.get("players"):
                    break
        rew, terms = self._reward(self.prev, st)
        if st.get("players"):
            self.prev = st
        wrong_map = self.level is not None and st.get("level") not in (None, self.level)
        # Only the game decides when a match is over (score limit, or the clock plus whatever
        # tie-break follows); ending at time_left == 0 would cut off sudden death.
        done = bool(st.get("game_over")) or not st.get("in_match") or wrong_map
        obs = {1: observe(st, 1), 2: observe(st, 2)} if st.get("players") else None
        return obs, rew, done, {"state": st, "terms": terms}

    def _reward(self, a, b):
        rew, terms = compute_reward(a, b, self.w, self.frame_skip)
        if getattr(self, "gr", None):
            self.gr.apply(a, b, self.w, self.frame_skip, terms, rew)
        return rew, terms

    def close(self):
        if self.b:
            for i in (0, 1):
                self.b.cmd(f"pad clear {i}")
            if self.fast > 1:
                self.b.cmd("fast off")
            self._sync(False)


def demo(steps=600, fast=0):
    """Scripted agents: run toward the ball and wave the pole."""
    env = PoleRidersEnv(frame_skip=4, fast=fast)
    env.start_match()
    obs = env.reset()
    t0, total = time.time(), {1: 0.0, 2: 0.0}
    for t in range(steps):
        acts = {}
        for a in (1, 2):
            o = obs[a]
            ball_dx = o[30] - o[0]
            phase = math.sin(t / 15 + a)
            acts[a] = (max(-1, min(1, ball_dx)), math.cos(phase), abs(math.sin(phase)))
        obs, rew, done, info = env.step(acts)
        for a in (1, 2):
            total[a] += rew[a]
        if done or obs is None:
            obs = env.reset()
    dt = time.time() - t0
    st = info["state"]
    print(f"{steps} steps x {env.frame_skip} ticks in {dt:.1f}s = {steps * env.frame_skip / dt:.0f} ticks/s "
          f"({steps * env.frame_skip / 60 / dt:.1f}x real time); score {st.get('score')}, rewards {total}")
    env.close()


if __name__ == "__main__":
    demo(int(sys.argv[1]) if len(sys.argv) > 1 else 600, int(sys.argv[2]) if len(sys.argv) > 2 else 0)
