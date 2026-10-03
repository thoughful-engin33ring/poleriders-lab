#!/usr/bin/env python3
"""Batched native Pole Riders env: the native C++ match (sim/) as a drop-in for the real-game farm.

    host = NativeHost(n_hosts=14, matches=16, frame_skip=4, sd_frac=0.15, max_steps=5400)
    obs = host.reset_all()                  # float32[N, 2, 62] (observe2), N = n_hosts * matches
    out = host.step(actions)                # actions float32[N, 2, 3]: (run, aim_x, aim_y) per agent
    out["obs"], out["final"], out["rew"], out["terms"], out["reset"], out["trunc"], out["sd"],
    out["score"], out["sd_start"]           # exactly farm_worker.py's `out` dict, for all N matches
    out["touch"], out["touch_final"]        # ball-touch bookkeeping float32[N, 2, 5] (not observed; see TOUCH_*)
    host.snapshot(k, slot); host.restore(k, slot)   # exact save / restore of match k (per-match arenas)
    host.plan_init(owner); host.plan_begin(); host.plan_step(acts); host.plan_end()   # branch pool (planner.py)

Semantics are those of bridge/farm_worker.py + farm_env.FarmHost, value for value: the same stick
mapping (PoleRidersEnv._stick, "%.4f" on the wire), observations (observe2 from the %.4f snapshot
values), rewards (compute_reward, DEFAULT_REWARD incl. position shaping), done / max_steps
truncation, auto-reset with sudden-death starts (probability sd_frac, clock 0.05). Host h of the
batch draws its reset coins from random.Random(seed + h) in match order, like FarmHost(seed=h).
Checked by sim/tools/test_env_equiv.py.

The arrays in `out` are zero-copy views of the env's buffers: every step() overwrites them (copy
what you keep). Build the library first: `make -C sim env` (SPRSIM_LIB=path loads another build, e.g. a
test build from `make -C sim env B=build-dev`).

Reward rule `goal_refund` (default ON = the training default since 2026-10-02; GoalRefundState in
sim/src/env/features.h, Python mirror spr_env.GoalRefund): no non-goal reward between a goal and the
respawn, and a death is refunded in full (the kill taken back) when the victim's team scores while it is
dead. It only changes `rew` (`terms` stay the raw events); statistics that count kills through `rew`
(eval_h2h's COUNT_KILLS) pass goal_refund=False. With goal_refund=False the env is bit-identical to before.

For train_farm.py (--native): NativeFarm gives Pipe-like endpoints that speak host_worker's
protocol ("ready" / ("step", acts) -> ("ok", out)), one per logical host, backed by one in-process
NativeHost stepped on a C++ thread pool; plus optional `farm state` sockets for the dashboard.
"""
import ctypes, os, socket, threading, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
LIB_PATH = os.environ.get("SPRSIM_LIB") or os.path.join(HERE, "..", "sim", "build", "libsprsim.dylib")
OBS2_SIZE = 62
# menu arena (spr_env.start_match / train_farm --arena) -> PlayState level id. The training map is
# menu "arena 2" = level id 1 (sim/tools/oracle.py MENU_CELL); its geometry is arenas/arena2.json.
ARENA_LEVEL = {2: 1, 4: 2}
DEFAULT_REWARD = {"goal": 1.0, "kill": 0.3, "ball_progress": 0.003, "position": 0.1}  # spr_env's
# out["touch"][k, rider, i] fields (rider 0 = player 1, 1 = player 2; sim/src/env/batch_env.h EnvBuffers::touch)
TOUCH_AGE, TOUCH_WHAT, TOUCH_KICK_AGE, TOUCH_NOW, TOUCH_GAME_LAST = range(5)
TOUCH_BODY, TOUCH_POLE, TOUCH_HANDLE = 1, 2, 4  # TOUCH_WHAT bits (handle = a pole segment the game tags as the rider: kicks like the body)
TOUCH_NEVER = 999.0            # seconds: no touch / kick yet this match


class _Bufs(ctypes.Structure):
    _fields_ = [("obs", ctypes.POINTER(ctypes.c_float)), ("final", ctypes.POINTER(ctypes.c_float)),
                ("rew", ctypes.POINTER(ctypes.c_float)), ("terms", ctypes.POINTER(ctypes.c_float)),
                ("reset", ctypes.POINTER(ctypes.c_uint8)), ("trunc", ctypes.POINTER(ctypes.c_uint8)),
                ("sd", ctypes.POINTER(ctypes.c_uint8)), ("sd_start", ctypes.POINTER(ctypes.c_uint8)),
                ("score", ctypes.POINTER(ctypes.c_int32))]


class _PlanBufs(ctypes.Structure):
    _fields_ = [("obs", ctypes.POINTER(ctypes.c_float)), ("rew", ctypes.POINTER(ctypes.c_float)),
                ("terms", ctypes.POINTER(ctypes.c_float)), ("done", ctypes.POINTER(ctypes.c_uint8)),
                ("trunc", ctypes.POINTER(ctypes.c_uint8)), ("score", ctypes.POINTER(ctypes.c_int32)),
                ("touch", ctypes.POINTER(ctypes.c_float))]


_lib = None


def load_lib(path=LIB_PATH):
    global _lib
    if _lib is not None:
        return _lib
    if not os.path.exists(path):
        raise FileNotFoundError(f"{path} missing: run `make -C sim env`")
    L = ctypes.CDLL(path)
    vp, sz, cp = ctypes.c_void_p, ctypes.c_size_t, ctypes.c_char_p
    L.sprenv_create.restype = vp
    L.sprenv_create.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_double, ctypes.c_int,
                                ctypes.c_ulonglong, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int,
                                ctypes.POINTER(ctypes.c_int), ctypes.c_float]
    L.sprenv_destroy.argtypes = [vp]
    L.sprenv_set_pit.argtypes = [vp, ctypes.c_double, ctypes.c_float, ctypes.c_float]
    L.sprenv_pit_start.argtypes, L.sprenv_pit_start.restype = [vp, ctypes.c_int], ctypes.c_int
    L.sprenv_set_ghost_lock.argtypes = [vp, ctypes.c_int, ctypes.c_int]
    L.sprenv_set_legacy_obs.argtypes = [vp, ctypes.c_int, ctypes.c_int]
    L.sprenv_set_end_on_sd.argtypes = [vp, ctypes.c_int]
    L.sprenv_set_format.argtypes = [vp, ctypes.c_int, ctypes.c_float]
    L.sprenv_set_scramble.argtypes = [vp, ctypes.c_int]
    L.sprenv_set_death_weight.argtypes = [vp, ctypes.c_double]
    L.sprenv_set_rotation_weight.argtypes = [vp, ctypes.c_double]
    L.sprenv_set_random_pole.argtypes = [vp, ctypes.c_int]
    L.sprenv_set_goal_spawn.argtypes = [vp, ctypes.c_double]
    L.sprenv_n.argtypes, L.sprenv_n.restype = [vp], ctypes.c_int
    L.sprenv_buffers.argtypes = [vp, ctypes.POINTER(_Bufs)]
    L.sprenv_reset_all.argtypes = [vp]
    L.sprenv_step.argtypes = [vp, vp]
    L.sprenv_set_record.argtypes = [vp, ctypes.c_int]
    L.sprenv_json.argtypes, L.sprenv_json.restype = [vp, ctypes.c_int, ctypes.c_int, cp, sz], sz
    L.sprenv_state_json.argtypes, L.sprenv_state_json.restype = [vp, ctypes.c_int, ctypes.c_int, cp, sz], sz
    L.sprenv_level_json.argtypes, L.sprenv_level_json.restype = [vp, cp, sz], sz
    L.sprenv_terrain_floor.argtypes, L.sprenv_terrain_floor.restype = [vp, ctypes.c_double, ctypes.c_double], ctypes.c_double
    L.sprenv_terrain_above.argtypes = [vp, ctypes.c_double, ctypes.c_double, ctypes.POINTER(ctypes.c_double)]
    L.sprenv_terrain_above.restype = ctypes.c_int
    L.sprenv_terrain_column.argtypes = [vp, ctypes.c_int, ctypes.POINTER(ctypes.c_double), ctypes.c_int]
    L.sprenv_terrain_column.restype = ctypes.c_int
    L.sprenv_raw_act.argtypes = [vp, vp, ctypes.c_int]
    L.sprenv_raw_reset.argtypes, L.sprenv_raw_reset.restype = [vp, ctypes.c_int, ctypes.c_float, cp, sz], sz
    L.sprenv_stick.argtypes = [ctypes.c_int, ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float)]
    # added 2026-10-02 (goal refund, ball touches): optional, so this file still runs on an older build
    L.has_goal_refund = hasattr(L, "sprenv_set_goal_refund")
    if L.has_goal_refund:
        L.sprenv_set_goal_refund.argtypes = [vp, ctypes.c_int]
    L.has_touch = hasattr(L, "sprenv_touch_buffers")
    if L.has_touch:
        fpp = ctypes.POINTER(ctypes.POINTER(ctypes.c_float))
        L.sprenv_touch_buffers.argtypes = [vp, fpp, fpp]
        L.sprenv_touch_size.restype = ctypes.c_int
        L.sprenv_debug_event.argtypes, L.sprenv_debug_event.restype = [vp, ctypes.c_int, ctypes.c_int, ctypes.c_int], ctypes.c_int
    # added 2026-10-02 (per-match arenas): snapshot / restore + planning branches
    L.has_snapshot = hasattr(L, "sprenv_snapshot")
    if L.has_snapshot:
        L.sprenv_snapshot.argtypes = L.sprenv_restore.argtypes = [vp, ctypes.c_int, ctypes.c_int]
        L.sprenv_snapshot.restype = L.sprenv_restore.restype = ctypes.c_int
        L.sprenv_arena_used.argtypes, L.sprenv_arena_used.restype = [vp, ctypes.c_int], sz
        L.sprenv_check_arena.argtypes, L.sprenv_check_arena.restype = [vp, ctypes.c_int, ctypes.c_int], ctypes.c_int
        L.sprenv_arena_counters.argtypes = [ctypes.POINTER(ctypes.c_ulonglong)]
        L.sprenv_plan_init.argtypes, L.sprenv_plan_init.restype = [vp, ctypes.POINTER(ctypes.c_int), ctypes.c_int], ctypes.c_int
        L.sprenv_plan_buffers.argtypes = [vp, ctypes.POINTER(_PlanBufs)]
        L.sprenv_plan_begin.argtypes, L.sprenv_plan_begin.restype = [vp, vp], ctypes.c_int
        L.sprenv_plan_step.argtypes = [vp, vp]
        L.sprenv_plan_end.argtypes = [vp]
        L.sprenv_plan_json.argtypes, L.sprenv_plan_json.restype = [vp, ctypes.c_int, cp, sz], sz
    _lib = L
    return L


def _str(fn, *args, cap=1 << 16):
    """Call a C function that writes a string into (buf, cap) and returns its full length. The first
    try (64 KB) always fits a single match, so side-effecting calls (raw_reset) run exactly once."""
    buf = ctypes.create_string_buffer(cap)
    n = fn(*args, buf, cap)
    if n >= cap:
        buf = ctypes.create_string_buffer(n + 1)
        fn(*args, buf, n + 1)
    return buf.value.decode()


class NativeHost:
    act_size = 3  # run, aim x, aim y (per agent, in its own frame)
    def __init__(self, n_hosts=1, matches=16, frame_skip=4, sd_frac=0.0, max_steps=5400, seed=0, threads=0,
                 reward=None, level=1, heads=(0, 0), sd_clock=0.05, pit_frac=0.0, end_on_sd=False,
                 match_format=None, scramble=False, random_pole=False, goal_spawn=0.0, goal_refund=True):
        self.L = load_lib()
        self.n_hosts, self.K, self.n = n_hosts, matches, n_hosts * matches
        self.frame_skip = frame_skip
        w = dict(DEFAULT_REWARD, **(reward or {}))
        wts = (ctypes.c_double * 4)(w["goal"], w["kill"], w["ball_progress"], w["position"])
        hd = (ctypes.c_int * 2)(*heads)
        self.h = self.L.sprenv_create(n_hosts, matches, frame_skip, sd_frac, max_steps, seed, threads, wts, level,
                                      hd, sd_clock)
        if not self.h:
            raise ValueError("sprenv_create failed")
        b = _Bufs()
        self.L.sprenv_buffers(self.h, ctypes.byref(b))
        N, D = self.n, OBS2_SIZE
        arr = np.ctypeslib.as_array
        self.out = {"obs": arr(b.obs, (N, 2, D)), "final": arr(b.final, (N, 2, D)), "rew": arr(b.rew, (N, 2)),
                    "terms": arr(b.terms, (N, 2, 2)), "reset": arr(b.reset, (N,)).view(bool),
                    "trunc": arr(b.trunc, (N,)).view(bool), "sd": arr(b.sd, (N,)).view(bool),
                    "score": arr(b.score, (N, 2)), "sd_start": arr(b.sd_start, (N,)).view(bool)}
        if self.L.has_touch:
            tb, tf = ctypes.POINTER(ctypes.c_float)(), ctypes.POINTER(ctypes.c_float)()
            self.L.sprenv_touch_buffers(self.h, ctypes.byref(tb), ctypes.byref(tf))
            T = self.L.sprenv_touch_size()
            self.out["touch"], self.out["touch_final"] = arr(tb, (N, 2, T)), arr(tf, (N, 2, T))
        self._acts = np.zeros((N, 2, 3), np.float32)
        self.goal_refund = bool(goal_refund) and self.L.has_goal_refund
        if goal_refund:  # training reward rule (the default): see the module doc
            if self.L.has_goal_refund:
                self.L.sprenv_set_goal_refund(self.h, 1)
            else:
                print("native_env: this libsprsim build has no goal refund (rebuild: make -C sim env); rule OFF",
                      flush=True)
        if pit_frac > 0:
            self.set_pit(pit_frac)
        if end_on_sd:  # training rule: ties at the end of time end the match (no sudden death)
            self.L.sprenv_set_end_on_sd(self.h, 1)
        if match_format:  # (goals to win, seconds); default = the game config the oracle uses, (3, 180)
            self.L.sprenv_set_format(self.h, int(match_format[0]), float(match_format[1]))
        if scramble:  # training curriculum: random re-spawn of both riders + ball after every goal / kill
            self.L.sprenv_set_scramble(self.h, 1)
        if w.get("death") is not None:  # kills paid w["kill"], own deaths cost w["death"] (not zero-sum)
            self.L.sprenv_set_death_weight(self.h, float(w["death"]))
        if w.get("rotation"):
            self.set_rotation_weight(w["rotation"])
        if random_pole:  # training curriculum: random pole angle (forward .. back-up) at every (re)spawn
            self.L.sprenv_set_random_pole(self.h, 1)
        if goal_spawn:  # training curriculum: random re-spawn of both riders + ball after a goal (prob.)
            self.L.sprenv_set_goal_spawn(self.h, float(goal_spawn))

    def set_rotation_weight(self, w):
        """Reward weight of backward pole rotation (spr_env.backward_rotation); may change between steps."""
        self.L.sprenv_set_rotation_weight(self.h, float(w))

    def set_pit(self, frac, x_px=1152.0, y_px=46.0):
        """Training curriculum: this fraction of non-sudden-death resets drops player 1, player 2 or both
        into their ENEMY pit with the pole pointing up (ball in the middle as usual)."""
        self.L.sprenv_set_pit(self.h, frac, x_px, y_px)

    def set_ghost_lock(self, k, mask):
        """Training-only: keep riders of match k (bitmask 1 = player 1, 2 = player 2) in the game's ghost
        state — terrain collisions only (no rider/ball contact, no kills). Persists across resets."""
        self.L.sprenv_set_ghost_lock(self.h, k, mask)

    def set_legacy_obs(self, k, mask):
        """Evaluation-only: riders of match k (bitmask) observe dead riders the way policies trained before
        the dead-rider fix (farm1, manager2, worker2) did: the corpse as a live rider. Persists; applies
        from the next observation."""
        self.L.sprenv_set_legacy_obs(self.h, k, mask)

    def pit_start(self, k):
        """Riders dropped into the enemy pit at match k's last reset: bitmask 1 = player 1, 2 = player 2."""
        return self.L.sprenv_pit_start(self.h, k)

    def reset_all(self):
        """Every match restarted (reset coins drawn in match order). Returns the obs view."""
        self.L.sprenv_reset_all(self.h)
        return self.out["obs"]

    def step(self, actions):
        a = np.asarray(actions, dtype=np.float32)
        if a.shape != (self.n, 2, 3) or not a.flags.c_contiguous:
            self._acts[...] = a.reshape(self.n, 2, 3)
            a = self._acts
        self.L.sprenv_step(self.h, a.ctypes.data)
        return self.out

    # ---- snapshot / restore (per match) and planning branches (sim/src/env/batch_env.h)
    def snapshot(self, k, slot=0):
        """Save match k (its whole native state + the env's bookkeeping: reward rule state, touch counters,
        curriculum RNG streams) in snapshot slot `slot` (0..15). ~5-10 us."""
        rc = self.L.sprenv_snapshot(self.h, k, slot)
        if rc:
            raise RuntimeError(f"snapshot({k}, {slot}) failed: {rc}")

    def restore(self, k, slot=0):
        """Put match k back into the state saved in `slot`: stepping it with the same actions then gives exactly
        the trajectory it had from there (obs, rewards, terms, touch, snapshot JSON). Rewrites out["obs"][k] and
        out["touch"][k] (the other output rows keep the last step's values). Not restored: the host's
        sudden-death coin stream (sd_frac > 0), config switches. Restore as often as you like."""
        rc = self.L.sprenv_restore(self.h, k, slot)
        if rc:
            raise RuntimeError(f"restore({k}, {slot}) failed: {rc}")

    def arena_used(self, k):
        return self.L.sprenv_arena_used(self.h, k)

    def check_arena(self, k=0, branch=-1):
        """Debug: game objects of match k (or of planning branch `branch`) outside its arena; must be 0."""
        return self.L.sprenv_check_arena(self.h, k, branch)

    def plan_init(self, owner):
        """Planning pool: branch b is a clone of match owner[b] (an env-owned match in its own arena, kept in step
        with match owner[b] by re-playing its actions). Sets self.plan = output views [B, ...]: obs [B, 2, 62],
        rew [B, 2], terms [B, 2, 2], done [B], trunc [B], score [B, 2], touch [B, 2, 5]."""
        owner = np.ascontiguousarray(owner, dtype=np.int32)
        B = len(owner)
        if self.L.sprenv_plan_init(self.h, owner.ctypes.data_as(ctypes.POINTER(ctypes.c_int)), B):
            raise ValueError("plan_init: bad owner index")
        b = _PlanBufs()
        self.L.sprenv_plan_buffers(self.h, ctypes.byref(b))
        arr = np.ctypeslib.as_array
        self.plan_n = B
        self.plan = {"obs": arr(b.obs, (B, 2, OBS2_SIZE)), "rew": arr(b.rew, (B, 2)), "terms": arr(b.terms, (B, 2, 2)),
                     "done": arr(b.done, (B,)).view(bool), "trunc": arr(b.trunc, (B,)).view(bool),
                     "score": arr(b.score, (B, 2)),
                     "touch": arr(b.touch, (B, 2, self.L.sprenv_touch_size()))}
        self._plan_acts = np.zeros((B, 2, 3), np.float32)
        self._plan_active = np.ones(B, np.uint8)
        return self.plan

    def plan_begin(self, active=None):
        """Start a search: every active branch (bool [B], None = all) becomes an exact copy of its match's current
        state (plan["obs"] = the match's observation, done/rew cleared). Returns the plan views."""
        if active is None:
            ptr = None
        else:
            self._plan_active[:] = np.asarray(active, bool)
            ptr = self._plan_active.ctypes.data
        if self.L.sprenv_plan_begin(self.h, ptr):
            raise RuntimeError("plan_begin: a match cannot be replayed (raw_act / raw_reset since its reset)")
        return self.plan

    def plan_step(self, actions):
        """One decision on every active branch that has not ended: actions float32[B, 2, 3] (both riders). No
        auto-reset: a branch whose match ends stays frozen with done = True (and rew 0 from then on)."""
        a = np.asarray(actions, dtype=np.float32)
        if a.shape != (self.plan_n, 2, 3) or not a.flags.c_contiguous:
            self._plan_acts[...] = a.reshape(self.plan_n, 2, 3)
            a = self._plan_acts
        self.L.sprenv_plan_step(self.h, a.ctypes.data)
        return self.plan

    def plan_end(self):
        """Every branch back to its root (cheap; the next plan_begin only re-plays the matches' new steps)."""
        self.L.sprenv_plan_end(self.h)

    def plan_json(self, b):
        """Snapshot JSON of planning branch b's current state (same format as json(k))."""
        return _str(self.L.sprenv_plan_json, self.h, b)

    # ---- inspection / tests
    def record(self, on=True):
        """Keep each step's post-step snapshot JSON and each reset's fresh-match JSON (json(k, 1/2))."""
        self.L.sprenv_set_record(self.h, int(on))

    def json(self, k, which=0):
        """Bridge snapshot_ps JSON of match k: 0 current, 1 last post-step state, 2 last reset,
        3 raw `farm dump` format (sim/tools/diff_dump.py)."""
        return _str(self.L.sprenv_json, self.h, k, which)

    def state_json(self, k0=0, k1=None):
        """`farm state`-style {"matches": [...]} for matches k0..k1-1."""
        return _str(self.L.sprenv_state_json, self.h, k0, self.n if k1 is None else k1)

    def level_json(self):
        return _str(self.L.sprenv_level_json, self.h)

    def terrain_floor(self, x, y):
        return self.L.sprenv_terrain_floor(self.h, x, y)

    def terrain_above(self, x, y):
        v = ctypes.c_double()
        return v.value if self.L.sprenv_terrain_above(self.h, x, y, ctypes.byref(v)) else None

    def terrain_column(self, i):
        buf = (ctypes.c_double * 64)()
        n = self.L.sprenv_terrain_column(self.h, i, buf, 64)
        return None if n < 0 else list(buf[:n])

    def debug_event(self, k, kind, who):
        """Tests only (changes match k): kind "kill" = rider `who` (0 = player 1, 1 = player 2) is killed by
        the other one (the kill branch's bookkeeping: death/kill stats, 3 s respawn timer, off the pole), kind
        "goal" = the ball is put against the wall team `who` scores on, moving into it (the game scores it)."""
        rc = self.L.sprenv_debug_event(self.h, k, {"kill": 1, "goal": 2}[kind], who)
        if rc:
            raise RuntimeError(f"debug_event({k}, {kind}, {who}) failed: {rc}")

    def raw_act(self, sticks, n):
        """Bridge `farm act n ...` on every match: sticks float32[N, 6] as the game receives them."""
        a = np.ascontiguousarray(sticks, dtype=np.float32).reshape(self.n, 6)
        self.L.sprenv_raw_act(self.h, a.ctypes.data, n)

    def raw_reset(self, k, clock=0.0):
        """Bridge `farm reset k [clock]`: returns the fresh match's snapshot JSON."""
        return _str(self.L.sprenv_raw_reset, self.h, k, clock)

    def close(self):
        if self.h:
            self.L.sprenv_destroy(self.h)
            self.h = None

    def __del__(self):
        try:
            self.close()
        except Exception:  # noqa: BLE001
            pass


def stick(agent, action):
    """Native PoleRidersEnv._stick + "%.4f" + bridge strtof (for tests)."""
    L = load_lib()
    a = (ctypes.c_float * 3)(*action)
    o = (ctypes.c_float * 3)()
    L.sprenv_stick(agent, a, o)
    return list(o)


# ---------------------------------------------------------------- train_farm.py integration

class _Conn:
    """One logical host's end of host_worker's Pipe protocol."""

    def __init__(self, farm, h):
        self.farm, self.h, self.inbox = farm, h, []

    def send(self, msg):
        self.farm._send(self.h, msg)

    def recv(self):
        if self.inbox:
            return self.inbox.pop(0)
        return self.farm._recv(self.h)


class NativeFarm:
    """H logical hosts x K matches, stepped together: `conns[h]` behaves like a farm_worker Pipe.
    socks: optional per-host Unix socket paths answering `farm state` (dashboard.py farm mode).

    groups > 1 splits the hosts into that many independent native envs (hosts g*H/G .. ), each
    steppable in the background (`step_async(g, acts)` / `wait(g)`; the C call releases the GIL), so a
    trainer can compute one group's actions while another group's physics runs. Host h's reset RNG is
    random.Random(seed + h) either way. The Pipe emulation (`conns`) needs groups == 1."""

    def __init__(self, H, K, cfg, threads=0, seed=0, socks=None, heads=(0, 0), groups=1, hier=None):
        if H % groups:
            raise ValueError(f"{H} hosts don't split into {groups} groups")
        self.H, self.K, self.G, self.Hg = H, K, groups, H // groups
        kw = dict(sd_frac=cfg["sd_frac"], max_steps=cfg["episode_steps"], threads=threads,
                  level=cfg.get("level", 1), heads=heads, pit_frac=cfg.get("pit_frac", 0.0),
                  reward=cfg.get("reward"), end_on_sd=cfg.get("end_on_sd", False),
                  match_format=cfg.get("match_format"), scramble=cfg.get("scramble", False),
                  random_pole=cfg.get("random_pole", False), goal_spawn=cfg.get("goal_spawn", 0.0),
                  goal_refund=cfg.get("goal_refund", True))
        if hier:  # (worker checkpoint, macro): the trained policy is a manager over a frozen worker
            from hier_env import HierHost
            self.hosts = [HierHost(hier[0], hier[1], self.Hg, K, cfg["frame_skip"], seed=seed + g * self.Hg, **kw)
                          for g in range(groups)]
        else:
            self.hosts = [NativeHost(self.Hg, K, cfg["frame_skip"], seed=seed + g * self.Hg, **kw)
                          for g in range(groups)]
        self.host = self.hosts[0]
        self.acts = np.zeros((H * K, 2, self.host.act_size), np.float32)
        self.sent, self.ready = [False] * H, [False] * H
        self._pending = [None] * groups
        obs = np.concatenate([hh.reset_all() for hh in self.hosts])
        sd0 = np.concatenate([hh.out["sd_start"] for hh in self.hosts])
        self.conns = [_Conn(self, h) for h in range(H)]
        for h, c in enumerate(self.conns):
            sl = slice(h * K, (h + 1) * K)
            c.inbox.append(("ready", {"obs": obs[sl].copy(), "sd_start": sd0[sl].copy()}))
        self.server = _StateServer(self, socks) if socks else None

    def set_flat(self, j, agent, opp):
        """Global match j (= host * K + k): flat league opponent for rider `agent` (0 = none); --hier only."""
        host, lh = self.host_of(j // self.K)
        host.set_flat(lh * self.K + j % self.K, agent, opp)

    def host_of(self, h):
        """(NativeHost, local host index) of logical host h."""
        return self.hosts[h // self.Hg], h % self.Hg

    def step_async(self, g, acts):
        """Start stepping group g (acts float32[Hg*K, 2, 3]) on a background thread."""
        if self._pending[g]:
            raise RuntimeError(f"group {g} is already stepping")
        a = np.ascontiguousarray(acts, dtype=np.float32).reshape(self.Hg * self.K, 2, -1)
        th = threading.Thread(target=self.hosts[g].step, args=(a,), daemon=True)
        self._pending[g] = (th, a)
        th.start()

    def wait(self, g):
        """Finish group g's step; returns its output views (overwritten by its next step)."""
        th, _ = self._pending[g]
        th.join()
        self._pending[g] = None
        if self.server:
            self.server.refresh(g)
        return self.hosts[g].out

    def _send(self, h, msg):
        cmd, acts = msg
        if cmd == "step":
            self.acts[h * self.K:(h + 1) * self.K] = acts
            self.sent[h] = True

    def _recv(self, h):
        if self.G != 1:
            raise RuntimeError("the Pipe emulation needs groups == 1 (use step_async/wait)")
        if not self.ready[h]:
            if not self.sent[h]:
                raise RuntimeError(f"recv on native host {h} without a step")
            self.host.step(self.acts)  # every host that sent this round steps together
            for i in range(self.H):
                self.ready[i], self.sent[i] = self.sent[i], False
            if self.server:
                self.server.refresh()
        self.ready[h] = False
        sl = slice(h * self.K, (h + 1) * self.K)
        return "ok", {k: v[sl] for k, v in self.host.out.items()}

    def close(self):
        if self.server:
            self.server.close()
        for hh in self.hosts:
            hh.close()


class _StateServer:
    """Answers `farm state` on one Unix socket per logical host from a snapshot cache the trainer
    thread refreshes (at most 10x/s, and only while someone polled in the last 5 s)."""

    def __init__(self, farm, socks):
        self.farm, self.socks = farm, socks
        self.cache = [b'{"matches":[]}\n'] * len(socks)
        self.last_poll, self.last_refresh = 0.0, [0.0] * farm.G
        self.listeners = []
        for h, path in enumerate(socks):
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
            s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
            s.bind(path)
            s.listen(4)
            self.listeners.append(s)
            threading.Thread(target=self._accept, args=(s, h), daemon=True).start()

    def refresh(self, g=0):
        """Re-snapshot group g's hosts (call only while group g is not stepping)."""
        now = time.time()
        if now - self.last_poll > 5.0 or now - self.last_refresh[g] < 0.1:
            return
        self.last_refresh[g] = now
        K, f = self.farm.K, self.farm
        cache = list(self.cache)
        for h in range(g * f.Hg, min((g + 1) * f.Hg, len(self.socks))):
            host, lh = f.host_of(h)
            cache[h] = (host.state_json(lh * K, (lh + 1) * K) + "\n").encode()
        self.cache = cache

    def _accept(self, s, h):
        while True:
            try:
                c, _ = s.accept()
            except OSError:
                return
            threading.Thread(target=self._serve, args=(c, h), daemon=True).start()

    def _serve(self, c, h):
        with c, c.makefile("r") as rf:
            for line in rf:
                self.last_poll = time.time()
                if line.strip() == "farm state":
                    c.sendall(self.cache[h])
                else:
                    c.sendall(b'{"error":"native farm: only `farm state`"}\n')

    def close(self):
        for s, path in zip(self.listeners, self.socks):
            s.close()
            try:
                os.unlink(path)
            except FileNotFoundError:
                pass
