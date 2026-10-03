#!/usr/bin/env python3
"""Concept analysis of recorded real-game matches (runs/human/*.jsonl): the human vs the bot, per match and pooled.

Every log goes through concepts.py's play-log path (playlog_read -> playlog_resample -> obs_from_states, one row
per 4 game ticks = 1/15 s as in the sim; the bridge logs a snapshot every 4th RENDERED frame, which on the user's
120 Hz display is every ~2 ticks). Ball touches are not in the logs, so touch rows are rebuilt from the full-rate
snapshots (see `touch_rows`):
  - KICKS are exact (to +-2 ticks): the game's per-rider stats.kicks counter, minus stomp kills (the game's
    PlayerCollision adds a kick to the killer's stats too: sim/src/game/contacts.cpp).
  - other touches are INFERRED: a jump of the ball's velocity that its free motion (gravity + the zipline spring,
    a linear model fitted per match on moments with nobody near the ball) doesn't explain, with a rider's body or
    pole within 1.3 m of the ball; attributed to the rider who kicked then, else the nearest one.
  - the game's lastTouch: the last (inferred) toucher; at goals the game's own goals / own_goals counters.
Which match was against whom: result rows (runs/gauntlet/human.jsonl, runs/human/series.jsonl: opp, mine, theirs,
t) matched to logs by time order and final score (the earliest unused row written after the log started with the
same score). Unmatched / aborted / short logs are listed and left out of the pools.

Per rider (human, bot) and match: state shares (of the rider's in-play time) and entries per minute, events per
minute; pooled over the human's matches (all, vs each opponent group) with the value of each concept for the human
= own goal diff over the next 10 s after the moment minus the human's average in that pool (correlational; n =
moments / entries / events, flagged when small). Differences human - bot ranked by a paired t over matches.

    .venv/bin/python analysis/human_concepts.py                    # all runs/human logs
    .venv/bin/python analysis/human_concepts.py --logs 'runs/human/20261002-17*.jsonl' --out /tmp/x
"""
import argparse, glob, json, os, re, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import concepts as C  # noqa: E402

H = 10 * C.HZ               # value horizon: 10 s
BALL_R = 24.0 / 64          # m (sim/src/game/ball.cpp)
TOUCH_DV = 3.0              # m/s: unexplained ball velocity change that counts as a touch
TOUCH_D = 1.3               # m: ball centre to body / pole line (at either end of the snapshot interval)
LOW_N = 20                  # flag values with fewer events / state entries
NEVER = 999.0
EXTRA_STATES = ["still"]                       # not in concepts.py (see analyse_match)
EXTRA_EVENTS = ["ball_kick", "ball_touch"]
STATES = C.STATE_NAMES + EXTRA_STATES
EVENTS = C.EVENT_NAMES + EXTRA_EVENTS
OUTCOME_EVENTS = ("goal_for", "goal_against")  # results, left out of the style ranking

GLOSS = {  # plain-language meaning, tied to docs/strategy.md
    "good_side": "you-ball-goal: body behind the ball",
    "near_ball": "within half a pole of the ball (where farm1 got bad)",
    "tip_behind_ball": "near the ball, tip hooked to push it on (round 3: the tip decides)",
    "tip_past_ball": "near the ball, tip past it: can only pull it back",
    "sit_right_far": "good side with room to vault in",
    "sit_right_close": "good side but too close: time to back up",
    "sit_A": "situation A (you-opp-ball)",
    "sit_B": "situation B (opp-you-ball, 'you're fucked')",
    "sit_both_wrong": "both riders past the ball",
    "opp_out": "kill window: opponent dead / respawning",
    "opp_wrong_side": "opponent on his wrong side",
    "airborne": "feet off everything",
    "pole_planted": "tip on the floor (slingshot loading)",
    "in_pit": "in a pit",
    "pole_fwd_down": "pole forward-down (normal low carry)",
    "pole_fwd_up": "pole forward-up",
    "pole_back_up": "pole back-up (hooked to push near the ball)",
    "pole_back_down": "pole back-down",
    "backswing_in_flight": "backward (CCW) swing in flight: prolong / fine-tune a jump",
    "pole_on_opp": "my pole on the opponent (pressing)",
    "opp_pole_on_me": "his pole on me",
    "pole_over_opp_head": "pole over his head (A option 2: kill threat)",
    "opp_on_my_pole": "he stands on my pole (throw setup)",
    "on_opp_pole": "I stand on his pole (he can throw me)",
    "tip_peek": "tip peeking past the ball on his side",
    "rhythmic_pressing": "B defence: rhythmic press/release",
    "committed": "committed: in the air touching nothing (touch part inferred)",
    "at_mercy": "committed while he can act near ball / me",
    "high_energy": "energy >= 165 J/kg (farm2's top quarter in the sim)",
    "goal_for": "goal scored", "goal_against": "goal conceded", "kill": "kill", "death": "death",
    "back_pass": "pole over the top backward", "fwd_pass": "pole over the top forward",
    "vault": "slingshot vault", "pogo_vault": "landing on the pole into the next vault",
    "jump_over_ball": "A option 1: long jump over the ball", "pushed_opp_wrong_side": "pushed him to the wrong side (GG)",
    "tip_got_behind_ball": "re-hooked the tip behind the ball", "catapult": "the throw: flung him off my pole",
    "peek_touch": "peek pays off: tip touches him as he goes for the ball",
    "takeoff": "every jump", "fast_takeoff": "jump at >= 14 m/s",
    "lunge_at_opp": "jump at the opponent from in front of the ball", "lunge_from_pit": "lunge from my own hole",
    "overcommit": "lunge backfired: flipped to [ball, opp, me]", "overcommit_opp_still": "overcommit while he did nothing",
    "got_flung": "thrown off his pole", "flung_own_goal": "flung, then own goal (inferred touch)",
    "second_touch": "hit the ball second (INFERRED touches)", "second_kick": "kicked second (exact kicks)",
    "impale_plant": "impale attempt (flying backward, tip into the floor back-down)",
    "impale_brake": "impale that shed >= 3 m/s",
    "still": "doing nothing: body < 2 m/s and pole turning < 1.5 rad/s (overcommit_opp_still's test)",
    "ball_kick": "ball kicks (exact, game stats)", "ball_touch": "ball touches incl. kicks (INFERRED)",
    "mutual_pole_contact": "each pole on the other rider (now)", "interlocked": "interlocked >= 1 s (poles on each other)",
    "pole_interlock": "an interlock ended", "interlock_let_go": "interlocked with my pole servo off (stick centred)",
    "let_go_escape": "let go to end an interlock", "open_goal": "open goal: he's dead / > 5.5 m on his wrong side",
    "open_goal_conversion": "scored from an open goal", "aim_back_pass": "stick sweeps the pole aim backward over the top",
}


# ------------------------------------------------------------------------------------------------ matching

def log_start(path):
    m = re.search(r"(\d{8}-\d{6})", os.path.basename(path))
    return time.mktime(time.strptime(m.group(1), "%Y%m%d-%H%M%S")) if m else os.path.getmtime(path)


def group_of(opp):
    """Opponent group: "farm2-family" (farm2, farm2@200M), "search" (the ~20M search trials t27/t2/t7/t24), or a frozen
    long run on its own ("t20_300M (300M steps)" -> "t20@300M", the 2026-10-03 series)."""
    if opp is None:
        return None
    m = re.match(r"(t\d+)_(\d+M)\b", opp)
    if m:
        return f"{m.group(1)}@{m.group(2)}"
    if re.match(r"t\d+\b", opp):
        return "search"
    return opp.split("@")[0].split()[0] + "-family"


def load_results(paths):
    rows = []
    for p in paths:
        if not os.path.exists(p):
            continue
        src = "gauntlet" if "gauntlet" in p else ("series" if "series" in p else os.path.basename(p))
        for line in open(p):
            try:
                r = json.loads(line)
            except ValueError:
                continue
            rows.append({"opp": r.get("opp"), "mine": r.get("mine"), "theirs": r.get("theirs"), "t": r.get("t", 0),
                         "src": src, "used": False})
    return sorted(rows, key=lambda r: r["t"])


def match_result(results, start, score, window=30 * 60):
    for r in results:
        if not r["used"] and start - 60 <= r["t"] <= start + window and (r["mine"], r["theirs"]) == tuple(score):
            r["used"] = True
            return r
    return None


# ------------------------------------------------------------------------------------------------ touch rows

def _seg_dist(px, py, ax, ay, bx, by):
    vx, vy = bx - ax, by - ay
    u = np.clip(((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy + 1e-9), 0, 1)
    return np.hypot(px - ax - u * vx, py - ay - u * vy)


def ball_distances(states):
    """[N, 2] distance from the ball centre to each rider's body / pole polyline (99 while dead)."""
    N = len(states)
    d = np.full((N, 2), 99.0)
    for t, s in enumerate(states):
        b = s["ball"]
        for k, p in enumerate(sorted(s["players"], key=lambda q: q["id"])):
            if p.get("dead"):
                continue
            P = np.array([[q["x"], q["y"]] for q in p["pole"]])
            pts = np.vstack([P[0] - 0.5 * (P[1] - P[0]), P, P[-1] + 0.5 * (P[-1] - P[-2])])
            dp = _seg_dist(b["x"], b["y"], pts[:-1, 0], pts[:-1, 1], pts[1:, 0], pts[1:, 1]).min()
            d[t, k] = min(dp, np.hypot(b["x"] - p["x"], b["y"] - p["y"]))
    return d


def touch_rows(states, ticks, idx):
    """Touch rows [T, 2 views, 2 (me, opp), 5] in native_env's TOUCH_* layout for the picked snapshots idx, from all
    snapshots (see the module doc), + diagnostics."""
    N = len(states)
    pl = [sorted(s["players"], key=lambda q: q["id"]) for s in states]
    kicks = np.array([[p["stats"]["kicks"] for p in ps] for ps in pl], np.int64)
    kills = np.array([[p["stats"]["kills"] for p in ps] for ps in pl], np.int64)
    goals = np.array([[p["stats"]["goals"] for p in ps] for ps in pl], np.int64)
    owng = np.array([[p["stats"]["own_goals"] for p in ps] for ps in pl], np.int64)
    dk = np.zeros((N, 2), np.int64); dk[1:] = np.diff(kicks, axis=0)
    dkill = np.zeros((N, 2), np.int64); dkill[1:] = np.diff(kills, axis=0)
    ball_kick = np.maximum(dk - dkill, 0) > 0                     # [N, 2] a ball kick shows at snapshot i
    b = np.array([[s["ball"]["x"], s["ball"]["y"], s["ball"]["vx"], s["ball"]["vy"]] for s in states])
    tr = np.array([s.get("trolley", [s["ball"]["x"], s["ball"]["y"]]) for s in states])
    dist = ball_distances(states)
    dt = np.diff(ticks) / 60.0
    tl = np.array([s["time_left"] for s in states])
    running = np.zeros(N - 1, bool); running[:] = (tl[:-1] - tl[1:]) > 1e-4
    dv = b[1:, 2:] - b[:-1, 2:]
    X = np.hstack([b[:-1, :2] - tr[:-1], b[:-1, 2:], np.ones((N - 1, 1))])
    far = (dist.min(1)[1:] > 2.5) & (dist.min(1)[:-1] > 2.5) & running & (dt > 0)
    acc = dv / np.maximum(dt, 1e-6)[:, None]
    coef = np.zeros((2, 5))
    if far.sum() > 50:
        for j in range(2):
            coef[j], *_ = np.linalg.lstsq(X[far], acc[far, j], rcond=None)
    res = np.hypot(*(dv - (X @ coef.T) * dt[:, None]).T)          # [N-1] interval i-1 -> i at index i-1
    jump = np.zeros(N, bool); jump[1:] = res > TOUCH_DV            # shows at snapshot i
    dm = np.full((N, 2), 99.0); dm[1:] = np.minimum(dist[1:], dist[:-1])
    near = dm < TOUCH_D
    kick_near = np.zeros((N, 2), bool)                            # a ball kick at i-1, i or i+1
    for s_ in (-1, 0, 1):
        kick_near |= np.roll(ball_kick, s_, axis=0)
    touch = np.zeros((N, 2), bool)
    cand = jump & near.any(1)
    kicker = kick_near & cand[:, None]
    by_kick = kicker.any(1)
    touch[by_kick] = kicker[by_kick]
    nk = cand & ~by_kick
    nearest = dm.argmin(1)
    touch[nk, nearest[nk]] = True
    touch |= ball_kick                                            # a kick is a touch
    # the game's lastTouch: the last toucher; at a goal the scorer (goals) or the own-goaler (own_goals)
    last = np.full(N, -1)
    cur = -1
    dg = np.zeros((N, 2), np.int64); dg[1:] = np.diff(goals + owng, axis=0)
    for i in range(N):
        if touch[i].any():
            cur = int(np.argmax(touch[i])) if touch[i].sum() == 1 else int(dm[i].argmin())
        if dg[i].any():
            cur = int(np.argmax(dg[i]))
        last[i] = cur

    def ages(ev):
        """seconds since the last ev (per rider) at each picked snapshot."""
        out = np.full((len(idx), 2), NEVER, np.float32)
        lt = np.full(2, -np.inf)
        j = 0
        for i in range(N):
            lt = np.where(ev[i], ticks[i], lt)
            while j < len(idx) and idx[j] == i:
                out[j] = np.where(np.isfinite(lt), (ticks[i] - lt) / 60.0, NEVER)
                j += 1
        return out
    t_age, k_age = ages(touch), ages(ball_kick)
    gl = last[idx]
    TC = np.zeros((len(idx), 2, 2, 5), np.float32)
    for v in (0, 1):
        for slot, r in ((0, v), (1, 1 - v)):
            TC[:, v, slot, 0] = t_age[:, r]
            TC[:, v, slot, 2] = k_age[:, r]
            TC[:, v, slot, 4] = gl == r
    # diagnostics: kicks with an inferred velocity jump credited to the kicker within +-1 snapshot
    jumpw = jump | np.roll(jump, 1) | np.roll(jump, -1)
    kk = np.nonzero(ball_kick)
    rec = float(np.mean([jumpw[i] and (dm[max(0, i - 1):i + 2, k].min() < TOUCH_D) for i, k in zip(*kk)])) if len(kk[0]) else None
    diag = {"ball_kicks": ball_kick.sum(0).tolist(), "stomp_kicks_removed": int(np.minimum(dk, dkill).clip(0).sum()),
            "inferred_touches": (touch & ~ball_kick).sum(0).tolist(),
            "kicks_seen_as_velocity_jump_near_kicker": rec,
            "velocity_jumps_with_nobody_near": int((jump & ~near.any(1)).sum()),
            "free_motion_residual_p50_p99": [float(np.percentile(res[far[:len(res)]], q)) for q in (50, 99)] if far.any() else None,
            "touch_dv": TOUCH_DV, "touch_d": TOUCH_D}
    return TC, diag


# ------------------------------------------------------------------------------------------------ one match

def stick_activity(path, frames_wanted, pad):
    """Share of 1/15 s steps in which the human's sticks moved (> 0.05 on any axis since the previous step)."""
    want = set(int(f) for f in frames_wanted)
    ax = {}
    for line in open(path):
        if '"s"' not in line:
            continue
        try:
            r = json.loads(line)
        except ValueError:
            continue
        if r["f"] in want:
            ax[r["f"]] = r["ax"][pad]
    a = np.array([ax[f] for f in frames_wanted if f in ax])
    return float((np.abs(np.diff(a, axis=0)).max(1) > 0.05).mean()) if len(a) > 2 else None


def analyse_match(path, mid, frames, states, kinds):
    idx, same_r, info = C.playlog_resample(frames, states)
    picked = [states[i] for i in idx]
    O, live, same = C.obs_from_states(picked)
    same &= same_r[:, None]
    pads = {p["id"]: p.get("pad") for p in states[-1]["players"]}
    humans = np.array([pads.get(i) is not None and pads[i] < len(kinds) and kinds[pads[i]] in (1, 3, 4) for i in (1, 2)])
    ticks, _, _ = C.playlog_ticks(frames, states)
    TC, tdiag = touch_rows(states, ticks, idx)
    ACT = C.playlog_actions(path, frames[idx], pads)   # both riders' sticks in the bots' terms (pole servo concepts)
    F = C.Frames(O, live, same, touch=TC, act=ACT)
    S, E = C.state_concepts(F), C.event_concepts(F)
    S["still"] = F.alive & (F.speed < 2.0) & (np.abs(O[..., C.PW]) < 1.5)
    E["ball_kick"] = C._ev(F.kicked)
    E["ball_touch"] = C._ev(C._onset(F.touched, F.same) | (F.touched & ~F.same))
    T = len(O)
    G = np.zeros((T, 2))
    G[1:] = (np.diff(O[:, :, C.MY_SCORE], axis=0) - np.diff(O[:, :, C.OPP_SCORE], axis=0)) * same[1:]
    cg = np.concatenate([np.zeros((1, 2)), np.cumsum(G, 0)])
    fut = cg[np.minimum(np.arange(T) + H, T)] - cg[np.arange(T)]
    inside = live & (np.arange(T)[:, None] + H <= T)
    last = picked[-1]
    score = [int(v) for v in last["score"]]
    stats = {p["id"]: p["stats"] for p in last["players"]}
    # sanity: velocity vs finite differences at the 1/15 s cadence, positions, pole tip distance
    ok = same[1:] & F.alive[1:] & F.alive[:-1] & (np.abs(F.vx[1:]) > 2)
    fd = ((F.x[1:] - F.x[:-1]) / C.STEP_S)[ok] / (0.5 * (F.vx[1:] + F.vx[:-1]))[ok]
    tipd = np.hypot(O[..., C.TIPDX], O[..., C.TIPDY])[F.alive]
    sanity = {"rows": T, "play_seconds": info["seconds"], "ticks_per_snapshot": info["ticks_per_snapshot"],
              "step_ticks": info["step_ticks"], "aborted": info["aborted"], "missing_snapshots": info["missing_snapshots"],
              "start_time_left": float(states[0]["time_left"]), "fd_over_v_median": float(np.median(fd)) if len(fd) else None,
              "x_range": [float(F.x[F.alive].min()), float(F.x[F.alive].max())] if F.alive.any() else None,
              "ball_y_range": [float(F.by[live].min()), float(F.by[live].max())],
              "tip_dist_p1_p99": [float(np.percentile(tipd, 1)), float(np.percentile(tipd, 99))] if len(tipd) else None,
              "touch": tdiag}
    return {"path": path, "match_id": mid, "O": O, "live": live, "same": same, "F": F, "S": S, "E": E, "fut": fut,
            "inside": inside, "humans": humans, "score": score, "stats": stats, "sanity": sanity, "idx": idx,
            "frames": frames[idx], "pads": pads}


# ------------------------------------------------------------------------------------------------ statistics

def rider_view(m, r):
    """Arrays of one rider's view (r = 0 rider 1, 1 rider 2)."""
    F = m["F"]
    return {"alive": F.alive[:, r], "live": m["live"][:, r], "same": m["same"][:, r], "fut": m["fut"][:, r],
            "inside": m["inside"][:, r], "S": {k: v[:, r] for k, v in m["S"].items()},
            "E": {k: v[:, r] for k, v in m["E"].items()}, "energy": F.energy[:, r], "oenergy": F.oenergy[:, r],
            "speed": F.speed[:, r], "opp_alive": F.opp_alive[:, r], "ball_vx": m["O"][:, r, 54],
            "takeoff_speed": C.takeoff_speed(F)[:, r], "ospeed": F.ospeed[:, r], "sit": C.situation(F)[:, r],
            "oground": F.oground[:, r]}


def rates(v):
    """Shares / rates of one rider in one match (or a pooled list of views)."""
    vs = v if isinstance(v, list) else [v]
    mins = sum(x["live"].sum() for x in vs) * C.STEP_S / 60
    alive = sum(x["alive"].sum() for x in vs)
    out = {"minutes": float(mins), "states": {}, "events": {}}
    for n in STATES:
        on = sum(x["S"][n].sum() for x in vs)
        ent = sum(C._onset(x["S"][n], x["same"]).sum() for x in vs)
        out["states"][n] = {"share": float(on / max(1, alive)), "entries_per_min": float(ent / max(1e-9, mins)),
                            "entries": int(ent)}
    for n in EVENTS:
        k = sum((x["E"][n] > 0).sum() for x in vs)
        out["events"][n] = {"n": int(k), "per_min": float(k / max(1e-9, mins))}
    return out


def values(vs):
    """Value of each concept for the rider of views vs (pooled): own goal diff over the next 10 s minus the rider's
    baseline. States: mean over moments, se over entries (episode means); events: mean over events, se."""
    cat = lambda f: np.concatenate([f(x) for x in vs])  # noqa: E731
    fut, ok_s, inside = cat(lambda x: x["fut"]), cat(lambda x: x["alive"] & x["inside"]), cat(lambda x: x["inside"])
    base = float(fut[ok_s].mean())
    out = {"baseline_10s": base, "states": {}, "events": {}}
    for n in STATES:
        ep_means = []
        for x in vs:
            m = x["S"][n] & x["alive"] & x["inside"]
            if not m.any():
                continue
            seg = np.cumsum(C._onset(x["S"][n], x["same"]) | (m & ~np.roll(m, 1)))
            for sid in np.unique(seg[m]):
                ep_means.append(x["fut"][m & (seg == sid)].mean())
        m = cat(lambda x: x["S"][n] & x["alive"] & x["inside"])
        ep = np.array(ep_means)
        out["states"][n] = {"value": float(fut[m].mean() - base) if m.any() else None, "n_moments": int(m.sum()),
                            "n_entries": len(ep), "se": float(ep.std() / np.sqrt(len(ep))) if len(ep) > 1 else None,
                            "low_n": len(ep) < LOW_N}
    for n in EVENTS:
        m = cat(lambda x: (x["E"][n] > 0) & x["inside"])
        f = fut[m]
        out["events"][n] = {"value": float(f.mean() - base) if len(f) else None, "n": int(len(f)),
                            "se": float(f.std() / np.sqrt(len(f))) if len(f) > 1 else None, "low_n": len(f) < LOW_N}
    return out


def _within(ev, k):
    T = len(ev)
    c = np.concatenate([[0], np.cumsum(ev)])
    t = np.arange(T)
    return (c[np.minimum(t + 1 + k, T)] - c[t + 1]) > 0


def _later(a, k):
    out = np.full(a.shape, np.nan, np.float32)
    out[:len(a) - k] = a[k:]
    return out


def outcomes(vs):
    """Consequences: lunges / overcommits, kills -> goals, energy, take-off speed, second kick / touch, impales."""
    acc = {}

    def add(key, arr):
        acc.setdefault(key, []).append(arr)
    for x in vs:
        E, ok, ins = x["E"], x["alive"] & x["inside"], x["inside"]
        ga5, gf5 = _within(E["goal_against"] > 0, 5 * C.HZ), _within(E["goal_for"] > 0, 5 * C.HZ)
        for k, v in (("ok", ok), ("ins", ins), ("ga5", ga5), ("gf5", gf5), ("fut", x["fut"]), ("energy", x["energy"]),
                     ("lead", np.where(x["opp_alive"], x["energy"] - x["oenergy"], np.nan)),
                     ("tko", x["takeoff_speed"]), ("ball_my_way", _later(x["ball_vx"], C.HZ // 2) > 0),
                     ("ospeed", x["ospeed"]), ("sit", x["sit"]), ("still", x["S"]["still"]),
                     ("opp_air", x["opp_alive"] & ~x["oground"])):
            add(k, v)
        for n in EVENTS:
            add("E_" + n, E[n] > 0)
    a = {k: np.concatenate(v) for k, v in acc.items()}
    p = lambda b, m: (float(b[m].mean()) if m.any() else None)  # noqa: E731
    base = float(a["fut"][a["ok"]].mean())
    ins = a["ins"]
    o = {"n_rows": int(a["ok"].sum()), "p_goal_against_5s_baseline": p(a["ga5"], a["ok"]),
         "p_goal_for_5s_baseline": p(a["gf5"], a["ok"])}
    lu, oc, ocs = a["E_lunge_at_opp"], a["E_overcommit"], a["E_overcommit_opp_still"]
    o["lunge"] = {"n": int(lu.sum()), "overcommits": int(oc.sum()), "overcommit_per_lunge": float(oc.sum() / max(1, lu.sum())),
                  "p_goal_against_5s_after_lunge": p(a["ga5"], lu & ins), "p_goal_against_5s_after_overcommit": p(a["ga5"], oc & ins),
                  "opp_still_overcommits": int(ocs.sum()), "n_from_pit": int(a["E_lunge_from_pit"].sum()),
                  "opp_speed_at_overcommit_median": float(np.median(a["ospeed"][oc])) if oc.any() else None}
    kl, de = a["E_kill"], a["E_death"]
    o["kill"] = {"n": int(kl.sum()), "p_goal_for_5s_after": p(a["gf5"], kl & ins), "p_goal_against_5s_after": p(a["ga5"], kl & ins)}
    o["death"] = {"n": int(de.sum()), "p_goal_against_5s_after": p(a["ga5"], de & ins), "p_goal_for_5s_after": p(a["gf5"], de & ins)}

    def fixed_bins(x, mask, edges):
        idx = np.searchsorted(edges, x[mask])
        f = a["fut"][mask]
        return {"edges": list(edges), "n": [int((idx == i).sum()) for i in range(len(edges) + 1)],
                "value": [float(f[idx == i].mean() - base) if (idx == i).any() else None for i in range(len(edges) + 1)]}
    ok = a["ok"]
    o["energy_median"] = float(np.median(a["energy"][ok]))
    o["energy_quartiles"] = [float(q) for q in np.percentile(a["energy"][ok], [25, 50, 75])]
    o["share_high_energy"] = float((a["energy"][ok] >= C.E_HIGH).mean())
    o["energy_bins_farm2_sim_edges"] = fixed_bins(a["energy"], ok, [35.0, 100.0, 165.0])
    lm = ok & np.isfinite(a["lead"])
    o["energy_lead_median"] = float(np.median(a["lead"][lm])) if lm.any() else None
    o["energy_lead_bins"] = fixed_bins(a["lead"], lm, [-70.0, 0.0, 70.0])
    tk = a["E_takeoff"] & ins
    sp = np.nan_to_num(a["tko"])
    o["takeoff_speed_median"] = float(np.median(sp[a["E_takeoff"]])) if a["E_takeoff"].any() else None
    o["takeoff_speed_quartiles"] = [float(q) for q in np.percentile(sp[a["E_takeoff"]], [25, 50, 75])] if a["E_takeoff"].any() else None
    o["takeoff_speed_bins_farm2_sim_edges"] = fixed_bins(sp, tk, [9.0, 11.9, 14.1])
    for name in ("second_kick", "second_touch"):
        m = a["E_" + name] & ins
        o[name] = {"n": int(a["E_" + name].sum()), "p_ball_my_way_0.5s": p(a["ball_my_way"], m),
                   "p_goal_for_5s": p(a["gf5"], m), "p_goal_against_5s": p(a["ga5"], m)}
    o["impale"] = {"plants": int(a["E_impale_plant"].sum()), "brakes": int(a["E_impale_brake"].sum())}
    st, sit, alive = a["still"], a["sit"], a["sit"] >= 0
    o["still"] = {"share": float(st.sum() / max(1, alive.sum())),
                  "share_while_opp_airborne": float((st & a["opp_air"]).sum() / max(1, st.sum())),
                  "by_situation": {nm: float((st & (sit == c)).sum() / max(1, (sit == c).sum()))
                                   for c, nm in enumerate(["right_far", "right_close", "A", "B", "both_wrong", "opp_out"])}}
    return o


def paired(per_match, getter):
    """Paired comparison over matches: (mean diff, t, n_matches with diff > 0, n)."""
    d = np.array([getter(m) for m in per_match], float)
    d = d[np.isfinite(d)]
    if len(d) < 2:
        return None
    sd = d.std(ddof=1)
    return {"mean_diff": float(d.mean()), "t": float(d.mean() / (sd / np.sqrt(len(d)))) if sd > 0 else None,
            "pos": int((d > 0).sum()), "n": len(d)}


def welch(a, b):
    a, b = np.array(a, float), np.array(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else None


# ------------------------------------------------------------------------------------------------ report

def metric_list():
    return ([("state", n, "share") for n in STATES] + [("event", n, "per_min") for n in EVENTS])


def get_metric(r, kind, name):
    return r["states"][name]["share"] if kind == "state" else r["events"][name]["per_min"]


def fmt_metric(kind, v):
    return f"{100 * v:5.1f}%" if kind == "state" else f"{v:6.2f}"


def sim_reference():
    """Bot numbers from the native-sim concept report (analysis/concepts_report_v2.json), for the sanity check."""
    p = os.path.join(HERE, "concepts_report_v2.json")
    if not os.path.exists(p):
        return {}
    r = json.load(open(p))["runs"]
    out = {}
    for key, who, label in (("farm2 self-play", "farm2", "farm2 sim self-play"),):
        if key in r and who in r[key]:
            out[label] = r[key][who]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default=os.path.join(HERE, "..", "runs", "human", "*.jsonl"))
    p.add_argument("--results", nargs="*", default=[os.path.join(HERE, "..", "runs", "gauntlet", "human.jsonl"),
                                                     os.path.join(HERE, "..", "runs", "human", "series.jsonl")])
    p.add_argument("--opp", action="append", default=[], help="LOGNAME=opponent override (also includes the log)")
    p.add_argument("--min-play-s", type=float, default=60.0)
    p.add_argument("--out", default=os.path.join(HERE, "human_concepts_report"))
    a = p.parse_args()
    overrides = dict(x.split("=", 1) for x in a.opp)
    results = load_results(a.results)
    logs = sorted(f for f in glob.glob(a.logs) if not f.endswith("series.jsonl"))
    matches, skipped = [], []
    for path in logs:
        name = os.path.basename(path)
        for mid, (frames, states, kinds) in C.playlog_read(path).items():
            t0 = time.time()
            m = analyse_match(path, mid, frames, states, kinds)
            san = m["sanity"]
            h = np.nonzero(m["humans"])[0]
            why = None
            if san["aborted"]:
                why = f"aborted: the log ends after {san['play_seconds']:.0f} s of play, score {m['score']}"
            elif san["play_seconds"] < a.min_play_s:
                why = f"only {san['play_seconds']:.0f} s of play"
            elif len(h) != 1:
                why = f"{len(h)} human riders"
            res = None
            if why is None:
                hr = int(h[0])
                mine, theirs = m["score"][hr], m["score"][1 - hr]
                if name in overrides:
                    res = {"opp": overrides[name], "mine": mine, "theirs": theirs, "src": "override", "t": None}
                else:
                    res = match_result(results, log_start(path), (mine, theirs))
                if res is None:
                    why = (f"no result row with score {mine}-{theirs} after the log started (format "
                           f"{san['start_time_left']:.0f} s, pads kinds {kinds}): a test match, opponent unknown")
            if why:
                skipped.append({"log": name, "match_id": mid, "why": why, "sanity": san})
                print(f"skip {name} m{mid}: {why}", flush=True)
                continue
            m.update(human=hr, bot=1 - hr, opp=res["opp"], group=group_of(res["opp"]), src=res["src"],
                     result=(res["mine"], res["theirs"]), log=name)
            print(f"{name} m{mid}: human {mine}-{theirs} {res['opp']} ({res['src']}), {san['play_seconds']:.0f} s, "
                  f"{time.time() - t0:.1f} s", flush=True)
            matches.append(m)
    if not matches:
        sys.exit("no usable matches")

    # ---- per match / pooled numbers
    for m in matches:
        m["hv"], m["bv"] = rider_view(m, m["human"]), rider_view(m, m["bot"])
        m["hr"], m["br"] = rates(m["hv"]), rates(m["bv"])
        hp = m["pads"][m["human"] + 1]
        m["stick_activity"] = stick_activity(m["path"], list(m["frames"]), hp)
    groups = sorted(set(m["group"] for m in matches))
    pools = {"all": matches, **{g: [m for m in matches if m["group"] == g] for g in groups}}
    # group comparisons (Welch t over matches): the farm2 family vs the search trials, and every frozen long run
    # (e.g. t20@300M) vs all the other matches
    comparisons = []
    if "search" in groups and any(g.endswith("-family") for g in groups):
        comparisons.append(([g for g in groups if g.endswith("-family")][0], "search"))
    for g in groups:
        if "@" in g and len(pools[g]) < len(matches):
            pools["not " + g] = [m for m in matches if m["group"] != g]
            comparisons.append((g, "not " + g))
    pooled = {}
    for pn, ms in pools.items():
        pooled[pn] = {"matches": len(ms), "human": rates([m["hv"] for m in ms]), "bot": rates([m["bv"] for m in ms]),
                      "human_values": values([m["hv"] for m in ms]), "bot_values": values([m["bv"] for m in ms]),
                      "human_outcomes": outcomes([m["hv"] for m in ms]), "bot_outcomes": outcomes([m["bv"] for m in ms])}

    # ---- differences
    diffs = []
    for kind, n, _ in metric_list():
        pr = paired(matches, lambda m: get_metric(m["hr"], kind, n) - get_metric(m["br"], kind, n))
        hv, bv = get_metric(pooled["all"]["human"], kind, n), get_metric(pooled["all"]["bot"], kind, n)
        cnt_h = pooled["all"]["human"]["events"][n]["n"] if kind == "event" else pooled["all"]["human"]["states"][n]["entries"]
        cnt_b = pooled["all"]["bot"]["events"][n]["n"] if kind == "event" else pooled["all"]["bot"]["states"][n]["entries"]
        row = {"kind": kind, "name": n, "human": hv, "bot": bv, "ratio": (hv / bv) if bv > 0 else None,
               "count_human": cnt_h, "count_bot": cnt_b, "paired": pr}
        row["by_group"] = {g: {"human": get_metric(pooled[g]["human"], kind, n), "bot": get_metric(pooled[g]["bot"], kind, n)}
                           for g in pools if g != "all"}
        row["comparisons"] = {f"{ga}|{gb}": {
            "human_welch_t": welch([get_metric(m["hr"], kind, n) for m in pools[ga]], [get_metric(m["hr"], kind, n) for m in pools[gb]]),
            "bot_welch_t": welch([get_metric(m["br"], kind, n) for m in pools[ga]], [get_metric(m["br"], kind, n) for m in pools[gb]])}
            for ga, gb in comparisons}
        diffs.append(row)
    ranked = sorted([d for d in diffs if d["paired"] and d["paired"]["t"] is not None
                     and d["count_human"] + d["count_bot"] >= 10 and d["name"] not in OUTCOME_EVENTS], key=lambda d: -abs(d["paired"]["t"]))

    # ---- text
    L = []
    w = L.append
    w(f"Human vs bot concept report ({time.strftime('%Y-%m-%d %H:%M')}): {len(matches)} real-game matches from "
      f"{len(set(m['log'] for m in matches))} play logs ({', '.join(f'{g}: {len(pools[g])}' for g in groups)}). "
      f"Concepts: concepts.py (docs/strategy.md); rows resampled to one per 4 game ticks (1/15 s) by the match clock.")
    w("Value = the human's own goal diff over the next 10 s after the moment minus the human's average over the pool "
      "(correlational; n = events, or entries for states; '*' = n < %d: a hint, not a finding)." % LOW_N)
    w("Touches: KICKS exact (game stats, stomp kills removed); other touches INFERRED from ball velocity jumps near a "
      "rider (second_touch, flung_own_goal, committed/at_mercy's ball part).")
    w("\n== Data sanity")
    for s in skipped:
        w(f"  excluded {s['log']} (match {s['match_id']}): {s['why']}")
    w(f"  {'log':26s} {'opp':16s} {'src':8s} {'score':>6s} {'result':>6s} {'play s':>6s} {'tk/snap':>7s} {'fd/v':>5s} "
      f"{'goals h/b (stats)':>18s} {'kills h/b concept|game':>23s} {'kicks h/b':>9s} {'jump@kick':>9s} {'stick act':>9s}")
    for m in matches:
        s, hr, br = m["sanity"], m["hr"]["events"], m["br"]["events"]
        st = m["stats"]
        sh, sb = st[m["human"] + 1], st[m["bot"] + 1]
        w(f"  {m['log']:26s} {m['opp'][:16]:16s} {m['src']:8s} {m['score'][m['human']]:>2d}-{m['score'][m['bot']]:<3d} "
          f"{m['result'][0]:>2d}-{m['result'][1]:<3d} {s['play_seconds']:6.0f} {s['ticks_per_snapshot']:7.1f} "
          f"{s['fd_over_v_median']:5.2f} {hr['goal_for']['n']:>3d}/{br['goal_for']['n']:<2d} ({sh['goals']}+{sb['own_goals']}og/"
          f"{sb['goals']}+{sh['own_goals']}og) {hr['kill']['n']:>3d}/{br['kill']['n']:<3d}|{sh['kills']:>2d}/{sb['kills']:<2d}      "
          f"{s['touch']['ball_kicks'][m['human']]:>3d}/{s['touch']['ball_kicks'][m['bot']]:<3d}   "
          f"{s['touch']['kicks_seen_as_velocity_jump_near_kicker'] or 0:7.2f}   {m['stick_activity'] or 0:7.2f}")
    allfd = [m["sanity"]["fd_over_v_median"] for m in matches]
    st_all = {}
    for m in matches:
        for k, v in m["sanity"]["step_ticks"].items():
            st_all[k] = st_all.get(k, 0) + v
    tot = sum(st_all.values())
    w(f"  cadence: the logs hold a snapshot every 4 rendered frames = {matches[0]['sanity']['ticks_per_snapshot']:.0f} "
      f"game ticks (120 Hz display); resampled step lengths (ticks): "
      + ", ".join(f"{k}: {100 * v / tot:.1f}%" for k, v in sorted(st_all.items()))
      + f"; finite-difference / reported velocity {min(allfd):.2f}-{max(allfd):.2f} (1.00 = rows are 1/15 s apart)")
    xr = [m["sanity"]["x_range"] for m in matches]
    tr_ = [m["sanity"]["tip_dist_p1_p99"] for m in matches]
    by = [m["sanity"]["ball_y_range"] for m in matches]
    w(f"  ranges: rider x {min(r[0] for r in xr):.1f}..{max(r[1] for r in xr):.1f} m, ball y {min(r[0] for r in by):.1f}.."
      f"{max(r[1] for r in by):.1f} m (zipline), pole tip distance p1-p99 {min(r[0] for r in tr_):.2f}-{max(r[1] for r in tr_):.2f} m")
    td = [m["sanity"]["touch"] for m in matches]
    w(f"  touches: {sum(sum(t['ball_kicks']) for t in td)} ball kicks ({sum(t['stomp_kicks_removed'] for t in td)} stomp-kill "
      f"'kicks' removed), {sum(sum(t['inferred_touches']) for t in td)} inferred non-kick touches, "
      f"{sum(t['velocity_jumps_with_nobody_near'] for t in td)} ball velocity jumps with nobody within {TOUCH_D} m; "
      f"kicks that show as a velocity jump near the kicker: {np.mean([t['kicks_seen_as_velocity_jump_near_kicker'] for t in td]):.2f}")
    ref = sim_reference()

    # pooled concept table
    hdr = "  {:22s} {:>16s} {:>16s}".format("", "human (all)", "bot (all)")
    for g in groups:
        hdr += f" {('human vs ' + g)[:16]:>16s} {('bot ' + g)[:16]:>16s}"
    for lab in ref:
        hdr += f" {lab[:19]:>19s}"
    hdr += f" {'human value (n)':>22s}"
    w("\n== Concepts pooled: states = share of in-play time / entries per min; events = per min")
    w(hdr)
    for kind, n, _ in metric_list():
        def cell(r):
            if kind == "state":
                s = r["states"][n]
                return f"{100 * s['share']:5.1f}% {s['entries_per_min']:6.2f}/m"
            return f"{r['events'][n]['per_min']:8.2f}/m ({r['events'][n]['n']})"
        line = f"  {n:22s} {cell(pooled['all']['human']):>16s} {cell(pooled['all']['bot']):>16s}"
        for g in groups:
            line += f" {cell(pooled[g]['human']):>16s} {cell(pooled[g]['bot']):>16s}"
        for lab, rr in ref.items():
            if n not in rr["states" if kind == "state" else "events"]:
                line += f" {'-':>19s}"
            elif kind == "state":
                s = rr["states"][n]
                line += f" {100 * s['share']:6.1f}% {s['entries_per_min']:6.2f}/m"
            else:
                line += f" {rr['events'][n]['per_min']:15.2f}/m "
        hv = pooled["all"]["human_values"]["states" if kind == "state" else "events"][n]
        nn = hv["n_entries"] if kind == "state" else hv["n"]
        vs = "n/a" if hv["value"] is None else f"{hv['value']:+.2f}+-{hv['se'] or 0:.2f}"
        line += f" {vs:>14s} ({nn}){'*' if hv['low_n'] else ''}"
        w(line)
    w(f"  (human baseline goal diff next 10 s: {pooled['all']['human_values']['baseline_10s']:+.3f}; bot "
      f"{pooled['all']['bot_values']['baseline_10s']:+.3f}; sim columns: analysis/concepts_report_v2 (touch-based "
      f"concepts there use exact touches: second_touch not comparable with the inferred ones here))")

    # values by pool
    w("\n== Value for the human by opponent group (goal diff next 10 s minus the pool's average; n; * = n < %d)" % LOW_N)
    w("  " + f"{'':22s}" + "".join(f" {g:>22s}" for g in groups))
    for kind, n, _ in metric_list():
        cells = []
        for g in groups:
            hv = pooled[g]["human_values"]["states" if kind == "state" else "events"][n]
            nn = hv["n_entries"] if kind == "state" else hv["n"]
            cells.append("n/a" if hv["value"] is None else f"{hv['value']:+.2f} ({nn}){'*' if hv['low_n'] else ''}")
        w(f"  {n:22s}" + "".join(f" {c:>22s}" for c in cells))

    # ranked differences
    w("\n== Biggest differences human - bot (paired over matches; t = mean diff / its se; 'h>b' = matches where the human is higher)")
    w(f"  {'concept':22s} {'kind':5s} {'human':>7s} {'bot':>7s} {'ratio':>6s} {'t':>6s} {'h>b':>6s}  meaning")
    for d in ranked[:30]:
        pr = d["paired"]
        w(f"  {d['name']:22s} {d['kind']:5s} {fmt_metric(d['kind'], d['human']):>7s} {fmt_metric(d['kind'], d['bot']):>7s} "
          f"{(d['ratio'] or 0):6.2f} {pr['t']:+6.1f} {pr['pos']:>3d}/{pr['n']:<2d}  {GLOSS.get(d['name'], '')}")
    for ga, gb in comparisons:
        key = f"{ga}|{gb}"
        w(f"\n== Opponent groups: {ga} ({len(pools[ga])} matches) vs {gb} ({len(pools[gb])}) — Welch t over matches"
          + ("  (few matches: |t| < 3 is noise)" if min(len(pools[ga]), len(pools[gb])) < 5 else ""))
        w(f"  {'concept':22s} {('human vs ' + ga)[:16]:>16s} {('human vs ' + gb)[:16]:>16s} {'t':>6s}   {('bot ' + ga)[:14]:>14s} "
          f"{('bot ' + gb)[:14]:>14s} {'t':>6s}")
        rows = [d for d in diffs if d["comparisons"][key]["bot_welch_t"] is not None or d["comparisons"][key]["human_welch_t"] is not None]
        rows.sort(key=lambda d: -max(abs(d["comparisons"][key]["bot_welch_t"] or 0), abs(d["comparisons"][key]["human_welch_t"] or 0)))
        for d in rows[:24]:
            bg, cp = d["by_group"], d["comparisons"][key]
            w(f"  {d['name']:22s} {fmt_metric(d['kind'], bg[ga]['human']):>16s} {fmt_metric(d['kind'], bg[gb]['human']):>16s} "
              f"{(cp['human_welch_t'] or 0):+6.1f}   {fmt_metric(d['kind'], bg[ga]['bot']):>14s} "
              f"{fmt_metric(d['kind'], bg[gb]['bot']):>14s} {(cp['bot_welch_t'] or 0):+6.1f}")

    # outcomes
    def f_(v, s="{:.2f}"):
        return "n/a" if v is None else s.format(v)
    w("\n== Outcomes (pooled; P = probability within 5 s)")
    for pn in ["all"] + groups:
        for who in ("human", "bot"):
            o = pooled[pn][who + "_outcomes"]
            lu, k, d_ = o["lunge"], o["kill"], o["death"]
            w(f"  [{pn}] {who}: P(goal for / against 5 s) baseline {f_(o['p_goal_for_5s_baseline'], '{:.3f}')} / "
              f"{f_(o['p_goal_against_5s_baseline'], '{:.3f}')}")
            w(f"    lunges {lu['n']} ({lu['n_from_pit']} from the pit) -> overcommits {lu['overcommits']} "
              f"({100 * lu['overcommit_per_lunge']:.0f}%), P(goal against) after an overcommit "
              f"{f_(lu['p_goal_against_5s_after_overcommit'])}, opponent still in {lu['opp_still_overcommits']}")
            w(f"    kills {k['n']}: P(goal for) after {f_(k['p_goal_for_5s_after'])}; deaths {d_['n']}: P(goal against) after "
              f"{f_(d_['p_goal_against_5s_after'])}")
            eb, tb, lb = o["energy_bins_farm2_sim_edges"], o["takeoff_speed_bins_farm2_sim_edges"], o["energy_lead_bins"]
            w(f"    energy median {o['energy_median']:.0f} J/kg (quartiles {', '.join(f'{q:.0f}' for q in o['energy_quartiles'])}), "
              f"high-energy share {100 * o['share_high_energy']:.0f}%; value by energy <35/35-100/100-165/>165: "
              + ", ".join(f"{f_(v, '{:+.2f}')} (n={n})" for v, n in zip(eb["value"], eb["n"])))
            w(f"    energy lead median {f_(o['energy_lead_median'], '{:+.0f}')}; value by lead <-70/-70-0/0-70/>70: "
              + ", ".join(f"{f_(v, '{:+.2f}')} (n={n})" for v, n in zip(lb["value"], lb["n"])))
            w(f"    take-off speed median {f_(o['takeoff_speed_median'], '{:.1f}')} m/s (quartiles "
              f"{', '.join(f'{q:.1f}' for q in (o['takeoff_speed_quartiles'] or []))}); value by <9/9-11.9/11.9-14.1/>14.1: "
              + ", ".join(f"{f_(v, '{:+.2f}')} (n={n})" for v, n in zip(tb["value"], tb["n"])))
            for nm in ("second_kick", "second_touch"):
                x = o[nm]
                w(f"    {nm}{' (inferred)' if nm == 'second_touch' else ''}: n={x['n']}, P(ball moving my way 0.5 s later) "
                  f"{f_(x['p_ball_my_way_0.5s'])}, P(goal for / against 5 s) {f_(x['p_goal_for_5s'])} / {f_(x['p_goal_against_5s'])}")
            w(f"    impale plants {o['impale']['plants']}, brakes {o['impale']['brakes']}")
            sl = o["still"]
            w(f"    still (doing nothing) {100 * sl['share']:.1f}% of in-play time, {100 * sl['share_while_opp_airborne']:.0f}% of it "
              f"while the opponent is airborne; share of each situation spent still: "
              + ", ".join(f"{k} {100 * v:.1f}%" for k, v in sl["by_situation"].items()))
        # who gets the last word: second kicks per kick of the other rider (an opportunity to reply)
        hk, bk = pooled[pn]["human"]["events"]["ball_kick"]["n"], pooled[pn]["bot"]["events"]["ball_kick"]["n"]
        hs, bs = pooled[pn]["human"]["events"]["second_kick"]["n"], pooled[pn]["bot"]["events"]["second_kick"]["n"]
        pooled[pn]["kick_reply"] = {"human_kicks": hk, "bot_kicks": bk, "human_second_kicks": hs, "bot_second_kicks": bs,
                                    "human_replies_per_bot_kick": hs / max(1, bk), "bot_replies_per_human_kick": bs / max(1, hk)}
        w(f"  [{pn}] kicks human {hk} / bot {bk}; second kicks per kick of the other: human {hs / max(1, bk):.2f} "
          f"({hs}/{bk}), bot {bs / max(1, hk):.2f} ({bs}/{hk})")

    # per match
    tables = [("states: share of in-play time, human/bot %",
               ["good_side", "tip_behind_ball", "tip_past_ball", "sit_A", "sit_B", "opp_out", "pole_fwd_up",
                "pole_back_down", "backswing_in_flight", "pole_on_opp", "at_mercy", "still"]),
              ("events per minute, human/bot",
               ["takeoff", "fast_takeoff", "back_pass", "kill", "death", "lunge_at_opp", "overcommit", "ball_kick",
                "second_kick", "catapult", "jump_over_ball", "impale_plant"]),
              ("2026-10-03 concepts — states: share of in-play time, human/bot %",
               ["mutual_pole_contact", "interlocked", "interlock_let_go", "open_goal", "sit_A"]),
              ("2026-10-03 concepts — events per minute, human/bot",
               ["pole_interlock", "let_go_escape", "open_goal_conversion", "aim_back_pass", "fwd_pass"])]
    w("\n== Per match (consistency; the user: less engaged in some matches; 'stick' = share of 1/15 s steps in which "
      "the human's sticks moved)")
    for title, names in tables:
        kind = "state" if "states" in title else "event"
        w(f"  -- {title}")
        w(f"  {'log':15s} {'opp':10s} {'score':>5s} {'min':>4s} {'stick':>5s}" + "".join(f" {n[:11]:>11s}" for n in names))
        for m in matches:
            line = (f"  {m['log'][9:24]:15s} {m['opp'][:10]:10s} {m['result'][0]:>2d}-{m['result'][1]:<2d} "
                    f"{m['hr']['minutes']:4.1f} {m['stick_activity'] or 0:5.2f}")
            for n in names:
                h, b_ = get_metric(m["hr"], kind, n), get_metric(m["br"], kind, n)
                line += f" {(f'{100 * h:.0f}/{100 * b_:.0f}' if kind == 'state' else f'{h:.1f}/{b_:.1f}'):>11s}"
            w(line)

    text = "\n".join(L) + "\n"
    open(a.out + ".txt", "w").write(text)
    report = {"generated": time.strftime("%Y-%m-%d %H:%M"), "horizon_s": H / C.HZ, "low_n": LOW_N,
              "touch_inference": {"dv": TOUCH_DV, "dist": TOUCH_D}, "skipped": skipped,
              "matches": [{"log": m["log"], "match_id": m["match_id"], "opp": m["opp"], "group": m["group"], "src": m["src"],
                           "result": m["result"], "score_from_log": [m["score"][m["human"]], m["score"][m["bot"]]],
                           "human_rider": m["human"] + 1, "game_stats": m["stats"], "sanity": m["sanity"],
                           "stick_activity": m["stick_activity"], "human": m["hr"], "bot": m["br"]} for m in matches],
              "pooled": pooled, "differences": diffs, "ranked": [d["name"] + ":" + d["kind"] for d in ranked],
              "gloss": GLOSS}
    json.dump(report, open(a.out + ".json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(text)
    print(f"-> {a.out}.txt / .json")


if __name__ == "__main__":
    main()
