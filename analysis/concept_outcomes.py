#!/usr/bin/env python3
"""What follows the 2026-10-03 concepts (concepts.py: pole_interlock / let_go_escape, open_goal, the situation classes,
the pole servo), as episode rows + summaries. Works on any concepts.Frames (sim rows [T, matches, views] or one play-log
match [T, 2 views]); used by analysis/t20_vs_user.py, analysis/concept_report.py and analysis/human_concepts.py.

Rows are dicts per episode seen from one view (`col` = the Frames batch index, me = that view's rider). Outcomes
count goals / ball kicks after the episode within a window, stopping at the end of the match segment. Kicks need touch
rows (sim: exact; play logs: exact kicks rebuilt from the game's stats, human_concepts.touch_rows)."""
import os, sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import concepts as C  # noqa: E402

HZ = C.HZ


class Ctx:
    """Per-Frames arrays shared by the row builders."""

    def __init__(self, F):
        self.F = F
        self.T = len(F.x)
        self.gf = C.goal_for(F) > 0
        self.ga = C.goal_against(F) > 0
        self.sit = C.situation(F)
        if F.touch is not None:
            self.kick = F.kicked
            self.okick = F.live & (F.ok_age < C.STEP_S - 1e-4)
        else:
            self.kick = self.okick = None
        # index of the last row of each row's contiguous segment (same chain)
        last = np.zeros(F.x.shape, np.int64)
        cur = np.full(F.x.shape[1:], self.T - 1, np.int64)
        for t in range(self.T - 1, -1, -1):
            last[t] = cur
            cur = np.where(F.same[t], cur, t - 1)
        self.seg_last = last

    def win(self, t0, k, col):
        """Rows (t0, t0 + k] of col within its segment, as a slice of time."""
        hi = min(t0 + k, int(self.seg_last[(max(t0, 0),) + col]))
        return slice(t0 + 1, max(t0 + 1, hi + 1))

    def any_in(self, a, t0, k, col):
        return bool(a[(self.win(t0, k, col),) + col].any())

    def first_in(self, a, t0, k, col):
        """Steps after t0 of the first true in (t0, t0 + k], or None."""
        sl = self.win(t0, k, col)
        nz = np.nonzero(a[(sl,) + col])[0]
        return int(nz[0]) + 1 if len(nz) else None


def _p(x):
    x = [v for v in x if v is not None]
    return float(np.mean(x)) if x else None


# ------------------------------------------------------------------------------------------------ interlocks

def interlock_rows(F, ctx=None, views=None):
    """Interlock episodes (concepts.interlock_episodes) + outcomes after the last contact: who kicks the ball first
    within 3 s (me / opp / none; None without touch rows), goals within 5 / 10 s. views: keep only these values of
    the last batch index (e.g. [0] for self-play, so each interlock counts once)."""
    ctx = ctx or Ctx(F)
    rows = []
    for ep in C.interlock_episodes(F):
        col, e = ep["col"], ep["last"]
        if views is not None and col[-1] not in views:
            continue
        r = dict(ep)
        if ctx.kick is not None:
            a, b = ctx.first_in(ctx.kick, e, 3 * HZ, col), ctx.first_in(ctx.okick, e, 3 * HZ, col)
            r["first_kick"] = "none" if a is None and b is None else ("me" if b is None or (a is not None and a <= b) else "opp")
        else:
            r["first_kick"] = None
        for k in (5, 10):
            r[f"gf{k}"] = ctx.any_in(ctx.gf, e, k * HZ, col)
            r[f"ga{k}"] = ctx.any_in(ctx.ga, e, k * HZ, col)
        t2 = min(e + 2 * HZ, int(ctx.seg_last[(e,) + col]))
        att = ep["attacker"]
        if att != 0:
            gs = (F.bx[(t2,) + col] > F.x[(t2,) + col]) if att > 0 else (F.ox[(t2,) + col] >= F.bx[(t2,) + col])
            r["attacker_still_good_2s"] = bool(gs)
        else:
            r["attacker_still_good_2s"] = None
        r["col"] = list(col)
        rows.append(r)
    return rows


def side_rows(rows):
    """Each interlock twice: from me and from the opponent (role, let go?, kicked first?, scored / conceded in 5 s)."""
    out = []
    for r in rows:
        for who, sg in (("me", 1), ("opp", -1)):
            role = {1: "attacker", -1: "defender", 0: "neutral"}[r["attacker"] * sg]
            top = {1: "top", -1: "bottom", 0: "level"}[r["top"] * sg]
            fk = r["first_kick"]
            out.append({"who": who, "role": role, "top": top, "let_go": r["my_let_go"] if sg > 0 else r["his_let_go"],
                        "other_let_go": r["his_let_go"] if sg > 0 else r["my_let_go"], "dur_s": r["dur_s"],
                        "kick_first": None if fk is None else fk == who,
                        "gf5": r["gf5"] if sg > 0 else r["ga5"], "ga5": r["ga5"] if sg > 0 else r["gf5"],
                        "gf10": r["gf10"] if sg > 0 else r["ga10"], "ga10": r["ga10"] if sg > 0 else r["gf10"]})
    return out


def summarize_interlocks(rows, minutes):
    """Frequency, duration, configuration, outcomes by role and by let-go."""
    n = len(rows)
    d = np.array([r["dur_s"] for r in rows]) if n else np.zeros(0)
    s = {"n": n, "per_min": n / max(minutes, 1e-9), "long_n": int((d >= C.INTERLOCK_LONG / HZ).sum()),
         "long_per_min": float((d >= C.INTERLOCK_LONG / HZ).sum() / max(minutes, 1e-9)),
         "share_of_time": float(d.sum() / 60 / max(minutes, 1e-9)),
         "dur_median_s": float(np.median(d)) if n else None, "dur_p90_s": float(np.percentile(d, 90)) if n else None,
         "dur_max_s": float(d.max()) if n else None,
         "stacked_share": _p([r["stacked"] for r in rows]),
         "attacker_at_start": {k: sum(1 for r in rows if r["attacker"] == v) for k, v in (("me", 1), ("opp", -1), ("none", 0))},
         "top": {k: sum(1 for r in rows if r["top"] == v) for k, v in (("me", 1), ("opp", -1), ("none", 0))},
         "let_go": {"me": sum(r["my_let_go"] for r in rows), "opp": sum(r["his_let_go"] for r in rows)},
         "attacker_still_good_2s": _p([r["attacker_still_good_2s"] for r in rows])}
    sr = side_rows(rows)

    def block(sel):
        return {"n": len(sel), "kick_first": _p([x["kick_first"] for x in sel]), "gf5": _p([x["gf5"] for x in sel]),
                "ga5": _p([x["ga5"] for x in sel]), "gf10": _p([x["gf10"] for x in sel]), "ga10": _p([x["ga10"] for x in sel])}
    s["by_role"] = {role: block([x for x in sr if x["role"] == role]) for role in ("attacker", "defender", "neutral")}
    s["by_top"] = {t: block([x for x in sr if x["top"] == t]) for t in ("top", "bottom", "level")}
    s["let_go_vs_not"] = {
        "released": block([x for x in sr if x["let_go"]]),
        "released_attacker": block([x for x in sr if x["let_go"] and x["role"] == "attacker"]),
        "released_top": block([x for x in sr if x["let_go"] and x["top"] == "top"]),
        "nobody_released": block([x for x in sr if not x["let_go"] and not x["other_let_go"]]),
        "nobody_released_attacker": block([x for x in sr if not x["let_go"] and not x["other_let_go"] and x["role"] == "attacker"]),
        "nobody_released_top": block([x for x in sr if not x["let_go"] and not x["other_let_go"] and x["top"] == "top"]),
        "other_released": block([x for x in sr if x["other_let_go"] and not x["let_go"]])}
    s["by_who"] = {w: {"let_go": sum(1 for x in sr if x["who"] == w and x["let_go"]),
                       "kick_first": _p([x["kick_first"] for x in sr if x["who"] == w]),
                       "gf5": _p([x["gf5"] for x in sr if x["who"] == w]), "ga5": _p([x["ga5"] for x in sr if x["who"] == w])}
                   for w in ("me", "opp")}
    return s


# ------------------------------------------------------------------------------------------------ open goal

OPEN_MIN = HZ // 2       # steps: windows shorter than 0.5 s are flicker
OPEN_MERGE = C.OPEN_GRACE


def merged_runs(b, same, gap, min_len=1):
    """runs_of(b) with gaps <= gap merged (same column, same segment), kept if >= min_len steps long."""
    runs = sorted(C.runs_of(b, same), key=lambda r: (r[0], r[1]))
    out = []
    for col, s, e in runs:
        if out and out[-1][0] == col and s - out[-1][2] - 1 <= gap and same[(slice(out[-1][2] + 1, s + 1),) + col].all():
            out[-1] = (col, out[-1][1], e)
        else:
            out.append((col, s, e))
    return [r for r in out if r[2] - r[1] + 1 >= min_len]


def open_goal_rows(F, ctx=None, views=None):
    """Open-goal windows (open_goal runs, gaps <= 1 s merged, >= 0.5 s): cause (opp dead / out of position), the ball's
    distance to the goal I attack at the start, converted (my goal before the window closed + 1 s), time to goal,
    goal within 5 s / 10 s of the start, goal against within 10 s."""
    ctx = ctx or Ctx(F)
    og = C.open_goal(F)
    rows = []
    for col, s, e in merged_runs(og, F.same, OPEN_MERGE, OPEN_MIN):
        if views is not None and col[-1] not in views:
            continue
        at = lambda a: a[(s,) + col]  # noqa: E731
        k_conv = e - s + C.OPEN_GRACE
        ttg = ctx.first_in(ctx.gf, s - 1, k_conv + 1, col)
        rows.append({"col": list(col), "start": s, "end": e, "dur_s": (e - s + 1) / HZ,
                     "cause": "dead" if not at(F.opp_alive) else "out_of_position",
                     "ball_to_goal": float(C.GOAL_X - at(F.bx)), "opp_dx": float(at(F.bx) - at(F.ox)) if at(F.opp_alive) else None,
                     "me_to_ball": float(at(F.bx) - at(F.x)),
                     "converted": ttg is not None, "time_to_goal_s": None if ttg is None else ttg / HZ,
                     "gf5": ctx.any_in(ctx.gf, s - 1, 5 * HZ + 1, col), "gf10": ctx.any_in(ctx.gf, s - 1, 10 * HZ + 1, col),
                     "ga10": ctx.any_in(ctx.ga, s - 1, 10 * HZ + 1, col)})
    return rows


def summarize_open_goal(rows, minutes):
    def block(sel):
        tt = [r["time_to_goal_s"] for r in sel if r["converted"]]
        return {"n": len(sel), "converted": _p([r["converted"] for r in sel]), "gf5": _p([r["gf5"] for r in sel]),
                "gf10": _p([r["gf10"] for r in sel]), "ga10": _p([r["ga10"] for r in sel]),
                "time_to_goal_median_s": float(np.median(tt)) if tt else None,
                "dur_median_s": float(np.median([r["dur_s"] for r in sel])) if sel else None}
    s = {"per_min": len(rows) / max(minutes, 1e-9), "all": block(rows),
         "dead": block([r for r in rows if r["cause"] == "dead"]),
         "out_of_position": block([r for r in rows if r["cause"] == "out_of_position"])}
    s["by_ball_to_goal"] = {lab: block([r for r in rows if lo <= r["ball_to_goal"] < hi])
                            for lab, lo, hi in (("<10 m", -99, 10), ("10-20 m (middle)", 10, 20), (">=20 m (my half)", 20, 99))}
    return s


# ------------------------------------------------------------------------------------------------ situations

SIT_NAMES = ["right_far", "right_close", "A", "B", "both_wrong", "opp_out"]


def situation_rows(F, cls=2, ctx=None, views=None, merge=3, min_len=5):
    """Episodes of situation `cls` from each view (gaps <= merge steps merged, >= min_len steps): duration, whether the
    opponent was within a pole (6 m) at the entry, goals for / against within 10 s of the entry, escape to the good
    side within 3 / 5 s, how it ended (the situation at the step after), interlocked share, pole rotation in it."""
    ctx = ctx or Ctx(F)
    il = C.interlocked(F)
    ob = C.jump_over_ball(F) > 0
    good = (ctx.sit == 0) | (ctx.sit == 1)
    rows = []
    for col, s, e in merged_runs(ctx.sit == cls, F.same, merge, min_len):
        if views is not None and col[-1] not in views:
            continue
        sl = (slice(s, e + 1),) + col
        nxt = int(ctx.sit[(e + 1,) + col]) if e + 1 <= int(ctx.seg_last[(s,) + col]) else None
        gf10, ga10 = ctx.any_in(ctx.gf, s - 1, 10 * HZ + 1, col), ctx.any_in(ctx.ga, s - 1, 10 * HZ + 1, col)
        sl10 = (ctx.win(s - 1, 10 * HZ + 1, col),) + col
        gd10 = int(ctx.gf[sl10].sum()) - int(ctx.ga[sl10].sum())
        esc3 = ctx.first_in(good, s, 3 * HZ, col)
        esc5 = ctx.first_in(good, s, 5 * HZ, col)
        alive = F.alive[sl]
        w = F.w[sl][alive]
        rows.append({"col": list(col), "start": s, "end": e, "dur_s": (e - s + 1) / HZ,
                     "close": bool(abs(F.x[(s,) + col] - F.ox[(s,) + col]) < 6.0),
                     "gf10": gf10, "ga10": ga10, "gd10": gd10, "escaped_3s": esc3 is not None, "escaped_5s": esc5 is not None,
                     "ga10_before_escape": ga10 and (esc5 is None),
                     "exit": "end" if nxt is None else ({-1: "me_out"}.get(nxt) or SIT_NAMES[nxt]),
                     "jump_over": bool(ob[(slice(s, min(e + C.HZ, int(ctx.seg_last[(s,) + col])) + 1),) + col].any()),
                     "interlocked_share": float(il[sl].mean()), "back_rot_share": float((w > 1.0).mean()) if len(w) else None,
                     "fwd_rot_share": float((w < -1.0).mean()) if len(w) else None})
    return rows


def summarize_situation(rows, minutes, alive_minutes=None):
    def block(sel):
        return {"n": len(sel), "dur_median_s": float(np.median([r["dur_s"] for r in sel])) if sel else None,
                "time_s": float(sum(r["dur_s"] for r in sel)),
                "gd10": _p([r["gd10"] for r in sel]), "gf10": _p([r["gf10"] for r in sel]), "ga10": _p([r["ga10"] for r in sel]),
                "escaped_3s": _p([r["escaped_3s"] for r in sel]), "escaped_5s": _p([r["escaped_5s"] for r in sel]),
                "ga10_before_escape": _p([r["ga10_before_escape"] for r in sel]),
                "jump_over": _p([r["jump_over"] for r in sel]), "interlocked_share": _p([r["interlocked_share"] for r in sel]),
                "back_rot_share": _p([r["back_rot_share"] for r in sel]), "fwd_rot_share": _p([r["fwd_rot_share"] for r in sel]),
                "exit": {k: sum(1 for r in sel if r["exit"] == k) for k in ["right_far", "right_close", "B", "both_wrong",
                                                                           "opp_out", "me_out", "end"]}}
    s = {"entries_per_min": len(rows) / max(minutes, 1e-9), "all": block(rows),
         "close": block([r for r in rows if r["close"]]), "far": block([r for r in rows if not r["close"]]),
         "pole_back_ge30": block([r for r in rows if (r["back_rot_share"] or 0) >= 0.3]),
         "pole_back_lt30": block([r for r in rows if (r["back_rot_share"] or 0) < 0.3])}
    if alive_minutes:
        s["share_of_time"] = float(sum(r["dur_s"] for r in rows) / 60 / alive_minutes)
    return s


def baseline(F, ctx=None, k=10):
    """P(goal for / against within k s) and mean goal diff over k s from a random in-play moment of the views."""
    ctx = ctx or Ctx(F)
    T = ctx.T
    t = np.arange(T).reshape((T,) + (1,) * (F.x.ndim - 1))
    lim = np.minimum(t + k * HZ, ctx.seg_last)

    def cnt(a):
        c = np.concatenate([np.zeros((1,) + a.shape[1:]), np.cumsum(a, 0)])
        return np.take_along_axis(c, lim + 1, 0) - c[1:]
    gf, ga = cnt(ctx.gf), cnt(ctx.ga)
    ok = F.alive & (t + k * HZ <= ctx.seg_last)
    return {"p_gf": float((gf[ok] > 0).mean()), "p_ga": float((ga[ok] > 0).mean()), "gd": float((gf - ga)[ok].mean()),
            "n": int(ok.sum())}


# ------------------------------------------------------------------------------------------------ pole rotation / servo

def _wrap(d):
    return (d + 180.0) % 360.0 - 180.0


def rotation_arrays(F, mask):
    """The per-moment pieces of rotation_stats for the alive moments of `mask` (pool several with rotation_stats)."""
    m = mask & F.alive
    out = {"w": F.w[m], "bp": (C.back_pass(F) > 0)[m], "fp": (C.fwd_pass(F) > 0)[m]}
    if F.act is not None:
        on = ~F.released & np.isfinite(F.aim)
        a0, a1 = F.aim[:-1], F.aim[1:]
        on2 = np.zeros(F.x.shape, bool)
        on2[1:] = F.same[1:] & on[1:] & on[:-1] & F.alive[1:] & F.alive[:-1]
        abp = np.zeros(F.x.shape, bool); afp = np.zeros(F.x.shape, bool)
        abp[1:] = on2[1:] & (a0 < 90) & (a1 >= 90) & (a1 - a0 < 90)
        afp[1:] = on2[1:] & (a0 >= 90) & (a1 < 90) & (a0 - a1 < 90)
        out.update({"released": F.released[m], "offset": _wrap(F.aim - F.ang)[m & on], "aim_on": F.aim[m & on],
                    "abp": abp[m], "afp": afp[m]})
    return out


def rotation_stats(parts):
    """How the pole turns in the given moments: `parts` = [rotation_arrays(F, mask), ...] (or one Frames-mask pair as
    (F, mask)). Share of time it turns backward (w > 1 rad/s, counter-clockwise in the attack frame) / forward
    (w < -1), back / fwd passes over the top per minute; with act rows: the servo-off share, where the stick aims
    relative to the pole (offset = aim - pole angle wrapped to +-180: > +30 drives the pole backward, < -30 forward;
    the servo always takes the short way), and stick sweeps over the top (aim passes) per minute."""
    if isinstance(parts, tuple):
        parts = [rotation_arrays(*parts)]
    cat = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]} if parts else {}
    n = len(cat.get("w", []))
    if n == 0:
        return {"n_steps": 0}
    mins = n * C.STEP_S / 60
    w = cat["w"]
    out = {"n_steps": n, "minutes": mins, "w_back_share": float((w > 1.0).mean()), "w_fwd_share": float((w < -1.0).mean()),
           "w_median": float(np.median(w)), "back_pass_per_min": float(cat["bp"].sum() / mins),
           "fwd_pass_per_min": float(cat["fp"].sum() / mins)}
    if "offset" in cat:
        off, k = cat["offset"], len(cat["offset"])
        aim = cat["aim_on"]
        out.update({"servo_off_share": float(cat["released"].mean()),
                    "aim_drives_back_share": float((off > 30).mean()) if k else None,
                    "aim_drives_fwd_share": float((off < -30).mean()) if k else None,
                    "aim_holds_share": float((np.abs(off) <= 30).mean()) if k else None,
                    "aim_offset_p10_p50_p90": [float(q) for q in np.percentile(off, [10, 50, 90])] if k else None,
                    "aim_back_pass_per_min": float(cat["abp"].sum() / mins), "aim_fwd_pass_per_min": float(cat["afp"].sum() / mins),
                    "aim_back_half_share": float(((aim >= 90) & (aim < 270)).mean()) if k else None})
    return out
