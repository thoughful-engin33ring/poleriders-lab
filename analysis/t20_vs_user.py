#!/usr/bin/env python3
"""t20 (300M) vs the user: the 2026-10-03 real-game series + the 2026-10-03 concepts (pole_interlock, let_go_escape,
open_goal, the bot-behind situation) and the pole-rotation question, compared with t20 / farm2 self-play in the sim.

    .venv/bin/python analysis/t20_vs_user.py gen --name t20 --ckpt runs/gauntlet/ckpt/t20_300M.pt --matches 128
    .venv/bin/python analysis/t20_vs_user.py gen --name farm2 --ckpt runs/gauntlet/ckpt/farm2.pt --matches 128
    .venv/bin/python analysis/t20_vs_user.py calib --ckpt runs/gauntlet/ckpt/t20_300M.pt --matches 8
    .venv/bin/python analysis/t20_vs_user.py report        # -> analysis/t20_vs_user_report.txt / .json

`gen` plays self-play matches in the native sim (7 goals / 5 min, no sudden death, the training rules; frozen
checkpoint, sampled actions as in the real game) through analysis/concept_report.play and caches observations, goal
terms, touch rows and the actions taken in runs/concepts_cache/. `calib` plays a few matches with full snapshots
(pole polylines) to check the straight-line pole contact proxies against the real geometry. `report` reads the
caches + every runs/human play log (analysis/human_concepts.py's matching and touch rebuilding).
"""
import argparse, glob, json, os, sys, time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.join(HERE, "..")
sys.path.insert(0, ROOT); sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(ROOT, "bridge"))
import concepts as C  # noqa: E402

CACHE = os.path.join(ROOT, "runs", "concepts_cache")


def cache_path(name, matches, seed):
    return os.path.join(CACHE, f"{name}_self_{matches}_s{seed}.npz")


def cmd_gen(a):
    import torch
    import concept_report as CR
    torch.set_num_threads(1)
    os.makedirs(CACHE, exist_ok=True)
    t0 = time.time()
    pol, ck = CR.load(a.ckpt)
    O, G, LIVE, TC, ACT = CR.play(pol, pol, a.matches, a.threads, a.seed, with_actions=True)
    out = cache_path(a.name, a.matches, a.seed)
    np.savez(out, O=O, G=G, LIVE=LIVE, TC=TC if TC is not None else np.zeros(0), ACT=ACT,
             steps=int(ck.get("steps", 0)), ckpt=os.path.abspath(a.ckpt))
    print(f"{a.name}: {a.matches} matches, {len(O)} steps, {time.time() - t0:.0f} s -> {out}", flush=True)


def cmd_calib(a):
    """Snapshots with the real pole polylines (t20 self-play) for the contact-proxy check."""
    import torch
    import concept_report as CR
    import eval_h2h as e
    from native_env import NativeHost
    torch.set_num_threads(1)
    os.makedirs(CACHE, exist_ok=True)
    pol, ck = CR.load(a.ckpt)
    env = NativeHost(1, a.matches, 4, sd_frac=0.0, seed=a.seed, threads=1, end_on_sd=True, match_format=e.FORMAT,
                     max_steps=e.MAX_STEPS, goal_refund=False)
    obs = env.reset_all().copy()
    done = np.zeros(a.matches, bool)
    snaps = [[] for _ in range(a.matches)]
    acts = [[] for _ in range(a.matches)]
    for k in range(a.matches):
        snaps[k].append(env.json(k))
    while not done.all():
        act = np.clip(e.sample(pol, obs.reshape(-1, obs.shape[-1])), -1, 1).reshape(a.matches, 2, 3).astype(np.float32)
        out = env.step(act)
        for k in np.nonzero(~done)[0]:
            acts[k].append(act[k].copy())
            if out["reset"][k]:
                done[k] = True
            else:
                snaps[k].append(env.json(k))
        obs = out["obs"].copy()
    env.close()
    with open(os.path.join(CACHE, f"calib_{a.matches}_s{a.seed}.jsonl"), "w") as f:
        for k in range(a.matches):
            f.write(json.dumps({"snaps": snaps[k], "acts": np.array(acts[k]).tolist()}) + "\n")
    print(f"calib: {a.matches} matches, {sum(len(s) for s in snaps)} snapshots", flush=True)


def main():
    p = argparse.ArgumentParser()
    sub = p.add_subparsers(dest="cmd", required=True)
    g = sub.add_parser("gen")
    g.add_argument("--name", required=True)
    g.add_argument("--ckpt", required=True)
    g.add_argument("--matches", type=int, default=128)
    g.add_argument("--threads", type=int, default=3)
    g.add_argument("--seed", type=int, default=4242)
    c = sub.add_parser("calib")
    c.add_argument("--ckpt", default=os.path.join(ROOT, "runs", "gauntlet", "ckpt", "t20_300M.pt"))
    c.add_argument("--matches", type=int, default=8)
    c.add_argument("--seed", type=int, default=777)
    r = sub.add_parser("report")
    r.add_argument("--matches", type=int, default=128)
    r.add_argument("--seed", type=int, default=4242)
    r.add_argument("--out", default=os.path.join(HERE, "t20_vs_user_report"))
    a = p.parse_args()
    {"gen": cmd_gen, "calib": cmd_calib, "report": lambda a: cmd_report(a)}[a.cmd](a)


# ------------------------------------------------------------------------------------------------ report: data

HALF_W, HALF_H, POLE_R = 24.5 / 64, 56.0 / 64, 8.0 / 64   # rider hull half-extents, pole half-width (m)


def _pt_seg(p, a, b):
    v = b - a
    u = np.clip(((p - a) * v).sum(-1) / ((v * v).sum(-1) + 1e-9), 0, 1)
    return np.linalg.norm(p - a - u[..., None] * v, axis=-1)


def _seg_seg(p0, p1, q0, q1):
    """Distance between segment p0-p1 and the segments q0[i]-q1[i] (arrays [n, 2]); 0 if they cross."""
    d = np.minimum.reduce([_pt_seg(q0, p0[None], p1[None]), _pt_seg(q1, p0[None], p1[None]),
                           _pt_seg(np.broadcast_to(p0, q0.shape), q0, q1), _pt_seg(np.broadcast_to(p1, q0.shape), q0, q1)])
    cr = lambda a, b, c: (b[..., 0] - a[..., 0]) * (c[..., 1] - a[..., 1]) - (b[..., 1] - a[..., 1]) * (c[..., 0] - a[..., 0])  # noqa: E731
    x = (cr(p0[None], p1[None], q0) * cr(p0[None], p1[None], q1) < 0) & (cr(q0, q1, p0[None]) * cr(q0, q1, p1[None]) < 0)
    return np.where(x, 0.0, d)


def true_pole_contact(st):
    """[2] bool: rider k's real pole polyline (half-width 0.125 m) touches the other rider's hull (box 0.77 x 1.75 m,
    as a capsule along the body axis) within 0.1 m; both alive. The geometry the straight-line concepts approximate."""
    pl = sorted(st["players"], key=lambda q: q["id"])
    out = np.zeros(2, bool)
    if len(pl) != 2 or any(p.get("dead") or p.get("ghost") for p in pl):
        return out
    for k in (0, 1):
        P = np.array([[q["x"], q["y"]] for q in pl[k]["pole"]])
        o = pl[1 - k]
        c, a = np.array([o["x"], o["y"]]), o.get("a", 0.0)
        ax = np.array([-np.sin(a), np.cos(a)]) * (HALF_H - HALF_W)
        out[k] = _seg_seg(c - ax, c + ax, P[:-1], P[1:]).min() <= HALF_W + POLE_R + 0.1
    return out


def contact_agreement(F, states, col=0):
    """Row and episode agreement of concepts.mutual_pole_contact / interlocks with the real geometry (one match)."""
    con = C.mutual_pole_contact(F)[:, col]
    tm = np.array([true_pole_contact(s) for s in states]).all(1) & F.live[:, col]
    il_c = C._interlock(F)[0][:, col]
    il_t = C._interlock(F, np.repeat(tm[:, None], F.x.shape[1], 1))[0][:, col]
    r = {"tp": int((con & tm).sum()), "fp": int((con & ~tm).sum()), "fn": int((~con & tm).sum()),
         "true_il": 0, "true_il_found": 0, "concept_il": 0, "concept_il_true": 0}
    same = F.same[:, col:col + 1]
    for _, s0, e0 in C.runs_of(il_t[:, None], same):
        r["true_il"] += 1; r["true_il_found"] += bool(il_c[s0:e0 + 1].any())
    for _, s0, e0 in C.runs_of(il_c[:, None], same):
        r["concept_il"] += 1; r["concept_il_true"] += bool(il_t[s0:e0 + 1].any())
    return r


def load_human():
    """Every usable real-game match (human_concepts' matching): analyse_match dicts + labels + contact agreement."""
    import human_concepts as HC
    results = HC.load_results([os.path.join(ROOT, "runs", "gauntlet", "human.jsonl"),
                               os.path.join(ROOT, "runs", "human", "series.jsonl")])
    out = []
    for path in sorted(glob.glob(os.path.join(ROOT, "runs", "human", "*.jsonl"))):
        if path.endswith("series.jsonl"):
            continue
        for mid, (frames, states, kinds) in C.playlog_read(path).items():
            m = HC.analyse_match(path, mid, frames, states, kinds)
            h = np.nonzero(m["humans"])[0]
            if m["sanity"]["aborted"] or m["sanity"]["play_seconds"] < 60 or len(h) != 1:
                continue
            hr = int(h[0])
            res = HC.match_result(results, HC.log_start(path), (m["score"][hr], m["score"][1 - hr]))
            if res is None:
                continue
            m.update(human=hr, bot=1 - hr, opp=res["opp"], group=HC.group_of(res["opp"]), result=(res["mine"], res["theirs"]),
                     log=os.path.basename(path))
            m["contact_check"] = contact_agreement(m["F"], [states[i] for i in m["idx"]], hr)
            m["hv"], m["bv"] = HC.rider_view(m, hr), HC.rider_view(m, 1 - hr)
            m["hr"], m["br"] = HC.rates(m["hv"]), HC.rates(m["bv"])
            print(f"  {m['log']}: {m['opp']} {m['result'][0]}-{m['result'][1]}", flush=True)
            out.append(m)
    return out


def load_sim(name, matches, seed):
    d = np.load(cache_path(name, matches, seed))
    O, LIVE, TC, ACT = d["O"], d["LIVE"], d["TC"], d["ACT"]
    live = np.repeat(LIVE[:, :, None], 2, 2)
    F = C.Frames(O, live, touch=TC if TC.size else None, act=ACT)
    return F, {"steps": int(d["steps"]), "ckpt": str(d["ckpt"]), "matches": matches}


# ------------------------------------------------------------------------------------------------ report: groups

class Group:
    """Pooled rows of one rider role (e.g. the user vs t20, t20 in the real game, t20 in sim self-play)."""

    def __init__(self, label):
        self.label = label
        self.minutes = 0.0; self.alive_minutes = 0.0; self.matches = 0
        self.il, self.og, self.sa, self.sb = [], [], [], []
        self.rot_all, self.rot_A, self.rot_il = [], [], []
        self.base = []
        self.extra = {}

    def add(self, F, view_idx, il_views, mins, ctx=None):
        """view_idx: Frames batch indices (tuples) of this role; il_views: the views whose interlocks count (each once)."""
        import concept_outcomes as CO
        ctx = ctx or CO.Ctx(F)
        last = view_idx
        self.minutes += mins
        self.alive_minutes += float(sum(F.alive[(slice(None),) + v].sum() for v in view_idx)) * C.STEP_S / 60
        self.il += CO.interlock_rows(F, ctx, views=il_views) if il_views is not None else []
        keep = lambda rows: [r for r in rows if tuple(r["col"])[-1] in [v[-1] for v in last]]  # noqa: E731
        self.og += keep(CO.open_goal_rows(F, ctx))
        self.sa += keep(CO.situation_rows(F, 2, ctx))
        self.sb += keep(CO.situation_rows(F, 3, ctx))
        sel = np.zeros(F.x.shape, bool)
        for v in view_idx:
            sel[(slice(None),) + v] = True
        self.rot_all.append(CO.rotation_arrays(F, sel))
        self.rot_A.append(CO.rotation_arrays(F, sel & C.sit_A(F)))
        self.rot_il.append(CO.rotation_arrays(F, sel & C.interlocked(F)))
        b = CO.baseline(F, ctx)
        self.base.append((b["n"], b))
        return ctx

    def summary(self, il_minutes=None):
        import concept_outcomes as CO
        n = sum(k for k, _ in self.base)
        base = {key: sum(k * b[key] for k, b in self.base) / max(1, n) for key in ("p_gf", "p_ga", "gd")}
        return {"label": self.label, "minutes": self.minutes, "alive_minutes": self.alive_minutes,
                "baseline_10s": base, "interlock": CO.summarize_interlocks(self.il, il_minutes or self.minutes),
                "open_goal": CO.summarize_open_goal(self.og, self.minutes),
                "sit_A": CO.summarize_situation(self.sa, self.minutes, self.alive_minutes),
                "sit_B": CO.summarize_situation(self.sb, self.minutes, self.alive_minutes),
                "rotation_all": CO.rotation_stats(self.rot_all), "rotation_A": CO.rotation_stats(self.rot_A),
                "rotation_interlocked": CO.rotation_stats(self.rot_il), **self.extra}


def servo_demo():
    """The pole servo takes the short way: riders standing at kick-off (pole forward-down, about -10..-20 deg), the
    stick held at a fixed aim; the pole angle after 0.13 / 0.27 / 0.53 s (native sim)."""
    from native_env import NativeHost
    rows = []
    for aim in (120, 150, 170, 190, 210, 240):
        env = NativeHost(1, 1, 4, seed=1, threads=1, match_format=(7, 300), goal_refund=False)
        o = env.reset_all().copy()
        a0 = float(C.Frames(o[None]).ang[0, 0, 0])
        act = np.zeros((1, 2, 3), np.float32)
        act[0, :, 1], act[0, :, 2] = np.cos(np.radians(aim)), np.sin(np.radians(aim))
        traj = []
        for t in range(8):
            o = env.step(act)["obs"].copy()
            if t in (1, 3, 7):
                traj.append(round(float(C.Frames(o[None]).ang[0, 0, 0]), 0))
        env.close()
        short = "backward (CCW)" if ((aim - a0 + 180) % 360 - 180) > 0 else "forward (CW)"
        rows.append({"aim": aim, "start": round(a0, 0), "angles": traj, "short_way": short,
                     "went": "backward (CCW)" if (traj[0] - a0 + 180) % 360 - 180 > 0 else "forward (CW)"})
    return rows


def welch(a, b):
    a, b = np.array(a, float), np.array(b, float)
    if len(a) < 2 or len(b) < 2:
        return None
    se = np.sqrt(a.var(ddof=1) / len(a) + b.var(ddof=1) / len(b))
    return float((a.mean() - b.mean()) / se) if se > 0 else None


# ------------------------------------------------------------------------------------------------ report

def cmd_report(a):
    import concept_outcomes as CO
    import human_concepts as HC
    t0 = time.time()
    print("real-game logs:", flush=True)
    hm = load_human()
    T20 = [m for m in hm if m["group"] == "t20@300M"]
    EARLY = [m for m in hm if m["group"] != "t20@300M"]
    G = {k: Group(lab) for k, lab in (("user_t20", "user vs t20 (real, 3)"), ("t20_real", "t20 vs user (real, 3)"),
                                       ("user_early", "user vs earlier bots (real, 13)"),
                                       ("bots_early", "earlier bots vs user (real, 13)"),
                                       ("t20_sim", "t20 self-play (sim)"), ("farm2_sim", "farm2 self-play (sim)"))}
    for m in hm:
        F, h, b = m["F"], m["human"], m["bot"]
        mins = float(m["live"][:, h].sum()) * C.STEP_S / 60
        gu, gb = (G["user_t20"], G["t20_real"]) if m in T20 else (G["user_early"], G["bots_early"])
        ctx = gu.add(F, [(h,)], [h], mins)
        gb.add(F, [(b,)], None, mins, ctx)
        gu.matches += 1; gb.matches += 1
    sims = {}
    for key, name in (("t20_sim", "t20_300M"), ("farm2_sim", "farm2")):
        print(f"sim {name} ...", flush=True)
        F, info = load_sim(name, a.matches, a.seed)
        sims[key] = info
        mins = float(F.live[:, :, 0].sum()) * C.STEP_S / 60
        g = G[key]
        ctx = g.add(F, [(slice(None), 0), (slice(None), 1)], [0], 2 * mins)
        g.matches = a.matches
        g.extra["match_minutes"] = mins
        # sim vs real sanity: a few concept rates of the policy itself
        S = {n: C.CONCEPTS[n][1](F) for n in ("airborne", "good_side", "sit_A", "sit_B", "pole_on_opp", "interlocked",
                                                "open_goal", "mutual_pole_contact")}
        E = {n: C.CONCEPTS[n][1](F) > 0 for n in ("takeoff", "back_pass", "fwd_pass", "kill", "goal_for")}
        al = F.alive.sum()
        g.extra["rates"] = {"states": {n: float(v.sum() / al) for n, v in S.items()},
                            "events_per_min": {n: float(v.sum() / (2 * mins)) for n, v in E.items()}}
        # run lengths of gap-bridged mutual contact (calibration of INTERLOCK_MIN), view 0
        g.extra["contact_runs"] = _contact_runs([C.mutual_pole_contact(F)[:, :, 0]], [F.same[:, :, 0]])
        # open goal: windows with the opponent 3-5.5 m on his wrong side (below OPEN_DX) for the threshold check
        # open goal threshold: out-of-position windows defined with other dx thresholds
        old = C.OPEN_DX
        g.extra["open_goal_dx_check"] = {}
        for dxv in (3.0, 5.5, 8.0):
            C.OPEN_DX = dxv
            rows3 = [r for r in CO.open_goal_rows(F, ctx) if r["cause"] == "out_of_position"]
            g.extra["open_goal_dx_check"][f"{dxv} m"] = CO.summarize_open_goal(rows3, 2 * mins)["all"]
        C.OPEN_DX = old
        del F, ctx
    real_runs = _contact_runs([C.mutual_pole_contact(m["F"])[:, m["human"]:m["human"] + 1] for m in hm],
                              [m["F"].same[:, m["human"]:m["human"] + 1] for m in hm])
    S = {k: g.summary(il_minutes=(g.extra.get("match_minutes") if k.endswith("_sim") else None)) for k, g in G.items()}
    print(f"summaries {time.time() - t0:.0f} s", flush=True)

    # ---- task 1: the user vs t20 / earlier, t20 vs farm2 as an opponent (per-match concept rates, human_concepts)
    metrics = HC.metric_list()

    def vals(ms, who, kind, n):
        return [HC.get_metric(m["hr" if who == "h" else "br"], kind, n) for m in ms]
    FAM = [m for m in EARLY if m["group"] == "farm2-family"]
    comp = []
    for kind, n, _ in metrics:
        if n in HC.OUTCOME_EVENTS:
            continue
        rh_t, rh_e = np.mean(vals(T20, "h", kind, n)), np.mean(vals(EARLY, "h", kind, n))
        rb_t, rb_f = np.mean(vals(T20, "b", kind, n)), np.mean(vals(FAM, "b", kind, n))
        comp.append({"kind": kind, "name": n, "user_vs_t20": rh_t, "user_vs_earlier": rh_e,
                     "t_user": welch(vals(T20, "h", kind, n), vals(EARLY, "h", kind, n)),
                     "t20_bot": rb_t, "farm2_bot": rb_f, "t_bot": welch(vals(T20, "b", kind, n), vals(FAM, "b", kind, n)),
                     "per_match_user_t20": vals(T20, "h", kind, n), "per_match_t20": vals(T20, "b", kind, n)})

    calib = {"sim": _calib_sim(), "real": {}}
    cc = [m["contact_check"] for m in hm]
    calib["real"] = {k: sum(c[k] for c in cc) for k in cc[0]}
    calib["real_t20"] = {k: sum(m["contact_check"][k] for m in T20) for k in cc[0]}
    calib["contact_runs_real"] = real_runs
    demo = servo_demo()

    # the real t20 matches' interlocks one by one
    il_list = []
    for m in T20:
        for r in CO.interlock_rows(m["F"], None, views=[m["human"]]):
            il_list.append({"log": m["log"], "t_s": r["start"] / C.HZ, "dur_s": r["dur_s"], "stacked": r["stacked"],
                            "top": {1: "user", -1: "t20", 0: "-"}[r["top"]],
                            "attacker": {1: "user", -1: "t20", 0: "-"}[r["attacker"]],
                            "user_let_go": r["my_let_go"], "t20_let_go": r["his_let_go"],
                            "first_kick_3s": {"me": "user", "opp": "t20"}.get(r["first_kick"], r["first_kick"]),
                            "goal_5s": "user" if r["gf5"] else ("t20" if r["ga5"] else "-"),
                            "user_sit": C_SIT[r["sit"]] if r["sit"] >= 0 else "-"})
    lunges = {"lunges": int(sum((m["E"]["lunge_at_opp"][:, m["human"]] > 0).sum() for m in T20)),
              "overcommits": int(sum((m["E"]["overcommit"][:, m["human"]] > 0).sum() for m in T20)),
              "lunges_earlier": int(sum((m["E"]["lunge_at_opp"][:, m["human"]] > 0).sum() for m in EARLY)),
              "overcommits_earlier": int(sum((m["E"]["overcommit"][:, m["human"]] > 0).sum() for m in EARLY))}
    report = {"generated": time.strftime("%Y-%m-%d %H:%M"), "sim": sims, "groups": S, "comparison": comp, "user_lunges": lunges,
              "calibration": calib, "servo_demo": demo, "t20_real_interlocks": il_list,
              "matches": [{"log": m["log"], "opp": m["opp"], "group": m["group"], "result": m["result"],
                           "contact_check": m["contact_check"]} for m in hm],
              "params": {"INTERLOCK_MIN_s": C.INTERLOCK_MIN / C.HZ, "INTERLOCK_GAP_s": C.INTERLOCK_GAP / C.HZ,
                         "INTERLOCK_LONG_s": C.INTERLOCK_LONG / C.HZ, "LETGO_W_s": C.LETGO_W / C.HZ,
                         "OPEN_DX_m": C.OPEN_DX, "OPEN_GRACE_s": C.OPEN_GRACE / C.HZ, "SERVO_DZ": C.SERVO_DZ,
                         "SERVO_MIN": C.SERVO_MIN}}
    text = render(report, T20, EARLY)
    open(a.out + ".txt", "w").write(text)
    json.dump(report, open(a.out + ".json", "w"), indent=1, default=lambda o: o.item() if hasattr(o, "item") else str(o))
    print(text)
    print(f"-> {a.out}.txt / .json ({time.time() - t0:.0f} s)")


C_SIT = ["right_far", "right_close", "A", "B", "both_wrong", "opp_out"]


def _contact_runs(cs, sames):
    """Lengths of gap-bridged mutual-contact runs (gaps <= INTERLOCK_GAP, any length): share of runs at least
    0.5 / 1 / 2 / 3 / 5 / 10 s long. cs, sames: lists of bool[T, n] (contact, row continues the previous)."""
    lens = []
    for c, same in zip(cs, sames):
        for j in range(c.shape[1]):
            s = last = -1
            for t in np.nonzero(c[:, j] | ~same[:, j])[0]:
                if not same[t, j] or (s >= 0 and t - last > C.INTERLOCK_GAP + 1):
                    if s >= 0:
                        lens.append(last - s + 1)
                    s = last = -1
                if c[t, j]:
                    s = t if s < 0 else s
                    last = t
            if s >= 0:
                lens.append(last - s + 1)
    L = np.array(lens) / C.HZ
    return {"n": len(L), **{f">={x}s": float((L >= x).mean()) if len(L) else None for x in (0.5, 1, 2, 3, 5, 10)}}


def _calib_sim():
    p = os.path.join(CACHE, "calib_8_s777.jsonl")
    if not os.path.exists(p):
        return None
    tot = None
    for line in open(p):
        snaps = [json.loads(x) if isinstance(x, str) else x for x in json.loads(line)["snaps"]]
        O, live, same = C.obs_from_states(snaps)
        r = contact_agreement(C.Frames(O, live, same), snaps, 0)
        tot = r if tot is None else {k: tot[k] + r[k] for k in r}
    return tot



# ------------------------------------------------------------------------------------------------ report: text

SUGGESTIONS = [
    "Start states from situation A (the named [t20, user, ball]): a share of resets (like --pit-frac / --goal-spawn) that "
    "put one rider past the ball with the opponent between it and the ball, incl. the stacked variant (the opponent standing "
    "on its upright pole) — or restore real states harvested at A entries / interlock starts (NativeHost.snapshot/restore, "
    "the sim and the 16 real logs). Measure: P(conceded 10 s) and 'back on the good side in 3 s' from A (section 4).",
    "Action parameterization for the pole: aim RELATIVE to the current pole angle (a signed rotation offset in (-180, 180) "
    "plus a strength) instead of an absolute stick angle. The servo already works on the error to the aimed angle, so "
    "'keep rotating backward' becomes one constant output instead of a stick sweep that has to stay < 180 deg ahead of a "
    "moving pole; a forward-biased Gaussian on (aim x, aim y) (t20 drives forward ~75% of the time) rarely produces it. "
    "Could be an --action-head option, A/B-tested like gated/resmlp.",
    "A real 'let go' action: bots almost never centre the stick (servo off 1-2% of play; the user 30%), because a Gaussian "
    "over (aim x, aim y) seldom lands in the 0.28 dead box. A discrete release bit (or the gated head's gate on the aim, if "
    "n1_t20_gated has one) makes letting go learnable; check whether n1_t20_gated ever releases in interlocks.",
    "Concept prediction targets (auxiliary heads, no reward change): from each state predict 'conceded within 10 s', "
    "'back on the good side within 3 s' (from A), the outcome of the current interlock (who kicks first after it), and "
    "open-goal conversion; plus the concept states themselves (sit_A, interlocked, open_goal) 1 s ahead.",
    "Interlocks: t20 self-play treats them as safe for the attacker (scores 30% / concedes 2% within 5 s after) and holds "
    "some to the final whistle (4.5 min); humans break them by letting go. A league opponent that releases its pole in "
    "interlocks (a scripted override of the frozen opponent's aim while interlocked) would price that in. Note the user "
    "rejected random do-nothing opponents; this is a targeted expert move, so it needs the user's call.",
    "Open-goal finishing: kill-window start states (opponent dead with 1-3 s of respawn left, ball at various x) and "
    "opponent-out-of-position states, scored by conversion within the window; the sim conversion (t20 43%, farm2 28%) is "
    "the baseline to beat.",
    "Behaviour cloning on the user's demos (runs/human/demos.npz) restricted to A / interlock states as an auxiliary loss: "
    "the user's backward drive in A (54% of the time vs t20 7%) and his let-gos are in those rows.",
]


def headlines(R, T20):
    G = R["groups"]
    u, b, ts, fs, ue, be = (G[k] for k in ("user_t20", "t20_real", "t20_sim", "farm2_sim", "user_early", "bots_early"))
    comp = {c["name"]: c for c in R["comparison"]}
    c = lambda n, k: comp[n][k]  # noqa: E731
    out = []
    out.append(f"Results vs t20_300M: " + ", ".join(f"{m['result'][0]}-{m['result'][1]}" for m in T20)
               + " (the user's first non-wins in the logs; 13/13 won before).")
    out.append(f"Interlocks are a t20 trait: {u['interlock']['per_min']:.2f} per match-minute vs the user (n={u['interlock']['n']}, "
               f"{pc(u['interlock']['share_of_time'])} of play) vs {ue['interlock']['per_min']:.2f} with the earlier bots "
               f"(n={ue['interlock']['n']}); t20 self-play {ts['interlock']['per_min']:.2f}/min (n={ts['interlock']['n']}, "
               f"{pc(ts['interlock']['share_of_time'])} of play, p90 {ts['interlock']['dur_p90_s']:.0f} s, longest "
               f"{ts['interlock']['dur_max_s']:.0f} s = to the final whistle), farm2 {fs['interlock']['per_min']:.2f}/min. "
               f"Detector vs real geometry: precision {R['calibration']['sim']['tp'] / max(1, R['calibration']['sim']['tp'] + R['calibration']['sim']['fp']):.2f}, "
               f"recall {R['calibration']['sim']['tp'] / max(1, R['calibration']['sim']['tp'] + R['calibration']['sim']['fn']):.2f} (sim rows).")
    a_s = ts["interlock"]["by_role"]["attacker"]
    out.append(f"In t20 self-play the attacker at the start of an interlock scores within 5 s after it {pc(a_s['gf5'])} and concedes "
               f"{pc(a_s['ga5'])} (n={a_s['n']}): for bots it is a safe hold, not a coin flip. Bots never let go: "
               f"{ts['interlock']['let_go']['me'] + ts['interlock']['let_go']['opp']} let-go escapes in {ts['interlock']['n']} t20 "
               f"interlocks, servo off {pc(ts['rotation_all'].get('servo_off_share'))} of play (the user: "
               f"{pc(u['rotation_all'].get('servo_off_share'))}).")
    lg = u["interlock"]["let_go_vs_not"]["released_top"]
    tops = [r for r in R["t20_real_interlocks"] if r["user_let_go"] and r["top"] == "user"]
    out.append(f"The user let go to end {u['interlock']['let_go']['me']} of the {u['interlock']['n']} interlocks vs t20; on top "
               f"{lg['n']} times (" + ", ".join(f"{r['log'][:15]} at {r['t_s']:.0f} s after {r['dur_s']:.1f} s, attacker {r['attacker']}, "
                                                 f"first kick {r['first_kick_3s']}, goal {r['goal_5s']}" for r in tops)
               + f"). n={lg['n']}: consistent with the user's demonstrations, not a measurement.")
    og_b, og_u, og_s, og_e = b["open_goal"], u["open_goal"], ts["open_goal"], be["open_goal"]
    out.append(f"Open goals (opponent dead or > 5.5 m on his wrong side): t20 converts {pc(og_b['all']['converted'])} vs the user "
               f"(n={og_b['all']['n']}), the user {pc(og_u['all']['converted'])} vs t20 (n={og_u['all']['n']}), t20 self-play "
               f"{pc(og_s['all']['converted'])} (n={og_s['all']['n']}), the earlier bots vs the user only {pc(og_e['all']['converted'])} "
               f"(n={og_e['all']['n']}). With the user dead: t20 {og_b['dead']['n']} windows, {pc(og_b['dead']['converted'])}; "
               f"from the middle (ball 10-20 m out): t20 {pc(og_b['by_ball_to_goal']['10-20 m (middle)']['converted'])} "
               f"(n={og_b['by_ball_to_goal']['10-20 m (middle)']['n']}) vs earlier bots "
               f"{pc(og_e['by_ball_to_goal']['10-20 m (middle)']['converted'])} (n={og_e['by_ball_to_goal']['10-20 m (middle)']['n']}). "
               "Real-game samples are small.")
    sa_b, sa_u, sa_s, sa_f = b["sit_A"]["all"], u["sit_A"]["all"], ts["sit_A"]["all"], fs["sit_A"]["all"]
    out.append(f"The named situation [t20, user, ball] is situation A for t20. From it t20 conceded within 10 s "
               f"{pc(sa_b['ga10'])} of {sa_b['n']} entries and was back on the good side within 3 s {pc(sa_b['escaped_3s'])}; the "
               f"user in the mirror: {pc(sa_u['ga10'])} / {pc(sa_u['escaped_3s'])} (n={sa_u['n']}); t20 self-play "
               f"{pc(sa_s['ga10'])} / {pc(sa_s['escaped_3s'])} (n={sa_s['n']}), farm2 {pc(sa_f['ga10'])} (baseline P(conceded 10 s) "
               f"~{pc(ts['baseline_10s']['p_ga'])}). t20 is interlocked {pc(sa_b['interlocked_share'])} of its real A time.")
    ra_b, ra_u, ra_s = b["rotation_A"], u["rotation_A"], ts["rotation_A"]
    out.append(f"Pole rotation in A (the user's 'he doesn't want to move his pole clockwise'): t20's stick drives the pole backward "
               f"{pc(ra_b.get('aim_drives_back_share'))} of the time in the real games and {pc(ra_s.get('aim_drives_back_share'))} in "
               f"self-play, aims into the back half {pc(ra_b.get('aim_back_half_share'))} / {pc(ra_s.get('aim_back_half_share'))}; the "
               f"user in A: {pc(ra_u.get('aim_drives_back_share'))} backward, {pc(ra_u.get('aim_back_half_share'))} back half; the pole "
               f"actually turns backward {pc(ra_b.get('w_back_share'))} (t20) vs {pc(ra_u.get('w_back_share'))} (user). Elsewhere t20 "
               f"sweeps backward plenty (back passes {ts['rotation_all']['back_pass_per_min']:.1f}/min vs farm2 "
               f"{fs['rotation_all']['back_pass_per_min']:.1f}): it is the situation, not the skill. Caveat: in self-play, A stays "
               f"with the pole turning backward >= 30% of the time are not better on goals (conceded 10 s "
               f"{pc(ts['sit_A']['pole_back_ge30']['ga10'])} vs {pc(ts['sit_A']['pole_back_lt30']['ga10'])}; back on the good side "
               f"in 3 s {pc(ts['sit_A']['pole_back_ge30']['escaped_3s'])} vs {pc(ts['sit_A']['pole_back_lt30']['escaped_3s'])}): the "
               "data show t20 rarely tries it, not that trying it (the way t20 would) pays.")
    out.append(f"The user vs t20 (vs the earlier bots): the user's pole on the bot {100 * c('pole_on_opp', 'user_vs_t20'):.0f}% vs "
               f"{100 * c('pole_on_opp', 'user_vs_earlier'):.0f}% (t {c('pole_on_opp', 't_user'):+.1f}), the user at the bot's mercy "
               f"{100 * c('at_mercy', 'user_vs_t20'):.0f}% vs {100 * c('at_mercy', 'user_vs_earlier'):.0f}% ({c('at_mercy', 't_user'):+.1f}), "
               f"good side {100 * c('good_side', 'user_vs_t20'):.0f}% vs {100 * c('good_side', 'user_vs_earlier'):.0f}% "
               f"({c('good_side', 't_user'):+.1f}), lunges {c('lunge_at_opp', 'user_vs_t20'):.2f} vs "
               f"{c('lunge_at_opp', 'user_vs_earlier'):.2f}/min ({c('lunge_at_opp', 't_user'):+.1f}; {R['user_lunges']['overcommits']} of "
               f"{R['user_lunges']['lunges']} overcommitted vs {R['user_lunges']['overcommits_earlier']} of "
               f"{R['user_lunges']['lunges_earlier']} before), ball kicks "
               f"{c('ball_kick', 'user_vs_t20'):.1f} vs {c('ball_kick', 'user_vs_earlier'):.1f}/min ({c('ball_kick', 't_user'):+.1f}); "
               f"kills unchanged ({c('kill', 'user_vs_t20'):.2f} vs {c('kill', 'user_vs_earlier'):.2f}/min).")
    out.append(f"t20 vs farm2 as the user's opponent: airborne {100 * c('airborne', 't20_bot'):.0f}% vs {100 * c('airborne', 'farm2_bot'):.0f}% "
               f"(t {c('airborne', 't_bot'):+.1f}), still {100 * c('still', 't20_bot'):.1f}% vs {100 * c('still', 'farm2_bot'):.1f}% "
               f"({c('still', 't_bot'):+.1f}), tip behind the ball {100 * c('tip_behind_ball', 't20_bot'):.0f}% vs "
               f"{100 * c('tip_behind_ball', 'farm2_bot'):.0f}% ({c('tip_behind_ball', 't_bot'):+.1f}), at the user's mercy "
               f"{100 * c('at_mercy', 't20_bot'):.1f}% vs {100 * c('at_mercy', 'farm2_bot'):.1f}% ({c('at_mercy', 't_bot'):+.1f}), back "
               f"passes {c('back_pass', 't20_bot'):.1f} vs {c('back_pass', 'farm2_bot'):.1f}/min, second kicks "
               f"{c('second_kick', 't20_bot'):.1f} vs {c('second_kick', 'farm2_bot'):.1f}/min: it stays down, keeps the tip hooked and "
               f"gives fewer free hits. Real-game t20 matches its sim self-play (airborne {100 * G['t20_sim']['rates']['states']['airborne']:.0f}%, "
               f"back passes {G['t20_sim']['rates']['events_per_min']['back_pass']:.1f}/min in sim).")
    return out

def f_(v, fmt="{:.2f}"):
    return "n/a" if v is None else fmt.format(v)


def pc(v):
    return "n/a" if v is None else f"{100 * v:.0f}%"


ORDER = ["user_t20", "t20_real", "user_early", "bots_early", "t20_sim", "farm2_sim"]
SHORT = {"user_t20": "user v t20", "t20_real": "t20 real", "user_early": "user v early", "bots_early": "early bots",
         "t20_sim": "t20 sim", "farm2_sim": "farm2 sim"}


def render(R, T20, EARLY):
    G = R["groups"]
    L = []
    w = L.append
    W = 17

    def table(title, keys, rows):
        if title:
            w(title)
        w(f"  {'':34s}" + "".join(f"{SHORT[k]:>{W}s}" for k in keys))
        for lab, fn in rows:
            cells = []
            for k in keys:
                try:
                    cells.append(fn(G[k]))
                except (KeyError, TypeError):
                    cells.append("n/a")
            w(f"  {lab:34s}" + "".join(f"{c:>{W}s}" for c in cells))

    sims = R["sim"]
    w(f"t20 (300M) vs the user — the 2026-10-03 series + new concepts ({R['generated']})")
    w(f"Real games: {len(T20)} matches vs t20_300M (" + ", ".join(f"{m['result'][0]}-{m['result'][1]}" for m in T20)
      + f"; the 5-5 cut at the end of regulation, the sudden-death goal ignored) and {len(EARLY)} earlier matches (farm2 "
      f"family 9, ~20M search trials 4: all won by the user). The user is always the left rider (real controller).")
    w(f"Sim: {sims['t20_sim']['matches']} matches t20_300M self-play ({sims['t20_sim']['steps'] / 1e6:.0f}M steps) and "
      f"{sims['farm2_sim']['matches']} farm2 self-play ({sims['farm2_sim']['steps'] / 1e6:.0f}M), native sim, 7 goals / 5 min, "
      "no sudden death, sampled actions (as in the real game). Both sim riders are pooled (each interlock counted once).")
    w("Sample sizes: 3 real matches vs t20 = ~13 min of play. Anything below ~20 episodes is a hint, not a finding "
      "(n is printed everywhere). Goal outcomes are correlational. Concepts: concepts.py; outcome code: "
      "analysis/concept_outcomes.py.")

    # ---------------- headlines (numbers from below)
    w("\n== Headlines")
    for line in headlines(R, T20):
        w("  - " + line)

    # ---------------- task 1
    w("\n== 1. The user vs t20 and vs the earlier bots; t20 vs farm2 as an opponent (per-match concept rates, Welch t)")
    w("  t_user: user vs t20 (3 matches) against user vs earlier (13); t_bot: t20 in the real game (3) against the farm2-family "
      "bots (9). With 3 matches |t| < ~3 is noise.")
    comp = {c["name"]: c for c in R["comparison"]}

    def crow(n, kind):
        c = comp[n]
        fm = (lambda v: f"{100 * v:5.1f}%") if kind == "state" else (lambda v: f"{v:6.2f}")
        pm = "/".join((f"{100 * v:.0f}" if kind == "state" else f"{v:.1f}") for v in c["per_match_user_t20"])
        pb = "/".join((f"{100 * v:.0f}" if kind == "state" else f"{v:.1f}") for v in c["per_match_t20"])
        return (f"  {n:22s} {fm(c['user_vs_t20']):>8s} {fm(c['user_vs_earlier']):>8s} {f_(c['t_user'], '{:+.1f}'):>6s}   "
                f"{fm(c['t20_bot']):>8s} {fm(c['farm2_bot']):>8s} {f_(c['t_bot'], '{:+.1f}'):>6s}   {pm:>14s} {pb:>14s}")
    hdr = (f"  {'concept':22s} {'user v20':>8s} {'user vE':>8s} {'t':>6s}   {'t20 bot':>8s} {'farm2 b':>8s} {'t':>6s}   "
           f"{'user per match':>14s} {'t20 per match':>14s}")
    groups = [("pressure / contact (the user: 'pressure is pointless vs it')",
               [("pole_on_opp", "state"), ("opp_pole_on_me", "state"), ("pole_over_opp_head", "state"),
                ("rhythmic_pressing", "state"), ("at_mercy", "state"), ("mutual_pole_contact", "state"),
                ("interlocked", "state"), ("lunge_at_opp", "event"), ("kill", "event"), ("death", "event"),
                ("catapult", "event"), ("got_flung", "event")]),
              ("position", [("good_side", "state"), ("sit_A", "state"), ("sit_B", "state"), ("tip_behind_ball", "state"),
                            ("tip_past_ball", "state"), ("opp_out", "state"), ("open_goal", "state")]),
              ("technique / tempo", [("still", "state"), ("airborne", "state"), ("takeoff", "event"), ("fast_takeoff", "event"),
                                     ("back_pass", "event"), ("fwd_pass", "event"), ("aim_back_pass", "event"),
                                     ("backswing_in_flight", "state"), ("pole_back_down", "state"), ("pole_fwd_up", "state"),
                                     ("ball_kick", "event"), ("second_kick", "event"), ("impale_plant", "event")])]
    for title, items in groups:
        w(f"  -- {title}")
        w(hdr)
        for n, kind in items:
            if n in comp:
                w(crow(n, kind))
    top_u = sorted([c for c in R["comparison"] if c["t_user"] is not None], key=lambda c: -abs(c["t_user"]))[:10]
    top_b = sorted([c for c in R["comparison"] if c["t_bot"] is not None], key=lambda c: -abs(c["t_bot"]))[:10]
    w("  -- largest |t|, the user vs t20 against vs the earlier bots: "
      + ", ".join(f"{c['name']} {c['t_user']:+.1f}" for c in top_u))
    w("  -- largest |t|, t20 against farm2 as the user's opponent: " + ", ".join(f"{c['name']} {c['t_bot']:+.1f}" for c in top_b))

    # ---------------- interlocks
    w("\n== 2. pole_interlock: both riders' poles on each other for >= 1 s (gaps <= 1/3 s bridged)")
    w("  Game fact: the two poles never collide (collision filter, pole_ctor.cpp); 'interlocked poles' = each pole resting on the "
      "other RIDER. Stacked = one rider standing on the other's pole for >= half of it (in the real games: the user balanced on "
      "t20's upright pole, his own pole pointing down onto t20). Real games: from the user's view; sim: each interlock once.")
    IL = ["user_t20", "user_early", "t20_sim", "farm2_sim"]
    table("", IL, [
        ("interlocks per match-minute (n)", lambda g: f"{g['interlock']['per_min']:.2f} ({g['interlock']['n']})"),
        ("  >= 3 s per match-minute (n)", lambda g: f"{g['interlock']['long_per_min']:.2f} ({g['interlock']['long_n']})"),
        ("  share of play time interlocked", lambda g: pc(g["interlock"]["share_of_time"])),
        ("  duration median / p90 (s)", lambda g: f"{f_(g['interlock']['dur_median_s'], '{:.1f}')}/{f_(g['interlock']['dur_p90_s'], '{:.1f}')}"),
        ("  longest (s)", lambda g: f_(g["interlock"]["dur_max_s"], "{:.0f}")),
        ("  stacked", lambda g: pc(g["interlock"]["stacked_share"])),
        ("  attacker at start me/opp/none", lambda g: "/".join(str(v) for v in g["interlock"]["attacker_at_start"].values())),
        ("  on top me/opp/none", lambda g: "/".join(str(v) for v in g["interlock"]["top"].values())),
        ("  attacker still on good side 2 s after", lambda g: pc(g["interlock"]["attacker_still_good_2s"])),
        ("  let-go escapes me/opp", lambda g: f"{g['interlock']['let_go']['me']}/{g['interlock']['let_go']['opp']}"),
    ])
    w("  ('me' = the user in the real columns, rider A in sim. Outcomes after the last contact, per side of each interlock:)")
    cellf = lambda b: f"{b['n']}/{pc(b['kick_first'])}/{pc(b['gf5'])}/{pc(b['ga5'])}"  # noqa: E731
    W = 18
    rows = []
    for role_key, roles in (("by_role", ("attacker", "defender")), ("by_top", ("top", "bottom"))):
        for ro in roles:
            rows.append((ro, lambda g, ro=ro, rk=role_key: cellf(g["interlock"][rk][ro])))
    table("  cells: n / kicks the ball first within 3 s / scores within 5 s / concedes within 5 s", IL, rows)
    w("  -- letting go (let_go_escape: servo off at the end, released <= 0.5 s before the last contact) vs nobody letting go")
    rows = [(lab, lambda g, key=key: cellf(g["interlock"]["let_go_vs_not"][key]))
            for key, lab in (("released", "released (any role)"), ("released_attacker", "released, attacker"),
                             ("released_top", "released, on top"), ("other_released", "the OTHER released"),
                             ("nobody_released", "nobody released (any role)"),
                             ("nobody_released_attacker", "nobody released: attacker"),
                             ("nobody_released_top", "nobody released: on top"))]
    table("  cells as above, for the side in question", IL, rows)
    W = 17
    by = {k: G[k]["interlock"]["by_who"] for k in IL}
    w(f"  -- who lets go, real games vs t20: user {by['user_t20']['me']['let_go']}, t20 {by['user_t20']['opp']['let_go']}; "
      f"vs earlier bots: user {by['user_early']['me']['let_go']}, bots {by['user_early']['opp']['let_go']}")
    w("  -- the real t20 matches, every interlock (t = seconds into the match; first kick within 3 s of the last contact; goal within 5 s):")
    w(f"  {'log':26s} {'t':>6s} {'dur':>5s} {'stacked':>7s} {'top':>5s} {'attacker':>8s} {'let go u/t20':>12s} {'1st kick':>8s} {'goal':>5s} {'user sit':>11s}")
    for r in R["t20_real_interlocks"]:
        w(f"  {r['log']:26s} {r['t_s']:6.1f} {r['dur_s']:5.1f} {str(r['stacked']):>7s} {r['top']:>5s} {r['attacker']:>8s} "
          f"{('Y' if r['user_let_go'] else '-') + '/' + ('Y' if r['t20_let_go'] else '-'):>12s} {str(r['first_kick_3s']):>8s} "
          f"{r['goal_5s']:>5s} {r['user_sit']:>11s}")
    w("  -- servo off ('pole let go': right stick within the dead zone) share of time")
    table("", ORDER, [("overall", lambda g: pc(g["rotation_all"].get("servo_off_share"))),
                      ("while interlocked", lambda g: pc(g["rotation_interlocked"].get("servo_off_share"))),
                      ("  (interlocked minutes of the rider)", lambda g: f_(g["rotation_interlocked"].get("minutes"), "{:.1f}"))])

    # ---------------- open goal
    w("\n== 3. open_goal / open_goal_conversion: the opponent dead (respawning) or > 5.5 m on his wrong side of the ball, me on the "
      "good side; windows >= 0.5 s (gaps <= 1 s merged); converted = my goal before the window closes + 1 s")
    og = lambda g, sub="all": g["open_goal"][sub] if sub in ("all", "dead", "out_of_position") else g["open_goal"]["by_ball_to_goal"][sub]  # noqa: E731
    rows = [("windows per rider-minute", lambda g: f"{g['open_goal']['per_min']:.2f}")]
    for sub, lab in (("all", "all"), ("dead", "opp dead"), ("out_of_position", "opp out of position"),
                     ("<10 m", "ball < 10 m from goal"), ("10-20 m (middle)", "ball 10-20 m (middle)"),
                     (">=20 m (my half)", "ball in my half (>= 20 m)")):
        rows.append((f"{lab}: n / converted", lambda g, sub=sub: f"{og(g, sub)['n']}/{pc(og(g, sub)['converted'])}"))
    rows += [("goal within 5 s / 10 s of the start", lambda g: f"{pc(og(g)['gf5'])}/{pc(og(g)['gf10'])}"),
             ("time to goal median (converted, s)", lambda g: f_(og(g)["time_to_goal_median_s"], "{:.1f}")),
             ("window length median (s)", lambda g: f_(og(g)["dur_median_s"], "{:.1f}")),
             ("conceded within 10 s of the start", lambda g: pc(og(g)["ga10"])),
             ("baseline P(goal for within 10 s)", lambda g: pc(g["baseline_10s"]["p_gf"]))]
    table("", ORDER, rows)
    for k in ("t20_sim", "farm2_sim"):
        dx = G[k].get("open_goal_dx_check")
        if dx:
            w(f"  threshold check, {SHORT[k]}: out-of-position windows when 'out of position' = more than dx on his wrong side: "
              + ", ".join(f"dx {kk}: n={v['n']} converted {pc(v['converted'])}, goal 5 s {pc(v['gf5'])}" for kk, v in dx.items()))

    # ---------------- named situation
    w("\n== 4. The user's named situation: x order [t20, user, ball] (t20 behind the user, on its wrong side)")
    w("  t20 attacks -x, so in its attack frame the order is [ball, user, t20] = ball < opp < me = situation A (sit_A, "
      "'you-opp-ball', situations.py class 2). The user's mirror = the user in A ([ball, t20, user]).")
    SA = ORDER
    sa = lambda g, sub="all": g["sit_A"][sub]  # noqa: E731
    table("", SA, [
        ("share of alive time in A", lambda g: pc(g["sit_A"].get("share_of_time"))),
        ("entries per rider-minute (n)", lambda g: f"{g['sit_A']['entries_per_min']:.2f} ({sa(g)['n']})"),
        ("median stay (s)", lambda g: f_(sa(g)["dur_median_s"], "{:.1f}")),
        ("goal diff 10 s after entry", lambda g: f_(sa(g)["gd10"], "{:+.2f}")),
        ("P(conceded within 10 s)", lambda g: pc(sa(g)["ga10"])),
        ("  baseline P(conceded 10 s)", lambda g: pc(g["baseline_10s"]["p_ga"])),
        ("P(scored within 10 s)", lambda g: pc(sa(g)["gf10"])),
        ("back on the good side within 3 / 5 s", lambda g: f"{pc(sa(g)['escaped_3s'])}/{pc(sa(g)['escaped_5s'])}"),
        ("conceded 10 s, not out within 5 s", lambda g: pc(sa(g)["ga10_before_escape"])),
        ("jump over the ball from it", lambda g: pc(sa(g)["jump_over"])),
        ("interlocked share of A time", lambda g: pc(sa(g)["interlocked_share"])),
        ("exit to B / both-wrong / opp out / I died", lambda g: "/".join(str(sa(g)["exit"][k]) for k in ("B", "both_wrong", "opp_out", "me_out"))),
        ("close (opp within 6 m): n / ga10", lambda g: f"{sa(g, 'close')['n']}/{pc(sa(g, 'close')['ga10'])}"),
        ("far: n / ga10", lambda g: f"{sa(g, 'far')['n']}/{pc(sa(g, 'far')['ga10'])}"),
        ("pole turning back >= 30% of the stay: n/ga10/out 3 s", lambda g: (lambda b: f"{b['n']}/{pc(b['ga10'])}/{pc(b['escaped_3s'])}")(sa(g, "pole_back_ge30"))),
        ("  < 30%: n/ga10/out 3 s", lambda g: (lambda b: f"{b['n']}/{pc(b['ga10'])}/{pc(b['escaped_3s'])}")(sa(g, "pole_back_lt30"))),
        ("for reference, B: n / ga10 / escaped 5 s", lambda g: f"{g['sit_B']['all']['n']}/{pc(g['sit_B']['all']['ga10'])}/{pc(g['sit_B']['all']['escaped_5s'])}"),
    ])

    # ---------------- rotation
    w("\n== 5. Pole rotation (the user: t20 'doesn't want to move his pole clockwise' in [t20, user, ball]) and the pole servo")
    w("  Frame: every number is in the rider's attack frame (0 deg = toward the goal it attacks, 90 = up). Backward = "
      "counter-clockwise there = CLOCKWISE ON SCREEN for t20 (it attacks left), counter-clockwise on screen for the user.")
    w("  How the stick drives the pole (sim/src/game/player.cpp pole servo, input.cpp; checked in the sim below):")
    w("   - target = atan2(stick y, stick x): the stick sets an ABSOLUTE pole angle. The error to the handle's angle is wrapped to")
    w("     +-180 deg, so the pole always turns the SHORT way; motor speed and max torque both scale with the error (kp 0.06, kd 0.004,")
    w("     clamp 0.4: never reached by the error alone), i.e. the push is strongest just under 180 deg ahead and flips direction past it.")
    w("   - per-axis dead zone 0.1, then the servo works only if the stick length > 0.2: a centred stick = zero torque, the pole")
    w("     hangs free ('letting go'). A policy's (aim x, aim y) maps to the stick directly (x mirrored for the right rider).")
    w("   - so a full backward rotation needs the stick swept round backward, kept ahead of the pole by < 180 deg the whole way")
    w("     (a human: a continuous circular motion of the thumb). Aiming straight 'behind' does not rotate backward: from a")
    w("     forward-down pole, any aim more than 180 deg away going backward makes the pole go FORWARD (demo).")
    w("  Demo (native sim, kick-off stance, stick held at a fixed aim; pole angle after 0.13 / 0.27 / 0.53 s):")
    for d in R["servo_demo"]:
        w(f"    aim {d['aim']:3d} deg from {d['start']:+.0f}: {d['angles']} -> went {d['went']} (short way: {d['short_way']})")
    for sub, lab in (("rotation_all", "all play"), ("rotation_A", "in situation A (for t20 in the real games: the named situation)"),
                     ("rotation_interlocked", "while interlocked")):
        w(f"  -- {lab}")
        table("", ORDER, [
            ("minutes", lambda g, s=sub: f_(g[s].get("minutes"), "{:.1f}")),
            ("pole turning backward (w > 1 rad/s)", lambda g, s=sub: pc(g[s].get("w_back_share"))),
            ("pole turning forward (w < -1)", lambda g, s=sub: pc(g[s].get("w_fwd_share"))),
            ("back passes / fwd passes per min", lambda g, s=sub: f"{f_(g[s].get('back_pass_per_min'), '{:.1f}')}/{f_(g[s].get('fwd_pass_per_min'), '{:.1f}')}"),
            ("stick drives back / fwd / holds", lambda g, s=sub: f"{pc(g[s].get('aim_drives_back_share'))}/{pc(g[s].get('aim_drives_fwd_share'))}/{pc(g[s].get('aim_holds_share'))}"),
            ("aim offset p10/p50/p90 (deg)", lambda g, s=sub: "/".join(f"{v:.0f}" for v in g[s]["aim_offset_p10_p50_p90"]) if g[s].get("aim_offset_p10_p50_p90") else "n/a"),
            ("stick sweeps over top back/fwd /min", lambda g, s=sub: f"{f_(g[s].get('aim_back_pass_per_min'), '{:.1f}')}/{f_(g[s].get('aim_fwd_pass_per_min'), '{:.1f}')}"),
            ("stick aimed into the back half", lambda g, s=sub: pc(g[s].get("aim_back_half_share"))),
            ("servo off", lambda g, s=sub: pc(g[s].get("servo_off_share"))),
        ])

    # ---------------- calibration
    w("\n== 6. Calibration / sanity")
    cal = R["calibration"]
    for lab, c in (("sim t20 self-play, 8 matches with full snapshots", cal["sim"]), ("real games, all 16", cal["real"]),
                   ("real games vs t20", cal["real_t20"])):
        if not c:
            continue
        w(f"  mutual_pole_contact (straight body->tip lines, 1 m) vs the real pole polylines / hulls, {lab}: precision "
          f"{c['tp'] / max(1, c['tp'] + c['fp']):.2f}, recall {c['tp'] / max(1, c['tp'] + c['fn']):.2f} (rows: {c['tp'] + c['fn']} true); "
          f"interlocks from the real geometry {c['true_il']}, found by the concept {c['true_il_found']}; concept interlocks "
          f"{c['concept_il']}, real {c['concept_il_true']}")
    for lab, r in (("t20 sim", G["t20_sim"].get("contact_runs")), ("farm2 sim", G["farm2_sim"].get("contact_runs")),
                   ("real (all 16, user view)", cal.get("contact_runs_real"))):
        if r:
            w(f"  mutual-contact runs (gaps <= 1/3 s bridged), {lab}: n={r['n']}, share lasting >= 0.5/1/2/3/5/10 s: "
              + "/".join(pc(r[f'>={x}s']) for x in (0.5, 1, 2, 3, 5, 10)))
    for k in ("t20_sim", "farm2_sim"):
        rr = G[k].get("rates")
        if rr:
            w(f"  {SHORT[k]} rates: " + ", ".join(f"{n} {pc(v)}" for n, v in rr["states"].items()) + "; per min: "
              + ", ".join(f"{n} {v:.2f}" for n, v in rr["events_per_min"].items()))
    w("\n== 7. Training ideas for the weaknesses above (suggestions only, nothing implemented)")
    for line in SUGGESTIONS:
        w("  - " + line)
    p = R["params"]
    w(f"\n  thresholds: interlock >= {p['INTERLOCK_MIN_s']:.0f} s, gaps <= {p['INTERLOCK_GAP_s']:.2f} s, long >= {p['INTERLOCK_LONG_s']:.0f} s; "
      f"let-go window {p['LETGO_W_s']:.0f} s; open goal dx {p['OPEN_DX_m']} m, grace {p['OPEN_GRACE_s']:.0f} s; servo dead zone "
      f"{p['SERVO_DZ']}, off below stick length {p['SERVO_MIN']}")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    main()
