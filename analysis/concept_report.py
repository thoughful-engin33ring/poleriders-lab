#!/usr/bin/env python3
"""How flat policies use the concepts of concepts.py (docs/strategy.md), and what each is worth.

Plays matches in the native sim (7 goals / 5 min, no sudden death): self-play of one policy (both riders
pooled) or A vs B head to head (sides swapped every other match). Per policy, from its own observations:
  states: share of its live time, how often it enters the state (per minute), and the goal difference over the
          next 10 s while it holds, minus the policy's baseline (all its live moments) — correlational, like
          analysis/situations.py
  events: per minute of play (one rider's minutes), and the goal difference over the next 10 s after the step
          it completes, minus the baseline (+- standard error, events treated as independent)
Rows are each rider's observe2 view; the terminal observation of each match is kept (out["final"]) so the
deciding goal is seen. The native env's ball-touch rows (out["touch"], not observed by the policies) go with
them for the touch concepts. "outcomes" (2026-10-02 concepts): what follows a lunge at the opponent (overcommit,
goal against within 5 s, was he still), value by energy / energy lead / take-off speed quartile, what follows a
second touch / kick (ball going my way, goals, my speed and position after), the impale brake's speed drop.

    analysis/concept_report.py --self runs/farm2/ckpt.pt --h2h runs/farm3/ckpt.pt:runs/farm2/ckpt.pt \\
        --matches 128 --out analysis/concepts_report
"""
import argparse, json, os, shutil, sys, tempfile, time

import numpy as np
import torch

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
import concepts as C  # noqa: E402
import eval_h2h as e  # noqa: E402
from native_env import NativeHost  # noqa: E402

H = 10 * C.HZ  # value horizon: 10 s


def load(path):
    """Load a checkpoint from a private copy (training runs overwrite theirs)."""
    with tempfile.TemporaryDirectory() as d:
        tmp = os.path.join(d, "ckpt.pt")
        for _ in range(5):
            try:
                shutil.copyfile(path, tmp)
                net, norm, ck = e.load_policy(tmp)
                return (net, norm), ck
            except Exception:  # noqa: BLE001 — caught mid-write
                time.sleep(2)
        raise RuntimeError(f"cannot load {path}")


def play(pa, pb, games, threads, seed, with_actions=False):
    """O [T, games, 2, 62] (rows of A, B), G [T, games, 2] goal term per step, LIVE [T, games], TC [T, games, 2
    (A, B rows), 2 (me, opp), 5] ball-touch rows (None if the native library has none); with_actions: also
    ACT [T, games, 2 (A, B rows), 2 (me, opp), 3] the (clipped) actions taken at each row, in each rider's own
    frame (run, aim x, aim y) — for concepts.py's `act` (the pole servo / let-go concepts)."""
    env = NativeHost(1, games, 4, sd_frac=0.0, seed=seed, threads=threads, end_on_sd=True,
                     match_format=e.FORMAT, max_steps=e.MAX_STEPS, goal_refund=False)
    idx = np.arange(games)
    a_rider = idx % 2  # which rider A drives in match k
    cur = env.reset_all().copy()
    has_t = "touch" in env.out
    cur_t = env.out["touch"].copy() if has_t else None
    state = np.zeros(games, int)  # 0 playing, 1 terminal row pending, 2 done
    O, G, LIVE, TC, ACT = [], [], [], [], []
    while (state < 2).any():
        O.append(np.stack([cur[idx, a_rider], cur[idx, 1 - a_rider]], 1)); LIVE.append(state < 2)
        if has_t:
            ta, tb = cur_t[idx, a_rider], cur_t[idx, 1 - a_rider]
            TC.append(np.stack([np.stack([ta, tb], 1), np.stack([tb, ta], 1)], 1))
        act = np.zeros((games, 2, 3), np.float32)
        act[idx, a_rider] = np.clip(e.sample(pa, cur[idx, a_rider]), -1, 1)
        act[idx, 1 - a_rider] = np.clip(e.sample(pb, cur[idx, 1 - a_rider]), -1, 1)
        if with_actions:
            aa, ab = act[idx, a_rider], act[idx, 1 - a_rider]
            ACT.append(np.stack([np.stack([aa, ab], 1), np.stack([ab, aa], 1)], 1))
        out = env.step(act)
        t = out["terms"]
        G.append(np.stack([t[idx, a_rider, 0], t[idx, 1 - a_rider, 0]], 1) * (state == 0)[:, None])
        ended = (state == 0) & out["reset"]
        cur = out["obs"].copy()
        cur[ended] = out["final"][ended]
        if has_t:
            cur_t = out["touch"].copy()
            cur_t[ended] = out["touch_final"][ended]
        state[state == 1] = 2
        state[ended] = 1
    env.close()
    res = (np.array(O), np.array(G), np.array(LIVE), (np.array(TC) if has_t else None))
    return res + (np.array(ACT),) if with_actions else res


def _bins(x, mask, fut, base, q=(25, 50, 75)):
    """Value (own goal diff next 10 s minus baseline) of moments in mask, by quartile of x."""
    v = x[mask]
    if len(v) < 20:
        return None
    edges = np.percentile(v, q)
    idx = np.searchsorted(edges, v)
    f = fut[mask]
    return {"edges": [float(a) for a in edges],
            "value": [float(f[idx == i].mean() - base) if (idx == i).any() else None for i in range(len(q) + 1)],
            "n": [int((idx == i).sum()) for i in range(len(q) + 1)]}


def outcomes(F, E, fut, ok, inside, base, live):
    """The 2026-10-02 concepts' consequences (see the module doc)."""
    T = len(F.x)
    t_idx = np.arange(T)

    def within(ev, k):  # ev happens in (t, t + k]
        c = np.concatenate([np.zeros((1,) + ev.shape[1:]), np.cumsum(ev, 0)])
        return (c[np.minimum(t_idx + 1 + k, T)] - c[t_idx + 1]) > 0

    def later(a, k):  # a at t + k (nan past the end)
        out = np.full(a.shape, np.nan, np.float32)
        out[:T - k] = a[k:]
        return out

    ga5 = within(E["goal_against"] > 0, 5 * C.HZ)
    gf5 = within(E["goal_for"] > 0, 5 * C.HZ)
    p = lambda b, m: float(b[m].mean()) if m.any() else None  # noqa: E731
    o = {"p_goal_against_5s_baseline": p(ga5, ok), "p_goal_for_5s_baseline": p(gf5, ok)}
    lunge, oc, ocs = E["lunge_at_opp"] > 0, E["overcommit"] > 0, E["overcommit_opp_still"] > 0
    o["lunge"] = {"n": int(lunge.sum()), "overcommit_per_lunge": float(oc.sum() / max(1, lunge.sum())),
                  "p_goal_against_5s": p(ga5, lunge & inside),
                  "p_goal_against_5s_after_overcommit": p(ga5, oc & inside),
                  "p_goal_against_5s_after_overcommit_opp_still": p(ga5, ocs & inside),
                  "opp_still_share_of_overcommits": float(ocs.sum() / max(1, oc.sum())),
                  "n_from_pit": int((E["lunge_from_pit"] > 0).sum()),
                  "opp_speed_at_overcommit": float(np.median(F.ospeed[oc])) if oc.any() else None}
    both = ok & F.opp_alive
    o["energy_bins"] = _bins(F.energy, ok, fut, base)
    o["energy_lead_bins"] = _bins(F.energy - F.oenergy, both, fut, base)
    o["energy_median"] = float(np.median(F.energy[ok])) if ok.any() else None
    tk = E["takeoff"] > 0
    sp = C.takeoff_speed(F)
    o["takeoff_speed_bins"] = _bins(np.nan_to_num(sp), tk & inside, fut, base)
    if F.touch is not None:
        bvx_later = later(F.O[..., 54], C.HZ // 2)          # ball vx in my attack frame 0.5 s later
        x_later, bx_later = later(F.x, C.HZ), later(F.bx, C.HZ)
        smax = np.full(F.x.shape, np.nan, np.float32)
        for k in range(T):
            smax[k] = np.nanmax(np.where(live[k:k + C.HZ // 2 + 1], F.speed[k:k + C.HZ // 2 + 1], np.nan), 0)
        for name in ("second_touch", "second_kick"):
            m = (E[name] > 0) & inside
            o[name] = {"n": int(m.sum()), "p_ball_my_way_0.5s": p(bvx_later > 0, m), "p_goal_for_5s": p(gf5, m),
                       "p_goal_against_5s": p(ga5, m), "speed_at": float(np.mean(F.speed[m])) if m.any() else None,
                       "max_speed_next_0.5s": float(np.nanmean(smax[m])) if m.any() else None,
                       "p_past_ball_1s": p(x_later > bx_later, m),
                       "dx_1s": float(np.nanmean((x_later - F.x)[m])) if m.any() else None}
        fl = (E["got_flung"] > 0)
        o["got_flung"] = {"n": int(fl.sum()), "own_goals_after": int((E["flung_own_goal"] > 0).sum())}
    ip, ib = E["impale_plant"] > 0, E["impale_brake"] > 0
    o["impale_brake"] = {"n_plants": int(ip.sum()), "n": int(ib.sum())}
    return o


def outcomes_2026_10_03(F, rider_minutes, n_sides):
    """Interlocks (each once: view 0 of self-play), let-go, open-goal conversion, situation A and the pole rotation in it
    (analysis/concept_outcomes.py; the user's 2026-10-03 notes)."""
    import concept_outcomes as CO
    ctx = CO.Ctx(F)
    views = [0] if n_sides == 2 else None
    s = CO.summarize_interlocks(CO.interlock_rows(F, ctx, views=views), rider_minutes / n_sides)
    return {"interlock": {k: v for k, v in s.items() if k not in ("by_top",)},
            "open_goal": CO.summarize_open_goal(CO.open_goal_rows(F, ctx), rider_minutes),
            "sit_A": CO.summarize_situation(CO.situation_rows(F, 2, ctx), rider_minutes),
            "rotation_A": CO.rotation_stats((F, C.sit_A(F))), "rotation_all": CO.rotation_stats((F, F.alive))}


def measure(O, G, LIVE, sides, TC=None, ACT=None):
    """Concept stats for the rows of `sides` (list of 0/1 = A/B views) pooled."""
    T = len(O)
    O = O[:, :, sides]; G = G[:, :, sides]
    live = np.repeat(LIVE[:, :, None], len(sides), 2)
    F = C.Frames(O, live, touch=None if TC is None else TC[:, :, sides], act=None if ACT is None else ACT[:, :, sides])
    S, E = C.state_concepts(F), C.event_concepts(F)
    cg = np.concatenate([np.zeros((1,) + G.shape[1:]), np.cumsum(G, 0)])
    fut = cg[np.minimum(np.arange(T) + H, T)] - cg[np.arange(T)]  # own goal diff over the next 10 s
    inside = live & (np.arange(T)[:, None, None] + H <= T)
    ok = F.alive & inside  # states: moments I'm in play; events: any live step (a death leaves me out of play)
    mins = live.sum() * C.STEP_S / 60
    base = float(fut[ok].mean())
    r = {"rider_minutes": float(mins), "goals_per_min": float((G > 0).sum() / mins),
         "baseline_10s": base, "states": {}, "events": {}}
    alive = F.alive.sum()
    for n, v in S.items():
        m = v & ok
        r["states"][n] = {"share": float(v.sum() / alive), "entries_per_min": float(C._onset(v, F.same).sum() / mins),
                          "value": float(fut[m].mean() - base) if m.any() else None, "n": int(m.sum())}
    for n, v in E.items():
        m = (v > 0) & inside
        f = fut[m]
        r["events"][n] = {"per_min": float((v > 0).sum() / mins), "n": int((v > 0).sum()),
                          "value": float(f.mean() - base) if len(f) else None,
                          "se": float(f.std() / np.sqrt(len(f))) if len(f) > 1 else None}
    # energy-storage check for the pogo proxy: apex body height in the 1.5 s after a pogo vs another vault
    v, pg = E["vault"] > 0, E["pogo_vault"] > 0
    apex = np.full(F.h.shape, np.nan, np.float32)
    for k in range(T):
        apex[k] = np.max(np.where(live[k:k + 23], F.h[k:k + 23], -np.inf), 0)
    r["vault_apex_m"] = {"pogo": float(np.nanmedian(apex[pg])) if pg.any() else None,
                         "other": float(np.nanmedian(apex[v & ~pg])) if (v & ~pg).any() else None}
    r["outcomes"] = outcomes(F, E, fut, ok, inside, base, live)
    r["outcomes"].update(outcomes_2026_10_03(F, mins, len(sides)))
    r["act_rows"] = ACT is not None
    r["touch_rows"] = TC is not None
    return r


def _f(v, fmt="{:+.3f}"):
    return "n/a" if v is None else fmt.format(v)


def fmt_outcomes(labels, results):
    out = ["  -- outcomes (2026-10-02 concepts; goal diff values relative to the baseline, p = probability)"]
    for l, r in zip(labels, results):
        o = r["outcomes"]
        lu = o["lunge"]
        out.append(f"  {l}: P(goal against within 5 s) baseline {_f(o['p_goal_against_5s_baseline'], '{:.3f}')}, "
                   f"P(goal for within 5 s) {_f(o['p_goal_for_5s_baseline'], '{:.3f}')}")
        out.append(f"    lunge at opp: n={lu['n']} ({lu['n_from_pit']} from the pit), overcommit per lunge "
                   f"{lu['overcommit_per_lunge']:.2f}, P(goal against 5 s) after lunge {_f(lu['p_goal_against_5s'], '{:.3f}')}, "
                   f"after overcommit {_f(lu['p_goal_against_5s_after_overcommit'], '{:.3f}')}, after overcommit with the "
                   f"opponent still {_f(lu['p_goal_against_5s_after_overcommit_opp_still'], '{:.3f}')}; opponent still in "
                   f"{100 * lu['opp_still_share_of_overcommits']:.0f}% of overcommits (his median speed at the flip "
                   f"{_f(lu['opp_speed_at_overcommit'], '{:.1f}')} m/s)")
        for key, lab, unit in (("energy_bins", "energy (J/kg)", ".0f"), ("energy_lead_bins", "energy lead me-opp", ".0f"),
                               ("takeoff_speed_bins", "take-off speed (m/s)", ".1f")):
            b = o[key]
            if b:
                e = b["edges"]
                rng = [f"<{e[0]:{unit}}"] + [f"{e[i]:{unit}}-{e[i + 1]:{unit}}" for i in range(len(e) - 1)] + [f">{e[-1]:{unit}}"]
                out.append(f"    value by {lab} quartile: " + ", ".join(f"{q} {_f(v)} (n={n})" for q, v, n in
                                                                       zip(rng, b["value"], b["n"])))
        for name in ("second_touch", "second_kick"):
            if name in o:
                x = o[name]
                out.append(f"    {name}: n={x['n']}, P(ball moving my way 0.5 s later) {_f(x['p_ball_my_way_0.5s'], '{:.2f}')}, "
                           f"P(goal for / against within 5 s) {_f(x['p_goal_for_5s'], '{:.3f}')} / "
                           f"{_f(x['p_goal_against_5s'], '{:.3f}')}, my speed {_f(x['speed_at'], '{:.1f}')} -> max "
                           f"{_f(x['max_speed_next_0.5s'], '{:.1f}')} m/s in 0.5 s, P(past the ball 1 s later) "
                           f"{_f(x['p_past_ball_1s'], '{:.2f}')}, dx 1 s later {_f(x['dx_1s'], '{:+.1f}')} m")
        if "got_flung" in o:
            out.append(f"    got flung: n={o['got_flung']['n']}, own goals after (flung_own_goal) "
                       f"{o['got_flung']['own_goals_after']}; impale plants n={o['impale_brake']['n_plants']}, of which brakes "
                       f"(speed drop) n={o['impale_brake']['n']}")
        if "interlock" in o:
            il, og, sa, ra = o["interlock"], o["open_goal"], o["sit_A"]["all"], o["rotation_A"]
            out.append(f"    interlocks (both poles on each other >= 1 s): {il['per_min']:.2f}/match-min (n={il['n']}), >= 3 s "
                       f"{il['long_per_min']:.2f}/min, {100 * il['share_of_time']:.0f}% of play, median {_f(il['dur_median_s'], '{:.1f}')} s, "
                       f"longest {_f(il['dur_max_s'], '{:.0f}')} s, stacked {_f(il['stacked_share'], '{:.2f}')}; let-go escapes "
                       f"{il['let_go']['me']}+{il['let_go']['opp']}; attacker kicks first / scores 5 s / concedes 5 s: "
                       f"{_f(il['by_role']['attacker']['kick_first'], '{:.2f}')} / {_f(il['by_role']['attacker']['gf5'], '{:.2f}')} / "
                       f"{_f(il['by_role']['attacker']['ga5'], '{:.2f}')}")
            out.append(f"    open goal (opp dead / > 5.5 m on his wrong side, >= 0.5 s): {og['per_min']:.2f}/min (n={og['all']['n']}), "
                       f"converted {_f(og['all']['converted'], '{:.2f}')} (opp dead {_f(og['dead']['converted'], '{:.2f}')}, out of "
                       f"position {_f(og['out_of_position']['converted'], '{:.2f}')}), median time to goal "
                       f"{_f(og['all']['time_to_goal_median_s'], '{:.1f}')} s")
            out.append(f"    situation A entries n={sa['n']}: P(conceded 10 s) {_f(sa['ga10'], '{:.2f}')}, back on the good side in 3 s "
                       f"{_f(sa['escaped_3s'], '{:.2f}')}; pole in A: turning backward {_f(ra.get('w_back_share'), '{:.2f}')}, stick drives "
                       f"it backward {_f(ra.get('aim_drives_back_share'), '{:.2f}')} of the time")
        if not r["touch_rows"]:
            out.append("    (no touch rows: this libsprsim build has no touch export; touch concepts are empty)")
    return "\n".join(out)


def fmt(name, results, labels):
    W = 30
    out = [f"\n== {name}"]
    for l, r in zip(labels, results):
        ap = r["vault_apex_m"]
        out.append(f"  {l}: {r['rider_minutes']:.0f} rider-min, {r['goals_per_min']:.2f} goals/min, baseline goal diff "
                   f"next 10 s {r['baseline_10s']:+.3f}; vault apex median: pogo {ap['pogo'] or float('nan'):.2f} m, "
                   f"other {ap['other'] or float('nan'):.2f} m")
    out.append(f"  {'state':22s}" + "".join(f"{l + ': share entries/min value':>{W + 12}s}" for l in labels))
    for n in C.STATE_NAMES:
        cells = []
        for r in results:
            s = r["states"][n]
            v = "n/a" if s["value"] is None else f"{s['value']:+.3f}"
            cells.append(f"{100 * s['share']:5.1f}% {s['entries_per_min']:7.2f} {v:>7s}")
        out.append(f"  {n:22s}" + "".join(f"{c:>{W + 12}s}" for c in cells))
    out.append(f"  {'event':22s}" + "".join(f"{l + ': per min value +- se':>{W + 12}s}" for l in labels))
    for n in C.EVENT_NAMES:
        cells = []
        for r in results:
            s = r["events"][n]
            v = "n/a" if s["value"] is None else f"{s['value']:+.3f} +- {s['se'] or 0:.3f}"
            cells.append(f"{s['per_min']:6.2f} {v:>16s}")
        out.append(f"  {n:22s}" + "".join(f"{c:>{W + 12}s}" for c in cells))
    out.append(fmt_outcomes(labels, results))
    return "\n".join(out)


def name_of(path):
    """runs/<run>/ckpt.pt -> <run>; a frozen copy (runs/gauntlet/ckpt/<name>.pt) -> <name>."""
    d = os.path.basename(os.path.dirname(os.path.abspath(path)))
    return os.path.splitext(os.path.basename(path))[0] if d == "ckpt" else d


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--self", dest="selfplay", action="append", default=[], help="checkpoint for self-play")
    p.add_argument("--h2h", action="append", default=[], help="A_ckpt:B_ckpt head to head")
    p.add_argument("--matches", type=int, default=128)
    p.add_argument("--threads", type=int, default=4)
    p.add_argument("--seed", type=int, default=4242)
    p.add_argument("--out", default=os.path.join(HERE, "concepts_report"), help="writes <out>.txt and <out>.json")
    a = p.parse_args()
    torch.set_num_threads(1)
    report, text = {"matches": a.matches, "horizon_s": H / C.HZ, "concepts": C.describe(), "runs": {}}, []
    text.append(f"Concept report ({time.strftime('%Y-%m-%d %H:%M')}): {a.matches} matches per line-up, native sim, "
                f"7 goals / 5 min, no sudden death. Value = own goal diff over the next {H // C.HZ} s minus the policy's "
                f"baseline (correlational). Rates per minute of one rider's play. Definitions: concepts.py.")
    jobs = [("self", s, s) for s in a.selfplay] + [("h2h",) + tuple(x.split(":")) for x in a.h2h]
    for kind, pa_path, pb_path in jobs:
        t0 = time.time()
        pa, cka = load(pa_path)
        pb, ckb = (pa, cka) if kind == "self" else load(pb_path)
        na, nb = name_of(pa_path), name_of(pb_path)
        O, G, LIVE, TC, ACT = play(pa, pb, a.matches, a.threads, a.seed, with_actions=True)
        steps = {na: int(cka.get("steps", 0)), nb: int(ckb.get("steps", 0))}
        if kind == "self":
            key, labels, res = f"{na} self-play", [na], [measure(O, G, LIVE, [0, 1], TC, ACT)]
        else:
            key, labels = f"{na} vs {nb}", [na, nb]
            res = [measure(O, G, LIVE, [0], TC, ACT), measure(O, G, LIVE, [1], TC, ACT)]
            gd = G[:, :, 0].sum(0)
            res[0]["h2h"] = {"goal_diff_per_match": float(gd.mean()), "w": int((gd > 0).sum()),
                             "d": int((gd == 0).sum()), "l": int((gd < 0).sum())}
        report["runs"][key] = {"steps": steps, **dict(zip(labels, res))}
        title = key + "  (" + ", ".join(f"{k} {v / 1e6:.0f}M steps" for k, v in steps.items()) + ")"
        if kind == "h2h":
            h = res[0]["h2h"]
            title += f"  {na} W-D-L {h['w']}-{h['d']}-{h['l']}, goal diff {h['goal_diff_per_match']:+.2f}/match"
        text.append(fmt(title, res, labels))
        print(text[-1], f"\n  ({time.time() - t0:.0f} s)", flush=True)
    open(a.out + ".txt", "w").write("\n".join(text) + "\n")
    json.dump(report, open(a.out + ".json", "w"), indent=1)
    print(f"-> {a.out}.txt / .json")


if __name__ == "__main__":
    main()
