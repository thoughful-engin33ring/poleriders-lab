#!/usr/bin/env python3
"""Proves a farm shadow is the real game: from the same tick-0 kickoff, a shadow match and the
real rendered match get identical random stick inputs, one tick at a time, and their riders,
poles and ball must stay identical. Uses a throwaway game instance.

    python3 bridge/test_farm_equivalence.py [ticks]
"""
import json, os, random, sys, time
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from spr_env import PoleRidersEnv, SOCK_DIR, launch_game  # noqa: E402

TICKS = int(sys.argv[1]) if len(sys.argv) > 1 else 1200


def key(st):
    """Everything physical in a snapshot, as a comparable tuple."""
    out = []
    for p in st["players"]:
        out += [p["x"], p["y"], p["a"], p["vx"], p["vy"], p["w"], p["on_pole"], p["on_ground"], p["ghost"],
                p.get("dead"), p.get("respawn_t")]
        out += [c for seg in p["pole"] for c in (seg["x"], seg["y"], seg["a"])]
    b = st["ball"]
    return out + [b["x"], b["y"], b["vx"], b["vy"]] + st["score"]


sock = os.path.join(SOCK_DIR, "spr-eq.sock")
proc = launch_game(sock)
try:
    time.sleep(16)
    env = PoleRidersEnv(sock=sock)
    env.start_match(arena=2)
    b = env.b
    b.cmd("sync on")
    # restart the real match and catch it at tick 0 (the creation frame doesn't tick)
    old = json.loads(b.cmd("step 1"))["match_id"]
    b.cmd("reset")
    for _ in range(3000):
        st = json.loads(b.cmd("step 1"))
        if st.get("in_match") and st["match_id"] != old and st.get("players"):
            break
    # After a restart the real match is frozen while the loading screen fades in. Step it with
    # zero input until its state first changes: that frame was its first real tick.
    zero = "act 1 0 0 0 0 0 0"
    prev, frozen = key(st), 0
    for _ in range(3000):
        st = json.loads(b.cmd(zero))
        if key(st) != prev:
            break
        frozen += 1
    print(f"real match was frozen for {frozen} frames after creation")
    print(json.loads(b.cmd("farm create 1")))
    shadow = json.loads(b.cmd("farm act 1 0 0 0 0 0 0"))["matches"][0]  # its first tick too
    print("after 1 tick each:", "same" if key(st) == key(shadow) else "DIFFERENT")
    rng = random.Random(7)
    first_diff = None
    hist = []
    for t in range(TICKS):
        a = [round(rng.uniform(-1, 1), 4) for _ in range(6)]
        real = json.loads(b.cmd("act 1 " + " ".join(map(str, a))))
        shadow = json.loads(b.cmd("farm act 1 " + " ".join(map(str, a))))["matches"][0]
        kr, ks = key(real), key(shadow)
        hist.append((t, real, shadow))
        if kr != ks and first_diff is None:
            first_diff = t
            diffs = [(i, u, v) for i, (u, v) in enumerate(zip(kr, ks)) if u != v]
            print(f"tick {t}: {len(diffs)} values differ, first few {diffs[:5]}")
            for tt, r, sh in hist[-6:]:
                ev = lambda st: [(p["id"], p["ghost"], p["on_pole"], p["on_ground"], p["stats"]["deaths"],
                                  p["stats"]["suicides"], p["stats"]["jumps"], p["stats"]["kicks"]) for p in st["players"]]
                print(f"  t={tt} real {ev(r)} ball_vy {r['ball']['vy']:.3f}")
                print(f"  t={tt} shad {ev(sh)} ball_vy {sh['ball']['vy']:.3f}")
            break
    print(f"real vs shadow over {TICKS} ticks: " + ("IDENTICAL" if first_diff is None else f"diverged at tick {first_diff}"))
    print("final real  :", real["score"], [round(p['x'], 3) for p in real["players"]], round(real["ball"]["x"], 3))
    print("final shadow:", shadow["score"], [round(p['x'], 3) for p in shadow["players"]], round(shadow["ball"]["x"], 3))
    b.cmd("farm destroy"); b.cmd("sync off")
finally:
    proc.kill()
