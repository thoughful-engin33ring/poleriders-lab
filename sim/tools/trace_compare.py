#!/usr/bin/env python3
"""Tick-by-tick comparison of the native sim against the real game (oracle).

    trace_compare.py [--ticks 600] [--seed 1] [--hold 12] [--zero] [--max 30] [--sock NAME] [--level 1]

Generates an input script (random stick positions held for --hold ticks; --zero = no input), runs it
in an oracle shadow (dumping every tick) and in build/gamecheck, and reports the first tick whose
dumps differ with the differing fields (tools/diff_dump.py). Traces are kept in traces/cmp/.
"""
import argparse, json, os, random, struct, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from diff_dump import diff, load_layouts  # noqa: E402
from oracle import SOCK, Oracle, dump_level  # noqa: E402


def script(ticks, seed, hold, zero):
    rng = random.Random(seed)
    rows, cur = [], [0.0] * 6
    for t in range(ticks):
        if not zero and t % hold == 0:
            cur = [rng.choice([-1.0, 1.0, 0.0, rng.uniform(-1, 1)]) for _ in range(6)]
        rows.append(" ".join(f"{v:.6f}" for v in cur))
    return rows


def engine_bin(d0):
    """gamecheck's engine.bin: the shadow's initial pad state, session records and level id."""
    o = d0["objects"]
    raw = lambda k: bytes.fromhex(o[k]["raw"])
    return (raw("gamepad0") + raw("gamepad1") + raw("pad0") + raw("pad1")
            + raw("player0")[0x220:0x23c] + raw("player1")[0x220:0x23c] + struct.pack("<i", dump_level(d0)))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hold", type=int, default=12)
    ap.add_argument("--zero", action="store_true")
    ap.add_argument("--max", type=int, default=30)
    ap.add_argument("--sock", default=None, help="oracle socket name (default spr-oracle.sock)")
    ap.add_argument("--reuse", action="store_true", help="reuse the saved oracle trace for this script")
    ap.add_argument("--level", type=int, default=1, help="level id (PlayState+0x214): 1 Kalgoorlie, 2 Flatirons")
    args = ap.parse_args()
    out = os.path.join(SIM, "traces", "cmp")
    os.makedirs(out, exist_ok=True)
    tag = f"L{args.level}_s{args.seed}_h{args.hold}_t{args.ticks}" + ("_zero" if args.zero else "")
    rows = script(args.ticks, args.seed, args.hold, args.zero)
    sp = os.path.join(out, f"{tag}.script")
    open(sp, "w").write("\n".join(rows) + "\n")
    real_path = os.path.join(out, f"{tag}.real.jsonl")
    if not (args.reuse and os.path.exists(real_path)):
        sock = os.path.join(os.path.dirname(SOCK), args.sock) if args.sock else SOCK
        o = Oracle(sock=sock, level=args.level)
        try:
            o.boot(launch=False)
        except Exception:  # noqa: BLE001 — no oracle game on this socket yet
            o.boot(launch=True)
        o.reset(0)
        t0 = time.time()
        with open(real_path, "w") as f:
            f.write(json.dumps(o.dump(0)) + "\n")
            for r in rows:
                o.act([float(x) for x in r.split()], n=1)
                f.write(json.dumps(o.dump(0)) + "\n")
        print(f"oracle: {args.ticks} ticks in {time.time() - t0:.1f}s")
    real = [json.loads(l) for l in open(real_path)]
    if dump_level(real[0]) != args.level:
        sys.exit(f"oracle trace is level {dump_level(real[0])}, want {args.level}")
    eb = os.path.join(out, f"{tag}.engine.bin")
    open(eb, "wb").write(engine_bin(real[0]))
    nat_path = os.path.join(out, f"{tag}.native.jsonl")
    r = subprocess.run([os.path.join(SIM, "build", "gamecheck"), eb, sp, nat_path])
    if r.returncode:
        sys.exit(f"gamecheck failed ({r.returncode})")
    native = [json.loads(l) for l in open(nat_path)]
    L = load_layouts()
    for t, (a, b) in enumerate(zip(real, native)):
        n, lines = diff(a, b, args.max, layouts=L)
        if n:
            print(f"FIRST DIVERGENCE at tick {t} ({n} differences); input before it: "
                  f"{rows[t - 1] if t else '(construction)'}")
            print("\n".join(lines))
            sys.exit(1)
    if len(native) != len(real):
        sys.exit(f"native produced {len(native)} dumps, oracle {len(real)}")
    print(f"IDENTICAL for all {len(real) - 1} ticks")


if __name__ == "__main__":
    main()
