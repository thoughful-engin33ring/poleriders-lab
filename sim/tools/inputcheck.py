#!/usr/bin/env python3
"""Package D check: the native input chain (build/inputcheck) against the real game's, tick by tick.

    inputcheck.py [--ticks 600] [--seed 1] [--hold 1] [--sock NAME] [--reuse]

Generates a stick script that stresses the chain (full deflections, diagonals past the unit circle, -0 and
zero crossings, values at the 0.1 deadzone +-1e-6, around the 0.4/0.8/0.2 flick thresholds, per-tick
changes), runs it in an oracle shadow (dump every tick; only the input objects are kept) and through
build/inputcheck, and compares after every tick:
  - gamepad0/1 (atrophy::GamepadSDL) and pad0/1 (Common::Input::_pads) bytes, bit for bit;
  - VirtualRightStick/VirtualLeftStick of session 0/1 vs the real Player+0x19c..+0x1bb (the values the
    player read this tick) whenever Player+0x1d0 (active) is set — except on the tick the player respawns
    (PlayState+0x1a0 timer runs out, or a new pole/body): SpawnPlayerBody zeroes them after the read.
Pad+0xf4/+0xf8 (rumble) are written by Player::Update's VirtualUpdateVibration after _UpdatePad; they are
fed back from the real trace (not an input-chain output) so the next tick sees the same state.
"""
import argparse, json, os, random, struct, subprocess, sys, time

HERE = os.path.dirname(os.path.abspath(__file__))
SIM = os.path.join(HERE, "..")
sys.path.insert(0, HERE)
from oracle import SOCK, Oracle  # noqa: E402

KEEP = ["gamepad0", "gamepad1", "pad0", "pad1"]


def stick_value(rng, prev):
    r = rng.random()
    if r < 0.25:
        return prev
    c = rng.randrange(10)
    if c == 0:
        return rng.choice([-1.0, 1.0])
    if c == 1:
        return rng.choice([0.0, -0.0, 1e-7, -1e-7])
    if c == 2:  # the 0.1 deadzone +- a few 1e-6
        return rng.choice([-1, 1]) * (0.1 + rng.randint(-3, 3) * 1e-6)
    if c == 3:  # processed 0.4 / 0.8 / 0.2 thresholds: raw = 0.1 + 0.9 * t
        t = rng.choice([0.4, 0.8, 0.2])
        return rng.choice([-1, 1]) * (0.1 + 0.9 * t + rng.randint(-4, 4) * 1e-6)
    if c == 4:  # diagonals (x^2 + y^2 > 1 -> divided by the squared length)
        return rng.choice([-1, 1]) * rng.choice([0.7071, 0.70711, 0.75, 0.8, 0.9, 0.95])
    if c == 5:  # sign flip (zero crossing)
        return -prev
    if c == 6:
        return rng.uniform(-0.15, 0.15)
    return rng.uniform(-1, 1)


def script(ticks, seed, hold=1):
    rng = random.Random(seed)
    rows, cur = [], [0.0] * 6
    for t in range(ticks):
        if t % hold:
            pass
        elif rng.random() < 0.1:  # both sticks of a pad on the same diagonal
            p, v = rng.randrange(2), rng.choice([-1, 1]) * rng.choice([0.7071, 0.8, 1.0])
            cur[3 * p:3 * p + 3] = [v, v, rng.choice([-v, v])]
        else:
            cur = [stick_value(rng, x) for x in cur]
        rows.append(" ".join(f"{v:.6f}" for v in cur))
    return rows


def keep(d):
    o = d["objects"]
    r = {k: o[k]["raw"] for k in KEEP}
    for i in range(2):
        p = bytes.fromhex(o[f"player{i}"]["raw"])
        r[f"player{i}"] = {"sticks": p[0x19c:0x1bc].hex(), "active": p[0x1d0], "session": p[0x220:0x23c].hex(),
                           "pole_body": p[0x18:0x20].hex() + p[0x30:0x38].hex()}  # Player+0x18 pole, +0x30 body
    ps = bytes.fromhex(o["playstate"]["raw"])
    r["respawnTimer"] = list(struct.unpack("<2f", ps[0x1a0:0x1a8]))  # PlayState+0x1a0 [player]
    return r


def f4(h):
    return [struct.unpack("<f", bytes.fromhex(h[i:i + 8]))[0] for i in range(0, len(h), 8)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ticks", type=int, default=600)
    ap.add_argument("--seed", type=int, default=1)
    ap.add_argument("--hold", type=int, default=1, help="new stick values every HOLD ticks (hold timers)")
    ap.add_argument("--sock", default=None, help="oracle socket name (default spr-oracle.sock)")
    ap.add_argument("--reuse", action="store_true", help="reuse the saved oracle trace for this script")
    ap.add_argument("--max", type=int, default=20)
    args = ap.parse_args()
    out = os.path.join(SIM, "traces", "cmp")
    os.makedirs(out, exist_ok=True)
    tag = f"input_s{args.seed}_h{args.hold}_t{args.ticks}"
    rows = script(args.ticks, args.seed, args.hold)
    sp = os.path.join(out, f"{tag}.script")
    open(sp, "w").write("\n".join(rows) + "\n")
    real_path = os.path.join(out, f"{tag}.real.jsonl")
    if not (args.reuse and os.path.exists(real_path)):
        sock = os.path.join(os.path.dirname(SOCK), args.sock) if args.sock else SOCK
        o = Oracle(sock=sock)
        try:
            o.boot(launch=False)
        except Exception:  # noqa: BLE001 — no oracle game on this socket yet
            o.boot(launch=True)
        o.reset(0)
        t0 = time.time()
        with open(real_path, "w") as f:
            f.write(json.dumps(keep(o.dump(0))) + "\n")
            for r in rows:
                o.act([float(x) for x in r.split()], n=1)
                f.write(json.dumps(keep(o.dump(0))) + "\n")
        print(f"oracle: {args.ticks} ticks in {time.time() - t0:.1f}s")
    real = [json.loads(l) for l in open(real_path)]
    r0 = real[0]
    eb = os.path.join(out, f"{tag}.engine.bin")
    open(eb, "wb").write(b"".join(bytes.fromhex(r0[k]) for k in KEEP)
                         + bytes.fromhex(r0["player0"]["session"]) + bytes.fromhex(r0["player1"]["session"]))
    rp = os.path.join(out, f"{tag}.rumble")
    with open(rp, "w") as f:
        for d in real[1:]:
            w = []
            for k in ("pad0", "pad1"):
                b = bytes.fromhex(d[k])
                w += [f"{struct.unpack('<I', b[o:o + 4])[0]:08x}" for o in (0xf4, 0xf8)]
            f.write(" ".join(w) + "\n")
    res = subprocess.run([os.path.join(SIM, "build", "inputcheck"), eb, sp, rp], capture_output=True, text=True)
    if res.returncode:
        sys.exit(f"inputcheck failed ({res.returncode}): {res.stderr}")
    native = [l.split() for l in res.stdout.splitlines()]
    if len(native) != len(real):
        sys.exit(f"native produced {len(native)} states, oracle {len(real)}")
    nbad, stick_checked, stick_skipped, respawned = 0, 0, 0, 0
    for t, (d, n) in enumerate(zip(real, native)):
        lines = []
        for j, k in enumerate(KEEP):
            a, b = bytes.fromhex(d[k]), bytes.fromhex(n[j])
            diffs = [o for o in range(len(a)) if a[o] != b[o]]
            if diffs:
                lines.append(f"  {k}: {len(diffs)} bytes differ, first at +0x{diffs[0]:x}: "
                             f"real {a[diffs[0] & ~3:(diffs[0] & ~3) + 4].hex()} native {b[diffs[0] & ~3:(diffs[0] & ~3) + 4].hex()}")
        if t > 0:
            for i in range(2):
                pl = d[f"player{i}"]
                nat = n[4 + 2 * i] + n[5 + 2 * i]  # VirtualRightStick (+0x19c) then VirtualLeftStick (+0x1ac)
                if not pl["active"]:
                    stick_skipped += 1
                    continue
                respawn = (real[t - 1]["respawnTimer"][i] > 0 >= d["respawnTimer"][i]
                           or real[t - 1][f"player{i}"]["pole_body"] != pl["pole_body"])
                if respawn and int(pl["sticks"], 16) == 0:
                    respawned += 1  # Respawn -> SpawnPlayerBody zeroed +0x19c..+0x1bb after this tick's read
                    continue
                stick_checked += 1
                if nat != pl["sticks"]:
                    lines.append(f"  player{i} sticks: real {f4(pl['sticks'])} native {f4(nat)} "
                                 f"(bits {pl['sticks']} vs {nat})")
        if lines:
            nbad += 1
            if nbad <= args.max:
                print(f"tick {t} (input {rows[t - 1] if t else '(construction)'}):")
                print("\n".join(lines))
    print(f"{len(real) - 1} ticks: {'IDENTICAL' if not nbad else f'{nbad} ticks differ'} "
          f"(gamepad/pad bytes every tick; player sticks checked {stick_checked}x, "
          f"skipped {stick_skipped}x inactive, {respawned}x zeroed by a respawn)")
    sys.exit(1 if nbad else 0)


if __name__ == "__main__":
    main()
